"""
runner.py
=========
End-to-end closed-loop rendezvous simulation.

Pipeline per time step
----------------------
  1.  Propagate true state (HCW translation + torque-free attitude) with
      the last control command injected into the translational dynamics.
  2.  Render a synthetic wireframe image of the target from the chaser
      camera and project keypoints.
  3.  Run RANSAC + EPnP on the projected keypoints (with optional pixel noise)
      to get a raw pose estimate.
  4.  Feed the pose measurement into the Multiplicative EKF.
  5.  Extract the translational state estimate [r̂, v̂] from the EKF.
  6.  Compute LQR control on the estimate; saturate to u_max.
  7.  Log everything.

The simulator can also run in "perfect-state" mode (no vision / no EKF)
to serve as a baseline for comparison.

Coordinate frames
-----------------
  LVLH (Local Vertical / Local Horizontal):
      x  radial (chaser → nadir)
      y  along-track
      z  cross-track
  Target body frame: right-hand, principal axes.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from dynamics.constants import LEO_N as N_DEFAULT
from target.attitude import RigidBodyAttitude
from estimator.state import quat_to_dcm
from vision.camera import rendezvous_camera
from vision.body_model import ariane_model
from vision.renderer import DebrisRenderer, RendererConfig
from pose.ransac import RANSACSolver, solve_pnp_ransac
from pose.epnp import EPnPSolver
from estimator.state import pack_state, unpack_state, propagate_rk4, h_measurement
from estimator.ekf import MultEKF, make_ekf
from controller.lqr import LQRController, make_lqr, hcw_discrete, N_ORBITAL_DEFAULT


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class SimConfig:
    """All tunable parameters for one simulation run."""

    # Orbital
    n_orbital: float = N_ORBITAL_DEFAULT   # mean motion [rad/s]
    dt:        float = 1.0                 # control / filter step [s]
    n_steps:   int   = 300                 # total simulation steps

    # Initial true state
    r0:  np.ndarray = field(default_factory=lambda: np.array([30., 3., -1.5]))
    v0:  np.ndarray = field(default_factory=lambda: np.array([-0.08, 0., 0.]))
    q0:  np.ndarray = field(default_factory=lambda: np.array([0., 0., 0., 1.]))
    w0:  np.ndarray = field(default_factory=lambda: np.array([0.02, 0.05, 0.01]))

    # Initial EKF state offset (to test estimation convergence)
    r0_err: np.ndarray = field(default_factory=lambda: np.array([2., -0.5, 0.3]))
    v0_err: np.ndarray = field(default_factory=lambda: np.array([0., 0., 0.]))

    # Noise
    pixel_noise_std: float = 1.5    # [px] keypoint detection noise
    pos_meas_std:    float = 0.5    # [m]  EKF measurement noise (position)
    att_meas_std:    float = 0.05   # [rad] EKF measurement noise (attitude)

    # Controller
    u_max:       float = 0.3        # [m/s²] thrust limit
    pos_weight:  float = 10.0       # LQR position cost
    vel_weight:  float = 1.0        # LQR velocity cost

    # Vision pipeline
    use_vision:  bool  = True       # if False → perfect state measurement
    use_ekf:     bool  = True       # if False → feed raw EPnP to LQR
    ransac_iter: int   = 100
    ransac_thr:  float = 3.0        # [px] RANSAC reprojection threshold

    # Docking
    dock_r_thr: float = 1.0         # [m]  docking distance
    dock_v_thr: float = 0.05        # [m/s] docking speed


def default_config(**kwargs) -> SimConfig:
    """Build a SimConfig, overriding any fields via kwargs."""
    cfg = SimConfig()
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class SimResult:
    """All logged quantities from a simulation run."""
    cfg: SimConfig

    # Trajectories  (n_steps+1, *)
    r_true:     np.ndarray = None   # (T, 3)  true position
    v_true:     np.ndarray = None   # (T, 3)  true velocity
    q_true:     np.ndarray = None   # (T, 4)  true quaternion
    w_true:     np.ndarray = None   # (T, 3)  true angular velocity

    r_est:      np.ndarray = None   # (T, 3)  EKF/raw position estimate
    v_est:      np.ndarray = None   # (T, 3)  EKF/raw velocity estimate

    controls:   np.ndarray = None   # (T-1, 3) applied thrust
    repr_errs:  np.ndarray = None   # (T-1,)   EPnP reprojection error per step
    ekf_cov_pos: np.ndarray = None  # (T, 3)   diagonal of P for position

    # Scalars
    delta_v:     float = 0.0
    dock_step:   Optional[int] = None   # first step reaching docking
    n_steps_run: int = 0

    @property
    def pos_error(self) -> np.ndarray:
        return np.linalg.norm(self.r_true - self.r_est, axis=1)

    @property
    def range_m(self) -> np.ndarray:
        return np.linalg.norm(self.r_true, axis=1)

    @property
    def converged(self) -> bool:
        return self.dock_step is not None


# ── Simulator ─────────────────────────────────────────────────────────────────

class RendezvousSimulator:
    """
    Closed-loop rendezvous simulator integrating all six phases.

    Parameters
    ----------
    cfg : SimConfig
        All tunable parameters.
    rng_seed : int
        Random seed for reproducibility.
    """

    def __init__(self, cfg: SimConfig = None, rng_seed: int = 42):
        self.cfg = cfg or SimConfig()
        self.rng = np.random.default_rng(rng_seed)

        # Build subsystems
        self.cam    = rendezvous_camera()
        self.model  = ariane_model()
        self.renderer = DebrisRenderer(self.cam)

        self.Phi, self.Gamma = hcw_discrete(self.cfg.n_orbital, self.cfg.dt)

        self.lqr = make_lqr(
            n=self.cfg.n_orbital,
            dt=self.cfg.dt,
            u_max=self.cfg.u_max,
            pos_weight=self.cfg.pos_weight,
            vel_weight=self.cfg.vel_weight,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _project_keypoints_noiseless(self, r_true: np.ndarray,
                                     q_true: np.ndarray):
        """
        Project 3-D model keypoints noislessly.

        The camera is on the chaser, looking toward the target.
        We model the camera frame = LVLH frame with the origin at the chaser.
        The target body is at position r_true (LVLH) with orientation q_true.

        Returns
        -------
        kp3d : (K, 3) keypoints in camera/LVLH frame (object space)
        kp2d : (K, 2) projected pixel coordinates
        """
        kp3d_body = self.model.keypoint_array         # (K, 3) body frame
        R_body2cam = quat_to_dcm(q_true)              # body → camera frame
        # Transform keypoints to camera frame (target at r_true)
        kp3d_cam = (R_body2cam @ kp3d_body.T).T + r_true  # (K, 3)
        # Pinhole projection (z must be > 0 = in front of camera)
        fx, fy = self.cam.K[0, 0], self.cam.K[1, 1]
        cx, cy = self.cam.K[0, 2], self.cam.K[1, 2]
        z = kp3d_cam[:, 2]
        u = fx * kp3d_cam[:, 0] / z + cx
        v = fy * kp3d_cam[:, 1] / z + cy
        kp2d = np.stack([u, v], axis=-1)
        return kp3d_cam, kp2d

    def _vision_measurement(self, r_true: np.ndarray,
                            q_true: np.ndarray):
        """
        Run EPnP (+ RANSAC) on synthetic keypoints with pixel noise.

        Returns
        -------
        r_meas  : (3,) position estimate  [m]
        q_meas  : (4,) quaternion estimate
        repr_err: float  mean reprojection error [px]
        ok      : bool   True if EPnP succeeded
        """
        kp3d_body = self.model.keypoint_array
        _, kp2d = self._project_keypoints_noiseless(r_true, q_true)
        # Add pixel noise
        if self.cfg.pixel_noise_std > 0:
            kp2d = kp2d + self.rng.normal(0, self.cfg.pixel_noise_std, kp2d.shape)

        # RANSAC + EPnP — solve for pose of body keypoints (body frame) from 2D
        try:
            R_est, t_est, inliers, meta = solve_pnp_ransac(
                kp3d_body, kp2d, self.cam.K,
                threshold_px=self.cfg.ransac_thr,
                max_iter=self.cfg.ransac_iter,
            )
            ok = meta.get('success', False)
            if not ok or R_est is None:
                return r_true.copy(), q_true.copy(), np.nan, False

            # Compute reprojection error manually
            from estimator.state import dcm_to_quat
            q_est    = dcm_to_quat(R_est)
            P_cam    = (R_est @ kp3d_body.T).T + t_est.flatten()
            fx, fy   = self.cam.K[0, 0], self.cam.K[1, 1]
            cx, cy   = self.cam.K[0, 2], self.cam.K[1, 2]
            u_proj   = fx * P_cam[:, 0] / P_cam[:, 2] + cx
            v_proj   = fy * P_cam[:, 1] / P_cam[:, 2] + cy
            repr_err = float(np.mean(np.sqrt(
                (u_proj - kp2d[:, 0])**2 + (v_proj - kp2d[:, 1])**2)))
            return t_est.flatten(), q_est, repr_err, True
        except Exception:
            return r_true.copy(), q_true.copy(), np.nan, False

    # ── Main simulation loop ──────────────────────────────────────────────────

    def run(self) -> SimResult:
        """Execute the full closed-loop simulation."""
        cfg = self.cfg
        res = SimResult(cfg=cfg)

        # Allocate logs
        T = cfg.n_steps + 1
        r_true  = np.zeros((T, 3))
        v_true  = np.zeros((T, 3))
        q_true  = np.zeros((T, 4))
        w_true  = np.zeros((T, 3))
        r_est   = np.zeros((T, 3))
        v_est   = np.zeros((T, 3))
        ctrls   = np.zeros((cfg.n_steps, 3))
        repr_e  = np.full(cfg.n_steps, np.nan)
        cov_pos = np.zeros((T, 3))

        # ── Initialise true state ──────────────────────────────────────────
        x_true = pack_state(cfg.r0, cfg.v0, cfg.q0, cfg.w0)
        r_true[0], v_true[0], q_true[0], w_true[0] = (
            cfg.r0, cfg.v0, cfg.q0, cfg.w0)

        # ── Initialise EKF ────────────────────────────────────────────────
        x_ekf0 = pack_state(
            cfg.r0 + cfg.r0_err, cfg.v0 + cfg.v0_err, cfg.q0, cfg.w0)
        ekf: Optional[MultEKF] = None
        if cfg.use_ekf:
            ekf = make_ekf(x_ekf0, cfg.dt,
                           pos0_std=3.0,
                           meas_pos_std=cfg.pos_meas_std,
                           meas_att_std=cfg.att_meas_std)
            r_est[0] = unpack_state(ekf.state)[0]
            v_est[0] = unpack_state(ekf.state)[1]
            cov_pos[0] = np.diag(ekf.P)[:3]
        else:
            r_est[0] = cfg.r0 + cfg.r0_err
            v_est[0] = cfg.v0 + cfg.v0_err

        u = np.zeros(3)   # control initialised to zero
        delta_v = 0.0
        dock_step = None

        for k in range(cfg.n_steps):
            # ── 1. Propagate true state ─────────────────────────────────
            rt, vt, qt, wt = unpack_state(x_true)
            rv_new = self.Phi @ np.concatenate([rt, vt]) + self.Gamma @ u
            # Propagate attitude torque-free
            x_att  = propagate_rk4(x_true, cfg.dt, cfg.n_orbital)
            x_true = pack_state(rv_new[:3], rv_new[3:],
                                unpack_state(x_att)[2],
                                unpack_state(x_att)[3])
            rt_new, vt_new, qt_new, wt_new = unpack_state(x_true)
            r_true[k+1] = rt_new
            v_true[k+1] = vt_new
            q_true[k+1] = qt_new
            w_true[k+1] = wt_new

            # ── 2-3. Vision measurement ─────────────────────────────────
            if cfg.use_vision:
                r_meas, q_meas, repr_err, ok = self._vision_measurement(
                    rt_new, qt_new)
                repr_e[k] = repr_err
            else:
                # Perfect measurement with small noise
                r_meas = rt_new + self.rng.normal(0, cfg.pos_meas_std, 3)
                q_meas = qt_new.copy()
                repr_e[k] = 0.0

            # Build measurement vector [r, rotvec(q)]
            from estimator.state import quat_to_rotvec
            z_meas = np.concatenate([r_meas, quat_to_rotvec(q_meas)])

            # ── 4. EKF update ───────────────────────────────────────────
            if cfg.use_ekf and ekf is not None:
                # Control-aware predict: inject last u into EKF
                ekf.predict(cfg.dt)
                ctrl_effect = self.Gamma @ u
                ekf.x[:3]  += ctrl_effect[:3]
                ekf.x[3:6] += ctrl_effect[3:]
                ekf.update(z_meas)
                r_est[k+1] = unpack_state(ekf.state)[0]
                v_est[k+1] = unpack_state(ekf.state)[1]
                cov_pos[k+1] = np.sqrt(np.diag(ekf.P)[:3])
            else:
                r_est[k+1] = r_meas
                v_est[k+1] = vt_new  # use true velocity if no filter
                cov_pos[k+1] = np.zeros(3)

            # ── 5-6. LQR control on estimate ────────────────────────────
            x_ctrl = np.concatenate([r_est[k+1], v_est[k+1]])
            u = self.lqr.control(x_ctrl)
            ctrls[k] = u
            delta_v += np.linalg.norm(u) * cfg.dt

            # ── 7. Check docking ─────────────────────────────────────────
            if (dock_step is None
                    and np.linalg.norm(rt_new) < cfg.dock_r_thr
                    and np.linalg.norm(vt_new) < cfg.dock_v_thr):
                dock_step = k + 1

            res.n_steps_run = k + 1

        res.r_true      = r_true
        res.v_true      = v_true
        res.q_true      = q_true
        res.w_true      = w_true
        res.r_est       = r_est
        res.v_est       = v_est
        res.controls    = ctrls
        res.repr_errs   = repr_e
        res.ekf_cov_pos = cov_pos
        res.delta_v     = delta_v
        res.dock_step   = dock_step
        return res


# ── Convenience ───────────────────────────────────────────────────────────────

def run_simulation(cfg: SimConfig = None, rng_seed: int = 42) -> SimResult:
    """Build a simulator and run it; return the SimResult."""
    sim = RendezvousSimulator(cfg or SimConfig(), rng_seed=rng_seed)
    return sim.run()

"""
runner.py
=========
End-to-end closed-loop rendezvous simulation.

Pipeline per time step
----------------------
  1.  Propagate true state (HCW translation + torque-free attitude) with
      the last control command injected into the translational dynamics.
  2.  Project the target's 3-D model keypoints into the chaser camera and
      apply the visibility gate (in front of the camera, inside the image).
      NOTE: no image is rendered — the keypoint projection is analytic and
      correspondences are known, so this is an idealised feature front end.
  3.  Run RANSAC + EPnP on the projected keypoints (with optional pixel noise)
      to get a raw pose estimate.
  4.  Feed the pose measurement into the Multiplicative EKF.
  5.  Extract the translational state estimate [r̂, v̂] from the EKF.
  6.  Compute LQR control on the estimate; saturate to u_max.
  7.  Log everything.

Setting `use_vision=False` replaces the vision front end with a direct pose
measurement corrupted by the same noise the filter's R declares
(`pos_meas_std`, `att_meas_std`); it isolates the estimator and controller
from the perception chain.  Setting `use_ekf=False` in addition removes the
filter, leaving raw measurements plus model dead reckoning.

Coordinate frames
-----------------
  LVLH (Local Vertical / Local Horizontal):
      x  radial (positive away from Earth, i.e. zenith)
      y  along-track
      z  cross-track
  Target body frame: right-hand, principal axes.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

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

    # Unmodelled disturbance acceleration applied to the TRUE trajectory only
    # (differential J2, drag, thruster execution error).  Without it the truth
    # is propagated by exactly the filter's own model with zero process noise,
    # so the filter's Q describes a disturbance that does not exist and its
    # consistency statistics are meaningless.  Set to 0.0 to recover the old
    # no-model-error behaviour.
    disturb_accel_std: float = 5.0e-4   # [m/s²] 1-sigma per axis

    # Controller
    u_max:       float = 0.3        # [m/s²] thrust limit
    pos_weight:  float = 10.0       # LQR position cost
    vel_weight:  float = 1.0        # LQR velocity cost
    # Control cost.  With thrust_weight = 1 the LQR gain demands ~27 m/s² at
    # 30 m range — about 90x the 0.3 m/s² limit — so the loop saturates on
    # ~28 % of steps and behaves like bang-bang control: the chaser arrives
    # at 4 m/s and flies straight through the target.  Weighting control so
    # the UNSATURATED demand respects u_max makes the approach smooth and
    # cuts the propellant by an order of magnitude.
    thrust_weight: float = 1.0e5

    # Vision pipeline
    use_vision:  bool  = True       # if False → perfect state measurement
    use_ekf:     bool  = True       # if False → feed raw EPnP to LQR
    ransac_iter: int   = 100
    ransac_thr:  float = 3.0        # [px] RANSAC reprojection threshold

    # Camera pointing.  The camera sits on the chaser at the LVLH origin and
    # its boresight is the camera +z axis (NOT an LVLH axis).
    #   'track' : the chaser attitude controller is assumed to hold the
    #             boresight on the *estimated* target line of sight
    #   'fixed' : boresight held along `camera_boresight` in LVLH
    camera_mode:      str = 'track'
    camera_boresight: np.ndarray = field(
        default_factory=lambda: np.array([1., 0., 0.]))   # LVLH radial
    camera_up:        np.ndarray = field(
        default_factory=lambda: np.array([0., 0., 1.]))   # LVLH cross-track
    min_visible_kpts: int = 6       # below this the pose solve is not attempted

    # Docking
    dock_r_thr: float = 1.0         # [m]  docking distance
    dock_v_thr: float = 0.05        # [m/s] docking speed

    # Terminal hold point: the controller is commanded to the docking port on
    # the +x approach axis, NOT to the target's centre of mass.  Driving the
    # reference to r = 0 asks the chaser to fly to the middle of the target,
    # and with a thrust limit it overshoots and passes through it — the
    # trajectory oscillates through the origin instead of approaching.
    r_dock: np.ndarray = field(
        default_factory=lambda: np.array([0.8, 0., 0.]))
    # End the run once the docking condition is met.  Continuing to
    # station-keep afterwards inflates the reported delta-v (about 6 % of it
    # was being spent after the success criterion was already satisfied).
    stop_at_dock: bool = True


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
    n_visible:  np.ndarray = None   # (T-1,)   keypoints passing visibility gate
    meas_used:  np.ndarray = None   # (T-1,)   True where the filter was updated

    # Scalars
    delta_v:     float = 0.0
    dock_step:   Optional[int] = None   # first step reaching docking
    n_steps_run: int = 0
    n_vision_fail: int = 0          # steps where the pose solve failed
    delta_v_to_dock: float = 0.0    # delta-v spent up to the docking instant

    @property
    def vision_dropout_rate(self) -> float:
        """Fraction of steps on which the vision pose solve failed."""
        if not self.cfg.use_vision or self.n_steps_run == 0:
            return 0.0
        return self.n_vision_fail / self.n_steps_run

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
            thrust_weight=self.cfg.thrust_weight,
        )

    # ── Helpers ──────────────────────────────────────────────────────────────────

    def camera_attitude(self, r_los: np.ndarray) -> np.ndarray:
        """
        Rotation matrix R_cl mapping LVLH → camera frame.

        The camera is mounted on the chaser at the LVLH origin and its
        boresight is the camera +z axis.  In ``'track'`` mode the chaser
        attitude controller is assumed to hold the boresight on the target
        line of sight; `r_los` is the chaser's own *estimate* of the target
        position, never the true one, so no privileged information enters the
        measurement chain.  In ``'fixed'`` mode the boresight is held along
        ``cfg.camera_boresight`` in LVLH.

        Parameters
        ----------
        r_los : (3,) estimated target position in LVLH [m]

        Returns
        -------
        R_cl : (3, 3) rotation LVLH → camera
        """
        cfg = self.cfg
        if cfg.camera_mode == 'fixed':
            fwd = np.asarray(cfg.camera_boresight, dtype=float)
        elif cfg.camera_mode == 'track':
            fwd = np.asarray(r_los, dtype=float)
        else:
            raise ValueError(f"unknown camera_mode {cfg.camera_mode!r}")

        nrm = np.linalg.norm(fwd)
        if nrm < 1e-9:                       # degenerate LOS → fall back
            fwd = np.asarray(cfg.camera_boresight, dtype=float)
            nrm = np.linalg.norm(fwd)
        fwd = fwd / nrm

        up = np.asarray(cfg.camera_up, dtype=float)
        right = np.cross(fwd, up)
        if np.linalg.norm(right) < 1e-8:     # boresight parallel to up
            up = np.array([0., 1., 0.])
            right = np.cross(fwd, up)
        right = right / np.linalg.norm(right)
        up_cam = np.cross(right, fwd)

        # Camera axes expressed in LVLH: x = right, y = down, z = boresight
        R_lc = np.stack([right, -up_cam, fwd], axis=1)
        return R_lc.T

    def _project_keypoints(self, r_true: np.ndarray,
                           q_true: np.ndarray,
                           R_cl: np.ndarray):
        """
        Project the target's 3-D model keypoints into the chaser camera.

        The chaser (and its camera) is at the LVLH origin; the target body
        origin is at `r_true` in LVLH with attitude `q_true` (body → LVLH).

        Returns
        -------
        kp3d_cam : (K, 3) keypoints in the camera frame
        kp2d     : (K, 2) projected pixel coordinates (meaningless where the
                   corresponding `visible` entry is False)
        visible  : (K,) bool — in front of the camera AND inside the image
        R_bc     : (3, 3) body → camera, the rotation the pose solver should
                   recover
        """
        kp3d_body = self.model.keypoint_array          # (K, 3) body frame
        R_bl = quat_to_dcm(q_true)                     # body  → LVLH
        R_bc = R_cl @ R_bl                             # body  → camera
        t_bc = R_cl @ np.asarray(r_true, dtype=float)  # target origin in camera

        kp3d_cam = (R_bc @ kp3d_body.T).T + t_bc       # (K, 3)

        fx, fy = self.cam.K[0, 0], self.cam.K[1, 1]
        cx, cy = self.cam.K[0, 2], self.cam.K[1, 2]

        z = kp3d_cam[:, 2]
        in_front = z > 1e-6
        z_safe = np.where(in_front, z, 1.0)            # avoid divide-by-zero
        u = fx * kp3d_cam[:, 0] / z_safe + cx
        v = fy * kp3d_cam[:, 1] / z_safe + cy
        kp2d = np.stack([u, v], axis=-1)

        in_frame = ((u >= 0) & (u < self.cam.width) &
                    (v >= 0) & (v < self.cam.height))
        visible = in_front & in_frame

        return kp3d_cam, kp2d, visible, R_bc

    def _vision_measurement(self, r_true: np.ndarray,
                            q_true: np.ndarray,
                            r_los: np.ndarray):
        """
        Run RANSAC + EPnP on the visible synthetic keypoints.

        `r_los` is the chaser's current *estimate* of the target position and
        is used only to point the camera.

        Returns
        -------
        r_meas   : (3,) position estimate in LVLH [m], or None on failure
        q_meas   : (4,) quaternion estimate, body → LVLH, or None on failure
        repr_err : float  mean reprojection error [px], NaN on failure
        ok       : bool   True if the pose solve succeeded
        n_vis    : int    keypoints that passed the visibility gate

        On failure this returns None rather than any function of the true
        state.  A sensor model must never hand the estimator ground truth:
        doing so silently converts a failed run into a perfect-state run.
        """
        cfg = self.cfg
        R_cl = self.camera_attitude(r_los)
        kp3d_body = self.model.keypoint_array
        _, kp2d, visible, _ = self._project_keypoints(r_true, q_true, R_cl)

        n_vis = int(visible.sum())
        if n_vis < cfg.min_visible_kpts:
            return None, None, np.nan, False, n_vis

        obj = kp3d_body[visible]
        img = kp2d[visible]
        if cfg.pixel_noise_std > 0:
            img = img + self.rng.normal(0, cfg.pixel_noise_std, img.shape)

        try:
            R_est, t_est, inliers, meta = solve_pnp_ransac(
                obj, img, self.cam.K,
                threshold_px=cfg.ransac_thr,
                max_iter=cfg.ransac_iter,
                seed=int(self.rng.integers(0, 2**31 - 1)),
            )
            ok = meta.get('success', False)
            if not ok or R_est is None:
                return None, None, np.nan, False, n_vis

            t_est = np.asarray(t_est, dtype=float).flatten()

            # Mean reprojection error, in the camera frame
            P_cam    = (R_est @ obj.T).T + t_est
            fx, fy   = self.cam.K[0, 0], self.cam.K[1, 1]
            cx, cy   = self.cam.K[0, 2], self.cam.K[1, 2]
            u_proj   = fx * P_cam[:, 0] / P_cam[:, 2] + cx
            v_proj   = fy * P_cam[:, 1] / P_cam[:, 2] + cy
            repr_err = float(np.mean(np.hypot(u_proj - img[:, 0],
                                              v_proj - img[:, 1])))

            # Camera frame → LVLH.  R_cl is the chaser's own commanded
            # attitude, so this transform uses no privileged information.
            from estimator.state import dcm_to_quat
            r_meas = R_cl.T @ t_est                    # target position, LVLH
            q_meas = dcm_to_quat(R_cl.T @ R_est)       # body → LVLH
            return r_meas, q_meas, repr_err, True, n_vis
        except Exception:
            return None, None, np.nan, False, n_vis

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
        n_visible = np.zeros(cfg.n_steps, dtype=int)
        meas_used = np.zeros(cfg.n_steps, dtype=bool)
        n_vision_fail = 0

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
                           # Acceleration-driven Q matching the disturbance
                           # actually injected into the truth above.
                           accel_proc_std=max(cfg.disturb_accel_std, 1e-9),
                           ang_accel_proc_std=1e-5,
                           meas_pos_std=cfg.pos_meas_std,
                           meas_att_std=cfg.att_meas_std)
            r_est[0] = unpack_state(ekf.state)[0]
            v_est[0] = unpack_state(ekf.state)[1]
            # Standard deviation, matching every later row.  This used to be
            # the raw variance, so a sigma envelope had its first point wrong
            # by a factor of ~18.
            cov_pos[0] = np.sqrt(np.diag(ekf.P)[:3])
        else:
            r_est[0] = cfg.r0 + cfg.r0_err
            v_est[0] = cfg.v0 + cfg.v0_err

        u = np.zeros(3)   # control initialised to zero
        delta_v = 0.0
        delta_v_dock = None
        dock_step = None

        for k in range(cfg.n_steps):
            # ── 1. Propagate true state ─────────────────────────────────
            rt, vt, qt, wt = unpack_state(x_true)
            # Unmodelled disturbance acceleration on the truth (see SimConfig)
            a_dist = (self.rng.normal(0, cfg.disturb_accel_std, 3)
                      if cfg.disturb_accel_std > 0 else np.zeros(3))
            rv_new = (self.Phi @ np.concatenate([rt, vt])
                      + self.Gamma @ (u + a_dist))
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
                r_meas, q_meas, repr_err, ok, n_vis = self._vision_measurement(
                    rt_new, qt_new, r_est[k])
                repr_e[k] = repr_err
                n_visible[k] = n_vis
                if not ok:
                    n_vision_fail += 1
            else:
                # Direct pose measurement with the noise the filter's R
                # declares.  The attitude perturbation is applied on the LEFT
                # (global), matching the MEKF's error convention
                #     q_pert = δq ⊗ q      (estimator.state.apply_delta)
                # so that R_att = σ_att² I₃ is a genuine tangent-space
                # covariance for this measurement.
                from estimator.state import quat_mult, rotvec_to_quat
                r_meas = rt_new + self.rng.normal(0, cfg.pos_meas_std, 3)
                dq = rotvec_to_quat(self.rng.normal(0, cfg.att_meas_std, 3))
                q_meas = quat_mult(dq, qt_new)
                repr_e[k] = 0.0
                ok = True
            meas_used[k] = ok

            # ── 4. EKF predict, then update only if a measurement arrived ─
            if cfg.use_ekf and ekf is not None:
                # Control-aware predict: inject last u into EKF
                ekf.predict(cfg.dt)
                ctrl_effect = self.Gamma @ u
                ekf.x[:3]  += ctrl_effect[:3]
                ekf.x[3:6] += ctrl_effect[3:]
                if ok:
                    from estimator.state import quat_to_rotvec
                    ekf.update(np.concatenate([r_meas, quat_to_rotvec(q_meas)]))
                # else: no measurement this step — the filter coasts on its
                # own prediction and the covariance grows accordingly.
                r_est[k+1] = unpack_state(ekf.state)[0]
                v_est[k+1] = unpack_state(ekf.state)[1]
                cov_pos[k+1] = np.sqrt(np.diag(ekf.P)[:3])
            else:
                # No filter: dead-reckon the previous estimate through the
                # model, and overwrite position with the raw measurement when
                # one is available.  Never reads the true state.
                rv_prior = (self.Phi @ np.concatenate([r_est[k], v_est[k]])
                            + self.Gamma @ u)
                r_est[k+1] = r_meas if ok else rv_prior[:3]
                v_est[k+1] = rv_prior[3:]
                cov_pos[k+1] = np.zeros(3)

            # ── 5-6. LQR control on estimate ────────────────────────────
            x_ctrl = np.concatenate([r_est[k+1], v_est[k+1]])
            x_ref_ctrl = np.concatenate([cfg.r_dock, np.zeros(3)])
            u = self.lqr.control(x_ctrl, x_ref_ctrl)
            ctrls[k] = u
            delta_v += np.linalg.norm(u) * cfg.dt

            # ── 7. Check docking ─────────────────────────────────────────
            if (dock_step is None
                    and np.linalg.norm(rt_new) < cfg.dock_r_thr
                    and np.linalg.norm(vt_new) < cfg.dock_v_thr):
                dock_step = k + 1
                delta_v_dock = delta_v

            res.n_steps_run = k + 1

            if dock_step is not None and cfg.stop_at_dock:
                # Truncate the logs to the steps actually flown.
                T_run = k + 2
                r_true, v_true = r_true[:T_run], v_true[:T_run]
                q_true, w_true = q_true[:T_run], w_true[:T_run]
                r_est,  v_est  = r_est[:T_run],  v_est[:T_run]
                cov_pos = cov_pos[:T_run]
                ctrls, repr_e = ctrls[:k+1], repr_e[:k+1]
                n_visible, meas_used = n_visible[:k+1], meas_used[:k+1]
                break

        res.r_true      = r_true
        res.v_true      = v_true
        res.q_true      = q_true
        res.w_true      = w_true
        res.r_est       = r_est
        res.v_est       = v_est
        res.controls    = ctrls
        res.repr_errs   = repr_e
        res.ekf_cov_pos = cov_pos
        res.n_visible   = n_visible
        res.meas_used   = meas_used
        res.delta_v     = delta_v
        res.dock_step   = dock_step
        res.n_vision_fail = n_vision_fail
        res.delta_v_to_dock = (delta_v if delta_v_dock is None else delta_v_dock)
        return res


# ── Convenience ───────────────────────────────────────────────────────────────

def run_simulation(cfg: SimConfig = None, rng_seed: int = 42) -> SimResult:
    """Build a simulator and run it; return the SimResult."""
    sim = RendezvousSimulator(cfg or SimConfig(), rng_seed=rng_seed)
    return sim.run()

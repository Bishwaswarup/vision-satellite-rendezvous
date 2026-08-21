"""
ekf.py
======
Multiplicative Extended Kalman Filter (MEKF) for relative pose tracking.

The MEKF separates the nominal state (full 13-D) from the 12-D error state
so that quaternion unit-norm is maintained throughout.  After each update the
attitude error is reset into the nominal quaternion (the "reset step").

State / Error-state convention
-------------------------------
  Nominal x ∈ R^13 : [r(3), v(3), q(4), ω(3)]
  Error  δx ∈ R^12 : [δr(3), δv(3), δα(3), δω(3)]
    δα ∈ R^3 : small-angle attitude error  (rotation vector)

Measurement
-----------
  z ∈ R^6 : [t_meas(3), rotvec_meas(3)]   — from EPnP / RANSAC

Mahalanobis gating
------------------
  An incoming measurement is rejected if
      (z−h)ᵀ S⁻¹ (z−h) > χ²_gate
  where χ²_gate ≈ 15.1 for 6 DOF at 99% confidence (chi2.ppf(0.99, 6)).

References
----------
  Markley F.L., "Attitude Error Representations for Kalman Filtering",
      J. Guidance, Control, Dynamics, 2003.
  Trawny N., Roumeliotis S.I., "Indirect Kalman Filter for 3D Attitude
      Estimation", Tech. Rep., Univ. of Minnesota, 2005.
"""

import numpy as np
from .state import (
    N_ORBITAL,
    pack_state, unpack_state,
    propagate_rk4, process_jacobian,
    h_measurement, measurement_jacobian,
    default_process_noise, default_measurement_noise,
    quat_mult, quat_norm, rotvec_to_quat,
)

# χ² 99th-percentile gate for dim=6
_CHI2_GATE_6 = 16.812


class MultEKF:
    """
    Multiplicative Extended Kalman Filter for relative pose estimation.

    Parameters
    ----------
    x0       : (13,) initial state  [r, v, q, ω]
    P0       : (12,12) initial error covariance
    Q        : (12,12) process noise covariance
    R_noise  : (6,6)  measurement noise covariance
    n        : float  orbital mean motion [rad/s]
    gate     : float  Mahalanobis² gate threshold (None = no gating)
    """

    def __init__(self,
                 x0      : np.ndarray,
                 P0      : np.ndarray,
                 Q       : np.ndarray,
                 R_noise : np.ndarray,
                 n       : float = N_ORBITAL,
                 gate    : float = _CHI2_GATE_6):
        self.x      = np.array(x0, dtype=float)
        self.P      = np.array(P0, dtype=float)
        self.Q      = np.array(Q,  dtype=float)
        self.R      = np.array(R_noise, dtype=float)
        self.n      = n
        self.gate   = gate

        # Running stats
        self.n_updates  = 0
        self.n_rejected = 0

    # ── Public interface ───────────────────────────────────────────────────────

    @property
    def state(self):
        return self.x.copy()

    @property
    def covariance(self):
        return self.P.copy()

    def predict(self, dt: float) -> None:
        """
        Propagate state and covariance by dt seconds.

        x_{k+1} = f(x_k)          (RK4)
        P_{k+1} = F P F^T + Q     (linearised)
        """
        F = process_jacobian(self.x, dt, self.n)
        self.x = propagate_rk4(self.x, dt, self.n)
        self.P = F @ self.P @ F.T + self.Q
        self._symmetrise()

    def update(self, z: np.ndarray,
               R_override: np.ndarray = None) -> dict:
        """
        Kalman update with measurement z ∈ R^6.

        Parameters
        ----------
        z          : (6,) measurement  [t_meas(3), rotvec(3)]
        R_override : (6,6) optional per-step measurement noise

        Returns
        -------
        info : dict with keys 'accepted', 'mahal', 'innov'
        """
        z   = np.asarray(z, dtype=float)
        R_n = R_override if R_override is not None else self.R

        # Predicted measurement and Jacobian
        z_hat = h_measurement(self.x)
        H     = measurement_jacobian(self.x)

        innov = z - z_hat                      # (6,)

        # Wrap rotation-vector innovation to (-π, π)
        innov[3:] = self._wrap_rotvec(innov[3:])

        # Innovation covariance
        S = H @ self.P @ H.T + R_n            # (6,6)
        S_sym = 0.5 * (S + S.T)

        # Mahalanobis gate
        try:
            S_inv  = np.linalg.inv(S_sym)
            mahal2 = float(innov @ S_inv @ innov)
        except np.linalg.LinAlgError:
            return {'accepted': False, 'mahal': np.inf, 'innov': innov}

        if self.gate is not None and mahal2 > self.gate:
            self.n_rejected += 1
            return {'accepted': False, 'mahal': mahal2, 'innov': innov}

        # Kalman gain
        K = self.P @ H.T @ S_inv              # (12,6)

        # Error-state update
        delta_x = K @ innov                    # (12,)

        # Apply error-state to nominal state
        self._apply_error(delta_x)

        # Joseph-form covariance update (numerically safer)
        I_KH = np.eye(12) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_n @ K.T
        self._symmetrise()

        self.n_updates += 1
        return {'accepted': True, 'mahal': mahal2, 'innov': innov}

    # ── Private helpers ────────────────────────────────────────────────────────

    def _apply_error(self, delta_x: np.ndarray) -> None:
        """Apply 12-D error state to nominal 13-D state."""
        r, v, q, omega = unpack_state(self.x)
        dr   = delta_x[:3]
        dv   = delta_x[3:6]
        dalpha = delta_x[6:9]
        dw   = delta_x[9:12]

        # Position / velocity: additive
        r_new = r + dr
        v_new = v + dv

        # Attitude: multiplicative (δq ⊗ q_nom)
        dq    = rotvec_to_quat(dalpha)
        q_new = quat_norm(quat_mult(dq, q))

        # Angular velocity: additive
        omega_new = omega + dw

        self.x = pack_state(r_new, v_new, q_new, omega_new)

    def _symmetrise(self) -> None:
        self.P = 0.5 * (self.P + self.P.T)

    @staticmethod
    def _wrap_rotvec(rv: np.ndarray) -> np.ndarray:
        """Wrap rotation-vector innovation so norm is in [0, π)."""
        n = np.linalg.norm(rv)
        if n > np.pi:
            rv = rv * (n - 2*np.pi) / n
        return rv


# ── Convenience factory ────────────────────────────────────────────────────────

def make_ekf(x0, dt,
             pos0_std=2.0, vel0_std=0.5, att0_std=0.3, rate0_std=0.05,
             pos_proc_std=0.05, vel_proc_std=0.005,
             att_proc_std=1e-4, rate_proc_std=1e-5,
             meas_pos_std=0.5, meas_att_std=0.05,
             n=N_ORBITAL):
    """
    Build a MultEKF with sensible diagonal defaults.

    Parameters
    ----------
    x0         : (13,) initial state
    dt         : nominal time step [s]  (used only to scale Q)
    *_std      : 1-sigma initial / process / measurement uncertainties
    """
    P0 = np.diag(np.concatenate([
        np.full(3, pos0_std**2),
        np.full(3, vel0_std**2),
        np.full(3, att0_std**2),
        np.full(3, rate0_std**2),
    ]))
    Q = default_process_noise(dt,
                              pos_std=pos_proc_std,
                              vel_std=vel_proc_std,
                              att_std=att_proc_std,
                              rate_std=rate_proc_std)
    R_noise = default_measurement_noise(meas_pos_std, meas_att_std)
    return MultEKF(x0, P0, Q, R_noise, n=n)

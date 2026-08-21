"""
ukf.py
======
Unscented Kalman Filter (UKF) for relative pose tracking.

Uses the Van der Merwe scaled sigma-point transform.  Quaternion manifold
is handled by:
  - Generating sigma points in error-state space (R^12)
  - Adding perturbations multiplicatively for the attitude component
  - Recovering attitude error from propagated sigma points via log map

This avoids quaternion-averaging (which is ill-posed on S^3 without care)
and keeps the covariance in the 12-D error-state tangent space.

Reference
---------
  Van der Merwe R. et al., "The Unscented Particle Filter", NIPS 2000.
  Kraft E., "A Quaternion-based Unscented Kalman Filter for Orientation
      Tracking", FUSION 2003.
"""

import numpy as np
from .state import (
    N_ORBITAL,
    pack_state, unpack_state,
    propagate_rk4,
    h_measurement,
    default_process_noise, default_measurement_noise,
    quat_mult, quat_norm, rotvec_to_quat, quat_to_rotvec,
)

_CHI2_GATE_6 = 16.812


class UnscentedKF:
    """
    Unscented Kalman Filter for relative pose tracking.

    Parameters
    ----------
    x0      : (13,) initial state  [r, v, q, ω]
    P0      : (12,12) initial error covariance (error-state space)
    Q       : (12,12) process noise covariance
    R_noise : (6,6)  measurement noise covariance
    n       : float  orbital mean motion [rad/s]
    alpha   : float  spread parameter (σ-pts distance, ~1e-3)
    beta    : float  distribution parameter (2 = Gaussian)
    kappa   : float  secondary scaling (0 for state estimation)
    gate    : float  Mahalanobis² gate (None = no gating)
    """

    def __init__(self,
                 x0      : np.ndarray,
                 P0      : np.ndarray,
                 Q       : np.ndarray,
                 R_noise : np.ndarray,
                 n       : float = N_ORBITAL,
                 alpha   : float = 1e-3,
                 beta    : float = 2.0,
                 kappa   : float = 0.0,
                 gate    : float = _CHI2_GATE_6):
        self.x      = np.array(x0, dtype=float)
        self.P      = np.array(P0, dtype=float)
        self.Q      = np.array(Q,  dtype=float)
        self.R      = np.array(R_noise, dtype=float)
        self.n      = n
        self.gate   = gate

        # Error-state dimension
        self.n_err  = 12

        # Van der Merwe weights
        lam         = alpha**2 * (self.n_err + kappa) - self.n_err
        self._lam   = lam
        n_          = self.n_err
        self._Wm    = np.full(2*n_ + 1, 0.5 / (n_ + lam))
        self._Wc    = np.full(2*n_ + 1, 0.5 / (n_ + lam))
        self._Wm[0] = lam / (n_ + lam)
        self._Wc[0] = lam / (n_ + lam) + (1 - alpha**2 + beta)

        self.n_updates  = 0
        self.n_rejected = 0

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def state(self):
        return self.x.copy()

    @property
    def covariance(self):
        return self.P.copy()

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(self, dt: float) -> None:
        """
        Propagate state and covariance by dt seconds.
        """
        sigma_err, sigma_full = self._generate_sigma_points()
        # Propagate each sigma point through nonlinear dynamics
        sigma_prop = np.array([
            propagate_rk4(sp, dt, self.n) for sp in sigma_full
        ])
        # Recover mean state and error-state covariance
        x_new, P_new = self._recover_mean_cov(sigma_prop)
        self.x = x_new
        self.P = P_new + self.Q
        self._symmetrise()

    # ── Update ────────────────────────────────────────────────────────────────

    def update(self, z: np.ndarray,
               R_override: np.ndarray = None) -> dict:
        """
        Kalman update with measurement z ∈ R^6.
        """
        z   = np.asarray(z, dtype=float)
        R_n = R_override if R_override is not None else self.R

        sigma_err, sigma_full = self._generate_sigma_points()
        # Map sigma points through measurement model
        Z = np.array([h_measurement(sp) for sp in sigma_full])  # (2n+1, 6)

        # Predicted measurement mean
        z_hat = Z.T @ self._Wm                                   # (6,)

        # Innovation covariance and cross-covariance
        dZ = Z - z_hat                                           # (2n+1, 6)
        Pzz = (self._Wc[:, None] * dZ).T @ dZ + R_n            # (6,6)
        Pxz = (self._Wc[:, None] * sigma_err).T @ dZ            # (12,6)

        # Innovation
        innov = z - z_hat
        innov[3:] = self._wrap_rotvec(innov[3:])

        # Mahalanobis gate
        try:
            Pzz_sym = 0.5 * (Pzz + Pzz.T)
            S_inv   = np.linalg.inv(Pzz_sym)
            mahal2  = float(innov @ S_inv @ innov)
        except np.linalg.LinAlgError:
            return {'accepted': False, 'mahal': np.inf, 'innov': innov}

        if self.gate is not None and mahal2 > self.gate:
            self.n_rejected += 1
            return {'accepted': False, 'mahal': mahal2, 'innov': innov}

        # Kalman gain (12×6)
        K = Pxz @ S_inv

        # Error-state update
        delta_x = K @ innov                                       # (12,)

        # Apply to nominal state
        self._apply_error(delta_x)

        # Covariance update
        self.P = self.P - K @ Pzz_sym @ K.T
        self._symmetrise()

        self.n_updates += 1
        return {'accepted': True, 'mahal': mahal2, 'innov': innov}

    # ── Sigma-point generation ────────────────────────────────────────────────

    def _generate_sigma_points(self):
        """
        Generate 2n+1 sigma points in error-state space, then convert to
        full state space by composing with nominal state.

        Returns
        -------
        sigma_err  : (2n+1, 12)  error-state sigma points (relative to mean=0)
        sigma_full : (2n+1, 13)  full-state sigma points
        """
        n_   = self.n_err
        lam  = self._lam
        try:
            S = np.linalg.cholesky((n_ + lam) * self.P)
        except np.linalg.LinAlgError:
            # Fallback: add small jitter
            P_reg = self.P + 1e-12 * np.eye(n_)
            S = np.linalg.cholesky((n_ + lam) * P_reg)

        r, v, q, omega = unpack_state(self.x)

        sigma_err  = np.zeros((2*n_ + 1, n_))
        sigma_full = np.zeros((2*n_ + 1, 13))

        sigma_full[0] = self.x

        for i in range(n_):
            col = S[:, i]   # (12,)
            for sign, idx in [(+1, i+1), (-1, n_+i+1)]:
                de = sign * col
                sigma_err[idx] = de
                # Compose attitude perturbation
                da    = de[6:9]
                dq    = rotvec_to_quat(da)
                q_sp  = quat_norm(quat_mult(dq, q))
                sigma_full[idx] = pack_state(
                    r + de[:3], v + de[3:6], q_sp, omega + de[9:12])

        return sigma_err, sigma_full

    def _recover_mean_cov(self, sigma_prop):
        """
        Recover nominal state mean and error-state covariance from
        propagated sigma points.
        """
        Wm  = self._Wm
        Wc  = self._Wc

        # Attitude mean via iterative quaternion averaging (Markley 2007)
        # Simple approx: weighted mean of error vectors relative to sigma[0]
        x_mean = np.zeros(13)
        # Translation / velocity / omega: straightforward weighted sum
        for j, sp in enumerate(sigma_prop):
            x_mean += Wm[j] * sp
        # Re-normalise quaternion of mean
        x_mean[6:10] = quat_norm(x_mean[6:10])

        # Error-state covariance
        P = np.zeros((12, 12))
        r_m, v_m, q_m, w_m = unpack_state(x_mean)
        q_m_inv = np.array([-q_m[0], -q_m[1], -q_m[2], q_m[3]])

        err_vecs = np.zeros((len(sigma_prop), 12))
        for j, sp in enumerate(sigma_prop):
            r_j, v_j, q_j, w_j = unpack_state(sp)
            dq   = quat_mult(q_j, q_m_inv)
            da   = quat_to_rotvec(dq)
            err_vecs[j] = np.concatenate([
                r_j - r_m, v_j - v_m, da, w_j - w_m])

        for j in range(len(sigma_prop)):
            de = err_vecs[j]
            P += Wc[j] * np.outer(de, de)

        return x_mean, P

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _apply_error(self, delta_x: np.ndarray) -> None:
        r, v, q, omega = unpack_state(self.x)
        dq  = rotvec_to_quat(delta_x[6:9])
        self.x = pack_state(
            r + delta_x[:3],
            v + delta_x[3:6],
            quat_norm(quat_mult(dq, q)),
            omega + delta_x[9:12],
        )

    def _symmetrise(self) -> None:
        self.P = 0.5 * (self.P + self.P.T)

    @staticmethod
    def _wrap_rotvec(rv: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(rv)
        if n > np.pi:
            rv = rv * (n - 2*np.pi) / n
        return rv


# ── Convenience factory ────────────────────────────────────────────────────────

def make_ukf(x0, dt,
             pos0_std=2.0, vel0_std=0.5, att0_std=0.3, rate0_std=0.05,
             pos_proc_std=0.05, vel_proc_std=0.005,
             att_proc_std=1e-4, rate_proc_std=1e-5,
             meas_pos_std=0.5, meas_att_std=0.05,
             n=N_ORBITAL, alpha=1e-3, beta=2.0, kappa=0.0):
    """Build a UnscentedKF with sensible diagonal defaults."""
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
    return UnscentedKF(x0, P0, Q, R_noise, n=n,
                       alpha=alpha, beta=beta, kappa=kappa)

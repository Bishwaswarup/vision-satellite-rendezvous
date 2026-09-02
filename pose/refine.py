"""
refine.py
=========
Gauss-Newton non-linear refinement of PnP pose estimates.

Minimises total squared reprojection error:
    E(r, t) = Σ_i || p_i - π(R(r) P_i + t) ||²

where r is the Rodrigues rotation vector (3 DOF), t is translation (3 DOF).

Parameterisation
---------------
  The rotation is carried as a MATRIX and updated on the manifold:

      R <- exp([δω]×) R        t <- t + δt

  rather than as a global Rodrigues vector r with R = exp([r]×).  A global
  r is singular at ||r|| = π — and the canonical camera poses in this project
  (`vision.renderer.look_at_rotation` for an on-axis view) are *exactly*
  θ = π, where the Rodrigues parameterisation and its derivative blow up.
  The increment δω is always near zero, so the local parameterisation is
  exact and well conditioned wherever the optimiser happens to be.

Jacobian
--------
  ∂π/∂P_cam  :  2×3  (perspective projection Jacobian)
  ∂P_cam/∂δω : -[R P_w]×      since  exp([δ]×) v ≈ v + δ × v = v - [v]× δ
  ∂P_cam/∂δt :  I₃

  J_i = [∂π/∂P_cam] @ [-[R P_w]× | I₃]   shape (2, 6)

Update:  δx = -(JᵀJ + λI)⁻¹ Jᵀ e   (damped Levenberg-Marquardt step)

Any step that puts a point behind the camera is assigned infinite cost and
therefore rejected, so the residual and the Jacobian always describe the same
function and the optimiser cannot walk off to a spurious minimum.

Usage
-----
  from pose.refine import GaussNewtonRefiner
  refiner = GaussNewtonRefiner(max_iter=20, tol=1e-6)
  R_ref, t_ref, cost = refiner.refine(R_init, t_init, pts3d, pts2d, K)
"""

import numpy as np


class GaussNewtonRefiner:
    """
    Levenberg-Marquardt / Gauss-Newton pose refiner.

    Parameters
    ----------
    max_iter : int    Maximum LM iterations
    tol      : float  Convergence tolerance on update norm
    lam_init : float  Initial LM damping (0 = pure Gauss-Newton)
    """

    LAM_MIN   = 1e-10     # damping is clamped so it cannot run away in either
    LAM_MAX   = 1e10      # direction over a long sequence of accept/reject
    MIN_DEPTH = 1e-6      # [m] a point at or behind this is a chirality failure

    def __init__(self,
                 max_iter : int   = 20,
                 tol      : float = 1e-7,
                 lam_init : float = 1e-3):
        self.max_iter = max_iter
        self.tol      = tol
        self.lam_init = lam_init

    # ── Public interface ──────────────────────────────────────────────────────
    def refine(self,
               R_init : np.ndarray,
               t_init : np.ndarray,
               pts3d  : np.ndarray,
               pts2d  : np.ndarray,
               K      : np.ndarray,
               weights: np.ndarray = None,
               return_info: bool = False) -> tuple:
        """
        Refine pose from an EPnP / RANSAC initial estimate.

        Parameters
        ----------
        R_init  : (3,3)  initial rotation (world → camera)
        t_init  : (3,)   initial translation [m]
        pts3d   : (N,3)  3D world points [m]
        pts2d   : (N,2)  2D pixel observations [px]
        K       : (3,3)  camera intrinsics
        weights : (N,) or None  per-point weights (inlier = 1, outlier = 0)

        Returns
        -------
        R_ref   : (3,3)  refined rotation
        t_ref   : (3,)   refined translation
        cost    : float  final mean squared reprojection error [px²]
        info    : dict   only when `return_info` — convergence flag, iteration
                  and accepted-step counts, final damping
        """
        pts3d   = np.asarray(pts3d,  dtype=float)
        pts2d   = np.asarray(pts2d,  dtype=float)
        K       = np.asarray(K,      dtype=float)
        R_init  = np.asarray(R_init, dtype=float)
        t_init  = np.asarray(t_init, dtype=float).ravel()
        N       = len(pts3d)

        if weights is None:
            weights = np.ones(N)
        weights = np.asarray(weights, dtype=float)

        R = R_init.copy()
        t = t_init.copy()

        lam = self.lam_init
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        cost = self._cost(R, t, pts3d, pts2d, fx, fy, cx, cy, weights)
        n_accept, converged, n_it = 0, False, 0

        # If the INITIAL pose already puts points behind the camera there is no
        # valid linearisation to take, so report the failure instead of
        # iterating on a meaningless Jacobian.
        if not np.isfinite(cost):
            if return_info:
                return R, t, float(cost), {
                    'converged': False, 'iterations': 0, 'n_accepted': 0,
                    'cost': float(cost), 'lam': float(lam), 'success': False,
                }
            return R, t, float(cost)

        for n_it in range(1, self.max_iter + 1):
            J, e = self._jacobian_and_residual(
                R, t, pts3d, pts2d, fx, fy, cx, cy, weights)

            JtJ = J.T @ J
            Jte = J.T @ e
            try:
                delta = np.linalg.solve(JtJ + lam * np.eye(6), -Jte)
            except np.linalg.LinAlgError:
                lam = min(lam * 10.0, self.LAM_MAX)
                continue
            if not np.all(np.isfinite(delta)):
                lam = min(lam * 10.0, self.LAM_MAX)
                continue

            # Manifold update: left-multiplicative increment on SO(3)
            R_new = _rodrigues_to_dcm(delta[:3]) @ R
            t_new = t + delta[3:]

            cost_new = self._cost(R_new, t_new, pts3d, pts2d,
                                  fx, fy, cx, cy, weights)

            if cost_new < cost:
                R, t = R_new, t_new
                cost = cost_new
                lam  = max(lam * 0.1, self.LAM_MIN)
                n_accept += 1
                # Convergence is only meaningful after an ACCEPTED step.  A run
                # of rejections shrinks delta by 10x each time and would
                # otherwise trigger a false "converged" at a high cost.
                d_rot = np.linalg.norm(delta[:3])                  # [rad]
                d_tra = np.linalg.norm(delta[3:])                  # [m]
                scale = 1.0 + np.linalg.norm(t)
                if d_rot < self.tol and d_tra < self.tol * scale:
                    converged = True
                    break
            else:
                lam = min(lam * 10.0, self.LAM_MAX)
                if lam >= self.LAM_MAX:
                    break

        # Re-orthonormalise against accumulated round-off
        U, _, Vt = np.linalg.svd(R)
        R = U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt

        if return_info:
            info = {
                'converged'  : bool(converged),
                'iterations' : int(n_it),
                'n_accepted' : int(n_accept),
                'cost'       : float(cost),
                'lam'        : float(lam),
                'success'    : bool(np.isfinite(cost)),
            }
            return R, t, float(cost), info
        return R, t, float(cost)

    # ── Internal ──────────────────────────────────────────────────────────────
    @classmethod
    def _cost(cls, R, t, pts3d, pts2d, fx, fy, cx, cy, weights):
        """
        Mean weighted squared reprojection error, or +inf if the pose puts any
        weighted point at or behind the camera.

        Returning inf (rather than clamping Z) is what keeps the optimiser
        honest: a step through the image plane is rejected instead of being
        scored against a clamped, fictitious cost that the Jacobian does not
        describe.
        """
        P_cam = (R @ pts3d.T).T + t
        z = P_cam[:, 2]
        if np.any((z <= cls.MIN_DEPTH) & (weights > 0)):
            return np.inf
        u_hat = fx * P_cam[:, 0] / z + cx
        v_hat = fy * P_cam[:, 1] / z + cy
        err   = (u_hat - pts2d[:, 0])**2 + (v_hat - pts2d[:, 1])**2
        return float((weights * err).mean())

    @staticmethod
    def _jacobian_and_residual(R, t, pts3d, pts2d, fx, fy, cx, cy, weights):
        """
        Stacked Jacobian J (2N, 6) and residual e (2N,) at the current pose.

        Columns 0:3 are the rotation increment δω of the left-multiplicative
        manifold update, columns 3:6 the translation increment.  The same Z is
        used for the residual and the Jacobian — evaluation only ever happens
        at a pose that passed the chirality test in `_cost`.
        """
        N = len(pts3d)
        P_cam = (R @ pts3d.T).T + t      # (N, 3)
        RP    = P_cam - t                # (N, 3) = R @ P_world, rotated only

        X, Y, Z = P_cam[:, 0], P_cam[:, 1], P_cam[:, 2]
        res_u = fx * X / Z + cx - pts2d[:, 0]
        res_v = fy * Y / Z + cy - pts2d[:, 1]

        J = np.zeros((2 * N, 6))
        e = np.zeros(2 * N)

        for i in range(N):
            w = np.sqrt(max(weights[i], 0.0))
            Zi = Z[i]

            # ∂π/∂P_cam
            dpi_dP = np.array([
                [fx / Zi, 0.0,     -fx * X[i] / Zi**2],
                [0.0,     fy / Zi, -fy * Y[i] / Zi**2],
            ])

            # ∂P_cam/∂δω = -[R P_w]×   and   ∂P_cam/∂δt = I₃
            J_rot = dpi_dP @ (-_skew(RP[i]))
            J_tra = dpi_dP

            J[2*i,     :3] = w * J_rot[0]
            J[2*i,     3:] = w * J_tra[0]
            J[2*i + 1, :3] = w * J_rot[1]
            J[2*i + 1, 3:] = w * J_tra[1]
            e[2*i]     = w * res_u[i]
            e[2*i + 1] = w * res_v[i]

        return J, e


def _skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix [v]× such that [v]× a = v × a."""
    return np.array([
        [0.0,  -v[2],  v[1]],
        [v[2],  0.0,  -v[0]],
        [-v[1], v[0],  0.0],
    ])


# ── Rodrigues helpers ─────────────────────────────────────────────────────────

def _rodrigues_to_dcm(r: np.ndarray) -> np.ndarray:
    """Rodrigues vector → rotation matrix (Rodrigues' formula)."""
    r = np.asarray(r, dtype=float)
    theta = np.linalg.norm(r)
    if theta < 1e-12:
        return np.eye(3)
    k = r / theta
    K_skew = np.array([
        [ 0.,    -k[2],  k[1]],
        [ k[2],  0.,    -k[0]],
        [-k[1],  k[0],  0.  ],
    ])
    return (np.eye(3) + np.sin(theta) * K_skew
            + (1 - np.cos(theta)) * K_skew @ K_skew)


def _dcm_to_rodrigues(R: np.ndarray) -> np.ndarray:
    """
    Rotation matrix → Rodrigues vector.

    The naive formula divides the off-diagonal terms by 2 sin θ, which is
    singular at θ = π — and R + I is exactly rank 1 there, so the axis has to
    come from a different route.  At θ = π, R = 2aaᵀ - I, hence

        R + I = 2 a aᵀ

    whose columns are all parallel to the axis a.  Taking the column of
    largest norm and fixing the sign from the (still informative) off-diagonal
    differences recovers a stably.  Without this branch, `look_at_rotation`
    poses (trace = -1 exactly) return [0, 0, 0] — a 180° error.
    """
    R = np.asarray(R, dtype=float)

    # w = 2 sin(theta) * axis, so ||w||/2 = sin(theta) and (tr-1)/2 = cos(theta).
    # Taking theta = atan2(sin, cos) is well conditioned for every angle,
    # unlike arccos(cos), whose derivative is unbounded at 0 and pi.
    w = np.array([R[2, 1] - R[1, 2],
                  R[0, 2] - R[2, 0],
                  R[1, 0] - R[0, 1]])
    sin_theta = np.linalg.norm(w) / 2.0
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arctan2(sin_theta, cos_theta)

    if theta < 1e-10:
        return np.zeros(3)

    if sin_theta > 1e-6:
        return theta * (w / (2.0 * sin_theta))

    # θ ≈ π: recover the axis from the columns of R + I = 2 a aᵀ
    S = R + np.eye(3)
    col = int(np.argmax(np.linalg.norm(S, axis=0)))
    axis = S[:, col]
    nrm = np.linalg.norm(axis)
    if nrm < 1e-12:                       # should not happen for a valid R
        return np.zeros(3)
    axis = axis / nrm

    # Sign: at exactly θ = π, r and -r describe the same rotation, so any
    # consistent choice is valid.  Just off π the off-diagonal differences
    # still carry the sign, so use them when they are informative.
    if np.dot(w, axis) < 0:
        axis = -axis
    elif np.linalg.norm(w) < 1e-12:
        # Exactly π — pick a canonical sign so the map is deterministic.
        for c in axis:
            if abs(c) > 1e-12:
                if c < 0:
                    axis = -axis
                break

    return theta * axis


def _rodrigues_jacobian(r: np.ndarray) -> np.ndarray:
    """
    Analytic Jacobian of R(r) w.r.t. r.
    Returns dR: shape (3, 3, 3) where dR[i, j, k] = ∂R[i,j]/∂r[k].
    """
    r = np.asarray(r, dtype=float)
    theta = np.linalg.norm(r)
    if theta < 1e-12:
        # Near identity: ∂R/∂r ≈ [e_k]× (skew-sym generators)
        dR = np.zeros((3, 3, 3))
        dR[:, :, 0] = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])
        dR[:, :, 1] = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
        dR[:, :, 2] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
        return dR

    k = r / theta
    s, c = np.sin(theta), np.cos(theta)
    K = np.array([
        [ 0.,    -k[2],  k[1]],
        [ k[2],  0.,    -k[0]],
        [-k[1],  k[0],  0.  ],
    ])
    R = np.eye(3) + s * K + (1 - c) * K @ K

    dR = np.zeros((3, 3, 3))
    for m in range(3):
        # ∂k/∂r_m and ∂theta/∂r_m
        dtheta_drm = r[m] / theta
        dk_drm     = (np.eye(3)[m] - k[m] * k) / theta  # (3,)

        # ∂K/∂r_m  (skew of dk_drm)
        dK_drm = np.array([
            [0.,           -dk_drm[2],  dk_drm[1]],
            [dk_drm[2],   0.,          -dk_drm[0]],
            [-dk_drm[1],  dk_drm[0],   0.        ],
        ])

        dKK_drm = dK_drm @ K + K @ dK_drm

        dR[:, :, m] = (c * dtheta_drm * K + s * dK_drm
                       + s * dtheta_drm * K @ K + (1 - c) * dKK_drm)

    return dR


# ── Convenience function ──────────────────────────────────────────────────────

def refine_pose(R_init  : np.ndarray,
                t_init  : np.ndarray,
                pts3d   : np.ndarray,
                pts2d   : np.ndarray,
                K       : np.ndarray,
                weights : np.ndarray = None,
                max_iter: int        = 20) -> tuple:
    """
    One-shot Gauss-Newton refinement wrapper.
    Returns (R_ref, t_ref, cost).
    """
    return GaussNewtonRefiner(max_iter=max_iter).refine(
        R_init, t_init, pts3d, pts2d, K, weights)

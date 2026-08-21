"""
refine.py
=========
Gauss-Newton non-linear refinement of PnP pose estimates.

Minimises total squared reprojection error:
    E(r, t) = Σ_i || p_i - π(R(r) P_i + t) ||²

where r is the Rodrigues rotation vector (3 DOF), t is translation (3 DOF).

Jacobian
--------
  ∂π/∂P_cam  :  2×3  (perspective projection Jacobian)
  ∂P_cam/∂r  :  3×3  (Rodrigues derivative, analytic)
  ∂P_cam/∂t  :  I₃   (identity)

  J_i = [∂π/∂P_cam] @ [∂P_cam/∂r | ∂P_cam/∂t]   shape (2, 6)

Update:  δx = -(JᵀJ + λI)⁻¹ Jᵀ e   (damped Levenberg-Marquardt step)

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
               weights: np.ndarray = None) -> tuple:
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

        # Parameterise as Rodrigues vector
        r = _dcm_to_rodrigues(R_init)
        t = t_init.copy()

        lam = self.lam_init
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        cost = self._cost(r, t, pts3d, pts2d, fx, fy, cx, cy, weights)

        for _ in range(self.max_iter):
            J, e = self._jacobian_and_residual(
                r, t, pts3d, pts2d, fx, fy, cx, cy, weights)

            JtJ = J.T @ J
            Jte = J.T @ e
            delta = np.linalg.solve(JtJ + lam * np.eye(6), -Jte)

            r_new = r + delta[:3]
            t_new = t + delta[3:]

            cost_new = self._cost(r_new, t_new, pts3d, pts2d,
                                  fx, fy, cx, cy, weights)

            if cost_new < cost:
                r, t  = r_new, t_new
                cost  = cost_new
                lam  *= 0.1
            else:
                lam  *= 10.0

            if np.linalg.norm(delta) < self.tol:
                break

        R_ref = _rodrigues_to_dcm(r)
        return R_ref, t, float(cost)

    # ── Internal ──────────────────────────────────────────────────────────────
    @staticmethod
    def _cost(r, t, pts3d, pts2d, fx, fy, cx, cy, weights):
        R = _rodrigues_to_dcm(r)
        P_cam = (R @ pts3d.T).T + t
        z = np.maximum(P_cam[:, 2], 1e-6)
        u_hat = fx * P_cam[:, 0] / z + cx
        v_hat = fy * P_cam[:, 1] / z + cy
        err   = (u_hat - pts2d[:, 0])**2 + (v_hat - pts2d[:, 1])**2
        return float((weights * err).mean())

    @staticmethod
    def _jacobian_and_residual(r, t, pts3d, pts2d, fx, fy, cx, cy, weights):
        """
        Compute stacked Jacobian J (2N, 6) and residual e (2N,).
        """
        N = len(pts3d)
        R = _rodrigues_to_dcm(r)
        dR_dr = _rodrigues_jacobian(r)   # (3, 3, 3): dR[i,j]/dr[k]

        P_cam = (R @ pts3d.T).T + t      # (N, 3)
        X = P_cam[:, 0]
        Y = P_cam[:, 1]
        Z = np.maximum(P_cam[:, 2], 1e-6)

        u_hat = fx * X / Z + cx
        v_hat = fy * Y / Z + cy
        res_u = u_hat - pts2d[:, 0]
        res_v = v_hat - pts2d[:, 1]

        J = np.zeros((2 * N, 6))
        e = np.zeros(2 * N)

        for i in range(N):
            w = np.sqrt(max(weights[i], 0.0))
            Xi, Yi, Zi = P_cam[i]

            # ∂u/∂P_cam = [fx/Z, 0, -fx*X/Z²]
            # ∂v/∂P_cam = [0, fy/Z, -fy*Y/Z²]
            dpi_dP = np.array([
                [fx/Zi,   0.,     -fx*Xi/Zi**2],
                [0.,      fy/Zi,  -fy*Yi/Zi**2],
            ])

            # ∂P_cam_i/∂r = dR/dr @ P_world_i  → shape (3, 3)
            Pw = pts3d[i]
            dP_dr = np.array([
                dR_dr[:, :, k] @ Pw for k in range(3)
            ]).T                                # (3, 3)

            # ∂P_cam/∂t = I
            dP_dt = np.eye(3)

            J_r = dpi_dP @ dP_dr   # (2, 3)
            J_t = dpi_dP @ dP_dt   # (2, 3)

            J[2*i,   :3] = w * J_r[0]
            J[2*i,   3:] = w * J_t[0]
            J[2*i+1, :3] = w * J_r[1]
            J[2*i+1, 3:] = w * J_t[1]
            e[2*i]     = w * res_u[i]
            e[2*i + 1] = w * res_v[i]

        return J, e


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
    """Rotation matrix → Rodrigues vector."""
    R = np.asarray(R, dtype=float)
    # cos(θ) = (tr(R) - 1) / 2
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if abs(theta) < 1e-10:
        return np.zeros(3)
    axis = np.array([R[2,1] - R[1,2],
                     R[0,2] - R[2,0],
                     R[1,0] - R[0,1]]) / (2.0 * np.sin(theta))
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

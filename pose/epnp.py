"""
epnp.py
=======
EPnP — Efficient Perspective-n-Point pose estimation.

Reference
---------
  Lepetit V., Moreno-Noguer F., Fua P.,
  "EPnP: An Accurate O(n) Solution to the PnP Problem",
  IJCV 81(2), 2009.  DOI: 10.1007/s11263-008-0152-6

Algorithm summary
-----------------
  Given N ≥ 4 correspondences  {P_i ↔ p_i}  where
    P_i ∈ ℝ³  — 3D world-frame point
    p_i ∈ ℝ²  — 2D observed pixel  [u, v]

  1. Choose 4 control points  c_0..c_3  (centroid + PCA axes of P_i)
  2. Express each P_i as weighted sum:  P_i = Σ_j α_ij c_j  (barycentric)
  3. Camera-frame counterparts:  P_i^c = Σ_j α_ij c_j^c
  4. Projection gives 2 linear equations per point in the 12 unknowns
     x = [c_0^c; c_1^c; c_2^c; c_3^c] ∈ ℝ¹²
  5. Stack → M x = 0  (M is 2N×12)
  6. x lives in the null space of M; solve for β coefficients via L*β = ρ
  7. Recover R, t from control points in camera frame (sign fix via chirality)

Returns
-------
  R : (3,3)  rotation world → camera
  t : (3,)   translation [m] (camera frame, i.e. P_cam = R @ P_world + t)
  reprojection_errors : (N,) per-point reprojection error [px]

Usage
-----
  from pose.epnp import EPnPSolver
  solver = EPnPSolver()
  R, t, err = solver.solve(pts3d, pts2d, K)
"""

import numpy as np


class EPnPSolver:
    """
    EPnP pose solver.

    Parameters
    ----------
    max_reprojection_error : float
        Points with reprojection error above this [px] are flagged (not used
        inside EPnP itself — passed out for the caller to use).
    """

    def __init__(self, max_reprojection_error: float = 8.0):
        self.max_repr_err = max_reprojection_error

    # ── Public interface ──────────────────────────────────────────────────────
    def solve(self,
              pts3d : np.ndarray,
              pts2d : np.ndarray,
              K     : np.ndarray) -> tuple:
        """
        Solve PnP for N ≥ 4 point correspondences.

        Parameters
        ----------
        pts3d : (N, 3)  3D world-frame points [m]
        pts2d : (N, 2)  2D pixel observations  [px]
        K     : (3, 3)  camera intrinsic matrix

        Returns
        -------
        R    : (3, 3)  rotation matrix  (world → camera)
        t    : (3,)    translation [m]  (P_cam = R @ P_world + t)
        repr_errs : (N,) reprojection errors [px]
        success   : bool  (False if degenerate)
        """
        pts3d = np.asarray(pts3d, dtype=float)
        pts2d = np.asarray(pts2d, dtype=float)
        K     = np.asarray(K,     dtype=float)
        N     = len(pts3d)

        if N < 4:
            raise ValueError(f"EPnP requires N≥4 correspondences, got {N}")

        # 1. Control points
        ctrl  = self._choose_control_points(pts3d)   # (4, 3)

        # 2. Barycentric coordinates  α  (N, 4)
        alpha = self._barycentric(pts3d, ctrl)       # (N, 4)

        # 3. Build M matrix  (2N, 12)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        M = self._build_M(alpha, pts2d, fx, fy, cx, cy)

        # 4. Null space of M via SVD (take last 4 columns of V^T)
        _, _, Vt = np.linalg.svd(M, full_matrices=True)
        # Null vectors: rows of Vt with smallest singular values
        # EPnP uses up to N=4 null vectors; start with N=1 (simplest)
        V = Vt[-4:, :].T           # (12, 4)  null space basis

        # 5. Solve for betas — try N=1,2,4 and pick best
        R_best, t_best, err_best = None, None, np.inf

        for n_null in [1, 2, 4]:
            try:
                R_c, t_c = self._solve_for_betas(V[:, -n_null:], ctrl, alpha, N)
                if R_c is None:
                    continue
                errs = self._reprojection_errors(pts3d, pts2d, R_c, t_c, K)
                mean_err = errs.mean()
                if mean_err < err_best:
                    R_best, t_best, err_best = R_c, t_c, mean_err
                    repr_errs = errs
            except Exception:
                continue

        if R_best is None:
            return None, None, np.full(N, np.inf), False

        return R_best, t_best, repr_errs, True

    # ── Step 1: Control points ────────────────────────────────────────────────
    @staticmethod
    def _choose_control_points(pts3d: np.ndarray) -> np.ndarray:
        """
        4 control points:
          c_0 = centroid of pts3d
          c_1, c_2, c_3 = centroid ± principal component axes (scaled)
        """
        c0 = pts3d.mean(axis=0)
        centered = pts3d - c0
        # SVD for PCA
        _, s, Vt = np.linalg.svd(centered, full_matrices=False)
        # Scale by sqrt(eigenvalue / N) so control points span the cloud
        scale = s / np.sqrt(len(pts3d))
        c1 = c0 + scale[0] * Vt[0]
        c2 = c0 + scale[1] * Vt[1]
        c3 = c0 + scale[2] * Vt[2]
        return np.array([c0, c1, c2, c3])

    # ── Step 2: Barycentric coordinates ───────────────────────────────────────
    @staticmethod
    def _barycentric(pts3d: np.ndarray,
                     ctrl : np.ndarray) -> np.ndarray:
        """
        Express each P_i in barycentric coordinates w.r.t. the 4 control points.
        P_i = α_i0 c_0 + α_i1 c_1 + α_i2 c_2 + α_i3 c_3
        with Σ_j α_ij = 1  (affine combination).

        Uses lstsq instead of inv for robustness when control points are
        nearly coplanar (e.g. in 4-point minimal samples).
        """
        N  = len(pts3d)
        # Solve:  [c_1-c_0, c_2-c_0, c_3-c_0] α_vec = P_i - c_0
        #   where α_vec = [α_i1, α_i2, α_i3],  α_i0 = 1 - sum(α_vec)
        C    = (ctrl[1:] - ctrl[0]).T          # (3, 3)
        diff = pts3d - ctrl[0]                 # (N, 3)
        # Solve C @ α_vec.T = diff.T  →  α_vec (N, 3)
        alpha_vec, _, _, _ = np.linalg.lstsq(C, diff.T, rcond=None)
        alpha_vec = alpha_vec.T                # (N, 3)
        alpha = np.zeros((N, 4))
        alpha[:, 1:] = alpha_vec
        alpha[:, 0]  = 1.0 - alpha_vec.sum(axis=1)
        return alpha

    # ── Step 3: Build M matrix ────────────────────────────────────────────────
    @staticmethod
    def _build_M(alpha, pts2d, fx, fy, cx, cy) -> np.ndarray:
        """
        Build 2N×12 matrix M from projection equations.
        For each point i, two rows:
          [α_i0 fx, 0, α_i0(cx-u_i),  α_i1 fx, 0, α_i1(cx-u_i),  ...]
          [0, α_i0 fy, α_i0(cy-v_i),  0, α_i1 fy, α_i1(cy-v_i),  ...]
        """
        N = len(alpha)
        M = np.zeros((2 * N, 12))
        for i in range(N):
            u, v = pts2d[i]
            for j in range(4):
                a = alpha[i, j]
                col = j * 3
                # Row 2i: u equation
                M[2*i,   col]     =  a * fx
                M[2*i,   col + 1] =  0.0
                M[2*i,   col + 2] =  a * (cx - u)
                # Row 2i+1: v equation
                M[2*i+1, col]     =  0.0
                M[2*i+1, col + 1] =  a * fy
                M[2*i+1, col + 2] =  a * (cy - v)
        return M

    # ── Step 4/5: Solve for betas → control points in camera frame ────────────
    def _solve_for_betas(self, V_null, ctrl, alpha, N):
        """
        Recover control points in camera frame from null space vectors.

        For n_null = 1:  x = β v₁  →  solve β from ||c_j^c - c_k^c||² = ||c_j - c_k||²
        For n_null = 2:  x = β₁ v₁ + β₂ v₂  →  solve L*[β₁²,β₁β₂,β₂²]ᵀ = ρ
        For n_null = 4:  x = Σ βᵢ vᵢ  →  solve L*β = ρ  (linearised)
        """
        n = V_null.shape[1]

        if n == 1:
            # x = β * v → estimate β by matching inter-control distances
            v = V_null[:, 0]
            # Try positive and negative β
            betas = self._dist_beta_n1(v, ctrl)
            best  = None
            for b in betas:
                ctrl_cam = (b * v).reshape(4, 3)
                if ctrl_cam[:, 2].min() > 0:    # chirality
                    best = ctrl_cam
                    break
            if best is None:
                return None, None
            ctrl_cam = best

        elif n == 2:
            v1, v2 = V_null[:, 0], V_null[:, 1]
            betas = self._solve_betas_n2(v1, v2, ctrl)
            ctrl_cam = (betas[0]*v1 + betas[1]*v2).reshape(4, 3)
            if ctrl_cam[:, 2].mean() < 0:
                ctrl_cam = (-betas[0]*v1 - betas[1]*v2).reshape(4, 3)

        else:   # n == 4
            L, rho = self._build_L_rho(V_null, ctrl)
            # Solve L*β = ρ in least-squares sense
            betas, _, _, _ = np.linalg.lstsq(L, rho, rcond=None)
            ctrl_cam = V_null @ betas.reshape(-1, 1)
            ctrl_cam = ctrl_cam.reshape(4, 3)
            if ctrl_cam[:, 2].mean() < 0:
                ctrl_cam = -ctrl_cam

        # Recover R, t from control points
        return self._recover_pose(ctrl_cam, ctrl, alpha)

    def _dist_beta_n1(self, v, ctrl):
        """Estimate β (n=1) from mean of pairwise distances."""
        betas = []
        v_ctrl = v.reshape(4, 3)
        for i in range(4):
            for j in range(i+1, 4):
                d_world = np.linalg.norm(ctrl[i] - ctrl[j])
                d_cam   = np.linalg.norm(v_ctrl[i] - v_ctrl[j])
                if d_cam > 1e-10:
                    betas.append(d_world / d_cam)
        if not betas:
            return [1.0, -1.0]
        b = float(np.median(betas))
        return [b, -b]

    def _solve_betas_n2(self, v1, v2, ctrl):
        """Solve for β₁, β₂ (n=2) via distance constraints → L*[β₁²,β₁β₂,β₂²] = ρ."""
        L, rho = self._build_L_rho(np.column_stack([v1, v2]), ctrl,
                                    cross_terms=True)
        try:
            x, _, _, _ = np.linalg.lstsq(L, rho, rcond=None)
        except Exception:
            return [1.0, 0.0]
        b11, b12, b22 = x[0], x[1], x[2]
        # β₁ = sqrt(|b11|), β₂ = b12/β₁ or sqrt(|b22|) sign-matched
        b1 = np.sqrt(abs(b11))
        if b1 < 1e-10:
            b2 = np.sqrt(abs(b22))
            return [0.0, b2]
        b2 = b12 / b1
        return [b1, b2]

    @staticmethod
    def _build_L_rho(V_null, ctrl, cross_terms=False):
        """
        Build constraint matrix L and RHS ρ from inter-control distances.
        Pairs (i,j): || Σ_k β_k (v_k[i] - v_k[j]) ||² = ||c_i - c_j||²
        """
        n   = V_null.shape[1]
        pairs = [(i, j) for i in range(4) for j in range(i+1, 4)]
        V4  = V_null.reshape(4, 3, n)   # (ctrl_idx, xyz, null_dim)

        L_rows, rho_vals = [], []
        for (i, j) in pairs:
            dv = V4[i] - V4[j]           # (3, n) differences
            d2_world = np.sum((ctrl[i] - ctrl[j])**2)
            if cross_terms and n == 2:
                # products: β₁², β₁β₂, β₂²
                row = [
                    np.dot(dv[:, 0], dv[:, 0]),
                    2*np.dot(dv[:, 0], dv[:, 1]),
                    np.dot(dv[:, 1], dv[:, 1]),
                ]
            else:
                # general: β_k * β_l terms
                row = []
                for k in range(n):
                    for l in range(k, n):
                        factor = 1.0 if k == l else 2.0
                        row.append(factor * np.dot(dv[:, k], dv[:, l]))
            L_rows.append(row)
            rho_vals.append(d2_world)
        return np.array(L_rows), np.array(rho_vals)

    # ── R, t recovery ─────────────────────────────────────────────────────────
    @staticmethod
    def _recover_pose(ctrl_cam, ctrl_world, alpha):
        """
        Given control points in camera frame, recover R and t via
        weighted Procrustes (SVD-based absolute orientation).
        """
        # Reconstruct 3D points in camera frame using barycentric coords
        N = len(alpha)
        pts_cam   = alpha @ ctrl_cam      # (N, 3)
        pts_world = alpha @ ctrl_world    # (N, 3)

        # Centroid
        mu_cam   = pts_cam.mean(axis=0)
        mu_world = pts_world.mean(axis=0)

        A = (pts_cam   - mu_cam).T    @ \
            (pts_world - mu_world)        # (3, 3) cross-covariance

        U, _, Vt = np.linalg.svd(A)
        # Ensure proper rotation (det = +1)
        d = np.linalg.det(U @ Vt)
        D = np.diag([1., 1., d])
        R = U @ D @ Vt                     # world → camera

        t = mu_cam - R @ mu_world

        return R, t

    # ── Reprojection error ────────────────────────────────────────────────────
    @staticmethod
    def _reprojection_errors(pts3d, pts2d, R, t, K) -> np.ndarray:
        """Per-point reprojection error [px]."""
        P_cam = (R @ pts3d.T).T + t        # (N, 3)
        # Avoid division by zero
        z = np.maximum(P_cam[:, 2], 1e-6)
        u_hat = K[0, 0] * P_cam[:, 0] / z + K[0, 2]
        v_hat = K[1, 1] * P_cam[:, 1] / z + K[1, 2]
        du = u_hat - pts2d[:, 0]
        dv = v_hat - pts2d[:, 1]
        return np.sqrt(du**2 + dv**2)


# ── Convenience function ──────────────────────────────────────────────────────

def solve_epnp(pts3d: np.ndarray,
               pts2d: np.ndarray,
               K    : np.ndarray) -> tuple:
    """
    Single-call EPnP wrapper.

    Returns (R, t, reprojection_errors, success).
    """
    return EPnPSolver().solve(pts3d, pts2d, K)

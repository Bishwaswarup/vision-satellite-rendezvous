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

  1. Choose control points (centroid + PCA axes of P_i).  Four of them for a
     general 3-D cloud; THREE when the cloud is coplanar (Lepetit sec. 3.5) —
     a 4th control point would coincide with the centroid, making the
     barycentric system singular and zeroing three columns of M.
  2. Express each P_i as weighted sum:  P_i = Σ_j α_ij c_j  (barycentric)
  3. Camera-frame counterparts:  P_i^c = Σ_j α_ij c_j^c
  4. Projection gives 2 linear equations per point in the 12 unknowns
     x = [c_0^c; c_1^c; c_2^c; c_3^c] ∈ ℝ¹²
  5. Stack → M x = 0  (M is 2N×12)
  6. x lives in the null space of M; solve for the β coefficients from the
     inter-control-point distance constraints, then refine them with
     Gauss-Newton (Lepetit sec. 3.3)
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
        A solution whose mean reprojection error exceeds this [px] is reported
        as a failure (`success=False`).
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

        # 4. Null space of M via SVD.  The dimension of x is 3*n_ctrl, so the
        #    number of candidate null vectors is capped accordingly (4 for a
        #    general cloud, 3 for a coplanar one).
        n_ctrl  = ctrl.shape[0]
        n_max   = 4 if n_ctrl == 4 else 3
        _, _, Vt = np.linalg.svd(M, full_matrices=True)
        V = Vt[-n_max:, :].T            # (3*n_ctrl, n_max) null-space basis

        # 5. Solve for the betas — try each null-space dimension, keep the
        #    candidate with the smallest reprojection error.
        R_best, t_best, err_best = None, None, np.inf
        repr_errs = np.full(N, np.inf)

        for n_null in range(1, n_max + 1):
            try:
                R_c, t_c = self._solve_for_betas(V[:, -n_null:], ctrl, alpha, N)
                if R_c is None:
                    continue
                errs = self._reprojection_errors(pts3d, pts2d, R_c, t_c, K)
                mean_err = float(errs.mean())
                if np.isfinite(mean_err) and mean_err < err_best:
                    R_best, t_best, err_best = R_c, t_c, mean_err
                    repr_errs = errs
            except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                # A specific guard, not a bare `except Exception`: the N=4
                # branch used to raise a shape error on every call and the
                # broad catch hid it completely.
                continue

        if R_best is None:
            return None, None, np.full(N, np.inf), False

        # A pose whose own reprojection error is implausible is a failure, not
        # a solution.  Without this the planar/degenerate cases return
        # success=True alongside errors of order 1e9 px.
        success = bool(err_best < self.max_repr_err)

        return R_best, t_best, repr_errs, success

    # ── Step 1: Control points ────────────────────────────────────────────────
    @staticmethod
    def _choose_control_points(pts3d: np.ndarray,
                              planar_tol: float = 1e-8) -> np.ndarray:
        """
        Control points: centroid plus the scaled principal axes of the cloud.

        Returns 4 control points for a general 3-D cloud and **3** for a
        coplanar one (Lepetit sec. 3.5).  This is not an optimisation: for a
        coplanar cloud the third singular value is zero, so a 4th control point
        would equal the centroid, `C = (ctrl[1:] - ctrl[0]).T` would be
        singular, and the barycentric solve would silently return alpha[:,3]=0
        — which zeroes three columns of M and makes its null space
        artificially 4-dimensional.

        Parameters
        ----------
        pts3d      : (N, 3) world points
        planar_tol : relative tolerance on s[2]/s[0] below which the cloud is
                     treated as coplanar

        Returns
        -------
        ctrl : (n_ctrl, 3) with n_ctrl in {3, 4}
        """
        c0 = pts3d.mean(axis=0)
        centered = pts3d - c0
        _, s, Vt = np.linalg.svd(centered, full_matrices=False)

        # Scale by sqrt(eigenvalue / N) so the control points span the cloud
        scale = s / np.sqrt(len(pts3d))

        s0 = s[0] if s[0] > 0 else 1.0
        planar = (len(s) < 3) or (s[2] / s0 < planar_tol)

        ctrl = [c0, c0 + scale[0] * Vt[0], c0 + scale[1] * Vt[1]]
        if not planar:
            ctrl.append(c0 + scale[2] * Vt[2])
        return np.array(ctrl)

    # ── Step 2: Barycentric coordinates ───────────────────────────────────────
    @staticmethod
    def _barycentric(pts3d: np.ndarray,
                     ctrl : np.ndarray) -> np.ndarray:
        """
        Express each P_i in barycentric coordinates w.r.t. the 4 control points.
        P_i = Σ_j α_ij c_j
        with Σ_j α_ij = 1  (affine combination).

        Works for 3 or 4 control points.  With 3 (the coplanar case) the
        system is 3x2 and lstsq gives the exact in-plane coefficients.
        """
        N      = len(pts3d)
        n_ctrl = len(ctrl)
        # Solve:  [c_1-c_0, ..., c_{m}-c_0] α_vec = P_i - c_0
        #   where α_vec = [α_i1, ...],  α_i0 = 1 - sum(α_vec)
        C    = (ctrl[1:] - ctrl[0]).T          # (3, n_ctrl-1)
        diff = pts3d - ctrl[0]                 # (N, 3)
        alpha_vec, _, _, _ = np.linalg.lstsq(C, diff.T, rcond=None)
        alpha_vec = alpha_vec.T                # (N, n_ctrl-1)
        alpha = np.zeros((N, n_ctrl))
        alpha[:, 1:] = alpha_vec
        alpha[:, 0]  = 1.0 - alpha_vec.sum(axis=1)
        return alpha

    # ── Step 3: Build M matrix ────────────────────────────────────────────────
    @staticmethod
    def _build_M(alpha, pts2d, fx, fy, cx, cy) -> np.ndarray:
        """
        Build the 2N x 3*n_ctrl matrix M from the projection equations.
        For each point i, two rows:
          [α_i0 fx, 0, α_i0(cx-u_i),  α_i1 fx, 0, α_i1(cx-u_i),  ...]
          [0, α_i0 fy, α_i0(cy-v_i),  0, α_i1 fy, α_i1(cy-v_i),  ...]
        """
        N = len(alpha)
        n_ctrl = alpha.shape[1]
        M = np.zeros((2 * N, 3 * n_ctrl))
        for i in range(N):
            u, v = pts2d[i]
            for j in range(n_ctrl):
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
        Recover the control points in the camera frame from the null-space
        basis, for any null-space dimension n and 3 or 4 control points.

        x = Σ_k β_k v_k must satisfy the inter-control-point distance
        constraints  ||c_i^c - c_j^c||² = ||c_i - c_j||².  We form an initial
        β from the linearised system in the products β_kβ_l and then refine β
        directly with Gauss-Newton (Lepetit sec. 3.3), which is what makes the
        n>1 cases accurate rather than merely close.
        """
        n      = V_null.shape[1]
        n_ctrl = ctrl.shape[0]

        if n == 1:
            v = V_null[:, 0]
            betas = np.array([self._dist_beta_n1(v, ctrl)[0]])
        else:
            # Linearised system in the products b_kl = β_k β_l  (k <= l)
            L, rho = self._build_L_rho(V_null, ctrl)
            b, _, _, _ = np.linalg.lstsq(L, rho, rcond=None)
            betas = self._betas_from_products(b, n)

        # Gauss-Newton refinement of β against the exact distance constraints
        betas = self._gauss_newton_betas(betas, V_null, ctrl)

        ctrl_cam = (V_null @ betas).reshape(n_ctrl, 3)

        # Chirality: every reconstructed point must lie in front of the camera.
        # The sign of x is free, so try both and reject if neither works.
        pts_cam = alpha @ ctrl_cam
        if pts_cam[:, 2].min() <= 0:
            ctrl_cam = -ctrl_cam
            pts_cam = alpha @ ctrl_cam
            if pts_cam[:, 2].min() <= 0:
                return None, None

        return self._recover_pose(ctrl_cam, ctrl, alpha)

    @staticmethod
    def _betas_from_products(b, n):
        """
        Extract β from the least-squares estimate of the products
        b_kl = β_k β_l (k <= l), stored in the order produced by _build_L_rho.

        Uses the standard relinearisation: β_1 = sqrt(|b_11|) and
        β_k = b_1k / β_1.  The overall sign is free and is resolved later by
        the chirality test.
        """
        b = np.asarray(b, dtype=float)
        idx = {}
        m = 0
        for k in range(n):
            for l in range(k, n):
                idx[(k, l)] = m
                m += 1

        b11 = b[idx[(0, 0)]]
        beta1 = np.sqrt(abs(b11))
        betas = np.zeros(n)
        if beta1 < 1e-12:
            # Degenerate first coefficient — fall back to the diagonal terms
            for k in range(n):
                betas[k] = np.sqrt(abs(b[idx[(k, k)]]))
            return betas
        betas[0] = beta1
        for k in range(1, n):
            betas[k] = b[idx[(0, k)]] / beta1
        return betas

    @staticmethod
    def _gauss_newton_betas(betas, V_null, ctrl, n_iter: int = 10):
        """
        Refine β so that  f_ij(β) = ||Σ_k β_k (v_k[i] - v_k[j])||² -
        ||c_i - c_j||²  vanishes for every control-point pair.

        With  d_ij(β) = dv_ij @ β,  the residual and Jacobian are
            f_ij      = d·d - ||c_i - c_j||²
            ∂f_ij/∂β_m = 2 d · dv_ij[:, m]

        This is the step the original implementation omitted: the linearised
        product solve is only an initialisation in the reference algorithm.
        """
        betas  = np.asarray(betas, dtype=float).copy()
        n      = V_null.shape[1]
        n_ctrl = ctrl.shape[0]
        V4     = V_null.reshape(n_ctrl, 3, n)
        pairs  = [(i, j) for i in range(n_ctrl) for j in range(i + 1, n_ctrl)]

        dvs = [V4[i] - V4[j] for (i, j) in pairs]                    # (3, n)
        d2w = np.array([np.sum((ctrl[i] - ctrl[j]) ** 2) for (i, j) in pairs])

        for _ in range(n_iter):
            resid = np.empty(len(pairs))
            J     = np.empty((len(pairs), n))
            for r, dv in enumerate(dvs):
                d = dv @ betas                       # (3,)
                resid[r] = d @ d - d2w[r]
                J[r, :]  = 2.0 * (d @ dv)            # (n,)
            try:
                delta, _, _, _ = np.linalg.lstsq(J, -resid, rcond=None)
            except np.linalg.LinAlgError:
                break
            if not np.all(np.isfinite(delta)):
                break
            betas = betas + delta
            if np.linalg.norm(delta) < 1e-12 * (1.0 + np.linalg.norm(betas)):
                break
        return betas

    def _dist_beta_n1(self, v, ctrl):
        """Estimate β (n=1) from the median of the pairwise distance ratios."""
        n_ctrl = ctrl.shape[0]
        betas = []
        v_ctrl = v.reshape(n_ctrl, 3)
        for i in range(n_ctrl):
            for j in range(i + 1, n_ctrl):
                d_world = np.linalg.norm(ctrl[i] - ctrl[j])
                d_cam   = np.linalg.norm(v_ctrl[i] - v_ctrl[j])
                if d_cam > 1e-10:
                    betas.append(d_world / d_cam)
        if not betas:
            return [1.0, -1.0]
        b = float(np.median(betas))
        return [b, -b]

    @staticmethod
    def _build_L_rho(V_null, ctrl):
        """
        Build the constraint matrix L and RHS ρ from the inter-control-point
        distances, in the unknowns b_kl = β_k β_l for k <= l:

            || Σ_k β_k (v_k[i] - v_k[j]) ||² = ||c_i - c_j||²

        L is (n_pairs, n(n+1)/2) with n_pairs = 6 for four control points and
        3 for three.
        """
        n      = V_null.shape[1]
        n_ctrl = ctrl.shape[0]
        pairs  = [(i, j) for i in range(n_ctrl) for j in range(i + 1, n_ctrl)]
        V4     = V_null.reshape(n_ctrl, 3, n)

        L_rows, rho_vals = [], []
        for (i, j) in pairs:
            dv = V4[i] - V4[j]                 # (3, n)
            row = []
            for k in range(n):
                for l in range(k, n):
                    factor = 1.0 if k == l else 2.0
                    row.append(factor * np.dot(dv[:, k], dv[:, l]))
            L_rows.append(row)
            rho_vals.append(np.sum((ctrl[i] - ctrl[j]) ** 2))
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

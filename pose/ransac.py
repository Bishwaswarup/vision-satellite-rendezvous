"""
ransac.py
=========
RANSAC (Random Sample Consensus) wrapper for robust PnP pose estimation.

Algorithm
---------
  Repeat for max_iter iterations:
    1. Draw a minimal random sample (n_min=4 points)
    2. Fit hypothesis (EPnP on 4 points)
    3. Count inliers: points with reprojection error < threshold [px]
    4. Track best hypothesis (most inliers; tie-break: lower mean error)

  After loop:
    5. Refit using ALL inliers of the best hypothesis (increases accuracy)
    6. Optionally iterate inlier refit (LO-RANSAC style, 1 round)

Reference
---------
  Fischler M.A., Bolles R.C., "Random Sample Consensus", CACM 1981.
  Lebeda K. et al., "Fixing the Locally Optimized RANSAC", BMVC 2012.

Usage
-----
  from pose.ransac import RANSACSolver
  solver = RANSACSolver(threshold_px=2.0, max_iter=200, confidence=0.999)
  R, t, inlier_mask, meta = solver.solve(pts3d, pts2d, K)
"""

import numpy as np
from .epnp import EPnPSolver


class RANSACSolver:
    """
    RANSAC-EPnP robust pose estimator.

    Parameters
    ----------
    threshold_px : float   Reprojection error inlier threshold [px]
    max_iter     : int     Maximum RANSAC iterations
    confidence   : float   Desired probability that at least one all-inlier
                           sample is drawn (used for adaptive termination)
    n_min        : int     Minimum sample size (default 4 for EPnP)
    lo_iters     : int     Local optimisation (inlier refit) iterations
    """

    def __init__(self,
                 threshold_px : float = 3.0,
                 max_iter     : int   = 200,
                 confidence   : float = 0.999,
                 n_min        : int   = 6,
                 lo_iters     : int   = 2):
        self.thresh    = threshold_px
        self.max_iter  = max_iter
        self.conf      = confidence
        self.n_min     = n_min
        self.lo_iters  = lo_iters
        self._epnp     = EPnPSolver()

    # ── Public interface ──────────────────────────────────────────────────────
    def solve(self,
              pts3d : np.ndarray,
              pts2d : np.ndarray,
              K     : np.ndarray,
              seed  : int = None) -> tuple:
        """
        Robust PnP via RANSAC + EPnP + inlier refit.

        Parameters
        ----------
        pts3d : (N, 3)  3D world-frame points  [m]
        pts2d : (N, 2)  2D pixel observations  [px]
        K     : (3, 3)  camera intrinsic matrix
        seed  : int or None  RNG seed

        Returns
        -------
        R           : (3,3)   best rotation   (None if failed)
        t           : (3,)    best translation (None if failed)
        inlier_mask : (N,) bool   True for inliers of final estimate
        meta        : dict {
                        'n_inliers'  : int,
                        'n_iters'    : int,
                        'inlier_ratio': float,
                        'mean_repr_err': float,  # over inliers
                        'success'    : bool,
                      }
        """
        pts3d = np.asarray(pts3d, dtype=float)
        pts2d = np.asarray(pts2d, dtype=float)
        K     = np.asarray(K,     dtype=float)
        N     = len(pts3d)
        rng   = np.random.default_rng(seed)

        if N < self.n_min:
            return None, None, np.zeros(N, bool), {
                'n_inliers': 0, 'n_iters': 0,
                'inlier_ratio': 0.0, 'mean_repr_err': np.inf,
                'success': False}

        best_mask  = np.zeros(N, bool)
        best_count = 0
        best_err   = np.inf
        best_R     = None
        best_t     = None

        n_iter_done = 0
        max_iter    = self.max_iter

        for it in range(max_iter):
            n_iter_done += 1

            # 1. Minimal random sample
            idx = rng.choice(N, self.n_min, replace=False)

            # 2. Fit (guard against degenerate minimal samples)
            try:
                R_h, t_h, errs_h, ok = self._epnp.solve(
                    pts3d[idx], pts2d[idx], K)
            except Exception:
                continue
            if not ok or R_h is None:
                continue

            # 3. Score on ALL points
            errs_all = self._epnp._reprojection_errors(pts3d, pts2d, R_h, t_h, K)
            mask     = errs_all < self.thresh
            count    = mask.sum()
            mean_err = errs_all[mask].mean() if count > 0 else np.inf

            # 4. Update best
            if (count > best_count) or (count == best_count and mean_err < best_err):
                best_count = count
                best_err   = mean_err
                best_mask  = mask
                best_R, best_t = R_h, t_h

                # Adaptive iteration count  (Hartley & Zisserman eq. 4.18)
                inlier_ratio = count / N
                if inlier_ratio > 1e-6:
                    denom = np.log(max(1.0 - inlier_ratio**self.n_min, 1e-300))
                    n_needed = int(np.ceil(np.log(1.0 - self.conf) / denom))
                    max_iter = min(max_iter, n_needed)

        # 5. Local optimisation: refit on inliers (LO-RANSAC style)
        if best_count >= self.n_min:
            for _ in range(self.lo_iters):
                inlier_idx = np.where(best_mask)[0]
                try:
                    R_lo, t_lo, _, ok_lo = self._epnp.solve(
                        pts3d[inlier_idx], pts2d[inlier_idx], K)
                except Exception:
                    break
                if not ok_lo or R_lo is None:
                    break
                errs_lo = self._epnp._reprojection_errors(
                    pts3d, pts2d, R_lo, t_lo, K)
                mask_lo = errs_lo < self.thresh
                count_lo = mask_lo.sum()
                if count_lo >= best_count:
                    best_count = count_lo
                    best_mask  = mask_lo
                    best_R, best_t = R_lo, t_lo

        success = best_count >= self.n_min and best_R is not None
        mean_repr = (self._epnp._reprojection_errors(
            pts3d, pts2d, best_R, best_t, K)[best_mask].mean()
            if success else np.inf)

        meta = {
            'n_inliers'    : int(best_count),
            'n_iters'      : n_iter_done,
            'inlier_ratio' : float(best_count / N),
            'mean_repr_err': float(mean_repr),
            'success'      : success,
        }
        return best_R, best_t, best_mask, meta


# ── Convenience function ──────────────────────────────────────────────────────

def solve_pnp_ransac(pts3d       : np.ndarray,
                     pts2d       : np.ndarray,
                     K           : np.ndarray,
                     threshold_px: float = 3.0,
                     max_iter    : int   = 200,
                     seed        : int   = None) -> tuple:
    """
    One-shot RANSAC-EPnP wrapper.

    Returns (R, t, inlier_mask, meta).
    """
    solver = RANSACSolver(threshold_px=threshold_px, max_iter=max_iter)
    return solver.solve(pts3d, pts2d, K, seed=seed)

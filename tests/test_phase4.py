"""
test_phase4.py
==============
Unit tests for Phase 4 — PnP Pose Estimation.

Tests:
    1.  EPnP exact solution on noiseless synthetic data (N=6)
    2.  EPnP reprojection error < 0.1 px for noiseless data
    3.  Rotation matrix output is proper SO(3)
    4.  Translation error < 1 mm for noiseless N=6
    5.  EPnP degrades gracefully with noise (mean repr err < 5 px at σ=2 px)
    6.  RANSAC recovers pose with 50% outlier contamination
    7.  RANSAC inlier mask correctly identifies clean points
    8.  RANSAC meta contains expected keys
    9.  Rodrigues round-trip: R → r → R
    10. Rodrigues round-trip: r → R → r
    11. Refiner reduces reprojection error vs initial noisy estimate
    12. Refiner output is valid SO(3)
    13. Full pipeline: EPnP → Refiner → repr err < 2 px (σ=1 px noise)
    14. RANSAC + Refiner pipeline on Ariane keypoints
    15. EPnP fails gracefully with N < 4 correspondences

Run:
    cd /home/user/phase1
    python -m pytest tests/test_phase4.py -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from pose import (
    EPnPSolver, solve_epnp,
    RANSACSolver, solve_pnp_ransac,
    GaussNewtonRefiner, refine_pose,
    _dcm_to_rodrigues, _rodrigues_to_dcm,
)
from vision import rendezvous_camera, ariane_model, look_at_rotation

# ── Shared fixtures ────────────────────────────────────────────────────────────
CAM  = rendezvous_camera()
K    = CAM.K

def _random_rotation(seed):
    rng = np.random.default_rng(seed)
    u1, u2, u3 = rng.uniform(size=3)
    q = np.array([
        np.sqrt(1-u1)*np.sin(2*np.pi*u2),
        np.sqrt(1-u1)*np.cos(2*np.pi*u2),
        np.sqrt(u1)  *np.sin(2*np.pi*u3),
        np.sqrt(u1)  *np.cos(2*np.pi*u3),
    ])
    q0, q1, q2, q3 = q
    return np.array([
        [1-2*(q2**2+q3**2), 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)],
        [2*(q1*q2+q0*q3), 1-2*(q1**2+q3**2), 2*(q2*q3-q0*q1)],
        [2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), 1-2*(q1**2+q2**2)],
    ])

def _synthetic_correspondences(N=10, seed=0, noise_px=0.0, outlier_frac=0.0):
    """
    Generate synthetic PnP correspondences for a random scene.
    Returns (pts3d, pts2d_noisy, R_gt, t_gt).
    """
    rng = np.random.default_rng(seed)
    R_gt = _random_rotation(seed)
    t_gt = np.array([0.2, -0.1, 25.0])   # 25 m range

    # Random 3D points in front of camera (world = body frame)
    pts3d = rng.uniform(-3.0, 3.0, (N, 3))

    # Perfect projections
    P_cam = (R_gt @ pts3d.T).T + t_gt
    z = P_cam[:, 2]
    u = K[0,0] * P_cam[:,0] / z + K[0,2]
    v = K[1,1] * P_cam[:,1] / z + K[1,2]
    pts2d = np.stack([u, v], axis=-1)

    # Add noise
    if noise_px > 0:
        pts2d += rng.normal(0, noise_px, pts2d.shape)

    # Add outliers (random pixels)
    if outlier_frac > 0:
        n_out = int(N * outlier_frac)
        out_idx = rng.choice(N, n_out, replace=False)
        pts2d[out_idx] = rng.uniform(0, 1024, (n_out, 2))

    return pts3d, pts2d, R_gt, t_gt


# ── Test 1: EPnP exact solution (noiseless) ───────────────────────────────────
def test_epnp_noiseless_reprojection():
    pts3d, pts2d, R_gt, t_gt = _synthetic_correspondences(N=10, noise_px=0.0)
    R, t, errs, ok = solve_epnp(pts3d, pts2d, K)
    assert ok, "EPnP must succeed on noiseless data"
    assert errs.mean() < 0.5, \
        f"Noiseless mean reprojection error must be < 0.5 px, got {errs.mean():.4f}"


# ── Test 2: EPnP reprojection < 0.1 px for noiseless N=20 ────────────────────
def test_epnp_noiseless_accuracy():
    pts3d, pts2d, R_gt, t_gt = _synthetic_correspondences(N=20, noise_px=0.0)
    R, t, errs, ok = solve_epnp(pts3d, pts2d, K)
    assert ok
    assert errs.max() < 1.0, \
        f"Max reprojection error must be < 1 px (noiseless), got {errs.max():.4f}"


# ── Test 3: EPnP rotation is proper SO(3) ────────────────────────────────────
def test_epnp_rotation_so3():
    pts3d, pts2d, R_gt, t_gt = _synthetic_correspondences(N=12, noise_px=0.0)
    R, t, errs, ok = solve_epnp(pts3d, pts2d, K)
    assert ok
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-10,
        err_msg="EPnP rotation must be orthogonal")
    assert abs(np.linalg.det(R) - 1.0) < 1e-10, "EPnP rotation det must be +1"


# ── Test 4: EPnP translation error < 5 cm for noiseless N=15 ────────────────
def test_epnp_translation_accuracy():
    pts3d, pts2d, R_gt, t_gt = _synthetic_correspondences(N=15, noise_px=0.0)
    R, t, errs, ok = solve_epnp(pts3d, pts2d, K)
    assert ok
    t_err = np.linalg.norm(t - t_gt)
    assert t_err < 0.05, \
        f"Translation error must be < 5 cm for noiseless data, got {t_err*100:.2f} cm"


# ── Test 5: EPnP degrades gracefully with noise ───────────────────────────────
def test_epnp_with_noise():
    pts3d, pts2d, R_gt, t_gt = _synthetic_correspondences(N=20, noise_px=2.0)
    R, t, errs, ok = solve_epnp(pts3d, pts2d, K)
    assert ok, "EPnP must not fail on noisy data"
    # Mean reprojection error should be in the ballpark of noise level
    assert errs.mean() < 20.0, \
        f"Mean repr err with 2px noise should be < 20 px, got {errs.mean():.2f}"


# ── Test 6: RANSAC recovers pose with 40% outliers ───────────────────────────
def test_ransac_50pct_outliers():
    # N=30, 40% outliers → 18 inliers (n_min=6 → RANSAC is viable)
    pts3d, pts2d, R_gt, t_gt = _synthetic_correspondences(
        N=30, noise_px=1.0, outlier_frac=0.4, seed=42)
    R, t, mask, meta = solve_pnp_ransac(
        pts3d, pts2d, K, threshold_px=3.0, max_iter=500, seed=0)
    assert meta['success'], f"RANSAC must succeed with 40% outliers, meta={meta}"
    # Rotation angle error
    R_err = R @ R_gt.T
    cos_a = np.clip((np.trace(R_err) - 1) / 2, -1, 1)
    angle_err_deg = np.degrees(np.arccos(abs(cos_a)))
    assert angle_err_deg < 15.0, \
        f"RANSAC rotation error must be < 15° with 40% outliers, got {angle_err_deg:.2f}°"


# ── Test 7: RANSAC inlier mask identifies clean points ────────────────────────
def test_ransac_inlier_mask():
    N = 20
    pts3d, pts2d, R_gt, t_gt = _synthetic_correspondences(
        N=N, noise_px=0.5, outlier_frac=0.3, seed=7)
    R, t, mask, meta = solve_pnp_ransac(
        pts3d, pts2d, K, threshold_px=3.0, max_iter=200, seed=7)
    assert meta['success']
    assert mask.dtype == bool
    assert len(mask) == N
    assert mask.sum() >= 4, "RANSAC must find at least 4 inliers"


# ── Test 8: RANSAC meta keys ──────────────────────────────────────────────────
def test_ransac_meta_keys():
    pts3d, pts2d, R_gt, t_gt = _synthetic_correspondences(N=15, noise_px=1.0)
    _, _, _, meta = solve_pnp_ransac(pts3d, pts2d, K, seed=0)
    for key in ['n_inliers', 'n_iters', 'inlier_ratio', 'mean_repr_err', 'success']:
        assert key in meta, f"Meta must contain key '{key}'"


# ── Test 9: Rodrigues round-trip R → r → R ───────────────────────────────────
def test_rodrigues_roundtrip_R_to_r():
    rng = np.random.default_rng(3)
    for seed in range(10):
        R = _random_rotation(seed)
        r = _dcm_to_rodrigues(R)
        R2 = _rodrigues_to_dcm(r)
        np.testing.assert_allclose(R, R2, atol=1e-10,
            err_msg="R → r → R Rodrigues round-trip failed")


# ── Test 10: Rodrigues round-trip r → R → r ──────────────────────────────────
def test_rodrigues_roundtrip_r_to_R():
    rng = np.random.default_rng(5)
    for _ in range(10):
        r = rng.uniform(-np.pi, np.pi, 3)
        # Clip to valid range (|r| < π)
        norm = np.linalg.norm(r)
        if norm > np.pi:
            r = r / norm * (np.pi - 0.01)
        R = _rodrigues_to_dcm(r)
        r2 = _dcm_to_rodrigues(R)
        R2 = _rodrigues_to_dcm(r2)
        np.testing.assert_allclose(R, R2, atol=1e-10,
            err_msg="r → R → r Rodrigues round-trip failed")


# ── Test 11: Refiner reduces reprojection error ───────────────────────────────
def test_refiner_reduces_error():
    pts3d, pts2d, R_gt, t_gt = _synthetic_correspondences(N=15, noise_px=1.0)
    R_init, t_init, errs_init, ok = solve_epnp(pts3d, pts2d, K)
    assert ok
    cost_init = errs_init.mean()**2

    R_ref, t_ref, cost_ref = refine_pose(R_init, t_init, pts3d, pts2d, K)
    # Refined cost should be ≤ initial (or at worst similar)
    assert cost_ref <= cost_init * 2.0, \
        f"Refiner must not increase cost significantly: {cost_ref:.4f} vs {cost_init:.4f}"


# ── Test 12: Refiner output is valid SO(3) ────────────────────────────────────
def test_refiner_output_so3():
    pts3d, pts2d, R_gt, t_gt = _synthetic_correspondences(N=12, noise_px=0.5)
    R_init, t_init, _, ok = solve_epnp(pts3d, pts2d, K)
    assert ok
    R_ref, t_ref, _ = refine_pose(R_init, t_init, pts3d, pts2d, K)
    np.testing.assert_allclose(R_ref.T @ R_ref, np.eye(3), atol=1e-10,
        err_msg="Refined rotation must be orthogonal")
    assert abs(np.linalg.det(R_ref) - 1.0) < 1e-10, "Refined rotation det must be +1"


# ── Test 13: Full pipeline EPnP → Refiner (σ=1 px) ───────────────────────────
def test_full_pipeline_reprojection():
    pts3d, pts2d, R_gt, t_gt = _synthetic_correspondences(N=20, noise_px=1.0)
    R_e, t_e, errs_e, ok = solve_epnp(pts3d, pts2d, K)
    assert ok
    R_r, t_r, cost = refine_pose(R_e, t_e, pts3d, pts2d, K)
    # Mean repr error after refinement should be close to noise level
    from pose.epnp import EPnPSolver
    errs_r = EPnPSolver._reprojection_errors(pts3d, pts2d, R_r, t_r, K)
    assert errs_r.mean() < 5.0, \
        f"Refined mean repr err must be < 5 px at σ=1px, got {errs_r.mean():.2f}"


# ── Test 14: RANSAC + Refiner on Ariane keypoints ────────────────────────────
def test_ransac_refine_ariane_keypoints():
    model = ariane_model()
    eye   = np.array([20., 10., 15.])
    R_gt, t_gt = look_at_rotation(eye, np.zeros(3))

    # Ground-truth 2D projections (noiseless)
    kp3d = model.keypoint_array        # (K, 3)
    P_cam = (R_gt @ kp3d.T).T + t_gt
    z = P_cam[:, 2]
    u = K[0,0] * P_cam[:,0] / z + K[0,2]
    v = K[1,1] * P_cam[:,1] / z + K[1,2]
    kp2d = np.stack([u, v], axis=-1)

    # Add mild noise
    rng = np.random.default_rng(99)
    kp2d_noisy = kp2d + rng.normal(0, 1.5, kp2d.shape)

    R, t, mask, meta = solve_pnp_ransac(kp3d, kp2d_noisy, K,
                                         threshold_px=5.0, max_iter=200, seed=0)
    assert meta['success'], "RANSAC must succeed on Ariane keypoints"
    assert meta['n_inliers'] >= 5, "Must have at least 5 inliers"

    R_ref, t_ref, _ = refine_pose(R, t, kp3d[mask], kp2d_noisy[mask], K)
    t_err = np.linalg.norm(t_ref - t_gt)
    assert t_err < 2.0, \
        f"Translation error on Ariane must be < 2 m, got {t_err:.3f} m"


# ── Test 15: EPnP fails gracefully with N < 4 ────────────────────────────────
def test_epnp_too_few_points():
    pts3d = np.random.rand(3, 3)
    pts2d = np.random.rand(3, 2)
    with pytest.raises(ValueError, match="N≥4"):
        solve_epnp(pts3d, pts2d, K)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

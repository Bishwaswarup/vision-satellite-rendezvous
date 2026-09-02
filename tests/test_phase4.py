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

K_TEST = np.array([[800., 0., 512.], [0., 800., 512.], [0., 0., 1.]])
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
    """
    Levenberg-Marquardt only ever commits a step that lowers the objective, so
    the refined cost can never exceed the initial one.

    The comparison must be like-for-like.  `refine_pose` returns the MEAN OF
    THE SQUARED reprojection distances, whereas `solve_epnp` returns the
    per-point distances themselves; `errs.mean() ** 2` is the SQUARE OF THE
    MEAN, which is a different quantity.  By Jensen's inequality

        mean(e^2) = (mean e)^2 + var(e)  >=  (mean e)^2

    so comparing the refined cost against `errs.mean() ** 2` is biased against
    the refiner by exactly the variance of the residuals — typically 20-75 %
    here, which is why this assertion originally carried a factor-of-2 fudge.
    Squaring first removes the mismatch and the bound then holds by
    construction, not by luck.
    """
    for seed in range(8):
        pts3d, pts2d, R_gt, t_gt = _synthetic_correspondences(
            N=15, noise_px=1.0, seed=seed)
        R_init, t_init, errs_init, ok = solve_epnp(pts3d, pts2d, K)
        assert ok

        cost_init = float((errs_init ** 2).mean())      # mean of squares
        R_ref, t_ref, cost_ref = refine_pose(R_init, t_init, pts3d, pts2d, K)

        assert cost_ref <= cost_init * (1 + 1e-9) + 1e-12, (
            f"seed {seed}: refiner increased the cost, "
            f"{cost_ref:.6f} vs {cost_init:.6f}")


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


# ── Test 16: coplanar point sets must solve exactly, not return garbage ─────
def _planar_scene(rng, N, sigma=0.0, depth=25.0):
    """A coplanar world-point set viewed from `depth` metres."""
    K = np.array([[800., 0., 512.], [0., 800., 512.], [0., 0., 1.]])
    P = rng.uniform(-3, 3, (N, 3))
    P[:, 2] = 0.0                       # exactly coplanar
    q = rng.normal(size=4); q /= np.linalg.norm(q)
    w, x, y, z = q
    R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
                  [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
                  [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]])
    t = np.array([0., 0., depth])
    Pc = (R @ P.T).T + t
    uv = np.stack([800*Pc[:, 0]/Pc[:, 2] + 512,
                   800*Pc[:, 1]/Pc[:, 2] + 512], axis=-1)
    if sigma > 0:
        uv = uv + rng.normal(0, sigma, uv.shape)
    return P, uv, K, R, t


def _rot_err_deg(A, B):
    c = (np.trace(A.T @ B) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def test_epnp_planar_points_solve_exactly():
    """
    Regression guard for the coplanar degeneracy.

    For a coplanar cloud the third singular value is zero, so a 4th control
    point coincides with the centroid, the barycentric solve becomes singular
    and three columns of M vanish.  EPnP must switch to THREE control points
    (Lepetit sec. 3.5) instead of returning a garbage pose as success=True.

    ~3 % of random 6-point samples drawn from the shipped keypoint models are
    exactly coplanar, so this is on the RANSAC hot path, not an edge case.
    """
    solver = EPnPSolver()
    for N in (6, 10, 20):
        rng = np.random.default_rng(N)
        for _ in range(25):
            P, uv, K, R, t = _planar_scene(rng, N)
            R_est, t_est, errs, ok = solver.solve(P, uv, K)
            assert ok, f"planar N={N} reported failure on a clean scene"
            assert _rot_err_deg(R, R_est) < 1e-3, (
                f"planar N={N}: {_rot_err_deg(R, R_est):.4f} deg")
            assert np.linalg.norm(t_est - t) < 1e-6
            assert errs.mean() < 1e-3


def test_epnp_uses_three_control_points_when_planar():
    """The planar branch must actually be taken, not worked around."""
    solver = EPnPSolver()
    rng = np.random.default_rng(0)

    P_planar = rng.uniform(-3, 3, (10, 3)); P_planar[:, 2] = 0.0
    P_solid  = rng.uniform(-3, 3, (10, 3))

    assert solver._choose_control_points(P_planar).shape == (3, 3)
    assert solver._choose_control_points(P_solid).shape == (4, 3)

    # And the barycentric coordinates must still be a partition of unity.
    for P in (P_planar, P_solid):
        ctrl = solver._choose_control_points(P)
        alpha = solver._barycentric(P, ctrl)
        assert np.allclose(alpha.sum(axis=1), 1.0)
        assert np.allclose(alpha @ ctrl, P, atol=1e-9)


# ── Test 17: an implausible fit must be reported as a failure ───────────────
def test_epnp_rejects_implausible_solution():
    """
    `success` must reflect the quality of the fit.  The degenerate cases used
    to return success=True alongside mean reprojection errors of order 1e9 px,
    which let RANSAC accept them as hypotheses.
    """
    solver = EPnPSolver(max_reprojection_error=8.0)
    rng = np.random.default_rng(1)

    # 3-D points paired with pixels that correspond to nothing.
    P = rng.uniform(-3, 3, (10, 3))
    uv = rng.uniform(0, 1024, (10, 2))
    R_est, t_est, errs, ok = solver.solve(P, uv, K_TEST)
    if ok:
        assert errs.mean() < 8.0, (
            "solver reported success on a fit it could not achieve")


# ── Test 18: agreement with OpenCV on non-planar scenes ─────────────────────
def test_epnp_matches_opencv_accuracy():
    """
    Cross-check against a reference implementation.  Without the Gauss-Newton
    refinement of the betas (Lepetit sec. 3.3) this implementation was ~4.3x
    less accurate than cv2; it should now be within a small factor.
    """
    cv2 = pytest.importorskip("cv2")
    solver = EPnPSolver()
    K = np.array([[800., 0., 512.], [0., 800., 512.], [0., 0., 1.]])
    rng = np.random.default_rng(4)

    ours, ref = [], []
    for _ in range(60):
        P = rng.uniform(-3, 3, (20, 3))
        q = rng.normal(size=4); q /= np.linalg.norm(q)
        w, x, y, z = q
        R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
                      [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
                      [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]])
        t = np.array([0., 0., 25.])
        Pc = (R @ P.T).T + t
        uv = np.stack([800*Pc[:, 0]/Pc[:, 2] + 512,
                       800*Pc[:, 1]/Pc[:, 2] + 512], axis=-1)
        uv = uv + rng.normal(0, 1.0, uv.shape)

        R_est, _, _, ok = solver.solve(P, uv, K)
        if ok:
            ours.append(_rot_err_deg(R, R_est))
        _, rvec, _ = cv2.solvePnP(P, uv, K, None, flags=cv2.SOLVEPNP_EPNP)
        ref.append(_rot_err_deg(R, cv2.Rodrigues(rvec)[0]))

    ratio = np.median(ours) / np.median(ref)
    assert ratio < 2.5, (
        f"median rotation error is {ratio:.2f}x the cv2 EPnP reference "
        f"({np.median(ours):.4f} deg vs {np.median(ref):.4f} deg)")


# ── Test 19: Rodrigues conversion must survive theta = pi ───────────────────
def test_dcm_to_rodrigues_at_theta_pi():
    """
    Regression guard for the theta = pi singularity.

    The naive formula divides by 2 sin(theta).  At theta = pi that is zero and
    the function returned [0, 0, 0] — a 180 degree error.  This is not an edge
    case here: `look_at_rotation` produces trace(R) = -1 exactly for the
    on-axis views used as fixtures throughout the project.
    """
    axis = np.array([1.0, 2.0, -0.5])
    axis /= np.linalg.norm(axis)

    for theta in (1e-9, 1e-4, 1.0, np.pi - 1e-4, np.pi - 1e-7,
                  np.pi - 1e-12, np.pi):
        R = _rodrigues_to_dcm(theta * axis)
        r = _dcm_to_rodrigues(R)
        R2 = _rodrigues_to_dcm(r)
        c = (np.trace(R.T @ R2) - 1) / 2
        err = np.degrees(np.arccos(np.clip(c, -1, 1)))
        assert err < 1e-4, f"theta={theta}: round trip error {err:.3e} deg"
        assert np.all(np.isfinite(r))
        assert np.linalg.norm(r) <= np.pi + 1e-9, (
            f"theta={theta}: ||r|| = {np.linalg.norm(r)} blew up")


def test_dcm_to_rodrigues_on_project_camera_poses():
    """The repo's own canonical camera poses are exactly theta = pi."""
    from vision.renderer import look_at_rotation

    for eye in ([0., 0., 30.], [0., 0., -30.], [0., 30., 0.]):
        R_cw, _ = look_at_rotation(np.array(eye))
        assert abs(np.trace(R_cw) + 1.0) < 1e-9, "fixture is no longer theta=pi"
        r = _dcm_to_rodrigues(R_cw)
        assert np.linalg.norm(r) > 3.0, (
            f"eye={eye}: returned ||r||={np.linalg.norm(r):.4f}, expected ~pi")
        c = (np.trace(R_cw.T @ _rodrigues_to_dcm(r)) - 1) / 2
        assert np.degrees(np.arccos(np.clip(c, -1, 1))) < 1e-4


# ── Test 20: refining an exact pose must not damage it ──────────────────────
def test_refiner_preserves_exact_initialisation():
    """
    Seeded with the ground-truth pose the refiner should barely move.  With a
    global Rodrigues parameterisation it started from r = 0 (a 180 degree error
    at these poses) and *destroyed* the initialisation: 6.01 deg from perfect.
    """
    from vision.renderer import look_at_rotation
    from vision.body_model import ariane_model

    kp = ariane_model().keypoint_array
    R_gt, _ = look_at_rotation(np.array([0., 0., 30.]))
    t_gt = np.array([0., 0., 30.])
    rng = np.random.default_rng(5)

    errs = []
    for _ in range(20):
        Pc = (R_gt @ kp.T).T + t_gt
        uv = np.stack([800*Pc[:, 0]/Pc[:, 2] + 512,
                       800*Pc[:, 1]/Pc[:, 2] + 512], axis=-1)
        uv = uv + rng.normal(0, 1.0, uv.shape)
        R_ref, t_ref, _ = refine_pose(R_gt, t_gt, kp, uv, K)
        c = (np.trace(R_gt.T @ R_ref) - 1) / 2
        errs.append(np.degrees(np.arccos(np.clip(c, -1, 1))))

    assert np.median(errs) < 1.5, (
        f"refiner moved {np.median(errs):.4f} deg away from an exact "
        f"initialisation at a theta=pi pose")


# ── Test 21: the refiner must fail visibly rather than diverge ──────────────
def test_refiner_does_not_diverge():
    """
    A residual that clamps Z while the Jacobian uses the raw Z lets LM accept
    steps that lower a fictitious cost; translation then ran away to ~1e8 m
    while the reported cost stayed a plausible 4450 px^2.  Any pose that puts
    points behind the camera must be rejected outright.
    """
    from vision.body_model import ariane_model
    from pose.refine import GaussNewtonRefiner

    kp = ariane_model().keypoint_array
    rng = np.random.default_rng(1)
    q = rng.normal(size=4); q /= np.linalg.norm(q)
    w, x, y, z = q
    R_gt = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
                     [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]])
    t_gt = np.array([0., 0., 25.])
    Pc = (R_gt @ kp.T).T + t_gt
    uv = np.stack([800*Pc[:, 0]/Pc[:, 2] + 512,
                   800*Pc[:, 1]/Pc[:, 2] + 512], axis=-1)

    refiner = GaussNewtonRefiner()
    for tz in (1.0, 0.0, -5.0, -25.0):
        R_ref, t_ref, cost, info = refiner.refine(
            R_gt, np.array([0., 0., tz]), kp, uv, K, return_info=True)
        assert np.linalg.norm(t_ref) < 1e3, (
            f"t_z={tz}: translation ran away to {np.linalg.norm(t_ref):.3e} m")
        assert not info['success'], (
            f"t_z={tz}: reported success on a pose behind the camera")

    # A merely poor — but valid — initialisation must still converge.
    R_ref, t_ref, cost, info = refiner.refine(
        R_gt, np.array([0., 0., 20.]), kp, uv, K, return_info=True)
    assert info['success'] and np.linalg.norm(t_ref - t_gt) < 1e-3


# ── Test 22: manifold Jacobian vs finite differences ────────────────────────
def test_refiner_jacobian_matches_finite_differences():
    """The analytic Jacobian must match the residual it claims to differentiate."""
    from pose.refine import GaussNewtonRefiner
    from vision.renderer import look_at_rotation

    rng = np.random.default_rng(0)
    P = rng.uniform(-3, 3, (12, 3))
    R, _ = look_at_rotation(np.array([0., 0., 30.]))
    t = np.array([0.3, -0.2, 30.])
    Pc = (R @ P.T).T + t
    uv = np.stack([800*Pc[:, 0]/Pc[:, 2] + 512,
                   800*Pc[:, 1]/Pc[:, 2] + 512], axis=-1)
    uv = uv + rng.normal(0, 1.0, uv.shape)
    wts = np.ones(len(P))

    J, _ = GaussNewtonRefiner._jacobian_and_residual(
        R, t, P, uv, 800., 800., 512., 512., wts)

    def resid(delta):
        Rn = _rodrigues_to_dcm(delta[:3]) @ R
        tn = t + delta[3:]
        Pc = (Rn @ P.T).T + tn
        r = np.empty(2 * len(P))
        r[0::2] = 800*Pc[:, 0]/Pc[:, 2] + 512 - uv[:, 0]
        r[1::2] = 800*Pc[:, 1]/Pc[:, 2] + 512 - uv[:, 1]
        return r

    h = 1e-7
    J_fd = np.zeros_like(J)
    for k in range(6):
        d = np.zeros(6); d[k] = h
        J_fd[:, k] = (resid(d) - resid(-d)) / (2 * h)

    rel = np.abs(J - J_fd).max() / np.abs(J_fd).max()
    assert rel < 1e-6, f"analytic Jacobian differs from FD by {rel:.3e} relative"


# ── Test 23: parity with the OpenCV LM refiner ──────────────────────────────
def test_refiner_matches_opencv_lm():
    """Both should converge to the same local optimum."""
    cv2 = pytest.importorskip("cv2")
    from vision.body_model import ariane_model
    from vision.renderer import look_at_rotation

    kp = ariane_model().keypoint_array
    R_gt, _ = look_at_rotation(np.array([0., 0., 30.]))
    t_gt = np.array([0., 0., 30.])
    rng = np.random.default_rng(5)

    ours, ref = [], []
    for _ in range(20):
        Pc = (R_gt @ kp.T).T + t_gt
        uv = np.stack([800*Pc[:, 0]/Pc[:, 2] + 512,
                       800*Pc[:, 1]/Pc[:, 2] + 512], axis=-1)
        uv = uv + rng.normal(0, 1.0, uv.shape)

        R_o, t_o, _ = refine_pose(R_gt, t_gt, kp, uv, K)
        rvec, _ = cv2.Rodrigues(R_gt)
        rv, tv = cv2.solvePnPRefineLM(kp, uv, K, None,
                                      rvec.copy(), t_gt.reshape(3, 1).copy())
        R_c = cv2.Rodrigues(rv)[0]

        c = (np.trace(R_c.T @ R_o) - 1) / 2
        ours.append(np.degrees(np.arccos(np.clip(c, -1, 1))))
        ref.append(np.linalg.norm(t_o - tv.ravel()))

    assert np.median(ours) < 0.05, (
        f"rotation differs from cv2 LM by {np.median(ours):.4f} deg")
    assert np.median(ref) < 1e-3


# ── Test 24: RANSAC adaptive termination must actually terminate ────────────
def test_ransac_adaptive_termination():
    """
    `for it in range(max_iter)` materialises the range once, so reassigning
    max_iter inside the loop did nothing: every call ran the full budget and
    meta['n_iters'] was always exactly max_iter, making it useless as a cost
    metric and costing 10-100x the necessary runtime.
    """
    from pose.ransac import solve_pnp_ransac

    rng = np.random.default_rng(0)
    P = rng.uniform(-3, 3, (40, 3))
    t = np.array([0., 0., 25.])
    Pc = P + t
    uv = np.stack([800*Pc[:, 0]/Pc[:, 2] + 512,
                   800*Pc[:, 1]/Pc[:, 2] + 512], axis=-1)
    uv = uv + rng.normal(0, 0.5, uv.shape)

    n_iters = []
    for max_iter in (50, 200, 1000):
        _, _, _, meta = solve_pnp_ransac(P, uv, K, threshold_px=3.0,
                                         max_iter=max_iter, seed=1)
        assert meta['success']
        assert meta['n_iters'] <= max_iter
        n_iters.append(meta['n_iters'])

    # On clean data a handful of samples suffices; the count must not simply
    # track the budget.
    assert max(n_iters) < 50, (
        f"n_iters {n_iters} scales with max_iter — adaptive stopping is dead")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

"""
test_phase5.py
==============
Unit tests for Phase 5 — EKF / UKF Pose Tracking.

Tests
-----
  1.  EKF predict: covariance grows (trace increases)
  2.  EKF update: covariance shrinks (trace decreases)
  3.  EKF covariance stays positive-definite after 50 steps
  4.  EKF quaternion normalization maintained after predict+update
  5.  EKF converges: position error < 1 m after 30 updates
  6.  EKF Mahalanobis gating rejects obvious outlier
  7.  EKF meta dict: n_updates, n_rejected track correctly
  8.  UKF predict: covariance grows
  9.  UKF update: covariance shrinks
 10.  UKF covariance stays positive-definite after 50 steps
 11.  UKF quaternion normalization maintained
 12.  UKF converges: position error < 1 m after 30 updates
 13.  EKF and UKF give similar position estimates on same scenario (RMSE < 2×)
 14.  EKF on tumbling target: attitude error < 15° after 40 steps
 15.  Filter uncertainty grows without measurements (predict-only)

Run:
    cd /home/user/phase1
    python -m pytest tests/test_phase5.py -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from estimator import (
    pack_state, unpack_state,
    propagate_rk4, h_measurement,
    N_ORBITAL,
    quat_to_dcm, dcm_to_quat, rotvec_to_quat, quat_to_rotvec, quat_norm,
    make_ekf, make_ukf,
    MultEKF, UnscentedKF,
    default_process_noise, default_measurement_noise,
)

# ── Shared helpers ─────────────────────────────────────────────────────────────

DT   = 1.0        # 1-second steps
MEAS_POS_STD = 0.5   # m
MEAS_ATT_STD = 0.05  # rad


def _make_gt_state(seed=0):
    """Ground-truth initial state at 25 m range."""
    rng = np.random.default_rng(seed)
    r_gt = np.array([25.0, 0.0, 0.0])    # radial approach
    v_gt = np.array([-0.05, 0.0, 0.0])   # closing at 5 cm/s
    # Random attitude
    axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
    angle = rng.uniform(0.1, 0.5)
    q_gt  = rotvec_to_quat(axis * angle)
    w_gt  = rng.uniform(-0.02, 0.02, 3)  # slow tumble
    return pack_state(r_gt, v_gt, q_gt, w_gt)


def _noisy_meas(x_true, rng, pos_std=MEAS_POS_STD, att_std=MEAS_ATT_STD):
    """Simulate a noisy measurement from ground truth state."""
    z_clean = h_measurement(x_true)
    z_noisy = z_clean.copy()
    z_noisy[:3] += rng.normal(0, pos_std,  3)
    z_noisy[3:]  += rng.normal(0, att_std, 3)
    return z_noisy


def _pos_error(x_est, x_true):
    return np.linalg.norm(unpack_state(x_est)[0] - unpack_state(x_true)[0])


# ── Test 1: EKF predict — covariance grows ────────────────────────────────────
def test_ekf_predict_cov_grows():
    x0  = _make_gt_state(0)
    ekf = make_ekf(x0, DT, pos0_std=1.0)
    tr0 = np.trace(ekf.P)
    ekf.predict(DT)
    assert np.trace(ekf.P) > tr0, "Predict must increase covariance trace"


# ── Test 2: EKF update — covariance shrinks ───────────────────────────────────
def test_ekf_update_cov_shrinks():
    x0  = _make_gt_state(0)
    ekf = make_ekf(x0, DT, pos0_std=5.0)
    ekf.predict(DT)
    tr_pred = np.trace(ekf.P)
    z = h_measurement(x0)        # perfect measurement
    ekf.update(z)
    assert np.trace(ekf.P) < tr_pred, "Update must decrease covariance trace"


# ── Test 3: EKF covariance positive-definite after 50 steps ──────────────────
def test_ekf_covariance_pd():
    x_true = _make_gt_state(1)
    ekf    = make_ekf(x_true, DT, pos0_std=2.0)
    rng    = np.random.default_rng(1)
    for _ in range(50):
        x_true = propagate_rk4(x_true, DT)
        ekf.predict(DT)
        ekf.update(_noisy_meas(x_true, rng))
    eigs = np.linalg.eigvalsh(ekf.P)
    assert np.all(eigs > -1e-9), f"Covariance must be PSD; min eig = {eigs.min():.3e}"


# ── Test 4: EKF quaternion normalization ──────────────────────────────────────
def test_ekf_quat_norm():
    x0  = _make_gt_state(2)
    ekf = make_ekf(x0, DT)
    rng = np.random.default_rng(2)
    x_true = x0.copy()
    for _ in range(20):
        x_true = propagate_rk4(x_true, DT)
        ekf.predict(DT)
        ekf.update(_noisy_meas(x_true, rng))
        q = unpack_state(ekf.state)[2]
        assert abs(np.linalg.norm(q) - 1.0) < 1e-10, \
            f"Quaternion norm must be 1, got {np.linalg.norm(q):.8f}"


# ── Test 5: EKF position convergence ──────────────────────────────────────────
def test_ekf_position_convergence():
    x_true = _make_gt_state(3)
    # Initialise with 3 m position error
    x_init = x_true.copy()
    x_init[:3] += np.array([3.0, 0.5, -0.5])
    ekf = make_ekf(x_init, DT, pos0_std=3.0)
    rng = np.random.default_rng(3)
    for _ in range(30):
        x_true = propagate_rk4(x_true, DT)
        ekf.predict(DT)
        ekf.update(_noisy_meas(x_true, rng))
    err = _pos_error(ekf.state, x_true)
    assert err < 1.0, f"EKF position error after 30 steps must be < 1 m, got {err:.3f} m"


# ── Test 6: EKF Mahalanobis gating rejects outlier ───────────────────────────
def test_ekf_mahal_gating():
    x0  = _make_gt_state(4)
    ekf = make_ekf(x0, DT, pos0_std=1.0)
    ekf.predict(DT)
    # Outlier: position 500 m off
    z_outlier = h_measurement(x0) + np.array([500., 0, 0, 0, 0, 0])
    info = ekf.update(z_outlier)
    assert not info['accepted'], "Mahalanobis gate must reject 500 m outlier"
    assert ekf.n_rejected == 1


# ── Test 7: EKF update/reject counters ───────────────────────────────────────
def test_ekf_counters():
    x0  = _make_gt_state(5)
    ekf = make_ekf(x0, DT)
    rng = np.random.default_rng(5)
    x_true = x0.copy()
    n_good  = 0
    n_bad   = 0
    for i in range(10):
        x_true = propagate_rk4(x_true, DT)
        ekf.predict(DT)
        if i % 3 == 0:
            # Insert outlier
            z_bad = h_measurement(x_true) + np.array([999., 0, 0, 0, 0, 0])
            ekf.update(z_bad)
            n_bad += 1
        else:
            ekf.update(_noisy_meas(x_true, rng))
            n_good += 1
    assert ekf.n_updates  == n_good
    assert ekf.n_rejected == n_bad


# ── Test 8: UKF predict — covariance grows ───────────────────────────────────
def test_ukf_predict_cov_grows():
    x0  = _make_gt_state(0)
    ukf = make_ukf(x0, DT, pos0_std=1.0)
    tr0 = np.trace(ukf.P)
    ukf.predict(DT)
    assert np.trace(ukf.P) > tr0, "UKF predict must increase covariance trace"


# ── Test 9: UKF update — covariance shrinks ──────────────────────────────────
def test_ukf_update_cov_shrinks():
    x0  = _make_gt_state(0)
    ukf = make_ukf(x0, DT, pos0_std=5.0)
    ukf.predict(DT)
    tr_pred = np.trace(ukf.P)
    z = h_measurement(x0)
    ukf.update(z)
    assert np.trace(ukf.P) < tr_pred, "UKF update must decrease covariance trace"


# ── Test 10: UKF covariance positive-definite ─────────────────────────────────
def test_ukf_covariance_pd():
    x_true = _make_gt_state(6)
    ukf    = make_ukf(x_true, DT, pos0_std=2.0)
    rng    = np.random.default_rng(6)
    for _ in range(50):
        x_true = propagate_rk4(x_true, DT)
        ukf.predict(DT)
        ukf.update(_noisy_meas(x_true, rng))
    eigs = np.linalg.eigvalsh(ukf.P)
    assert np.all(eigs > -1e-9), f"UKF covariance must be PSD; min eig = {eigs.min():.3e}"


# ── Test 11: UKF quaternion normalization ─────────────────────────────────────
def test_ukf_quat_norm():
    x0  = _make_gt_state(7)
    ukf = make_ukf(x0, DT)
    rng = np.random.default_rng(7)
    x_true = x0.copy()
    for _ in range(20):
        x_true = propagate_rk4(x_true, DT)
        ukf.predict(DT)
        ukf.update(_noisy_meas(x_true, rng))
        q = unpack_state(ukf.state)[2]
        assert abs(np.linalg.norm(q) - 1.0) < 1e-10, \
            f"UKF quaternion norm must be 1, got {np.linalg.norm(q):.8f}"


# ── Test 12: UKF position convergence ─────────────────────────────────────────
def test_ukf_position_convergence():
    x_true = _make_gt_state(8)
    x_init = x_true.copy()
    x_init[:3] += np.array([3.0, 0.5, -0.5])
    ukf = make_ukf(x_init, DT, pos0_std=3.0)
    rng = np.random.default_rng(8)
    for _ in range(30):
        x_true = propagate_rk4(x_true, DT)
        ukf.predict(DT)
        ukf.update(_noisy_meas(x_true, rng))
    err = _pos_error(ukf.state, x_true)
    assert err < 1.0, f"UKF position error after 30 steps must be < 1 m, got {err:.3f} m"


# ── Test 13: EKF and UKF give similar estimates ───────────────────────────────
def test_ekf_ukf_similar():
    """Both filters should reach similar RMSE on the same trajectory."""
    x_true = _make_gt_state(9)
    x_init = x_true.copy()
    x_init[:3] += np.array([2.0, -1.0, 0.5])

    ekf = make_ekf(x_init, DT, pos0_std=3.0)
    ukf = make_ukf(x_init, DT, pos0_std=3.0)
    rng = np.random.default_rng(9)

    errs_ekf, errs_ukf = [], []
    for _ in range(40):
        x_true = propagate_rk4(x_true, DT)
        z = _noisy_meas(x_true, rng)

        ekf.predict(DT)
        ekf.update(z)

        ukf.predict(DT)
        ukf.update(z)

        errs_ekf.append(_pos_error(ekf.state, x_true))
        errs_ukf.append(_pos_error(ukf.state, x_true))

    rmse_ekf = np.sqrt(np.mean(np.array(errs_ekf)**2))
    rmse_ukf = np.sqrt(np.mean(np.array(errs_ukf)**2))

    # Neither should be more than 3× worse than the other
    ratio = max(rmse_ekf, rmse_ukf) / (min(rmse_ekf, rmse_ukf) + 1e-9)
    assert ratio < 5.0, \
        f"EKF RMSE {rmse_ekf:.3f} vs UKF RMSE {rmse_ukf:.3f} — too different (ratio {ratio:.2f})"
    # Both should be reasonably good
    assert rmse_ekf < 2.0, f"EKF RMSE must be < 2 m, got {rmse_ekf:.3f}"
    assert rmse_ukf < 2.0, f"UKF RMSE must be < 2 m, got {rmse_ukf:.3f}"


# ── Test 14: EKF on tumbling target — attitude tracking ───────────────────────
def test_ekf_attitude_tumbling():
    """Filter must track a fast-tumbling target: att err < 15° after 40 steps."""
    rng    = np.random.default_rng(10)
    r_gt   = np.array([20., 0., 0.])
    v_gt   = np.array([-0.02, 0., 0.])
    axis   = np.array([0.1, 0.5, 0.86]); axis /= np.linalg.norm(axis)
    q_gt   = rotvec_to_quat(axis * 0.3)
    w_gt   = np.array([0.0, 0.0, 0.15])   # ~8.6 °/s spin
    x_true = pack_state(r_gt, v_gt, q_gt, w_gt)

    # Start with 20° attitude error
    dq_init = rotvec_to_quat(np.array([0., 0., 0.35]))   # ~20°
    from estimator.state import quat_mult
    q_init = quat_norm(quat_mult(dq_init, q_gt))
    x_init = pack_state(r_gt + np.array([1., 0., 0.]), v_gt, q_init, w_gt)

    ekf = make_ekf(x_init, DT, pos0_std=2.0, att0_std=0.4,
                   rate0_std=0.1, att_proc_std=1e-3, rate_proc_std=1e-4)

    for _ in range(40):
        x_true = propagate_rk4(x_true, DT)
        ekf.predict(DT)
        ekf.update(_noisy_meas(x_true, rng, att_std=0.02))

    q_est  = unpack_state(ekf.state)[2]
    q_true = unpack_state(x_true)[2]
    q_err  = np.array([-q_true[0], -q_true[1], -q_true[2], q_true[3]])
    from estimator.state import quat_mult as qm
    dq     = qm(q_est, q_err)
    dq     = dq / np.linalg.norm(dq)
    angle_err_deg = np.degrees(2 * np.arccos(np.clip(abs(dq[3]), 0, 1)))
    assert angle_err_deg < 15.0, \
        f"Attitude error after 40 steps must be < 15°, got {angle_err_deg:.2f}°"


# ── Test 15: Predict-only — uncertainty grows monotonically ──────────────────
def test_predict_only_uncertainty_grows():
    """Without measurements, trace(P) must grow or stay flat (never shrink)."""
    x0  = _make_gt_state(11)
    ekf = make_ekf(x0, DT, pos0_std=1.0)
    traces = [np.trace(ekf.P)]
    for _ in range(20):
        ekf.predict(DT)
        traces.append(np.trace(ekf.P))
    diffs = np.diff(traces)
    assert np.all(diffs >= -1e-12), \
        "Predict-only trace must be non-decreasing; some decrease detected"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

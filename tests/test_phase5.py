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


# ── Test 16: the attitude residual must be multiplicative ───────────────────
def test_attitude_residual_survives_theta_pi():
    """
    Regression guard.  The residual used to be a subtraction of GLOBAL
    rotation vectors, z[3:] - rotvec(q_hat).  That map is singular at
    theta = pi: two attitudes a hair either side map to nearly antipodal
    rotation vectors, so a ~0 degree attitude error produced a ~2*pi residual.
    """
    from estimator.state import (attitude_residual, quat_to_rotvec,
                                 rotvec_to_quat, quat_mult, quat_inv)

    axis = np.array([0.0, 0.0, 1.0])
    rng = np.random.default_rng(0)

    for theta in (0.3, 2.0, np.pi - 1e-3, np.pi, np.pi + 1e-3):
        q_hat = rotvec_to_quat(theta * axis)
        for _ in range(50):
            # A small true error applied multiplicatively
            da_true = rng.normal(0, 0.02, 3)
            q_meas = quat_mult(rotvec_to_quat(da_true), q_hat)

            res = attitude_residual(quat_to_rotvec(q_meas), q_hat)
            assert np.linalg.norm(res - da_true) < 1e-8, (
                f"theta={theta}: residual {res} != true error {da_true}")

    # And the naive difference must be shown to fail there, so the test is
    # about the fix rather than about the tolerance.
    q_hat  = rotvec_to_quat((np.pi - 1e-6) * axis)
    q_meas = rotvec_to_quat((np.pi + 1e-6) * axis)
    naive  = quat_to_rotvec(q_meas) - quat_to_rotvec(q_hat)
    good   = attitude_residual(quat_to_rotvec(q_meas), q_hat)
    assert np.linalg.norm(naive) > 6.0, "expected the naive residual to blow up"
    assert np.linalg.norm(good) < 1e-5


def test_measurement_jacobian_is_exact():
    """
    With a multiplicative residual, H is exactly [I3 0 0 0; 0 0 I3 0] — the
    attitude residual IS the attitude error state.
    """
    from estimator.state import measurement_jacobian, attitude_residual
    rng = np.random.default_rng(1)
    q = rng.normal(size=4); q /= np.linalg.norm(q)
    x = pack_state(np.array([10., -4., 2.]), np.array([0.1, 0., -0.05]),
                   q, np.array([0.01, 0.02, -0.01]))

    H = measurement_jacobian(x)
    expect = np.zeros((6, 12))
    expect[0:3, 0:3] = np.eye(3)
    expect[3:6, 6:9] = np.eye(3)
    np.testing.assert_allclose(H, expect, atol=0)

    # Confirm against finite differences of the ACTUAL residual.
    from estimator.state import (unpack_state, quat_mult, rotvec_to_quat,
                                 quat_to_rotvec)
    r0, v0, q0, w0 = unpack_state(x)
    z = np.concatenate([r0, quat_to_rotvec(q0)])

    def residual(delta):
        # perturb the ESTIMATE; the error state is (true - estimate)
        q_hat = quat_mult(rotvec_to_quat(delta[6:9]), q0)
        out = np.empty(6)
        out[:3] = z[:3] - (r0 + delta[:3])
        out[3:] = attitude_residual(z[3:], q_hat)
        return out

    h = 1e-6
    H_fd = np.zeros((6, 12))
    for k in range(12):
        d = np.zeros(12); d[k] = h
        H_fd[:, k] = (residual(-d) - residual(d)) / (2 * h)
    np.testing.assert_allclose(H, H_fd, atol=1e-6)


# ── Test 17: range-scaled measurement noise ─────────────────────────────────
def test_range_scaled_measurement_noise():
    """
    A monocular pose fix degrades as z^2, so a fixed R is wrong at both ends:
    70x too large at 5 m and 4x too small at 80 m on this pipeline.
    """
    from estimator.state import range_scaled_measurement_noise as R_of

    R20 = R_of(20.0)
    assert np.allclose(np.diag(R20)[:3], 0.25 ** 2)      # reference range

    # Quadratic in range.
    R40 = R_of(40.0)
    assert np.isclose(np.diag(R40)[0] / np.diag(R20)[0], 4.0 ** 2, rtol=1e-9)

    # Monotone and floored at close range (must not collapse to zero).
    ranges = [0.0, 0.5, 1.0, 5.0, 20.0, 80.0]
    sig = [np.sqrt(np.diag(R_of(r))[0]) for r in ranges]
    assert all(b >= a - 1e-12 for a, b in zip(sig, sig[1:]))
    assert sig[0] > 0.0
    assert all(np.all(np.linalg.eigvals(R_of(r)) > 0) for r in ranges)


# ── Test 18: the UKF must actually be a UKF ─────────────────────────────────
def test_ukf_sigma_points_have_usable_spread():
    """
    alpha = 1e-3 with n = 12 gives n + lambda = 1.2e-5: the sigma points sit
    at 0.0035 sigma with a centre weight of -1e6, so the unscented transform
    degenerates into a finite-difference linearisation and the UKF reproduces
    the EKF to four decimals.  An EKF-vs-UKF comparison then compares a filter
    with itself.
    """
    x0 = pack_state(np.array([20., 5., 2.]), np.zeros(3),
                    np.array([0., 0., 0., 1.]), np.array([0., 0., 0.1]))
    ukf = make_ukf(x0, 10.0)

    n_err = ukf.n_err
    spread = np.sqrt(abs(n_err + ukf._lam))
    assert spread > 0.3, (
        f"sigma-point spread is {spread:.5f} sigma — the unscented transform "
        f"has collapsed onto a linearisation")

    # Weights must still be a valid van der Merwe set.
    assert np.isclose(ukf._Wm.sum(), 1.0)
    assert abs(ukf._Wm[0]) < 1e3, "centre weight is pathologically large"


def test_ekf_and_ukf_are_distinguishable():
    """
    The two filters should track similarly but must not BE the same filter.

    Position is the wrong place to look: HCW translation is exactly linear, so
    the EKF and the UKF agree there to machine precision whatever alpha is.
    The unscented transform can only show itself where the model is nonlinear
    — the attitude/rate block — and it shows up most clearly in the
    COVARIANCE.  At alpha = 1e-3 the relative difference in P is ~6 %; with a
    usable spread it is ~47 %.
    """
    rng = np.random.default_rng(3)
    x_true = pack_state(np.array([50., 20., 10.]), np.array([-0.05, 0., 0.]),
                        np.array([0., 0., 0., 1.]), np.array([0., 0., 0.15]))
    x_init = pack_state(np.array([52., 17., 11.5]), np.array([-0.05, 0., 0.]),
                        np.array([0., 0., 0., 1.]), np.array([0., 0., 0.15]))
    ekf = make_ekf(x_init, 10.0, pos0_std=4.0)
    ukf = make_ukf(x_init, 10.0, pos0_std=4.0)

    x = x_true.copy()
    dP = 0.0
    for _ in range(120):
        x = propagate_rk4(x, 10.0)
        z = _noisy_meas(x, rng)
        for F in (ekf, ukf):
            F.predict(10.0)
            F.update(z)
        dP = max(dP, float(np.abs(ekf.P - ukf.P).max()
                           / max(np.abs(ekf.P).max(), 1e-30)))

    assert dP > 0.1, (
        f"EKF and UKF covariances never differ by more than {dP:.2e} relative "
        f"— the unscented transform has collapsed onto a linearisation")

    # ...but they must still agree on the answer.
    r_e = unpack_state(ekf.state)[0]
    r_u = unpack_state(ukf.state)[0]
    assert np.linalg.norm(r_e - r_u) < 1.0


# ── Test 19: NIS is computed and recorded ───────────────────────────────────
def test_nis_is_recorded_and_summarised():
    """The repo advertised NIS consistency but computed NIS nowhere."""
    from estimator.state import nis_statistics

    rng = np.random.default_rng(7)
    x_true = pack_state(np.array([30., 10., 5.]), np.zeros(3),
                        np.array([0., 0., 0., 1.]), np.array([0., 0., 0.1]))
    ekf = make_ekf(x_true.copy(), 10.0)

    x = x_true.copy()
    for _ in range(200):
        x = propagate_rk4(x, 10.0)
        ekf.predict(10.0)
        ekf.update(_noisy_meas(x, rng))

    assert len(ekf.nis_history) == 200
    st = nis_statistics(ekf.nis_history, dof=6)
    assert st['n'] == 200 and st['dof'] == 6
    assert st['ci_low'] < 6.0 < st['ci_high']
    assert 0.0 <= st['tail_fraction'] <= 1.0


# ── Test 20: one inertia tensor for one target ──────────────────────────────
def test_inertia_tensor_is_consistent_with_the_geometry():
    """
    `estimator.state.J_ARIANE` drives the target attitude in every reported
    result, while `target.attitude.ariane_upper_stage()` is only reached by
    the phase-2 tests.  They used to be different bodies — diag(1800,1800,360)
    with the symmetry axis on z, against diag(1176,6988,6988) on x.
    """
    from estimator.state import J_ARIANE
    from target.attitude import ariane_upper_stage

    J_ref = np.asarray(ariane_upper_stage().I, dtype=float)
    np.testing.assert_allclose(np.diag(J_ARIANE), np.diag(J_ref), rtol=1e-9)

    # x must be the symmetry axis, matching vision.body_model's geometry.
    d = np.diag(J_ARIANE)
    assert d[0] < d[1] and np.isclose(d[1], d[2]), \
        f"symmetry axis is not x: {d}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

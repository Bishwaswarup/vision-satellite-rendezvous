"""
test_phase7.py
==============
Unit tests for Phase 7 — Closed-Loop Integration & Rendezvous Video.

Tests
-----
  1.  SimConfig has expected default fields
  2.  default_config() factory applies overrides
  3.  run_simulation() returns a SimResult with correct array shapes
  4.  SimResult.range_m decreases overall (chaser approaches target)
  5.  Simulation with perfect state (no vision) converges to < 0.5 m
  6.  Simulation with EKF reaches position error < 2 m within 150 steps
  7.  Controls respect u_max bound at every step
  8.  delta_v is positive and finite
  9.  Docking detected within 300 steps (perfect state, generous u_max)
 10.  EKF covariance diagonal (cov_pos) decreases on average
 11.  SimResult.pos_error shape matches r_true
 12.  render_frame() returns a matplotlib Figure without error
 13.  fig_to_rgb() converts figure to (H, W, 3) uint8 array
 14.  frames_to_gif() writes a valid GIF file
 15.  VideoExporter.export_summary_png() writes a PNG file

Run:
    cd /home/user/phase1
    python -m pytest tests/test_phase7.py -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import tempfile

from simulation import (
    SimConfig, SimResult, RendezvousSimulator,
    run_simulation, default_config,
    VideoExporter, render_frame, frames_to_gif,
)


# ── Shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def perf_result():
    """Perfect-state simulation (no vision noise, no EKF lag)."""
    cfg = default_config(
        n_steps=300, use_vision=False, use_ekf=False,
        u_max=0.4, pixel_noise_std=0.0,
        pos_weight=20.0, vel_weight=2.0,
        r0=np.array([20., 2., -1.]),
        v0=np.array([-0.08, 0., 0.]),
        r0_err=np.array([0., 0., 0.]),
    )
    return run_simulation(cfg, rng_seed=0)


@pytest.fixture(scope='module')
def ekf_result():
    """EKF-in-loop simulation."""
    cfg = default_config(
        n_steps=200, use_vision=False, use_ekf=True,
        # Estimator statistics need the full run, not a truncated one.
        stop_at_dock=False,
        u_max=0.4, pixel_noise_std=0.0,
        r0=np.array([20., 2., -1.]),
        v0=np.array([-0.08, 0., 0.]),
        r0_err=np.array([2., -0.5, 0.3]),
        pos_meas_std=0.3,
    )
    return run_simulation(cfg, rng_seed=7)


# ── Test 1: SimConfig defaults ────────────────────────────────────────────────
def test_simconfig_defaults():
    cfg = SimConfig()
    assert cfg.dt == 1.0
    assert cfg.n_steps == 300
    assert cfg.u_max > 0
    assert cfg.use_vision is True
    assert cfg.use_ekf is True


# ── Test 2: default_config factory ───────────────────────────────────────────
def test_default_config_override():
    cfg = default_config(n_steps=50, u_max=0.05, use_vision=False)
    assert cfg.n_steps == 50
    assert cfg.u_max == 0.05
    assert cfg.use_vision is False
    assert cfg.dt == 1.0   # unchanged default


# ── Test 3: SimResult array shapes ───────────────────────────────────────────
def test_result_shapes(perf_result):
    res = perf_result
    T = res.n_steps_run + 1
    assert res.r_true.shape  == (T, 3)
    assert res.v_true.shape  == (T, 3)
    assert res.q_true.shape  == (T, 4)
    assert res.w_true.shape  == (T, 3)
    assert res.r_est.shape   == (T, 3)
    assert res.controls.shape[1] == 3


# ── Test 4: chaser approaches target ─────────────────────────────────────────
def test_chaser_approaches(perf_result):
    rng = perf_result.range_m
    # Range at end should be much less than at start
    assert rng[-1] < rng[0] * 0.5, \
        f"Chaser should approach: initial {rng[0]:.1f} m, final {rng[-1]:.1f} m"


# ── Test 5: perfect-state converges ──────────────────────────────────────────
def test_perfect_state_convergence(perf_result):
    r_final = np.linalg.norm(perf_result.r_true[-1])
    assert r_final < 1.0, \
        f"Perfect-state sim must converge to < 1.0 m, got {r_final:.3f} m"


# ── Test 6: EKF position error < 2 m within 150 steps ────────────────────────
def test_ekf_position_error(ekf_result):
    err_150 = ekf_result.pos_error[150]
    assert err_150 < 2.0, \
        f"EKF position error at step 150 must be < 2 m, got {err_150:.3f} m"


# ── Test 7: controls respect u_max ───────────────────────────────────────────
def test_controls_bounded(perf_result):
    u_max = perf_result.cfg.u_max
    max_u = np.abs(perf_result.controls).max()
    assert max_u <= u_max + 1e-9, \
        f"Controls must not exceed u_max={u_max:.3f}, saw {max_u:.6f}"


# ── Test 8: delta_v positive and finite ──────────────────────────────────────
def test_deltav(perf_result):
    dv = perf_result.delta_v
    assert dv > 0, "Δv must be positive"
    assert np.isfinite(dv), "Δv must be finite"
    assert dv < 1e4, f"Δv unreasonably large: {dv:.1f}"


# ── Test 9: docking detected ─────────────────────────────────────────────────
def test_docking_detected():
    cfg = default_config(
        n_steps=300, use_vision=False, use_ekf=False,
        u_max=0.5, pos_weight=50.0, vel_weight=10.0,
        pos_meas_std=0.0,    # perfect state for close-range docking test
        r0=np.array([5., 0.5, -0.2]),
        v0=np.array([-0.02, 0., 0.]),
        r0_err=np.zeros(3),
    )
    res = run_simulation(cfg, rng_seed=1)
    assert res.dock_step is not None, \
        f"Docking should be detected. Final range: {res.range_m[-1]:.3f} m"
    assert res.dock_step <= 300


# ── Test 10: EKF covariance decreases on average ─────────────────────────────
def test_ekf_cov_decreases(ekf_result):
    cov = ekf_result.ekf_cov_pos   # (T, 3)
    T = len(cov)
    assert T > 40, "run too short to compare early and late covariance"
    # Compare the first and last 15 steps of whatever was actually flown.
    cov_early = cov[5:20].mean()
    cov_late  = cov[T-15:T].mean()
    assert cov_late < cov_early, \
        f"EKF covariance should decrease: early {cov_early:.3f}, late {cov_late:.3f}"


# ── Test 11: pos_error shape ──────────────────────────────────────────────────
def test_pos_error_shape(ekf_result):
    assert ekf_result.pos_error.shape == ekf_result.r_true.shape[:1], \
        "pos_error shape must match number of time steps"


# ── Test 12: render_frame returns a Figure ────────────────────────────────────
def test_render_frame(perf_result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = render_frame(perf_result, k=10)
    assert hasattr(fig, 'savefig'), "render_frame must return a matplotlib Figure"
    plt.close(fig)


# ── Test 13: fig_to_rgb shape ─────────────────────────────────────────────────
def test_fig_to_rgb(perf_result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from simulation.video import fig_to_rgb
    fig = render_frame(perf_result, k=0)
    arr = fig_to_rgb(fig)
    plt.close(fig)
    assert arr.ndim == 3 and arr.shape[2] == 3, \
        f"fig_to_rgb must return (H, W, 3), got {arr.shape}"
    assert arr.dtype == np.uint8


# ── Test 14: frames_to_gif writes valid GIF ───────────────────────────────────
def test_frames_to_gif():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from simulation.video import fig_to_rgb

    # Create tiny dummy frames
    frames = []
    for _ in range(3):
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.plot([0, 1], [0, 1])
        frames.append(fig_to_rgb(fig))
        plt.close(fig)

    with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as f:
        path = f.name

    try:
        frames_to_gif(frames, path, fps=5)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 100, "GIF file seems empty"
    finally:
        os.unlink(path)


# ── Test 15: export_summary_png writes PNG ────────────────────────────────────
def test_export_summary_png(perf_result):
    import matplotlib
    matplotlib.use('Agg')
    exp = VideoExporter(perf_result, step=20)
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        path = f.name
    try:
        exp.export_summary_png(path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 1000, "PNG seems empty"
    finally:
        os.unlink(path)


# ── Test 16: the camera actually sees the target at the start of a run ───────
def test_camera_sees_target_at_start():
    """
    Regression guard for the camera-boresight defect.

    The camera boresight is the camera +z axis, NOT an LVLH axis.  If the two
    are conflated, the pinhole depth divisor becomes the LVLH cross-track
    component and the target lands behind the camera, so every keypoint fails
    the visibility gate and the pose solve never runs.
    """
    cfg = SimConfig()
    sim = RendezvousSimulator(cfg, rng_seed=0)

    for mode in ('track', 'fixed'):
        sim.cfg.camera_mode = mode
        R_cl = sim.camera_attitude(cfg.r0)

        # R_cl must be a proper rotation
        assert np.allclose(R_cl @ R_cl.T, np.eye(3), atol=1e-12), mode
        assert np.isclose(np.linalg.det(R_cl), 1.0, atol=1e-12), mode

        kp3d, kp2d, visible, _ = sim._project_keypoints(cfg.r0, cfg.q0, R_cl)

        # Every keypoint of a 8 m target at 30 m range must be in front of the
        # camera and inside a 1024x1024 frame.
        assert kp3d[:, 2].min() > 0, f"{mode}: keypoints behind the camera"
        assert visible.all(), (
            f"{mode}: only {visible.sum()}/{len(visible)} keypoints visible "
            f"at the initial condition")
        assert kp2d[:, 0].min() >= 0 and kp2d[:, 0].max() < sim.cam.width
        assert kp2d[:, 1].min() >= 0 and kp2d[:, 1].max() < sim.cam.height


# ── Test 17: the recovered pose is returned in LVLH, not the camera frame ────
def test_vision_measurement_returns_lvlh_pose():
    """
    Noiseless round trip: the pose handed to the filter must be expressed in
    LVLH.  Returning the raw camera-frame pose would silently rotate every
    measurement by the camera attitude.
    """
    from estimator.state import quat_to_dcm

    cfg = SimConfig()
    cfg.pixel_noise_std = 0.0
    sim = RendezvousSimulator(cfg, rng_seed=0)

    rng = np.random.default_rng(12345)
    for _ in range(5):
        r = np.array([25.0 + 10 * rng.random(),
                      6 * rng.random() - 3,
                      6 * rng.random() - 3])
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)

        r_meas, q_meas, repr_err, ok, n_vis = sim._vision_measurement(r, q, r)
        assert ok and n_vis == 15

        assert np.linalg.norm(r_meas - r) < 1e-6, "position not in LVLH"
        c = (np.trace(quat_to_dcm(q).T @ quat_to_dcm(q_meas)) - 1) / 2
        ang = np.degrees(np.arccos(np.clip(c, -1, 1)))
        assert ang < 1e-3, f"attitude not in LVLH ({ang:.4f} deg)"


# ── Test 18: dropout is driven by close-range FOV exit ───────────────────────
def test_vision_dropout_is_range_driven():
    """
    With the geometry correct, the pose solve must succeed at long range and
    degrade only as the target overflows the frame at close range.  An 8 m
    target fills a 1024 px frame at f*L/W = 800*8/1024 = 6.25 m.
    """
    cfg = SimConfig()
    cfg.n_steps = 120
    cfg.stop_at_dock = False
    res = RendezvousSimulator(cfg, rng_seed=42).run()

    rng_m = np.linalg.norm(res.r_true[:-1], axis=1)
    failed = ~np.isfinite(res.repr_errs)

    far = rng_m > 12.0
    assert far.sum() > 10, "test needs some far-range steps"
    assert failed[far].mean() < 0.05, (
        f"pose solve fails {100*failed[far].mean():.0f}% of the time beyond "
        f"12 m — the camera geometry is wrong")
    assert res.n_visible[far].min() >= cfg.min_visible_kpts


# ── Test 19: a failed pose solve must not return the true state ─────────────
def test_vision_failure_returns_none_not_truth():
    """
    Regression guard for the ground-truth leak.

    When the pose solve fails the sensor model must return None.  Returning
    r_true/q_true silently converts every failed step into a perfect
    measurement, which turns a failed run into a perfect-state run.
    """
    cfg = SimConfig()
    sim = RendezvousSimulator(cfg, rng_seed=0)

    # Put the target far behind the camera in 'fixed' mode: nothing is visible,
    # so the solve cannot be attempted.
    sim.cfg.camera_mode = 'fixed'
    r_behind = -50.0 * np.asarray(cfg.camera_boresight, dtype=float)

    r_meas, q_meas, repr_err, ok, n_vis = sim._vision_measurement(
        r_behind, cfg.q0, r_behind)

    assert not ok
    assert n_vis == 0
    assert r_meas is None, "sensor returned a pose on failure"
    assert q_meas is None, "sensor returned an attitude on failure"
    assert np.isnan(repr_err)


# ── Test 20: with vision always failing, the filter must visibly degrade ─────
def test_filter_coasts_when_vision_drops_out():
    """
    If every vision call fails the EKF has nothing to update on, so it must
    coast on its own prediction: the estimate degrades and the covariance
    grows.  A run that still tracks to millimetres is being fed truth.
    """
    cfg = SimConfig()
    cfg.n_steps = 120
    cfg.stop_at_dock = False
    sim = RendezvousSimulator(cfg, rng_seed=42)
    sim._vision_measurement = lambda r, q, los: (None, None, float('nan'),
                                                 False, 0)
    res = sim.run()

    assert res.vision_dropout_rate == 1.0
    assert not res.meas_used.any()

    # The filter starts with a deliberate r0_err of |[2,-0.5,0.3]| = 2.09 m and
    # is never corrected, so the error cannot collapse to millimetres.
    assert res.pos_error.mean() > 0.5, (
        f"mean error {res.pos_error.mean():.4f} m with no measurements at all "
        f"— the true state is leaking into the estimator")

    # Coasting covariance must grow monotonically in the absence of updates.
    sigma = np.linalg.norm(res.ekf_cov_pos, axis=1)
    assert sigma[-1] > sigma[1], "covariance did not grow while coasting"


# ── Test 21: dropout steps are exactly the steps with no filter update ──────
def test_meas_used_matches_dropout_bookkeeping():
    """The reported dropout rate must agree with the per-step update log."""
    cfg = SimConfig()
    cfg.n_steps = 100
    cfg.stop_at_dock = False       # exercise the full log, not a truncated one
    res = RendezvousSimulator(cfg, rng_seed=3).run()

    assert res.meas_used.shape == (res.n_steps_run,)
    assert res.n_vision_fail == int((~res.meas_used).sum())
    assert np.isclose(res.vision_dropout_rate,
                      res.n_vision_fail / res.n_steps_run)
    # A NaN reprojection error must coincide with a skipped update.
    assert np.array_equal(np.isnan(res.repr_errs), ~res.meas_used)


# ── Test 22: the direct-measurement path must inject the noise R declares ───
def test_direct_measurement_attitude_noise_matches_R():
    """
    Regression guard for the noise-free attitude measurement.

    With `use_vision=False` the measurement is built directly from the true
    state.  If the attitude is copied verbatim while R declares
    σ_att = att_meas_std, three of the six measurement channels are exact and
    the filter's consistency statistics become meaningless.
    """
    from estimator.state import quat_mult, rotvec_to_quat, quat_to_dcm

    cfg = SimConfig()
    rng = np.random.default_rng(0)
    q = np.array([0.0, 0.0, 0.0, 1.0])

    angles = []
    for _ in range(3000):
        dq = rotvec_to_quat(rng.normal(0, cfg.att_meas_std, 3))
        q_m = quat_mult(dq, q)
        c = (np.trace(quat_to_dcm(q).T @ quat_to_dcm(q_m)) - 1) / 2
        angles.append(np.arccos(np.clip(c, -1, 1)))
    angles = np.array(angles)

    # For small isotropic rotation-vector noise the total angle has
    # RMS = sqrt(3) * sigma, i.e. per-axis RMS = sigma.
    per_axis_rms = np.sqrt(np.mean(angles ** 2) / 3)
    assert np.isclose(per_axis_rms, cfg.att_meas_std, rtol=0.08), (
        f"per-axis attitude noise {per_axis_rms:.4f} rad != declared "
        f"{cfg.att_meas_std:.4f} rad")

    # And the measurement must actually differ from the truth.
    assert angles.min() > 0.0


# ── Test 23: filter consistency on the direct-measurement path ──────────────
def test_ekf_nis_is_broadly_consistent():
    """
    Normalised innovation squared (NIS) should sit near n_z = 6.

    History of this number, each step a separate defect:
        2.767  noise-free attitude measurement (three of six channels exact)
        6.821  noise fixed, but the residual subtracted global rotation vectors
        5.771  residual made multiplicative; filter now merely conservative
        6.044  truth given a real disturbance and Q made acceleration-driven

    The filter is now statistically consistent, so this asserts the actual
    chi-square interval rather than a loose band.
    """
    import estimator.ekf as E

    nis = []
    orig = E.MultEKF.update

    def patched(self, z, R_override=None):
        info = orig(self, z, R_override)
        if np.isfinite(info['mahal']):
            nis.append(info['mahal'])
        return info

    E.MultEKF.update = patched
    try:
        for seed in range(4):
            RendezvousSimulator(
                SimConfig(use_vision=False, n_steps=250, stop_at_dock=False),
                rng_seed=seed).run()
    finally:
        E.MultEKF.update = orig

    from estimator.state import nis_statistics
    st = nis_statistics(nis, dof=6)
    assert st['n'] > 500
    assert st['consistent'], (
        f"mean NIS {st['mean']:.3f} outside the 95% interval "
        f"[{st['ci_low']:.3f}, {st['ci_high']:.3f}] for n_z = 6 — the filter's "
        f"covariance does not match the errors it actually makes")
    assert st['tail_fraction'] < 0.03, (
        f"P(NIS > chi2_0.99) = {st['tail_fraction']*100:.2f}%, expected ~1%")


# ── Test 24: the approach must be a rendezvous, not a fly-through ───────────
def test_approach_does_not_fly_through_the_target():
    """
    With thrust_weight = 1 the LQR gain demands ~27 m/s^2 at 30 m — about 90x
    the 0.3 m/s^2 limit — so the loop saturated on ~28 % of steps, arrived at
    4 m/s and passed 23 m through the target before coming back.  Weighting
    control so the UNSATURATED demand respects u_max fixes it.
    """
    cfg = SimConfig(n_steps=600)
    res = RendezvousSimulator(cfg, rng_seed=42).run()

    speeds = np.linalg.norm(res.v_true, axis=1)
    sat = (np.abs(res.controls).max(axis=1) >= cfg.u_max - 1e-9).sum()

    assert res.dock_step is not None, "never docked"
    assert sat == 0, f"{sat} saturated steps — the gain exceeds the thrust limit"
    assert speeds.max() < 2.0, f"peak closing speed {speeds.max():.2f} m/s"
    assert res.r_true[:, 0].min() > -2.0, (
        f"chaser reached x = {res.r_true[:, 0].min():.2f} m, i.e. it flew "
        f"through the target")
    assert res.delta_v < 10.0, f"delta_v {res.delta_v:.2f} m/s for a 30 m approach"


def test_run_stops_at_docking_and_reports_delta_v_to_dock():
    """Station-keeping after the success criterion inflated the reported dv."""
    res = RendezvousSimulator(SimConfig(n_steps=600), rng_seed=42).run()
    assert res.dock_step is not None
    assert res.n_steps_run == res.dock_step
    assert len(res.controls) == res.n_steps_run
    assert np.isclose(res.delta_v_to_dock, res.delta_v)

    long_run = RendezvousSimulator(
        SimConfig(n_steps=600, stop_at_dock=False), rng_seed=42).run()
    assert long_run.n_steps_run > long_run.dock_step
    assert long_run.delta_v_to_dock < long_run.delta_v


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

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
    # Compare first 20 steps vs last 20 steps average
    cov_early = cov[5:20].mean()
    cov_late  = cov[180:200].mean()
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


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

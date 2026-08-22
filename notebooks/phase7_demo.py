"""
phase7_demo.py
==============
Phase 7 demonstration — Closed-Loop Integration + Rendezvous Video.

Figures
-------
  Fig 30 : Full pipeline trajectory (true vs EKF, 3-D LVLH)
  Fig 31 : Range + estimation error vs time
  Fig 32 : Thrust profiles + cumulative Δv
  Fig 33 : EPnP reprojection error over the approach
  Fig 34 : EKF covariance convergence

Animated GIF
------------
  outputs/phase7_rendezvous.gif — wireframe camera feed + LVLH trace

Run:
    cd /home/user/phase1
    python notebooks/phase7_demo.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

from simulation import (
    SimConfig, default_config, run_simulation,
    VideoExporter, render_frame,
)

OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTDIR, exist_ok=True)

plt.style.use('dark_background')
BLUE   = '#4FC3F7'
AMBER  = '#FFB74D'
GREEN  = '#81C784'
RED    = '#EF9A9A'
PURPLE = '#CE93D8'
GREY   = '#78909C'

# ══════════════════════════════════════════════════════════════════════════════
# Run the full closed-loop simulation
# ══════════════════════════════════════════════════════════════════════════════
print("Running Phase 7 closed-loop simulation (EKF + LQR + vision)...")

cfg_full = default_config(
    n_steps     = 250,
    use_vision  = True,
    use_ekf     = True,
    u_max       = 0.3,
    pos_weight  = 15.0,
    vel_weight  = 1.5,
    pixel_noise_std = 1.5,
    pos_meas_std    = 0.5,
    r0  = np.array([30., 3., -1.5]),
    v0  = np.array([-0.08, 0., 0.]),
    w0  = np.array([0.02, 0.05, 0.01]),
    r0_err = np.array([2.5, -0.8, 0.5]),
)
res_full = run_simulation(cfg_full, rng_seed=42)

# Also run perfect-state baseline for comparison
cfg_perf = default_config(
    n_steps     = 250,
    use_vision  = False,
    use_ekf     = False,
    u_max       = 0.3,
    pos_weight  = 15.0,
    vel_weight  = 1.5,
    pos_meas_std = 0.0,
    r0  = cfg_full.r0,
    v0  = cfg_full.v0,
    w0  = cfg_full.w0,
    r0_err = np.zeros(3),
)
res_perf = run_simulation(cfg_perf, rng_seed=0)

T  = res_full.n_steps_run + 1
t  = np.arange(T)

print(f"  Full pipeline: final range = {res_full.range_m[-1]:.2f} m, "
      f"Δv = {res_full.delta_v:.2f} m/s")
print(f"  Perfect state: final range = {res_perf.range_m[-1]:.2f} m, "
      f"Δv = {res_perf.delta_v:.2f} m/s")
if res_full.dock_step:
    print(f"  Docked at step {res_full.dock_step}")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 30 — 3-D LVLH trajectory
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Fig 30: 3-D LVLH trajectory...")

fig = plt.figure(figsize=(12, 5))
fig.suptitle('Fig 30 — Full Pipeline: LVLH Trajectory', fontsize=13, color='white')

# 3-D panel
ax3 = fig.add_subplot(1, 2, 1, projection='3d')
ax3.set_facecolor('#1a1a2e')
ax3.plot(res_full.r_true[:T, 0], res_full.r_true[:T, 1], res_full.r_true[:T, 2],
         color=BLUE, lw=2, label='True (full pipeline)')
ax3.plot(res_full.r_est[:T, 0],  res_full.r_est[:T, 1],  res_full.r_est[:T, 2],
         color=AMBER, lw=1.5, ls='--', label='EKF estimate')
ax3.plot(res_perf.r_true[:T, 0], res_perf.r_true[:T, 1], res_perf.r_true[:T, 2],
         color=GREY, lw=1.5, ls=':', label='Perfect state baseline')
ax3.scatter(*res_full.r_true[0], color=GREEN, s=80, zorder=5, label='Start')
ax3.scatter(0, 0, 0, color=RED, s=100, marker='*', zorder=5, label='Target')
ax3.set_xlabel('X radial [m]', color='white', fontsize=8)
ax3.set_ylabel('Y along-track [m]', color='white', fontsize=8)
ax3.set_zlabel('Z cross-track [m]', color='white', fontsize=8)
ax3.set_title('3-D LVLH', color='white')
ax3.legend(fontsize=7)

# X-Y top view
ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(res_full.r_true[:T, 0], res_full.r_true[:T, 1],
         color=BLUE, lw=2, label='True (full pipeline)')
ax2.plot(res_full.r_est[:T, 0],  res_full.r_est[:T, 1],
         color=AMBER, lw=1.5, ls='--', label='EKF estimate')
ax2.plot(res_perf.r_true[:T, 0], res_perf.r_true[:T, 1],
         color=GREY, lw=1.5, ls=':', label='Perfect state')
ax2.scatter(*res_full.r_true[0, :2], color=GREEN, s=80, zorder=5)
ax2.scatter(0, 0, color=RED, s=100, marker='*', zorder=5, label='Target')
ax2.set_xlabel('X radial [m]', color='white'); ax2.set_ylabel('Y along-track [m]', color='white')
ax2.set_title('Top view (X-Y)', color='white')
ax2.legend(fontsize=7); ax2.grid(alpha=0.2)

plt.tight_layout()
fname = os.path.join(OUTDIR, 'fig30_fullpipeline_trajectory.png')
plt.savefig(fname, dpi=150, bbox_inches='tight'); plt.close()
print(f"  -> {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 31 — Range + position error
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Fig 31: Range and estimation error...")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('Fig 31 — Range & Estimation Error vs Time', fontsize=13, color='white')

ax = axes[0]
ax.plot(t, res_full.range_m[:T], color=BLUE,  lw=2, label='Full pipeline range')
ax.plot(t, res_perf.range_m[:T], color=GREY, lw=1.5, ls=':', label='Perfect state range')
if res_full.dock_step:
    ax.axvline(res_full.dock_step, color=GREEN, ls='--', lw=1.5,
               label=f'Docked @ step {res_full.dock_step}')
ax.set_xlabel('Step'); ax.set_ylabel('Range [m]')
ax.set_title('Range to target', color='white')
ax.legend(fontsize=8); ax.grid(alpha=0.2)

ax = axes[1]
ax.plot(t, res_full.pos_error[:T], color=AMBER, lw=2, label='‖r_true − r_est‖')
ax.fill_between(t,
    res_full.pos_error[:T] - res_full.ekf_cov_pos[:T].mean(axis=1),
    res_full.pos_error[:T] + res_full.ekf_cov_pos[:T].mean(axis=1),
    color=AMBER, alpha=0.2, label='±1σ EKF')
ax.set_xlabel('Step'); ax.set_ylabel('Error [m]')
ax.set_title('Position estimation error', color='white')
ax.legend(fontsize=8); ax.grid(alpha=0.2)

plt.tight_layout()
fname = os.path.join(OUTDIR, 'fig31_range_and_error.png')
plt.savefig(fname, dpi=150, bbox_inches='tight'); plt.close()
print(f"  -> {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 32 — Thrust profiles + cumulative Δv
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Fig 32: Thrust profiles...")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('Fig 32 — Control Thrust & Cumulative Δv', fontsize=13, color='white')

t_ctrl = np.arange(res_full.n_steps_run)
labels = ['$u_x$ radial', '$u_y$ along-track', '$u_z$ cross-track']
cols   = [BLUE, AMBER, GREEN]

ax = axes[0]
for i, (lab, col) in enumerate(zip(labels, cols)):
    ax.plot(t_ctrl, res_full.controls[:res_full.n_steps_run, i],
            color=col, lw=1.5, label=lab)
ax.axhline( cfg_full.u_max, color=RED, ls='--', lw=1, label=f'±u_max={cfg_full.u_max}')
ax.axhline(-cfg_full.u_max, color=RED, ls='--', lw=1)
ax.set_xlabel('Step'); ax.set_ylabel('Thrust [m/s²]')
ax.set_title('Per-axis thrust commands', color='white')
ax.legend(fontsize=7); ax.grid(alpha=0.2)

ax = axes[1]
cum_dv_full = np.cumsum(np.linalg.norm(res_full.controls, axis=1)) * cfg_full.dt
cum_dv_perf = np.cumsum(np.linalg.norm(res_perf.controls, axis=1)) * cfg_perf.dt
ax.plot(t_ctrl, cum_dv_full[:res_full.n_steps_run], color=BLUE,  lw=2, label='Full pipeline')
ax.plot(t_ctrl, cum_dv_perf[:res_perf.n_steps_run], color=GREY,  lw=1.5, ls=':', label='Perfect state')
ax.set_xlabel('Step'); ax.set_ylabel('Cumulative Δv [m/s]')
ax.set_title('Cumulative Δv consumption', color='white')
ax.legend(fontsize=8); ax.grid(alpha=0.2)

plt.tight_layout()
fname = os.path.join(OUTDIR, 'fig32_thrust_profiles.png')
plt.savefig(fname, dpi=150, bbox_inches='tight'); plt.close()
print(f"  -> {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 33 — EPnP reprojection error
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Fig 33: EPnP reprojection error...")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('Fig 33 — Vision Pipeline: Reprojection Error vs Range', fontsize=13, color='white')

valid = np.isfinite(res_full.repr_errs)
t_v   = t_ctrl[valid[:res_full.n_steps_run]]
re_v  = res_full.repr_errs[:res_full.n_steps_run][valid[:res_full.n_steps_run]]

ax = axes[0]
ax.plot(t_v, re_v, color=RED, lw=1.5, alpha=0.8)
ax.set_xlabel('Step'); ax.set_ylabel('Reprojection error [px]')
ax.set_title('EPnP reprojection error vs time', color='white')
ax.grid(alpha=0.2)

ax = axes[1]
rng_v = res_full.range_m[:res_full.n_steps_run][valid[:res_full.n_steps_run]]
sc = ax.scatter(rng_v, re_v, c=t_v, cmap='plasma', s=12, alpha=0.7)
plt.colorbar(sc, ax=ax, label='Step')
ax.set_xlabel('Range to target [m]'); ax.set_ylabel('Reprojection error [px]')
ax.set_title('Reprojection error vs range', color='white')
ax.grid(alpha=0.2)

plt.tight_layout()
fname = os.path.join(OUTDIR, 'fig33_reprojection_error.png')
plt.savefig(fname, dpi=150, bbox_inches='tight'); plt.close()
print(f"  -> {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 34 — EKF covariance convergence
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Fig 34: EKF covariance convergence...")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('Fig 34 — EKF State Estimation Convergence', fontsize=13, color='white')

ax = axes[0]
cov_labels = ['σ_x', 'σ_y', 'σ_z']
cov_cols   = [BLUE, AMBER, GREEN]
for i, (lab, col) in enumerate(zip(cov_labels, cov_cols)):
    ax.plot(t, res_full.ekf_cov_pos[:T, i], color=col, lw=1.5, label=lab)
ax.set_xlabel('Step'); ax.set_ylabel('Position std-dev [m]')
ax.set_title('EKF position covariance (1σ)', color='white')
ax.legend(fontsize=8); ax.grid(alpha=0.2)

ax = axes[1]
perr = res_full.pos_error[:T]
ax.semilogy(t, np.maximum(perr, 1e-3), color=AMBER, lw=2, label='Position error')
# Rolling mean
win = 10
if len(perr) > win:
    rolling = np.convolve(perr, np.ones(win)/win, mode='valid')
    ax.semilogy(t[win-1:], np.maximum(rolling, 1e-3), color=RED, lw=2,
                ls='--', label=f'{win}-step average')
ax.set_xlabel('Step'); ax.set_ylabel('Error [m]  (log scale)')
ax.set_title('Estimation error convergence', color='white')
ax.legend(fontsize=8); ax.grid(alpha=0.2)

plt.tight_layout()
fname = os.path.join(OUTDIR, 'fig34_ekf_convergence.png')
plt.savefig(fname, dpi=150, bbox_inches='tight'); plt.close()
print(f"  -> {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# Animated GIF — rendezvous video
# ══════════════════════════════════════════════════════════════════════════════
print("Generating animated GIF (wireframe rendezvous fly-through)...")

gif_path = os.path.join(OUTDIR, 'phase7_rendezvous.gif')
# step=15 → ~17 frames; fast enough to run in the demo
exp = VideoExporter(res_full, fps=6, step=15)
exp.export_gif(gif_path, verbose=True)


# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("Phase 7 Summary")
print("=" * 60)
print(f"  Final range (full pipeline)  : {res_full.range_m[-1]:.3f} m")
print(f"  Final range (perfect state)  : {res_perf.range_m[-1]:.3f} m")
print(f"  Final est. error (EKF)       : {res_full.pos_error[-1]:.3f} m")
print(f"  Total Δv  (full pipeline)    : {res_full.delta_v:.2f} m/s")
valid_re = res_full.repr_errs[np.isfinite(res_full.repr_errs)]
if len(valid_re):
    print(f"  Median EPnP repr. error      : {np.median(valid_re):.2f} px")
print(f"  Docking achieved             : {'Yes @ step ' + str(res_full.dock_step) if res_full.dock_step else 'No'}")
print()
print("Figures saved to outputs/  (fig30 through fig34 + phase7_rendezvous.gif)")
print("Run tests: python -m pytest tests/test_phase7.py -v")

"""
Experiment D — MEKF vs UKF Relative Pose Tracking
===================================================
Simulates a rendezvous approach scenario and compares the Multiplicative
Extended Kalman Filter (MEKF) against the Unscented Kalman Filter (UKF)
for sequential pose estimation.

Setup
-----
  True trajectory  : HCW translation + torque-free Euler attitude, RK4.
  Initial range    : 50 m along-track.
  Target tumble    : ω = [0.01, 0.02, -0.015] rad/s (slow uncontrolled spin).
  Time step        : dt = 10 s  (camera at 0.1 Hz).
  Horizon          : N = 300 steps  (3 000 s ≈ 0.5 orbital period).
  Measurement noise: position σ = 0.5 m,  attitude σ = 0.05 rad  (~3°).
  Filter init error: position 4 m,  attitude ~17°  (realistic cold-start).

Metrics
-------
  Position error  [m]  : ||r_est − r_true||
  Attitude error  [°]  : 2 arccos |q_err_w|   (geodesic angle)
  Velocity error  [m/s]: ||v_est − v_true||

Outputs
-------
  outputs/expD_fig4_filter_tracking.png   — time histories
  outputs/expD_fig5_filter_rmse.png       — RMSE bar comparison

Run from the project root:
    python experiments/experiment_D.py
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from estimator import (
    make_ekf, make_ukf,
    pack_state, unpack_state,
    propagate_rk4, h_measurement,
    rotvec_to_quat, quat_mult, quat_norm,
    N_ORBITAL,
)

OUT = Path('outputs')
OUT.mkdir(exist_ok=True)

# ── style tokens (light / professional) ───────────────────────────────────────
BG    = '#FFFFFF'
PANEL = '#FAFAFA'
TEXT  = '#111111'
MUTED = '#555555'
GRID  = '#DDDDDD'
C1    = '#111111'   # MEKF (solid)
C2    = '#555555'   # UKF  (dashed)

def style_ax(ax, ylabel, title, xlabel=None):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(MUTED)
        spine.set_linewidth(0.8)
    ax.tick_params(colors=TEXT, labelsize=9.5, direction='in')
    ax.set_ylabel(ylabel, fontsize=10.5, color=TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.set_title(title, color=TEXT, fontsize=11, pad=6)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=1.0)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10.5, color=TEXT)

# ── simulation parameters ─────────────────────────────────────────────────────
RNG      = np.random.default_rng(42)
DT       = 10.0        # [s]  camera cadence
N_STEPS  = 300         # total steps → 3 000 s ≈ 0.5 orbital period

MEAS_POS_STD = 0.5    # [m]   position measurement noise 1σ
MEAS_ATT_STD = 0.05   # [rad] attitude measurement noise 1σ (~3°)

# ── true initial state ────────────────────────────────────────────────────────
n    = N_ORBITAL
r0   = np.array([0.0, 50.0, 0.0])           # 50 m along-track
v0   = np.array([0.0, 0.0,  0.0])
q0   = np.array([0.0, 0.0,  0.0, 1.0])      # identity attitude
w0   = np.array([0.01, 0.02, -0.015])        # slow tumble [rad/s]
x_true = pack_state(r0, v0, q0, w0)

# ── filter initial estimate (cold-start with deliberate errors) ───────────────
r_init = r0 + np.array([ 2.0, -3.0,  1.5])   # ~4 m position error
v_init = v0 + np.array([ 0.1, -0.10, 0.05])
# ~17° initial attitude error about x-axis
q_init = quat_norm(quat_mult(rotvec_to_quat(np.array([0.30, 0.0, 0.0])), q0))
w_init = w0 + np.array([ 0.005, -0.005, 0.002])
x_init = pack_state(r_init, v_init, q_init, w_init)

# ── build filters ─────────────────────────────────────────────────────────────
ekf = make_ekf(x_init, DT,
               pos0_std=4.0, vel0_std=0.5, att0_std=0.35, rate0_std=0.05,
               meas_pos_std=MEAS_POS_STD, meas_att_std=MEAS_ATT_STD,
               n=n)

ukf = make_ukf(x_init, DT,
               pos0_std=4.0, vel0_std=0.5, att0_std=0.35, rate0_std=0.05,
               meas_pos_std=MEAS_POS_STD, meas_att_std=MEAS_ATT_STD,
               n=n)

# ── helpers ───────────────────────────────────────────────────────────────────
def pos_err(x_est, x_true):
    return np.linalg.norm(x_est[:3] - x_true[:3])

def vel_err(x_est, x_true):
    return np.linalg.norm(x_est[3:6] - x_true[3:6])

def att_err_deg(x_est, x_true):
    """Geodesic rotation angle between estimated and true quaternion [°]."""
    q_e = x_est[6:10]
    q_t = x_true[6:10]
    # q_rel = q_est ⊗ q_true^{-1}
    q_t_inv = np.array([-q_t[0], -q_t[1], -q_t[2], q_t[3]])
    q_rel   = quat_mult(q_e, q_t_inv)
    w_abs   = np.clip(abs(q_rel[3]), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(w_abs))

# ── main simulation loop ──────────────────────────────────────────────────────
t_vec     = np.arange(N_STEPS) * DT / 60.0   # [min] for x-axis

ekf_pos   = np.empty(N_STEPS)
ekf_att   = np.empty(N_STEPS)
ekf_vel   = np.empty(N_STEPS)
ukf_pos   = np.empty(N_STEPS)
ukf_att   = np.empty(N_STEPS)
ukf_vel   = np.empty(N_STEPS)

ekf_accepted = 0
ukf_accepted = 0

t0_ekf = time.perf_counter()

x_sim = x_true.copy()
for k in range(N_STEPS):
    # ── true state at step k ──────────────────────────────────────────────────
    r_t, v_t, q_t, w_t = unpack_state(x_sim)

    # ── noisy measurement ─────────────────────────────────────────────────────
    z_clean = h_measurement(x_sim)          # [r(3), rotvec(3)]
    noise   = np.concatenate([
        RNG.normal(0.0, MEAS_POS_STD, 3),
        RNG.normal(0.0, MEAS_ATT_STD, 3),
    ])
    z_noisy = z_clean + noise

    # ── EKF predict + update ──────────────────────────────────────────────────
    ekf.predict(DT)
    info_e = ekf.update(z_noisy)
    if info_e['accepted']:
        ekf_accepted += 1

    # ── UKF predict + update ──────────────────────────────────────────────────
    ukf.predict(DT)
    info_u = ukf.update(z_noisy)
    if info_u['accepted']:
        ukf_accepted += 1

    # ── record errors ─────────────────────────────────────────────────────────
    ekf_pos[k] = pos_err(ekf.state, x_sim)
    ekf_att[k] = att_err_deg(ekf.state, x_sim)
    ekf_vel[k] = vel_err(ekf.state, x_sim)
    ukf_pos[k] = pos_err(ukf.state, x_sim)
    ukf_att[k] = att_err_deg(ukf.state, x_sim)
    ukf_vel[k] = vel_err(ukf.state, x_sim)

    # ── propagate true state ──────────────────────────────────────────────────
    x_sim = propagate_rk4(x_sim, DT, n)

t_ekf_total = time.perf_counter() - t0_ekf

# Time UKF separately (prediction-only approximation is unfair;
# we already have the full run above — just separate timing)
t0_ukf = time.perf_counter()
_ukf2  = make_ukf(x_init, DT, n=n)
_x2    = x_true.copy()
_rng2  = np.random.default_rng(42)
for _ in range(N_STEPS):
    z2 = h_measurement(_x2) + np.concatenate([
        _rng2.normal(0, MEAS_POS_STD, 3),
        _rng2.normal(0, MEAS_ATT_STD, 3)])
    _ukf2.predict(DT); _ukf2.update(z2)
    _x2 = propagate_rk4(_x2, DT, n)
t_ukf_total = time.perf_counter() - t0_ukf

# ── summary ───────────────────────────────────────────────────────────────────
def rmse(v): return np.sqrt(np.mean(v**2))

print("=" * 60)
print(f"{'Metric':<28} {'MEKF':>12}  {'UKF':>12}")
print("-" * 60)
print(f"{'Pos RMSE [m]':<28} {rmse(ekf_pos):>12.4f}  {rmse(ukf_pos):>12.4f}")
print(f"{'Att RMSE [°]':<28} {rmse(ekf_att):>12.4f}  {rmse(ukf_att):>12.4f}")
print(f"{'Vel RMSE [m/s]':<28} {rmse(ekf_vel):>12.4f}  {rmse(ukf_vel):>12.4f}")
print(f"{'Pos max err [m]':<28} {ekf_pos.max():>12.4f}  {ukf_pos.max():>12.4f}")
print(f"{'Att max err [°]':<28} {ekf_att.max():>12.4f}  {ukf_att.max():>12.4f}")
print(f"{'Meas accepted':<28} {ekf_accepted:>12d}  {ukf_accepted:>12d}")
print(f"{'Wall time [s]':<28} {t_ekf_total:>12.3f}  {t_ukf_total:>12.3f}")
print("=" * 60)

# ── Fig 4 — time histories ────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(9, 9), facecolor=BG)
fig.subplots_adjust(hspace=0.50, left=0.13, right=0.96, top=0.92, bottom=0.08)

xlim = (t_vec[0], t_vec[-1])

# panel 1 — position error
style_ax(axes[0], 'Position error  [m]', 'Position error')
l1, = axes[0].plot(t_vec, ekf_pos, color=C1, linewidth=1.5, label='MEKF')
l2, = axes[0].plot(t_vec, ukf_pos, color=C2, linewidth=1.5, linestyle='--', label='UKF')
axes[0].set_xlim(*xlim)
axes[0].legend(handles=[l1, l2], fontsize=8.5, facecolor=PANEL,
               edgecolor=MUTED, labelcolor=TEXT, loc='upper right')

# panel 2 — attitude error
style_ax(axes[1], 'Attitude error  [°]', 'Attitude error')
axes[1].plot(t_vec, ekf_att, color=C1, linewidth=1.5, label='MEKF')
axes[1].plot(t_vec, ukf_att, color=C2, linewidth=1.5, linestyle='--', label='UKF')
axes[1].set_xlim(*xlim)
axes[1].legend(fontsize=8.5, facecolor=PANEL, edgecolor=MUTED,
               labelcolor=TEXT, loc='upper right')

# panel 3 — velocity error
style_ax(axes[2], 'Velocity error  [m s⁻¹]', 'Velocity error',
         xlabel='Time  [min]')
axes[2].plot(t_vec, ekf_vel, color=C1, linewidth=1.5, label='MEKF')
axes[2].plot(t_vec, ukf_vel, color=C2, linewidth=1.5, linestyle='--', label='UKF')
axes[2].set_xlim(*xlim)
axes[2].legend(fontsize=8.5, facecolor=PANEL, edgecolor=MUTED,
               labelcolor=TEXT, loc='upper right')

fig.suptitle('MEKF vs UKF — Pose Tracking Error over Time'
             f'  (dt = {DT:.0f} s,  σ_pos = {MEAS_POS_STD} m,'
             f'  σ_att = {MEAS_ATT_STD} rad)',
             color=TEXT, fontsize=11, y=0.97)

out1 = OUT / 'expD_fig4_filter_tracking.png'
fig.savefig(out1, dpi=180, bbox_inches='tight', facecolor=BG)
plt.close(fig)
print(f"Saved → {out1}")

# ── Fig 5 — RMSE bar comparison ───────────────────────────────────────────────
metrics      = ['Position\nRMSE  [m]', 'Attitude\nRMSE  [°]', 'Velocity\nRMSE  [m/s]']
ekf_vals     = [rmse(ekf_pos), rmse(ekf_att), rmse(ekf_vel)]
ukf_vals     = [rmse(ukf_pos), rmse(ukf_att), rmse(ukf_vel)]

x_pos = np.arange(len(metrics))
bar_w = 0.30

fig2, ax = plt.subplots(figsize=(7, 5), facecolor=BG)
fig2.subplots_adjust(left=0.15, right=0.94, top=0.88, bottom=0.15)

ax.set_facecolor(PANEL)
for spine in ax.spines.values():
    spine.set_edgecolor(MUTED)
    spine.set_linewidth(0.8)
ax.tick_params(colors=TEXT, labelsize=9.5, direction='in')
ax.yaxis.label.set_color(TEXT)
ax.xaxis.label.set_color(TEXT)
ax.grid(True, color=GRID, linewidth=0.7, alpha=1.0, axis='y')
ax.set_axisbelow(True)

bars1 = ax.bar(x_pos - bar_w/2, ekf_vals, width=bar_w,
               color=C1, alpha=0.80, label='MEKF', zorder=3)
bars2 = ax.bar(x_pos + bar_w/2, ukf_vals, width=bar_w,
               color=C2, alpha=0.80, label='UKF',  zorder=3)

# Value labels above each bar
for bar in list(bars1) + list(bars2):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h * 1.02,
            f'{h:.3f}', ha='center', va='bottom',
            fontsize=8, color=TEXT)

ax.set_xticks(x_pos)
ax.set_xticklabels(metrics, fontsize=10, color=TEXT)
ax.set_ylabel('RMSE', fontsize=10.5, color=TEXT)
ax.set_title('MEKF vs UKF — RMSE Comparison', color=TEXT, fontsize=11, pad=8)
ax.legend(fontsize=9, facecolor=PANEL, edgecolor=MUTED, labelcolor=TEXT)
ax.set_ylim(0, max(ekf_vals + ukf_vals) * 1.25)

fig2.suptitle(f'N = {N_STEPS} steps,  dt = {DT:.0f} s'
              f'   |   MEKF wall-time {t_ekf_total:.2f} s'
              f',  UKF {t_ukf_total:.2f} s',
              color=MUTED, fontsize=9, y=0.97)

out2 = OUT / 'expD_fig5_filter_rmse.png'
fig2.savefig(out2, dpi=180, bbox_inches='tight', facecolor=BG)
plt.close(fig2)
print(f"Saved → {out2}")

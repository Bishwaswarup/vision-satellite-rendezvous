"""
Experiment E — LQR vs MPC Rendezvous Control
=============================================
Compares the infinite-horizon discrete LQR against receding-horizon MPC
for a spacecraft rendezvous manoeuvre in the LVLH frame under HCW dynamics.

Both controllers share the same Q, R, and u_max — the comparison is purely
on trajectory quality, propellant budget, and computation cost.  MPC uses
a finite prediction horizon (N=20 steps) with the LQR terminal cost; LQR
uses the infinite-horizon DARE solution.

Note on corridor constraints: MPC supports an approach-cone constraint,
but it requires the initial state to lie inside the cone (otherwise the QP
is infeasible from step 1 and SLSQP silently returns zero thrust).  For a
clean head-to-head comparison both controllers run unconstrained here.

Setup
-----
  Initial state   : [0, 50, 0, 0, 0, 0] m / m·s⁻¹  (50 m along-track)
  Target          : origin (rendezvous point)
  Control step    : dt = 10 s
  Horizon         : N_STEPS = 200  (2 000 s ≈ 0.33 orbital period)
  Thrust limit    : u_max = 0.1 m/s²
  LQR             : unconstrained infinite-horizon DARE
  MPC             : N = 20-step horizon, LQR terminal cost, no corridor

Metrics
-------
  Final range      [m]    : ||r_final||
  Total Δv         [m/s]  : Σ ||u_k|| dt
  Solve time per step [ms]: wall-clock / N_STEPS

Outputs
-------
  outputs/expE_fig6_lqr_vs_mpc_trajectory.png   — LVLH path + time histories
  outputs/expE_fig7_lqr_vs_mpc_comparison.png   — summary bar chart

Run from the project root:
    python experiments/experiment_E.py
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

from controller import make_lqr, make_mpc, N_ORBITAL_DEFAULT

OUT = Path('outputs')
OUT.mkdir(exist_ok=True)

# ── style tokens (light / professional) ───────────────────────────────────────
BG    = '#FFFFFF'
PANEL = '#FAFAFA'
TEXT  = '#111111'
MUTED = '#555555'
GRID  = '#DDDDDD'
C1    = '#111111'   # LQR  (solid)
C2    = '#555555'   # MPC  (dashed)

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
n        = N_ORBITAL_DEFAULT
DT       = 10.0        # [s]  control step
N_STEPS  = 200         # total steps → 2 000 s
N_HOR    = 20          # MPC prediction horizon [steps]
U_MAX    = 0.1         # [m/s²] per-axis thrust limit

# Initial state: 50 m along-track offset (classic V-bar hold point)
x0 = np.array([0.0, 50.0, 0.0, 0.0, 0.0, 0.0])

# ── build controllers ─────────────────────────────────────────────────────────
lqr = make_lqr(n=n, dt=DT,
               pos_weight=10.0, vel_weight=1.0, thrust_weight=1.0,
               u_max=U_MAX)

mpc = make_mpc(n=n, dt=DT, N=N_HOR,
               pos_weight=10.0, vel_weight=1.0, thrust_weight=1.0,
               u_max=U_MAX, cone_half_angle=None)   # unconstrained for fair comparison

print(f"LQR closed-loop stable : {lqr.is_stable()}")
print(f"LQR closed-loop eigs   : max|λ| = {np.abs(lqr.closed_loop_eigs()).max():.6f}")
print()

# ── simulate both controllers ─────────────────────────────────────────────────
rng = np.random.default_rng(0)

print("Running LQR simulation…")
t0 = time.perf_counter()
res_lqr = lqr.simulate(x0, n_steps=N_STEPS, rng=rng)
t_lqr   = time.perf_counter() - t0

print("Running MPC simulation… (this may take ~30–60 s)")
t0 = time.perf_counter()
res_mpc = mpc.simulate(x0, n_steps=N_STEPS, rng=rng)
t_mpc   = time.perf_counter() - t0

# ── unpack results ────────────────────────────────────────────────────────────
st_l = res_lqr['states']    # (N+1, 6)
ct_l = res_lqr['controls']  # (N,   3)
st_m = res_mpc['states']
ct_m = res_mpc['controls']

t_vec   = np.arange(N_STEPS + 1) * DT / 60.0      # [min]
t_ctrl  = np.arange(N_STEPS)     * DT / 60.0

# Range and control magnitude
range_l = np.linalg.norm(st_l[:, :3], axis=1)
range_m = np.linalg.norm(st_m[:, :3], axis=1)
umag_l  = np.linalg.norm(ct_l, axis=1)
umag_m  = np.linalg.norm(ct_m, axis=1)

# Cumulative Δv
dv_l = np.cumsum(umag_l) * DT
dv_m = np.cumsum(umag_m) * DT

# ── summary table ─────────────────────────────────────────────────────────────
print()
print("=" * 60)
print(f"{'Metric':<30} {'LQR':>12}  {'MPC':>12}")
print("-" * 60)
print(f"{'Final range [m]':<30} {range_l[-1]:>12.4f}  {range_m[-1]:>12.4f}")
print(f"{'Total Δv [m/s]':<30} {res_lqr['delta_v']:>12.4f}  {res_mpc['delta_v']:>12.4f}")
print(f"{'Max control [m/s²]':<30} {umag_l.max():>12.4f}  {umag_m.max():>12.4f}")
print(f"{'Sim wall-time [s]':<30} {t_lqr:>12.3f}  {t_mpc:>12.3f}")
print(f"{'Time per step [ms]':<30} {1e3*t_lqr/N_STEPS:>12.2f}  {1e3*t_mpc/N_STEPS:>12.2f}")
print("=" * 60)

# ── Fig 6 — trajectory + time histories ──────────────────────────────────────
fig = plt.figure(figsize=(11, 9), facecolor=BG)
fig.subplots_adjust(hspace=0.50, wspace=0.35,
                    left=0.09, right=0.96, top=0.92, bottom=0.08)

# 2-column layout: left = LVLH trajectory, right = 3 time-history panels
ax_traj = fig.add_subplot(1, 2, 1)
ax_r    = fig.add_subplot(3, 2, 2)
ax_u    = fig.add_subplot(3, 2, 4)
ax_dv   = fig.add_subplot(3, 2, 6)

# ── trajectories ─────────────────────────────────────────────────────────────
ax_traj.plot(st_l[:, 0], st_l[:, 1], color=C1, linewidth=1.6,
             label='LQR', zorder=3)
ax_traj.plot(st_m[:, 0], st_m[:, 1], color=C2, linewidth=1.6,
             linestyle='--', label='MPC', zorder=3)
ax_traj.plot(*x0[:2], marker='o', color=C1, markersize=7,
             markeredgecolor=BG, zorder=4)
ax_traj.plot(0, 0, marker='*', color=C1, markersize=10, zorder=5)
ax_traj.annotate('Start', xy=x0[:2], xytext=(x0[0]+2, x0[1]+2),
                 fontsize=8, color=TEXT)
ax_traj.annotate('Target', xy=(0, 0), xytext=(2, 4),
                 fontsize=8, color=TEXT)

ax_traj.set_facecolor(PANEL)
for spine in ax_traj.spines.values():
    spine.set_edgecolor(MUTED); spine.set_linewidth(0.8)
ax_traj.tick_params(colors=TEXT, labelsize=9.5, direction='in')
ax_traj.set_xlabel('Radial  x  [m]', fontsize=10.5, color=TEXT)
ax_traj.set_ylabel('Along-track  y  [m]', fontsize=10.5, color=TEXT)
ax_traj.set_title('LVLH Trajectory', color=TEXT, fontsize=11, pad=6)
ax_traj.grid(True, color=GRID, linewidth=0.7, alpha=1.0)
ax_traj.set_axisbelow(True)
ax_traj.legend(fontsize=8.5, facecolor=PANEL, edgecolor=MUTED,
               labelcolor=TEXT, loc='upper right')
ax_traj.set_aspect('equal', adjustable='datalim')

# ── range vs time ─────────────────────────────────────────────────────────────
style_ax(ax_r, 'Range  [m]', 'Range to target')
l1, = ax_r.plot(t_vec, range_l, color=C1, linewidth=1.5, label='LQR')
l2, = ax_r.plot(t_vec, range_m, color=C2, linewidth=1.5,
                linestyle='--', label='MPC')
ax_r.set_xlim(t_vec[0], t_vec[-1])
ax_r.legend(handles=[l1, l2], fontsize=8, facecolor=PANEL,
            edgecolor=MUTED, labelcolor=TEXT, loc='upper right')

# ── control magnitude vs time ─────────────────────────────────────────────────
style_ax(ax_u, '‖u‖  [m s⁻²]', 'Control magnitude')
ax_u.plot(t_ctrl, umag_l, color=C1, linewidth=1.3, label='LQR')
ax_u.plot(t_ctrl, umag_m, color=C2, linewidth=1.3, linestyle='--', label='MPC')
ax_u.axhline(U_MAX, color=MUTED, linewidth=0.8, linestyle=':', alpha=0.8,
             label=f'u_max={U_MAX}')
ax_u.set_xlim(t_vec[0], t_vec[-1])
ax_u.legend(fontsize=8, facecolor=PANEL, edgecolor=MUTED,
            labelcolor=TEXT, loc='upper right')

# ── cumulative Δv ─────────────────────────────────────────────────────────────
style_ax(ax_dv, 'Cumulative Δv  [m/s]', 'Propellant budget',
         xlabel='Time  [min]')
ax_dv.plot(t_ctrl, dv_l, color=C1, linewidth=1.5, label='LQR')
ax_dv.plot(t_ctrl, dv_m, color=C2, linewidth=1.5, linestyle='--', label='MPC')
ax_dv.set_xlim(t_vec[0], t_vec[-1])
ax_dv.legend(fontsize=8, facecolor=PANEL, edgecolor=MUTED,
             labelcolor=TEXT, loc='upper left')

fig.suptitle(f'LQR vs MPC — HCW Rendezvous  '
             f'(dt = {DT:.0f} s,  u_max = {U_MAX} m/s²,  MPC horizon N = {N_HOR})',
             color=TEXT, fontsize=11, y=0.97)

out1 = OUT / 'expE_fig6_lqr_vs_mpc_trajectory.png'
fig.savefig(out1, dpi=180, bbox_inches='tight', facecolor=BG)
plt.close(fig)
print(f"\nSaved → {out1}")

# ── Fig 7 — summary bar chart ─────────────────────────────────────────────────
metrics  = ['Final range\n[m]',
            'Total Δv\n[m/s]',
            'Max thrust\n[m/s²]',
            'Time/step\n[ms]']
lqr_vals = [range_l[-1],
            res_lqr['delta_v'],
            umag_l.max(),
            1e3 * t_lqr / N_STEPS]
mpc_vals = [range_m[-1],
            res_mpc['delta_v'],
            umag_m.max(),
            1e3 * t_mpc / N_STEPS]

x_pos = np.arange(len(metrics))
bar_w = 0.30

fig2, ax2 = plt.subplots(figsize=(8, 5), facecolor=BG)
fig2.subplots_adjust(left=0.12, right=0.95, top=0.88, bottom=0.15)

ax2.set_facecolor(PANEL)
for spine in ax2.spines.values():
    spine.set_edgecolor(MUTED); spine.set_linewidth(0.8)
ax2.tick_params(colors=TEXT, labelsize=9.5, direction='in')
ax2.yaxis.label.set_color(TEXT)
ax2.grid(True, color=GRID, linewidth=0.7, alpha=1.0, axis='y')
ax2.set_axisbelow(True)

bars1 = ax2.bar(x_pos - bar_w/2, lqr_vals, width=bar_w,
                color=C1, alpha=0.80, label='LQR', zorder=3)
bars2 = ax2.bar(x_pos + bar_w/2, mpc_vals, width=bar_w,
                color=C2, alpha=0.80, label='MPC', zorder=3)

for bar in list(bars1) + list(bars2):
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, h * 1.02,
             f'{h:.2f}', ha='center', va='bottom',
             fontsize=8, color=TEXT)

ax2.set_xticks(x_pos)
ax2.set_xticklabels(metrics, fontsize=9.5, color=TEXT)
ax2.set_title('LQR vs MPC — Performance Summary', color=TEXT, fontsize=11, pad=8)
ax2.legend(fontsize=9, facecolor=PANEL, edgecolor=MUTED, labelcolor=TEXT)
ax2.set_ylim(0, max(lqr_vals + mpc_vals) * 1.28)

fig2.suptitle(f'N_STEPS = {N_STEPS},  dt = {DT:.0f} s,  '
              f'MPC horizon N = {N_HOR},  u_max = {U_MAX} m/s²',
              color=MUTED, fontsize=9, y=0.97)

out2 = OUT / 'expE_fig7_lqr_vs_mpc_comparison.png'
fig2.savefig(out2, dpi=180, bbox_inches='tight', facecolor=BG)
plt.close(fig2)
print(f"Saved → {out2}")

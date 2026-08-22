"""
Experiment A — HCW Analytical vs Numerical Propagation
=======================================================
Compares the closed-form STM against the built-in DOP853 integrator
over 3 orbital periods on a 100 m periodic HCW orbit.
Also includes a numerical tolerance convergence test to validate
that discrepancies vanish as the solver tightens.

Outputs
-------
outputs/expA_fig1_dynamics_validation.png
outputs/expA_fig2_tolerance_convergence.png

Run from the project root:
    python experiments/experiment_A.py
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from dynamics import HCWPropagator
from dynamics.constants import LEO_N, LEO_PERIOD

OUT = Path('outputs')
OUT.mkdir(exist_ok=True)

# ── setup ─────────────────────────────────────────────────────────────────────
N     = LEO_N
T_ORB = LEO_PERIOD
prop  = HCWPropagator(n=N)
x0    = HCWPropagator.periodic_initial_condition(rho=100.0, n=N, theta0=0.0)

N_ORBITS = 3
t_end    = N_ORBITS * T_ORB
t_eval   = np.linspace(0.0, t_end, 3000)

# ── propagate (Time History) ──────────────────────────────────────────────────
x_anal = np.array([prop.propagate_analytical(x0, t) for t in t_eval])

num    = prop.propagate_numerical(x0, t_span=(0.0, t_end), t_eval=t_eval,
                                  rtol=1e-12, atol=1e-14)
x_num  = num['y'].T

# ── errors ────────────────────────────────────────────────────────────────────
pos_err = np.linalg.norm(x_anal[:, :3] - x_num[:, :3], axis=1)
vel_err = np.linalg.norm(x_anal[:, 3:] - x_num[:, 3:], axis=1)
t_hours = t_eval / 3600.0

print("=== Single Tolerance Test (rtol=1e-12) ===")
print(f"Max position error : {pos_err.max():.4e} m ({pos_err.max()*1e9:.2f} nm)")
print(f"Max velocity error : {vel_err.max():.4e} m/s ({vel_err.max()*1e9:.2f} nm/s)")
print("-" * 40)

# ── figure — clean, minimal (Time History) ────────────────────────────────────
DARK  = '#0D1B2A'
PANEL = '#162232'
GOLD  = '#E8A020'
BLUE  = '#5BA4CF'
GREEN = '#4CAF50'
GRID  = '#263A56'
TEXT  = '#D8E4F4'
MUTED = '#7A90AA'

fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 6), facecolor=DARK)
fig.subplots_adjust(hspace=0.42, left=0.12, right=0.96, top=0.93, bottom=0.10)

def style(ax, ylabel):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.tick_params(colors=MUTED, labelsize=9.5)
    ax.yaxis.label.set_color(TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.set_ylabel(ylabel, fontsize=10.5)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.7)

style(ax0, 'Position error  [nm]')
ax0.set_xlim(0, t_hours[-1])
for k in range(1, N_ORBITS + 1):
    ax0.axvline(k * T_ORB / 3600, color=MUTED, linewidth=0.7, linestyle='--', alpha=0.5)
ax0.plot(t_hours, pos_err * 1e9, color=GOLD, linewidth=1.4)
ax0.set_title('Position error', color=TEXT, fontsize=11, pad=6)

style(ax1, 'Velocity error  [nm s⁻¹]')
ax1.set_xlim(0, t_hours[-1])
for k in range(1, N_ORBITS + 1):
    ax1.axvline(k * T_ORB / 3600, color=MUTED, linewidth=0.7, linestyle='--', alpha=0.5)
ax1.plot(t_hours, vel_err * 1e9, color=BLUE, linewidth=1.4)
ax1.set_title('Velocity error', color=TEXT, fontsize=11, pad=6)
ax1.set_xlabel('Time  [h]', fontsize=10.5, color=TEXT)

out_path = OUT / 'expA_fig1_dynamics_validation.png'
fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor=DARK)
plt.close(fig)
print(f"Saved → {out_path}")

# ── convergence test (Multiple Tolerance Sweeps) ──────────────────────────────
print("\n=== Tolerance Convergence Tests ===")
print(f"{'DOP853 rtol':<12} | {'atol':<12} | {'Max Pos Error (m)':<18} | {'Max Vel Error (m/s)'}")
print("-" * 68)

# The specific tolerances to test for convergence validation
rtols = [1e-6, 1e-8, 1e-10, 1e-12, 1e-13]
max_pos_errors = []
max_vel_errors = []

for rtol in rtols:
    # Scale atol to be tighter than rtol to ensure solver respects bounds
    atol = rtol * 1e-2

    num_sweep = prop.propagate_numerical(x0, t_span=(0.0, t_end), t_eval=t_eval,
                                         rtol=rtol, atol=atol)
    x_num_sweep = num_sweep['y'].T

    # Calculate maximum errors across the entire propagation
    p_err = np.linalg.norm(x_anal[:, :3] - x_num_sweep[:, :3], axis=1).max()
    v_err = np.linalg.norm(x_anal[:, 3:] - x_num_sweep[:, 3:], axis=1).max()

    max_pos_errors.append(p_err)
    max_vel_errors.append(v_err)

    # Print formatted row
    print(f"{rtol:<12.0e} | {atol:<12.0e} | {p_err:<18.4e} | {v_err:.4e}")

# ── figure — tolerance convergence ────────────────────────────────────────────
fig2, (ax_pos, ax_vel) = plt.subplots(2, 1, figsize=(8, 7), facecolor=DARK)
fig2.subplots_adjust(hspace=0.35, left=0.15, right=0.92, top=0.90, bottom=0.10)

# Position Error Convergence Subplot
style(ax_pos, 'Max Pos Error  [m]')
ax_pos.loglog(rtols, max_pos_errors, marker='o', markersize=6, color=GREEN,
              linewidth=1.8, markeredgecolor=DARK, markeredgewidth=1.5)
ax_pos.invert_xaxis() # Loose (left) to Tight (right)
ax_pos.set_title('Position Error Convergence', color=TEXT, fontsize=11, pad=6)
ax_pos.grid(True, which="minor", color=GRID, linewidth=0.3, alpha=0.4)

# Velocity Error Convergence Subplot
style(ax_vel, 'Max Vel Error  [m/s]')
ax_vel.loglog(rtols, max_vel_errors, marker='s', markersize=6, color=BLUE,
              linewidth=1.8, markeredgecolor=DARK, markeredgewidth=1.5)
ax_vel.invert_xaxis() # Loose (left) to Tight (right)
ax_vel.set_title('Velocity Error Convergence', color=TEXT, fontsize=11, pad=6)
ax_vel.set_xlabel('Relative Tolerance (rtol)', fontsize=10.5, color=TEXT)
ax_vel.grid(True, which="minor", color=GRID, linewidth=0.3, alpha=0.4)

fig2.suptitle('DOP853 Numerical Solver Convergence', color=TEXT, fontsize=12, y=0.97)

out_path_conv = OUT / 'expA_fig2_tolerance_convergence.png'
fig2.savefig(out_path_conv, dpi=180, bbox_inches='tight', facecolor=DARK)
plt.close(fig2)
print(f"\nSaved → {out_path_conv}")
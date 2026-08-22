"""
Experiment A — HCW Analytical vs Numerical Propagation
=======================================================
Compares the closed-form STM against the built-in DOP853 integrator
over 3 orbital periods on a 100 m periodic HCW orbit.

Outputs
-------
outputs/expA_fig1_dynamics_validation.png

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

# ── propagate ─────────────────────────────────────────────────────────────────
x_anal = np.array([prop.propagate_analytical(x0, t) for t in t_eval])

num    = prop.propagate_numerical(x0, t_span=(0.0, t_end), t_eval=t_eval,
                                  rtol=1e-12, atol=1e-14)
x_num  = num['y'].T

# ── errors ────────────────────────────────────────────────────────────────────
pos_err = np.linalg.norm(x_anal[:, :3] - x_num[:, :3], axis=1)
vel_err = np.linalg.norm(x_anal[:, 3:] - x_num[:, 3:], axis=1)
t_hours = t_eval / 3600.0

print(f"Max position error : {pos_err.max():} m ({pos_err.max()*1e9:.2f} nm)")
print(f"Max velocity error : {vel_err.max():} m/s ({vel_err.max()*1e9:.2f} nm/s)")

# ── figure — clean, minimal ───────────────────────────────────────────────────
DARK  = '#0D1B2A'
PANEL = '#162232'
GOLD  = '#E8A020'
BLUE  = '#5BA4CF'
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
    ax.set_xlim(0, t_hours[-1])
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
    # orbit boundary ticks
    for k in range(1, N_ORBITS + 1):
        ax.axvline(k * T_ORB / 3600, color=MUTED, linewidth=0.7,
                   linestyle='--', alpha=0.5)

ax0.plot(t_hours, pos_err * 1e9, color=GOLD, linewidth=1.4)
style(ax0, 'Position error  [nm]')
ax0.set_title('Position error', color=TEXT, fontsize=11, pad=6)

ax1.plot(t_hours, vel_err * 1e9, color=BLUE, linewidth=1.4)
style(ax1, 'Velocity error  [nm s⁻¹]')
ax1.set_title('Velocity error', color=TEXT, fontsize=11, pad=6)
ax1.set_xlabel('Time  [h]', fontsize=10.5, color=TEXT)

out_path = OUT / 'expA_fig1_dynamics_validation.png'
fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor=DARK)
plt.close(fig)
print(f"Saved → {out_path}")

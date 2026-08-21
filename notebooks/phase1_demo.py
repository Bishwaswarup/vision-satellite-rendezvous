"""
phase1_demo.py
==============
Visualisation and validation demo for Phase 1 — Orbital Dynamics Engine.

Generates 4 publication-quality plots:
  Fig 1 — Drift-free HCW periodic orbit (LVLH x-y plane)
  Fig 2 — HCW analytical vs numerical (position error over 3 orbits)
  Fig 3 — J2 differential disturbance magnitude over one orbit
  Fig 4 — Pure HCW vs J2-perturbed relative trajectory comparison

Run from phase1/ directory:
    python notebooks/phase1_demo.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator

from dynamics import (
    HCWPropagator, YAStatTransition,
    j2_accel_eci, keplerian_to_eci, lvlh_to_eci_rotation,
    propagate_chief_eci, differential_j2_lvlh,
    MU_EARTH, J2, R_EARTH, LEO_SMA, LEO_N, LEO_PERIOD,
)
from dynamics.j2_perturb import J2PerturbedHCW

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':        'serif',
    'font.size':          11,
    'axes.linewidth':     1.1,
    'axes.grid':          True,
    'grid.alpha':         0.3,
    'grid.linestyle':     '--',
    'legend.framealpha':  0.85,
    'figure.dpi':         150,
})

BLUE   = '#1A6FBF'
ORANGE = '#E87722'
GREEN  = '#2CA02C'
RED    = '#D62728'
TEAL   = '#006D6D'

N   = LEO_N
T   = LEO_PERIOD
SMA = LEO_SMA
RHO = 50.0   # orbit half-amplitude [m]

os.makedirs('outputs', exist_ok=True)

# ── Helper ────────────────────────────────────────────────────────────────────
def fmt_orbit(ax):
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='both', direction='in', top=True, right=True)

print("Phase 1 Demo — Vision-Based Rendezvous Simulator")
print("=" * 55)

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Drift-Free HCW Periodic Orbit
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Fig 1: Drift-free periodic orbit...")

hcw = HCWPropagator(n=N)
n_pts = 2000
t_eval = np.linspace(0, 5 * T, n_pts * 5)

x0 = HCWPropagator.periodic_initial_condition(RHO, N, theta0=0.0)
sol = hcw.propagate_numerical(x0, (0, 5*T), t_eval=t_eval)
xs, ys, zs = sol['y'][0], sol['y'][1], sol['y'][2]
ts = sol['t'] / 3600.0   # hours

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# x-y plane
ax = axes[0]
sc = ax.scatter(xs, ys, c=ts, cmap='plasma', s=2, zorder=3)
ax.plot(x0[0], x0[1], 'k^', ms=8, label='IC', zorder=5)
ax.set_xlabel(r'Radial $x$ [m]')
ax.set_ylabel(r'Along-track $y$ [m]')
ax.set_title(r'HCW Drift-Free 2:1 Ellipse (5 orbits, LVLH $x$–$y$)')
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label('Time [hours]')
ax.set_aspect('equal')
fmt_orbit(ax)

# x-z plane
ax = axes[1]
x0_3d = HCWPropagator.periodic_initial_condition(RHO, N, theta0=0.3)
x0_3d[2]  = 30.0     # add cross-track amplitude
x0_3d[5]  = 0.0      # start at cross-track max
sol3d = hcw.propagate_numerical(x0_3d, (0, 2*T),
                                t_eval=np.linspace(0, 2*T, 4000))
ax.plot(sol3d['y'][0], sol3d['y'][2], color=TEAL, lw=1.2, label='x–z')
ax.plot(sol3d['y'][1], sol3d['y'][2], color=ORANGE, lw=1.2, ls='--', label='y–z')
ax.set_xlabel('[m]')
ax.set_ylabel(r'Cross-track $z$ [m]')
ax.set_title('Out-of-Plane + In-Plane (2 orbits, 3D trajectory)')
ax.legend(fontsize=9)
fmt_orbit(ax)

plt.tight_layout()
plt.savefig('outputs/fig1_periodic_orbit.png', dpi=150, bbox_inches='tight')
plt.close()
print("  -> outputs/fig1_periodic_orbit.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Analytical vs Numerical: Position Error
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Fig 2: Analytical vs Numerical error...")

n_pts = 3000
t_eval2 = np.linspace(0, 3*T, n_pts)
x0p     = HCWPropagator.periodic_initial_condition(RHO, N)

# Analytical
x_anal = np.column_stack([hcw.propagate_analytical(x0p, t) for t in t_eval2])
# Numerical
sol2   = hcw.propagate_numerical(x0p, (0, 3*T), t_eval=t_eval2)
x_num  = sol2['y']

err_pos = np.linalg.norm(x_anal[:3] - x_num[:3], axis=0)  # [m]
err_vel = np.linalg.norm(x_anal[3:] - x_num[3:], axis=0)  # [m/s]
t_hrs   = t_eval2 / 3600.0

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
ax1.semilogy(t_hrs, np.maximum(err_pos, 1e-16), color=BLUE, lw=1.4)
ax1.set_ylabel(r'Position error $\|\Delta\mathbf{r}\|_2$ [m]')
ax1.set_title('Analytical STM vs DOP853 Numerical Integration (3 orbits)')
ax1.axhline(1e-6, ls=':', color='gray', label='1 μm threshold')
ax1.legend(fontsize=9)
fmt_orbit(ax1)

ax2.semilogy(t_hrs, np.maximum(err_vel, 1e-19), color=ORANGE, lw=1.4)
ax2.set_xlabel('Time [hours]')
ax2.set_ylabel(r'Velocity error $\|\Delta\dot{\mathbf{r}}\|_2$ [m/s]')
ax2.axhline(1e-9, ls=':', color='gray', label='1 nm/s threshold')
ax2.legend(fontsize=9)
fmt_orbit(ax2)

# Mark orbit boundaries
for ax in (ax1, ax2):
    for k in [1, 2, 3]:
        ax.axvline(k * T / 3600, ls='--', color='lightgray', lw=0.8)

plt.tight_layout()
plt.savefig('outputs/fig2_analytical_vs_numerical.png', dpi=150, bbox_inches='tight')
plt.close()
print("  -> outputs/fig2_analytical_vs_numerical.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — J2 Differential Disturbance over One Orbit
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Fig 3: J2 differential disturbance...")

rv0 = keplerian_to_eci(SMA, 0.001, np.radians(51.6), 0.0, 0.0, 0.0)
t_eval3 = np.linspace(0, T, 1000)
rv_chief = propagate_chief_eci(rv0, t_eval3, include_j2=True)

# Deputy at a fixed LVLH offset
dr_lvlh = np.array([100.0, 0.0, 0.0])   # 100 m radial

da_list = []
for k in range(len(t_eval3)):
    r_c = rv_chief[:3, k]
    v_c = rv_chief[3:, k]
    da  = differential_j2_lvlh(r_c, v_c, dr_lvlh)
    da_list.append(da)

da_arr = np.array(da_list)   # (N, 3)
da_mag = np.linalg.norm(da_arr, axis=1)

fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
ax = axes[0]
ax.plot(t_eval3/3600, da_arr[:, 0]*1e6, color=BLUE,   lw=1.2, label=r'$\Delta a_x$ (radial)')
ax.plot(t_eval3/3600, da_arr[:, 1]*1e6, color=ORANGE, lw=1.2, label=r'$\Delta a_y$ (along-track)')
ax.plot(t_eval3/3600, da_arr[:, 2]*1e6, color=GREEN,  lw=1.2, label=r'$\Delta a_z$ (cross-track)')
ax.set_ylabel(r'Differential J$_2$ accel [$\mu$m/s²]')
ax.set_title(r'Differential J$_2$ Perturbation on Deputy ($\Delta r_x = 100$ m, $i=51.6°$)')
ax.legend(fontsize=9, ncol=3)
fmt_orbit(ax)

ax2 = axes[1]
ax2.plot(t_eval3/3600, da_mag*1e6, color=RED, lw=1.4, label=r'$|\Delta \mathbf{a}|$')
ax2.set_xlabel('Time [hours]')
ax2.set_ylabel(r'Magnitude [$\mu$m/s²]')
ax2.legend(fontsize=9)
fmt_orbit(ax2)

plt.tight_layout()
plt.savefig('outputs/fig3_j2_disturbance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  -> outputs/fig3_j2_disturbance.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Pure HCW vs J2-Perturbed Relative Trajectory
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Fig 4: HCW vs J2-perturbed trajectory...")

rv0_chief = keplerian_to_eci(SMA, 0.001, np.radians(51.6), 0.0, 0.0, 0.0)
t_eval4   = np.linspace(0, 5*T, 5000)
rv_chief4 = propagate_chief_eci(rv0_chief, t_eval4, include_j2=True)

# Build chief interpolator for J2 perturbed model
from scipy.interpolate import interp1d
interp_r = interp1d(t_eval4, rv_chief4[:3], axis=1, kind='cubic', fill_value='extrapolate')
interp_v = interp1d(t_eval4, rv_chief4[3:], axis=1, kind='cubic', fill_value='extrapolate')

def chief_ode(t):
    return interp_r(t).flatten(), interp_v(t).flatten()

# Initial state (100 m radial, drift-free)
x0_hcw = HCWPropagator.periodic_initial_condition(100.0, N)

# Pure HCW
sol_hcw = hcw.propagate_numerical(x0_hcw, (0, 5*T), t_eval=t_eval4)

# J2-perturbed
j2model = J2PerturbedHCW(n=N, chief_ode=chief_ode)
sol_j2  = j2model.propagate(x0_hcw, (0, 5*T), t_eval=t_eval4)

fig = plt.figure(figsize=(13, 5))
gs  = gridspec.GridSpec(1, 2, width_ratios=[1.4, 1])

# x-y plane comparison
ax1 = fig.add_subplot(gs[0])
ax1.plot(sol_hcw['y'][0], sol_hcw['y'][1],
         color=BLUE, lw=1.2, alpha=0.85, label='Pure HCW (ideal)')
ax1.plot(sol_j2['y'][0],  sol_j2['y'][1],
         color=RED,  lw=1.0, alpha=0.85, ls='--', label=r'HCW + $J_2$ perturbation')
ax1.set_xlabel(r'Radial $x$ [m]')
ax1.set_ylabel(r'Along-track $y$ [m]')
ax1.set_title(r'5-Orbit Relative Trajectory: HCW vs $J_2$-Perturbed ($i=51.6°$)')
ax1.legend(fontsize=9)
fmt_orbit(ax1)

# Separation error over time
ax2 = fig.add_subplot(gs[1])
delta = np.linalg.norm(sol_hcw['y'][:3] - sol_j2['y'][:3], axis=0)
ax2.plot(t_eval4/3600, delta, color=ORANGE, lw=1.3)
ax2.set_xlabel('Time [hours]')
ax2.set_ylabel(r'$|\mathbf{r}_{\rm HCW} - \mathbf{r}_{J_2}|$ [m]')
ax2.set_title(r'$J_2$ Model Divergence from HCW')
fmt_orbit(ax2)

# Mark orbit periods
for k in range(1, 6):
    ax2.axvline(k * T / 3600, ls=':', color='gray', lw=0.7)

plt.tight_layout()
plt.savefig('outputs/fig4_j2_vs_hcw.png', dpi=150, bbox_inches='tight')
plt.close()
print("  -> outputs/fig4_j2_vs_hcw.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Summary statistics
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 55)
print("Phase 1 Validation Summary")
print("=" * 55)

# Periodic orbit closure
xf = hcw.propagate_analytical(x0p, T)
pos_close = np.linalg.norm(xf[:3] - x0p[:3])
vel_close = np.linalg.norm(xf[3:] - x0p[3:])
print(f"  Periodic orbit position closure : {pos_close:.3e} m   (target < 1e-6 m)")
print(f"  Periodic orbit velocity closure : {vel_close:.3e} m/s (target < 1e-9 m/s)")

# Max analytical-vs-numerical error over 3 orbits
print(f"  Max position error (anal vs num): {err_pos.max():.3e} m")
print(f"  Max velocity error (anal vs num): {err_vel.max():.3e} m/s")

# J2 disturbance magnitude
print(f"  Max J2 diff accel (100 m offset): {da_mag.max()*1e6:.4f} μm/s²")

# 5-orbit J2 divergence
print(f"  Max J2 trajectory divergence 5T : {delta.max():.3f} m")
print()
print("All figures saved to outputs/")
print("Run tests with:  python -m pytest tests/test_phase1.py -v")

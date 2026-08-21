"""
phase2_demo.py
==============
Visualisation demo for Phase 2 — Target Kinematics & Tumbling Model.

Generates 4 plots:
  Fig 1 — Angular velocity components over time (tumble signature)
  Fig 2 — Polhode: ω trajectory in body frame (torque-free invariant curve)
  Fig 3 — Euler angles (roll, pitch, yaw) over time
  Fig 4 — Energy & angular momentum conservation error

Run from phase1/ directory:
    python notebooks/phase2_demo.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import AutoMinorLocator

from target import (
    RigidBodyAttitude, ariane_upper_stage, cubesat_3u,
    q_to_euler321, euler321_to_q, attitude_error_deg,
)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.linewidth': 1.1, 'axes.grid': True,
    'grid.alpha': 0.3, 'grid.linestyle': '--',
    'legend.framealpha': 0.85, 'figure.dpi': 150,
})
BLUE, ORANGE, GREEN, RED = '#1A6FBF', '#E87722', '#2CA02C', '#D62728'
TEAL, PURPLE = '#006D6D', '#9467BD'

os.makedirs('outputs', exist_ok=True)

print("Phase 2 Demo — Target Kinematics & Tumbling Model")
print("=" * 55)

# ── Setup: Ariane upper stage tumbling ────────────────────────────────────────
body   = ariane_upper_stage()
T_SIM  = 3 * 3600      # 3 hours
N_PTS  = 10000

# Initial attitude: small tilt from identity
q0 = euler321_to_q(np.radians(5), np.radians(10), np.radians(0))

# Tumbling initial condition:
# - Fast spin about intermediate axis (most unstable — Dzhanibekov effect)
# - Small transverse components to seed the tumble
w0 = np.array([0.002, 0.030, 0.001])   # rad/s  (~1.7 deg/s net)

t_eval = np.linspace(0, T_SIM, N_PTS)

print(f"Propagating Ariane upper stage for {T_SIM/3600:.0f} hours...")
result = body.propagate(q0, w0, (0, T_SIM), t_eval=t_eval)
t_hrs  = result['t'] / 3600.0
q_arr  = result['q']
w_arr  = result['w']
euler  = result['euler']

print(f"  Success: {result['success']}")

# ── Conservation report ───────────────────────────────────────────────────────
report = body.check_conservation(result, rtol=1e-5)
print(f"  T0 = {report['T0']:.4f} J,  H0 = {report['H0_mag']:.4f} kg·m²/s")
print(f"  Max energy error    : {report['max_E_err']:.2e}  -> {'PASS' if report['energy_ok'] else 'FAIL'}")
print(f"  Max momentum error  : {report['max_H_err']:.2e}  -> {'PASS' if report['momentum_ok'] else 'FAIL'}")

def fmt(ax):
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='both', direction='in', top=True, right=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Angular velocity components
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Fig 1: Angular velocity components...")
fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
labels = [r'$\omega_1$ (axial)',
          r'$\omega_2$ (transverse 1)',
          r'$\omega_3$ (transverse 2)']
colors = [BLUE, ORANGE, GREEN]

for i, (ax, lbl, col) in enumerate(zip(axes, labels, colors)):
    ax.plot(t_hrs, np.degrees(w_arr[i]), color=col, lw=0.9, label=lbl)
    ax.set_ylabel(r'$\omega$ [deg/s]')
    ax.legend(fontsize=9, loc='upper right')
    fmt(ax)

axes[-1].set_xlabel('Time [hours]')
axes[0].set_title('Ariane 44L Upper Stage — Tumbling Angular Velocity (3 hours)')
plt.tight_layout()
plt.savefig('outputs/fig5_angular_velocity.png', dpi=150, bbox_inches='tight')
plt.close()
print("  -> outputs/fig5_angular_velocity.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Polhode (ω trajectory in body frame) + Herpolhode
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Fig 2: Polhode...")

fig = plt.figure(figsize=(13, 5))

# Polhode in body frame (3D)
ax3d = fig.add_subplot(121, projection='3d')
w1, w2, w3 = np.degrees(w_arr[0]), np.degrees(w_arr[1]), np.degrees(w_arr[2])
sc = ax3d.scatter(w1, w2, w3, c=t_hrs, cmap='plasma', s=0.3, alpha=0.7)
ax3d.set_xlabel(r'$\omega_1$ [deg/s]', fontsize=9)
ax3d.set_ylabel(r'$\omega_2$ [deg/s]', fontsize=9)
ax3d.set_zlabel(r'$\omega_3$ [deg/s]', fontsize=9)
ax3d.set_title('Polhode ($\omega$ in Body Frame)', fontsize=10)
fig.colorbar(sc, ax=ax3d, label='Time [hr]', shrink=0.6)

# Energy ellipsoid cross-section (ω₁-ω₂ plane)
ax2 = fig.add_subplot(122)
ax2.plot(w1, w2, color=BLUE, lw=0.7, alpha=0.8, label='Polhode projection')
ax2.set_xlabel(r'$\omega_1$ [deg/s]')
ax2.set_ylabel(r'$\omega_2$ [deg/s]')
ax2.set_title(r'Polhode Projection ($\omega_1$–$\omega_2$)', fontsize=10)
ax2.set_aspect('equal')
fmt(ax2)

plt.tight_layout()
plt.savefig('outputs/fig6_polhode.png', dpi=150, bbox_inches='tight')
plt.close()
print("  -> outputs/fig6_polhode.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Euler angles
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Fig 3: Euler angles...")
fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
angle_labels = ['Roll $\\phi$ [deg]', 'Pitch $\\theta$ [deg]', 'Yaw $\\psi$ [deg]']
angle_colors = [BLUE, ORANGE, GREEN]

for i, (ax, lbl, col) in enumerate(zip(axes, angle_labels, angle_colors)):
    ax.plot(t_hrs, np.degrees(euler[i]), color=col, lw=0.8)
    ax.set_ylabel(lbl)
    fmt(ax)

axes[-1].set_xlabel('Time [hours]')
axes[0].set_title('Euler 3-2-1 Angles During Torque-Free Tumble')
plt.tight_layout()
plt.savefig('outputs/fig7_euler_angles.png', dpi=150, bbox_inches='tight')
plt.close()
print("  -> outputs/fig7_euler_angles.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Conservation errors
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Fig 4: Conservation errors...")

T_arr  = np.array([body.rotational_ke(w_arr[:, k]) for k in range(N_PTS)])
H_arr  = np.array([
    np.linalg.norm(body.angular_momentum_inertial(q_arr[:, k], w_arr[:, k]))
    for k in range(N_PTS)
])
T0, H0 = T_arr[0], H_arr[0]
E_err  = np.abs(T_arr - T0) / abs(T0)
H_err  = np.abs(H_arr - H0) / H0

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
ax1.semilogy(t_hrs, np.maximum(E_err, 1e-18), color=RED, lw=1.2)
ax1.set_ylabel('Relative KE error')
ax1.set_title('Conservation Errors — DOP853 Integration (rtol=1e-10, atol=1e-12)')
ax1.axhline(1e-6, ls='--', color='gray', lw=0.8, label='1 ppm threshold')
ax1.legend(fontsize=9)
fmt(ax1)

ax2.semilogy(t_hrs, np.maximum(H_err, 1e-18), color=TEAL, lw=1.2)
ax2.set_ylabel('Relative |H| error')
ax2.set_xlabel('Time [hours]')
ax2.axhline(1e-6, ls='--', color='gray', lw=0.8, label='1 ppm threshold')
ax2.legend(fontsize=9)
fmt(ax2)

plt.tight_layout()
plt.savefig('outputs/fig8_conservation.png', dpi=150, bbox_inches='tight')
plt.close()
print("  -> outputs/fig8_conservation.png")

# ── Additional: CubeSat vs Ariane comparison ──────────────────────────────────
print("Generating Fig extra: CubeSat tumble comparison...")

cube = cubesat_3u()
q0c  = euler321_to_q(np.radians(2), np.radians(5), 0)
w0c  = np.array([0.01, 0.08, 0.005])   # faster tumble for small sat
t_c  = np.linspace(0, 600, 5000)       # 10 minutes
res_c = cube.propagate(q0c, w0c, (0, 600), t_eval=t_c)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
t_c_min = res_c['t'] / 60.0

ax = axes[0]
for i, (col, lbl) in enumerate(zip([BLUE, ORANGE, GREEN],
                                    [r'$\omega_1$', r'$\omega_2$', r'$\omega_3$'])):
    ax.plot(t_c_min, np.degrees(res_c['w'][i]), color=col, lw=0.9, label=lbl)
ax.set_xlabel('Time [min]')
ax.set_ylabel('[deg/s]')
ax.set_title('3U CubeSat Tumble (10 min)')
ax.legend(fontsize=9)
fmt(ax)

ax = axes[1]
t_sub = t_hrs[:3000]
for i, (col, lbl) in enumerate(zip([BLUE, ORANGE, GREEN],
                                    [r'$\omega_1$', r'$\omega_2$', r'$\omega_3$'])):
    ax.plot(t_sub, np.degrees(w_arr[i, :3000]), color=col, lw=0.6, alpha=0.8, label=lbl)
ax.set_xlabel('Time [hours]')
ax.set_ylabel('[deg/s]')
ax.set_title('Ariane Upper Stage Tumble (first hour)')
ax.legend(fontsize=9)
fmt(ax)

plt.suptitle('Debris Body Comparison', fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('outputs/fig9_body_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  -> outputs/fig9_body_comparison.png")

# ── Summary ───────────────────────────────────────────────────────────────────
q_norm_drift = np.abs(np.linalg.norm(q_arr, axis=0) - 1.0).max()
print()
print("=" * 55)
print("Phase 2 Validation Summary")
print("=" * 55)
print(f"  Ariane inertia [I1, I2, I3] : "
      f"{np.diag(body.I)[0]:.0f}, {np.diag(body.I)[1]:.0f}, {np.diag(body.I)[2]:.0f} kg·m²")
print(f"  Simulation duration          : {T_SIM/3600:.0f} hours,  {N_PTS} steps")
print(f"  Max quaternion norm drift    : {q_norm_drift:.2e}  (target < 1e-10)")
print(f"  Max KE relative error        : {report['max_E_err']:.2e}  (target < 1e-6)")
print(f"  Max |H| relative error       : {report['max_H_err']:.2e}  (target < 1e-6)")
print(f"  Energy conserved             : {'YES' if report['energy_ok'] else 'NO'}")
print(f"  Momentum conserved           : {'YES' if report['momentum_ok'] else 'NO'}")
print()
print("Figures saved to outputs/  (fig5 through fig9)")
print("Run tests: python -m pytest tests/test_phase2.py -v")

"""
phase6_demo.py
==============
Phase 6 demonstration — LQR / MPC Rendezvous Control.

Figures
-------
  Fig 25 : LQR rendezvous trajectory (3-D LVLH + error vs time)
  Fig 26 : MPC rendezvous trajectory vs LQR comparison
  Fig 27 : Thrust profiles — LQR vs MPC (per-axis)
  Fig 28 : Approach corridor constraint (MPC with cone)
  Fig 29 : Full chain: EKF + LQR on Ariane approach (30→0 m)

Run:
    cd /home/user/phase1
    python notebooks/phase6_demo.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from controller import make_lqr, make_mpc, hcw_discrete, N_ORBITAL_DEFAULT
from estimator import (
    pack_state, unpack_state, propagate_rk4, h_measurement,
    rotvec_to_quat, make_ekf,
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

DT = 1.0
N  = N_ORBITAL_DEFAULT


# ══════════════════════════════════════════════════════════════════════════════
# Fig 25 — LQR rendezvous trajectory
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Fig 25: LQR rendezvous trajectory...")

x0  = np.array([20., 5., -3., -0.1, 0.02, 0.01])
lqr = make_lqr(u_max=0.2, pos_weight=10., vel_weight=1.)
sim = lqr.simulate(x0, n_steps=300)

states = sim['states']
t = np.arange(len(states)) * DT
r_norm = np.linalg.norm(states[:, :3], axis=1)
v_norm = np.linalg.norm(states[:, 3:], axis=1)

fig = plt.figure(figsize=(14, 5))
fig.suptitle('Fig 25 — LQR Rendezvous Trajectory', fontsize=13, color='white')

# 3-D trajectory
ax3 = fig.add_subplot(131, projection='3d')
ax3.plot(states[:, 0], states[:, 1], states[:, 2],
         color=BLUE, lw=1.5, label='Chaser path')
ax3.scatter(*states[0, :3], color=AMBER, s=60, zorder=5, label='Start')
ax3.scatter(0, 0, 0, color=GREEN, s=80, marker='*', zorder=5, label='Docking port')
ax3.set_xlabel('X [m]'); ax3.set_ylabel('Y [m]'); ax3.set_zlabel('Z [m]')
ax3.set_title('LVLH Trajectory', color='white')
ax3.legend(fontsize=8)

# Position error
ax1 = fig.add_subplot(132)
ax1.semilogy(t, r_norm + 1e-6, color=BLUE, lw=1.5)
ax1.axhline(0.5, ls='--', color=RED, lw=1, label='Docking threshold 0.5 m')
ax1.set_xlabel('Time [s]'); ax1.set_ylabel('‖r‖ [m]')
ax1.set_title('Position Error', color='white'); ax1.legend(fontsize=9); ax1.grid(alpha=0.2)

# Velocity
ax2 = fig.add_subplot(133)
ax2.plot(t, v_norm, color=AMBER, lw=1.5)
ax2.axhline(0.05, ls='--', color=GREEN, lw=1, label='Safe dock speed 0.05 m/s')
ax2.set_xlabel('Time [s]'); ax2.set_ylabel('‖v‖ [m/s]')
ax2.set_title('Closing Speed', color='white'); ax2.legend(fontsize=9); ax2.grid(alpha=0.2)

plt.tight_layout()
fname = os.path.join(OUTDIR, 'fig25_lqr_trajectory.png')
plt.savefig(fname, dpi=150, bbox_inches='tight'); plt.close()
print(f"  -> {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 26 — LQR vs MPC comparison
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Fig 26: LQR vs MPC comparison...")

x0c = np.array([20., 4., -2., -0.1, 0., 0.])
lqr2 = make_lqr(u_max=0.2)
mpc2 = make_mpc(N=20, u_max=0.2)

sim_l = lqr2.simulate(x0c, n_steps=200)
sim_m = mpc2.simulate(x0c, n_steps=200)

t2 = np.arange(201) * DT
r_l = np.linalg.norm(sim_l['states'][:, :3], axis=1)
r_m = np.linalg.norm(sim_m['states'][:, :3], axis=1)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('Fig 26 — LQR vs MPC Rendezvous Performance', fontsize=13, color='white')

axes[0].semilogy(t2, r_l + 1e-6, color=BLUE,  lw=2, label=f'LQR  Δv={sim_l["delta_v"]:.2f} m/s')
axes[0].semilogy(t2, r_m + 1e-6, color=AMBER, lw=2, ls='--', label=f'MPC  Δv={sim_m["delta_v"]:.2f} m/s')
axes[0].axhline(0.5, ls=':', color=RED, lw=1)
axes[0].set_xlabel('Time [s]'); axes[0].set_ylabel('‖r‖ [m]')
axes[0].set_title('Position Error', color='white'); axes[0].legend(); axes[0].grid(alpha=0.2)

# Δv budget bar chart
axes[1].bar(['LQR', 'MPC'], [sim_l['delta_v'], sim_m['delta_v']],
            color=[BLUE, AMBER], width=0.5)
axes[1].set_ylabel('Total Δv [m/s]')
axes[1].set_title('Fuel Consumption', color='white'); axes[1].grid(alpha=0.2, axis='y')
for i, v in enumerate([sim_l['delta_v'], sim_m['delta_v']]):
    axes[1].text(i, v + 0.05, f'{v:.2f}', ha='center', color='white', fontsize=11)

plt.tight_layout()
fname = os.path.join(OUTDIR, 'fig26_lqr_vs_mpc.png')
plt.savefig(fname, dpi=150, bbox_inches='tight'); plt.close()
print(f"  -> {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 27 — Thrust profiles
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Fig 27: Thrust profiles...")

fig, axes = plt.subplots(3, 2, figsize=(12, 7), sharex='col')
fig.suptitle('Fig 27 — Thrust Profiles: LQR (left) vs MPC (right)', fontsize=13, color='white')
labels = ['aₓ [m/s²]', 'a_y [m/s²]', 'a_z [m/s²]']
colors = [BLUE, AMBER, GREEN]
t_ctrl = np.arange(200) * DT

for i in range(3):
    axes[i, 0].step(t_ctrl, sim_l['controls'][:, i], where='post', color=colors[i], lw=1.2)
    axes[i, 0].axhline( 0.2, ls='--', color=GREY, lw=0.8)
    axes[i, 0].axhline(-0.2, ls='--', color=GREY, lw=0.8)
    axes[i, 0].set_ylabel(labels[i], color='white')
    axes[i, 0].grid(alpha=0.2)

    axes[i, 1].step(t_ctrl, sim_m['controls'][:, i], where='post', color=colors[i], lw=1.2)
    axes[i, 1].axhline( 0.2, ls='--', color=GREY, lw=0.8)
    axes[i, 1].axhline(-0.2, ls='--', color=GREY, lw=0.8)
    axes[i, 1].grid(alpha=0.2)

axes[0, 0].set_title('LQR', color='white')
axes[0, 1].set_title('MPC', color='white')
axes[-1, 0].set_xlabel('Time [s]', color='white')
axes[-1, 1].set_xlabel('Time [s]', color='white')

plt.tight_layout()
fname = os.path.join(OUTDIR, 'fig27_thrust_profiles.png')
plt.savefig(fname, dpi=150, bbox_inches='tight'); plt.close()
print(f"  -> {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 28 — Approach cone constraint
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Fig 28: Approach corridor constraint...")

# Setup:  chaser starts INSIDE the 30° cone (angle = arctan(14/30) ≈ 25°)
# but with an outward along-track velocity (+0.08 m/s) that would push it
# beyond the corridor without the cone constraint.
# Free MPC: ignores the corridor → drifts to ~87° off-axis during approach.
# Cone MPC: enforces |y| ≤ x·tan(30°) at each horizon step → stays narrow.
CONE_ANGLE = 30.0
x0_cone = np.array([30., 14., 0., -0.12, 0.08, 0.])  # inside cone, outward drift
mpc_free = make_mpc(N=20, u_max=0.3, cone_half_angle=None)
mpc_cone = make_mpc(N=20, u_max=0.3, cone_half_angle=CONE_ANGLE)

sim_free = mpc_free.simulate(x0_cone, n_steps=200)
sim_cone = mpc_cone.simulate(x0_cone, n_steps=200)

# Angle-vs-time for the right panel
tan_c = np.tan(np.radians(CONE_ANGLE))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f'Fig 28 — MPC Approach Cone Constraint (half-angle {CONE_ANGLE}°)',
             fontsize=13, color='white')

# ── Left: X-Y trajectory overlay ──────────────────────────────────────────────
ax = axes[0]
x_grid = np.linspace(0, x0_cone[0] + 2, 400)

ax.fill_between(x_grid,  x_grid * tan_c, (x0_cone[1] + 6) * np.ones_like(x_grid),
                color=RED, alpha=0.15, label='Forbidden zone')
ax.fill_between(x_grid, -(x0_cone[1] + 6) * np.ones_like(x_grid), -x_grid * tan_c,
                color=RED, alpha=0.15)
ax.plot(x_grid,  x_grid * tan_c, '--', color=AMBER, lw=1.5, label=f'±{CONE_ANGLE}° boundary')
ax.plot(x_grid, -x_grid * tan_c, '--', color=AMBER, lw=1.5)

sf = sim_free['states']
sc = sim_cone['states']
ax.plot(sf[:, 0], sf[:, 1], color=RED,  lw=2.5, label='Free MPC',       zorder=4)
ax.plot(sc[:, 0], sc[:, 1], color=BLUE, lw=2.5, label='Cone MPC (30°)', zorder=5)
ax.scatter(*x0_cone[:2],  color=GREEN, s=80,  zorder=6, label='Start')
ax.scatter(0, 0,           color='white', s=100, marker='*', zorder=6, label='Docking port')
ax.set_xlabel('X radial [m]', color='white')
ax.set_ylabel('Y along-track [m]', color='white')
ax.set_title('X-Y trajectory (top view)', color='white')
ax.legend(fontsize=8)
ax.grid(alpha=0.2)
ax.set_xlim(-2, x0_cone[0] + 3)
ax.set_ylim(-(x0_cone[1] + 5), x0_cone[1] + 5)

# ── Right: approach angle vs time ─────────────────────────────────────────────
ax2 = axes[1]
t = np.arange(len(sf))
angle_free = np.degrees(np.arctan2(np.abs(sf[:, 1]), np.abs(sf[:, 0]) + 1e-6))
angle_cone = np.degrees(np.arctan2(np.abs(sc[:, 1]), np.abs(sc[:, 0]) + 1e-6))
ax2.plot(t, angle_free, color=RED,  lw=2, label='Free MPC')
ax2.plot(t, angle_cone, color=BLUE, lw=2, label=f'Cone MPC ({CONE_ANGLE}°)')
ax2.axhline(CONE_ANGLE, color=AMBER, ls='--', lw=1.5, label=f'{CONE_ANGLE}° limit')
ax2.fill_between(t, CONE_ANGLE, angle_free.max() + 5,
                 color=RED, alpha=0.10, label='Forbidden region')
ax2.set_xlabel('Time step', color='white')
ax2.set_ylabel('Off-axis angle [deg]', color='white')
ax2.set_title('Approach angle vs time', color='white')
ax2.legend(fontsize=8); ax2.grid(alpha=0.2)
ax2.set_ylim(0, angle_free.max() + 5)

plt.tight_layout()
fname = os.path.join(OUTDIR, 'fig28_approach_cone.png')
plt.savefig(fname, dpi=150, bbox_inches='tight'); plt.close()
print(f"  -> {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 29 — Full chain: EKF + LQR on Ariane approach
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Fig 29: Full EKF + LQR pipeline...")

Phi_d, Gamma_d = hcw_discrete(N, DT)

r0f = np.array([30., 3., -1.5])
v0f = np.array([-0.1, 0., 0.])
q0f = rotvec_to_quat(np.array([0., 0., 0.15]))
w0f = np.array([0., 0., 0.02])
x_true = pack_state(r0f, v0f, q0f, w0f)

x_init_ekf = pack_state(r0f + np.array([3., -1., 0.5]), v0f, q0f, w0f)
ekf = make_ekf(x_init_ekf, DT, pos0_std=3.0, att0_std=0.2, meas_pos_std=0.6)
lqr_f = make_lqr(u_max=0.25)

rng  = np.random.default_rng(7)
N_F  = 250

gt_pos, ekf_pos, ctrl_hist = [], [], []
u = np.zeros(3)

for _ in range(N_F):
    # EKF predict + inject known thrust
    ekf.predict(DT)
    ekf.x[:3]  += (Gamma_d @ u)[:3]
    ekf.x[3:6] += (Gamma_d @ u)[3:]

    # LQR on EKF estimate
    r_e, v_e = unpack_state(ekf.state)[:2]
    u = lqr_f.control(np.concatenate([r_e, v_e]))
    ctrl_hist.append(u.copy())

    # True dynamics
    r_t, v_t, q_t, w_t = unpack_state(x_true)
    rv_new = Phi_d @ np.concatenate([r_t, v_t]) + Gamma_d @ u
    x_att  = propagate_rk4(pack_state(r_t, v_t, q_t, w_t), DT)
    x_true = pack_state(rv_new[:3], rv_new[3:],
                        unpack_state(x_att)[2], unpack_state(x_att)[3])

    # Measurement update (every step, 0.6 m noise)
    z = h_measurement(x_true)
    z[:3] += rng.normal(0, 0.6, 3)
    z[3:]  += rng.normal(0, 0.05, 3)
    ekf.update(z)

    gt_pos.append(unpack_state(x_true)[0].copy())
    ekf_pos.append(r_e.copy())

gt_pos  = np.array(gt_pos)
ekf_pos = np.array(ekf_pos)
ctrl_h  = np.array(ctrl_hist)
t_f     = np.arange(N_F) * DT

fig, axes = plt.subplots(2, 2, figsize=(13, 7))
fig.suptitle('Fig 29 — Full Chain: EKF + LQR on Ariane Rendezvous (30 m → Docking)', fontsize=12, color='white')

# Position components
for i, (lbl, col) in enumerate(zip(['X radial', 'Y along-track', 'Z cross-track'],
                                    [BLUE, AMBER, GREEN])):
    ax = axes[i // 2, i % 2]
    ax.plot(t_f, gt_pos[:, i],  color=GREY,  lw=1.5, ls='--', label='Ground truth')
    ax.plot(t_f, ekf_pos[:, i], color=col,   lw=2,   label='EKF estimate')
    ax.set_ylabel(f'{lbl} [m]', color='white')
    ax.set_xlabel('Time [s]', color='white')
    ax.legend(fontsize=9); ax.grid(alpha=0.2)

# Range
ax4 = axes[1, 1]
range_gt  = np.linalg.norm(gt_pos,  axis=1)
range_ekf = np.linalg.norm(ekf_pos, axis=1)
ax4.semilogy(t_f, range_gt  + 1e-3, color=GREY,  lw=1.5, ls='--', label='True range')
ax4.semilogy(t_f, range_ekf + 1e-3, color=RED,   lw=2,   label='EKF range estimate')
ax4.axhline(0.5, ls=':', color=GREEN, lw=1.2, label='Docking threshold 0.5 m')
ax4.set_ylabel('Range [m]', color='white')
ax4.set_xlabel('Time [s]', color='white')
ax4.set_title('Closing Range', color='white')
ax4.legend(fontsize=9); ax4.grid(alpha=0.2)

plt.tight_layout()
fname = os.path.join(OUTDIR, 'fig29_ekf_lqr_pipeline.png')
plt.savefig(fname, dpi=150, bbox_inches='tight'); plt.close()
print(f"  -> {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
r_lqr_final = np.linalg.norm(sim_l['states'][-1, :3])
r_mpc_final = np.linalg.norm(sim_m['states'][-1, :3])
r_full_final = np.linalg.norm(unpack_state(x_true)[0])

print()
print("=" * 55)
print("Phase 6 Validation Summary")
print("=" * 55)
print(f"  LQR final position error       : {r_lqr_final:.4f} m")
print(f"  MPC final position error       : {r_mpc_final:.4f} m")
print(f"  LQR total Δv                   : {sim_l['delta_v']:.3f} m/s")
print(f"  MPC total Δv                   : {sim_m['delta_v']:.3f} m/s")
print(f"  Full EKF+LQR final range       : {r_full_final:.3f} m")
print()
print("Figures saved to outputs/  (fig25 through fig29)")
print("Run tests: python -m pytest tests/test_phase6.py -v")

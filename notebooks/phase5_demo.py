"""
phase5_demo.py
==============
Phase 5 demonstration — EKF / UKF Relative Pose Tracking.

Figures
-------
  Fig 20 : EKF position and attitude error vs time
  Fig 21 : UKF position and attitude error vs time
  Fig 22 : EKF vs UKF RMSE comparison (position + attitude, 20 MC runs)
  Fig 23 : Covariance consistency — Normalised Innovation Squared (NIS)
  Fig 24 : RANSAC + EPnP → EKF pipeline on Ariane rendezvous approach

Run:
    cd /home/user/phase1
    python notebooks/phase5_demo.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from estimator import (
    pack_state, unpack_state,
    propagate_rk4, h_measurement,
    rotvec_to_quat, quat_to_rotvec, quat_norm,
    make_ekf, make_ukf, N_ORBITAL,
)
from estimator.state import quat_mult, default_measurement_noise
from pose import solve_epnp, solve_pnp_ransac, refine_pose
from vision import rendezvous_camera, ariane_model, look_at_rotation

OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTDIR, exist_ok=True)

plt.style.use('dark_background')
BLUE   = '#4FC3F7'
AMBER  = '#FFB74D'
GREEN  = '#81C784'
RED    = '#EF9A9A'
PURPLE = '#CE93D8'
GREY   = '#78909C'

DT   = 1.0
N_STEPS = 100
MEAS_POS_STD = 0.5
MEAS_ATT_STD = 0.05

# ── Shared helpers ─────────────────────────────────────────────────────────────

def make_gt_trajectory(seed=0, n_steps=N_STEPS):
    rng   = np.random.default_rng(seed)
    r0    = np.array([30.0, 2.0, -1.0])
    v0    = np.array([-0.06, 0.0, 0.0])
    axis  = rng.normal(size=3); axis /= np.linalg.norm(axis)
    q0    = rotvec_to_quat(axis * rng.uniform(0.1, 0.4))
    w0    = rng.uniform(-0.03, 0.03, 3)
    x     = pack_state(r0, v0, q0, w0)
    traj  = [x]
    for _ in range(n_steps - 1):
        x = propagate_rk4(x, DT)
        traj.append(x)
    return np.array(traj)   # (n_steps, 13)


def noisy_meas(x_true, rng):
    z = h_measurement(x_true)
    z[:3] += rng.normal(0, MEAS_POS_STD, 3)
    z[3:]  += rng.normal(0, MEAS_ATT_STD, 3)
    return z


def pos_err_deg(x_est, x_true):
    r_e, v_e, q_e, w_e = unpack_state(x_est)
    r_t, v_t, q_t, w_t = unpack_state(x_true)
    pos_err = np.linalg.norm(r_e - r_t)
    q_inv   = np.array([-q_t[0], -q_t[1], -q_t[2], q_t[3]])
    dq      = quat_norm(quat_mult(q_e, q_inv))
    att_deg = np.degrees(2 * np.arccos(np.clip(abs(dq[3]), 0, 1)))
    return pos_err, att_deg


def run_filter(filter_type, x_init_perturb, traj, rng, **kw):
    """Run EKF or UKF on a ground-truth trajectory."""
    x_init = traj[0].copy()
    x_init[:3] += x_init_perturb
    x_init[6:10] = quat_norm(
        quat_mult(rotvec_to_quat(np.array([0., 0., 0.2])), x_init[6:10]))

    if filter_type == 'ekf':
        filt = make_ekf(x_init, DT, pos0_std=3.0, att0_std=0.3, **kw)
    else:
        filt = make_ukf(x_init, DT, pos0_std=3.0, att0_std=0.3, **kw)

    pos_errs, att_errs, nis_vals = [], [], []

    for k in range(len(traj)):
        x_true = traj[k]
        filt.predict(DT)
        z = noisy_meas(x_true, rng)
        info = filt.update(z)

        pe, ae = pos_err_deg(filt.state, x_true)
        pos_errs.append(pe)
        att_errs.append(ae)
        if info['accepted']:
            nis_vals.append(info['mahal'])

    return np.array(pos_errs), np.array(att_errs), np.array(nis_vals)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 20 — EKF position & attitude error vs time
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Fig 20: EKF error vs time...")
traj = make_gt_trajectory(seed=0)
rng  = np.random.default_rng(42)
pos_e, att_e, _ = run_filter('ekf', np.array([3., -1., 0.5]), traj, rng)
t = np.arange(N_STEPS) * DT

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
fig.suptitle('Fig 20 — EKF: State Estimation Error vs Time', fontsize=13, color='white')
axes[0].plot(t, pos_e, color=BLUE, lw=1.5, label='Position error [m]')
axes[0].axhline(MEAS_POS_STD, ls='--', color=GREY, lw=0.8, label=f'Meas noise 1σ = {MEAS_POS_STD} m')
axes[0].set_ylabel('Position error [m]', color='white')
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.2)

axes[1].plot(t, att_e, color=AMBER, lw=1.5, label='Attitude error [°]')
axes[1].axhline(np.degrees(MEAS_ATT_STD), ls='--', color=GREY, lw=0.8,
                label=f'Meas noise 1σ = {np.degrees(MEAS_ATT_STD):.1f}°')
axes[1].set_ylabel('Attitude error [°]', color='white')
axes[1].set_xlabel('Time [s]', color='white')
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.2)

plt.tight_layout()
fname = os.path.join(OUTDIR, 'fig20_ekf_error_vs_time.png')
plt.savefig(fname, dpi=150, bbox_inches='tight')
plt.close()
print(f"  -> {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 21 — UKF position & attitude error vs time
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Fig 21: UKF error vs time...")
rng  = np.random.default_rng(42)   # same seed → same measurements
pos_u, att_u, _ = run_filter('ukf', np.array([3., -1., 0.5]), traj, rng)

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
fig.suptitle('Fig 21 — UKF: State Estimation Error vs Time', fontsize=13, color='white')
axes[0].plot(t, pos_u, color=GREEN, lw=1.5, label='Position error [m]')
axes[0].axhline(MEAS_POS_STD, ls='--', color=GREY, lw=0.8, label=f'1σ = {MEAS_POS_STD} m')
axes[0].set_ylabel('Position error [m]', color='white'); axes[0].legend(fontsize=9); axes[0].grid(alpha=0.2)
axes[1].plot(t, att_u, color=PURPLE, lw=1.5, label='Attitude error [°]')
axes[1].axhline(np.degrees(MEAS_ATT_STD), ls='--', color=GREY, lw=0.8)
axes[1].set_ylabel('Attitude error [°]', color='white')
axes[1].set_xlabel('Time [s]', color='white'); axes[1].legend(fontsize=9); axes[1].grid(alpha=0.2)
plt.tight_layout()
fname = os.path.join(OUTDIR, 'fig21_ukf_error_vs_time.png')
plt.savefig(fname, dpi=150, bbox_inches='tight')
plt.close()
print(f"  -> {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 22 — EKF vs UKF RMSE comparison (20 Monte-Carlo runs)
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Fig 22: EKF vs UKF MC comparison...")
N_MC = 20
ekf_pos_all = np.zeros((N_MC, N_STEPS))
ukf_pos_all = np.zeros((N_MC, N_STEPS))
ekf_att_all = np.zeros((N_MC, N_STEPS))
ukf_att_all = np.zeros((N_MC, N_STEPS))

for mc in range(N_MC):
    traj_mc = make_gt_trajectory(seed=mc)
    rng_mc  = np.random.default_rng(mc + 100)
    init_pert = np.array([2.0, 0, 0])
    ekf_pos_all[mc], ekf_att_all[mc], _ = run_filter('ekf', init_pert, traj_mc, rng_mc)

    rng_mc2 = np.random.default_rng(mc + 100)   # same measurements
    ukf_pos_all[mc], ukf_att_all[mc], _ = run_filter('ukf', init_pert, traj_mc, rng_mc2)

ekf_rmse_pos = np.sqrt((ekf_pos_all**2).mean(axis=0))
ukf_rmse_pos = np.sqrt((ukf_pos_all**2).mean(axis=0))
ekf_rmse_att = np.sqrt((ekf_att_all**2).mean(axis=0))
ukf_rmse_att = np.sqrt((ukf_att_all**2).mean(axis=0))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle(f'Fig 22 — EKF vs UKF RMSE ({N_MC} Monte-Carlo Runs)', fontsize=13, color='white')

axes[0].plot(t, ekf_rmse_pos, color=BLUE,  lw=2, label='EKF')
axes[0].plot(t, ukf_rmse_pos, color=GREEN, lw=2, ls='--', label='UKF')
axes[0].set_title('Position RMSE [m]', color='white')
axes[0].set_xlabel('Time [s]', color='white'); axes[0].legend(); axes[0].grid(alpha=0.2)

axes[1].plot(t, ekf_rmse_att, color=BLUE,  lw=2, label='EKF')
axes[1].plot(t, ukf_rmse_att, color=GREEN, lw=2, ls='--', label='UKF')
axes[1].set_title('Attitude RMSE [°]', color='white')
axes[1].set_xlabel('Time [s]', color='white'); axes[1].legend(); axes[1].grid(alpha=0.2)

plt.tight_layout()
fname = os.path.join(OUTDIR, 'fig22_ekf_vs_ukf_rmse.png')
plt.savefig(fname, dpi=150, bbox_inches='tight')
plt.close()
print(f"  -> {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 23 — NIS (Normalised Innovation Squared) consistency check
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Fig 23: NIS covariance consistency...")
traj_c = make_gt_trajectory(seed=7)
rng_c  = np.random.default_rng(77)
_, _, nis_ekf = run_filter('ekf', np.array([1., 0., 0.]), traj_c, rng_c)
rng_c2 = np.random.default_rng(77)
_, _, nis_ukf = run_filter('ukf', np.array([1., 0., 0.]), traj_c, rng_c2)

fig, ax = plt.subplots(figsize=(10, 4))
fig.suptitle('Fig 23 — NIS (Normalised Innovation Squared) — Covariance Consistency', fontsize=13)

t_nis = np.arange(len(nis_ekf))
ax.plot(t_nis, nis_ekf, '.', color=BLUE,  ms=4, alpha=0.7, label='EKF NIS')
ax.plot(np.arange(len(nis_ukf)), nis_ukf, 'x', color=GREEN, ms=4, alpha=0.7, label='UKF NIS')

# χ² 95th-percentile bounds for dof=6
ax.axhline(12.592, ls='--', color=AMBER, lw=1.2, label='χ²(6) 95% bound = 12.6')
ax.axhline(1.635,  ls='--', color=RED,   lw=1.2, label='χ²(6) 5% bound = 1.6')
ax.set_ylabel('NIS', color='white'); ax.set_xlabel('Measurement index', color='white')
ax.legend(fontsize=9); ax.grid(alpha=0.2)
ax.set_ylim(0, 25)
plt.tight_layout()
fname = os.path.join(OUTDIR, 'fig23_nis_consistency.png')
plt.savefig(fname, dpi=150, bbox_inches='tight')
plt.close()
print(f"  -> {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 24 — Full pipeline: RANSAC + EPnP → EKF on Ariane approach
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Fig 24: RANSAC+EPnP → EKF on Ariane approach...")

CAM   = rendezvous_camera()
K_cam = CAM.K
model = ariane_model()
kp3d  = model.keypoint_array          # (15, 3)

# Generate a closing trajectory (30 m → 5 m over 50 steps)
N_APP = 50
ranges = np.linspace(30, 8, N_APP)
r_gt_list = np.column_stack([ranges, np.zeros(N_APP), np.zeros(N_APP)])

# Fixed attitude: slowly tumbling
axis0 = np.array([0., 0., 1.])
omega_tgt = 0.02   # rad/s about z
q_list = []
for k in range(N_APP):
    angle = omega_tgt * k * DT
    q_list.append(rotvec_to_quat(axis0 * angle))

rng_app = np.random.default_rng(55)
meas_pos_pnp = []    # PnP raw position measurements
filt_pos_ekf  = []    # EKF filtered positions
gt_positions  = []    # ground truth

# Initialise EKF at step 0
x_init_app = pack_state(r_gt_list[0], np.array([-0.44, 0., 0.]),
                         q_list[0], np.array([0., 0., omega_tgt]))
ekf_app = make_ekf(x_init_app, DT,
                   pos0_std=1.0, att0_std=0.2,
                   meas_pos_std=0.8, meas_att_std=0.1)

for k in range(N_APP):
    r_t = r_gt_list[k]
    q_t = q_list[k]
    x_true_k = pack_state(r_t, np.array([-0.44, 0., 0.]), q_t, np.array([0., 0., omega_tgt]))

    # Project keypoints through camera
    from estimator.state import quat_to_dcm
    R_t = quat_to_dcm(q_t)
    P_cam = (R_t @ kp3d.T).T + r_t
    z_pts = P_cam[:, 2]
    u = K_cam[0,0] * P_cam[:, 0] / z_pts + K_cam[0, 2]
    v = K_cam[1,1] * P_cam[:, 1] / z_pts + K_cam[1, 2]
    kp2d = np.stack([u, v], axis=-1)
    kp2d_noisy = kp2d + rng_app.normal(0, 2.0, kp2d.shape)

    # RANSAC + EPnP
    R_pnp, t_pnp, mask, meta = solve_pnp_ransac(
        kp3d, kp2d_noisy, K_cam, threshold_px=6.0, max_iter=200, seed=k)

    if meta['success'] and mask.sum() >= 5:
        R_ref, t_ref, _ = refine_pose(R_pnp, t_pnp, kp3d[mask], kp2d_noisy[mask], K_cam)
        rv_pnp = quat_to_rotvec(np.array([-0., 0., 0., 1.]))   # placeholder
        from pose.epnp import EPnPSolver
        from estimator.state import dcm_to_quat
        q_pnp  = dcm_to_quat(R_ref)
        rv_pnp = quat_to_rotvec(q_pnp)
        z_ekf  = np.concatenate([t_ref, rv_pnp])
        meas_pos_pnp.append(t_ref.copy())
    else:
        z_ekf = None
        meas_pos_pnp.append(np.full(3, np.nan))

    ekf_app.predict(DT)
    if z_ekf is not None:
        ekf_app.update(z_ekf)

    filt_pos_ekf.append(unpack_state(ekf_app.state)[0].copy())
    gt_positions.append(r_t.copy())

gt_positions  = np.array(gt_positions)
filt_pos_ekf  = np.array(filt_pos_ekf)
meas_pos_pnp  = np.array(meas_pos_pnp)
t_app = np.arange(N_APP) * DT

fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
fig.suptitle('Fig 24 — RANSAC+EPnP → EKF: Ariane Rendezvous Approach (30→8 m)', fontsize=13, color='white')
labels_xyz = ['X (radial) [m]', 'Y (along-track) [m]', 'Z (cross-track) [m]']
for i, (ax, lbl) in enumerate(zip(axes, labels_xyz)):
    ax.plot(t_app, gt_positions[:, i],   color=GREY,   lw=1.5, ls='--', label='Ground truth')
    ax.plot(t_app, meas_pos_pnp[:, i],   '.', color=RED,    ms=5,  alpha=0.6, label='PnP raw')
    ax.plot(t_app, filt_pos_ekf[:, i],   color=BLUE,   lw=2,   label='EKF filtered')
    ax.set_ylabel(lbl, color='white'); ax.legend(fontsize=9, loc='upper right'); ax.grid(alpha=0.2)
axes[-1].set_xlabel('Time [s]', color='white')
plt.tight_layout()
fname = os.path.join(OUTDIR, 'fig24_ariane_ekf_pipeline.png')
plt.savefig(fname, dpi=150, bbox_inches='tight')
plt.close()
print(f"  -> {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 55)
print("Phase 5 Validation Summary")
print("=" * 55)

final_pos_err_ekf = np.linalg.norm(filt_pos_ekf[-1] - gt_positions[-1])
print(f"  EKF final position error (Ariane pipeline) : {final_pos_err_ekf:.3f} m")
print(f"  EKF mean pos RMSE ({N_MC} MC runs)             : {ekf_rmse_pos[50:].mean():.3f} m  (t>50s)")
print(f"  UKF mean pos RMSE ({N_MC} MC runs)             : {ukf_rmse_pos[50:].mean():.3f} m  (t>50s)")
print(f"  EKF mean att RMSE ({N_MC} MC runs)             : {ekf_rmse_att[50:].mean():.3f}°  (t>50s)")
print(f"  UKF mean att RMSE ({N_MC} MC runs)             : {ukf_rmse_att[50:].mean():.3f}°  (t>50s)")
print(f"  NIS in-bounds fraction (EKF)               : "
      f"{((nis_ekf > 1.635) & (nis_ekf < 12.592)).mean()*100:.1f}%  (expect ~90%)")
print()
print("Figures saved to outputs/  (fig20 through fig24)")
print("Run tests: python -m pytest tests/test_phase5.py -v")

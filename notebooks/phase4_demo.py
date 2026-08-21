"""
phase4_demo.py
==============
Visualisation demo for Phase 4 — PnP Pose Estimation.

Generates:
  Fig 15  — Reprojection overlay: ground truth vs EPnP vs EPnP+Refined
  Fig 16  — Rotation & translation error vs pixel noise level
  Fig 17  — RANSAC convergence: inlier count vs iteration
  Fig 18  — Reprojection error CDF (EPnP raw vs refined)
  Fig 19  — Pose estimation on full Ariane keypoint set (RANSAC + refine)

Run from phase1/ directory:
    python notebooks/phase4_demo.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from vision import rendezvous_camera, ariane_model, look_at_rotation, DebrisRenderer, RendererConfig
from pose import solve_epnp, solve_pnp_ransac, refine_pose
from pose.epnp import EPnPSolver

os.makedirs('outputs', exist_ok=True)

print("Phase 4 Demo — PnP Pose Estimation")
print("=" * 55)

CAM  = rendezvous_camera()
K    = CAM.K
BLUE, ORANGE, GREEN, RED = '#1A6FBF', '#E87722', '#2CA02C', '#D62728'
TEAL, PURPLE = '#006D6D', '#9467BD'

plt.rcParams.update({'font.family': 'serif', 'font.size': 10,
                     'axes.grid': True, 'grid.alpha': 0.3,
                     'grid.linestyle': '--', 'figure.dpi': 150})


def random_rotation(seed):
    rng = np.random.default_rng(seed)
    u1, u2, u3 = rng.uniform(size=3)
    q = np.array([np.sqrt(1-u1)*np.sin(2*np.pi*u2),
                  np.sqrt(1-u1)*np.cos(2*np.pi*u2),
                  np.sqrt(u1)*np.sin(2*np.pi*u3),
                  np.sqrt(u1)*np.cos(2*np.pi*u3)])
    q0, q1, q2, q3 = q
    return np.array([
        [1-2*(q2**2+q3**2), 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)],
        [2*(q1*q2+q0*q3), 1-2*(q1**2+q3**2), 2*(q2*q3-q0*q1)],
        [2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), 1-2*(q1**2+q2**2)],
    ])


def project(pts3d, R, t, K):
    P = (R @ pts3d.T).T + t
    z = np.maximum(P[:, 2], 1e-6)
    u = K[0,0]*P[:,0]/z + K[0,2]
    v = K[1,1]*P[:,1]/z + K[1,2]
    return np.stack([u, v], axis=-1)


def synth_data(N=20, noise=0.0, seed=0, outlier_frac=0.0):
    rng = np.random.default_rng(seed)
    R_gt = random_rotation(seed)
    t_gt = np.array([0.3, -0.2, 28.0])
    pts3d = rng.uniform(-4.0, 4.0, (N, 3))
    pts2d = project(pts3d, R_gt, t_gt, K)
    if noise > 0:
        pts2d += rng.normal(0, noise, pts2d.shape)
    if outlier_frac > 0:
        n_out = int(N * outlier_frac)
        idx = rng.choice(N, n_out, replace=False)
        pts2d[idx] = rng.uniform(100, 900, (n_out, 2))
    return pts3d, pts2d, R_gt, t_gt


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 15 — Reprojection overlay
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Fig 15: Reprojection overlay...")

pts3d, pts2d_noisy, R_gt, t_gt = synth_data(N=25, noise=2.0, seed=3)
pts2d_gt = project(pts3d, R_gt, t_gt, K)

R_e, t_e, errs_e, ok_e = solve_epnp(pts3d, pts2d_noisy, K)
R_r, t_r, cost_r       = refine_pose(R_e, t_e, pts3d, pts2d_noisy, K)
pts2d_epnp = project(pts3d, R_e, t_e, K)
pts2d_ref  = project(pts3d, R_r, t_r, K)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
titles = ['Ground Truth', 'EPnP Estimate', 'EPnP + GN Refined']
data2d = [pts2d_gt, pts2d_epnp, pts2d_ref]
colors = [GREEN, ORANGE, BLUE]

for ax, title, d2, col in zip(axes, titles, data2d, colors):
    ax.scatter(pts2d_noisy[:, 0], pts2d_noisy[:, 1],
               c='white', s=30, zorder=2, edgecolors='gray',
               linewidths=0.5, label='Observed (noisy)')
    ax.scatter(d2[:, 0], d2[:, 1],
               c=col, s=50, zorder=3, marker='+', linewidths=1.5,
               label='Projected')
    for obs, proj in zip(pts2d_noisy, d2):
        ax.plot([obs[0], proj[0]], [obs[1], proj[1]],
                color=col, lw=0.5, alpha=0.5)
    ax.set_xlim(300, 750); ax.set_ylim(300, 750)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('u [px]'); ax.set_ylabel('v [px]')
    ax.legend(fontsize=8)
    ax.set_aspect('equal')

errs_ref = EPnPSolver._reprojection_errors(pts3d, pts2d_noisy, R_r, t_r, K)
plt.suptitle(f'Reprojection Overlay  |  noise σ=2 px  |  '
             f'EPnP mean={errs_e.mean():.2f} px  '
             f'Refined mean={errs_ref.mean():.2f} px',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/fig15_reprojection_overlay.png', dpi=150, bbox_inches='tight')
plt.close()
print("  -> outputs/fig15_reprojection_overlay.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 16 — Error vs noise level
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Fig 16: Error vs noise level...")

noise_levels = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
n_trials = 20

rot_errs_e, rot_errs_r = [], []
t_errs_e,   t_errs_r   = [], []
repr_errs_e, repr_errs_r = [], []

for sigma in noise_levels:
    re_list, rr_list, te_list, tr_list, ree_list, rer_list = [], [], [], [], [], []
    for seed in range(n_trials):
        pts3d_t, pts2d_t, R_gt_t, t_gt_t = synth_data(N=20, noise=sigma, seed=seed)
        R_e_t, t_e_t, errs_t, ok_t = solve_epnp(pts3d_t, pts2d_t, K)
        if not ok_t or R_e_t is None:
            continue
        R_r_t, t_r_t, _ = refine_pose(R_e_t, t_e_t, pts3d_t, pts2d_t, K)

        def rot_err_deg(Ra, Rb):
            c = np.clip((np.trace(Ra @ Rb.T) - 1) / 2, -1, 1)
            return np.degrees(np.arccos(abs(c)))

        re_list.append(rot_err_deg(R_e_t, R_gt_t))
        rr_list.append(rot_err_deg(R_r_t, R_gt_t))
        te_list.append(np.linalg.norm(t_e_t - t_gt_t))
        tr_list.append(np.linalg.norm(t_r_t - t_gt_t))
        errs_r_t = EPnPSolver._reprojection_errors(pts3d_t, pts2d_t, R_r_t, t_r_t, K)
        ree_list.append(errs_t.mean())
        rer_list.append(errs_r_t.mean())

    rot_errs_e.append(np.median(re_list) if re_list else np.nan)
    rot_errs_r.append(np.median(rr_list) if rr_list else np.nan)
    t_errs_e.append(np.median(te_list) if te_list else np.nan)
    t_errs_r.append(np.median(tr_list) if tr_list else np.nan)
    repr_errs_e.append(np.median(ree_list) if ree_list else np.nan)
    repr_errs_r.append(np.median(rer_list) if rer_list else np.nan)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].plot(noise_levels, rot_errs_e, 'o-', color=ORANGE, lw=1.5, label='EPnP')
axes[0].plot(noise_levels, rot_errs_r, 's--', color=BLUE, lw=1.5, label='EPnP+GN')
axes[0].set_xlabel('Noise σ [px]')
axes[0].set_ylabel('Rotation error [deg]')
axes[0].set_title('Rotation Error vs Noise')
axes[0].legend()

axes[1].plot(noise_levels, t_errs_e, 'o-', color=ORANGE, lw=1.5, label='EPnP')
axes[1].plot(noise_levels, t_errs_r, 's--', color=BLUE, lw=1.5, label='EPnP+GN')
axes[1].set_xlabel('Noise σ [px]')
axes[1].set_ylabel('Translation error [m]')
axes[1].set_title('Translation Error vs Noise')
axes[1].legend()

axes[2].plot(noise_levels, repr_errs_e, 'o-', color=ORANGE, lw=1.5, label='EPnP')
axes[2].plot(noise_levels, repr_errs_r, 's--', color=BLUE, lw=1.5, label='EPnP+GN')
axes[2].set_xlabel('Noise σ [px]')
axes[2].set_ylabel('Mean repr. error [px]')
axes[2].set_title('Reprojection Error vs Noise')
axes[2].legend()

plt.suptitle('EPnP vs EPnP+GN Refinement (N=20 pts, median over 20 trials)',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/fig16_error_vs_noise.png', dpi=150, bbox_inches='tight')
plt.close()
print("  -> outputs/fig16_error_vs_noise.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 17 — RANSAC convergence (inliers vs iteration)
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Fig 17: RANSAC convergence...")

pts3d_r, pts2d_r, R_gtr, t_gtr = synth_data(N=40, noise=1.5,
                                              outlier_frac=0.5, seed=11)

from pose.ransac import RANSACSolver
from pose.epnp import EPnPSolver as _EPnP

# Manually trace RANSAC to get inlier count per iteration
rng = np.random.default_rng(0)
epnp = _EPnP()
best_counts = []
best_so_far = 0

for it in range(150):
    idx = rng.choice(len(pts3d_r), 4, replace=False)
    R_h, t_h, errs_h, ok = epnp.solve(pts3d_r[idx], pts2d_r[idx], K)
    if not ok or R_h is None:
        best_counts.append(best_so_far)
        continue
    all_errs = epnp._reprojection_errors(pts3d_r, pts2d_r, R_h, t_h, K)
    count = (all_errs < 3.0).sum()
    if count > best_so_far:
        best_so_far = count
    best_counts.append(best_so_far)

fig, ax = plt.subplots(figsize=(9, 4))
ax.step(range(len(best_counts)), best_counts, color=BLUE, lw=1.5,
        where='post', label='Best inlier count so far')
ax.axhline(20, color=GREEN, ls='--', lw=1.2,
           label='True inlier count (50% contamination)')
ax.set_xlabel('RANSAC iteration')
ax.set_ylabel('Best inlier count')
ax.set_title('RANSAC Convergence  |  N=40, 50% outliers, σ=1.5 px, threshold=3 px')
ax.legend()
plt.tight_layout()
plt.savefig('outputs/fig17_ransac_convergence.png', dpi=150, bbox_inches='tight')
plt.close()
print("  -> outputs/fig17_ransac_convergence.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 18 — Reprojection error CDF
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Fig 18: Reprojection error CDF...")

pts3d_c, pts2d_c, R_gtc, t_gtc = synth_data(N=50, noise=2.0, seed=21)
R_ec, t_ec, errs_ec, _ = solve_epnp(pts3d_c, pts2d_c, K)
R_rc, t_rc, _ = refine_pose(R_ec, t_ec, pts3d_c, pts2d_c, K)
errs_rc = EPnPSolver._reprojection_errors(pts3d_c, pts2d_c, R_rc, t_rc, K)

fig, ax = plt.subplots(figsize=(8, 5))
for errs, label, col, ls in [
    (errs_ec, 'EPnP', ORANGE, '-'),
    (errs_rc, 'EPnP + GN Refined', BLUE,   '--'),
]:
    sorted_e = np.sort(errs)
    cdf = np.arange(1, len(sorted_e)+1) / len(sorted_e)
    ax.plot(sorted_e, cdf, color=col, lw=2, ls=ls, label=label)

ax.axvline(2.0, color='gray', ls=':', lw=1.2, label='σ=2 px noise')
ax.set_xlabel('Reprojection error [px]')
ax.set_ylabel('CDF')
ax.set_title('Reprojection Error CDF  |  N=50 pts, σ=2 px noise')
ax.legend()
ax.set_xlim(0, None)
plt.tight_layout()
plt.savefig('outputs/fig18_repr_error_cdf.png', dpi=150, bbox_inches='tight')
plt.close()
print("  -> outputs/fig18_repr_error_cdf.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 19 — Full Ariane keypoint pose estimation
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Fig 19: Ariane keypoint pose estimation...")

model = ariane_model()
kp3d  = model.keypoint_array
eye   = np.array([18., 12., 22.])
R_gt19, t_gt19 = look_at_rotation(eye, np.zeros(3))

# Render the scene
cfg_rend = RendererConfig(draw_faces=True, draw_edges=False, draw_keypoints=False,
                           face_color=(160,185,210), noise_sigma=1.5,
                           star_density=0.0003, bg_color=(4,4,10))
rend = DebrisRenderer(CAM, cfg_rend)
R_body = random_rotation(7)
img19, _ = rend.render(model.transform(R_body), R_gt19, t_gt19, seed=19)

# Ground-truth 2D projections (after body rotation)
kp3d_rot = (R_body @ kp3d.T).T
kp2d_gt  = project(kp3d_rot, R_gt19, t_gt19, K)

# Add noise
rng19 = np.random.default_rng(19)
kp2d_noisy = kp2d_gt + rng19.normal(0, 2.0, kp2d_gt.shape)

# Clip to image
in_view = ((kp2d_gt[:,0] > 0) & (kp2d_gt[:,0] < CAM.width) &
           (kp2d_gt[:,1] > 0) & (kp2d_gt[:,1] < CAM.height))

kp3d_vis = kp3d_rot[in_view]
kp2d_vis = kp2d_noisy[in_view]

R_pnp, t_pnp, mask19, meta19 = solve_pnp_ransac(
    kp3d_vis, kp2d_vis, K, threshold_px=5.0, max_iter=300, seed=0)
R_ref19, t_ref19, _ = refine_pose(
    R_pnp, t_pnp, kp3d_vis[mask19], kp2d_vis[mask19], K)

kp2d_proj = project(kp3d_vis, R_ref19, t_ref19, K)
errs_final = EPnPSolver._reprojection_errors(kp3d_vis, kp2d_vis, R_ref19, t_ref19, K)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: rendered image + keypoints
ax = axes[0]
ax.imshow(img19)
ax.scatter(kp2d_vis[:, 0], kp2d_vis[:, 1], c='lime',
           s=40, zorder=3, label=f'Observed kpts ({len(kp2d_vis)})')
ax.scatter(kp2d_proj[:, 0], kp2d_proj[:, 1], c='red',
           s=40, marker='+', linewidths=2, zorder=4,
           label=f'Projected (refined, mean={errs_final.mean():.2f} px)')
for obs, proj in zip(kp2d_vis, kp2d_proj):
    ax.plot([obs[0], proj[0]], [obs[1], proj[1]], 'r-', lw=0.5, alpha=0.6)
ax.set_title('Ariane — RANSAC+EPnP+GN Pose on Synthetic Image')
ax.legend(fontsize=8, loc='lower right')
ax.axis('off')

# Right: error bar per keypoint
ax2 = axes[1]
sorted_idx = np.argsort(errs_final)
ax2.bar(range(len(errs_final)), errs_final[sorted_idx],
        color=BLUE, alpha=0.8, edgecolor='none')
ax2.axhline(5.0, color=RED, ls='--', lw=1.2, label='RANSAC threshold (5 px)')
ax2.axhline(errs_final.mean(), color=GREEN, ls='-', lw=1.5,
            label=f'Mean = {errs_final.mean():.2f} px')
ax2.set_xlabel('Keypoint index (sorted by error)')
ax2.set_ylabel('Reprojection error [px]')
ax2.set_title(f'Per-Keypoint Reprojection Error\n'
              f'N={len(kp2d_vis)} kpts  |  inliers={meta19["n_inliers"]}  |  '
              f'noise σ=2 px')
ax2.legend(fontsize=9)

plt.suptitle('Phase 4 — Full Ariane Pose Estimation Pipeline', fontsize=12,
             fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/fig19_ariane_pose.png', dpi=150, bbox_inches='tight')
plt.close()
print("  -> outputs/fig19_ariane_pose.png")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 55)
print("Phase 4 Validation Summary")
print("=" * 55)
pts3d_s, pts2d_s, _, _ = synth_data(N=20, noise=0.0)
R_s, t_s, errs_s, _ = solve_epnp(pts3d_s, pts2d_s, K)
R_rs, t_rs, _ = refine_pose(R_s, t_s, pts3d_s, pts2d_s, K)
errs_rs = EPnPSolver._reprojection_errors(pts3d_s, pts2d_s, R_rs, t_rs, K)
print(f"  EPnP noiseless (N=20) mean repr err : {errs_s.mean():.2e} px")
print(f"  GN refined noiseless mean repr err  : {errs_rs.mean():.2e} px")
print(f"  RANSAC on Ariane kpts ({len(kp2d_vis)} pts):")
print(f"    Inliers        : {meta19['n_inliers']} / {len(kp2d_vis)}")
print(f"    Mean repr err  : {errs_final.mean():.3f} px  (σ=2 px noise)")
print(f"    RANSAC iters   : {meta19['n_iters']}")
print()
print("Figures saved to outputs/  (fig15 through fig19)")
print("Run tests: python -m pytest tests/test_phase4.py -v")

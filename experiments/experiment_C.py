"""
Experiment C — RANSAC Robustness vs Outlier Fraction
=====================================================
Injects a controlled fraction of outlier keypoints (random pixel coords),
then compares bare EPnP against RANSAC+EPnP.

Fixed pixel noise: σ = 1.0 px on inlier points.
Outlier fractions: 0%, 5%, 10%, 20%, 30%.
Trials: 200 per condition.

A trial is a *failure* if the solver either reports failure
or the recovered rotation error exceeds 15°.

Metrics recorded (over successful trials only):
  - Translation RMSE  [cm]
  - Rotation RMSE     [deg]
Failure rate [%] is tracked separately for all trials.

Outputs
-------
outputs/expC_fig3_ransac_robustness.png

Run from the project root:
    python experiments/experiment_C.py
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from vision import rendezvous_camera, ariane_model, look_at_rotation
from pose import solve_epnp
from pose.ransac import solve_pnp_ransac

OUT = Path('outputs')
OUT.mkdir(exist_ok=True)

# ── style tokens (light / professional) ───────────────────────────────────────
BG    = '#FFFFFF'
PANEL = '#FAFAFA'
TEXT  = '#111111'
MUTED = '#555555'
GRID  = '#DDDDDD'
C1    = '#111111'   # EPnP bare
C2    = '#555555'   # RANSAC + EPnP
ALPHA = 0.13

def style_ax(ax, ylabel, title, xlabel='Outlier fraction  [%]'):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(MUTED)
        spine.set_linewidth(0.8)
    ax.tick_params(colors=TEXT, labelsize=9.5, direction='in')
    ax.set_ylabel(ylabel, fontsize=10.5, color=TEXT)
    ax.set_xlabel(xlabel, fontsize=10.5, color=TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.set_title(title, color=TEXT, fontsize=11, pad=6)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=1.0)
    ax.set_axisbelow(True)

# ── setup ─────────────────────────────────────────────────────────────────────
RNG         = np.random.default_rng(42)
N_TRIALS    = 200
SIGMA_PX    = 1.0          # inlier pixel noise [px]
OUTLIER_FRAC = [0.0, 0.05, 0.10, 0.20, 0.30]
FAIL_THRESH_DEG = 15.0     # rotation error > this → failure

# ── scene ─────────────────────────────────────────────────────────────────────
cam   = rendezvous_camera()
model = ariane_model()
K     = cam.K
kp3d  = model.keypoint_array
N_KP  = len(kp3d)

W = int(2 * K[0, 2])   # image width  ≈ 2 * cx
H = int(2 * K[1, 2])   # image height ≈ 2 * cy

eye = np.array([0., 0., 20.])
R_gt, t_gt = look_at_rotation(eye, target=np.zeros(3), up=np.array([0., 1., 0.]))

P_cam   = (R_gt @ kp3d.T).T + t_gt
u_true  = K[0,0] * P_cam[:,0] / P_cam[:,2] + K[0,2]
v_true  = K[1,1] * P_cam[:,1] / P_cam[:,2] + K[1,2]
kp2d_gt = np.stack([u_true, v_true], axis=1)

print(f"Keypoints: {N_KP}   Image: {W}×{H}")
print(f"{'frac':>6}  {'EPnP fail%':>10}  {'EPnP t[cm]':>11}  {'EPnP r[°]':>10}  "
      f"{'RANSAC fail%':>12}  {'RANSAC t[cm]':>13}  {'RANSAC r[°]':>11}")
print('-' * 90)

def t_err(t_est): return np.linalg.norm(t_est - t_gt)
def r_err_deg(R_est):
    c = np.clip((np.trace(R_est @ R_gt.T) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(c))

ep_tm_all, ep_ts_all, ep_rm_all, ep_rs_all, ep_fail_all   = [], [], [], [], []
ra_tm_all, ra_ts_all, ra_rm_all, ra_rs_all, ra_fail_all   = [], [], [], [], []

for frac in OUTLIER_FRAC:
    n_out = int(round(frac * N_KP))   # number of outlier points

    ep_t, ep_r, ep_fails = [], [], 0
    ra_t, ra_r, ra_fails = [], [], 0

    for trial in range(N_TRIALS):
        # ── build noisy + outlier-corrupted observations ──────────────────────
        noise  = RNG.normal(0.0, SIGMA_PX, kp2d_gt.shape)
        kp2d_n = kp2d_gt + noise

        if n_out > 0:
            out_idx = RNG.choice(N_KP, n_out, replace=False)
            # random pixel coords anywhere in image
            kp2d_n[out_idx, 0] = RNG.uniform(0, W, n_out)
            kp2d_n[out_idx, 1] = RNG.uniform(0, H, n_out)

        # ── bare EPnP ─────────────────────────────────────────────────────────
        R_e, t_e, _, ok_e = solve_epnp(kp3d, kp2d_n, K)
        if ok_e and R_e is not None:
            re = r_err_deg(R_e)
            if re > FAIL_THRESH_DEG:
                ep_fails += 1
            else:
                ep_t.append(t_err(t_e))
                ep_r.append(re)
        else:
            ep_fails += 1

        # ── RANSAC + EPnP ─────────────────────────────────────────────────────
        R_r, t_r, _, meta = solve_pnp_ransac(kp3d, kp2d_n, K,
                                              threshold_px=2.0,
                                              max_iter=500,
                                              seed=int(RNG.integers(1<<31)))
        if meta['success'] and R_r is not None:
            rr = r_err_deg(R_r)
            if rr > FAIL_THRESH_DEG:
                ra_fails += 1
            else:
                ra_t.append(t_err(t_r))
                ra_r.append(rr)
        else:
            ra_fails += 1

    def rmse(v): return np.sqrt(np.mean(np.array(v)**2)) if v else np.nan
    def std(v):  return np.std(np.array(v))               if v else np.nan

    ep_tm_all.append(rmse(ep_t)*100); ep_ts_all.append(std(ep_t)*100)
    ep_rm_all.append(rmse(ep_r));     ep_rs_all.append(std(ep_r))
    ep_fail_all.append(100.0 * ep_fails / N_TRIALS)

    ra_tm_all.append(rmse(ra_t)*100); ra_ts_all.append(std(ra_t)*100)
    ra_rm_all.append(rmse(ra_r));     ra_rs_all.append(std(ra_r))
    ra_fail_all.append(100.0 * ra_fails / N_TRIALS)

    print(f"{frac:>6.0%}  {ep_fail_all[-1]:>10.1f}  {ep_tm_all[-1]:>11.3f}  "
          f"{ep_rm_all[-1]:>10.3f}  {ra_fail_all[-1]:>12.1f}  "
          f"{ra_tm_all[-1]:>13.3f}  {ra_rm_all[-1]:>11.3f}")

fracs  = np.array(OUTLIER_FRAC) * 100   # percent for x-axis
ep_tm  = np.array(ep_tm_all); ep_ts = np.array(ep_ts_all)
ep_rm  = np.array(ep_rm_all); ep_rs = np.array(ep_rs_all)
ra_tm  = np.array(ra_tm_all); ra_ts = np.array(ra_ts_all)
ra_rm  = np.array(ra_rm_all); ra_rs = np.array(ra_rs_all)
ep_fail = np.array(ep_fail_all)
ra_fail = np.array(ra_fail_all)

# ── figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(9, 9), facecolor=BG)
fig.subplots_adjust(hspace=0.52, left=0.13, right=0.96, top=0.92, bottom=0.08)

def dual(ax, x, m1, s1, m2, s2, lab1, lab2):
    l1, = ax.plot(x, m1, color=C1, linewidth=1.4, linestyle='--',
                  marker='o', markersize=4.5, label=lab1, alpha=0.65)
    ax.fill_between(x, m1-s1, m1+s1, color=C1, alpha=ALPHA)
    l2, = ax.plot(x, m2, color=C2, linewidth=1.9,
                  marker='s', markersize=4.5, label=lab2)
    ax.fill_between(x, m2-s2, m2+s2, color=C2, alpha=ALPHA)
    # ±1σ proxy patch for legend
    band = mpatches.Patch(facecolor=MUTED, alpha=0.30, label='±1σ band')
    ax.legend(handles=[l1, l2, band],
              fontsize=8.5, facecolor=PANEL, edgecolor=MUTED,
              labelcolor=TEXT, loc='upper left')

def xlim_ticks(ax):
    ax.set_xlim(fracs[0] - 1.5, fracs[-1] + 1.5)
    ax.set_xticks(fracs)

# panel 1 — translation RMSE
dual(axes[0], fracs, ep_tm, ep_ts, ra_tm, ra_ts,
     'EPnP (bare)', 'RANSAC + EPnP')
style_ax(axes[0], 'Translation RMSE  [cm]', 'Translation error  (successful trials)')
xlim_ticks(axes[0])

# panel 2 — rotation RMSE
dual(axes[1], fracs, ep_rm, ep_rs, ra_rm, ra_rs,
     'EPnP (bare)', 'RANSAC + EPnP')
style_ax(axes[1], 'Rotation RMSE  [deg]', 'Rotation error  (successful trials)')
xlim_ticks(axes[1])

# panel 3 — failure rate (grouped bars, no shaded bands needed)
ax2 = axes[2]
ax2.set_facecolor(PANEL)
for spine in ax2.spines.values():
    spine.set_edgecolor(MUTED)
    spine.set_linewidth(0.8)
ax2.tick_params(colors=TEXT, labelsize=9.5, direction='in')
ax2.yaxis.label.set_color(TEXT)
ax2.xaxis.label.set_color(TEXT)
ax2.set_title('Failure rate  (rotation error > 15°)', color=TEXT, fontsize=11, pad=6)
ax2.set_ylabel('Failure rate  [%]', fontsize=10.5)
ax2.set_xlabel('Outlier fraction  [%]', fontsize=10.5, color=TEXT)
ax2.grid(True, color=GRID, linewidth=0.7, alpha=1.0, axis='y')
ax2.set_axisbelow(True)
ax2.set_ylim(0, 108)

bar_w = 2.0
ax2.bar(fracs - bar_w/2, ep_fail, width=bar_w, color=C1,
        alpha=0.70, label='EPnP (bare)', zorder=3)
ax2.bar(fracs + bar_w/2, ra_fail, width=bar_w, color=C2,
        alpha=0.70, label='RANSAC + EPnP', zorder=3)
ax2.set_xlim(fracs[0] - 1.5, fracs[-1] + 1.5)
ax2.set_xticks(fracs)
ax2.legend(fontsize=8.5, facecolor=PANEL, edgecolor=MUTED,
           labelcolor=TEXT, loc='upper left')

fig.suptitle('RANSAC Robustness vs Outlier Fraction  (N = 200 trials, σ = 1 px)',
             color=TEXT, fontsize=12, y=0.97)

out_path = OUT / 'expC_fig3_ransac_robustness.png'
fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor=BG)
plt.close(fig)
print(f"\nSaved → {out_path}")

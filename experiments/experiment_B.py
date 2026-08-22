"""
Experiment B — EPnP Pose Estimation vs Pixel Noise
====================================================
Sweeps pixel noise σ = 0.1 … 5 px (σ=0 excluded — GN diverges from
exact-zero residual due to J^T J singularity; EPnP alone is correct at σ=0).

For each σ, runs 200 Monte Carlo trials and records:
  - Raw EPnP:         translation RMSE, rotation RMSE, reprojection error
  - EPnP + GN refine: same metrics

Shows both curves with ±1σ shaded bands on 3-panel figure.

Outputs
-------
outputs/expB_fig2_pose_vs_noise.png

Run from the project root:
    python experiments/experiment_B.py
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
from pose.refine import refine_pose

OUT      = Path('outputs')
OUT.mkdir(exist_ok=True)

# ── style tokens (light / professional) ───────────────────────────────────────
BG    = '#FFFFFF'
PANEL = '#FAFAFA'
TEXT  = '#111111'
MUTED = '#555555'
GRID  = '#DDDDDD'
C1    = '#111111'   # EPnP raw (solid)
C2    = '#555555'   # EPnP + GN (dashed)
ALPHA_BAND = 0.13

def style_ax(ax, ylabel, title, xlabel='Pixel noise  σ  [px]'):
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
RNG      = np.random.default_rng(42)
N_TRIALS = 200
# σ=0 excluded: GN diverges from exact-zero residual (J^T J singular).
# EPnP is correct at σ=0 (see experiment_B_diag.py).
SIGMAS   = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

# ── scene ─────────────────────────────────────────────────────────────────────
cam   = rendezvous_camera()
model = ariane_model()
K     = cam.K
kp3d  = model.keypoint_array

eye = np.array([0., 0., 20.])
R_gt, t_gt = look_at_rotation(eye, target=np.zeros(3), up=np.array([0., 1., 0.]))

P_cam   = (R_gt @ kp3d.T).T + t_gt
u_true  = K[0,0] * P_cam[:,0] / P_cam[:,2] + K[0,2]
v_true  = K[1,1] * P_cam[:,1] / P_cam[:,2] + K[1,2]
kp2d_gt = np.stack([u_true, v_true], axis=1)

def t_err(t_est):
    return np.linalg.norm(t_est - t_gt)

def r_err_deg(R_est):
    c = np.clip((np.trace(R_est @ R_gt.T) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(c))

def reproj_err(R, t):
    Pc = (R @ kp3d.T).T + t
    u  = K[0,0]*Pc[:,0]/Pc[:,2] + K[0,2]
    v  = K[1,1]*Pc[:,1]/Pc[:,2] + K[1,2]
    return np.sqrt(((np.stack([u,v],1) - kp2d_gt)**2).sum(1)).mean()

# ── Monte Carlo ───────────────────────────────────────────────────────────────
results = {
    'epnp': {'t': [], 'r': [], 'rp': []},
    'gn':   {'t': [], 'r': [], 'rp': []},
}

print(f"{'sigma':>8}  {'EPnP t[cm]':>12}  {'EPnP r[°]':>10}  "
      f"{'GN t[cm]':>10}  {'GN r[°]':>8}  ok/N")
print('-' * 70)

for sigma in SIGMAS:
    ep_t, ep_r, ep_rp = [], [], []
    gn_t, gn_r, gn_rp = [], [], []
    ok = 0

    for _ in range(N_TRIALS):
        noise  = RNG.normal(0.0, sigma, kp2d_gt.shape)
        kp2d_n = kp2d_gt + noise

        R_e, t_e, _, solved = solve_epnp(kp3d, kp2d_n, K)
        if not solved:
            continue
        ok += 1

        ep_t.append(t_err(t_e))
        ep_r.append(r_err_deg(R_e))
        ep_rp.append(reproj_err(R_e, t_e))

        R_gn, t_gn, _ = refine_pose(R_e, t_e, kp3d, kp2d_n, K)
        gn_t.append(t_err(t_gn))
        gn_r.append(r_err_deg(R_gn))
        gn_rp.append(reproj_err(R_gn, t_gn))

    results['epnp']['t'].append(ep_t)
    results['epnp']['r'].append(ep_r)
    results['epnp']['rp'].append(ep_rp)
    results['gn']['t'].append(gn_t)
    results['gn']['r'].append(gn_r)
    results['gn']['rp'].append(gn_rp)

    def rmse(v): return np.sqrt(np.mean(np.array(v)**2)) if v else np.nan
    print(f"{sigma:>8.2f}  {rmse(ep_t)*100:>12.3f}  {np.mean(ep_r):>10.3f}  "
          f"{rmse(gn_t)*100:>10.3f}  {np.mean(gn_r):>8.3f}  {ok}/{N_TRIALS}")

# ── aggregated stats ──────────────────────────────────────────────────────────
sigmas = np.array(SIGMAS)

def stats(vals_list, scale=1.0):
    """Per-sigma mean and std from list-of-lists."""
    means, stds = [], []
    for v in vals_list:
        a = np.array(v) * scale
        means.append(a.mean() if len(a) else np.nan)
        stds.append(a.std()  if len(a) else np.nan)
    return np.array(means), np.array(stds)

ep_tm, ep_ts = stats(results['epnp']['t'],  scale=100)    # cm
ep_rm, ep_rs = stats(results['epnp']['r'])                # deg
ep_rp_m, ep_rp_s = stats(results['epnp']['rp'])           # px

gn_tm, gn_ts = stats(results['gn']['t'],   scale=100)
gn_rm, gn_rs = stats(results['gn']['r'])
gn_rp_m, gn_rp_s = stats(results['gn']['rp'])

# ── figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(9, 9), facecolor=BG)
fig.subplots_adjust(hspace=0.50, left=0.13, right=0.96, top=0.92, bottom=0.08)

def twin_lines(ax, x, m1, s1, m2, s2, lab1, lab2):
    # raw EPnP (dashed, lighter)
    l1, = ax.plot(x, m1, color=C1, linewidth=1.3, linestyle='--',
                  marker='o', markersize=4, label=lab1, alpha=0.65)
    ax.fill_between(x, m1-s1, m1+s1, color=C1, alpha=ALPHA_BAND)
    # GN refined (solid)
    l2, = ax.plot(x, m2, color=C2, linewidth=1.8,
                  marker='s', markersize=4.5, label=lab2)
    ax.fill_between(x, m2-s2, m2+s2, color=C2, alpha=ALPHA_BAND)
    # ±1σ proxy patch for legend
    band = mpatches.Patch(facecolor=MUTED, alpha=0.30, label='±1σ band')
    ax.legend(handles=[l1, l2, band],
              fontsize=8.5, facecolor=PANEL, edgecolor=MUTED,
              labelcolor=TEXT, loc='upper left')

# panel 1 — translation
twin_lines(axes[0], sigmas,
           ep_tm, ep_ts, gn_tm, gn_ts,
           'EPnP (raw)', 'EPnP + GN')
style_ax(axes[0], 'Translation RMSE  [cm]', 'Translation error')
axes[0].set_xlim(sigmas[0] - 0.05, sigmas[-1] + 0.1)
axes[0].set_xticks(sigmas)

# panel 2 — rotation
twin_lines(axes[1], sigmas,
           ep_rm, ep_rs, gn_rm, gn_rs,
           'EPnP (raw)', 'EPnP + GN')
style_ax(axes[1], 'Rotation RMSE  [deg]', 'Rotation error')
axes[1].set_xlim(sigmas[0] - 0.05, sigmas[-1] + 0.1)
axes[1].set_xticks(sigmas)

# panel 3 — reprojection
twin_lines(axes[2], sigmas,
           ep_rp_m, ep_rp_s, gn_rp_m, gn_rp_s,
           'EPnP (raw)', 'EPnP + GN')
style_ax(axes[2], 'Reprojection error  [px]', 'Reprojection error')
axes[2].set_xlim(sigmas[0] - 0.05, sigmas[-1] + 0.1)
axes[2].set_xticks(sigmas)

fig.suptitle('Pose Estimation Accuracy vs Pixel Noise  (N = 200 trials per σ)',
             color=TEXT, fontsize=12, y=0.97)

out_path = OUT / 'expB_fig2_pose_vs_noise.png'
fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor=BG)
plt.close(fig)
print(f"\nSaved → {out_path}")

"""
Experiment F — Monte Carlo Robustness & Noise Ablation
=======================================================
Evaluates the complete vision-satellite-rendezvous pipeline under statistical
variation and ablation of key system components / noise parameters.

Study 1 — Monte Carlo (N_MC trials × 3 configurations)
  Configurations compared:
    A. Perfect state  : ideal state knowledge, zero noise, zero init error
    B. EKF only       : noisy direct measurements → EKF → LQR  (no vision)
    C. Full pipeline  : EPnP vision → EKF → LQR  (nominal σ_px = 1.5 px)
  Metrics per trial  : final range [m], total Δv [m/s], docking success.

Study 2 — Pixel-noise ablation (N_MC_NOISE trials × 7 noise levels)
  Full pipeline run at σ_px ∈ {0.3, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0} px.
  Reports mean ± 1σ final range and docking success rate vs noise level.

Setup
-----
  Initial position : [30, 3, -1.5] m  (radial approach)
  Initial velocity : [-0.08, 0, 0] m/s
  Init pos error   : [2.5, -0.8, 0.5] m  (configs B & C)
  u_max            : 0.3 m/s²
  Steps per trial  : 250  (default dt)

Outputs
-------
  outputs/expF_fig8_montecarlo_boxplot.png  — violin + bar plots (Study 1)
  outputs/expF_fig9_ablation_noise.png      — range & success vs noise (Study 2)

Run from the project root:
    python experiments/experiment_F.py
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

from simulation import default_config, run_simulation

OUT = Path('outputs')
OUT.mkdir(exist_ok=True)

# ── style tokens ──────────────────────────────────────────────────────────────
BG         = '#FFFFFF'
PANEL      = '#FAFAFA'
TEXT       = '#111111'
MUTED      = '#555555'
GRID       = '#DDDDDD'
C1         = '#111111'   # config A / curve
C2         = '#555555'   # config B
C3         = '#999999'   # config C
ALPHA_BAND = 0.13


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
N_MC       = 25           # Monte Carlo trials per configuration
N_MC_NOISE = 20           # trials per noise ablation level
N_STEPS    = 250          # steps per trial (matches phase7 demo)
U_MAX      = 0.3          # [m/s²]
POS_W      = 15.0
VEL_W      = 1.5
POS_STD    = 0.5          # direct position measurement std [m]
PIX_STD    = 1.5          # nominal pixel noise std [px]

R0     = np.array([30.,  3., -1.5])    # initial position [m]
V0     = np.array([-0.08, 0., 0.])     # initial velocity [m/s]
W0     = np.array([0.02, 0.05, 0.01])  # initial angular rate [rad/s]
R0_ERR = np.array([2.5, -0.8, 0.5])   # initial position error [m]

NOISE_LEVELS = [0.3, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0]  # [px]

# ── three configurations ──────────────────────────────────────────────────────
#   key  : display label for plot x-axis
#   value: keyword overrides for default_config()
CONFIGS = {
    'Perfect\nstate': dict(
        use_vision=False, use_ekf=False,
        pos_meas_std=0.0, pixel_noise_std=0.0,
        r0_err=np.zeros(3),
    ),
    'EKF only\n(direct)': dict(
        use_vision=False, use_ekf=True,
        pos_meas_std=POS_STD, pixel_noise_std=0.0,
        r0_err=R0_ERR,
    ),
    'Full pipeline\n(vision+EKF)': dict(
        use_vision=True, use_ekf=True,
        pos_meas_std=POS_STD, pixel_noise_std=PIX_STD,
        r0_err=R0_ERR,
    ),
}

# ── Study 1: Monte Carlo ──────────────────────────────────────────────────────
print("=" * 65)
print(f"Study 1 — Monte Carlo  (N={N_MC} trials × {len(CONFIGS)} configs)")
print("=" * 65)

mc_results = {}   # label → {range, dv, success}

for label, cfg_kw in CONFIGS.items():
    short = label.replace('\n', ' ')
    print(f"\n  [{short}]")
    ranges, dvs, successes = [], [], []
    t0 = time.perf_counter()

    for seed in range(N_MC):
        cfg = default_config(
            n_steps=N_STEPS,
            u_max=U_MAX,
            pos_weight=POS_W,
            vel_weight=VEL_W,
            r0=R0, v0=V0, w0=W0,
            **cfg_kw,
        )
        res = run_simulation(cfg, rng_seed=seed)
        ranges.append(float(res.range_m[-1]))
        dvs.append(float(res.delta_v))
        successes.append(res.dock_step is not None)
        if (seed + 1) % 5 == 0:
            print(f"    seed {seed+1:3d}/{N_MC}  "
                  f"range = {res.range_m[-1]:.3f} m  "
                  f"dv = {res.delta_v:.2f} m/s")

    elapsed = time.perf_counter() - t0
    sr = np.mean(successes) * 100
    print(f"    ✓ {elapsed:.1f}s  |  "
          f"mean range = {np.mean(ranges):.2f} m  |  "
          f"success = {sr:.0f}%")

    mc_results[label] = {
        'range'  : np.array(ranges),
        'dv'     : np.array(dvs),
        'success': np.array(successes, dtype=float),
    }

# ── Study 2: Pixel-noise ablation ────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"Study 2 — Pixel-noise ablation  "
      f"(N={N_MC_NOISE} trials × {len(NOISE_LEVELS)} levels, full pipeline)")
print("=" * 65)

abl_means   = []
abl_stds    = []
abl_success = []

for px_std in NOISE_LEVELS:
    ranges = []
    for seed in range(N_MC_NOISE):
        cfg = default_config(
            n_steps=N_STEPS,
            u_max=U_MAX,
            pos_weight=POS_W,
            vel_weight=VEL_W,
            r0=R0, v0=V0, w0=W0,
            use_vision=True, use_ekf=True,
            pos_meas_std=POS_STD,
            pixel_noise_std=px_std,
            r0_err=R0_ERR,
        )
        res = run_simulation(cfg, rng_seed=seed)
        ranges.append(float(res.range_m[-1]))

    m, s = np.mean(ranges), np.std(ranges)
    sr   = np.mean([r < 2.0 for r in ranges]) * 100   # fallback docking check
    abl_means.append(m)
    abl_stds.append(s)
    abl_success.append(sr)
    print(f"  σ_px = {px_std:.1f} px  →  "
          f"range = {m:.2f} ± {s:.2f} m  |  success ≈ {sr:.0f}%")

abl_means   = np.array(abl_means)
abl_stds    = np.array(abl_stds)
abl_success = np.array(abl_success)
noise_x     = np.array(NOISE_LEVELS)

# ── summary table ─────────────────────────────────────────────────────────────
print()
print("=" * 65)
print(f"{'Configuration':<28} {'Range [m]':>12} {'Δv [m/s]':>10} {'Success %':>10}")
print("-" * 65)
for label, r in mc_results.items():
    short = label.replace('\n', ' ')
    print(f"{short:<28} {np.mean(r['range']):>10.2f}  "
          f"{np.mean(r['dv']):>10.2f}  "
          f"{r['success'].mean()*100:>10.0f}")
print("=" * 65)

# ── Fig 8 — Monte Carlo violin + success bar chart ────────────────────────────
cfg_labels = list(CONFIGS.keys())
cfg_colors = [C1, C2, C3]
positions  = [1, 2, 3]

fig, axes = plt.subplots(1, 3, figsize=(13, 5), facecolor=BG)
fig.subplots_adjust(wspace=0.42, left=0.08, right=0.97,
                    top=0.88, bottom=0.14)

# ── panel A: final range violin ───────────────────────────────────────────────
ax = axes[0]
style_ax(ax, 'Final range  [m]', f'Final Range  (N = {N_MC})')
data_range = [mc_results[k]['range'] for k in cfg_labels]

# Pad any degenerate (constant) distributions to avoid violin crash
data_range_safe = []
for d in data_range:
    if np.ptp(d) < 1e-9:          # all values identical → add tiny jitter
        d = d + np.random.default_rng(0).normal(0, 1e-6, len(d))
    data_range_safe.append(d)

vp1 = ax.violinplot(data_range_safe, positions=positions,
                    showmedians=True, showextrema=True, widths=0.55)
for i, pc in enumerate(vp1['bodies']):
    pc.set_facecolor(cfg_colors[i])
    pc.set_edgecolor(TEXT)
    pc.set_alpha(0.55)
for part in ('cmedians', 'cmins', 'cmaxes', 'cbars'):
    vp1[part].set_edgecolor(TEXT)
    vp1[part].set_linewidth(1.2)
ax.set_xticks(positions)
ax.set_xticklabels(cfg_labels, fontsize=8.5, color=TEXT)

# ── panel B: total Δv violin ──────────────────────────────────────────────────
ax = axes[1]
style_ax(ax, 'Total Δv  [m/s]', f'Propellant Budget  (N = {N_MC})')
data_dv = [mc_results[k]['dv'] for k in cfg_labels]

data_dv_safe = []
for d in data_dv:
    if np.ptp(d) < 1e-9:
        d = d + np.random.default_rng(0).normal(0, 1e-6, len(d))
    data_dv_safe.append(d)

vp2 = ax.violinplot(data_dv_safe, positions=positions,
                    showmedians=True, showextrema=True, widths=0.55)
for i, pc in enumerate(vp2['bodies']):
    pc.set_facecolor(cfg_colors[i])
    pc.set_edgecolor(TEXT)
    pc.set_alpha(0.55)
for part in ('cmedians', 'cmins', 'cmaxes', 'cbars'):
    vp2[part].set_edgecolor(TEXT)
    vp2[part].set_linewidth(1.2)
ax.set_xticks(positions)
ax.set_xticklabels(cfg_labels, fontsize=8.5, color=TEXT)

# ── panel C: docking success rate bar ─────────────────────────────────────────
ax = axes[2]
style_ax(ax, 'Docking success rate  [%]', f'Success Rate  (N = {N_MC})')
sr_vals = [mc_results[k]['success'].mean() * 100 for k in cfg_labels]
bars = ax.bar(positions, sr_vals, width=0.50,
              color=cfg_colors, alpha=0.80, zorder=3,
              edgecolor=TEXT, linewidth=0.7)
for bar, v in zip(bars, sr_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5,
            f'{v:.0f}%', ha='center', va='bottom', fontsize=9.5, color=TEXT)
ax.set_xticks(positions)
ax.set_xticklabels(cfg_labels, fontsize=8.5, color=TEXT)
ax.set_ylim(0, 118)

fig.suptitle(
    f'Monte Carlo Robustness Study  ·  {N_STEPS} steps  ·  '
    f'u_max = {U_MAX} m/s²  ·  σ_pos = {POS_STD} m  ·  σ_px = {PIX_STD} px',
    color=TEXT, fontsize=10.5, y=0.97)

out1 = OUT / 'expF_fig8_montecarlo_boxplot.png'
fig.savefig(out1, dpi=180, bbox_inches='tight', facecolor=BG)
plt.close(fig)
print(f"\nSaved → {out1}")

# ── Fig 9 — Pixel-noise ablation ─────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(11, 5), facecolor=BG)
fig2.subplots_adjust(wspace=0.42, left=0.10, right=0.97,
                     top=0.88, bottom=0.13)

# ── panel A: final range vs noise ─────────────────────────────────────────────
ax = axes2[0]
style_ax(ax, 'Final range  [m]',
         'Final Range vs Pixel Noise',
         xlabel='Pixel noise  σ_px  [px]')
l1, = ax.plot(noise_x, abl_means, color=C1, linewidth=1.8,
              marker='o', markersize=5, label='Mean final range')
ax.fill_between(noise_x,
                abl_means - abl_stds,
                abl_means + abl_stds,
                color=C1, alpha=ALPHA_BAND)
band = mpatches.Patch(facecolor=MUTED, alpha=0.30, label='±1σ band')
ax.legend(handles=[l1, band], fontsize=8.5,
          facecolor=PANEL, edgecolor=MUTED, labelcolor=TEXT)
ax.set_xlim(noise_x[0] - 0.1, noise_x[-1] + 0.1)

# ── panel B: success rate vs noise ────────────────────────────────────────────
ax = axes2[1]
style_ax(ax, 'Docking success rate  [%]',
         'Success Rate vs Pixel Noise',
         xlabel='Pixel noise  σ_px  [px]')
l2, = ax.plot(noise_x, abl_success, color=C1, linewidth=1.8,
              marker='s', markersize=5, label='Success rate')
ax.set_xlim(noise_x[0] - 0.1, noise_x[-1] + 0.1)
ax.set_ylim(-5, 115)
ax.legend(handles=[l2], fontsize=8.5,
          facecolor=PANEL, edgecolor=MUTED, labelcolor=TEXT)

fig2.suptitle(
    f'Vision Pipeline Noise Ablation  ·  Full pipeline (vision+EKF)  ·  '
    f'N = {N_MC_NOISE} trials per level',
    color=TEXT, fontsize=10.5, y=0.97)

out2 = OUT / 'expF_fig9_ablation_noise.png'
fig2.savefig(out2, dpi=180, bbox_inches='tight', facecolor=BG)
plt.close(fig2)
print(f"Saved → {out2}")

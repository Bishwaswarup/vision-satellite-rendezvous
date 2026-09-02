"""
Experiment F — Dispersed Monte Carlo & Noise Ablation
=====================================================
Thin driver over :mod:`simulation.montecarlo`.  All sampling, seeding and
statistics live in that module so they can be unit-tested; this file only
chooses the campaign sizes, runs them, prints the tables and draws the
figures.

What changed from the earlier version of this experiment
--------------------------------------------------------
The previous script looped over ``range(N)`` and passed the *same* r0, v0, w0
and q0 to every trial, varying only the simulator's RNG seed.  The reported
"distribution" was therefore the spread of one measurement-noise realisation
about one trajectory, and the perfect-state configuration — which has no noise
at all — produced N bit-identical runs that had to be jittered by 1e-6 to keep
``violinplot`` from crashing.  The campaign now disperses initial range,
bearing, closing rate, tumble magnitude and axis, initial attitude (uniform on
SO(3)) and navigation initialisation error, with per-trial seed sequences so
any trial can be replayed on its own.

Study 1 — three configurations, paired trial-by-trial
    A. perfect state  : truth fed to the controller, no noise, no nav error
    B. EKF, direct    : noisy direct pose measurement -> MEKF -> LQR
    C. vision + EKF   : EPnP/RANSAC front end -> MEKF -> LQR

Study 2 — pixel-noise ablation, paired across levels
    sigma_px in {0.3, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0} px, identical initial
    conditions at every level, so the level-to-level change is the parameter
    effect and not a fresh IC draw.

Study 3 — availability vs instantaneous range
    Measurement availability pooled per simulation step and binned by the
    range the camera is actually at, which is the campaign-level counterpart
    of the closed-form fill criterion z_fill = f L / W.  Binning by *initial*
    range would say nothing: every trial that closes passes through the whole
    interval, so a per-trial dropout rate averages the easy far field with the
    hard terminal phase.

Outputs
-------
  outputs/expF_fig8_montecarlo.png       Study 1 (violins + success bars)
  outputs/expF_fig9_ablation_noise.png   Study 2 (range, success, dropout)
  outputs/expF_fig10_availability.png    Study 3 (dropout & success vs range)
  outputs/mc_<config>.csv                per-trial records, all studies
  outputs/mc_tables.tex                  booktabs tables for the paper

Run from the project root:
    python experiments/experiment_F.py [--trials N] [--noise-trials N] [--jobs J]
"""

import argparse
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path

from simulation.montecarlo import (
    MonteCarloConfig, Dispersion, latex_table, run_campaign, standard_configs,
    sweep,
)
from viz.style import (
    GREY, HATCHES, add_panel_label, apply_style, hatch, save_fig, series_kw,
    style_ax,
)

OUT = Path('outputs')
OUT.mkdir(exist_ok=True)

NOISE_LEVELS = [0.3, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0]   # [px]
Z_FILL_M = 6.25          # f L / W = 800 x 8 / 1024  [m]


# ── figure helpers ───────────────────────────────────────────────────────────

def _violin(ax, data, labels):
    """Monochrome violin: grey bodies, hatched, with a median bar."""
    safe = [np.asarray([v for v in d if np.isfinite(v)], dtype=float)
            for d in data]
    pos = np.arange(1, len(safe) + 1)
    vp = ax.violinplot(safe, positions=pos, widths=0.6,
                       showmedians=True, showextrema=True)
    for i, body in enumerate(vp['bodies']):
        body.set_facecolor(GREY['light'])
        body.set_edgecolor(GREY['ink'])
        body.set_linewidth(0.8)
        body.set_alpha(1.0)
        body.set_hatch(hatch(i))
    for part in ('cmedians', 'cmins', 'cmaxes', 'cbars'):
        if part in vp:
            vp[part].set_edgecolor(GREY['ink'])
            vp[part].set_linewidth(1.1)
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, fontsize=8.5)


def fig_study1(results, path):
    labels = [r.cfg.label.replace(', ', ',\n').replace(' + ', ' +\n')
              for r in results]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8))

    _violin(axes[0], [r.column('final_range_m') for r in results], labels)
    style_ax(axes[0], ylabel='Final range  [m]')
    add_panel_label(axes[0], '(a)')

    _violin(axes[1], [r.column('delta_v_mps') for r in results], labels)
    style_ax(axes[1], ylabel=r'Total $\Delta v$  [m/s]')
    add_panel_label(axes[1], '(b)')

    ax = axes[2]
    rates = [r.stats()['dock_rate'] for r in results]
    pos = np.arange(1, len(results) + 1)
    err = np.array([[d['rate'] - d['ci_low'] for d in rates],
                    [d['ci_high'] - d['rate'] for d in rates]]) * 100
    for i, d in enumerate(rates):
        ax.bar(pos[i], d['rate'] * 100, width=0.55, zorder=3,
               facecolor=GREY['light'], edgecolor=GREY['ink'],
               linewidth=0.8, hatch=hatch(i))
    ax.errorbar(pos, [d['rate'] * 100 for d in rates], yerr=err, fmt='none',
                ecolor=GREY['ink'], elinewidth=1.0, capsize=3.5, zorder=4)
    for i, d in enumerate(rates):
        ax.text(pos[i], d['ci_high'] * 100 + 2.5, f"{d['rate']*100:.0f}%",
                ha='center', va='bottom', fontsize=8.5, color=GREY['ink'])
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, 118)
    style_ax(ax, ylabel='Docking success  [%]')
    add_panel_label(ax, '(c)')

    fig.tight_layout()
    return save_fig(fig, path)


def fig_study2(levels, curves, path):
    x = np.asarray(levels, dtype=float)
    stats = [c.stats() for c in curves]
    # Median with a 5th-95th percentile band: the distribution is tight with
    # a rare diverging tail, so a mean +- 1 sigma band extends below zero and
    # hides the fact that the median is flat.
    med = np.array([s['final_range_m']['median'] for s in stats])
    p05 = np.array([s['final_range_m']['p05'] for s in stats])
    p95 = np.array([s['final_range_m']['p95'] for s in stats])
    rate = np.array([s['dock_rate']['rate'] for s in stats]) * 100
    lo = np.array([s['dock_rate']['ci_low'] for s in stats]) * 100
    hi = np.array([s['dock_rate']['ci_high'] for s in stats]) * 100
    drop = np.array([s['dropout_rate']['mean'] for s in stats]) * 100
    perr = np.array([s['mean_pos_err_m']['mean'] for s in stats])

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))

    ax = axes[0]
    ax.fill_between(x, p05, p95, facecolor=GREY['light'], edgecolor='none',
                    zorder=1, label='5th--95th percentile')
    ax.plot(x, med, **series_kw(0, marker=True), label='median final range',
            zorder=3)
    ax.set_yscale('log')
    style_ax(ax, xlabel=r'Pixel noise  $\sigma_{px}$  [px]',
             ylabel='Final range  [m]', legend=True, legend_loc='upper left')
    add_panel_label(ax, '(a)')

    ax = axes[1]
    ax.fill_between(x, lo, hi, facecolor=GREY['light'], edgecolor='none',
                    zorder=1)
    ax.plot(x, rate, **series_kw(1, marker=True), label='docking success',
            zorder=3)
    ax.set_ylim(-5, 108)
    style_ax(ax, xlabel=r'Pixel noise  $\sigma_{px}$  [px]',
             ylabel='Docking success  [%]', legend=True)
    add_panel_label(ax, '(b)')

    ax = axes[2]
    ax.plot(x, drop, **series_kw(2, marker=True), label='vision dropout')
    ax.set_ylabel('Dropout  [%]')
    ax2 = ax.twinx()
    ax2.plot(x, perr, **series_kw(3, marker=True), label='mean pos. error')
    ax2.set_ylabel('Mean position error  [m]')
    ax2.spines['top'].set_visible(False)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc='upper left',
              fontsize=8)
    style_ax(ax, xlabel=r'Pixel noise  $\sigma_{px}$  [px]')
    add_panel_label(ax, '(c)')

    fig.tight_layout()
    return save_fig(fig, path)


def fig_study3(rows, path, z_fill=Z_FILL_M):
    """Availability vs instantaneous range, with the fill criterion marked."""
    mid = np.array([0.5 * (r['range_lo'] + r['range_hi']) for r in rows])
    avail = np.array([r['avail'] for r in rows]) * 100
    lo = np.array([r['avail_ci_low'] for r in rows]) * 100
    hi = np.array([r['avail_ci_high'] for r in rows]) * 100
    vis = np.array([r['mean_visible'] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5))

    ax = axes[0]
    ax.fill_between(mid, lo, hi, facecolor=GREY['light'], edgecolor='none',
                    zorder=1)
    ax.plot(mid, avail, **series_kw(0, marker=True),
            label='measurement availability', zorder=3)
    ax.axvline(z_fill, color=GREY['mid'], linestyle=':', linewidth=1.0,
               zorder=2)
    ax.annotate(r'$z_{\mathrm{fill}} = fL/W$',
                xy=(z_fill, 50), xytext=(z_fill * 1.35, 42),
                fontsize=8, color=GREY['mid'])
    ax.set_xscale('log')
    ax.set_ylim(-5, 108)
    style_ax(ax, xlabel='Instantaneous range  [m]',
             ylabel='Availability  [%]', legend=True)
    add_panel_label(ax, '(a)')

    ax = axes[1]
    ax.plot(mid, vis, **series_kw(1, marker=True), label='keypoints in frame')
    ax.axvline(z_fill, color=GREY['mid'], linestyle=':', linewidth=1.0)
    ax.axhline(6.0, color=GREY['mid'], linestyle='--', linewidth=0.9)
    ax.annotate('solver minimum', xy=(mid[-1], 6.0), xytext=(mid[3], 6.6),
                fontsize=8, color=GREY['mid'])
    ax.set_xscale('log')
    style_ax(ax, xlabel='Instantaneous range  [m]',
             ylabel='Mean keypoints in frame', legend=True,
             legend_loc='lower right')
    add_panel_label(ax, '(b)')

    fig.tight_layout()
    return save_fig(fig, path)


# ── main ─────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--trials', type=int, default=100,
                    help='trials per configuration in Study 1 (default 100)')
    ap.add_argument('--noise-trials', type=int, default=40,
                    help='trials per noise level in Study 2 (default 40)')
    ap.add_argument('--jobs', type=int, default=1,
                    help='parallel worker processes (default 1)')
    ap.add_argument('--seed', type=int, default=20260901,
                    help='campaign base seed')
    ap.add_argument('--steps', type=int, default=250,
                    help='simulation steps per trial')
    args = ap.parse_args(argv)

    apply_style()

    # ── Study 1 ──
    print('=' * 72)
    print(f'Study 1 — dispersed Monte Carlo  '
          f'(N = {args.trials} per configuration, paired)')
    print('=' * 72)

    cfgs = standard_configs(n_trials=args.trials, base_seed=args.seed)
    results = []
    for name, mc in cfgs.items():
        mc.base['n_steps'] = args.steps
        print(f'\n  [{name}]')
        res = run_campaign(mc, n_jobs=args.jobs, progress_every=10)
        print(res.report())
        res.to_csv(OUT / f"mc_{name.replace(' ', '_').replace(',', '')}.csv")
        results.append(res)

    print('\n' + fig_study1(results, OUT / 'expF_fig8_montecarlo.png'))

    # ── Study 2 ──
    print('\n' + '=' * 72)
    print(f'Study 2 — pixel-noise ablation  '
          f'(N = {args.noise_trials} per level, paired across levels)')
    print('=' * 72)

    mc_abl = MonteCarloConfig(
        n_trials=args.noise_trials, base_seed=args.seed, label='vision + EKF',
        dispersion=Dispersion(),
        base=dict(n_steps=args.steps, u_max=0.3, pos_weight=15.0,
                  vel_weight=1.5, use_vision=True, use_ekf=True),
    )
    curves = sweep(mc_abl, 'pixel_noise_std', NOISE_LEVELS, n_jobs=args.jobs)
    for lvl, c in zip(NOISE_LEVELS, curves):
        c.to_csv(OUT / f'mc_noise_{lvl:.1f}px.csv')

    print('\n' + fig_study2(NOISE_LEVELS, curves,
                            OUT / 'expF_fig9_ablation_noise.png'))

    # ── Study 3 ──
    print('\n' + '=' * 72)
    print('Study 3 — measurement availability vs instantaneous range')
    print('=' * 72)
    full = results[-1]                      # the vision + EKF campaign
    rows = full.availability_profile()
    print(f"  predicted fill range  z_fill = f L / W = {Z_FILL_M:.2f} m")
    print(f"  {'range [m]':>13}{'steps':>8}{'avail [%]':>11}"
          f"{'95% CI':>18}{'kpts':>8}")
    for r in rows:
        print(f"  {r['range_lo']:5.1f}–{r['range_hi']:<7.1f}{r['n_steps']:>8}"
              f"{r['avail']*100:>11.1f}"
              f"   [{r['avail_ci_low']*100:5.1f},{r['avail_ci_high']*100:6.1f}]"
              f"{r['mean_visible']:>8.1f}")
    coarse = full.dropout_vs_range()
    print('\n  trial-level dropout by initial range (dispersion coverage check)')
    for r in coarse:
        print(f"  {r['range_lo']:5.0f}–{r['range_hi']:<7.0f}{r['n']:>8}"
              f"{r['dropout_mean']*100:>11.1f}")
    print('\n' + fig_study3(rows, OUT / 'expF_fig10_availability.png'))

    # ── LaTeX tables ──
    tex = [
        latex_table(results,
                    caption=(f'Dispersed Monte Carlo campaign, '
                             f'$N = {args.trials}$ trials per configuration '
                             f'drawn from a common seed sequence so the three '
                             f'configurations are compared trial-by-trial. '
                             f'Docking success is reported with a Wilson '
                             f'95\\,\\% score interval.'),
                    label='tab:mc_configs'),
        '',
        latex_table(curves,
                    caption=(f'Pixel-noise ablation, $N = {args.noise_trials}$ '
                             f'paired trials per level.'),
                    label='tab:mc_noise'),
    ]
    tex_path = OUT / 'mc_tables.tex'
    tex_path.write_text('\n'.join(tex))
    print(f'\n{tex_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""
Experiment G — Sensitivity of the availability knee, and a causality test
=========================================================================
Run from the project root:
    python experiments/experiment_G.py [--trials 100] [--jobs 6]

Study G1 — minimum-correspondence sweep
    The visibility gate and the RANSAC hypothesis size are both 6 in the
    nominal pipeline, although EPnP admits 4.  The full vision + MEKF campaign
    (sigma_px = 1.5 px, the Study-1 dispersion and seeds) is rerun with both set
    to N_min = 4, 5 and 6.  Per instantaneous-range bin it reports
        A      = P(valid pose returned)
        A_vis  = P(at least N_min keypoints pass the visibility gate)
        A_PnP  = P(valid pose | gate passed)      so  A = A_vis * A_PnP,
    and docking success.  This separates where the knee comes from (field of
    view) from how deep it goes (the N_min the solver is configured with).

Study G2 — does availability alone explain the failures?
    For sigma_px = 1.5 and 4.0 px, the vision + MEKF campaign (the Study-2
    configuration and seeds) gives a per-step availability profile A(range).
    The direct-measurement + MEKF configuration -- same initial conditions,
    same filter, a 0.5 m / 0.05 rad sensor that never fails -- is then rerun
    (a) as is, and (b) with each measurement withheld with probability
    1 - A(true range).  If (b) reproduces the vision pipeline's docking rate,
    the loss of measurements accounts for the failures; if (b) docks where
    vision does not, the failures need inaccurate accepted poses as well.

Study G3 — RANSAC inlier threshold at high noise   (--g3, run on its own)
    At sigma_px = 4 px the fixed 3 px inlier threshold is below the noise
    level.  The 4 px vision campaign is rerun with the threshold at 3, 6 and
    12 px (12 px ~ 3 sigma) to separate the effect of noise from the effect of
    an unscaled threshold.

Outputs
    outputs/expG_fig11_nmin_availability.png
    outputs/expG_nmin.csv, outputs/expG_causality.csv
    outputs/expG_tables.tex
"""
import argparse
import csv
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from simulation.montecarlo import (AVAIL_BINS, Dispersion, MonteCarloConfig,
                                   trial_config, wilson_interval)
from simulation.runner import run_simulation
from viz.style import GREY, add_panel_label, apply_style, save_fig, series_kw, style_ax

OUT = pathlib.Path('outputs')
OUT.mkdir(exist_ok=True)
Z_FILL_M = 6.25
BASE = dict(n_steps=250, u_max=0.3, pos_weight=15.0, vel_weight=1.5)
EDGES = np.asarray(AVAIL_BINS)
NB = len(EDGES) - 1
# bins as in the paper's availability table: 10-50 m pooled
TABLE_BINS = [(10, 50), (8, 10), (6, 8), (5, 6), (4, 5), (3, 4), (2, 3), (1, 2), (0, 1)]


def _trial(args):
    """One trial -> outcome plus per-range-bin step / gate / solve counts."""
    mc, i, extra = args
    cfg, seed, _ = trial_config(mc, i)
    for k, v in extra.items():
        setattr(cfg, k, v)
    res = run_simulation(cfg, rng_seed=seed)
    n = res.n_steps_run
    rng_m = res.range_m[:n]
    used = np.asarray(res.meas_used[:n], dtype=bool)
    vis = np.asarray(res.n_visible[:n]) if res.n_visible is not None else np.zeros(n)
    gate = vis >= cfg.min_visible_kpts if cfg.use_vision else np.ones(n, bool)
    idx = np.clip(np.digitize(rng_m, EDGES) - 1, 0, NB - 1)
    steps = np.bincount(idx, minlength=NB)
    n_gate = np.bincount(idx, weights=gate, minlength=NB)
    n_used = np.bincount(idx, weights=used, minlength=NB)
    return dict(trial=i, docked=res.dock_step is not None,
                final_range=float(res.range_m[-1]), dv=float(res.delta_v),
                steps=steps, gate=n_gate, used=n_used,
                dropout=float(1 - used.mean()) if n else float('nan'))


def campaign(pool, mc, n, extra=None):
    return list(pool.map(_trial, [(mc, i, extra or {}) for i in range(n)]))


def pooled(rows):
    s = sum(r['steps'] for r in rows); g = sum(r['gate'] for r in rows); u = sum(r['used'] for r in rows)
    return s, g, u


def table_rows(rows):
    s, g, u = pooled(rows)
    out = []
    for lo, hi in TABLE_BINS:
        m = (EDGES[:-1] >= lo) & (EDGES[1:] <= hi)
        n, ng, nu = int(s[m].sum()), float(g[m].sum()), float(u[m].sum())
        if n == 0:
            continue
        out.append(dict(lo=lo, hi=hi, n=n, A=nu / n, A_vis=ng / n,
                        A_pnp=(nu / ng) if ng else float('nan'), ci=wilson_interval(int(nu), n)))
    return out


def dock_summary(rows):
    k = sum(r['docked'] for r in rows); n = len(rows)
    lo, hi = wilson_interval(k, n)
    fr = np.array([r['final_range'] for r in rows])
    return k, n, lo, hi, float(np.median(fr)), float(np.percentile(fr, 95))


def main(argv=None):
    ap = argparse.ArgumentParser(description='Experiment G')
    ap.add_argument('--trials', type=int, default=100)
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--seed', type=int, default=20260901)
    ap.add_argument('--g3', action='store_true', help='run only Study G3 (RANSAC threshold)')
    a = ap.parse_args(argv)
    if a.g3:
        return study_g3(a)
    apply_style()
    disp = Dispersion()
    tex = []

    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        # ── G1 ──────────────────────────────────────────────────────────────
        print('=' * 72); print(f'Study G1 — minimum-correspondence sweep (N = {a.trials}, sigma = 1.5 px)'); print('=' * 72)
        mc_v = MonteCarloConfig(n_trials=a.trials, base_seed=a.seed, label='vision + EKF',
                                dispersion=disp, base=dict(BASE, use_vision=True, use_ekf=True))
        g1 = {}
        with open(OUT / 'expG_nmin.csv', 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['n_min', 'range_lo', 'range_hi', 'steps', 'A', 'A_vis', 'A_pnp_given_vis', 'A_ci_low', 'A_ci_high'])
            for nmin in (4, 5, 6):
                rows = campaign(pool, mc_v, a.trials, dict(min_visible_kpts=nmin, ransac_n_min=nmin))
                g1[nmin] = rows
                k, n, lo, hi, med, p95 = dock_summary(rows)
                print(f'\n  N_min = {nmin}: docked {k}/{n} [{100*lo:.0f}, {100*hi:.0f}] %, '
                      f'median final range {med:.2f} m, p95 {p95:.2f} m, mean dropout '
                      f'{100*np.mean([r["dropout"] for r in rows]):.1f} %')
                print(f"  {'range [m]':>10}{'steps':>7}{'A [%]':>8}{'A_vis [%]':>11}{'A_PnP|vis [%]':>15}")
                for t in table_rows(rows):
                    print(f"  {t['lo']:>4}-{t['hi']:<5}{t['n']:>7}{100*t['A']:>8.1f}{100*t['A_vis']:>11.1f}{100*t['A_pnp']:>15.1f}")
                    w.writerow([nmin, t['lo'], t['hi'], t['n'], t['A'], t['A_vis'], t['A_pnp'], *t['ci']])

        # LaTeX: availability per bin for the three N_min, plus docking
        hdr = ' & '.join(f'$N_{{\\min}}={n}$' for n in (4, 5, 6))
        lines = ['\\begin{table}[!tb]', '\\caption{Measurement availability against instantaneous range for three',
                 'minimum correspondence counts (visibility gate and RANSAC sample size), full vision--MEKF',
                 f'pipeline, $\\sigma_{{px}} = 1.5$~px, the same {a.trials} dispersed trials in each column.}}',
                 '\\label{tab:nmin}', '\\centering', '\\begin{tabular}{lrrr}', '\\toprule',
                 f'Range [m] & {hdr} \\\\', '\\midrule']
        tr = {n: table_rows(g1[n]) for n in (4, 5, 6)}
        for j, (lo, hi) in enumerate(TABLE_BINS):
            lab = '$<1$' if lo == 0 else f'{lo}--{hi}'
            cells = []
            for n in (4, 5, 6):
                t = next((x for x in tr[n] if x['lo'] == lo), None)
                cells.append('--' if t is None else f"{100*t['A']:.1f}\\%")
            lines.append(f'{lab} & ' + ' & '.join(cells) + ' \\\\')
        dk = [dock_summary(g1[n]) for n in (4, 5, 6)]
        lines += ['\\midrule', 'Docking & ' + ' & '.join(f'{d[0]}/{d[1]}' for d in dk) + ' \\\\',
                  '\\bottomrule', '\\end{tabular}', '\\end{table}', '']
        tex += lines

        # figure
        fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5))
        for j, n in enumerate((4, 5, 6)):
            s, g, u = pooled(g1[n])
            ok = s > 0
            mid = 0.5 * (EDGES[:-1] + EDGES[1:])[ok]
            axes[0].plot(mid, 100 * u[ok] / s[ok], **series_kw(j, marker=True), label=f'$N_{{min}}={n}$')
            axes[1].plot(mid, 100 * g[ok] / s[ok], **series_kw(j, marker=True), label=f'$N_{{min}}={n}$')
        for ax, yl, pl in ((axes[0], 'Availability  [%]', '(a)'), (axes[1], 'Visibility gate passed  [%]', '(b)')):
            ax.axvline(Z_FILL_M, color=GREY['mid'], linestyle=':', linewidth=1.0)
            ax.set_xscale('log'); ax.set_ylim(-5, 108)
            style_ax(ax, xlabel='Instantaneous range  [m]', ylabel=yl, legend=True, legend_loc='lower right')
            add_panel_label(ax, pl)
        fig.tight_layout()
        print('\n' + save_fig(fig, OUT / 'expG_fig11_nmin_availability.png'))

        # ── G2 ──────────────────────────────────────────────────────────────
        print('\n' + '=' * 72); print(f'Study G2 — availability-matched dropout (N = {a.trials})'); print('=' * 72)
        g2 = []
        for sigma in (1.5, 4.0):
            mc_vis = MonteCarloConfig(n_trials=a.trials, base_seed=a.seed, label='vision + EKF',
                                      dispersion=replace(disp, pixel_noise_std=(sigma, sigma)),
                                      base=dict(BASE, use_vision=True, use_ekf=True))
            mc_dir = MonteCarloConfig(n_trials=a.trials, base_seed=a.seed, label='EKF, direct',
                                      dispersion=replace(disp, pixel_noise_std=(0.0, 0.0)),
                                      base=dict(BASE, use_vision=False, use_ekf=True))
            vis_rows = campaign(pool, mc_vis, a.trials)
            s, g, u = pooled(vis_rows)
            prof = tuple(float(u[b] / s[b]) if s[b] else 1.0 for b in range(NB))
            dir_rows = campaign(pool, mc_dir, a.trials)
            drop_rows = campaign(pool, mc_dir, a.trials, dict(dropout_profile=(tuple(EDGES), prof)))
            print(f'\n  sigma_px = {sigma} px   availability profile used: ' +
                  ', '.join(f'{EDGES[b]:g}-{EDGES[b+1]:g} m: {100*prof[b]:.0f}%' for b in range(NB) if s[b]))
            for name, rows in (('vision + MEKF', vis_rows), ('direct + MEKF', dir_rows),
                               ('direct + MEKF, availability-matched dropout', drop_rows)):
                k, n, lo, hi, med, p95 = dock_summary(rows)
                drop = 100 * np.mean([r['dropout'] for r in rows])
                print(f'    {name:<45} docked {k:>3}/{n} [{100*lo:3.0f},{100*hi:4.0f}] %  '
                      f'median {med:5.2f} m  p95 {p95:6.2f} m  dropout {drop:4.1f} %')
                g2.append(dict(sigma=sigma, config=name, docked=k, n=n, ci_low=lo, ci_high=hi,
                               median=med, p95=p95, dropout=drop))
        with open(OUT / 'expG_causality.csv', 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(g2[0])); w.writeheader(); w.writerows(g2)

        lines = ['\\begin{table}[!tb]', '\\caption{Causality test. The direct-measurement filter is rerun with each',
                 'measurement withheld at the availability the vision pipeline achieved at the same range;',
                 f'{a.trials} paired trials per row. Terminal range: median [95th percentile].}}',
                 '\\label{tab:causality}', '\\centering', '\\begin{tabular}{llrrr}', '\\toprule',
                 '$\\sigma_{px}$ & Configuration & Dropout & Terminal range [m] & Docking \\\\', '\\midrule']
        for r in g2:
            lines.append(f"{r['sigma']:.1f} & {r['config']} & {r['dropout']:.1f}\\% & {r['median']:.2f} [{r['p95']:.2f}] & "
                         f"{r['docked']}/{r['n']} [{100*r['ci_low']:.0f},\\,{100*r['ci_high']:.0f}] \\\\")
        lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
        tex += lines

    (OUT / 'expG_tables.tex').write_text('\n'.join(tex) + '\n')
    print(f'\n{OUT / "expG_tables.tex"}')
    return 0


def study_g3(a):
    print('=' * 72); print(f'Study G3 — RANSAC inlier threshold at sigma = 4 px (N = {a.trials})'); print('=' * 72)
    rows_out, lines = [], []
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        for sigma, thr in ((1.5, 3.0), (4.0, 3.0), (4.0, 6.0), (4.0, 12.0)):
            mc = MonteCarloConfig(n_trials=a.trials, base_seed=a.seed, label='vision + EKF',
                                  dispersion=replace(Dispersion(), pixel_noise_std=(sigma, sigma)),
                                  base=dict(BASE, use_vision=True, use_ekf=True))
            rows = campaign(pool, mc, a.trials, dict(ransac_thr=thr))
            k, n, lo, hi, med, p95 = dock_summary(rows)
            drop = 100 * np.mean([r['dropout'] for r in rows])
            tr = {(t['lo'], t['hi']): t['A'] for t in table_rows(rows)}
            prof = ', '.join(f"{lo_}-{hi_} m {100*v:.0f}%" for (lo_, hi_), v in tr.items())
            print(f'  sigma {sigma} px, threshold {thr:>4} px: docked {k}/{n} [{100*lo:.0f},{100*hi:.0f}] %, '
                  f'median {med:.2f} m, p95 {p95:.2f} m, dropout {drop:.1f} %')
            print(f'      availability: {prof}')
            rows_out.append(dict(sigma=sigma, threshold_px=thr, docked=k, n=n, ci_low=lo, ci_high=hi,
                                 median=med, p95=p95, dropout=drop,
                                 **{f'A_{lo_}_{hi_}m': v for (lo_, hi_), v in tr.items()}))
            lines.append(f"{sigma:.1f} & {thr:.0f} & {drop:.1f}\\% & {100*tr.get((10, 50), float('nan')):.1f}\\% & "
                         f"{100*tr.get((2, 3), float('nan')):.1f}\\% & {med:.2f} [{p95:.2f}] & {k}/{n} [{100*lo:.0f},\\,{100*hi:.0f}] \\\\")
    with open(OUT / 'expG3_threshold.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0])); w.writeheader(); w.writerows(rows_out)
    tex = ['\\begin{table}[!tb]', '\\caption{Effect of the RANSAC inlier threshold at high keypoint noise; full vision--MEKF',
           f'pipeline, {a.trials} paired dispersed trials per row. Terminal range: median [95th percentile].}}',
           '\\label{tab:threshold}', '\\centering', '\\begin{tabular}{rrrrrrr}', '\\toprule',
           '$\\sigma_{px}$ & Threshold [px] & Dropout & $A_{10\\text{--}50\\,\\mathrm{m}}$ & $A_{2\\text{--}3\\,\\mathrm{m}}$ & Terminal range [m] & Docking \\\\',
           '\\midrule'] + lines + ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    (OUT / 'expG3_tables.tex').write_text('\n'.join(tex) + '\n')
    print(f'\n{OUT / "expG3_tables.tex"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

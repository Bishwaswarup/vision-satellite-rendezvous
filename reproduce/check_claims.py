#!/usr/bin/env python3
"""
check_claims.py -- compare a fresh rerun against every number the paper
quotes from the experiment scripts.

Called by reproduce/rerun_all.sh; can also be run on its own:

    .venv/bin/python reproduce/check_claims.py --logs reproduce/logs --outputs outputs \
        [--backup outputs_before_rerun_<stamp>]

The reference values are those published in the paper, produced on macOS with
Python 3.11.9, NumPy 2.4.6 and SciPy 1.17.1.  On the same platform every value
should PASS.  On another CPU architecture or linear-algebra library the
vision-in-the-loop numbers can differ slightly -- see README, "Reproducibility".

Verdicts
  PASS   the rerun reproduces the paper's value at the precision the paper prints
  CLOSE  differs only in the last printed digit / within 3 % (floating-point or
         BLAS drift between machines) -- usually fine, but the text may need
         a one-digit touch-up
  FAIL   a real disagreement: the paper's claim is not reproduced
  INFO   machine-dependent (wall-clock timings) -- reported, not judged
"""
import argparse
import csv
import math
import pathlib
import re
import sys

ROWS = []          # (verdict, section, claim, paper, rerun, note)


def rhalf(x):      # round half away from zero (Python's round() is banker's)
    return math.floor(x + 0.5)


def judge(section, claim, paper, rerun, tol, note='', rel_close=0.03):
    """tol = half a unit in the paper's last printed digit."""
    if rerun is None or (isinstance(rerun, float) and math.isnan(rerun)):
        if isinstance(paper, float) and math.isnan(paper):
            v = 'PASS'
        else:
            v = 'FAIL'
            note = (note + '; ' if note else '') + 'value missing from rerun output'
    else:
        d = abs(rerun - paper)
        if d <= tol + 1e-12:
            v = 'PASS'
        elif d <= max(3 * tol, rel_close * abs(paper)):
            v = 'CLOSE'
        else:
            v = 'FAIL'
    ROWS.append((v, section, claim, paper, rerun, note))


def flag(section, claim, ok, paper_txt, rerun_txt, note=''):
    ROWS.append(('PASS' if ok else 'FAIL', section, claim, paper_txt, rerun_txt, note))


def info(section, claim, paper_txt, rerun_txt, note='machine-dependent'):
    ROWS.append(('INFO', section, claim, paper_txt, rerun_txt, note))


def read(p):
    try:
        return pathlib.Path(p).read_text(errors='replace')
    except FileNotFoundError:
        return ''


NUM = r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|nan)'


def fnum(s):
    return float('nan') if s.strip().lower() in ('nan', '--') else float(s)


# ─────────────────────────────────────────────────────────────────────────────
def check_tests(logs):
    t = read(logs / 'pytest.log')
    if not t:
        flag('Tests', 'pytest ran', False, 'all pass', 'no log')
        return
    tail = t.strip().splitlines()[-1] if t.strip() else ''
    g = lambda k: int(m.group(1)) if (m := re.search(r'(\d+) ' + k, tail)) else 0
    p, f, e, s = g('passed'), g('failed'), g('error'), g('skipped')
    flag('Tests', 'unit tests', f == 0 and e == 0 and p > 0,
         'all pass', f'{p} passed, {f} failed, {e} errors, {s} skipped',
         'skips are optional extras (vispy / OpenCV)' if s else '')


def check_A(logs):
    t = read(logs / 'A.log')
    m = re.search(r'Max position error\s*:\s*' + NUM, t)
    judge('Exp A', 'max position error vs DOP853 [m]', 4.63e-10,
          fnum(m.group(1)) if m else None, 0.005e-10, rel_close=0.10)
    m = re.search(r'Max velocity error\s*:\s*' + NUM, t)
    judge('Exp A', 'max velocity error [m/s]', 5.36e-13,
          fnum(m.group(1)) if m else None, 0.005e-13, rel_close=0.10)
    conv = {}
    for m in re.finditer(r'^\s*(1e-\d+)\s*\|\s*(1e-\d+)\s*\|\s*' + NUM, t, re.M):
        conv[m.group(1)] = fnum(m.group(3))
    judge('Exp A', 'pos. error at tol (1e-6, 1e-8) [m]', 4.50e-4,
          conv.get('1e-06'), 0.005e-4, rel_close=0.10)
    judge('Exp A', 'pos. error at tol (1e-13, 1e-15) [m]', 5.31e-11,
          conv.get('1e-13'), 0.005e-11, rel_close=0.10)


B_PAPER = {  # sigma: (EPnP t cm, EPnP r deg, LM t cm, LM r deg, solved)
    0.10: (0.395, 0.056, 0.362, 0.048, 200), 0.25: (1.136, 0.141, 0.982, 0.128, 200),
    0.50: (2.010, 0.273, 1.732, 0.249, 200), 1.00: (4.349, 0.550, 3.986, 0.511, 200),
    1.50: (6.316, 0.866, 5.695, 0.787, 200), 2.00: (7.876, 1.079, 7.065, 1.025, 200),
    3.00: (13.732, 1.678, 11.492, 1.533, 200), 5.00: (23.328, 2.868, 20.245, 2.569, 200),
}


def check_B(logs):
    t = read(logs / 'B.log')
    got = {}
    for m in re.finditer(r'^\s*(\d+\.\d\d)\s+' + r'\s+'.join([NUM] * 4) + r'\s+(\d+)/(\d+)\s*$', t, re.M):
        got[float(m.group(1))] = tuple(fnum(m.group(i)) for i in range(2, 6)) + (int(m.group(6)),)
    names = ('EPnP t [cm]', 'EPnP r [deg]', 'EPnP+LM t [cm]', 'EPnP+LM r [deg]')
    for s, ref in B_PAPER.items():
        g = got.get(s)
        for i, nm in enumerate(names):
            judge('Exp B (Table pose_noise)', f'sigma={s:.2f}px {nm}', ref[i],
                  g[i] if g else None, 0.0005)
        if g:
            flag('Exp B (Table pose_noise)', f'sigma={s:.2f}px solved', abs(g[4] - ref[4]) <= 0,
                 f'{ref[4]}/200', f'{g[4]}/200',
                 'off by one solve: stochastic edge case' if abs(g[4] - ref[4]) == 1 else '')
    if len(got) == len(B_PAPER):
        tr = [100 * (g[0] - g[2]) / g[0] for g in got.values()]
        at = [100 * (g[1] - g[3]) / g[1] for g in got.values()]
        flag('Exp B (text)', 'LM reduces translation error by 8-16 %',
             (rhalf(min(tr)), rhalf(max(tr))) == (8, 16), '8-16 %', f'{min(tr):.1f}-{max(tr):.1f} %')
        flag('Exp B (text)', 'LM reduces attitude error by 5-14 %',
             (rhalf(min(at)), rhalf(max(at))) == (5, 14), '5-14 %', f'{min(at):.1f}-{max(at):.1f} %')


C_PAPER = {  # frac%: (EPnP fail %, EPnP t, EPnP r, RANSAC fail %, RANSAC t, RANSAC r)
    0: (0, 4.278, 0.643, 0, 5.145, 0.853), 5: (99, 14.538, 0.682, 0, 5.551, 0.841),
    10: (100, math.nan, math.nan, 0, 5.939, 0.913), 20: (100, math.nan, math.nan, 0, 6.169, 0.921),
    30: (100, math.nan, math.nan, 0, 6.035, 1.026),
}


def check_C(logs):
    t = read(logs / 'C.log')
    got = {}
    for m in re.finditer(r'^\s*(\d+)%\s+' + r'\s+'.join([NUM] * 6) + r'\s*$', t, re.M):
        got[int(m.group(1))] = tuple(fnum(m.group(i)) for i in range(2, 8))
    names = ('EPnP fail %', 'EPnP t [cm]', 'EPnP r [deg]', 'RANSAC fail %', 'RANSAC t [cm]', 'RANSAC r [deg]')
    tols = (0.5, 0.0005, 0.0005, 0.5, 0.0005, 0.0005)
    for f, ref in C_PAPER.items():
        g = got.get(f)
        for i, nm in enumerate(names):
            if isinstance(ref[i], float) and math.isnan(ref[i]):
                continue
            judge('Exp C (Table outliers)', f'f_out={f}% {nm}', ref[i], g[i] if g else None, tols[i])
    if got:
        flag('Exp C (text)', 'RANSAC-EPnP solves every trial up to 30 % outliers',
             all(g[3] == 0 for g in got.values()), '0 % failures',
             ', '.join(f'{k}%:{v[3]:.1f}' for k, v in sorted(got.items())))


def check_D(logs):
    t = read(logs / 'D.log')
    spec = [('Pos RMSE \\[m\\]', 'Position RMSE [m]', 0.5997, 0.5997),
            ('Att RMSE \\[°\\]', 'Attitude RMSE [deg]', 1.2518, 1.2802),
            ('Vel RMSE \\[m/s\\]', 'Velocity RMSE [m/s]', 0.0226, 0.0226),
            ('Pos max err \\[m\\]', 'Max position error [m]', 1.5343, 1.5343),
            ('Att max err \\[°\\]', 'Max attitude error [deg]', 7.2282, 7.7671)]
    for rx, nm, pe, pu in spec:
        m = re.search(rx + r'\s+' + NUM + r'\s+' + NUM, t)
        judge('Exp D (Table MEKF/UKF)', f'MEKF {nm}', pe, fnum(m.group(1)) if m else None, 0.00005)
        judge('Exp D (Table MEKF/UKF)', f'UKF  {nm}', pu, fnum(m.group(2)) if m else None, 0.00005)
    m = re.search(r'Meas accepted\s+(\d+)\s+(\d+)', t)
    flag('Exp D (Table MEKF/UKF)', 'measurements accepted', bool(m) and m.group(1) == m.group(2) == '300',
         '300 / 300', f'{m.group(1)} / {m.group(2)}' if m else 'missing')
    m = re.search(r'Wall time \[s\]\s+' + NUM + r'\s+' + NUM, t)
    if m:
        e, u = fnum(m.group(1)), fnum(m.group(2))
        info('Exp D', 'wall time MEKF / UKF [s]', '0.958 / 1.021 (UKF 6.6 % slower)',
             f'{e:.3f} / {u:.3f} (UKF {100*(u-e)/e:+.1f} %)',
             'timing differs by machine; paper already calls the gap "no meaningful difference"')


def check_E(logs):
    t = read(logs / 'E.log')
    m = re.search(r'Total Δv \[m/s\]\s+' + NUM + r'\s+' + NUM, t)
    lq, mp = (fnum(m.group(1)), fnum(m.group(2))) if m else (None, None)
    judge('Exp E (LQR vs MPC)', 'LQR total delta-v [m/s]', 16.83, lq, 0.005)
    judge('Exp E (LQR vs MPC)', 'MPC total delta-v [m/s]', 5.00, mp, 0.005)
    if lq and mp:
        flag('Exp E (text)', 'MPC saves 70 % delta-v', rhalf(100 * (lq - mp) / lq) == 70,
             '70 %', f'{100*(lq-mp)/lq:.1f} %')
    m = re.search(r'LQR closed-loop stable\s*:\s*(\w+)', t)
    flag('Exp E', 'LQR closed loop stable', bool(m) and m.group(1) == 'True', 'True',
         m.group(1) if m else 'missing')
    m = re.search(r'Time per step \[ms\]\s+' + NUM + r'\s+' + NUM, t)
    if m:
        info('Exp E', 'MPC time per step [ms]', '0.26', f'{fnum(m.group(2)):.2f}')


TAB_RX = re.compile(r'^(.+?) & (\d+) & \$([\d.]+)\$ \[([\d.]+),\\,([\d.]+)\] & '
                    r'\$([\d.]+) \\pm ([\d.]+)\$ & \$(\d+)\$ \[(\d+),\\,(\d+)\]', re.M)
F1_PAPER = {'perfect state': (0.33, 0.13, 0.84, 2.53, 0.80, 100, 96, 100),
            'EKF, direct': (0.54, 0.24, 0.98, 2.65, 0.81, 100, 96, 100),
            'vision + EKF': (0.37, 0.11, 0.89, 2.53, 0.80, 100, 96, 100)}
F2_PAPER = {0.3: (0.35, 0.10, 0.90, 100, 96, 100), 0.7: (0.36, 0.09, 0.97, 100, 96, 100),
            1.0: (0.35, 0.08, 0.89, 100, 96, 100), 1.5: (0.37, 0.11, 0.89, 100, 96, 100),
            2.0: (0.37, 0.11, 0.88, 99, 95, 100), 3.0: (0.38, 0.13, 0.99, 96, 90, 98),
            4.0: (0.47, 0.14, 24.46, 85, 77, 91)}
# availability bins as printed in the paper; the 10-50 m row pools the four far-field
# bins the script prints (10-15, 15-20, 20-30, 30-50 m)
F3_PAPER = {50: (1682, 15.0, 100.0, 99.8, 100.0), 10: (206, 14.4, 100.0, 98.2, 100.0),
            8: (229, 10.8, 100.0, 98.4, 100.0), 6: (132, 8.7, 97.7, 93.5, 99.2),
            5: (151, 7.7, 84.8, 78.2, 89.6), 4: (171, 7.2, 69.6, 62.3, 76.0),
            3: (211, 4.8, 28.0, 22.3, 34.4), 2: (326, 3.3, 8.9, 6.3, 12.5),
            1: (807, 2.5, 1.7, 1.0, 2.9)}


def check_F(logs, outputs):
    tex = read(outputs / 'mc_tables.tex')
    rows = {m.group(1).strip(): m for m in TAB_RX.finditer(tex)}
    S1 = 'Exp F Study 1 (Table configs)'
    dv = {}
    for name, ref in F1_PAPER.items():
        m = rows.get(name)
        vals = [float(m.group(i)) for i in (3, 4, 5, 6, 7, 8, 9, 10)] if m else [None] * 8
        lab = ('median final range', 'p05 final range', 'p95 final range',
               'delta-v mean', 'delta-v std', 'docking %', 'CI low', 'CI high')
        for i, nm in enumerate(lab):
            judge(S1, f'{name}: {nm}', ref[i], vals[i], 0.005 if i < 5 else 0.5)
        if m:
            dv[name] = vals[3]
    if len(dv) == 3:
        spread = 100 * (max(dv.values()) - min(dv.values())) / min(dv.values())
        flag('Exp F (text)', 'configs within about 5 % in propellant', spread <= 6.0,
             '~5 %', f'{spread:.1f} %')
    S2 = 'Exp F Study 2 (Table ablation)'
    med, p95, dock = {}, {}, {}
    for lvl, ref in F2_PAPER.items():
        key = next((k for k in rows if k.endswith(f'pixel\\_noise\\_std={lvl}')), None)
        m = rows.get(key) if key else None
        vals = [float(m.group(i)) for i in (3, 4, 5, 8, 9, 10)] if m else [None] * 6
        lab = ('median final range', 'p05', 'p95', 'docking %', 'CI low', 'CI high')
        for i, nm in enumerate(lab):
            judge(S2, f'{lvl} px: {nm}', ref[i], vals[i], 0.005 if i < 3 else 0.5)
        if m:
            med[lvl], p95[lvl], dock[lvl] = vals[0], vals[2], vals[3]
    if len(med) == 7:
        span = max(med[l] for l in (0.3, 0.7, 1.0, 1.5, 2.0, 3.0)) - min(med[l] for l in (0.3, 0.7, 1.0, 1.5, 2.0, 3.0))
        judge('Exp F (text/abstract)', 'median terminal range varies by 3 cm (0.3-3.0 px) [m]',
              0.03, span, 0.005)
        flag('Exp F (text/abstract)', 'p95 0.88 -> 0.99 -> 24.46 m (2/3/4 px)',
             (p95[2.0], p95[3.0], p95[4.0]) == (0.88, 0.99, 24.46), '0.88 / 0.99 / 24.46',
             f'{p95[2.0]} / {p95[3.0]} / {p95[4.0]}')
        flag('Exp F (text/abstract)', 'docking 99 % -> 85 % between 2 and 4 px',
             (dock[2.0], dock[4.0]) == (99, 85), '99 -> 85', f'{dock[2.0]:.0f} -> {dock[4.0]:.0f}')

    # Study 3 from the log
    t = read(logs / 'F.log')
    s3 = t.split('Study 3', 1)[1] if 'Study 3' in t else ''
    m = re.search(r'z_fill = f L / W = ' + NUM, s3)
    judge('Exp F Study 3', 'fill range z_fill [m]', 6.25, fnum(m.group(1)) if m else None, 0.005)
    av = {}
    far = []
    for m in re.finditer(r'^\s*([\d.]+)–([\d.]+)\s+(\d+)\s+([\d.]+)\s+\[\s*([\d.]+),\s*([\d.]+)\]\s+([\d.]+)\s*$', s3, re.M):
        lo, hi = float(m.group(1)), float(m.group(2))
        row = (int(m.group(3)), float(m.group(7)), float(m.group(4)), float(m.group(5)), float(m.group(6)))
        if lo >= 10.0 - 1e-9:
            far.append(row)
        else:
            av[rhalf(hi)] = row
    if far:   # pool 10-50 m the way the paper's table does
        n = sum(r[0] for r in far); k = sum(r[0] * r[2] / 100 for r in far)
        z = 1.959963985; p = k / n; den = 1 + z * z / n
        c = (p + z * z / (2 * n)) / den; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        av[50] = (n, round(sum(r[0] * r[1] for r in far) / n, 1), round(100 * p, 1),
                  round(100 * (c - h), 1), round(100 * (c + h), 1))
    S3 = 'Exp F Study 3 (Table availability)'
    lab = {50: '10-50 m', 10: '8-10 m', 8: '6-8 m', 6: '5-6 m', 5: '4-5 m', 4: '3-4 m', 3: '2-3 m', 2: '1-2 m', 1: '<1 m'}
    for hi, ref in F3_PAPER.items():
        g = av.get(hi)
        flag(S3, f'{lab[hi]}: steps', bool(g) and g[0] == ref[0], ref[0], g[0] if g else 'missing')
        judge(S3, f'{lab[hi]}: mean visible kpts (log prints 1 dp)', ref[1], g[1] if g else None, 0.051)
        judge(S3, f'{lab[hi]}: availability %', ref[2], g[2] if g else None, 0.05)
        judge(S3, f'{lab[hi]}: CI low', ref[3], g[3] if g else None, 0.05)
        judge(S3, f'{lab[hi]}: CI high', ref[4], g[4] if g else None, 0.05)
    if len(av) == 9:
        flag('Exp F (abstract)', 'total pooled control steps = 3915', sum(v[0] for v in av.values()) == 3915,
             3915, sum(v[0] for v in av.values()))
        flag('Exp F (text)', 'first bin whose CI excludes 100 % is 5-6 m',
             av[6][4] < 100 and all(av[h][4] >= 100 for h in (50, 10, 8)), '5-6 m',
             f'5-6 m CI high {av[6][4]}; >6 m CI highs {[av[h][4] for h in (50, 10, 8)]}')
        flag('Exp F (text)', 'mean visible keypoints crosses 6 between 3-4 m and 2-3 m bins',
             av[4][1] >= 6 > av[3][1], '>=6 at 3-4 m, <6 at 2-3 m', f'{av[4][1]} / {av[3][1]}')
    dr = {}
    tail = s3.split('trial-level dropout', 1)[1] if 'trial-level dropout' in s3 else ''
    for m in re.finditer(r'^\s*(\d+)–(\d+)\s+(\d+)\s+([\d.]+)\s*$', tail, re.M):
        dr[(int(m.group(1)), int(m.group(2)))] = float(m.group(4))
    for band, ref in {(10, 20): 34.3, (20, 30): 33.3, (30, 50): 32.6}.items():
        judge('Exp F (text)', f'trial-level dropout, initial range {band[0]}-{band[1]} m [%]',
              ref, dr.get(band), 0.05)


def check_repro(outputs, backup):
    """Trial-by-trial comparison of the new Monte Carlo CSVs with the old ones."""
    if not backup or not backup.exists():
        return
    for old in sorted(backup.glob('mc_*.csv')):
        new = outputs / old.name
        if not new.exists():
            flag('Reproducibility', old.name, False, 'present', 'not regenerated')
            continue
        a = list(csv.DictReader(old.open())); b = list(csv.DictReader(new.open()))
        if len(a) != len(b):
            n = min(len(a), len(b))
            same = all(r['seed'] == s['seed'] and r['docked'] == s['docked'] for r, s in zip(a[:n], b[:n]))
            info('Reproducibility', f'{old.name}: old file was incomplete',
                 f'old file had {len(a)} trials', f'rerun has {len(b)} trials',
                 f'first {n} trials {"match" if same else "DIFFER"}; publish the regenerated CSVs '
                 f'(the Data availability statement promises the per-trial records)')
            continue
        dock_diff = sum(r['docked'] != s['docked'] for r, s in zip(a, b))
        seeds_same = all(r['seed'] == s['seed'] for r, s in zip(a, b))
        dmax = 0.0
        for r, s in zip(a, b):
            try:
                dmax = max(dmax, abs(float(r['final_range_m']) - float(s['final_range_m'])))
            except ValueError:
                pass
        ok = seeds_same and dock_diff == 0 and dmax < 1e-6
        v = 'PASS' if ok else ('CLOSE' if seeds_same and dock_diff == 0 and dmax < 1e-2 else 'FAIL')
        ROWS.append((v, 'Reproducibility', f'{old.name}: same trials as before',
                     'identical', f'{dock_diff} docking flips, max |d final range| = {dmax:.2e} m',
                     '' if seeds_same else 'SEEDS DIFFER'))


NOT_COVERED = [
    'Closed-loop 12-seed table (tab:closedloop): docking step 42.7, delta-v 2.562 m/s, nav RMSE 0.345 m, dropout 36.3 %',
    'NIS consistency: mean 6.044, CI [5.887, 6.113], 0.97 % above chi2_0.99; 6.82 / 5.77 ablations',
    'OpenCV cross-checks (factor 1.2 vs SOLVEPNP_EPNP; LM matches cv2 to 4 dp; coplanar subsets 0.0000 deg; 3 % of samples)',
    'MPC reproduces LQR to 4.8e-7; unsaturated LQR command up to 1.77 m/s^2; "ninety times u_max at 30 m"',
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logs', type=pathlib.Path, required=True)
    ap.add_argument('--outputs', type=pathlib.Path, required=True)
    ap.add_argument('--backup', type=pathlib.Path, default=None)
    a = ap.parse_args()

    check_tests(a.logs); check_A(a.logs); check_B(a.logs); check_C(a.logs)
    check_D(a.logs); check_E(a.logs); check_F(a.logs, a.outputs)
    check_repro(a.outputs, a.backup)

    counts = {k: sum(r[0] == k for r in ROWS) for k in ('PASS', 'CLOSE', 'FAIL', 'INFO')}
    print('=' * 100)
    print('CLAIMS CHECK -- "Vision-Based Closed-Loop Spacecraft Rendezvous: Measurement Availability as the Binding Constraint"')
    print('=' * 100)
    print(f"PASS {counts['PASS']}   CLOSE {counts['CLOSE']}   FAIL {counts['FAIL']}   INFO {counts['INFO']}")
    print()
    for want in ('FAIL', 'CLOSE', 'INFO'):
        sel = [r for r in ROWS if r[0] == want]
        if sel:
            print(f'--- {want} ' + '-' * (94 - len(want)))
            for v, sec, cl, p, r, n in sel:
                print(f'  [{sec}] {cl}\n      paper: {p}   rerun: {r}' + (f'   ({n})' if n else ''))
            print()
    print('--- PASS (full list) ' + '-' * 79)
    sec_prev = None
    for v, sec, cl, p, r, n in ROWS:
        if v != 'PASS':
            continue
        if sec != sec_prev:
            print(f'  {sec}'); sec_prev = sec
        print(f'     ok  {cl:<62} paper {p!s:<14} rerun {r}')
    print()
    print('--- NOT checked by this script (numbers come from tests/notebooks, not the experiment scripts) ---')
    for s in NOT_COVERED:
        print('  - ' + s)
    print()
    if counts['FAIL'] == 0 and counts['CLOSE'] == 0:
        print('VERDICT: every checked claim reproduces exactly.')
    elif counts['FAIL'] == 0:
        print('VERDICT: no real disagreements; CLOSE items differ only in the last digit (see above).')
    else:
        print('VERDICT: some claims did NOT reproduce -- see FAIL items above before submitting.')
    return 0


if __name__ == '__main__':
    sys.exit(main())

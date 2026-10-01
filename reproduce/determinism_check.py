#!/usr/bin/env python3
"""
determinism_check.py -- does THIS machine give bit-identical results when the
same seeded code runs twice?  (~1-2 min)

    cd /Volumes/Bishwa/VISION-BASED/vision-satellite-rendezvous
    .venv/bin/python reproduce/determinism_check.py

1. Runs Experiment B twice and compares the printed tables character by character.
2. Re-runs vision+EKF Monte Carlo trials 0-9 one at a time (serially) and
   compares them with the same trials in outputs/mc_vision_+_EKF.csv, which
   reproduce/rerun_all.sh produced with 4 parallel jobs.
Run it after reproduce/rerun_all.sh.  If both are identical, any difference from
the paper is a cross-platform floating-point effect, not randomness in the code.
"""
import csv
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PY = sys.executable

print('1) Experiment B, run twice ...')
outs = []
for k in range(2):
    r = subprocess.run([PY, 'experiments/experiment_B.py'], cwd=ROOT,
                       capture_output=True, text=True, env=dict(os.environ, MPLBACKEND='Agg'))
    outs.append([l for l in r.stdout.splitlines() if not l.startswith('Saved')])
same_b = outs[0] == outs[1]
print('   identical' if same_b else '   DIFFERENT between two runs on the same machine')
if not same_b:
    for a, b in zip(*outs):
        if a != b:
            print('   run1:', a, '\n   run2:', b)

print('2) vision + EKF trials 0-9, serial, vs the parallel rerun CSV ...')
from simulation.montecarlo import standard_configs, run_trial
mc = standard_configs(n_trials=100, base_seed=20260901)['vision + EKF']
mc.base['n_steps'] = 250
ref = list(csv.DictReader(open(ROOT / 'outputs' / 'mc_vision_+_EKF.csv')))
keys = ('final_range_m', 'dock_step', 'n_steps_run', 'dropout_rate', 'delta_v_mps')
mism = 0
for i in range(10):
    rec = run_trial(mc, i)
    diffs = [k for k in keys if repr(float(rec[k])) != repr(float(ref[i][k]))]
    mism += bool(diffs)
    print(f'   trial {i}: ' + ('identical' if not diffs else
          'DIFFERS in ' + ', '.join(f"{k} {ref[i][k]} -> {rec[k]}" for k in diffs)))

print()
if same_b and mism == 0:
    print('RESULT: this machine is fully deterministic. Any gap vs the paper comes from')
    print('        the platform, not from the code.')
else:
    print('RESULT: NOT deterministic on the same machine -- please report it as an issue.')

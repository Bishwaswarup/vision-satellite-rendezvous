"""
test_montecarlo.py
==================
Guard tests for the Monte Carlo campaign driver.

Each test corresponds to a specific way the *previous* Monte Carlo study was
wrong, or to a property the new one claims and must therefore be held to:

  * trials must actually be dispersed (the old study varied only the seed)
  * seeding must be per-trial and order-independent
  * a swept ablation must be paired (same ICs at every level)
  * a proportion's confidence interval must stay inside [0, 1]
  * a crashing trial must be counted as a failure, not silently dropped
"""

import math

import numpy as np
import pytest

from simulation.montecarlo import (
    AVAIL_BINS, CampaignResult, Dispersion, MonteCarloConfig,
    continuous_stats, rate_stats, run_campaign, run_trial, sweep,
    trial_config, trial_seeds, wilson_interval, latex_table,
)

FAST = dict(n_steps=40, u_max=0.3, pos_weight=15.0, vel_weight=1.5,
            use_vision=True, use_ekf=True)


def _mc(n=6, **kw):
    return MonteCarloConfig(n_trials=n, label='test', base=dict(FAST), **kw)


# ── dispersion ───────────────────────────────────────────────────────────────

def test_initial_conditions_actually_vary():
    """
    The defect this module replaces: the old campaign passed a fixed r0, v0
    and w0 to every trial, so only the noise realisation moved.  Every
    dispersed quantity must differ across trials.
    """
    mc = _mc(12)
    cfgs = [trial_config(mc, i)[0] for i in range(mc.n_trials)]

    for attr in ('r0', 'v0', 'q0', 'w0', 'r0_err'):
        vals = np.array([getattr(c, attr) for c in cfgs])
        spread = vals.std(axis=0).max()
        assert spread > 1e-6, f'{attr} is identical across trials (spread {spread:g})'

    # ... and the *ranges* must span a useful interval, not jitter about a point
    rng0 = np.array([np.linalg.norm(c.r0) for c in cfgs])
    assert rng0.max() / rng0.min() > 1.5


def test_dispersion_respects_bounds():
    d = Dispersion(range0=(10.0, 20.0), bearing0_deg=(0.0, 15.0),
                   closing0=(-0.1, -0.05), tumble0=(0.01, 0.05))
    rng = np.random.default_rng(0)
    for _ in range(300):
        s = d.sample(rng)
        rho = np.linalg.norm(s['r0'])
        assert 10.0 - 1e-9 <= rho <= 20.0 + 1e-9
        beta = math.degrees(math.atan2(np.linalg.norm(s['r0'][1:]), s['r0'][0]))
        assert -1e-9 <= beta <= 15.0 + 1e-6
        assert 0.01 - 1e-9 <= np.linalg.norm(s['w0']) <= 0.05 + 1e-9
        assert abs(np.linalg.norm(s['q0']) - 1.0) < 1e-12


def test_random_attitude_is_uniform_on_so3():
    """
    Shoemake's method must give a rotation angle distribution with density
    (1 - cos t)/pi on [0, pi]; a naive 'random axis + uniform angle' draw does
    not, and would bias every tumbling-target statistic in the campaign.
    """
    d = Dispersion(random_attitude=True)
    rng = np.random.default_rng(7)
    ang = []
    for _ in range(4000):
        q = d.sample(rng)['q0']
        ang.append(2.0 * math.acos(min(1.0, abs(float(q[3])))))
    ang = np.array(ang)
    # E[theta] = pi/2 + 2/pi for the Haar measure
    assert abs(ang.mean() - (math.pi / 2 + 2 / math.pi)) < 0.06
    # median solves theta - sin(theta) = pi/2  ->  2.3093 rad
    assert abs(np.median(ang) - 2.3093) < 0.08


def test_frozen_dispersion_reproduces_fixed_ic():
    """Collapsing an interval must freeze that variable exactly."""
    d = Dispersion(range0=(25.0, 25.0), bearing0_deg=(0.0, 0.0),
                   log_uniform_range=False, random_attitude=False,
                   pixel_noise_std=(2.0, 2.0))
    rng = np.random.default_rng(3)
    for _ in range(50):
        s = d.sample(rng)
        assert np.allclose(s['r0'], [25.0, 0.0, 0.0], atol=1e-9)
        assert np.allclose(s['q0'], [0.0, 0.0, 0.0, 1.0])
        assert s['pixel_noise_std'] == pytest.approx(2.0)


# ── seeding ──────────────────────────────────────────────────────────────────

def test_trial_seeding_is_order_independent():
    """
    Trial i must be reproducible in isolation and independent of N.  Seeding
    from a single shared Generator (the obvious implementation) breaks this:
    trial 5 of a 10-trial run would differ from trial 5 of a 200-trial run and
    no result could be replayed.
    """
    small, big = _mc(4), _mc(40)
    for i in range(4):
        c_s, s_s, _ = trial_config(small, i)
        c_b, s_b, _ = trial_config(big, i)
        assert s_s == s_b
        assert np.allclose(c_s.r0, c_b.r0)
        assert np.allclose(c_s.w0, c_b.w0)


def test_trial_seeds_are_distinct():
    mc = _mc(64)
    seeds = [trial_config(mc, i)[1] for i in range(mc.n_trials)]
    assert len(set(seeds)) == len(seeds)
    assert len({tuple(np.round(trial_config(mc, i)[0].r0, 9))
                for i in range(mc.n_trials)}) == mc.n_trials


def test_base_seed_changes_the_campaign():
    a = trial_config(_mc(4, base_seed=1), 0)[0]
    b = trial_config(_mc(4, base_seed=2), 0)[0]
    assert not np.allclose(a.r0, b.r0)


def test_campaign_is_deterministic():
    mc = _mc(4)
    a = run_campaign(mc, progress=False)
    b = run_campaign(mc, progress=False)
    for x, y in zip(a.records, b.records):
        assert x['seed'] == y['seed']
        assert x['final_range_m'] == pytest.approx(y['final_range_m'], abs=1e-12)
        assert x['delta_v_mps'] == pytest.approx(y['delta_v_mps'], abs=1e-12)


def test_base_overrides_win_over_dispersion():
    mc = _mc(3)
    mc.base['pixel_noise_std'] = 0.0
    cfg, _, draw = trial_config(mc, 0)
    assert cfg.pixel_noise_std == 0.0
    assert draw['pixel_noise_std'] > 0.0     # the draw happened, then was overridden


# ── sweep pairing ────────────────────────────────────────────────────────────

def test_sweep_is_paired_across_levels():
    """
    An ablation over pixel noise must reuse the same initial conditions at
    every level; otherwise the level-to-level difference mixes the parameter
    effect with a fresh IC draw and a 20-trial ablation says nothing.
    """
    mc = _mc(5)
    levels = [0.5, 3.0]
    cfgs = {}
    for v in levels:
        sub = MonteCarloConfig(
            n_trials=mc.n_trials, base_seed=mc.base_seed, label='s',
            dispersion=Dispersion(pixel_noise_std=(v, v)), base=dict(FAST))
        cfgs[v] = [trial_config(sub, i)[0] for i in range(mc.n_trials)]
    for i in range(mc.n_trials):
        assert np.allclose(cfgs[levels[0]][i].r0, cfgs[levels[1]][i].r0)
        assert np.allclose(cfgs[levels[0]][i].w0, cfgs[levels[1]][i].w0)
        assert cfgs[levels[0]][i].pixel_noise_std != cfgs[levels[1]][i].pixel_noise_std


def test_sweep_runs_and_labels_levels():
    mc = _mc(2)
    out = sweep(mc, 'pixel_noise_std', [0.5, 2.0], progress=False)
    assert len(out) == 2
    assert '0.5' in out[0].cfg.label and '2.0' in out[1].cfg.label
    for r in out:
        assert len(r.records) == 2


# ── statistics ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize('k,n', [(0, 25), (1, 25), (24, 25), (25, 25), (13, 25)])
def test_wilson_interval_stays_a_probability(k, n):
    """
    The Wald interval at 24/25 reaches 1.03 and at 0/25 has zero width; both
    are invalid statements about a success probability, and both cases occur
    in the noise ablation.
    """
    lo, hi = wilson_interval(k, n)
    assert 0.0 <= lo <= hi <= 1.0
    assert lo <= k / n <= hi
    assert hi > lo                          # never degenerate, even at k in {0, n}


def test_wilson_interval_narrows_with_n():
    w_small = wilson_interval(5, 10)
    w_big = wilson_interval(500, 1000)
    assert (w_big[1] - w_big[0]) < (w_small[1] - w_small[0])


def test_rate_stats_counts():
    s = rate_stats([True, True, False, True])
    assert s['n'] == 4 and s['k'] == 3
    assert s['rate'] == pytest.approx(0.75)


def test_continuous_stats_ignores_nan_and_handles_empty():
    s = continuous_stats([1.0, 2.0, 3.0, float('nan')])
    assert s['n'] == 3 and s['mean'] == pytest.approx(2.0)
    assert s['p05'] <= s['median'] <= s['p95']
    e = continuous_stats([])
    assert math.isnan(e['mean'])


def test_continuous_stats_single_sample_has_zero_std():
    s = continuous_stats([4.2])
    assert s['n'] == 1 and s['std'] == 0.0 and s['sem'] == 0.0


# ── result container ─────────────────────────────────────────────────────────

def test_failed_trial_is_counted_not_dropped():
    """A crashing trial must appear as a non-success, not vanish."""
    recs = [
        {'trial': 0, 'docked': True, 'range_success': True,
         'final_range_m': 0.4, 'delta_v_mps': 2.0, 'error': ''},
        {'trial': 1, 'docked': False, 'range_success': False,
         'final_range_m': float('nan'), 'delta_v_mps': float('nan'),
         'error': 'ValueError: boom'},
    ]
    res = CampaignResult(cfg=_mc(2), records=recs)
    s = res.stats()
    assert res.n_error == 1
    assert s['n_trials'] == 2
    assert s['dock_rate']['rate'] == pytest.approx(0.5)
    # the NaN must not poison the continuous summary
    assert s['final_range_m']['mean'] == pytest.approx(0.4)


def test_delta_v_conditioned_on_success_excludes_failures():
    recs = [
        {'trial': 0, 'docked': True, 'range_success': True,
         'final_range_m': 0.4, 'delta_v_mps': 2.0},
        {'trial': 1, 'docked': False, 'range_success': False,
         'final_range_m': 18.0, 'delta_v_mps': 40.0},
    ]
    s = CampaignResult(cfg=_mc(2), records=recs).stats()
    assert s['delta_v_mps']['mean'] == pytest.approx(21.0)
    assert s['delta_v_docked_mps']['mean'] == pytest.approx(2.0)


def test_records_carry_replay_information():
    """Every record must contain enough to re-run that exact trial."""
    r = run_trial(_mc(1), 0)
    for k in ('trial', 'seed', 'range0_m', 'sigma_px', 'final_range_m',
              'delta_v_mps', 'docked'):
        assert k in r
    mc = _mc(1)
    cfg, seed, _ = trial_config(mc, 0)
    assert r['seed'] == seed


def test_availability_profile_shows_the_fill_knee():
    """
    The physical claim of the paper: availability is a function of the range
    the camera is *at*, saturating at 100 % beyond the fill range
    z_fill = f L / W ~ 6 m and collapsing inside it.  Binning by initial
    range cannot show this (see the next test), so the reducer must pool
    per-step outcomes.
    """
    mc = _mc(8)
    mc.base['n_steps'] = 120
    rows = run_campaign(mc, progress=False).availability_profile()
    assert rows

    far = [r for r in rows if r['range_lo'] >= 10.0]
    near = [r for r in rows if r['range_hi'] <= 2.0]
    assert far and near
    assert min(r['avail'] for r in far) > 0.95
    assert max(r['avail'] for r in near) < 0.6
    # keypoint count must fall monotonically as the target overfills the frame
    assert (max(r['mean_visible'] for r in far)
            > max(r['mean_visible'] for r in near))
    for r in rows:
        assert 0.0 <= r['avail_ci_low'] <= r['avail'] <= r['avail_ci_high'] <= 1.0
        assert r['dropout'] == pytest.approx(1.0 - r['avail'])


def test_availability_profile_pools_every_step():
    mc = _mc(5)
    mc.base['n_steps'] = 80
    res = run_campaign(mc, progress=False)
    rows = res.availability_profile()
    pooled = sum(r['n_steps'] for r in rows)
    assert pooled == sum(r['n_steps_run'] for r in res.records)


def test_availability_profile_empty_without_vision():
    mc = _mc(3)
    mc.base.update(use_vision=False, use_ekf=True)
    res = run_campaign(mc, progress=False)
    assert res.availability_profile() == []


def test_avail_bins_are_increasing():
    assert list(AVAIL_BINS) == sorted(AVAIL_BINS)
    assert AVAIL_BINS[0] == 0.0


def test_dropout_vs_range_bins():
    mc = _mc(8)
    res = run_campaign(mc, progress=False)
    rows = res.dropout_vs_range(edges=(0, 20, 30, 50))
    assert rows
    assert sum(r['n'] for r in rows) == len(res.records)
    for r in rows:
        assert r['range_lo'] < r['range_hi']
        assert 0.0 <= r['dropout_mean'] <= 1.0


def test_to_csv_round_trip(tmp_path):
    import csv
    res = run_campaign(_mc(3), progress=False)
    p = res.to_csv(tmp_path / 'mc.csv')
    rows = list(csv.DictReader(p.open()))
    assert len(rows) == 3
    assert {int(r['trial']) for r in rows} == {0, 1, 2}
    assert float(rows[0]['final_range_m']) == pytest.approx(
        res.records[0]['final_range_m'], rel=1e-6)


def test_report_and_latex_table_render():
    res = run_campaign(_mc(3), progress=False)
    txt = res.report()
    assert 'docking success' in txt and '95% CI' in txt
    tex = latex_table([res])
    assert r'\toprule' in tex and r'\bottomrule' in tex
    assert tex.count(r'\\') >= 2


def test_latex_table_escapes_labels():
    """
    Sweep labels contain the parameter name, e.g. pixel_noise_std=1.5.  An
    unescaped underscore is a LaTeX compile error, so the table must escape it.
    """
    mc = MonteCarloConfig(n_trials=2, label='vision_pipeline & 50%',
                          base=dict(FAST))
    res = run_campaign(mc, progress=False)
    tex = latex_table([res])
    assert r'vision\_pipeline' in tex
    assert r'\&' in tex and r'50\%' in tex


def test_latex_table_reports_a_robust_range_statistic():
    """
    One diverging trial dominates the mean and its standard deviation, so the
    range column must be a median with percentiles, not mean +- std.
    """
    recs = [{'trial': i, 'docked': True, 'range_success': True,
             'final_range_m': 0.4, 'delta_v_mps': 2.0} for i in range(19)]
    recs.append({'trial': 19, 'docked': False, 'range_success': False,
                 'final_range_m': 400.0, 'delta_v_mps': 2.0})
    tex = latex_table([CampaignResult(cfg=_mc(20), records=recs)])
    assert '0.40' in tex
    assert '400' not in tex.split(r'\midrule')[1].split('&')[2]


# ── parallel equivalence ─────────────────────────────────────────────────────

def test_parallel_matches_serial():
    mc = _mc(4)
    a = run_campaign(mc, progress=False)
    b = run_campaign(mc, n_jobs=2, progress=False)
    assert [r['trial'] for r in b.records] == [0, 1, 2, 3]
    for x, y in zip(a.records, b.records):
        assert x['final_range_m'] == pytest.approx(y['final_range_m'], abs=1e-12)

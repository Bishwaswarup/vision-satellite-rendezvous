"""
montecarlo.py
=============
Monte Carlo campaign driver for the closed-loop rendezvous simulator.

Why this module exists
----------------------
The Monte Carlo study originally lived inside ``experiments/experiment_F.py``
and varied **only the RNG seed**: every trial started from the identical
initial position, velocity, tumble rate and attitude, so the "distribution"
it reported was the spread of the measurement-noise realisation alone.  For
the perfect-state configuration (no noise anywhere) all trials were
bit-identical and the violin plot had to be fed 1e-6 of synthetic jitter to
keep it from crashing — a clear symptom that nothing was actually being
dispersed.

A Monte Carlo campaign in the GN&C sense disperses the *initial conditions
and the environment*, not just the noise draw.  This module does that:

    r0   initial relative position   (log-uniform range, uniform bearing)
    v0   initial relative velocity   (uniform drift + closing rate)
    w0   initial tumble rate         (uniform magnitude, isotropic axis)
    q0   initial attitude            (uniform on SO(3), Shoemake's method)
    σ_px pixel-detection noise       (uniform or fixed)
    σ_pos direct-measurement noise   (fixed by default)

Reproducibility
---------------
Every trial's dispersion draw and every trial's simulator seed descend from a
single ``numpy.random.SeedSequence(base_seed)`` via ``spawn()``.  Trial *i* is
therefore reproducible in isolation (``run_trial(cfg, i)``) and independent of
how many other trials were run, of the order they ran in, and of whether the
campaign was executed serially or in parallel.  ``seed`` in each trial record
is the 64-bit integer actually handed to the simulator, so any trial can be
replayed exactly.

Statistics
----------
Continuous metrics are summarised by mean, std, median and the 5th/95th
percentiles.  Success *rates* are proportions, so they get a Wilson score
interval rather than a normal approximation: at N = 25 with 24 successes the
normal interval reaches above 1.0, which is not a valid statement about a
probability.

Usage
-----
    from simulation.montecarlo import MonteCarloConfig, run_campaign

    mc  = MonteCarloConfig(n_trials=200, label='full pipeline')
    res = run_campaign(mc)
    print(res.report())
    res.to_csv('outputs/mc_full.csv')

    # ablation over a single parameter
    from simulation.montecarlo import sweep
    curves = sweep(mc, 'pixel_noise_std', [0.3, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0])
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .runner import SimConfig, SimResult, default_config, run_simulation


# ── Dispersion specification ─────────────────────────────────────────────────

@dataclass
class Dispersion:
    """
    Bounds of the initial-condition and noise dispersion.

    All intervals are closed and sampled uniformly unless noted.  Setting the
    two ends of an interval equal freezes that variable, which is how a
    controlled ablation is expressed (see :func:`sweep`).
    """

    # Initial range along the line of sight [m].  Sampled log-uniformly so the
    # trials are spread evenly in *orders of magnitude* of range rather than
    # piling up at the far end, which is what matters for a study whose
    # headline result is a range-dependent visibility knee.
    range0:        Tuple[float, float] = (15.0, 45.0)
    log_uniform_range: bool = True

    # Bearing of the initial position away from the +x (radial) approach axis
    # [deg].  0 puts the chaser exactly on the approach corridor axis.
    bearing0_deg:  Tuple[float, float] = (0.0, 20.0)

    # Initial closing speed along the line of sight [m/s], negative = closing.
    closing0:      Tuple[float, float] = (-0.15, -0.02)
    # Transverse velocity error per axis [m/s], 1-sigma of a zero-mean normal.
    v0_transverse_std: float = 0.01

    # Tumble rate: magnitude [rad/s] with an isotropically distributed axis.
    tumble0:       Tuple[float, float] = (0.005, 0.09)
    # Uniform random initial attitude on SO(3).  Turn off to keep q0 = identity.
    random_attitude: bool = True

    # Navigation initial error: 1-sigma per axis [m] and [m/s].
    r0_err_std:    float = 1.5
    v0_err_std:    float = 0.02

    # Sensor noise.
    pixel_noise_std: Tuple[float, float] = (1.5, 1.5)
    pos_meas_std:    Tuple[float, float] = (0.5, 0.5)

    def sample(self, rng: np.random.Generator) -> Dict[str, Any]:
        """Draw one set of :class:`~simulation.runner.SimConfig` overrides."""
        # ── position ──
        lo, hi = self.range0
        if self.log_uniform_range and lo > 0.0 and hi > lo:
            rho = float(np.exp(rng.uniform(math.log(lo), math.log(hi))))
        else:
            rho = float(rng.uniform(lo, hi))

        beta = math.radians(float(rng.uniform(*self.bearing0_deg)))
        phi  = float(rng.uniform(0.0, 2.0 * math.pi))
        # Cone of half-angle beta about the +x approach axis.
        u_los = np.array([math.cos(beta),
                          math.sin(beta) * math.cos(phi),
                          math.sin(beta) * math.sin(phi)])
        r0 = rho * u_los

        # ── velocity: closing along the line of sight + transverse error ──
        v_close = float(rng.uniform(*self.closing0))
        v0 = v_close * u_los + rng.normal(0.0, self.v0_transverse_std, 3)

        # ── tumble: uniform magnitude, isotropic axis ──
        axis = rng.normal(0.0, 1.0, 3)
        nrm  = float(np.linalg.norm(axis))
        axis = axis / nrm if nrm > 1e-12 else np.array([0., 0., 1.])
        w0 = float(rng.uniform(*self.tumble0)) * axis

        # ── attitude: uniform on SO(3) (Shoemake 1992), scalar-LAST to match
        #    the estimator/runner convention q = [x, y, z, w] ──
        if self.random_attitude:
            u1, u2, u3 = rng.uniform(0.0, 1.0, 3)
            s1, s2 = math.sqrt(1.0 - u1), math.sqrt(u1)
            q0 = np.array([s1 * math.sin(2 * math.pi * u2),
                           s1 * math.cos(2 * math.pi * u2),
                           s2 * math.sin(2 * math.pi * u3),
                           s2 * math.cos(2 * math.pi * u3)])
            q0 = q0 / np.linalg.norm(q0)
        else:
            q0 = np.array([0., 0., 0., 1.])

        return dict(
            r0=r0, v0=v0, q0=q0, w0=w0,
            r0_err=rng.normal(0.0, self.r0_err_std, 3),
            v0_err=rng.normal(0.0, self.v0_err_std, 3),
            pixel_noise_std=float(rng.uniform(*self.pixel_noise_std)),
            pos_meas_std=float(rng.uniform(*self.pos_meas_std)),
        )


# ── Campaign specification ───────────────────────────────────────────────────

@dataclass
class MonteCarloConfig:
    """One Monte Carlo campaign: N dispersed trials of one system configuration."""

    n_trials:   int = 100
    base_seed:  int = 20260901
    label:      str = 'nominal'
    dispersion: Dispersion = field(default_factory=Dispersion)

    # Fixed SimConfig overrides applied to every trial *after* the dispersion
    # draw, so they win.  This is where use_vision / use_ekf / n_steps live.
    base: Dict[str, Any] = field(default_factory=lambda: dict(
        n_steps=250, u_max=0.3, pos_weight=15.0, vel_weight=1.5,
        use_vision=True, use_ekf=True,
    ))

    # Success is *docking*, i.e. the runner's own range + closing-speed gate.
    # `range_success_m` is a secondary, weaker criterion reported alongside it
    # so a run that ends close but too fast is still distinguishable from one
    # that never arrived.
    range_success_m: float = 2.0


def trial_seeds(mc: MonteCarloConfig) -> List[np.random.SeedSequence]:
    """Independent child seed sequences, one per trial. Order-independent."""
    return np.random.SeedSequence(mc.base_seed).spawn(mc.n_trials)


def trial_config(mc: MonteCarloConfig, i: int) -> Tuple[SimConfig, int, Dict[str, Any]]:
    """
    Build trial *i*'s SimConfig.

    Returns ``(cfg, sim_seed, draw)`` where *draw* is the raw dispersion
    sample (logged into the trial record so the campaign is auditable) and
    *sim_seed* is the integer handed to the simulator's own RNG.
    """
    ss = trial_seeds(mc)[i]
    # Two independent streams from the same child: one for the dispersion
    # draw, one for the simulator.  Splitting them means changing the
    # dispersion spec does not reshuffle the measurement-noise realisations.
    ss_draw, ss_sim = ss.spawn(2)
    rng  = np.random.default_rng(ss_draw)
    draw = mc.dispersion.sample(rng)

    sim_seed = int(ss_sim.generate_state(1, dtype=np.uint32)[0])

    kw = dict(draw)
    kw.update(mc.base)
    return default_config(**kw), sim_seed, draw


# Range bins used for the instantaneous-availability profile [m].  The
# closed-form fill criterion z_fill = f L / W puts the knee at a few metres,
# so the bins are fine there and coarse far away.
AVAIL_BINS = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0,
              20.0, 30.0, 50.0)


def _availability_histogram(res: SimResult) -> tuple:
    """
    Per-step measurement availability binned by *instantaneous* range.

    Binning by initial range says nothing: every trial that closes passes
    through the whole range interval, so a per-trial dropout rate averages
    the easy far field with the hard terminal phase.  The quantity the fill
    criterion predicts is availability as a function of the range the camera
    is actually at, which is a per-step statistic.
    """
    if res.meas_used is None or res.n_steps_run == 0:
        return ([0] * (len(AVAIL_BINS) - 1),) * 3
    rng = res.range_m[:res.n_steps_run]
    used = np.asarray(res.meas_used[:res.n_steps_run], dtype=bool)
    vis = (np.asarray(res.n_visible[:res.n_steps_run], dtype=float)
           if res.n_visible is not None else np.zeros_like(rng))
    idx = np.digitize(rng, AVAIL_BINS) - 1
    nb = len(AVAIL_BINS) - 1
    n_steps, n_used, n_vis = [0] * nb, [0] * nb, [0.0] * nb
    for b in range(nb):
        m = idx == b
        if not m.any():
            continue
        n_steps[b] = int(m.sum())
        n_used[b] = int(used[m].sum())
        n_vis[b] = float(vis[m].sum())
    return n_steps, n_used, n_vis


def _record(mc: MonteCarloConfig, i: int, cfg: SimConfig, seed: int,
            draw: Dict[str, Any], res: SimResult, wall: float) -> Dict[str, Any]:
    r0 = np.asarray(draw['r0'], dtype=float)
    rec = {
        'trial':        i,
        'seed':         seed,
        'range0_m':     float(np.linalg.norm(r0)),
        'bearing0_deg': float(math.degrees(math.atan2(
                            float(np.linalg.norm(r0[1:])), float(r0[0])))),
        'closing0_mps': float(np.dot(draw['v0'], r0) / max(np.linalg.norm(r0), 1e-12)),
        'tumble0_dps':  float(np.degrees(np.linalg.norm(draw['w0']))),
        'r0_err_m':     float(np.linalg.norm(draw['r0_err'])),
        'sigma_px':     float(draw['pixel_noise_std']),
        'final_range_m': float(res.range_m[-1]),
        'min_range_m':  float(res.range_m.min()),
        'delta_v_mps':  float(res.delta_v),
        'delta_v_to_dock_mps': float(res.delta_v_to_dock),
        'dock_step':    -1 if res.dock_step is None else int(res.dock_step),
        'docked':       bool(res.dock_step is not None),
        'range_success': bool(res.range_m[-1] < mc.range_success_m),
        'n_steps_run':  int(res.n_steps_run),
        'dropout_rate': float(res.vision_dropout_rate),
        'n_vision_fail': int(res.n_vision_fail),
        'mean_pos_err_m': float(np.mean(res.pos_error)),
        'final_pos_err_m': float(res.pos_error[-1]),
        'wall_s':       float(wall),
        'error':        '',
    }
    if cfg.use_vision:
        h_steps, h_used, h_vis = _availability_histogram(res)
        rec['avail_steps'] = json.dumps(h_steps)
        rec['avail_used'] = json.dumps(h_used)
        rec['avail_visible'] = json.dumps(h_vis)
    if res.n_visible is not None and len(res.n_visible):
        rec['mean_n_visible'] = float(np.mean(res.n_visible))
    else:
        rec['mean_n_visible'] = float('nan')
    return rec


def run_trial(mc: MonteCarloConfig, i: int) -> Dict[str, Any]:
    """
    Run one dispersed trial and return a flat record.

    A trial that raises is recorded with ``error`` set rather than aborting the
    campaign: a Monte Carlo study whose failures silently vanish reports a
    success rate conditioned on not crashing, which is not the quantity of
    interest.
    """
    cfg, seed, draw = trial_config(mc, i)
    t0 = time.perf_counter()
    try:
        res = run_simulation(cfg, rng_seed=seed)
    except Exception as exc:                      # noqa: BLE001 — see docstring
        return {
            'trial': i, 'seed': seed,
            'range0_m': float(np.linalg.norm(draw['r0'])),
            'sigma_px': float(draw['pixel_noise_std']),
            'docked': False, 'range_success': False,
            'final_range_m': float('nan'), 'delta_v_mps': float('nan'),
            'wall_s': time.perf_counter() - t0,
            'error': f'{type(exc).__name__}: {exc}',
        }
    return _record(mc, i, cfg, seed, draw, res, time.perf_counter() - t0)


# ── Statistics ───────────────────────────────────────────────────────────────

def continuous_stats(x: Sequence[float]) -> Dict[str, float]:
    """Mean / std / median / 5th–95th percentile of a finite sample."""
    a = np.asarray([v for v in x if np.isfinite(v)], dtype=float)
    if a.size == 0:
        return {k: float('nan') for k in
                ('n', 'mean', 'std', 'median', 'p05', 'p95', 'min', 'max', 'sem')}
    return {
        'n':      int(a.size),
        'mean':   float(a.mean()),
        'std':    float(a.std(ddof=1)) if a.size > 1 else 0.0,
        'sem':    float(a.std(ddof=1) / math.sqrt(a.size)) if a.size > 1 else 0.0,
        'median': float(np.median(a)),
        'p05':    float(np.percentile(a, 5)),
        'p95':    float(np.percentile(a, 95)),
        'min':    float(a.min()),
        'max':    float(a.max()),
    }


def wilson_interval(k: int, n: int, z: float = 1.959963985) -> Tuple[float, float]:
    """
    Wilson score interval for a binomial proportion.

    Preferred over the normal (Wald) interval because it stays inside [0, 1]
    and does not collapse to zero width at k = 0 or k = n — both of which
    occur in this study (100 % docking success at low pixel noise, 0 % at
    4 px).
    """
    if n <= 0:
        return (float('nan'), float('nan'))
    p  = k / n
    d  = 1.0 + z * z / n
    c  = p + z * z / (2 * n)
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    lo, hi = (c - hw) / d, (c + hw) / d
    # Clamp to [0, 1] and to either side of p: at k = 0 and k = n the
    # rounding of (c -+ hw)/d can land a hair inside the point estimate, and
    # an "interval" that excludes its own estimate breaks every consumer.
    return (min(max(0.0, lo), p), max(min(1.0, hi), p))


def rate_stats(flags: Sequence[bool]) -> Dict[str, float]:
    """Proportion with a Wilson 95 % interval."""
    f = [bool(v) for v in flags]
    n, k = len(f), int(sum(f))
    lo, hi = wilson_interval(k, n)
    return {'n': n, 'k': k,
            'rate': (k / n) if n else float('nan'),
            'ci_low': lo, 'ci_high': hi}


# ── Campaign result ──────────────────────────────────────────────────────────

CONTINUOUS_METRICS = ('final_range_m', 'min_range_m', 'delta_v_mps',
                      'delta_v_to_dock_mps', 'dropout_rate',
                      'mean_pos_err_m', 'final_pos_err_m', 'mean_n_visible')


@dataclass
class CampaignResult:
    cfg:      MonteCarloConfig
    records:  List[Dict[str, Any]]
    wall_s:   float = 0.0

    # ── accessors ──
    def column(self, key: str) -> np.ndarray:
        return np.array([r.get(key, float('nan')) for r in self.records], dtype=float)

    @property
    def n_error(self) -> int:
        return sum(1 for r in self.records if r.get('error'))

    @property
    def docked(self) -> np.ndarray:
        return np.array([bool(r.get('docked')) for r in self.records])

    def stats(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            'label':      self.cfg.label,
            'n_trials':   len(self.records),
            'n_error':    self.n_error,
            'wall_s':     self.wall_s,
            'dock_rate':  rate_stats([r.get('docked', False) for r in self.records]),
            'range_rate': rate_stats([r.get('range_success', False) for r in self.records]),
        }
        for m in CONTINUOUS_METRICS:
            out[m] = continuous_stats(self.column(m))
        # Propellant conditioned on success: the mean Δv over *all* trials
        # mixes successful approaches with runs that gave up, so the
        # unconditional mean is not a propellant budget.
        dv_ok = [r['delta_v_mps'] for r in self.records if r.get('docked')]
        out['delta_v_docked_mps'] = continuous_stats(dv_ok)
        return out

    def report(self) -> str:
        s = self.stats()
        d, rr = s['dock_rate'], s['range_rate']
        L = [
            f"Monte Carlo — {s['label']}   N = {s['n_trials']}"
            f"   ({s['wall_s']:.1f} s wall"
            + (f", {s['n_error']} errored)" if s['n_error'] else ")"),
            "-" * 72,
            f"  docking success   {d['rate']*100:6.1f} %   "
            f"95% CI [{d['ci_low']*100:.1f}, {d['ci_high']*100:.1f}]   "
            f"({d['k']}/{d['n']})",
            f"  within {self.cfg.range_success_m:.1f} m      "
            f"{rr['rate']*100:6.1f} %   "
            f"95% CI [{rr['ci_low']*100:.1f}, {rr['ci_high']*100:.1f}]",
            "",
            f"  {'metric':<24}{'mean':>10}{'std':>10}{'median':>10}"
            f"{'p05':>10}{'p95':>10}",
        ]
        for m in ('final_range_m', 'min_range_m', 'delta_v_mps',
                  'dropout_rate', 'mean_pos_err_m', 'mean_n_visible'):
            c = s[m]
            L.append(f"  {m:<24}{c['mean']:>10.3f}{c['std']:>10.3f}"
                     f"{c['median']:>10.3f}{c['p05']:>10.3f}{c['p95']:>10.3f}")
        dvk = s['delta_v_docked_mps']
        L.append(f"  {'delta_v | docked':<24}{dvk['mean']:>10.3f}{dvk['std']:>10.3f}"
                 f"{dvk['median']:>10.3f}{dvk['p05']:>10.3f}{dvk['p95']:>10.3f}")
        return "\n".join(L)

    # ── export ──
    def to_csv(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        keys: List[str] = []
        for r in self.records:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with path.open('w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in self.records:
                w.writerow(r)
        return path

    # ── reducers ──
    def availability_profile(self) -> List[Dict[str, Any]]:
        """
        Measurement availability as a function of *instantaneous* range,
        pooled over every step of every trial in the campaign.

        This is the campaign-level counterpart of the closed-form fill
        criterion z_fill = f L / W.  Pooling per-step outcomes rather than
        per-trial averages is what makes the knee visible: a trial-level
        dropout rate mixes the far field, where the target is a handful of
        pixels across and every keypoint projects inside the frame, with the
        terminal phase, where the target overfills the field of view.
        """
        nb = len(AVAIL_BINS) - 1
        steps = np.zeros(nb)
        used = np.zeros(nb)
        vis = np.zeros(nb)
        for r in self.records:
            if 'avail_steps' not in r:
                continue
            steps += np.asarray(json.loads(r['avail_steps']), dtype=float)
            used += np.asarray(json.loads(r['avail_used']), dtype=float)
            vis += np.asarray(json.loads(r['avail_visible']), dtype=float)
        rows = []
        for b in range(nb):
            if steps[b] == 0:
                continue
            k = int(used[b])
            n = int(steps[b])
            lo, hi = wilson_interval(k, n)
            rows.append({
                'range_lo': float(AVAIL_BINS[b]),
                'range_hi': float(AVAIL_BINS[b + 1]),
                'n_steps': n,
                'avail': k / n,
                'avail_ci_low': lo,
                'avail_ci_high': hi,
                'dropout': 1.0 - k / n,
                'mean_visible': float(vis[b] / n),
            })
        return rows

    def dropout_vs_range(self, edges: Sequence[float] = (0, 5, 10, 20, 30, 50)
                         ) -> List[Dict[str, Any]]:
        """
        Trial-level dropout rate binned by *initial* range.

        Retained as a coarse check that the dispersion covers the range
        interval; :meth:`availability_profile` is the reducer that carries
        the physical result.
        """
        rho = self.column('range0_m')
        drp = self.column('dropout_rate')
        dok = self.docked
        rows = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (rho >= lo) & (rho < hi)
            if not m.any():
                continue
            rows.append({
                'range_lo': float(lo), 'range_hi': float(hi),
                'n': int(m.sum()),
                'dropout_mean': float(np.nanmean(drp[m])),
                'dock_rate': float(dok[m].mean()),
                **{f'dock_{k}': v for k, v in
                   rate_stats(list(dok[m])).items() if k.startswith('ci')},
            })
        return rows


# ── Campaign driver ──────────────────────────────────────────────────────────

def run_campaign(mc: MonteCarloConfig,
                 n_jobs: int = 1,
                 progress: bool = True,
                 progress_every: int = 10) -> CampaignResult:
    """
    Run all *n_trials* of a campaign.

    ``n_jobs > 1`` uses a process pool.  Results are re-sorted by trial index
    afterwards, and because each trial's seeding is derived independently
    (see :func:`trial_config`) the numbers are identical to the serial run —
    a property the test suite checks rather than assumes.
    """
    t0 = time.perf_counter()
    n  = mc.n_trials

    if n_jobs and n_jobs > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        records: List[Dict[str, Any]] = []
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futs = {ex.submit(run_trial, mc, i): i for i in range(n)}
            for done, fut in enumerate(as_completed(futs), 1):
                records.append(fut.result())
                if progress and done % progress_every == 0:
                    print(f"    [{mc.label}] {done}/{n}", flush=True)
        records.sort(key=lambda r: r['trial'])
    else:
        records = []
        for i in range(n):
            records.append(run_trial(mc, i))
            if progress and (i + 1) % progress_every == 0:
                ok = sum(1 for r in records if r.get('docked'))
                print(f"    [{mc.label}] {i+1}/{n}  docked {ok}", flush=True)

    return CampaignResult(cfg=mc, records=records,
                          wall_s=time.perf_counter() - t0)


def sweep(mc: MonteCarloConfig,
          param: str,
          values: Sequence[Any],
          n_jobs: int = 1,
          progress: bool = True) -> List[CampaignResult]:
    """
    Repeat a campaign while varying one parameter, holding the dispersion draw
    fixed across levels.

    The *same* seed sequence is used at every level, so trial *i* has the same
    initial position, velocity, tumble and attitude at every value of *param*.
    The ablation therefore measures the effect of the parameter rather than the
    effect of the parameter plus a fresh set of initial conditions — a paired
    comparison, which is what makes a 20-trial ablation informative at all.

    Dispersion-interval parameters (``pixel_noise_std``, ``pos_meas_std``) are
    frozen to the swept value; anything else is set as a SimConfig override.
    """
    out: List[CampaignResult] = []
    for v in values:
        disp = mc.dispersion
        base = dict(mc.base)
        if param in ('pixel_noise_std', 'pos_meas_std'):
            disp = replace(disp, **{param: (float(v), float(v))})
        else:
            base[param] = v
        sub = MonteCarloConfig(
            n_trials=mc.n_trials, base_seed=mc.base_seed,
            label=f'{mc.label} · {param}={v}',
            dispersion=disp, base=base,
            range_success_m=mc.range_success_m,
        )
        if progress:
            print(f"  {param} = {v}", flush=True)
        res = run_campaign(sub, n_jobs=n_jobs, progress=progress)
        if progress:
            s = res.stats()
            print(f"    → range {s['final_range_m']['mean']:.2f} "
                  f"± {s['final_range_m']['std']:.2f} m  |  "
                  f"dock {s['dock_rate']['rate']*100:.0f} %  |  "
                  f"dropout {s['dropout_rate']['mean']*100:.1f} %", flush=True)
        out.append(res)
    return out


# ── Ready-made configurations used by the paper ──────────────────────────────

def standard_configs(n_trials: int = 100,
                     base_seed: int = 20260901) -> Dict[str, MonteCarloConfig]:
    """
    The three system configurations compared in the paper, sharing one seed
    sequence so the comparison is paired trial-by-trial.

    'perfect state' has no sensor noise, but — unlike the original study — its
    trials are still dispersed in initial condition, so it has a genuine
    distribution rather than 25 copies of one run.
    """
    common = dict(n_steps=250, u_max=0.3, pos_weight=15.0, vel_weight=1.5)
    disp   = Dispersion()

    return {
        'perfect state': MonteCarloConfig(
            n_trials=n_trials, base_seed=base_seed, label='perfect state',
            dispersion=replace(disp, r0_err_std=0.0, v0_err_std=0.0,
                               pixel_noise_std=(0.0, 0.0),
                               pos_meas_std=(0.0, 0.0)),
            base=dict(common, use_vision=False, use_ekf=False),
        ),
        'EKF, direct': MonteCarloConfig(
            n_trials=n_trials, base_seed=base_seed, label='EKF, direct',
            dispersion=replace(disp, pixel_noise_std=(0.0, 0.0)),
            base=dict(common, use_vision=False, use_ekf=True),
        ),
        'vision + EKF': MonteCarloConfig(
            n_trials=n_trials, base_seed=base_seed, label='vision + EKF',
            dispersion=disp,
            base=dict(common, use_vision=True, use_ekf=True),
        ),
    }


def _tex(s: str) -> str:
    """Escape the LaTeX-special characters that appear in campaign labels."""
    for a, b in (('\\', r'\textbackslash{}'), ('_', r'\_'), ('&', r'\&'),
                 ('%', r'\%'), ('#', r'\#'), ('$', r'\$'), ('·', r'$\cdot$')):
        s = s.replace(a, b)
    return s


def latex_table(results: Sequence[CampaignResult],
                caption: str = 'Monte Carlo campaign summary',
                label: str = 'tab:montecarlo') -> str:
    """Emit a booktabs table of campaign summaries, ready to \\input."""
    L = [r'\begin{table}[t]', r'\centering',
         r'\caption{' + caption + '}', r'\label{' + label + '}',
         r'\begin{tabular}{lrrrr}', r'\toprule',
         r'Configuration & $N$ & Final range [m] & $\Delta v$ [m/s] '
         r'& Docking [\%] \\', r'\midrule']
    for r in results:
        s = r.stats()
        fr, dv, d = s['final_range_m'], s['delta_v_mps'], s['dock_rate']
        # Final range is reported as median [p05, p95]: a single diverging
        # trial dominates the mean and its standard deviation, so mean +- std
        # would misdescribe a distribution that is tight with a rare tail.
        L.append(
            f"{_tex(s['label'])} & {s['n_trials']} & "
            f"${fr['median']:.2f}$ [{fr['p05']:.2f},\\,{fr['p95']:.2f}] & "
            f"${dv['mean']:.2f} \\pm {dv['std']:.2f}$ & "
            f"${d['rate']*100:.0f}$ "
            f"[{d['ci_low']*100:.0f},\\,{d['ci_high']*100:.0f}] \\\\")
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return "\n".join(L)


__all__ = [
    'Dispersion', 'MonteCarloConfig', 'CampaignResult',
    'trial_seeds', 'trial_config', 'run_trial', 'run_campaign', 'sweep',
    'AVAIL_BINS',
    'continuous_stats', 'rate_stats', 'wilson_interval',
    'standard_configs', 'latex_table',
]

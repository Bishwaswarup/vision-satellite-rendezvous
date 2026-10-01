#!/usr/bin/env python3
"""
verify_extra_claims.py -- recompute the paper's claims that the six experiment
scripts do not print, and compare each with the value quoted in the paper.

    .venv/bin/pip install opencv-python-headless     # once; for section C only
    .venv/bin/python reproduce/verify_extra_claims.py [--jobs 6]

Takes about 3-6 minutes.  Writes reproduce/extra_claims_report.txt.

  A  Closed-loop performance over 12 seeds (Table "closedloop")
  B  MEKF consistency: NIS 6.044 over 3600 updates, plus the two ablations
  C  EPnP / LM cross-checks against OpenCV, coplanar subsets, beta refinement
  D  Unconstrained MPC reproduces LQR (1.1e-11); unsaturated LQR demand 1.77 m/s^2
  E  Fixed-trajectory noise ablation: 65 of 100 dock at 4 px, std 60 m
  F  Untuned LQR: ~90x u_max at 30 m, saturation, fly-through, factor 16 / 3

Where the paper does not state a setting (seed list, standard-deviation
convention), the choice made here is printed next to the result.
"""
import argparse
import itertools
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulation.runner import SimConfig, RendezvousSimulator, default_config, run_simulation  # noqa: E402

LINES = []
TALLY = {'PASS': 0, 'DIFFERS': 0, 'INFO': 0}


def out(s=''):
    print(s, flush=True)
    LINES.append(s)


def verdict(ok, claim, paper, got, note=''):
    v = 'PASS' if ok else 'DIFFERS'
    TALLY[v] += 1
    out(f'  [{v:7}] {claim}\n            paper: {paper}\n            rerun: {got}' + (f'\n            note : {note}' if note else ''))


def info(claim, got, note=''):
    TALLY['INFO'] += 1
    out(f'  [INFO   ] {claim}: {got}' + (f'   ({note})' if note else ''))


def close(a, b, half_unit):
    return abs(a - b) <= half_unit + 1e-12


# ─────────────────────────────────────────────────────────────────────────────
# A. closed-loop 12 seeds
# ─────────────────────────────────────────────────────────────────────────────
def _closedloop_trial(seed):
    cfg = SimConfig()
    res = RendezvousSimulator(cfg, rng_seed=seed).run()
    speeds = np.linalg.norm(res.v_true, axis=1)
    sat = int((np.abs(res.controls).max(axis=1) >= cfg.u_max - 1e-9).sum())
    return dict(dock=res.dock_step if res.dock_step is not None else np.nan,
                dv=res.delta_v_to_dock,
                nav=float(np.sqrt(np.mean(res.pos_error ** 2))),
                term=float(res.range_m[-1]),
                vpk=float(speeds.max()),
                drop=100 * res.vision_dropout_rate,
                sat=sat)


def section_A(pool):
    out('\nA. Closed-loop performance, 12 seeds (SimConfig defaults: 30 m, w0 = [0.02, 0.05, 0.01] rad/s)')
    out('   seeds 0-11 assumed (the paper does not list them)')
    rows = list(pool.map(_closedloop_trial, range(12)))
    paper = {'dock': ('Docking step', 42.8, 0.5, 0.05, 0.05),
             'dv': ('Delta-v to docking [m/s]', 2.567, 0.050, 0.0005, 0.0005),
             'nav': ('Navigation RMSE [m]', 0.343, 0.013, 0.0005, 0.0005),
             'term': ('Terminal range [m]', 0.450, 0.185, 0.0005, 0.0005),
             'vpk': ('Peak closing speed [m/s]', 1.334, 0.024, 0.0005, 0.0005),
             'drop': ('Vision dropout [%]', 36.6, 0.9, 0.05, 0.05),
             'sat': ('Thrust-saturated steps', 0, 0, 0.5, 0.5)}
    docked = sum(np.isfinite(r['dock']) for r in rows)
    verdict(docked == 12, 'all 12 seeds dock', '12/12', f'{docked}/12')
    for k, (name, pm, ps, hm, hs) in paper.items():
        x = np.array([r[k] for r in rows], dtype=float)
        m, s0, s1 = np.nanmean(x), np.nanstd(x), np.nanstd(x, ddof=1)
        ok_m = close(m, pm, hm)
        ok_s = close(s1, ps, hs)          # paper: sample standard deviation
        verdict(ok_m and ok_s, name, f'{pm} +/- {ps}',
                f'{m:.4g} +/- {s0:.3g} (ddof=0) / {s1:.3g} (ddof=1)')


# ─────────────────────────────────────────────────────────────────────────────
# B. NIS
# ─────────────────────────────────────────────────────────────────────────────
def _nis_run(args):
    variant, seed = args
    import estimator.ekf as E
    import estimator.state as S
    orig_resid, orig_update = E.attitude_residual, E.MultEKF.update
    if variant == 'global':
        def global_residual(rv_meas, q_est):
            return np.asarray(rv_meas, float) - S.quat_to_rotvec(q_est)
        E.attitude_residual = global_residual
    nis = []

    def patched(self, z, R_override=None):
        inf = orig_update(self, z, R_override)
        if np.isfinite(inf['mahal']):
            nis.append(float(inf['mahal']))
        return inf
    E.MultEKF.update = patched
    kw = dict(use_vision=False, n_steps=300, stop_at_dock=False)
    if variant == 'no_disturbance':
        kw['disturb_accel_std'] = 0.0
    try:
        RendezvousSimulator(SimConfig(**kw), rng_seed=seed).run()
    finally:            # pool workers are reused: always undo the patches
        E.MultEKF.update, E.attitude_residual = orig_update, orig_resid
    return nis


def section_B(pool):
    from estimator.state import nis_statistics
    out('\nB. MEKF consistency (direct measurements, 12 seeds x 300 steps, seeds 0-11)')
    res = {}
    for variant in ('baseline', 'global', 'no_disturbance'):
        nis = [v for chunk in pool.map(_nis_run, [(variant, s) for s in range(12)]) for v in chunk]
        res[variant] = nis_statistics(nis, dof=6)
    b = res['baseline']
    verdict(b['n'] == 3600, 'number of updates', 3600, b['n'])
    verdict(close(b['mean'], 6.044, 0.0005), 'mean NIS', 6.044, f"{b['mean']:.4f}")
    verdict(close(b['ci_low'], 5.887, 0.0005) and close(b['ci_high'], 6.113, 0.0005),
            '95 % CI on the mean', '[5.887, 6.113]', f"[{b['ci_low']:.3f}, {b['ci_high']:.3f}]")
    verdict(close(100 * b['tail_fraction'], 0.97, 0.005), 'fraction above chi2_6,0.99 = 16.81 [%]',
            0.97, f"{100*b['tail_fraction']:.2f}")
    verdict(bool(b['consistent']), 'filter consistent (mean inside CI)', True, bool(b['consistent']))
    g, nd = res['global'], res['no_disturbance']
    verdict(close(g['mean'], 359, 0.5), 'ablation: global rotation-vector residual -> mean NIS', 359,
            f"{g['mean']:.3f}", 'residual replaced by rv_meas - rotvec(q_est); everything else as the final filter')
    verdict(close(nd['mean'], 5.94, 0.0051), 'ablation: no truth disturbance -> mean NIS', 5.94,
            f"{nd['mean']:.3f}", 'disturb_accel_std = 0')


# ─────────────────────────────────────────────────────────────────────────────
# C. EPnP / LM cross-checks
# ─────────────────────────────────────────────────────────────────────────────
def _rot(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
                     [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]])


def _rerr(A, B):
    return float(np.degrees(np.arccos(np.clip((np.trace(A.T @ B) - 1) / 2, -1, 1))))


def _proj(R, t, P):
    Pc = (R @ P.T).T + t
    return np.stack([800*Pc[:, 0]/Pc[:, 2] + 512, 800*Pc[:, 1]/Pc[:, 2] + 512], axis=-1)


K800 = np.array([[800., 0., 512.], [0., 800., 512.], [0., 0., 1.]])


def _epnp_vs_cv2(cv2, refine_betas=True):
    from pose import EPnPSolver
    saved = EPnPSolver._gauss_newton_betas
    if not refine_betas:
        EPnPSolver._gauss_newton_betas = staticmethod(lambda betas, V_null, ctrl, n_iter=10: betas)
    try:
        solver = EPnPSolver()
        rng = np.random.default_rng(4)          # same scene set as tests/test_phase4.py
        ours, ref = [], []
        for _ in range(60):
            P = rng.uniform(-3, 3, (20, 3))
            q = rng.normal(size=4); q /= np.linalg.norm(q)
            R = _rot(q); t = np.array([0., 0., 25.])
            uv = _proj(R, t, P) + rng.normal(0, 1.0, (20, 2))
            R_est, _, _, ok = solver.solve(P, uv, K800)
            if ok:
                ours.append(_rerr(R, R_est))
            if cv2 is not None:
                _, rvec, _ = cv2.solvePnP(P, uv, K800, None, flags=cv2.SOLVEPNP_EPNP)
                ref.append(_rerr(R, cv2.Rodrigues(rvec)[0]))
    finally:
        EPnPSolver._gauss_newton_betas = saved
    return np.median(ours), (np.median(ref) if ref else float('nan'))


def section_C():
    out('\nC. EPnP / LM cross-checks')
    try:
        import cv2
    except ImportError:
        cv2 = None
        out('  OpenCV not installed -> the OpenCV comparisons are skipped.')
        out('  Install it into the venv and rerun:  .venv/bin/pip install opencv-python-headless')
    from pose import EPnPSolver, refine_pose
    from vision.body_model import ariane_model
    from vision.renderer import look_at_rotation

    ours, ref = _epnp_vs_cv2(cv2, True)
    ours_nob, _ = _epnp_vs_cv2(cv2, False)
    if cv2 is not None:
        r = ours / ref
        verdict(r <= 1.45, 'EPnP vs cv2.SOLVEPNP_EPNP, median rotation error ratio (sigma = 1 px)',
                'within a factor of 1.4 (0.53 vs 0.39 deg)', f'{r:.2f}x  ({ours:.4f} vs {ref:.4f} deg)')
    f = ours_nob / ours
    verdict(3.35 <= f < 3.45, 'omitting the beta Gauss-Newton refinement degrades raw EPnP',
            'a factor of about 3.4', f'{f:.2f}x  ({ours_nob:.4f} vs {ours:.4f} deg median rotation error)')

    if cv2 is not None:
        kp = ariane_model().keypoint_array
        R_gt, _ = look_at_rotation(np.array([0., 0., 30.]))
        t_gt = np.array([0., 0., 30.])
        worst_r, worst_t = 0.0, 0.0
        for sigma in (0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0):
            rng = np.random.default_rng(5)
            for _ in range(20):
                uv = _proj(R_gt, t_gt, kp) + rng.normal(0, sigma, (len(kp), 2))
                R_o, t_o, _ = refine_pose(R_gt, t_gt, kp, uv, K800)
                rvec, _ = cv2.Rodrigues(R_gt)
                rv, tv = cv2.solvePnPRefineLM(kp, uv, K800, None, rvec.copy(), t_gt.reshape(3, 1).copy())
                worst_r = max(worst_r, _rerr(cv2.Rodrigues(rv)[0], R_o))
                worst_t = max(worst_t, float(np.linalg.norm(t_o - tv.ravel())))
        verdict(worst_r < 0.002 and worst_t < 3e-5,
                'LM refiner agrees with cv2.solvePnPRefineLM at every noise level',
                'within 0.002 deg and 0.03 mm', f'max difference {worst_r:.2e} deg, {worst_t:.2e} m '
                f'over 8 noise levels x 20 trials (30 m, Ariane keypoints)')

    solver = EPnPSolver()
    worst, cv_err = 0.0, []
    for N in (6, 10, 20):
        rng = np.random.default_rng(N)
        for _ in range(25):
            P = rng.uniform(-3, 3, (N, 3)); P[:, 2] = 0.0
            q = rng.normal(size=4); q /= np.linalg.norm(q)
            R = _rot(q); t = np.array([0., 0., 25.])
            uv = _proj(R, t, P)
            R_est, _, _, ok = solver.solve(P, uv, K800)
            worst = max(worst, _rerr(R, R_est) if ok else 180.0)
            if cv2 is not None:
                okc, rvec, _ = cv2.solvePnP(P, uv, K800, None, flags=cv2.SOLVEPNP_EPNP)
                cv_err.append(_rerr(R, cv2.Rodrigues(rvec)[0]) if okc else 180.0)
    verdict(worst < 5e-5, 'exactly coplanar point sets solved (noiseless, N = 6/10/20)',
            '0.0000 deg', f'max rotation error {worst:.2e} deg')
    if cv_err:
        info('cv2.SOLVEPNP_EPNP on the same coplanar sets',
             f'median {np.median(cv_err):.3f} deg, max {np.max(cv_err):.1f} deg',
             'paper: "a case the OpenCV EPnP implementation does not handle"')

    kp = ariane_model().keypoint_array
    combos = list(itertools.combinations(range(len(kp)), 6))
    planar = sum(solver._choose_control_points(kp[list(c)]).shape[0] == 3 for c in combos)
    frac = 100 * planar / len(combos)
    verdict(planar == 172, 'share of six-point keypoint samples that are exactly coplanar',
            '3.4 % (172 of 5005)', f'{frac:.2f} %  ({planar} of all {len(combos)} subsets of {len(kp)} keypoints)')


# ─────────────────────────────────────────────────────────────────────────────
# D. MPC vs LQR
# ─────────────────────────────────────────────────────────────────────────────
def section_D():
    from controller.lqr import make_lqr
    from controller.mpc import make_mpc
    out('\nD. MPC vs LQR (Experiment E setup: 50 m along-track, dt = 10 s, N = 20, Q = diag(10,10,10,1,1,1), R = I)')
    x0 = np.array([0.0, 50.0, 0.0, 0.0, 0.0, 0.0])
    lqr = make_lqr(dt=10.0, pos_weight=10.0, vel_weight=1.0, thrust_weight=1.0, u_max=None)
    sim = lqr.simulate(x0, n_steps=200)
    U = sim['controls']
    peak_axis, peak_norm = np.abs(U).max(), np.linalg.norm(U, axis=1).max()
    verdict(close(peak_axis, 1.77, 0.005) or close(peak_norm, 1.77, 0.005),
            'peak unsaturated LQR demand [m/s^2]', '1.77 (seventeen times the 0.1 limit)',
            f'{peak_axis:.3f} per axis, {peak_norm:.3f} Euclidean  ({peak_axis/0.1:.1f}x / {peak_norm/0.1:.1f}x the limit)')
    mpc = make_mpc(dt=10.0, N=20, pos_weight=10.0, vel_weight=1.0, thrust_weight=1.0, u_max=1e6, cone_half_angle=None)
    exact = max(np.abs(np.linalg.solve(mpc._H, -mpc._SuTQbarSx @ x)[:3] + lqr.K @ x).max()
                for x in sim['states'][:60])
    verdict(exact < 1e-10, 'unconstrained MPC (QP solved exactly) reproduces LQR', '1.1e-11 m/s^2',
            f'max |u_QP - u_LQR| = {exact:.2e} m/s^2 over 60 states along the LQR path')
    ctl = max(np.abs(mpc.control(x) - (-lqr.K @ x)).max() for x in sim['states'][:60])
    verdict(ctl < 1e-9, 'the controller itself (BVLS) reproduces LQR when unconstrained', '< 1e-9 m/s^2',
            f'max {ctl:.2e} m/s^2')


# ─────────────────────────────────────────────────────────────────────────────
# E. fixed-trajectory ablation (old Experiment F, git 96657f5)
# ─────────────────────────────────────────────────────────────────────────────
def _fixed_trial(seed):
    cfg = default_config(n_steps=250, u_max=0.3, pos_weight=15.0, vel_weight=1.5,
                         r0=np.array([30., 3., -1.5]), v0=np.array([-0.08, 0., 0.]),
                         w0=np.array([0.02, 0.05, 0.01]), use_vision=True, use_ekf=True,
                         pos_meas_std=0.5, pixel_noise_std=4.0, r0_err=np.array([2.5, -0.8, 0.5]))
    res = run_simulation(cfg, rng_seed=seed)
    return float(res.range_m[-1]), res.dock_step is not None


def section_E(pool):
    out('\nE. Fixed-trajectory ablation at 4 px (setup of the original Experiment F, git 96657f5:')
    out('   one initial condition, only the noise seed varies; "success" there meant final range < 2 m)')
    r = list(pool.map(_fixed_trial, range(100)))
    fr = np.array([a for a, _ in r]); dk = np.array([b for _, b in r])
    verdict(int(dk.sum()) == 65 and 55 <= fr.std() <= 65,
            'N = 100 (seeds 0-99): docking at 4 px, terminal-range std', '65 of 100 dock, std 60 m',
            f'{int(dk.sum())} of 100 dock ({100*np.mean(fr < 2.0):.0f} % within 2 m), std {fr.std():.1f} m, '
            f'median {np.median(fr):.2f} m')


# ─────────────────────────────────────────────────────────────────────────────
# F. untuned LQR
# ─────────────────────────────────────────────────────────────────────────────
def _closed(cfg, seed=42):
    sim = RendezvousSimulator(cfg, rng_seed=seed)
    res = sim.run()
    x0 = np.concatenate([cfg.r0, cfg.v0])
    xr = np.concatenate([cfg.r_dock, np.zeros(3)])
    u0 = -sim.lqr.K @ (x0 - xr)
    sat = (np.abs(res.controls).max(axis=1) >= cfg.u_max - 1e-9)
    return dict(u0_axis=float(np.abs(u0).max()), u0_norm=float(np.linalg.norm(u0)),
                sat_frac=100 * float(sat.mean()), vpk=float(np.linalg.norm(res.v_true, axis=1).max()),
                xmin=float(res.r_true[:, 0].min()), dv=float(res.delta_v),
                docked=res.dock_step is not None, steps=res.n_steps_run)


def section_F():
    out('\nF. Untuned LQR (thrust_weight = 1) against the final design (thrust_weight = 1e5, docking-port hold point)')
    out('   seed 42, 600 steps, as in tests/test_phase7.py')
    good = _closed(SimConfig(n_steps=600))
    bad1 = _closed(SimConfig(n_steps=600, thrust_weight=1.0))
    bad2 = _closed(SimConfig(n_steps=600, thrust_weight=1.0, r_dock=np.zeros(3)))
    umax = 0.3
    verdict(80 <= bad1['u0_axis'] / umax <= 100 or 80 <= bad1['u0_norm'] / umax <= 100,
            'untuned gain demand at 30 m', 'approximately ninety times u_max (0.3 m/s^2)',
            f"{bad1['u0_axis']:.1f} m/s^2 per axis = {bad1['u0_axis']/umax:.0f}x, "
            f"{bad1['u0_norm']:.1f} m/s^2 Euclidean = {bad1['u0_norm']/umax:.0f}x")
    nsat = int(round(bad1['sat_frac'] * bad1['steps'] / 100))
    verdict(nsat == 115 and bad1['steps'] == 118, 'untuned: saturated steps before docking', '115 of 118',
            f"{nsat} of {bad1['steps']}")
    verdict(3.95 <= bad1['vpk'] <= 4.05, 'untuned: arrival / peak speed', 'about 4 m/s (3.99)', f"{bad1['vpk']:.2f} m/s")
    verdict(-23.05 <= bad1['xmin'] <= -22.95, 'untuned: passes through the target', '23 m through',
            f"min x = {bad1['xmin']:.1f} m")
    fdv, fv = bad2['dv'] / good['dv'], bad2['vpk'] / good['vpk']
    verdict(close(bad2['dv'], 41.2, 0.05) and close(good['dv'], 2.61, 0.005) and 15.5 <= fdv <= 16.5,
            'delta-v: untuned + aimed at CoM vs final design', 'factor of sixteen (41.2 to 2.61 m/s)',
            f"{bad2['dv']:.1f} vs {good['dv']:.2f} m/s = {fdv:.1f}x")
    verdict(close(bad2['vpk'], 3.99, 0.005) and close(good['vpk'], 1.35, 0.005),
            'peak closing speed: untuned + aimed at CoM vs final design', 'factor of three (3.99 to 1.35 m/s)',
            f"{bad2['vpk']:.2f} vs {good['vpk']:.2f} m/s = {fv:.1f}x")
    info('final design', f"docked = {good['docked']}, saturated {good['sat_frac']:.1f} %, "
         f"peak speed {good['vpk']:.2f} m/s, delta-v {good['dv']:.2f} m/s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=4)
    a = ap.parse_args()
    import numpy, scipy
    out('EXTRA CLAIMS CHECK -- numbers in the paper not printed by experiments A-F')
    out(f'Python {sys.version.split()[0]}, NumPy {numpy.__version__}, SciPy {scipy.__version__}')
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        section_A(pool)
        section_B(pool)
        section_C()
        section_D()
        section_E(pool)
    section_F()
    out()
    out(f"SUMMARY: PASS {TALLY['PASS']}   DIFFERS {TALLY['DIFFERS']}   INFO {TALLY['INFO']}")
    (ROOT / 'reproduce' / 'extra_claims_report.txt').write_text('\n'.join(LINES) + '\n')
    print('\nSaved: reproduce/extra_claims_report.txt')


if __name__ == '__main__':
    main()

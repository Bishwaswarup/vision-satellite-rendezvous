#!/usr/bin/env python3
"""
main.py
=======
Single entry point for the vision-based satellite rendezvous simulator.

    python main.py all              run the tests, then every experiment,
                                    then the animation
    python main.py test             unit tests only
    python main.py figures          the six experiments -> outputs/*.png
    python main.py animate          closed-loop GIF + summary sheet
    python main.py demo             Phase-7 walkthrough figures
    python main.py check            fast smoke test of the whole pipeline

Every figure is written to ``outputs/`` in print-safe monochrome (see
``viz/style.py``): series are separated by grey level, line style and marker
rather than by hue, so the figures survive a greyscale journal printer.

Run ``python main.py <command> --help`` for the options of one command.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'

# The six experiments that produce the paper's figures, in order.
EXPERIMENTS = [
    ('A', 'experiment_A.py', 'Orbital dynamics validation'),
    ('B', 'experiment_B.py', 'Pose accuracy vs pixel noise'),
    ('C', 'experiment_C.py', 'RANSAC robustness vs outliers'),
    ('D', 'experiment_D.py', 'MEKF vs UKF'),
    ('E', 'experiment_E.py', 'LQR vs MPC'),
    ('F', 'experiment_F.py', 'End-to-end Monte Carlo'),
]


# ── Console helpers ───────────────────────────────────────────────────────────

def _width() -> int:
    return min(shutil.get_terminal_size((80, 20)).columns, 78)


def rule(char: str = '-') -> None:
    print(char * _width())


def banner(text: str) -> None:
    rule('=')
    print(text)
    rule('=')


def step(text: str) -> None:
    print(f'\n>> {text}')
    rule()


def _fmt(seconds: float) -> str:
    return f'{seconds:5.1f} s' if seconds < 60 else f'{seconds/60:5.1f} min'


# ── Command implementations ───────────────────────────────────────────────────

def run_tests(args) -> int:
    """Run the unit-test suite via pytest."""
    step('Unit tests')
    cmd = [sys.executable, '-m', 'pytest', 'tests/',
           '-q' if not args.verbose else '-v', '--tb=short']
    if args.k:
        cmd += ['-k', args.k]
    t0 = time.perf_counter()
    rc = subprocess.call(cmd, cwd=ROOT)
    print(f'\ntests finished in {_fmt(time.perf_counter() - t0)} '
          f'(exit {rc})')
    return rc


def run_figures(args) -> int:
    """Run the experiment scripts that generate the paper's figures."""
    OUT.mkdir(exist_ok=True)
    selected = [e for e in EXPERIMENTS
                if not args.only or e[0].upper() in
                {c.strip().upper() for c in args.only.split(',')}]
    if not selected:
        print(f'No experiment matches --only {args.only!r}. '
              f'Choose from {", ".join(e[0] for e in EXPERIMENTS)}.')
        return 2

    failures = []
    for tag, script, title in selected:
        step(f'Experiment {tag} — {title}')
        t0 = time.perf_counter()
        rc = subprocess.call([sys.executable, f'experiments/{script}'],
                             cwd=ROOT)
        dt = time.perf_counter() - t0
        if rc == 0:
            print(f'   experiment {tag} ok  ({_fmt(dt)})')
        else:
            print(f'   experiment {tag} FAILED (exit {rc})')
            failures.append(tag)

    if failures:
        print(f'\nFailed experiments: {", ".join(failures)}')
        return 1
    return 0


def run_animate(args) -> int:
    """Render the closed-loop rendezvous as an animated GIF plus a summary."""
    step('Closed-loop animation')
    OUT.mkdir(exist_ok=True)

    import numpy as np                                    # noqa: F401
    from simulation.runner import RendezvousSimulator, SimConfig
    from simulation.video import VideoExporter

    cfg = SimConfig(n_steps=args.steps, use_vision=not args.no_vision)
    if args.full_run:
        cfg.stop_at_dock = False

    print(f'   simulating {cfg.n_steps} steps '
          f'(vision {"on" if cfg.use_vision else "off"}) ...')
    t0 = time.perf_counter()
    res = RendezvousSimulator(cfg, rng_seed=args.seed).run()

    docked = 'yes, step %d' % res.dock_step if res.dock_step else 'no'
    print(f'   flown {res.n_steps_run} steps   docked: {docked}')
    print(f'   final range {float(np.linalg.norm(res.r_true[-1])):.3f} m   '
          f'delta-v {res.delta_v:.2f} m/s')
    if cfg.use_vision:
        print(f'   vision dropout {res.vision_dropout_rate * 100:.1f} %')

    exporter = VideoExporter(res, fps=args.fps, step=args.frame_step)

    gif = OUT / 'rendezvous.gif'
    print(f'   rendering frames -> {gif.name} ...')
    exporter.export_gif(str(gif), verbose=args.verbose)

    png = OUT / 'rendezvous_summary.png'
    exporter.export_summary_png(str(png))

    print(f'\n   {gif}   ({gif.stat().st_size / 1e6:.1f} MB)')
    print(f'   {png}')
    print(f'   done in {_fmt(time.perf_counter() - t0)}')
    return 0


def run_demo(args) -> int:
    """Run the Phase-7 walkthrough, which produces the tutorial figures."""
    step('Phase-7 demo figures')
    rc = subprocess.call([sys.executable, 'notebooks/phase7_demo.py'], cwd=ROOT)
    return rc


def run_check(args) -> int:
    """
    Fast end-to-end smoke test.

    Exercises every subsystem once — dynamics, attitude, vision, pose,
    filtering and control — without the cost of the full experiment suite.
    Useful after a fresh clone or a dependency upgrade.
    """
    step('Pipeline smoke test')
    import numpy as np
    from dynamics import HCWPropagator
    from dynamics.constants import LEO_N
    from vision.body_model import ariane_model
    from vision.camera import rendezvous_camera
    from pose.epnp import EPnPSolver
    from pose.refine import refine_pose
    from simulation.runner import RendezvousSimulator, SimConfig
    from estimator.state import nis_statistics

    ok = True

    def report(name, passed, detail):
        nonlocal ok
        ok &= bool(passed)
        print(f'   [{"ok " if passed else "FAIL"}]  {name:<34s} {detail}')

    # 1. Dynamics — a periodic orbit must close on itself.
    prop = HCWPropagator(LEO_N)
    x0 = HCWPropagator.periodic_initial_condition(rho=100.0, n=LEO_N)
    xT = prop.stm(2 * np.pi / LEO_N) @ x0
    report('HCW periodic orbit closes', np.linalg.norm(xT[:3] - x0[:3]) < 1e-6,
           f'closure {np.linalg.norm(xT[:3] - x0[:3]):.2e} m')

    # 2. Vision + pose — a noiseless solve must be exact.
    cam, model = rendezvous_camera(), ariane_model()
    kp = model.keypoint_array
    R_gt = np.eye(3)
    t_gt = np.array([0.0, 0.0, 25.0])
    Pc = (R_gt @ kp.T).T + t_gt
    uv = np.stack([cam.K[0, 0] * Pc[:, 0] / Pc[:, 2] + cam.K[0, 2],
                   cam.K[1, 1] * Pc[:, 1] / Pc[:, 2] + cam.K[1, 2]], axis=-1)
    R_e, t_e, errs, solved = EPnPSolver().solve(kp, uv, cam.K)
    report('EPnP noiseless solve', solved and errs.mean() < 1e-6,
           f'reprojection {errs.mean():.2e} px')

    R_r, t_r, cost = refine_pose(R_e, t_e, kp, uv, cam.K)
    report('Gauss-Newton refinement', np.isfinite(cost) and cost < 1e-6,
           f'cost {cost:.2e} px^2')

    # 3. Closed loop — must dock, and the filter must stay consistent.
    res = RendezvousSimulator(SimConfig(n_steps=400), rng_seed=42).run()
    report('Closed-loop docking', res.dock_step is not None,
           f'step {res.dock_step}, delta-v {res.delta_v:.2f} m/s')
    report('Navigation error', res.pos_error.mean() < 1.0,
           f'mean {res.pos_error.mean():.3f} m')

    res_long = RendezvousSimulator(
        SimConfig(n_steps=300, use_vision=False, stop_at_dock=False),
        rng_seed=0).run()
    report('Vision dropout reported', res.n_vision_fail >= 0,
           f'{res.vision_dropout_rate * 100:.1f} % of steps')

    print()
    print('   ' + ('all subsystems nominal' if ok
                   else 'ONE OR MORE CHECKS FAILED'))
    return 0 if ok else 1


def run_all(args) -> int:
    """Tests, then every experiment, then the animation."""
    banner('Vision-based satellite rendezvous — full run')
    rc = run_tests(args)
    if rc != 0 and not args.keep_going:
        print('\nTests failed; stopping. Re-run with --keep-going to continue.')
        return rc
    rc_fig = run_figures(args)
    rc_ani = run_animate(args)

    banner('Summary')
    print(f'   tests       exit {rc}')
    print(f'   figures     exit {rc_fig}')
    print(f'   animation   exit {rc_ani}')
    n = len(list(OUT.glob('*.png'))) if OUT.exists() else 0
    print(f'\n   {n} PNG files and the GIF are in {OUT}/')
    return max(rc, rc_fig, rc_ani)


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='main.py',
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-v', '--verbose', action='store_true',
                   help='more output from the underlying tools')
    sub = p.add_subparsers(dest='command', required=True)

    s = sub.add_parser('test', help='run the unit tests')
    s.add_argument('-k', metavar='EXPR', default=None,
                   help='only run tests matching this pytest -k expression')
    s.set_defaults(func=run_tests)

    s = sub.add_parser('figures', help='run the experiments that make figures')
    s.add_argument('--only', metavar='A,B,...', default=None,
                   help='comma-separated experiment letters (default: all)')
    s.set_defaults(func=run_figures)

    s = sub.add_parser('animate', help='render the rendezvous animation')
    s.add_argument('--steps', type=int, default=600,
                   help='simulation step budget (default 600)')
    s.add_argument('--fps', type=int, default=8, help='GIF frame rate')
    s.add_argument('--frame-step', type=int, default=1,
                   help='render every Nth simulation step (default 1)')
    s.add_argument('--seed', type=int, default=42, help='RNG seed')
    s.add_argument('--no-vision', action='store_true',
                   help='use the direct pose measurement instead of vision')
    s.add_argument('--full-run', action='store_true',
                   help='do not stop the simulation at docking')
    s.set_defaults(func=run_animate)

    s = sub.add_parser('demo', help='Phase-7 walkthrough figures')
    s.set_defaults(func=run_demo)

    s = sub.add_parser('check', help='fast end-to-end smoke test')
    s.set_defaults(func=run_check)

    s = sub.add_parser('all', help='tests + figures + animation')
    s.add_argument('-k', metavar='EXPR', default=None)
    s.add_argument('--only', metavar='A,B,...', default=None)
    s.add_argument('--steps', type=int, default=600)
    s.add_argument('--fps', type=int, default=8)
    s.add_argument('--frame-step', type=int, default=1)
    s.add_argument('--seed', type=int, default=42)
    s.add_argument('--no-vision', action='store_true')
    s.add_argument('--full-run', action='store_true')
    s.add_argument('--keep-going', action='store_true',
                   help='continue to the figures even if tests fail')
    s.set_defaults(func=run_all)
    return p


def main(argv=None) -> int:
    sys.path.insert(0, str(ROOT))
    args = build_parser().parse_args(argv)
    OUT.mkdir(exist_ok=True)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print('\ninterrupted')
        return 130


if __name__ == '__main__':
    raise SystemExit(main())

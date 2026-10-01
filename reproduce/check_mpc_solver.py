#!/usr/bin/env python3
"""
check_mpc_solver.py -- is the MPC formulation exactly LQR when unconstrained,
how well does the SLSQP solver converge, and does Experiment E's result
(MPC 5.00 vs saturated LQR 16.83 m/s) survive a high-accuracy QP solve?

    .venv/bin/python reproduce/check_mpc_solver.py          # ~1-2 min
"""
import pathlib
import sys
import time

import numpy as np
from scipy.optimize import minimize, Bounds

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from controller.lqr import make_lqr      # noqa: E402
from controller.mpc import make_mpc      # noqa: E402

LINES = []


def out(s=''):
    print(s, flush=True)
    LINES.append(s)


X0 = np.array([0.0, 50.0, 0.0, 0.0, 0.0, 0.0])          # Experiment E
KW = dict(dt=10.0, pos_weight=10.0, vel_weight=1.0, thrust_weight=1.0)

lqr_free = make_lqr(u_max=None, **KW)
path = lqr_free.simulate(X0, n_steps=200)['states'][:60]

# 1 ── formulation: exact minimiser of the unconstrained QP vs LQR ──────────────
mpc = make_mpc(N=20, u_max=1e6, cone_half_angle=None, **KW)
H, F = mpc._H, mpc._SuTQbarSx
exact = max(np.abs(np.linalg.solve(H, -F @ x)[:3] + lqr_free.K @ x).max() for x in path)
out('1. Unconstrained QP solved exactly (linear solve) vs infinite-horizon LQR')
out(f'   max |u_QP - u_LQR| over 60 states = {exact:.2e} m/s^2   -> the formulation IS the LQR')
out(f'   cond(H) = {np.linalg.cond(H):.2e}')

# 2 ── what SLSQP (the solver the MPC actually uses) returns ─────────────────────
errs, its, msgs = [], [], set()
for x in path:
    f = F @ x
    sc = 1.0 / max(1.0, float(np.linalg.norm(f)))
    r = minimize(lambda U: sc * (0.5 * U @ H @ U + f @ U), np.zeros(60),
                 jac=lambda U: sc * (H @ U + f), method='SLSQP',
                 bounds=Bounds(np.full(60, -1e6), np.full(60, 1e6)),
                 options={'ftol': 1e-9, 'maxiter': 300})
    errs.append(np.abs(r.x[:3] + lqr_free.K @ x).max()); its.append(r.nit); msgs.add(r.message)
out('\n2. Same QP solved with SLSQP as in controller/mpc.py (cold start, ftol 1e-9, maxiter 300)')
out(f'   max |u_SLSQP - u_LQR| = {max(errs):.2e} m/s^2, median {np.median(errs):.2e}; '
    f'iterations {min(its)}-{max(its)}; messages: {sorted(msgs)}')
for solver in ('slsqp', 'bvls'):
    m = make_mpc(N=20, u_max=1e6, cone_half_angle=None, **KW)
    m.qp_solver = solver
    ws = [np.abs(m.control(x) - (-lqr_free.K @ x)).max() for x in path]
    out(f'   controller in closed-loop use, qp_solver={solver!r:8}: max {max(ws):.2e}, median {np.median(ws):.2e} m/s^2')

# 3 ── Experiment E with SLSQP vs with a high-accuracy box-QP solve ─────────────
def accurate_qp(self, objective, gradient, U0, bounds, constraints, nU=None):
    r = minimize(objective, U0, jac=gradient, method='L-BFGS-B', bounds=bounds,
                 options={'ftol': 1e-15, 'gtol': 1e-12, 'maxiter': 50000, 'maxcor': 50})
    return np.asarray(r.x, float), True


out('\n3. Experiment E closed loop (u_max = 0.1 m/s^2, 200 steps)')
res = {}
for name, solver, patch in (('SLSQP (original)', 'slsqp', False),
                            ('BVLS (exact, new default)', 'bvls', False),
                            ('L-BFGS-B, tight tolerances', 'slsqp', True)):
    m = make_mpc(N=20, u_max=0.1, cone_half_angle=None, **KW)
    m.qp_solver = solver
    if patch:
        m._solve_qp = accurate_qp.__get__(m)
    t0 = time.perf_counter()
    s = m.simulate(X0, n_steps=200)
    dt = time.perf_counter() - t0
    res[name] = s
    out(f'   {name:<28} total delta-v {s["delta_v"]:.4f} m/s, final range '
        f'{np.linalg.norm(s["states"][-1, :3]):.4f} m, {1e3*dt/200:.2f} ms/step')
c = {k: res[k]['controls'] for k in res}
k = list(c)
out(f'   max control difference BVLS vs L-BFGS-B: {np.abs(c[k[1]] - c[k[2]]).max():.2e} m/s^2; '
    f'SLSQP vs L-BFGS-B: {np.abs(c[k[0]] - c[k[2]]).max():.2e} m/s^2')
lq = make_lqr(u_max=0.1, **KW).simulate(X0, n_steps=200)
out(f'   saturated LQR for reference: total delta-v {lq["delta_v"]:.4f} m/s')

# 4 ── which solver actually reaches the QP minimum? ────────────────────────────
out('\n4. QP optimality at 40 states along the BVLS closed loop (u_max = 0.1, cold start for the iterative solvers)')
mb = make_mpc(N=20, u_max=0.1, cone_half_angle=None, **KW)
traj = mb.simulate(X0, n_steps=200)['states']
H, F, um = mb._H, mb._SuTQbarSx, 0.1
box = Bounds(np.full(60, -um), np.full(60, um))
kkt, excess = [], {'SLSQP': [], 'L-BFGS-B': []}
for x in traj[:200:5]:
    f = F @ x
    Ub, _ = mb._solve_box_qp(f)
    g = H @ Ub + f
    at_lo, at_hi = Ub <= -um + 1e-9, Ub >= um - 1e-9
    pg = np.where(at_lo, np.minimum(g, 0), np.where(at_hi, np.maximum(g, 0), g))
    kkt.append(np.abs(pg).max() / max(1.0, np.abs(f).max()))
    J = lambda U: 0.5 * U @ H @ U + f @ U
    sc = 1.0 / max(1.0, float(np.linalg.norm(f)))
    for name, meth, opt in (('SLSQP', 'SLSQP', {'ftol': 1e-9, 'maxiter': 300}),
                            ('L-BFGS-B', 'L-BFGS-B', {'ftol': 1e-15, 'gtol': 1e-12, 'maxiter': 50000, 'maxcor': 50})):
        r = minimize(lambda U: sc * J(U), np.zeros(60), jac=lambda U: sc * (H @ U + f),
                     method=meth, bounds=box, options=opt)
        excess[name].append((J(r.x) - J(Ub)) / max(abs(J(Ub)), 1e-12))
out(f'   BVLS optimality (KKT) residual: max {max(kkt):.1e} (relative to |f|)')
for name, e in excess.items():
    e = np.array(e)
    out(f'   {name:<9} cost above BVLS: min {e.min():+.1e}, median {np.median(e):+.1e}, max {e.max():+.1e} (relative); '
        f'lower than BVLS at {(e < -1e-9).sum()} of {len(e)} states')

(ROOT / 'reproduce' / 'mpc_solver_report.txt').write_text('\n'.join(LINES) + '\n')
print('\nSaved: reproduce/mpc_solver_report.txt')

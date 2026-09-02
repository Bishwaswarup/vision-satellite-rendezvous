"""
mpc.py
======
Model Predictive Control (MPC) for spacecraft rendezvous via HCW dynamics.

Formulation
-----------
At each step solve the finite-horizon QP:

    min_{U}  Σ_{k=0}^{N-1} [ xₖᵀ Q xₖ + uₖᵀ R uₖ ] + x_N^T P x_N
    s.t.     x_{k+1} = Φ xₖ + Γ uₖ                  (HCW dynamics)
             ‖uₖ‖_∞ ≤ u_max                    for k=0…N-1  (thrust limit)
             √(y_k² + z_k²) ≤ x_k · tan(θ)     for k=0…N-1  (approach cone)

The approach cone is a genuine second-order cone about the +x (radial)
approach axis: it constrains BOTH transverse components and requires x_k > 0,
so the chaser cannot satisfy it by sitting behind the target.  It is convex,
and it is evaluated on the predicted state X(U) — not on the free response —
so it actually constrains the decision variable.

where P is the LQR terminal cost (ensures stability + consistent horizon end).

Condensed QP
------------
Substituting the dynamics recursively:

    X = S_x x₀ + S_u U

gives a dense QP in U = [u₀; …; u_{N-1}] ∈ R^{3N}:

    min  ½ Uᵀ H U + fᵀ U
    s.t. lb ≤ U ≤ ub           (per-axis thrust bounds)
         g(U) ≥ 0              (approach corridor, if active)

Solved via scipy.optimize.minimize (method='SLSQP') with bounds.

Infeasibility
-------------
A corridor constraint can be infeasible — most obviously when the chaser is
already outside the cone.  The solver result is therefore always checked, and
on failure the controller degrades explicitly rather than silently applying
the warm start (which is zero thrust on the first call):

    cone QP  →  thrust-only QP (always feasible)  →  saturated LQR

Every outcome is recorded in `last_status` and counted, so a run that never
solved its intended problem cannot be mistaken for a run that did.

Reference
---------
  Mayne D.Q. et al., "Constrained model predictive control: Stability and
      optimality", Automatica, 2000.
  Fehse W., "Automated Rendezvous and Docking of Spacecraft", CUP, 2003.
"""

import numpy as np
from scipy.optimize import minimize, Bounds
from .lqr import LQRController, hcw_discrete, N_ORBITAL_DEFAULT


# ── MPC Controller ────────────────────────────────────────────────────────────

class MPCController:
    """
    Receding-horizon MPC for HCW rendezvous.

    Parameters
    ----------
    n           : float   orbital mean motion [rad/s]
    dt          : float   control step [s]
    N           : int     prediction horizon [steps]
    Q           : (6,6)   stage state cost
    R           : (3,3)   stage control cost
    P_terminal  : (6,6)   terminal cost (default: LQR P matrix)
    u_max       : float   per-axis thrust bound [m/s²]
    cone_half_angle : float | None   approach cone half-angle [deg] (None = off)
    warm_start  : bool    initialise solver from previous solution
    """

    FEAS_TOL = 1e-6      # [m] / [m/s²] tolerance when certifying a solution
    max_iter = 300       # SLSQP iteration budget
    CONE_EPS = 1e-6      # [m] smoothing so the cone is differentiable on axis

    def __init__(self,
                 n                : float = N_ORBITAL_DEFAULT,
                 dt               : float = 1.0,
                 N                : int   = 20,
                 Q                : np.ndarray = None,
                 R                : np.ndarray = None,
                 P_terminal       : np.ndarray = None,
                 u_max            : float = 0.1,
                 cone_half_angle  : float = None,
                 cone_activation_range : float = None,
                 cone_soft        : bool  = True,
                 cone_penalty     : float = 1e4,
                 warm_start       : bool  = True):
        self.n    = n
        self.dt   = dt
        self.N    = N
        self.u_max = u_max
        self.cone_half_angle = cone_half_angle
        # An approach corridor is a TERMINAL constraint in practice: it applies
        # once the chaser is inside the approach sphere.  Enforcing it from an
        # arbitrary starting point makes the QP infeasible from step 1.
        self.cone_activation_range = cone_activation_range
        # A HARD corridor loses recursive feasibility: once the chaser is
        # outside the cone there may be no admissible input that returns the
        # first predicted state to it, and the QP is simply infeasible.  The
        # standard remedy is a soft corridor — slack s_k >= 0 with a large
        # linear penalty — which is always feasible and drives s_k to zero
        # whenever the hard constraint can be met.
        self.cone_soft    = cone_soft
        self.cone_penalty = cone_penalty
        self.warm_start = warm_start

        if Q is None:
            Q = np.diag([10., 10., 10., 1., 1., 1.])
        if R is None:
            R = np.eye(3) * 1.0

        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)

        self.Phi, self.Gamma = hcw_discrete(n, dt)
        nx, nu = 6, 3
        self.nx, self.nu = nx, nu

        # Terminal cost: use LQR P matrix by default.  The LQR is also kept as
        # the last-resort fallback law when the QP cannot be solved at all.
        self.lqr = LQRController(n=n, dt=dt, Q=Q, R=R, u_max=u_max)
        if P_terminal is None:
            P_terminal = self.lqr.P
        self.P_term = np.asarray(P_terminal, dtype=float)

        # Pre-build prediction matrices (constant, only depends on Phi/Gamma)
        self._build_prediction_matrices()
        # Pre-build QP cost matrices
        self._build_qp_cost()

        # Warm-start storage
        self._U_prev    = np.zeros(N * nu)
        self.delta_v    = 0.0

        # Outcome bookkeeping.  n_solved counts calls that solved the intended
        # problem; it is NOT a call counter.
        self.n_calls        = 0
        self.n_solved       = 0
        self.n_cone_relaxed = 0
        self.n_failed       = 0
        self.last_status    = 'none'
        self.max_violation  = 0.0

    # ── Prediction matrices ────────────────────────────────────────────────────

    def _build_prediction_matrices(self):
        """
        Build S_x (N·nx × nx) and S_u (N·nx × N·nu) such that
            X = S_x x0 + S_u U
        """
        nx, nu, N = self.nx, self.nu, self.N
        Phi, Gamma = self.Phi, self.Gamma

        S_x = np.zeros((N * nx, nx))
        S_u = np.zeros((N * nx, N * nu))

        Phi_k = Phi.copy()
        for k in range(N):
            S_x[k*nx:(k+1)*nx, :] = Phi_k
            for j in range(k + 1):
                idx_row = k * nx
                idx_col = j * nu
                S_u[idx_row:idx_row+nx, idx_col:idx_col+nu] = (
                    np.linalg.matrix_power(Phi, k - j) @ Gamma)
            Phi_k = Phi @ Phi_k

        self.S_x = S_x
        self.S_u = S_u

    def _build_qp_cost(self):
        """
        Build dense QP cost: ½ Uᵀ H U + fᵀ(x0) U
        H = S_uᵀ Q̄ S_u + R̄  (constant)
        f = S_uᵀ Q̄ S_x x0   (linear in x0)
        where Q̄ = blkdiag(Q, …, Q, P_term) and R̄ = blkdiag(R, …, R).
        """
        nx, nu, N = self.nx, self.nu, self.N

        # Block-diagonal Q̄ and R̄
        Q_bar = np.zeros((N * nx, N * nx))
        for k in range(N - 1):
            Q_bar[k*nx:(k+1)*nx, k*nx:(k+1)*nx] = self.Q
        Q_bar[(N-1)*nx:N*nx, (N-1)*nx:N*nx] = self.P_term

        R_bar = np.kron(np.eye(N), self.R)

        self._Q_bar = Q_bar
        self._R_bar = R_bar

        SuTQbar = self.S_u.T @ Q_bar
        self._H  = SuTQbar @ self.S_u + R_bar      # (N·nu × N·nu)
        self._H  = 0.5 * (self._H + self._H.T)     # symmetrise
        self._SuTQbarSx = SuTQbar @ self.S_x        # (N·nu × nx)

    # ── Control ───────────────────────────────────────────────────────────────

    def control(self, x: np.ndarray,
                x_ref: np.ndarray = None) -> np.ndarray:
        """
        Solve the MPC QP and return the first control action.

        Parameters
        ----------
        x     : (6,)  current state [r, v]
        x_ref : (6,)  reference (default: origin)

        Returns
        -------
        u : (3,)  thrust acceleration [m/s²]
        """
        x     = np.asarray(x, dtype=float)
        x_ref = np.zeros(6) if x_ref is None else np.asarray(x_ref, dtype=float)
        e     = x - x_ref                              # tracking error

        # QP linear term
        f = self._SuTQbarSx @ e                        # (N·nu,)

        # Bounds: per-element thrust limits
        lb = np.full(self.N * self.nu, -self.u_max)
        ub = np.full(self.N * self.nu,  self.u_max)
        bounds = Bounds(lb, ub)

        # Approach corridor (None when disabled or outside activation range)
        cone = self._cone_spec(e, x_ref)
        nU   = self.N * self.nu
        H    = self._H

        # SLSQP's line search is sensitive to scaling, and the raw cost grows
        # like the square of the range: at 20 m, ||f|| ~ 1e5 and cond(H) ~ 1e5,
        # which makes the solver fail with "positive directional derivative".
        # Normalising the objective leaves the minimiser unchanged, puts the
        # cost near O(1), and gives the slack penalty a meaningful relative
        # weight (a 1 m corridor violation must cost more than the tracking
        # error it buys).
        sc = 1.0 / max(1.0, float(np.linalg.norm(f)))

        self.n_calls += 1

        # Plain box-constrained QP: cost + thrust bounds only.  This is the
        # cheap problem and it is always feasible (H is positive definite).
        def objective(U):
            return sc * (0.5 * U @ H @ U + f @ U)

        def gradient(U):
            return sc * (H @ U + f)

        box = Bounds(np.full(nU, -self.u_max), np.full(nU, self.u_max))
        U0  = self._warm_start_U()

        U_opt, ok = self._solve_qp(objective, gradient, U0, box, [], nU=nU)
        status = 'ok' if ok else 'lqr_fallback'

        if cone is not None and ok:
            # Active-set shortcut: if the unconstrained-in-corridor optimum is
            # already inside the cone then the corridor is inactive and this
            # IS the constrained optimum.  Only pay for the harder problem
            # when the constraint actually binds — which also keeps the
            # per-step solve time down.
            if float(np.min(cone['fun'](U_opt))) < -(self.FEAS_TOL + self.CONE_EPS):
                U_cone, _ = self._solve_cone_qp(cone, f, H, sc, nU, U0)

                # Judge the corridor solve by what it is FOR: the total
                # corridor violation of the predicted trajectory.  With a soft
                # corridor the slack is an artifact of the formulation, so
                # demanding that the augmented constraint be satisfied to
                # solver tolerance throws away perfectly good solutions —
                # which is how a corridor that does reduce the violation ends
                # up never being applied.
                v_plain = self._cone_violation_sum(cone, U_opt)
                v_cone  = self._cone_violation_sum(cone, U_cone)
                in_box  = np.max(np.abs(U_cone)) <= self.u_max + self.FEAS_TOL

                if in_box and np.all(np.isfinite(U_cone)) and v_cone <= v_plain:
                    U_opt = U_cone
                else:
                    status = 'cone_relaxed'
                    self.n_cone_relaxed += 1

        if status == 'ok':
            self.n_solved += 1

        if status == 'lqr_fallback':
            # Last resort: the saturated LQR law.  Never silently return the
            # warm start — on the first call that is zero thrust, i.e. a
            # controller that does nothing while reporting a successful solve.
            u_lqr = self.lqr.control(e)
            U_opt = np.zeros(nU)
            U_opt[:self.nu] = np.clip(u_lqr, -self.u_max, self.u_max)
            self.n_failed += 1

        self.last_status = status
        self.max_violation = (0.0 if cone is None
                              else max(0.0, -float(np.min(cone['fun'](U_opt)))))
        self._U_prev = U_opt

        u = U_opt[:self.nu]                             # first control action
        self.delta_v += np.linalg.norm(u) * self.dt
        return u

    def _solve_cone_qp(self, cone, f, H, sc, nU, U_ws):
        """
        Solve the corridor-constrained problem.

        With `cone_soft` the corridor carries a slack s_k >= 0 penalised
        linearly, so the problem is always feasible: a hard corridor loses
        recursive feasibility the moment the chaser is outside the cone, and
        an infeasible QP is exactly what used to make the controller fall back
        to zero thrust.  The slack is warm-started at the violation the
        shifted input actually produces, so the initial point is feasible;
        starting it at zero hands SLSQP an infeasible point and it spends its
        whole iteration budget just finding the feasible set.
        """
        ns = cone['n']

        if not self.cone_soft:
            U, ok = self._solve_qp(
                lambda U: sc * (0.5 * U @ H @ U + f @ U),
                lambda U: sc * (H @ U + f),
                U_ws,
                Bounds(np.full(nU, -self.u_max), np.full(nU, self.u_max)),
                [{'type': 'ineq', 'fun': cone['fun'], 'jac': cone['jac']}],
                nU=nU)
            return U, ok

        rho   = self.cone_penalty
        s_ws  = np.maximum(0.0, -cone['fun'](U_ws))
        s_cap = float(max(1e3, 10.0 * s_ws.max() + 1.0))

        def objective(Z):
            U = Z[:nU]
            return sc * (0.5 * U @ H @ U + f @ U) + rho * float(Z[nU:].sum())

        def gradient(Z):
            return np.concatenate([sc * (H @ Z[:nU] + f), rho * np.ones(ns)])

        bounds = Bounds(
            np.concatenate([np.full(nU, -self.u_max), np.zeros(ns)]),
            np.concatenate([np.full(nU,  self.u_max), np.full(ns, s_cap)]),
        )
        constraints = [{
            'type': 'ineq',
            'fun' : lambda Z: cone['fun'](Z[:nU]) + Z[nU:],
            'jac' : lambda Z: np.hstack([cone['jac'](Z[:nU]), np.eye(ns)]),
        }]
        Z0 = np.concatenate([U_ws, s_ws])

        Z, ok = self._solve_qp(objective, gradient, Z0, bounds,
                               constraints, nU=nU)
        return Z[:nU], ok

    # ── QP solve + verification ───────────────────────────────────────────────

    def _solve_qp(self, objective, gradient, U0, bounds, constraints, nU=None):
        """
        Run SLSQP and verify the answer.

        `result.success` alone is not enough: SLSQP can report success on a
        point that still violates a constraint, and can report failure while
        having found a perfectly good one.  What matters is whether the
        returned U is feasible, so that is what is checked.

        Returns
        -------
        U  : (N*nu,) the candidate solution
        ok : bool    True if the solve converged AND U is feasible
        """
        result = minimize(
            objective, U0, jac=gradient,
            bounds=bounds, constraints=constraints,
            method='SLSQP',
            options={'ftol': 1e-9, 'maxiter': self.max_iter, 'disp': False},
        )
        U = np.asarray(result.x, dtype=float)

        if not np.all(np.isfinite(U)):
            return U0, False
        # Bounds are enforced by SLSQP, but check anyway — cheap and it makes
        # a returned solution self-certifying.
        nU = len(U) if nU is None else nU
        if np.max(np.abs(U[:nU])) > self.u_max + self.FEAS_TOL:
            return U, False
        if self._constraint_violation(U, constraints) > self.FEAS_TOL + self.CONE_EPS:
            return U, False

        # SLSQP frequently exits with "positive directional derivative for
        # linesearch" at a point it simply cannot improve on.  That is a
        # status message, not a verdict on the answer.  Accept any feasible
        # point that does not make the objective worse than the warm start;
        # that is what actually matters for closed-loop behaviour.
        if result.success:
            return U, True
        try:
            j_new = objective(U)
            j_ref = objective(np.asarray(U0, dtype=float))
            # A converged receding-horizon warm start is already near optimal,
            # so require only that the answer is no worse to within a relative
            # tolerance — a strict '<' fails on floating-point noise.
            improved = j_new <= j_ref + 1e-9 * (1.0 + abs(j_ref))
        except Exception:
            improved = False
        return U, bool(improved)

    @staticmethod
    def _cone_violation_sum(cone, U):
        """Total corridor violation of the predicted trajectory [m]."""
        return float(np.clip(-cone['fun'](np.asarray(U, dtype=float)),
                             0.0, None).sum())

    def _constraint_violation(self, U, constraints):
        """Largest violation of the inequality constraints g(U) >= 0."""
        worst = 0.0
        for c in constraints:
            g = np.atleast_1d(c['fun'](U))
            if g.size:
                worst = max(worst, float(-np.min(g)))
        return max(worst, 0.0)

    def simulate(self, x0: np.ndarray,
                 n_steps: int = 100,
                 x_ref: np.ndarray = None,
                 noise_std: float = 0.0,
                 rng: np.random.Generator = None) -> dict:
        """Simulate closed-loop MPC from x0."""
        x_ref = np.zeros(6) if x_ref is None else np.asarray(x_ref, dtype=float)
        if rng is None:
            rng = np.random.default_rng(0)

        x = np.asarray(x0, dtype=float).copy()
        self.delta_v = 0.0
        states, controls = [x.copy()], []

        for _ in range(n_steps):
            x_meas = x + rng.normal(0, noise_std, 6) if noise_std > 0 else x
            u = self.control(x_meas, x_ref)
            controls.append(u.copy())
            x = self.Phi @ x + self.Gamma @ u
            states.append(x.copy())

        return {
            'states'   : np.array(states),
            'controls' : np.array(controls),
            'delta_v'  : self.delta_v,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _warm_start_U(self) -> np.ndarray:
        """Shift previous solution by one step (receding horizon warm-start)."""
        if not self.warm_start:
            return np.zeros(self.N * self.nu)
        nu = self.nu
        U_shifted = np.zeros_like(self._U_prev)
        U_shifted[:-(nu)] = self._U_prev[nu:]
        return U_shifted

    def _cone_spec(self, e, x_ref):
        """
        Approach-cone constraint: the chaser must remain inside a cone of
        half-angle θ about the +x (radial) approach axis, for every step of
        the prediction horizon.

            √(y_k² + z_k²)  ≤  tan(θ) · x_k          k = 0 … N-1

        Written for SLSQP as g_k(U) ≥ 0 with

            g_k(U) = tan(θ)·x_k(U) − √(y_k(U)² + z_k(U)² + ε²)

        where [x_k, y_k, z_k] are taken from the predicted state
        X(U) = S_x e + S_u U.  Three properties matter and all three were
        missing before:

          * BOTH transverse components appear.  Constraining y alone leaves
            the cross-track axis completely free — it is a wedge, not a cone.
          * x_k is used signed, not |x_k|.  With the absolute value the cone
            is mirrored into x < 0 and the chaser satisfies it while sitting
            *behind* the target.  Since the right-hand side is non-negative,
            the signed form also implies x_k ≥ 0.
          * The radius depends on U.  Taking it from the free response
            S_x e makes the right-hand side a constant, so the constraint
            stops being a cone at all.

        The ε inside the square root (1 µm) keeps g differentiable on the
        axis; it makes the constraint very slightly conservative, which is the
        safe direction.

        Returns None when the cone is disabled or the chaser is outside the
        activation range, otherwise a dict with `fun`, `jac` and `n`.
        """
        if self.cone_half_angle is None:
            return None

        e     = np.asarray(e, dtype=float)
        x_ref = np.asarray(x_ref, dtype=float)

        # A corridor is a terminal-approach constraint: it applies once the
        # chaser is inside the approach sphere.  Range is measured from the
        # TARGET, not from the reference point.
        r_now = e[:3] + x_ref[:3]
        if (self.cone_activation_range is not None
                and np.linalg.norm(r_now) > self.cone_activation_range):
            return None

        tan_theta = np.tan(np.radians(self.cone_half_angle))
        nx, N = self.nx, self.N

        ix = [k * nx + 0 for k in range(N)]
        iy = [k * nx + 1 for k in range(N)]
        iz = [k * nx + 2 for k in range(N)]

        Su_x, Su_y, Su_z = self.S_u[ix], self.S_u[iy], self.S_u[iz]
        # The prediction X = S_x e + S_u U is the trajectory of the ERROR
        # e = x - x_ref.  The corridor is a statement about where the chaser
        # is relative to the TARGET, so the reference offset is added back:
        # constraining the error would make the cone degenerate at any
        # non-zero hold point (the error shrinks to zero inside it).
        xf = self.S_x[ix] @ e + x_ref[0]
        yf = self.S_x[iy] @ e + x_ref[1]
        zf = self.S_x[iz] @ e + x_ref[2]
        eps = self.CONE_EPS

        def _xyz(U):
            return (xf + Su_x @ U, yf + Su_y @ U, zf + Su_z @ U)

        def fun(U):
            xk, yk, zk = _xyz(U)
            return tan_theta * xk - np.sqrt(yk**2 + zk**2 + eps**2)

        def jac(U):
            xk, yk, zk = _xyz(U)
            rho = np.sqrt(yk**2 + zk**2 + eps**2)          # (N,)
            return (tan_theta * Su_x
                    - (yk[:, None] * Su_y + zk[:, None] * Su_z)
                    / rho[:, None])

        return {'fun': fun, 'jac': jac, 'n': N}

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def cone_violation(self, r: np.ndarray) -> float:
        """
        Signed cone violation at a position r [m], in metres.

        Positive means outside the corridor.  Provided so that a caller (or a
        test) can check the *achieved* trajectory rather than trusting the
        solver's own report.
        """
        if self.cone_half_angle is None:
            return 0.0
        r = np.asarray(r, dtype=float)
        tan_theta = np.tan(np.radians(self.cone_half_angle))
        return float(np.hypot(r[1], r[2]) - tan_theta * r[0])


# ── Convenience ───────────────────────────────────────────────────────────────

def make_mpc(n=N_ORBITAL_DEFAULT, dt=1.0, N=20,
             pos_weight=10.0, vel_weight=1.0, thrust_weight=1.0,
             u_max=0.1, cone_half_angle=None, cone_activation_range=None):
    """Build an MPCController with diagonal Q and R."""
    Q = np.diag([pos_weight]*3 + [vel_weight]*3)
    R = np.eye(3) * thrust_weight
    return MPCController(n=n, dt=dt, N=N, Q=Q, R=R, u_max=u_max,
                         cone_half_angle=cone_half_angle,
                         cone_activation_range=cone_activation_range)

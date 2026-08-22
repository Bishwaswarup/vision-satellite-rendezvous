"""
mpc.py
======
Model Predictive Control (MPC) for spacecraft rendezvous via HCW dynamics.

Formulation
-----------
At each step solve the finite-horizon QP:

    min_{U}  Σ_{k=0}^{N-1} [ xₖᵀ Q xₖ + uₖᵀ R uₖ ] + x_N^T P x_N
    s.t.     x_{k+1} = Φ xₖ + Γ uₖ                  (HCW dynamics)
             ‖uₖ‖_∞ ≤ u_max          for k=0…N-1    (thrust limit)
             |y_k| ≤ r_k · tan(θ)    for k=0…N-1    (approach cone, optional)

where P is the LQR terminal cost (ensures stability + consistent horizon end).

Condensed QP
------------
Substituting the dynamics recursively:

    X = S_x x₀ + S_u U

gives a dense QP in U = [u₀; …; u_{N-1}] ∈ R^{3N}:

    min  ½ Uᵀ H U + fᵀ U
    s.t. lb ≤ U ≤ ub           (per-axis thrust bounds)
         A_ineq U ≤ b_ineq     (approach corridor, if active)

Solved via scipy.optimize.minimize (method='SLSQP') with bounds.

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

    def __init__(self,
                 n                : float = N_ORBITAL_DEFAULT,
                 dt               : float = 1.0,
                 N                : int   = 20,
                 Q                : np.ndarray = None,
                 R                : np.ndarray = None,
                 P_terminal       : np.ndarray = None,
                 u_max            : float = 0.1,
                 cone_half_angle  : float = None,
                 warm_start       : bool  = True):
        self.n    = n
        self.dt   = dt
        self.N    = N
        self.u_max = u_max
        self.cone_half_angle = cone_half_angle
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

        # Terminal cost: use LQR P matrix by default
        if P_terminal is None:
            lqr = LQRController(n=n, dt=dt, Q=Q, R=R, u_max=u_max)
            P_terminal = lqr.P
        self.P_term = np.asarray(P_terminal, dtype=float)

        # Pre-build prediction matrices (constant, only depends on Phi/Gamma)
        self._build_prediction_matrices()
        # Pre-build QP cost matrices
        self._build_qp_cost()

        # Warm-start storage
        self._U_prev    = np.zeros(N * nu)
        self.delta_v    = 0.0
        self.n_solved   = 0

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

        # Linear constraints: approach corridor
        constraints = self._build_corridor_constraints(x, x_ref)

        # Warm start
        U0 = self._warm_start_U()

        # Solve QP via SLSQP
        H = self._H
        def objective(U):
            return 0.5 * U @ H @ U + f @ U
        def gradient(U):
            return H @ U + f

        result = minimize(
            objective, U0, jac=gradient,
            bounds=bounds, constraints=constraints,
            method='SLSQP',
            options={'ftol': 1e-9, 'maxiter': 200, 'disp': False},
        )

        U_opt = result.x if result.success else U0
        self._U_prev = U_opt
        self.n_solved += 1

        u = U_opt[:self.nu]                             # first control action
        self.delta_v += np.linalg.norm(u) * self.dt
        return u

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

    def _build_corridor_constraints(self, x, x_ref):
        """
        Optional approach cone constraint: chaser must stay within a cone
        of half-angle θ around the X (radial) approach axis.

        Linearised as: |y_k| ≤ |x_k| tan(θ) at each prediction step.
        Implemented as a linear inequality on U.
        """
        if self.cone_half_angle is None:
            return []

        tan_theta = np.tan(np.radians(self.cone_half_angle))
        nx, nu, N = self.nx, self.nu, self.N

        # Predicted positions: X = S_x e + S_u U  (error coords)
        e = x - x_ref
        X_free = self.S_x @ e    # (N*nx,) ignoring U

        rows_y, rows_x = [], []
        for k in range(N):
            rows_y.append(k * nx + 1)   # y component
            rows_x.append(k * nx + 0)   # x component

        # |y_k| ≤ |x_k| tan θ  →  -tan θ x_k ≤ y_k ≤ tan θ x_k
        # (approximated at free response for linearity)
        x_ref_k = np.abs(X_free[rows_x])   # (N,)
        ub_y    = tan_theta * x_ref_k        # (N,)

        # S_u rows corresponding to y
        Su_y = self.S_u[rows_y, :]           # (N, N*nu)

        A_ub = np.vstack([ Su_y, -Su_y])     # (2N, N*nu)
        b_ub = np.concatenate([
            ub_y - X_free[rows_y],
            ub_y + X_free[rows_y],
        ])                                    # (2N,)

        # SLSQP requires dict-format constraints (LinearConstraint is silently
        # ignored by SLSQP — it only works with trust-constr).
        # Inequality form: g(U) >= 0  →  b_ub - A_ub @ U >= 0
        return [{
            'type': 'ineq',
            'fun' : lambda U, A=A_ub, b=b_ub: b - A @ U,
            'jac' : lambda U, A=A_ub, b=b_ub: -A,
        }]


# ── Convenience ───────────────────────────────────────────────────────────────

def make_mpc(n=N_ORBITAL_DEFAULT, dt=1.0, N=20,
             pos_weight=10.0, vel_weight=1.0, thrust_weight=1.0,
             u_max=0.1, cone_half_angle=None):
    """Build an MPCController with diagonal Q and R."""
    Q = np.diag([pos_weight]*3 + [vel_weight]*3)
    R = np.eye(3) * thrust_weight
    return MPCController(n=n, dt=dt, N=N, Q=Q, R=R, u_max=u_max,
                         cone_half_angle=cone_half_angle)

"""
lqr.py
======
Linear Quadratic Regulator (LQR) for spacecraft rendezvous via HCW dynamics.

The rendezvous problem is cast as an infinite-horizon discrete LQR:

    min  Σ_{k=0}^∞  xₖᵀ Q xₖ + uₖᵀ R uₖ
    s.t. x_{k+1} = Φ xₖ + Γ uₖ          (discrete HCW, ZOH)
         ‖uₖ‖_∞ ≤ u_max                  (thrust saturation)

where the state is x = [r(3), v(3)] ∈ R⁶ (relative position and velocity in LVLH)
and the control is u = [aₓ, a_y, a_z] ∈ R³ (thrust acceleration [m/s²]).

The optimal gain K* is found by solving the Discrete Algebraic Riccati Equation
(DARE) and gives u* = -K* x.

Δv accounting
-------------
Total Δv [m/s] = Σ ‖uₖ‖₂ · dt  (impulse per step × number of steps)

References
----------
  Clohessy W.H., Wiltshire R.S., "Terminal Guidance System for Satellite
      Rendezvous", J. Aero Sci., 1960.
  Bryson A.E., Ho Y.-C., "Applied Optimal Control", 1975.
"""

import numpy as np
from scipy.linalg import solve_discrete_are, expm

# Default orbital rate (ISS ~400 km)
N_ORBITAL_DEFAULT = 1.1368e-3   # rad/s


# ── HCW discretization ────────────────────────────────────────────────────────

def hcw_matrices(n: float = N_ORBITAL_DEFAULT) -> tuple:
    """
    Continuous-time HCW state and input matrices (6×6, 6×3).
    State order: [x, y, z, ẋ, ẏ, ż]  (radial, along-track, cross-track)
    """
    A = np.array([
        [0,    0, 0,  1,   0, 0],
        [0,    0, 0,  0,   1, 0],
        [0,    0, 0,  0,   0, 1],
        [3*n**2, 0, 0, 0,  2*n, 0],
        [0,    0, 0, -2*n, 0,  0],
        [0,    0,-n**2, 0, 0,  0],
    ], dtype=float)
    B = np.zeros((6, 3))
    B[3:, :] = np.eye(3)
    return A, B


def hcw_discrete(n: float = N_ORBITAL_DEFAULT,
                 dt: float = 1.0) -> tuple:
    """
    Zero-Order Hold discretization of HCW.

    Returns
    -------
    Phi   : (6,6)  discrete state transition matrix
    Gamma : (6,3)  discrete input matrix
    """
    A, B = hcw_matrices(n)
    nx, nu = 6, 3

    # Augmented matrix for exact ZOH: expm([[A, B], [0, 0]] * dt)
    M = np.zeros((nx + nu, nx + nu))
    M[:nx, :nx] = A
    M[:nx, nx:] = B
    eM = expm(M * dt)

    Phi   = eM[:nx, :nx]
    Gamma = eM[:nx, nx:]
    return Phi, Gamma


# ── LQR solver ────────────────────────────────────────────────────────────────

class LQRController:
    """
    Infinite-horizon discrete LQR for HCW rendezvous.

    Parameters
    ----------
    n       : float   orbital mean motion [rad/s]
    dt      : float   control step [s]
    Q       : (6,6)   state cost matrix (default: position-weighted)
    R       : (3,3)   control cost matrix (default: I₃)
    u_max   : float   per-axis thrust saturation [m/s²]  (None = unlimited)
    """

    def __init__(self,
                 n     : float = N_ORBITAL_DEFAULT,
                 dt    : float = 1.0,
                 Q     : np.ndarray = None,
                 R     : np.ndarray = None,
                 u_max : float = 0.1):
        self.n     = n
        self.dt    = dt
        self.u_max = u_max

        # Default costs: penalise position 10× more than velocity
        if Q is None:
            Q = np.diag([10., 10., 10., 1., 1., 1.])
        if R is None:
            R = np.eye(3) * 1.0

        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)

        self.Phi, self.Gamma = hcw_discrete(n, dt)
        self._solve_dare()

        # Running Δv accumulator
        self.delta_v = 0.0

    # ── DARE ──────────────────────────────────────────────────────────────────

    def _solve_dare(self):
        """Solve DARE and compute optimal gain K."""
        P = solve_discrete_are(self.Phi, self.Gamma, self.Q, self.R)
        self.P = P                                           # value function matrix
        GtP   = self.Gamma.T @ P
        self.K = np.linalg.solve(GtP @ self.Gamma + self.R,
                                 GtP @ self.Phi)            # (3, 6)

    # ── Control law ───────────────────────────────────────────────────────────

    def control(self, x: np.ndarray,
                x_ref: np.ndarray = None,
                saturate: bool = True) -> np.ndarray:
        """
        Compute LQR control u = -K (x - x_ref).

        Parameters
        ----------
        x     : (6,)  current state  [r, v]
        x_ref : (6,)  reference state  (default: origin)
        saturate : bool  apply per-axis thrust saturation

        Returns
        -------
        u : (3,)  thrust acceleration [m/s²]
        """
        x     = np.asarray(x, dtype=float)
        x_ref = np.zeros(6) if x_ref is None else np.asarray(x_ref, dtype=float)
        e     = x - x_ref
        u     = -self.K @ e

        if saturate and self.u_max is not None:
            u = np.clip(u, -self.u_max, self.u_max)

        # Accumulate Δv
        self.delta_v += np.linalg.norm(u) * self.dt

        return u

    def lyapunov_value(self, x: np.ndarray,
                       x_ref: np.ndarray = None) -> float:
        """V(x) = xᵀ P x  — the LQR value function (Lyapunov certificate)."""
        x_ref = np.zeros(6) if x_ref is None else np.asarray(x_ref)
        e = np.asarray(x) - x_ref
        return float(e @ self.P @ e)

    def is_stable(self) -> bool:
        """
        Check closed-loop stability: all eigenvalues of (Φ - Γ K) inside unit circle.
        """
        A_cl = self.Phi - self.Gamma @ self.K
        eigs = np.linalg.eigvals(A_cl)
        return bool(np.all(np.abs(eigs) < 1.0))

    def closed_loop_eigs(self) -> np.ndarray:
        """Eigenvalues of the closed-loop matrix Φ - Γ K."""
        return np.linalg.eigvals(self.Phi - self.Gamma @ self.K)

    def simulate(self, x0: np.ndarray,
                 n_steps: int = 200,
                 x_ref: np.ndarray = None,
                 noise_std: float = 0.0,
                 rng: np.random.Generator = None) -> dict:
        """
        Simulate closed-loop rendezvous from initial state x0.

        Parameters
        ----------
        x0       : (6,)  initial state [r, v]
        n_steps  : int   number of steps
        x_ref    : (6,)  target state (default: origin)
        noise_std: float state measurement noise std [m / m/s]
        rng      : Generator  for reproducible noise

        Returns
        -------
        dict with 'states' (n+1,6), 'controls' (n,3), 'costs' (n,),
                  'delta_v' (float)
        """
        x_ref = np.zeros(6) if x_ref is None else np.asarray(x_ref, dtype=float)
        if rng is None:
            rng = np.random.default_rng(0)

        x = np.asarray(x0, dtype=float).copy()
        self.delta_v = 0.0

        states   = [x.copy()]
        controls = []
        costs    = []

        for _ in range(n_steps):
            x_meas = x + rng.normal(0, noise_std, 6) if noise_std > 0 else x
            u      = self.control(x_meas, x_ref)
            cost_k = float(x @ self.Q @ x + u @ self.R @ u)
            costs.append(cost_k)
            controls.append(u.copy())
            x = self.Phi @ x + self.Gamma @ u
            states.append(x.copy())

        return {
            'states'   : np.array(states),
            'controls' : np.array(controls),
            'costs'    : np.array(costs),
            'delta_v'  : self.delta_v,
        }


# ── Convenience ───────────────────────────────────────────────────────────────

def make_lqr(n=N_ORBITAL_DEFAULT, dt=1.0,
             pos_weight=10.0, vel_weight=1.0, thrust_weight=1.0,
             u_max=0.1):
    """Build an LQRController with diagonal Q and R."""
    Q = np.diag([pos_weight]*3 + [vel_weight]*3)
    R = np.eye(3) * thrust_weight
    return LQRController(n=n, dt=dt, Q=Q, R=R, u_max=u_max)

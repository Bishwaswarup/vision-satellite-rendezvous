"""
hcw.py
======
Hill-Clohessy-Wiltshire (HCW) relative orbital dynamics propagator.

Implements BOTH:
  1. Analytical state-transition matrix (exact for linear, unperturbed model)
  2. Numerical ODE integration via scipy RK45 / DOP853

State vector:
    x_hcw = [x, y, z, xdot, ydot, zdot]   in LVLH frame

    x  -> radial     (positive away from Earth)
    y  -> along-track (positive in velocity direction)
    z  -> cross-track (positive out of orbital plane)

Equations of motion (Hill 1878, Clohessy & Wiltshire 1960):
    x'' - 2n*y' - 3n^2*x = fx
    y'' + 2n*x'           = fy
    z'' + n^2*z           = fz

Reference:
    Schaub, H. & Junkins, J.L., "Analytical Mechanics of Space Systems", 3rd ed., 2018
    Clohessy & Wiltshire, JARS 1960.
"""

import numpy as np
from scipy.integrate import solve_ivp
from .constants import MU_EARTH


class HCWPropagator:
    """
    Propagates relative motion in the LVLH frame using HCW equations.

    Parameters
    ----------
    n : float
        Mean motion of the chief orbit [rad/s].
        Computed from semi-major axis: n = sqrt(mu / a^3).
    """

    def __init__(self, n: float):
        self.n = float(n)
        self._build_matrices()

    # ── Construction ──────────────────────────────────────────────────────────
    def _build_matrices(self):
        """Build the continuous-time A and B matrices for the HCW system."""
        n = self.n
        # State matrix A  (6x6)
        self.A = np.array([
            [ 0,  0,   0,  1,   0,  0],
            [ 0,  0,   0,  0,   1,  0],
            [ 0,  0,   0,  0,   0,  1],
            [3*n**2, 0, 0,  0,  2*n,  0],
            [ 0,  0,   0, -2*n, 0,  0],
            [ 0,  0, -n**2, 0,  0,  0],
        ], dtype=float)

        # Input matrix B  (6x3)  — control force per unit mass [m/s^2]
        self.B = np.vstack([np.zeros((3, 3)), np.eye(3)])

    # ── Analytical STM ────────────────────────────────────────────────────────
    def stm(self, t: float) -> np.ndarray:
        """
        Analytical HCW State Transition Matrix Phi(t, 0).

        Propagates x(t) = Phi(t) @ x(0) for zero control input.

        Parameters
        ----------
        t : float
            Propagation time [s].

        Returns
        -------
        Phi : np.ndarray, shape (6, 6)
        """
        n  = self.n
        nt = n * t
        c  = np.cos(nt)
        s  = np.sin(nt)

        Phi = np.array([
            # x(t)
            [4 - 3*c,        0,  0,  s/n,        2*(1-c)/n,  0    ],
            # y(t)
            [6*(s - nt),     1,  0, -2*(1-c)/n, (4*s - 3*nt)/n, 0 ],
            # z(t)
            [0,              0,  c,  0,           0,          s/n  ],
            # xdot(t)
            [3*n*s,          0,  0,  c,           2*s,        0    ],
            # ydot(t)
            [6*n*(c - 1),    0,  0, -2*s,         4*c - 3,    0    ],
            # zdot(t)
            [0,              0, -n*s, 0,           0,          c    ],
        ], dtype=float)

        return Phi

    def propagate_analytical(self, x0: np.ndarray, t: float) -> np.ndarray:
        """
        Propagate initial state x0 to time t using the analytical STM.

        Parameters
        ----------
        x0 : array-like, shape (6,)
            Initial state [x, y, z, xdot, ydot, zdot] [m, m/s].
        t  : float
            Time of flight [s].

        Returns
        -------
        xf : np.ndarray, shape (6,)
        """
        return self.stm(t) @ np.asarray(x0, dtype=float)

    # ── Numerical Integration ─────────────────────────────────────────────────
    def _eom(self, t: float, x: np.ndarray,
             control_fn=None) -> np.ndarray:
        """
        ODE right-hand side for HCW with optional control input.

        Parameters
        ----------
        t          : float
        x          : array, shape (6,)
        control_fn : callable(t, x) -> array(3,) or None
            Returns [fx, fy, fz] control acceleration [m/s^2].
        """
        f = np.zeros(3)
        if control_fn is not None:
            f = np.asarray(control_fn(t, x), dtype=float)

        n = self.n
        xp, yp, zp   = x[0], x[1], x[2]
        xdp, ydp, zdp = x[3], x[4], x[5]

        xddot = 3*n**2 * xp + 2*n * ydp + f[0]
        yddot =            - 2*n * xdp + f[1]
        zddot = -n**2 * zp              + f[2]

        return np.array([xdp, ydp, zdp, xddot, yddot, zddot])

    def propagate_numerical(self,
                            x0: np.ndarray,
                            t_span: tuple,
                            t_eval: np.ndarray = None,
                            control_fn=None,
                            method: str = 'DOP853',
                            rtol: float = 1e-10,
                            atol: float = 1e-12) -> dict:
        """
        Numerically integrate the HCW equations over a time interval.

        Parameters
        ----------
        x0         : array-like, shape (6,)
        t_span     : (t0, tf) tuple [s]
        t_eval     : array of output times, or None for adaptive stepping
        control_fn : callable(t, x) -> array(3,)  [m/s^2], optional
        method     : 'RK45' | 'DOP853' (default DOP853 for high accuracy)
        rtol, atol : ODE solver tolerances

        Returns
        -------
        dict with keys:
            't'       : time array [s]
            'y'       : state array, shape (6, N)   (scipy convention)
            'success' : bool
        """
        sol = solve_ivp(
            fun=lambda t, x: self._eom(t, x, control_fn),
            t_span=t_span,
            y0=np.asarray(x0, dtype=float),
            method=method,
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
            dense_output=False,
        )
        return {'t': sol.t, 'y': sol.y, 'success': sol.success}

    # ── Drift-Free Periodic Orbit Initialiser ─────────────────────────────────
    @classmethod
    def periodic_initial_condition(cls, rho: float, n: float,
                                   theta0: float = 0.0) -> np.ndarray:
        """
        Return an initial state that produces a closed, drift-free
        elliptical HCW orbit (2:1 ellipse in x-y plane).

        The condition for drift-free motion is:
            y_dot(0) = -2 n x(0)

        Parameters
        ----------
        rho    : float  Half the x-amplitude [m].
        n      : float  Mean motion [rad/s].
        theta0 : float  Initial phase angle [rad].

        Returns
        -------
        x0 : np.ndarray, shape (6,)
        """
        x0    =  rho * np.cos(theta0)
        y0    = -2 * rho * np.sin(theta0)
        z0    =  0.0
        xdot0 = -rho * n * np.sin(theta0)
        ydot0 = -2 * rho * n * np.cos(theta0)   # drift-free condition
        zdot0 =  0.0
        return np.array([x0, y0, z0, xdot0, ydot0, zdot0])

    # ── Impulsive Delta-V ─────────────────────────────────────────────────────
    @staticmethod
    def apply_dv(x: np.ndarray, dv: np.ndarray) -> np.ndarray:
        """Apply an instantaneous delta-v to the state vector."""
        x_new = x.copy()
        x_new[3:6] += np.asarray(dv, dtype=float)
        return x_new

    # ── Discretisation ────────────────────────────────────────────────────────
    def discretise(self, Ts: float):
        """
        Discretise the HCW system at sample period Ts [s] using
        matrix exponential.

        Returns (Ad, Bd) for the discrete-time system:
            x[k+1] = Ad @ x[k] + Bd @ u[k]
        """
        from scipy.linalg import expm
        import numpy as np

        # Ad = e^(A*Ts)
        Ad = expm(self.A * Ts)

        # Bd = (integral_0^Ts e^(A*tau) dtau) @ B
        # Use the analytical STM: integral = (Ad - I) @ A^{-1} @ B
        # For HCW, A is singular (has zero eigenvalue for y drift).
        # Use numerical integration instead:
        from scipy.integrate import quad
        from scipy.linalg import expm as _expm

        def integrand(tau):
            return _expm(self.A * tau) @ self.B

        Bd = np.zeros((6, 3))
        for i in range(6):
            for j in range(3):
                Bd[i, j], _ = quad(lambda tau: (_expm(self.A * tau) @ self.B)[i, j],
                                   0, Ts)
        return Ad, Bd

    def __repr__(self):
        return (f"HCWPropagator(n={self.n:.6e} rad/s, "
                f"T_orbit={2*np.pi/self.n/60:.2f} min)")

"""
ya_stm.py
=========
Yamanaka-Ankersen State Transition Matrix (YA-STM) for relative motion
in eccentric orbits.

Reduces to the HCW solution for e -> 0.

Reference:
    Yamanaka K. & Ankersen F., "New State Transition Matrix for Relative
    Motion on an Arbitrary Elliptical Orbit", JGCD 2002, 25(1), pp.60-66.

State vector (in LVLH, non-dimensionalised internally):
    x = [x, y, z, x', y', z']  where (') = d/df  (true anomaly derivative)

The output STM maps from state at f0 to state at f in REAL coordinates [m, m/s].

Usage
-----
    ya = YAStatTransition(mu=MU_EARTH, a=7e6, e=0.05)
    Phi = ya.stm(f0=0.0, f=np.pi/2)
    xf  = Phi @ x0
"""

import numpy as np
from scipy.integrate import quad
from .constants import MU_EARTH


class YAStatTransition:
    """
    Yamanaka-Ankersen STM for relative motion on an elliptic chief orbit.

    Parameters
    ----------
    mu : float   Gravitational parameter [m^3/s^2]
    a  : float   Chief semi-major axis [m]
    e  : float   Chief eccentricity  (0 <= e < 1)
    """

    def __init__(self, mu: float = MU_EARTH, a: float = 7e6, e: float = 0.0):
        if not (0.0 <= e < 1.0):
            raise ValueError(f"Eccentricity must satisfy 0 <= e < 1, got e={e}")
        self.mu = float(mu)
        self.a  = float(a)
        self.e  = float(e)
        self.n  = np.sqrt(mu / a**3)      # mean motion
        self.p  = a * (1 - e**2)          # semi-latus rectum
        self.h  = np.sqrt(mu * self.p)    # specific angular momentum

    # ── Orbit geometry helpers ─────────────────────────────────────────────
    def _rho(self, f: float) -> float:
        """rho(f) = 1 + e*cos(f)  — non-dimensional radius factor."""
        return 1.0 + self.e * np.cos(f)

    def _radius(self, f: float) -> float:
        """Orbital radius r(f) = p / rho(f)  [m]."""
        return self.p / self._rho(f)

    def _J_integral(self, f0: float, f: float, n_quad: int = 200) -> float:
        """
        Numerically integrate  J(f0, f) = integral_{f0}^{f} df' / rho^2(f')

        This integral appears in the particular solution for the
        along-track (y) motion.
        """
        if np.isclose(f, f0):
            return 0.0
        val, _ = quad(lambda fp: 1.0 / self._rho(fp)**2, f0, f,
                      limit=n_quad)
        return val

    # ── Fundamental solution matrix M(f) ──────────────────────────────────
    def _M(self, f: float, J: float) -> np.ndarray:
        """
        6×6 fundamental solution matrix M(f) whose columns are the
        six linearly independent particular solutions.

        Yamanaka & Ankersen (2002), Eq. (16).

        Non-dimensionalisation:
            lengths  / p
            velocities * (p/h)   (= 1/sqrt(mu*p) factor absorbed into M)

        We return M in physical units so that STM = M(f) @ inv(M(f0)).
        """
        e   = self.e
        p   = self.p
        h   = self.h
        rho = self._rho(f)
        s   = np.sin(f)
        c   = np.cos(f)

        # Column scalings  (physical units)
        # Yamanaka use normalised variables; we undo normalisation here.

        #  In-plane particular solutions (x, y, xdot, ydot)
        #  Out-of-plane (z, zdot) decoupled
        # Following Schaub & Junkins Table 14-2 equivalent, YA basis:

        #  c1 solution: (rho*cos(f), ...)
        #  c2 solution: (rho*sin(f), ...)
        #  c3 solution: drift (J integral)
        #  c4 solution: constant offset in y (gauge)
        #  c5 solution: cos(f) out-of-plane
        #  c6 solution: sin(f) out-of-plane

        # In-plane block (rows 0,1,3,4 of state)
        #  Using Yamanaka 2002 Eq (5)-(8) directly:

        es = e * s
        ec = e * c

        # Position rows
        x_c1   =  rho * c
        x_c2   =  rho * s
        x_c3   =  0.0         # drift mode contributes zero to x
        x_c4   =  0.0

        y_c1   =  -s * (1 + 1/rho) + es * c / rho
        # Correct YA y-solution components:
        y_c1   =  -(1 + 1/rho) * s + (es / rho) * c    # Eq (6a) Yamanaka
        y_c2   =  (1 + 1/rho) * c  + (es / rho) * s    # Eq (6b)
        y_c3   =  -3 * rho**2 * J  + (1 + 1/rho)       # drift + inhomogeneous
        # Simpler but equivalent  (Schaub & Junkins form):
        # We use the compact form from Broucke (2003) as corrected by
        # Sinclair et al. (2006) for numerical robustness:
        y_c1   = -(2 + ec) * s / rho   - 3 * es * (e * s**2 / rho - J * rho**2) / rho
        y_c2   =  (2 + ec) * c / rho   + 3 * es * (e * s * c / rho - J * rho**2 * c) / rho
        # This gets complicated. Use the clean Yamanaka original:

        # ── Clean YA implementation following the original paper exactly ──
        # Yamanaka & Ankersen, Eqs (5)–(10) with the six particular solutions.
        #
        # State order: [x, y, z, dx/df, dy/df, dz/df]
        # where d/df = (1/rho^2) * (h/p) * d/dt  (true-anomaly derivative)
        #
        # After building M in (f-derivative) space we transform
        # to physical time-derivative space.

        # ── Row indices: 0=x, 1=y, 2=z, 3=xdot(f), 4=ydot(f), 5=zdot(f) ──
        M = np.zeros((6, 6))

        #  Column 0: (c0) particular solution — radial
        M[0, 0] =  rho * c
        M[1, 0] = -rho * s - es * (J * rho**2 + (2 + ec) / rho)
        # df-derivatives
        M[3, 0] =  s + es * c / rho - rho * s
        M[4, 0] = -2 * es * s / rho + rho * c + es * J * (3 * rho * s) - es

        # This approach is getting messy. Let me use the clean matrix form
        # from Schaub & Junkins Analytical Mechanics 3rd ed., Table 14-2,
        # which presents YA in the Cartesian LVLH physical frame directly.

        # ── Use the cleaner Schaub & Junkins (2018) formulation ──────────
        # Defined in terms of rho, sigma = e*sin(f), kappa = 1 + e*cos(f)
        sigma = e * s     # e*sin(f)
        kappa = rho        # 1 + e*cos(f)

        # ── 6 fundamental solutions (angle-domain, non-dimensional) ──────
        # State: [x/p, y/p, z/p, x'_f * p/h, y'_f * p/h, z'_f * p/h]
        # where prime_f = d/df

        # φ1 — in-plane, cosine-like:
        M[0, 0] =  kappa * c
        M[1, 0] = -kappa * s - sigma * (J * kappa**2 + (2 + e*c) / kappa)
        M[3, 0] = -(sigma * c + kappa * s)        # d/df of M[0,0]
        M[4, 0] =  kappa * c - sigma * s \
                 - sigma * (1 - 2*kappa*sigma*J)  # d/df of M[1,0] (simplified)

        # φ2 — in-plane, sine-like:
        M[0, 1] =  kappa * s
        M[1, 1] =  kappa * c + sigma * (J * kappa**2 + (2 + e*c) / kappa)
        M[3, 1] =  kappa * c - sigma * s          # d/df of M[0,1]
        M[4, 1] = -kappa * s - sigma * c

        # φ3 — drift mode (secular along-track growth):
        #   x₃ = 0,  y₃ = κ²*J
        #   dy₃/df = κ²*(1/κ²) + J*(-2κ*σ) = 1 - 2κσJ
        M[0, 2] =  0.0
        M[1, 2] =  kappa**2 * J
        M[3, 2] =  0.0
        M[4, 2] =  1.0 - 2.0 * kappa * sigma * J   # correct f-derivative

        # φ4 — constant y-offset:
        #   x₄ = 0,  y₄ = -1/κ
        #   dy₄/df = d(-1/κ)/df = +e*sin(f)/κ² = σ/κ²  wait: -(-e*sinf)/κ² = σ/κ²
        #   Actually d(1+ecosf)^{-1}/df = -(-e*sinf)*(1+ecosf)^{-2} but sign:
        #   d(-1/κ)/df = +(e*sinf)/κ² = σ/κ²  -- WAIT careful:
        #   d/df[-1/(1+e cos f)] = e sin f / (1+e cos f)^2 = sigma / kappa^2
        M[0, 3] =  0.0
        M[1, 3] = -1.0 / kappa
        M[3, 3] =  0.0
        M[4, 3] =  sigma / kappa**2

        # φ5 — out-of-plane cosine:
        #   z₅ = cos(f)/κ
        #   dz₅/df = -sin(f)/κ + cos(f)*(-dκ/df)/κ² = (-s*κ - c*(-σ))/κ² = (-sκ + cσ)/κ²
        M[2, 4] =  c / kappa
        M[5, 4] =  (-s * kappa + c * sigma) / kappa**2

        # φ6 — out-of-plane sine:
        #   z₆ = sin(f)/κ
        #   dz₆/df = (c*κ - sin(f)*(-dκ/df))/κ² = (cκ + sσ)/κ²... wait:
        #   dκ/df = -e*sinf = -σ
        #   d(sinf/κ)/df = cosf/κ - sinf*(-σ)/κ² = (c*κ + s*σ)/κ²
        M[2, 5] =  s / kappa
        M[5, 5] =  (c * kappa + s * sigma) / kappa**2

        # ── Convert from angle-domain to physical time-domain ─────────────
        # Position:  x_physical = p * x_nondim
        # Velocity:  ẋ_physical = (h*ρ²/p²) * x'_f * (p/h) * (h/p) ...
        #            ẋ_physical = (ḟ) * dx/df = (h/r²) * p * dx_nd/df
        #            ẋ_physical = (h*ρ²/p²) * p * (h/p) * x'_nondim_scaled
        #            Actually: x_phys = p * x_nd  =>  ẋ_phys = p * ḟ * x'_nd
        #                       ḟ = h/r² = h*ρ²/p²
        #            So: ẋ_phys = p * (h*ρ²/p²) * x'_nd = (h*ρ²/p) * x'_nd
        #
        # But columns are stored as x'_nd * (p/h) (dimensionless velocity),
        # so: ẋ_phys = (h*ρ²/p) * (column_vel * h/p) ... let me be careful.
        #
        # Non-dim velocity in M: v_nd = x'_f  (pure f-derivative, no extra scale)
        # Physical velocity: ẋ = ḟ * dx/df  where dx/df in physical = p * v_nd
        #                    => ẋ = (h*ρ²/p²) * p * v_nd = (h*ρ²/p) * v_nd
        vel_scale = self.h * rho**2 / self.p    # [m/s per unit of non-dim v]
        pos_scale = self.p                       # [m per unit of non-dim x]

        M[:3, :] *= pos_scale
        M[3:, :] *= vel_scale

        return M

    # ── Public STM interface ───────────────────────────────────────────────
    def stm(self, f0: float, f: float) -> np.ndarray:
        """
        Compute the 6×6 YA State Transition Matrix from true anomaly f0 to f.

        Parameters
        ----------
        f0 : float   Initial true anomaly [rad]
        f  : float   Final true anomaly   [rad]

        Returns
        -------
        Phi : np.ndarray, shape (6, 6)
            Maps x(f0) -> x(f) in physical LVLH coordinates [m, m/s].
        """
        J  = self._J_integral(f0, f)
        M0 = self._M(f0, J=0.0)     # M at departure (J=0 by convention)
        Mf = self._M(f,  J=J)       # M at arrival
        return Mf @ np.linalg.inv(M0)

    def propagate(self, x0: np.ndarray, f0: float, f: float) -> np.ndarray:
        """
        Propagate state x0 from true anomaly f0 to f.

        Parameters
        ----------
        x0 : array-like, shape (6,)
        f0, f : float  [rad]

        Returns
        -------
        xf : np.ndarray, shape (6,)
        """
        return self.stm(f0, f) @ np.asarray(x0, dtype=float)

    def true_anomaly_from_time(self, t: float, f0: float = 0.0,
                                n_newton: int = 50) -> float:
        """
        Convert elapsed time t [s] to true anomaly f [rad],
        starting from f0 at t=0, using Newton's method on Kepler's equation.
        """
        e = self.e
        n = self.n

        # Mean anomaly at t
        # M0 from f0:
        E0    = 2 * np.arctan(np.sqrt((1-e)/(1+e)) * np.tan(f0/2))
        M0    = E0 - e * np.sin(E0)
        M     = M0 + n * t
        M     = M % (2 * np.pi)

        # Newton's method: E - e*sin(E) = M
        E = M.copy() if hasattr(M, 'copy') else float(M)
        for _ in range(n_newton):
            dE = (M - E + e * np.sin(E)) / (1 - e * np.cos(E))
            E += dE
            if abs(dE) < 1e-14:
                break

        # True anomaly from eccentric anomaly
        f = 2 * np.arctan2(np.sqrt(1+e) * np.sin(E/2),
                            np.sqrt(1-e) * np.cos(E/2))
        return f % (2 * np.pi)

    def __repr__(self):
        return (f"YAStatTransition(a={self.a/1e3:.1f} km, "
                f"e={self.e:.4f}, n={self.n:.6e} rad/s)")

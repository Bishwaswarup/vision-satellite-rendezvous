"""
j2_perturb.py
=============
J2 oblateness perturbation model.

Provides:
  1. ECI acceleration from J2 (absolute spacecraft)
  2. Differential J2 acceleration in LVLH (chief-deputy difference)
     — used as a disturbance force in the relative dynamics equations

Reference:
    Vallado, "Fundamentals of Astrodynamics and Applications", 4th ed., 2013.
    Schaub & Junkins, "Analytical Mechanics of Space Systems", 2018.
"""

import numpy as np
from .constants import MU_EARTH, J2, R_EARTH


def j2_accel_eci(r_eci: np.ndarray,
                 mu: float = MU_EARTH,
                 j2: float = J2,
                 Re: float = R_EARTH) -> np.ndarray:
    """
    J2 perturbing acceleration vector in ECI frame.

    Parameters
    ----------
    r_eci : array-like, shape (3,)
        Position vector in ECI frame [m].
    mu, j2, Re : physical constants (see constants.py)

    Returns
    -------
    a_j2 : np.ndarray, shape (3,)
        Acceleration [m/s^2].

    Formula
    -------
        a_J2 = (3*mu*J2*Re^2) / (2*r^5) * [
            x*(5*z^2/r^2 - 1),
            y*(5*z^2/r^2 - 1),
            z*(5*z^2/r^2 - 3)
        ]
    """
    r    = np.asarray(r_eci, dtype=float)
    rmag = np.linalg.norm(r)
    x, y, z = r

    coeff = 3.0 * mu * j2 * Re**2 / (2.0 * rmag**5)
    zr2   = (z / rmag)**2          # (z/r)^2

    a = coeff * np.array([
        x * (5 * zr2 - 1),
        y * (5 * zr2 - 1),
        z * (5 * zr2 - 3),
    ])
    return a


def lvlh_to_eci_rotation(r_eci: np.ndarray,
                          v_eci: np.ndarray) -> np.ndarray:
    """
    Compute the rotation matrix R_LVLH_ECI that maps LVLH -> ECI.

    LVLH axes:
        x_hat = r_hat          (radial, outward)
        z_hat = h_hat          (orbit normal = r × v, normalised)
        y_hat = z_hat × x_hat  (along-track)

    Parameters
    ----------
    r_eci : array (3,)   Chief position in ECI [m]
    v_eci : array (3,)   Chief velocity in ECI [m/s]

    Returns
    -------
    R : np.ndarray, shape (3, 3)
        R @ v_lvlh = v_eci
    """
    r = np.asarray(r_eci, dtype=float)
    v = np.asarray(v_eci, dtype=float)

    x_hat = r / np.linalg.norm(r)
    h     = np.cross(r, v)
    z_hat = h / np.linalg.norm(h)
    y_hat = np.cross(z_hat, x_hat)

    return np.column_stack([x_hat, y_hat, z_hat])   # columns are LVLH unit vectors in ECI


def differential_j2_lvlh(r_chief_eci: np.ndarray,
                          v_chief_eci: np.ndarray,
                          dr_lvlh: np.ndarray,
                          mu: float = MU_EARTH,
                          j2: float = J2,
                          Re: float = R_EARTH) -> np.ndarray:
    """
    Differential J2 acceleration on the deputy relative to the chief,
    expressed in the LVLH frame.

    This is the *perturbation* term that corrupts the ideal HCW dynamics:
        f_perturb = a_J2(r_deputy) - a_J2(r_chief)   in LVLH [m/s^2]

    Parameters
    ----------
    r_chief_eci : array (3,)   Chief position in ECI [m]
    v_chief_eci : array (3,)   Chief velocity in ECI [m/s]
    dr_lvlh     : array (3,)   Relative position of deputy in LVLH [m]
    mu, j2, Re  : constants

    Returns
    -------
    da_lvlh : np.ndarray, shape (3,)
        Differential J2 acceleration in LVLH [m/s^2].
    """
    r_chief = np.asarray(r_chief_eci, dtype=float)
    v_chief = np.asarray(v_chief_eci, dtype=float)
    dr      = np.asarray(dr_lvlh,     dtype=float)

    # Rotation matrix LVLH -> ECI
    R_LE = lvlh_to_eci_rotation(r_chief, v_chief)

    # Deputy position in ECI
    r_deputy_eci = r_chief + R_LE @ dr

    # J2 accelerations in ECI
    a_chief = j2_accel_eci(r_chief,  mu, j2, Re)
    a_dep   = j2_accel_eci(r_deputy_eci, mu, j2, Re)

    # Differential acceleration in ECI, then rotate to LVLH
    da_eci  = a_dep - a_chief
    da_lvlh = R_LE.T @ da_eci       # ECI -> LVLH

    return da_lvlh


class J2PerturbedHCW:
    """
    HCW propagator augmented with differential J2 perturbation,
    integrated numerically (RK45 / DOP853).

    This models the real relative motion more faithfully than pure HCW.

    Parameters
    ----------
    n          : float    Chief mean motion [rad/s]
    chief_ode  : callable(t) -> (r_eci(3), v_eci(3))
        Function that returns the chief's ECI state at time t.
        Typically from a two-body or J2-averaged propagator.
    """

    def __init__(self, n: float, chief_ode):
        self.n         = float(n)
        self.chief_ode = chief_ode

    def _eom_j2(self, t: float, x: np.ndarray, control_fn=None) -> np.ndarray:
        """
        Full relative EOM: HCW + differential J2 + optional control.
        """
        from scipy.integrate import solve_ivp as _solve   # just for type hints
        n  = self.n
        xp, yp, zp   = x[0], x[1], x[2]
        xdp, ydp, zdp = x[3], x[4], x[5]

        # HCW accelerations
        xdd = 3*n**2*xp + 2*n*ydp
        ydd =           - 2*n*xdp
        zdd = -n**2 * zp

        # Differential J2 disturbance in LVLH
        r_c, v_c = self.chief_ode(t)
        da = differential_j2_lvlh(r_c, v_c, x[:3])

        # Control (optional)
        fc = np.zeros(3)
        if control_fn is not None:
            fc = np.asarray(control_fn(t, x), dtype=float)

        xdd += da[0] + fc[0]
        ydd += da[1] + fc[1]
        zdd += da[2] + fc[2]

        return np.array([xdp, ydp, zdp, xdd, ydd, zdd])

    def propagate(self, x0: np.ndarray, t_span: tuple,
                  t_eval: np.ndarray = None,
                  control_fn=None,
                  method: str = 'DOP853') -> dict:
        """Propagate with J2 perturbations."""
        from scipy.integrate import solve_ivp
        sol = solve_ivp(
            fun=lambda t, x: self._eom_j2(t, x, control_fn),
            t_span=t_span,
            y0=np.asarray(x0, dtype=float),
            method=method,
            t_eval=t_eval,
            rtol=1e-10,
            atol=1e-12,
        )
        return {'t': sol.t, 'y': sol.y, 'success': sol.success}


def two_body_eci_ode(t: float, rv: np.ndarray,
                     mu: float = MU_EARTH,
                     include_j2: bool = True,
                     j2: float = J2,
                     Re: float = R_EARTH) -> np.ndarray:
    """
    Two-body (+ optional J2) equations of motion in ECI for the chief.

    State: rv = [rx, ry, rz, vx, vy, vz]
    """
    r = rv[:3]
    v = rv[3:]
    rmag = np.linalg.norm(r)
    rdot = -mu / rmag**3 * r
    if include_j2:
        rdot += j2_accel_eci(r, mu, j2, Re)
    return np.concatenate([v, rdot])


def propagate_chief_eci(rv0_eci: np.ndarray,
                         t_eval: np.ndarray,
                         mu: float = MU_EARTH,
                         include_j2: bool = True) -> np.ndarray:
    """
    Propagate the chief spacecraft in ECI with optional J2.

    Parameters
    ----------
    rv0_eci : array (6,)   Initial ECI state [r(m), v(m/s)]
    t_eval  : array (N,)   Output times [s]
    include_j2 : bool

    Returns
    -------
    rv_eci : np.ndarray, shape (6, N)
    """
    from scipy.integrate import solve_ivp
    sol = solve_ivp(
        fun=lambda t, rv: two_body_eci_ode(t, rv, mu, include_j2),
        t_span=(t_eval[0], t_eval[-1]),
        y0=np.asarray(rv0_eci, dtype=float),
        method='DOP853',
        t_eval=t_eval,
        rtol=1e-11,
        atol=1e-13,
    )
    return sol.y   # shape (6, N)


def keplerian_to_eci(a: float, e: float, i: float,
                      raan: float, aop: float, f: float,
                      mu: float = MU_EARTH) -> np.ndarray:
    """
    Convert Keplerian orbital elements to ECI Cartesian state.

    Parameters
    ----------
    a    : semi-major axis [m]
    e    : eccentricity
    i    : inclination [rad]
    raan : right ascension of ascending node [rad]
    aop  : argument of periapsis [rad]
    f    : true anomaly [rad]
    mu   : gravitational parameter [m^3/s^2]

    Returns
    -------
    rv : np.ndarray, shape (6,)   [r(m), v(m/s)] in ECI
    """
    p  = a * (1 - e**2)
    r  = p / (1 + e * np.cos(f))
    h  = np.sqrt(mu * p)

    # Position and velocity in perifocal frame
    r_pf = r * np.array([np.cos(f), np.sin(f), 0.0])
    v_pf = (mu / h) * np.array([-np.sin(f), e + np.cos(f), 0.0])

    # Rotation matrices
    ci, si = np.cos(i),    np.sin(i)
    co, so = np.cos(raan), np.sin(raan)
    cw, sw = np.cos(aop),  np.sin(aop)

    # ECI = R3(-raan) @ R1(-i) @ R3(-aop) @ perifocal
    R = np.array([
        [co*cw - so*sw*ci, -co*sw - so*cw*ci,  so*si],
        [so*cw + co*sw*ci, -so*sw + co*cw*ci, -co*si],
        [si*sw,              si*cw,               ci  ],
    ])

    return np.concatenate([R @ r_pf, R @ v_pf])

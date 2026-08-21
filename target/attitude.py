"""
attitude.py
===========
Rigid-body attitude dynamics for a non-cooperative tumbling debris object.

Implements:
  1. Euler's rotational equations of motion (body frame)
  2. Quaternion kinematic equation
  3. Full 7-state integrator  [q(4), ω(3)]
  4. Conservation checks (rotational kinetic energy, angular momentum)

State vector:
    x_att = [q0, q1, q2, q3, ω1, ω2, ω3]   shape (7,)

    q = unit quaternion   (body ← inertial rotation)
    ω = angular velocity in body frame  [rad/s]

Equations of motion:
    q̇  = ½ Ξ(q) ω                         (kinematics)
    Iω̇ = -ω × (Iω) + τ                   (Euler's equations)

For torque-free debris:  τ = 0

Reference:
    Schaub & Junkins, "Analytical Mechanics of Space Systems", 3rd ed., Ch. 3
    Hughes, P.C., "Spacecraft Attitude Dynamics", Dover, 2004
"""

import numpy as np
from scipy.integrate import solve_ivp
from .quaternion import qdot, qnormalize, q_to_dcm, q_to_euler321, xi_matrix


class RigidBodyAttitude:
    """
    Torque-free (and optionally torque-driven) rigid-body attitude integrator.

    Parameters
    ----------
    inertia : array-like, shape (3,) or (3, 3)
        If shape (3,): principal moments of inertia [I1, I2, I3] in kg·m².
        If shape (3,3): full inertia tensor (off-diagonal terms included).
    """

    def __init__(self, inertia):
        I = np.asarray(inertia, dtype=float)
        if I.ndim == 1:
            assert len(I) == 3, "Principal moments must be a 3-vector"
            self.I     = np.diag(I)
            self.I_inv = np.diag(1.0 / I)
            self.principal = True
        else:
            assert I.shape == (3, 3), "Inertia tensor must be 3×3"
            self.I     = I
            self.I_inv = np.linalg.inv(I)
            self.principal = False

    # ── Equations of Motion ───────────────────────────────────────────────────
    def _eom(self, t: float, x: np.ndarray,
             torque_fn=None) -> np.ndarray:
        """
        RHS of the 7-DOF attitude ODE.

        Parameters
        ----------
        t        : float          current time [s]
        x        : array (7,)     [q0, q1, q2, q3, ω1, ω2, ω3]
        torque_fn: callable(t, x) -> array(3,) or None
                   External torque in body frame [N·m]

        Returns
        -------
        xdot : np.ndarray, shape (7,)
        """
        q = x[:4]
        w = x[4:]

        # External torque (zero for torque-free tumbling)
        tau = np.zeros(3)
        if torque_fn is not None:
            tau = np.asarray(torque_fn(t, x), dtype=float)

        # Quaternion kinematics:  q̇ = ½ Ξ(q) ω
        q_dot = qdot(q, w)

        # Euler's rotational equations:  I ω̇ = τ - ω × (I ω)
        Iw    = self.I @ w
        w_dot = self.I_inv @ (tau - np.cross(w, Iw))

        return np.concatenate([q_dot, w_dot])

    # ── Integration ───────────────────────────────────────────────────────────
    def propagate(self,
                  q0: np.ndarray,
                  w0: np.ndarray,
                  t_span: tuple,
                  t_eval: np.ndarray = None,
                  torque_fn=None,
                  method: str = 'DOP853',
                  rtol: float = 1e-10,
                  atol: float = 1e-12,
                  renormalize: bool = True) -> dict:
        """
        Integrate attitude dynamics from initial conditions.

        Parameters
        ----------
        q0       : array (4,)   Initial unit quaternion  [q0, q1, q2, q3]
        w0       : array (3,)   Initial angular velocity [rad/s] (body frame)
        t_span   : (t0, tf)     Time interval [s]
        t_eval   : array (N,)   Output times (None = adaptive)
        torque_fn: callable or None
        method   : 'RK45' | 'DOP853'
        renormalize: bool  If True, renormalise q at each output step

        Returns
        -------
        dict with keys:
            't'   : time array [s]
            'q'   : quaternion array, shape (4, N)
            'w'   : angular velocity array, shape (3, N)
            'euler': Euler 321 angles array, shape (3, N)  [rad]
            'success': bool
        """
        x0  = np.concatenate([
            qnormalize(np.asarray(q0, float)),
            np.asarray(w0, float)
        ])

        sol = solve_ivp(
            fun=lambda t, x: self._eom(t, x, torque_fn),
            t_span=t_span,
            y0=x0,
            method=method,
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
            dense_output=False,
        )

        q_arr = sol.y[:4, :]
        w_arr = sol.y[4:, :]

        # Re-normalise quaternion columns (suppress numerical drift)
        if renormalize:
            norms = np.linalg.norm(q_arr, axis=0)
            q_arr = q_arr / norms

        # Euler angles at each step
        euler_arr = np.column_stack([q_to_euler321(q_arr[:, k])
                                     for k in range(q_arr.shape[1])])

        return {
            't':      sol.t,
            'q':      q_arr,
            'w':      w_arr,
            'euler':  euler_arr,
            'success': sol.success,
        }

    # ── Conservation quantities ───────────────────────────────────────────────
    def rotational_ke(self, w: np.ndarray) -> float:
        """
        Rotational kinetic energy  T = ½ ωᵀ I ω  [J].
        Conserved for torque-free motion.
        """
        w = np.asarray(w, dtype=float)
        return 0.5 * float(w @ self.I @ w)

    def angular_momentum_body(self, w: np.ndarray) -> np.ndarray:
        """
        Angular momentum in body frame  h = I ω  [kg·m²/s].
        """
        return self.I @ np.asarray(w, dtype=float)

    def angular_momentum_inertial(self, q: np.ndarray,
                                   w: np.ndarray) -> np.ndarray:
        """
        Angular momentum in inertial frame  H = C(q) h  [kg·m²/s].
        Magnitude conserved for torque-free motion.
        """
        h_body = self.angular_momentum_body(w)
        return q_to_dcm(q) @ h_body

    def check_conservation(self, result: dict,
                            rtol: float = 1e-6) -> dict:
        """
        Check energy and angular momentum conservation over a propagation.

        Parameters
        ----------
        result : dict output from propagate()
        rtol   : relative tolerance for pass/fail

        Returns
        -------
        report : dict with keys 'energy_ok', 'momentum_ok', 'max_E_err',
                 'max_H_err', 'T0', 'H0_mag'
        """
        q_arr = result['q']
        w_arr = result['w']
        N     = q_arr.shape[1]

        T0    = self.rotational_ke(w_arr[:, 0])
        H0    = self.angular_momentum_inertial(q_arr[:, 0], w_arr[:, 0])
        H0mag = np.linalg.norm(H0)

        E_errs = []
        H_errs = []
        for k in range(N):
            T_k   = self.rotational_ke(w_arr[:, k])
            H_k   = self.angular_momentum_inertial(q_arr[:, k], w_arr[:, k])
            E_errs.append(abs(T_k - T0) / (abs(T0) + 1e-30))
            H_errs.append(abs(np.linalg.norm(H_k) - H0mag) / (H0mag + 1e-30))

        max_E = max(E_errs)
        max_H = max(H_errs)

        return {
            'energy_ok':   max_E < rtol,
            'momentum_ok': max_H < rtol,
            'max_E_err':   max_E,
            'max_H_err':   max_H,
            'T0':          T0,
            'H0_mag':      H0mag,
        }

    def __repr__(self):
        diag = np.diag(self.I)
        return (f"RigidBodyAttitude(I=[{diag[0]:.2f}, {diag[1]:.2f}, "
                f"{diag[2]:.2f}] kg·m²)")


# ── Debris body presets ───────────────────────────────────────────────────────

def ariane_upper_stage() -> RigidBodyAttitude:
    """
    Approximate inertia tensor for an Ariane 44L upper stage (H10 engine):
        - Dry mass    : ~1200 kg
        - Length      : ~8.0 m
        - Outer radius: ~1.4 m
    Modelled as a hollow cylinder.

    I_axial      = ½ m r²              (spin axis along x)
    I_transverse = m(3r² + L²) / 12   (tumble axes y, z)
    """
    m  = 1200.0    # kg
    r  = 1.4       # m
    L  = 8.0       # m
    I1 = 0.5 * m * r**2                 # ~1176 kg·m²  axial (symmetry)
    I2 = m * (3 * r**2 + L**2) / 12    # ~7280 kg·m²  transverse
    I3 = I2
    return RigidBodyAttitude(np.array([I1, I2, I3]))


def cubesat_3u() -> RigidBodyAttitude:
    """
    Approximate inertia for a 3U CubeSat (10×10×30 cm, ~4 kg).
    Modelled as a rectangular box.

    I_i = m/12 * (a² + b²)  for each axis
    """
    m  = 4.0      # kg
    lx, ly, lz = 0.30, 0.10, 0.10   # m  (30 cm along x)
    I1 = m / 12 * (ly**2 + lz**2)   # ~0.0067 kg·m²
    I2 = m / 12 * (lx**2 + lz**2)   # ~0.034  kg·m²
    I3 = m / 12 * (lx**2 + ly**2)   # ~0.034  kg·m²
    return RigidBodyAttitude(np.array([I1, I2, I3]))


def custom_body(mass: float, length: float, radius: float) -> RigidBodyAttitude:
    """
    Solid cylinder approximation for an arbitrary debris body.

    Parameters
    ----------
    mass   : float  [kg]
    length : float  [m]
    radius : float  [m]

    Returns
    -------
    RigidBodyAttitude
    """
    I1 = 0.5 * mass * radius**2
    I2 = mass * (3 * radius**2 + length**2) / 12
    I3 = I2
    return RigidBodyAttitude(np.array([I1, I2, I3]))

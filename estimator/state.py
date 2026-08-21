"""
state.py
========
State vector definition, process model, and measurement model for
Phase 5 — EKF / UKF relative pose tracking.

State
-----
  x  ∈ R^13  : [r(3), v(3), q(4), ω(3)]
    r  : relative position,  LVLH frame  [m]
    v  : relative velocity,  LVLH frame  [m/s]
    q  : target attitude quaternion, body→LVLH  [qx, qy, qz, qw] (scalar last)
    ω  : target angular velocity, body frame  [rad/s]

Error state (for EKF covariance propagation)
---------------------------------------------
  δx ∈ R^12 : [δr(3), δv(3), δα(3), δω(3)]
    δα : attitude error as a rotation vector (small angle)

Orbital / target constants
--------------------------
  n  : mean orbital motion [rad/s]  (ISS ≈ 1.14 × 10⁻³)
  J  : Ariane 44L inertia tensor  [kg·m²]  (diagonal, tumbling regime)
"""

import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────

# ISS-like orbit: h ≈ 400 km, T ≈ 92 min
N_ORBITAL = 1.1368e-3          # rad/s   (μ/a³)^(1/2), a = 6778 km

# Ariane 44L upper-stage inertia (approximate, dry mass ~1200 kg)
J_ARIANE = np.diag([1800.0, 1800.0, 360.0])   # kg·m²
J_INV    = np.linalg.inv(J_ARIANE)


# ── Quaternion utilities ───────────────────────────────────────────────────────

def quat_mult(p, q):
    """Hamilton product p ⊗ q.  Convention: [x,y,z,w], scalar last."""
    px, py, pz, pw = p
    qx, qy, qz, qw = q
    return np.array([
        pw*qx + px*qw + py*qz - pz*qy,
        pw*qy - px*qz + py*qw + pz*qx,
        pw*qz + px*qy - py*qx + pz*qw,
        pw*qw - px*qx - py*qy - pz*qz,
    ])


def quat_norm(q):
    return q / np.linalg.norm(q)


def quat_to_dcm(q):
    """Quaternion [x,y,z,w] → 3×3 DCM (body→frame)."""
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),       1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),       2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def dcm_to_quat(R):
    """3×3 DCM → quaternion [x,y,z,w].  Shepperd method."""
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2,1] - R[1,2]) * s
        y = (R[0,2] - R[2,0]) * s
        z = (R[1,0] - R[0,1]) * s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        w = (R[2,1] - R[1,2]) / s
        x = 0.25 * s
        y = (R[0,1] + R[1,0]) / s
        z = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        w = (R[0,2] - R[2,0]) / s
        x = (R[0,1] + R[1,0]) / s
        y = 0.25 * s
        z = (R[1,2] + R[2,1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        w = (R[1,0] - R[0,1]) / s
        x = (R[0,2] + R[2,0]) / s
        y = (R[1,2] + R[2,1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w])


def rotvec_to_quat(rv):
    """Rotation vector → quaternion [x,y,z,w]."""
    angle = np.linalg.norm(rv)
    if angle < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = rv / angle
    s = np.sin(angle / 2)
    return np.array([axis[0]*s, axis[1]*s, axis[2]*s, np.cos(angle / 2)])


def quat_to_rotvec(q):
    """Quaternion [x,y,z,w] → rotation vector."""
    q = q / np.linalg.norm(q)
    if q[3] < 0:
        q = -q
    w = np.clip(q[3], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    s = np.sin(angle / 2)
    if s < 1e-12:
        return np.zeros(3)
    return q[:3] / s * angle


# ── State packing / unpacking ──────────────────────────────────────────────────

def pack_state(r, v, q, omega):
    """Pack (r,v,q,ω) → x (13,)."""
    return np.concatenate([r, v, q, omega])


def unpack_state(x):
    """x (13,) → (r, v, q, ω)."""
    return x[:3], x[3:6], x[6:10], x[10:13]


# ── HCW skew matrix ───────────────────────────────────────────────────────────

def _hcw_A(n):
    """
    Continuous-time HCW state matrix for [r, v] (6×6).
    x=radial, y=along-track, z=cross-track.
    """
    return np.array([
        [0,    0, 0,  1,    0, 0],
        [0,    0, 0,  0,    1, 0],
        [0,    0, 0,  0,    0, 1],
        [3*n**2, 0, 0,  0,  2*n, 0],
        [0,    0, 0, -2*n,  0, 0],
        [0,    0,-n**2, 0,  0, 0],
    ])


# ── Continuous dynamics (RHS) ──────────────────────────────────────────────────

def f_continuous(x, n=N_ORBITAL, J_inv=J_INV):
    """
    Continuous dynamics ẋ = f(x).

    HCW for translation; torque-free Euler for attitude.
    """
    r, v, q, omega = unpack_state(x)
    q = quat_norm(q)

    A = _hcw_A(n)
    rv_dot = A @ np.concatenate([r, v])    # (6,)

    # Quaternion kinematics: q̇ = 0.5 * Ω(ω) @ q
    # Ω(ω) is the right-quaternion-multiplication matrix for [0,ω]
    wx, wy, wz = omega
    Omega = 0.5 * np.array([
        [ 0,   wz, -wy,  wx],
        [-wz,  0,   wx,  wy],
        [ wy, -wx,  0,   wz],
        [-wx, -wy, -wz,  0 ],
    ])
    q_dot = Omega @ q                       # (4,)

    # Torque-free Euler: ω̇ = J⁻¹(-ω × Jω)
    Jw = J_ARIANE @ omega
    omega_dot = J_inv @ (-np.cross(omega, Jw))   # (3,)

    return np.concatenate([rv_dot, q_dot, omega_dot])


def propagate_rk4(x, dt, n=N_ORBITAL):
    """RK4 integration of f_continuous over one step dt."""
    k1 = f_continuous(x,        n)
    k2 = f_continuous(x + dt/2*k1, n)
    k3 = f_continuous(x + dt/2*k2, n)
    k4 = f_continuous(x + dt*k3,   n)
    x_new = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    # Re-normalise quaternion
    x_new[6:10] = quat_norm(x_new[6:10])
    return x_new


# ── Linearised process Jacobian (error state, 12×12) ─────────────────────────

def process_jacobian(x, dt, n=N_ORBITAL):
    """
    Linearised discrete-time state transition F such that
        δx_{k+1} ≈ F @ δx_k
    Error state: δx = [δr, δv, δα, δω]  (12,)
    δα is the attitude error rotation vector.

    Computed via finite differences for numerical robustness.
    """
    eps = 1e-5
    n12 = 12
    F = np.zeros((n12, n12))
    x_nom = propagate_rk4(x, dt, n)

    def apply_delta(delta):
        """Add error-state perturbation to nominal state."""
        dr, dv, da, dw = delta[:3], delta[3:6], delta[6:9], delta[9:12]
        r, v, q, omega = unpack_state(x)
        q_pert = quat_mult(rotvec_to_quat(da), q)
        return pack_state(r + dr, v + dv, q_pert, omega + dw)

    def state_to_error(x_pert, x_ref):
        """Map perturbed full state → error state relative to reference."""
        r_p, v_p, q_p, w_p = unpack_state(x_pert)
        r_r, v_r, q_r, w_r = unpack_state(x_ref)
        # Attitude error: δq = q_pert ⊗ q_ref⁻¹,  then → rotvec
        q_r_inv = np.array([-q_r[0], -q_r[1], -q_r[2], q_r[3]])
        dq = quat_mult(q_p, q_r_inv)
        da = quat_to_rotvec(dq)
        return np.concatenate([r_p - r_r, v_p - v_r, da, w_p - w_r])

    for i in range(n12):
        delta_plus  = np.zeros(n12); delta_plus[i]  =  eps
        delta_minus = np.zeros(n12); delta_minus[i] = -eps
        x_p = propagate_rk4(apply_delta(delta_plus),  dt, n)
        x_m = propagate_rk4(apply_delta(delta_minus), dt, n)
        de_p = state_to_error(x_p, x_nom)
        de_m = state_to_error(x_m, x_nom)
        F[:, i] = (de_p - de_m) / (2*eps)

    return F


# ── Measurement model ──────────────────────────────────────────────────────────

def h_measurement(x):
    """
    Measurement function h: state → z (6,).
    z = [t_meas(3),  r_meas(3)]
      t_meas : relative position (what EPnP returns as translation)
      r_meas : rotation vector from DCM of target attitude
    """
    r, v, q, omega = unpack_state(x)
    R = quat_to_dcm(q)
    rv = quat_to_rotvec(q)
    return np.concatenate([r, rv])


def measurement_jacobian(x):
    """
    Measurement Jacobian H (6×12) via finite differences.
    Maps error state δx → measurement residual δz.
    """
    eps = 1e-5
    z0 = h_measurement(x)
    r, v, q, omega = unpack_state(x)

    H = np.zeros((6, 12))
    for i in range(12):
        # Perturb error state
        de = np.zeros(12)
        de[i] = eps

        dr, dv, da, dw = de[:3], de[3:6], de[6:9], de[9:12]
        q_pert = quat_norm(quat_mult(rotvec_to_quat(da), q))
        x_pert = pack_state(r + dr, v + dv, q_pert, omega + dw)

        de_neg = np.zeros(12)
        de_neg[i] = -eps
        dr2, dv2, da2, dw2 = de_neg[:3], de_neg[3:6], de_neg[6:9], de_neg[9:12]
        q_pert2 = quat_norm(quat_mult(rotvec_to_quat(da2), q))
        x_pert2 = pack_state(r + dr2, v + dv2, q_pert2, omega + dw2)

        H[:, i] = (h_measurement(x_pert) - h_measurement(x_pert2)) / (2*eps)

    return H


# ── Default noise matrices ─────────────────────────────────────────────────────

def default_process_noise(dt, pos_std=0.1, vel_std=0.01,
                           att_std=1e-4, rate_std=1e-5):
    """
    Diagonal process noise Q (12×12).
    Scaled by dt so noise density is consistent across step sizes.
    """
    diag = np.concatenate([
        np.full(3, pos_std**2  * dt),
        np.full(3, vel_std**2  * dt),
        np.full(3, att_std**2  * dt),
        np.full(3, rate_std**2 * dt),
    ])
    return np.diag(diag)


def default_measurement_noise(pos_std=0.5, att_std=0.05):
    """
    Diagonal measurement noise R (6×6).
    pos_std [m] — position measurement uncertainty
    att_std [rad] — attitude (Rodrigues) measurement uncertainty
    """
    diag = np.concatenate([
        np.full(3, pos_std**2),
        np.full(3, att_std**2),
    ])
    return np.diag(diag)

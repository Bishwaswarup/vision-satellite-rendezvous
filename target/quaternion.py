"""
quaternion.py
=============
Pure-Python quaternion utilities for attitude kinematics.

Convention: q = [q0, q1, q2, q3]  where q0 is the SCALAR part.
Unit quaternion: ||q|| = 1.

All functions operate on numpy arrays. No external attitude libraries
are required — everything is built from first principles.

Reference:
    Shuster M.D., "A Survey of Attitude Representations", JAS 1993.
    Schaub & Junkins, "Analytical Mechanics of Space Systems", 3rd ed.
"""

import numpy as np


# ── Basic operations ──────────────────────────────────────────────────────────

def qnorm(q: np.ndarray) -> float:
    """Return ||q||."""
    return float(np.linalg.norm(q))


def qnormalize(q: np.ndarray) -> np.ndarray:
    """Return q / ||q||."""
    return np.asarray(q, dtype=float) / np.linalg.norm(q)


def qconjugate(q: np.ndarray) -> np.ndarray:
    """
    Quaternion conjugate: q* = [q0, -q_vec].
    For unit quaternions this equals the inverse.
    """
    q = np.asarray(q, dtype=float)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def qinverse(q: np.ndarray) -> np.ndarray:
    """
    Quaternion inverse: q^{-1} = q* / ||q||^2.
    For unit quaternions: q^{-1} = q*.
    """
    return qconjugate(q) / np.dot(q, q)


def qmultiply(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Hamilton product p ⊗ q.

    Convention: p ⊗ q represents first rotating by q, then by p
    (right-to-left composition, same as rotation matrices).

    p = [p0, p1, p2, p3],  q = [q0, q1, q2, q3]
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p0, p1, p2, p3 = p
    q0, q1, q2, q3 = q
    return np.array([
        p0*q0 - p1*q1 - p2*q2 - p3*q3,
        p0*q1 + p1*q0 + p2*q3 - p3*q2,
        p0*q2 - p1*q3 + p2*q0 + p3*q1,
        p0*q3 + p1*q2 - p2*q1 + p3*q0,
    ])


# ── Rotation matrix ───────────────────────────────────────────────────────────

def q_to_dcm(q: np.ndarray) -> np.ndarray:
    """
    Direction Cosine Matrix (DCM) from unit quaternion.

    C = (q0²  - ||q_vec||²) I₃  +  2 q_vec q_vec^T  +  2 q0 [q_vec]×

    Maps a vector from the body frame to the inertial frame:
        v_inertial = C @ v_body

    Parameters
    ----------
    q : array (4,)  unit quaternion [q0, q1, q2, q3]

    Returns
    -------
    C : np.ndarray, shape (3, 3)
    """
    q = qnormalize(np.asarray(q, dtype=float))
    q0, q1, q2, q3 = q

    C = np.array([
        [1 - 2*(q2**2 + q3**2),   2*(q1*q2 - q0*q3),   2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3),   1 - 2*(q1**2 + q3**2),   2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2),   2*(q2*q3 + q0*q1),   1 - 2*(q1**2 + q2**2)],
    ])
    return C


def dcm_to_q(C: np.ndarray) -> np.ndarray:
    """
    Convert a rotation matrix C to a unit quaternion (Shepperd's method).
    Returns the quaternion with positive scalar part (q0 ≥ 0).
    """
    C = np.asarray(C, dtype=float)
    trace = np.trace(C)

    if trace > 0:
        s  = 0.5 / np.sqrt(trace + 1.0)
        q0 = 0.25 / s
        q1 = (C[2, 1] - C[1, 2]) * s
        q2 = (C[0, 2] - C[2, 0]) * s
        q3 = (C[1, 0] - C[0, 1]) * s
    elif C[0, 0] > C[1, 1] and C[0, 0] > C[2, 2]:
        s  = 2.0 * np.sqrt(1.0 + C[0, 0] - C[1, 1] - C[2, 2])
        q0 = (C[2, 1] - C[1, 2]) / s
        q1 = 0.25 * s
        q2 = (C[0, 1] + C[1, 0]) / s
        q3 = (C[0, 2] + C[2, 0]) / s
    elif C[1, 1] > C[2, 2]:
        s  = 2.0 * np.sqrt(1.0 + C[1, 1] - C[0, 0] - C[2, 2])
        q0 = (C[0, 2] - C[2, 0]) / s
        q1 = (C[0, 1] + C[1, 0]) / s
        q2 = 0.25 * s
        q3 = (C[1, 2] + C[2, 1]) / s
    else:
        s  = 2.0 * np.sqrt(1.0 + C[2, 2] - C[0, 0] - C[1, 1])
        q0 = (C[1, 0] - C[0, 1]) / s
        q1 = (C[0, 2] + C[2, 0]) / s
        q2 = (C[1, 2] + C[2, 1]) / s
        q3 = 0.25 * s

    q = np.array([q0, q1, q2, q3])
    return qnormalize(q) if q[0] >= 0 else qnormalize(-q)


# ── Kinematics matrix Ξ(q) ───────────────────────────────────────────────────

def xi_matrix(q: np.ndarray) -> np.ndarray:
    """
    4×3 kinematic matrix Ξ(q) such that:
        q̇ = ½ Ξ(q) ω       (body-frame angular velocity ω [rad/s])

    Ξ(q) = [ -q_vec^T ]
            [ q0 I₃ + [q_vec]× ]

    Parameters
    ----------
    q : array (4,)  unit quaternion

    Returns
    -------
    Xi : np.ndarray, shape (4, 3)
    """
    q = np.asarray(q, dtype=float)
    q0, q1, q2, q3 = q
    return np.array([
        [-q1, -q2, -q3],
        [ q0, -q3,  q2],
        [ q3,  q0, -q1],
        [-q2,  q1,  q0],
    ])


def qdot(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """
    Time derivative of unit quaternion given body-frame angular velocity.

        q̇ = ½ Ξ(q) ω

    Parameters
    ----------
    q     : array (4,)  unit quaternion
    omega : array (3,)  angular velocity in body frame [rad/s]

    Returns
    -------
    dq : np.ndarray, shape (4,)
    """
    return 0.5 * xi_matrix(q) @ np.asarray(omega, dtype=float)


# ── Euler angles ──────────────────────────────────────────────────────────────

def q_to_euler321(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to Euler angles (3-2-1 / yaw-pitch-roll convention).

    Returns [roll φ, pitch θ, yaw ψ] in radians.
    Singularity at pitch θ = ±90°.
    """
    q = qnormalize(np.asarray(q, dtype=float))
    q0, q1, q2, q3 = q

    roll  = np.arctan2(2*(q0*q1 + q2*q3), 1 - 2*(q1**2 + q2**2))
    pitch = np.arcsin(np.clip(2*(q0*q2 - q3*q1), -1.0, 1.0))
    yaw   = np.arctan2(2*(q0*q3 + q1*q2), 1 - 2*(q2**2 + q3**2))

    return np.array([roll, pitch, yaw])


def euler321_to_q(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convert 3-2-1 Euler angles to unit quaternion.

    Parameters
    ----------
    roll, pitch, yaw : floats [rad]
    """
    cr, sr = np.cos(roll/2),  np.sin(roll/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cy, sy = np.cos(yaw/2),   np.sin(yaw/2)

    q0 = cr*cp*cy + sr*sp*sy
    q1 = sr*cp*cy - cr*sp*sy
    q2 = cr*sp*cy + sr*cp*sy
    q3 = cr*cp*sy - sr*sp*cy

    return qnormalize(np.array([q0, q1, q2, q3]))


# ── Angular error ─────────────────────────────────────────────────────────────

def attitude_error_deg(q_true: np.ndarray, q_est: np.ndarray) -> float:
    """
    Principal rotation angle error between two quaternions [degrees].

        θ_err = 2 * arccos(|q_true · q_est|)
    """
    # 2*arccos(|q.q|) loses about half the significant digits near dot = 1
    # and returns exactly 0.0 below ~1 microdegree, which floors any
    # convergence plot.  Taking the angle of the relative quaternion with
    # atan2 is exact across the whole range.
    q1 = qnormalize(np.asarray(q_true, float))
    q2 = qnormalize(np.asarray(q_est,  float))
    dq = qmultiply(q1, qconjugate(q2))
    return float(np.degrees(2.0 * np.arctan2(np.linalg.norm(dq[1:]),
                                             abs(dq[0]))))


# ── Rotation of vectors ───────────────────────────────────────────────────────

def rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Rotate vector v from body frame to inertial frame using quaternion q.

        v_inertial = C(q) @ v_body
    """
    return q_to_dcm(q) @ np.asarray(v, dtype=float)


def rotate_vector_inv(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Rotate vector v from inertial frame to body frame.

        v_body = C(q)^T @ v_inertial
    """
    return q_to_dcm(q).T @ np.asarray(v, dtype=float)

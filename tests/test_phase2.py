"""
test_phase2.py
==============
Unit tests for Phase 2 — Target Kinematics & Tumbling Model.

Tests:
    1.  Quaternion norm preserved after propagation
    2.  Rotation matrix is proper: C^T C = I, det(C) = +1
    3.  Rotational kinetic energy conserved (torque-free)
    4.  Angular momentum magnitude conserved (torque-free)
    5.  Identity quaternion gives zero euler angles
    6.  Quaternion kinematics: q̇ = 0 when ω = 0
    7.  qmultiply associativity
    8.  dcm_to_q round-trip: C -> q -> C
    9.  Euler 321 round-trip: angles -> q -> angles
    10. Axisymmetric body: spin-axis ω conserved (no wobble)
    11. Euler angle singularity-free for moderate pitch
    12. Quaternion conjugate reverses rotation
    13. Energy and momentum check pass via check_conservation()
    14. attitude_error_deg = 0 for identical quaternions
    15. ariane_upper_stage inertia ratios are physically reasonable

Run:
    cd /home/user/phase1
    python -m pytest tests/test_phase2.py -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from target import (
    qnormalize, qmultiply, qconjugate,
    q_to_dcm, dcm_to_q,
    q_to_euler321, euler321_to_q,
    qdot, attitude_error_deg,
    rotate_vector, rotate_vector_inv,
    RigidBodyAttitude, ariane_upper_stage, cubesat_3u,
)

# ── Shared fixtures ────────────────────────────────────────────────────────────
ARIANE = ariane_upper_stage()
T_SIM  = 600.0      # 10 minutes of tumble
N_PTS  = 3000

@pytest.fixture
def q_random():
    """A random unit quaternion."""
    rng = np.random.default_rng(42)
    q = rng.standard_normal(4)
    return qnormalize(q)

@pytest.fixture
def w_random():
    """A moderate random angular velocity [rad/s]."""
    rng = np.random.default_rng(7)
    return rng.uniform(-0.05, 0.05, size=3)   # ~3 deg/s

@pytest.fixture
def tumble_result(q_random, w_random):
    t_eval = np.linspace(0, T_SIM, N_PTS)
    return ARIANE.propagate(q_random, w_random, (0, T_SIM), t_eval=t_eval)

# ── Test 1: Quaternion norm preserved ─────────────────────────────────────────
def test_qnorm_preserved(tumble_result):
    norms = np.linalg.norm(tumble_result['q'], axis=0)
    np.testing.assert_allclose(norms, 1.0, atol=1e-10,
        err_msg="Quaternion unit norm must be preserved throughout propagation")

# ── Test 2: Rotation matrix is proper ─────────────────────────────────────────
def test_dcm_proper(q_random):
    C = q_to_dcm(q_random)
    np.testing.assert_allclose(C.T @ C, np.eye(3), atol=1e-14,
        err_msg="DCM must be orthogonal: C^T C = I")
    assert abs(np.linalg.det(C) - 1.0) < 1e-14, "DCM det must be +1 (proper rotation)"

# ── Test 3: Rotational KE conserved (torque-free) ─────────────────────────────
def test_energy_conservation(tumble_result):
    q, w = tumble_result['q'], tumble_result['w']
    T0 = ARIANE.rotational_ke(w[:, 0])
    T_arr = np.array([ARIANE.rotational_ke(w[:, k]) for k in range(w.shape[1])])
    rel_err = np.abs(T_arr - T0) / (abs(T0) + 1e-30)
    assert rel_err.max() < 1e-6, \
        f"Rotational KE must be conserved, max rel err = {rel_err.max():.2e}"

# ── Test 4: Angular momentum magnitude conserved ───────────────────────────────
def test_angular_momentum_conservation(tumble_result):
    q, w = tumble_result['q'], tumble_result['w']
    H0mag = np.linalg.norm(ARIANE.angular_momentum_inertial(q[:, 0], w[:, 0]))
    H_mags = np.array([
        np.linalg.norm(ARIANE.angular_momentum_inertial(q[:, k], w[:, k]))
        for k in range(q.shape[1])
    ])
    rel_err = np.abs(H_mags - H0mag) / (H0mag + 1e-30)
    assert rel_err.max() < 1e-6, \
        f"Angular momentum magnitude must be conserved, max rel err = {rel_err.max():.2e}"

# ── Test 5: Identity quaternion → zero Euler angles ───────────────────────────
def test_identity_quaternion_euler():
    q_id  = np.array([1.0, 0.0, 0.0, 0.0])
    euler = q_to_euler321(q_id)
    np.testing.assert_allclose(euler, 0.0, atol=1e-14,
        err_msg="Identity quaternion must give zero Euler angles")

# ── Test 6: q̇ = 0 when ω = 0 ─────────────────────────────────────────────────
def test_qdot_zero_at_zero_omega(q_random):
    dq = qdot(q_random, np.zeros(3))
    np.testing.assert_allclose(dq, 0.0, atol=1e-15,
        err_msg="Quaternion time derivative must be zero when angular velocity is zero")

# ── Test 7: qmultiply associativity ────────────────────────────────────────────
def test_qmultiply_associative():
    rng = np.random.default_rng(1)
    p = qnormalize(rng.standard_normal(4))
    q = qnormalize(rng.standard_normal(4))
    r = qnormalize(rng.standard_normal(4))
    pq_r = qmultiply(qmultiply(p, q), r)
    p_qr = qmultiply(p, qmultiply(q, r))
    np.testing.assert_allclose(pq_r, p_qr, atol=1e-14,
        err_msg="Quaternion multiplication must be associative")

# ── Test 8: dcm_to_q round-trip ───────────────────────────────────────────────
def test_dcm_q_roundtrip(q_random):
    C  = q_to_dcm(q_random)
    q2 = dcm_to_q(C)
    C2 = q_to_dcm(q2)
    np.testing.assert_allclose(C, C2, atol=1e-12,
        err_msg="DCM -> q -> DCM round-trip must recover original matrix")

# ── Test 9: Euler 321 round-trip ──────────────────────────────────────────────
def test_euler_roundtrip():
    for angles in [(0.3, 0.2, 0.1), (-0.5, 0.4, 1.2), (0.0, 0.0, 0.0)]:
        roll, pitch, yaw = angles
        q     = euler321_to_q(roll, pitch, yaw)
        back  = q_to_euler321(q)
        np.testing.assert_allclose(back, [roll, pitch, yaw], atol=1e-12,
            err_msg=f"Euler 321 round-trip failed for angles {angles}")

# ── Test 10: Axisymmetric spin: spin-axis ω₁ conserved ────────────────────────
def test_axisymmetric_spin_axis():
    """
    For an axisymmetric body (I2 = I3), pure spin about axis 1 is an
    equilibrium — ω1 must remain constant.
    """
    body   = ariane_upper_stage()   # I2 = I3
    q0     = np.array([1.0, 0.0, 0.0, 0.0])
    w0     = np.array([0.05, 0.0, 0.0])    # pure spin about symmetry axis
    t_eval = np.linspace(0, 3600, 5000)    # 1 hour
    res    = body.propagate(q0, w0, (0, 3600), t_eval=t_eval)
    w1_arr = res['w'][0, :]
    variation = np.ptp(w1_arr)
    assert variation < 1e-10, \
        f"Spin about symmetry axis must be constant, variation = {variation:.2e} rad/s"

# ── Test 11: Euler angles bounded for moderate tumble ─────────────────────────
def test_euler_angles_bounded(tumble_result):
    euler = tumble_result['euler']   # shape (3, N) — roll, pitch, yaw
    # Roll and yaw live in (-π, π), pitch in (-π/2, π/2)
    assert euler[0].min() >= -np.pi - 1e-10 and euler[0].max() <= np.pi + 1e-10, \
        "Roll must stay in (-π, π)"
    assert euler[1].min() >= -np.pi/2 - 1e-10 and euler[1].max() <= np.pi/2 + 1e-10, \
        "Pitch must stay in (-π/2, π/2)"

# ── Test 12: Quaternion conjugate reverses rotation ───────────────────────────
def test_conjugate_reverses_rotation(q_random):
    v     = np.array([1.0, 0.0, 0.0])
    v_rot = rotate_vector(q_random, v)
    v_back = rotate_vector_inv(q_random, v_rot)
    np.testing.assert_allclose(v_back, v, atol=1e-14,
        err_msg="Rotating then inverse-rotating must recover original vector")

# ── Test 13: check_conservation passes for DOP853 integration ─────────────────
def test_check_conservation_passes(tumble_result):
    report = ARIANE.check_conservation(tumble_result, rtol=1e-5)
    assert report['energy_ok'],   \
        f"Energy conservation check failed: max err = {report['max_E_err']:.2e}"
    assert report['momentum_ok'], \
        f"Momentum conservation check failed: max err = {report['max_H_err']:.2e}"

# ── Test 14: attitude_error_deg = 0 for identical quaternions ─────────────────
def test_attitude_error_identical(q_random):
    err = attitude_error_deg(q_random, q_random)
    assert err < 1e-10, f"Attitude error between identical quaternions must be 0, got {err:.2e}"

def test_attitude_error_opposite():
    # -q and q represent the same rotation
    q  = np.array([0.5, 0.5, 0.5, 0.5])
    err = attitude_error_deg(q, -q)
    assert err < 1e-10, "Antipodal quaternions represent same rotation — error must be 0"

# ── Test 15: Ariane inertia ratios are physically reasonable ──────────────────
def test_ariane_inertia_ratios():
    I = np.diag(ARIANE.I)
    # Transverse >> axial for a long cylinder
    assert I[1] > I[0] * 3, \
        f"Ariane transverse inertia should be >> axial, got ratio {I[1]/I[0]:.2f}"
    # Symmetry: I2 == I3
    np.testing.assert_allclose(I[1], I[2], rtol=1e-10,
        err_msg="Ariane body must be axisymmetric: I2 = I3")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

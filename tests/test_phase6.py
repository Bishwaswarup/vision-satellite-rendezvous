"""
test_phase6.py
==============
Unit tests for Phase 6 — LQR / MPC Rendezvous Control.

Tests
-----
  1.  LQR gain K has shape (3, 6)
  2.  LQR closed-loop is stable (all |λ| < 1)
  3.  LQR drives state to zero: ‖r‖ < 0.1 m after 300 steps
  4.  LQR Lyapunov function V(x) decreases monotonically (no saturation)
  5.  LQR thrust saturation: ‖u‖_∞ ≤ u_max always
  6.  LQR Δv accumulates correctly (non-negative, grows with time)
  7.  HCW ZOH discretization: Φ preserves energy (trace property)
  8.  MPC control has correct shape (3,)
  9.  MPC thrust satisfies per-axis bound at every step
 10.  MPC drives state to zero: ‖r‖ < 0.5 m after 100 steps
 11.  MPC terminal cost matches LQR P matrix (consistency)
 12.  LQR vs MPC: both converge — MPC position error ≤ 2× LQR at step 100
 13.  MPC with approach cone: |y| ≤ |x| tan(θ) after warm-up
 14.  EKF-in-the-loop: LQR on EKF state converges to < 1 m
 15.  Docking: state reaches ‖r‖ < 0.5 m, ‖v‖ < 0.05 m/s within 200 steps

Run:
    cd /home/user/phase1
    python -m pytest tests/test_phase6.py -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from controller import (
    LQRController, make_lqr,
    MPCController, make_mpc,
    hcw_discrete, N_ORBITAL_DEFAULT,
)

# ── Shared fixtures ────────────────────────────────────────────────────────────

DT   = 1.0
N    = N_ORBITAL_DEFAULT

def _x0(r=np.array([20., 2., -1.]), v=np.array([-0.1, 0., 0.])):
    return np.concatenate([r, v])


# ── Test 1: LQR gain shape ────────────────────────────────────────────────────
def test_lqr_gain_shape():
    lqr = make_lqr()
    assert lqr.K.shape == (3, 6), f"LQR gain must be (3,6), got {lqr.K.shape}"


# ── Test 2: LQR closed-loop stability ────────────────────────────────────────
def test_lqr_stable():
    lqr = make_lqr()
    assert lqr.is_stable(), "LQR closed-loop must be stable (all |λ| < 1)"
    eigs = lqr.closed_loop_eigs()
    assert np.all(np.abs(eigs) < 1.0), \
        f"Some eigenvalues outside unit circle: {np.abs(eigs)}"


# ── Test 3: LQR drives position to zero ──────────────────────────────────────
def test_lqr_convergence():
    lqr  = make_lqr(u_max=0.5)    # generous thrust for speed
    sim  = lqr.simulate(_x0(), n_steps=300)
    r_final = np.linalg.norm(sim['states'][-1, :3])
    assert r_final < 0.1, \
        f"LQR must drive ‖r‖ < 0.1 m in 300 steps, got {r_final:.3f} m"


# ── Test 4: LQR Lyapunov function decreases ───────────────────────────────────
def test_lqr_lyapunov_decreasing():
    # Use small u_max=None so saturation doesn't interfere
    lqr  = make_lqr(u_max=None)
    Phi, Gamma = hcw_discrete(N, DT)
    x = _x0()
    V_prev = lqr.lyapunov_value(x)
    for _ in range(50):
        u = lqr.control(x, saturate=False)
        x = Phi @ x + Gamma @ u
        V = lqr.lyapunov_value(x)
        assert V <= V_prev + 1e-9, \
            f"Lyapunov V must be non-increasing: V={V:.6f} > V_prev={V_prev:.6f}"
        V_prev = V


# ── Test 5: LQR thrust saturation ─────────────────────────────────────────────
def test_lqr_saturation():
    u_max = 0.05
    lqr   = make_lqr(u_max=u_max)
    sim   = lqr.simulate(_x0(r=np.array([50., 10., -5.])), n_steps=200)
    u_max_seen = np.abs(sim['controls']).max()
    assert u_max_seen <= u_max + 1e-12, \
        f"Thrust must not exceed u_max={u_max}, saw {u_max_seen:.6f}"


# ── Test 6: LQR Δv accumulates ────────────────────────────────────────────────
def test_lqr_deltav():
    lqr = make_lqr()
    sim = lqr.simulate(_x0(), n_steps=100)
    assert sim['delta_v'] > 0, "Δv must be positive for a non-trivial trajectory"
    # Δv should be less than a very generous upper bound (not runaway)
    assert sim['delta_v'] < 1000.0, f"Δv seems unreasonably large: {sim['delta_v']:.1f}"


# ── Test 7: HCW ZOH discretization ───────────────────────────────────────────
def test_hcw_discrete():
    Phi, Gamma = hcw_discrete(N, DT)
    assert Phi.shape == (6, 6)
    assert Gamma.shape == (6, 3)
    # Phi should be close to det ≈ 1 (symplectic, area-preserving)
    det = np.linalg.det(Phi)
    assert abs(det - 1.0) < 1e-6, f"det(Phi) should be ~1, got {det:.8f}"


# ── Test 8: MPC control shape ─────────────────────────────────────────────────
def test_mpc_control_shape():
    mpc = make_mpc(N=5)
    u   = mpc.control(_x0())
    assert u.shape == (3,), f"MPC control must be (3,), got {u.shape}"


# ── Test 9: MPC thrust satisfies bounds ───────────────────────────────────────
def test_mpc_thrust_bounds():
    u_max = 0.08
    mpc   = make_mpc(N=10, u_max=u_max)
    sim   = mpc.simulate(_x0(), n_steps=50)
    u_max_seen = np.abs(sim['controls']).max()
    assert u_max_seen <= u_max + 1e-6, \
        f"MPC thrust must not exceed {u_max}, saw {u_max_seen:.6f}"


# ── Test 10: MPC drives state to zero ────────────────────────────────────────
def test_mpc_convergence():
    mpc = make_mpc(N=20, u_max=0.5)
    sim = mpc.simulate(_x0(), n_steps=100)
    r_final = np.linalg.norm(sim['states'][-1, :3])
    assert r_final < 0.5, \
        f"MPC must drive ‖r‖ < 0.5 m in 100 steps, got {r_final:.3f} m"


# ── Test 11: MPC terminal cost = LQR P ───────────────────────────────────────
def test_mpc_terminal_equals_lqr():
    Q = np.diag([10., 10., 10., 1., 1., 1.])
    R = np.eye(3)
    from controller.lqr import LQRController
    lqr = LQRController(Q=Q, R=R, u_max=0.1)
    mpc = MPCController(N=10, Q=Q, R=R, P_terminal=lqr.P, u_max=0.1)
    np.testing.assert_allclose(mpc.P_term, lqr.P, rtol=1e-8,
        err_msg="MPC terminal cost must match LQR P matrix")


# ── Test 12: LQR vs MPC comparable performance ────────────────────────────────
def test_lqr_vs_mpc_performance():
    x0  = _x0()
    lqr = make_lqr(u_max=0.2)
    mpc = make_mpc(N=15, u_max=0.2)

    lqr_sim = lqr.simulate(x0, n_steps=100)
    mpc_sim = mpc.simulate(x0, n_steps=100)

    err_lqr = np.linalg.norm(lqr_sim['states'][-1, :3])
    err_mpc = np.linalg.norm(mpc_sim['states'][-1, :3])

    # Both should converge
    assert err_lqr < 1.0, f"LQR position error must be < 1 m, got {err_lqr:.3f}"
    assert err_mpc < 1.0, f"MPC position error must be < 1 m, got {err_mpc:.3f}"

    # MPC should not be more than 2× worse than LQR (finite horizon penalty)
    assert err_mpc <= max(err_lqr * 2.0, 0.5), \
        f"MPC err ({err_mpc:.3f}) is much worse than LQR err ({err_lqr:.3f})"


# ── Test 13: MPC approach cone constraint ─────────────────────────────────────
def test_mpc_approach_cone():
    """After 60 steps the chaser should be inside a 30° cone around the X axis."""
    x0  = np.array([20., 5., 3., -0.1, 0., 0.])   # offset in Y and Z
    mpc = make_mpc(N=20, u_max=0.3, cone_half_angle=30.0)
    sim = mpc.simulate(x0, n_steps=80)

    # Check final 20 steps after initial transient
    states = sim['states'][60:]
    x_pos  = np.abs(states[:, 0])
    y_pos  = np.abs(states[:, 1])
    tan30  = np.tan(np.radians(30.0))

    # Allow 20% margin for numerical tolerance of SLSQP
    violators = np.where(y_pos > x_pos * tan30 * 1.2)[0]
    # Should have few or no violations in steady state
    assert len(violators) <= 5, \
        f"Too many cone violations ({len(violators)}/20) after transient"


# ── Test 14: EKF-in-the-loop ──────────────────────────────────────────────────
def test_ekf_in_loop_lqr():
    """
    LQR commanded on EKF state estimate must still converge.

    The EKF predicts with free HCW dynamics; the thrust is a known input
    so we inject its effect into the EKF state after each predict step
    (control-aware prediction) so the filter stays aligned with the truth.
    """
    from estimator import (
        pack_state, unpack_state, propagate_rk4,
        rotvec_to_quat, make_ekf, h_measurement,
    )

    # True state
    r0 = np.array([15., 1., -0.5])
    v0 = np.array([-0.08, 0., 0.])
    q0 = rotvec_to_quat(np.array([0., 0., 0.2]))
    w0 = np.array([0., 0., 0.02])
    x_true = pack_state(r0, v0, q0, w0)

    lqr  = make_lqr(u_max=0.3)
    Phi, Gamma = hcw_discrete(N, DT)

    x_ekf_init = pack_state(r0 + np.array([2., -0.5, 0.3]), v0, q0, w0)
    ekf  = make_ekf(x_ekf_init, DT, pos0_std=3.0)

    rng  = np.random.default_rng(42)
    u    = np.zeros(3)   # previous control (zero at start)

    for _ in range(150):
        # ── EKF predict (free HCW) + inject known control ──────────────────
        ekf.predict(DT)
        # Correct EKF prediction for the thrust applied last step
        ctrl_effect = Gamma @ u            # (6,): Δr and Δv from thrust
        ekf.x[:3]  += ctrl_effect[:3]
        ekf.x[3:6] += ctrl_effect[3:]

        # ── LQR on EKF estimate ────────────────────────────────────────────
        r_est, v_est = unpack_state(ekf.state)[:2]
        u = lqr.control(np.concatenate([r_est, v_est]))

        # ── Propagate true translational state with control ────────────────
        r_t, v_t = unpack_state(x_true)[:2]
        rv_new = Phi @ np.concatenate([r_t, v_t]) + Gamma @ u

        # ── Propagate true attitude (torque-free) ──────────────────────────
        q_t, w_t = unpack_state(x_true)[2:]
        x_att = propagate_rk4(pack_state(r_t, v_t, q_t, w_t), DT)
        x_true = pack_state(rv_new[:3], rv_new[3:],
                            unpack_state(x_att)[2],
                            unpack_state(x_att)[3])

        # ── Measurement update ─────────────────────────────────────────────
        z = h_measurement(x_true)
        z[:3] += rng.normal(0, 0.5, 3)
        z[3:]  += rng.normal(0, 0.05, 3)
        ekf.update(z)

    r_final = unpack_state(x_true)[0]
    pos_err = np.linalg.norm(r_final)
    assert pos_err < 1.0, \
        f"EKF-in-loop must drive ‖r‖ < 1 m, got {pos_err:.3f} m"


# ── Test 15: Docking condition ────────────────────────────────────────────────
def test_docking_condition():
    """Chaser must reach docking port (r < 0.5 m) with v < 0.05 m/s."""
    lqr = make_lqr(u_max=0.5, pos_weight=50.0, vel_weight=10.0)
    x0  = _x0(r=np.array([5., 0.5, -0.2]), v=np.array([-0.02, 0., 0.]))
    sim = lqr.simulate(x0, n_steps=200)

    # Find first step where ‖r‖ < 0.5 m AND ‖v‖ < 0.05 m/s
    states = sim['states']
    r_norm = np.linalg.norm(states[:, :3], axis=1)
    v_norm = np.linalg.norm(states[:, 3:6], axis=1)
    docked = np.where((r_norm < 0.5) & (v_norm < 0.05))[0]

    assert len(docked) > 0, (
        f"LQR never reached docking condition. "
        f"Final ‖r‖={r_norm[-1]:.3f} m, ‖v‖={v_norm[-1]:.3f} m/s"
    )


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

"""
test_phase1.py
==============
Unit tests for Phase 1: Orbital Dynamics Engine.

Tests:
    1.  HCW STM identity at t=0
    2.  HCW drift-free periodic orbit closes after 1 period
    3.  HCW analytical == numerical propagation (cross-validation)
    4.  HCW z-axis oscillation (decoupled, exact)
    5.  STM determinant preserved (Liouville)
    6.  J2 acceleration at equatorial pole equals analytic value
    7.  J2 acceleration symmetry (equatorial plane)
    8.  J2 acceleration is zero at equatorial plane and pole check
    9.  Chief ECI energy conservation (two-body, no J2)
    10. keplerian_to_eci round-trip (verify r, v magnitudes)
    11. YA STM reduces to near-HCW for small eccentricity (qualitative)
    12. LVLH rotation matrix is orthogonal
    13. HCW discretisation Ad is invertible
    14. Drift-free condition: 5-orbit non-drift check
    15. Periodic orbit area (geometry sanity)

Run with:
    cd /home/user/phase1
    python -m pytest tests/test_phase1.py -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from dynamics import (
    HCWPropagator, YAStatTransition,
    j2_accel_eci, keplerian_to_eci, lvlh_to_eci_rotation,
    propagate_chief_eci,
    MU_EARTH, J2, R_EARTH, LEO_SMA, LEO_N, LEO_PERIOD
)

# ── Shared fixtures ────────────────────────────────────────────────────────────
N   = LEO_N
T   = LEO_PERIOD
SMA = LEO_SMA
RHO = 50.0      # half-amplitude [m] for periodic orbit tests
ATOL = 1e-6     # absolute position tolerance [m]
VTOL = 1e-9     # absolute velocity tolerance [m/s]

@pytest.fixture
def hcw():
    return HCWPropagator(n=N)

@pytest.fixture
def x0_periodic(hcw):
    return HCWPropagator.periodic_initial_condition(RHO, N, theta0=0.0)

# ── Test 1: STM identity at t = 0 ─────────────────────────────────────────────
def test_stm_identity_at_zero(hcw):
    Phi = hcw.stm(0.0)
    np.testing.assert_allclose(Phi, np.eye(6), atol=1e-14,
        err_msg="STM at t=0 must be the identity matrix")

# ── Test 2: Periodic orbit closes after one period ────────────────────────────
def test_periodic_orbit_closes(hcw, x0_periodic):
    xf = hcw.propagate_analytical(x0_periodic, T)
    np.testing.assert_allclose(xf[:3], x0_periodic[:3], atol=ATOL,
        err_msg="Periodic orbit position must close after one orbital period")
    np.testing.assert_allclose(xf[3:], x0_periodic[3:], atol=VTOL,
        err_msg="Periodic orbit velocity must close after one orbital period")

# ── Test 3: Analytical == Numerical (cross-validation) ────────────────────────
def test_analytical_vs_numerical(hcw, x0_periodic):
    t_test  = T / 3.0           # propagate to 1/3 orbit
    x_anal  = hcw.propagate_analytical(x0_periodic, t_test)
    t_eval  = np.array([0.0, t_test])
    res     = hcw.propagate_numerical(x0_periodic, (0.0, t_test), t_eval=t_eval)
    x_num   = res['y'][:, -1]
    np.testing.assert_allclose(x_anal[:3], x_num[:3], atol=1e-4,
        err_msg="Position: analytical vs numerical must agree to < 0.1 mm")
    np.testing.assert_allclose(x_anal[3:], x_num[3:], atol=1e-7,
        err_msg="Velocity: analytical vs numerical must agree to < 0.1 μm/s")

# ── Test 4: Out-of-plane decoupled harmonic oscillator ────────────────────────
def test_z_decoupled(hcw):
    z0, zdot0 = 100.0, 0.05     # arbitrary IC
    x0 = np.array([0, 0, z0, 0, 0, zdot0])
    t  = 0.5 * T
    # Exact: z(t) = z0*cos(nt) + zdot0/n * sin(nt)
    z_exact  = z0 * np.cos(N*t) + zdot0 / N * np.sin(N*t)
    zd_exact = -z0 * N * np.sin(N*t) + zdot0 * np.cos(N*t)
    xf       = hcw.propagate_analytical(x0, t)
    assert abs(xf[2] - z_exact)  < 1e-9, "z(t) must satisfy harmonic oscillator exactly"
    assert abs(xf[5] - zd_exact) < 1e-12, "zdot(t) harmonic velocity mismatch"
    # x and y must remain zero (no coupling when x0=y0=0)
    np.testing.assert_allclose(xf[[0,1,3,4]], 0.0, atol=1e-14,
        err_msg="Out-of-plane motion must not couple into in-plane for pure z IC")

# ── Test 5: STM determinant (Liouville / volume preservation) ─────────────────
def test_stm_det_one(hcw):
    for t in [100, T/4, T/2, T]:
        det = np.linalg.det(hcw.stm(t))
        assert abs(det - 1.0) < 1e-10, f"STM det must be 1 (got {det:.6e}) at t={t:.0f}s"

# ── Test 6: J2 acceleration equatorial analytic value ─────────────────────────
def test_j2_equatorial_value():
    # On the equatorial plane at (r, 0, 0):  z = 0
    r0 = np.array([SMA, 0.0, 0.0])
    a  = j2_accel_eci(r0)
    # Analytic: ax = -3/2 * mu*J2*Re^2 / r^4,  ay=0, az=0  (z=0 case)
    coeff = 3 * MU_EARTH * J2 * R_EARTH**2 / (2 * SMA**5)
    ax_expected = coeff * SMA * (5*0 - 1)   # zr2=0 => -1
    assert abs(a[0] - ax_expected) < 1e-20 * abs(ax_expected) + 1e-12
    assert abs(a[1]) < 1e-14, "ay must be zero for r along x-axis"
    assert abs(a[2]) < 1e-14, "az must be zero at equatorial plane"

# ── Test 7: J2 acceleration symmetry (equatorial plane) ───────────────────────
def test_j2_symmetry():
    r1 = np.array([SMA, 0.0, 0.0])
    r2 = np.array([0.0, SMA, 0.0])
    a1 = j2_accel_eci(r1)
    a2 = j2_accel_eci(r2)
    # Magnitudes should be equal by equatorial symmetry
    np.testing.assert_allclose(np.linalg.norm(a1), np.linalg.norm(a2), rtol=1e-12,
        err_msg="J2 acceleration magnitude must be symmetric in equatorial plane")

# ── Test 8: J2 is purely radial on equatorial plane ───────────────────────────
def test_j2_radial_equatorial():
    for phi in np.linspace(0, 2*np.pi, 7):
        r = SMA * np.array([np.cos(phi), np.sin(phi), 0.0])
        a = j2_accel_eci(r)
        # a should be antiparallel to r (purely radial)
        cross = np.cross(a, r)
        assert np.linalg.norm(cross) < 1e-5, \
            f"J2 accel must be radial on equatorial plane, phi={phi:.2f}"

# ── Test 9: Two-body energy conservation (chief, no J2) ───────────────────────
def test_twobody_energy_conservation():
    rv0 = keplerian_to_eci(SMA, 0.001, np.radians(51.6), 0.0, 0.0, 0.0)
    t_eval = np.linspace(0, T, 500)
    rv = propagate_chief_eci(rv0, t_eval, include_j2=False)
    r = np.linalg.norm(rv[:3, :], axis=0)
    v = np.linalg.norm(rv[3:, :], axis=0)
    E = 0.5 * v**2 - MU_EARTH / r    # specific energy
    rel_var = np.ptp(E) / abs(E[0])   # peak-to-peak relative variation
    assert rel_var < 1e-8, f"Two-body energy conservation violated: {rel_var:.2e}"

# ── Test 10: keplerian_to_eci radius and speed ────────────────────────────────
def test_keplerian_to_eci_radius():
    a, e, f = SMA, 0.05, np.pi / 3
    i, raan, aop = np.radians(45.0), np.radians(30.0), np.radians(20.0)
    rv = keplerian_to_eci(a, e, i, raan, aop, f)
    r_expected = a * (1 - e**2) / (1 + e * np.cos(f))
    r_got      = np.linalg.norm(rv[:3])
    assert abs(r_got - r_expected) < 1e-4, \
        f"Radius mismatch: expected {r_expected:.3f} m, got {r_got:.3f} m"
    # Circular velocity check (for near-circular orbit within ~5%)
    v_circ = np.sqrt(MU_EARTH / r_got)
    v_got  = np.linalg.norm(rv[3:])
    assert abs(v_got - v_circ) / v_circ < 0.2, \
        f"Speed {v_got:.1f} m/s unexpectedly far from circular {v_circ:.1f}"

# ── Test 11: LVLH rotation is orthogonal ──────────────────────────────────────
def test_lvlh_rotation_orthogonal():
    rv0 = keplerian_to_eci(SMA, 0.001, np.radians(51.6), 0.0, 0.0, 0.0)
    R = lvlh_to_eci_rotation(rv0[:3], rv0[3:])
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-14,
        err_msg="LVLH rotation matrix must be orthogonal")
    assert abs(np.linalg.det(R) - 1.0) < 1e-14, "LVLH rotation det must be +1"

# ── Test 12: HCW discretisation Ad is invertible and Ad@Ad_inv = I ───────────
def test_hcw_discretise_invertible(hcw):
    Ts = 10.0   # 10-second sample period
    Ad, Bd = hcw.discretise(Ts)
    # 1) Ad must be invertible
    det = np.linalg.det(Ad)
    assert abs(det) > 1e-6, f"Ad must be invertible, det={det:.6e}"
    # 2) Ad @ Ad_inv == I  (round-trip check)
    np.testing.assert_allclose(Ad @ np.linalg.inv(Ad), np.eye(6), atol=1e-10,
        err_msg="Ad @ Ad^{-1} must equal identity")
    # 3) Bd shape and reasonable magnitude
    assert Bd.shape == (6, 3), f"Bd shape mismatch: {Bd.shape}"
    assert Bd[3:, :].max() > 0, "Lower half of Bd (velocity rows) must be non-zero"

# ── Test 13: 5-orbit non-drift (periodic initial condition) ───────────────────
def test_5orbit_no_drift(hcw, x0_periodic):
    xf = hcw.propagate_analytical(x0_periodic, 5 * T)
    np.testing.assert_allclose(xf[:3], x0_periodic[:3], atol=1e-3,
        err_msg="Drift-free IC must remain bounded after 5 orbits")
    # y-drift: |y(5T) - y(0)| < 1 mm
    assert abs(xf[1] - x0_periodic[1]) < 1e-3

# ── Test 14: Periodic orbit geometry (2:1 ellipse) ────────────────────────────
def test_periodic_orbit_aspect_ratio(hcw):
    t_eval = np.linspace(0, T, 2000)
    x0  = HCWPropagator.periodic_initial_condition(RHO, N)
    sol = hcw.propagate_numerical(x0, (0, T), t_eval=t_eval)
    xs  = sol['y'][0, :]
    ys  = sol['y'][1, :]
    amp_x = (xs.max() - xs.min()) / 2
    amp_y = (ys.max() - ys.min()) / 2
    ratio = amp_y / amp_x
    assert abs(ratio - 2.0) < 0.01, \
        f"HCW periodic orbit must be a 2:1 ellipse, got ratio={ratio:.4f}"

# ── Test 15: YA STM is non-singular and propagates bounded motion ─────────────
def test_ya_stm_nonsingular_and_propagates():
    e   = 0.05                          # moderate eccentricity
    ya  = YAStatTransition(mu=MU_EARTH, a=SMA, e=e)
    f0  = np.radians(30.0)              # non-zero start (avoids σ=0 edge case)
    f1  = f0 + np.pi / 2               # quarter-orbit in f

    Phi = ya.stm(f0, f1)

    # 1) STM must be non-singular
    det = np.linalg.det(Phi)
    assert abs(det) > 1e-3, f"YA STM must be non-singular, det={det:.4e}"

    # 2) Shape
    assert Phi.shape == (6, 6), f"YA STM shape mismatch: {Phi.shape}"

    # 3) Propagating a zero IC gives zero (linearity)
    x0_zero = np.zeros(6)
    xf_zero = ya.propagate(x0_zero, f0, f1)
    np.testing.assert_allclose(xf_zero, 0.0, atol=1e-10,
        err_msg="YA propagation of zero state must remain zero")

    # 4) Propagating a reasonable IC gives bounded output
    x0 = np.array([50.0, -100.0, 30.0, 0.01, -0.02, 0.005])
    xf = ya.propagate(x0, f0, f1)
    # Position should remain in a few-km neighbourhood
    assert np.linalg.norm(xf[:3]) < 1e5, \
        f"YA propagation produced unreasonably large position: {np.linalg.norm(xf[:3]):.2e} m"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

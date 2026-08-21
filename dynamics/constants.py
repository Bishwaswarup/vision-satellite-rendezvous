"""
constants.py
============
Physical constants and Earth parameters for the orbital dynamics engine.
All values in SI units unless noted.

Reference:
    WGS-84 / EGM96 gravitational model
    Vallado, D. A., "Fundamentals of Astrodynamics and Applications", 4th ed.
"""

# ─── Gravitational Parameters ────────────────────────────────────────────────
MU_EARTH    = 3.986004418e14     # [m^3/s^2]  Earth gravitational parameter
MU_SUN      = 1.327124400e20     # [m^3/s^2]  Sun gravitational parameter (for later phases)

# ─── Earth Physical Properties ───────────────────────────────────────────────
R_EARTH     = 6.3781366e6        # [m]         Earth mean equatorial radius
J2          = 1.08262668e-3      # [-]         Earth oblateness coefficient (EGM96)
J3          = -2.53265648e-6     # [-]         3rd zonal harmonic (for high-fidelity, Phase 7)
OMEGA_EARTH = 7.2921150e-5       # [rad/s]     Earth sidereal rotation rate

# ─── Standard Atmosphere / Misc ──────────────────────────────────────────────
G0          = 9.80665            # [m/s^2]     Standard gravity (for Isp calcs)
AU          = 1.495978707e11     # [m]         Astronomical unit

# ─── Conversion Factors ──────────────────────────────────────────────────────
DEG2RAD     = 3.141592653589793 / 180.0
RAD2DEG     = 180.0 / 3.141592653589793

# ─── Typical LEO Simulation Defaults ─────────────────────────────────────────
LEO_ALTITUDE_KM  = 500.0         # [km]  Default chief orbit altitude
LEO_SMA          = R_EARTH + LEO_ALTITUDE_KM * 1e3   # [m]  Semi-major axis
LEO_N            = (MU_EARTH / LEO_SMA**3) ** 0.5    # [rad/s]  Mean motion
LEO_PERIOD       = 2 * 3.141592653589793 / LEO_N      # [s]  Orbital period

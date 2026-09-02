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
J_ARIANE = np.diag([1176.0, 6988.0, 6988.0])
# Ariane 44L upper stage, matching target.attitude.ariane_upper_stage()
# and the geometry in vision.body_model (x is the symmetry axis):
#   m = 1200 kg, r = 1.4 m, L = 8.0 m
#   I_xx = m r^2 / 2 = 1176,  I_yy = I_zz = m(3r^2 + L^2)/12 = 6988
# This used to be diag(1800, 1800, 360) — a different body with the
# symmetry axis on z instead of x.  Since propagate_rk4 uses THIS
# tensor, that wrong one drove the target attitude in every reported
# result while target/attitude.py was only ever reached by the tests.   # kg·m²
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


def quat_inv(q):
    """Inverse of a unit quaternion [x,y,z,w]."""
    q = np.asarray(q, dtype=float)
    return np.array([-q[0], -q[1], -q[2], q[3]])


def attitude_residual(rv_meas, q_est):
    """
    Multiplicative attitude residual, in the tangent space at `q_est`.

        delta_alpha = rotvec( q_meas  (x)  q_est^-1 )

    `rv_meas` is the measured attitude encoded as a rotation vector (which is
    what the measurement vector carries); it is converted back to a quaternion
    and the residual is taken on the group, not by subtracting rotation
    vectors.

    Subtracting global rotation vectors is wrong for two reasons.  The map
    q -> rotvec(q) is singular at theta = pi: two attitudes a hair either side
    of pi map to nearly ANTIPODAL rotation vectors, so the difference is
    O(2*pi) even though the attitudes are almost identical.  And away from the
    singularity the difference is still only an approximation of the true
    error, so R stops being a valid tangent-space covariance.  The
    multiplicative residual is exact everywhere and makes H_att exactly I3.
    """
    q_meas = rotvec_to_quat(np.asarray(rv_meas, dtype=float))
    q_est  = quat_norm(np.asarray(q_est, dtype=float))
    dq     = quat_mult(q_meas, quat_inv(q_est))
    if dq[3] < 0:                 # keep the short rotation
        dq = -dq
    return quat_to_rotvec(dq)


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
    rv = quat_to_rotvec(q)
    return np.concatenate([r, rv])


def measurement_jacobian(x):
    """
    Measurement Jacobian H (6×12) mapping the error state to the residual.

    With the multiplicative attitude residual of `attitude_residual`, H is
    exact and constant:

        innovation = [ r_meas - r_hat ,  delta_alpha ]
                   = [ I3  0  0  0 ;  0  0  I3  0 ] @ error_state

    because delta_alpha IS the attitude error state by construction.  No
    finite differencing is needed — and differencing the *global* rotation
    vector, as this used to, produces a Jacobian whose condition number blows
    up to ~2e5 near theta = pi.
    """
    H = np.zeros((6, 12))
    H[0:3, 0:3] = np.eye(3)      # position measures position
    H[3:6, 6:9] = np.eye(3)      # attitude residual measures attitude error
    return H


def _measurement_jacobian_fd(x):
    """Finite-difference Jacobian of the raw h(x); kept for cross-checks."""
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
                           att_std=1e-4, rate_std=1e-5,
                           accel_std=None, ang_accel_std=None):
    """
    Process noise Q (12×12).

    Two models are available.

    ``accel_std`` given — the physically correct one.  The disturbance is an
    unmodelled ACCELERATION, which drives position and velocity together:

        [Q_rr  Q_rv]   =  sigma_a^2  [ dt^3/3   dt^2/2 ]
        [Q_vr  Q_vv]                 [ dt^2/2   dt     ]

    with the same structure on attitude/rate for `ang_accel_std`.  Note the
    off-diagonal terms: position and velocity errors driven by a common
    acceleration are correlated, and a diagonal Q asserts they are not.

    ``accel_std`` omitted — the legacy diagonal random walk, kept so existing
    callers behave as before.  It is not physical: an independent position
    random walk of `pos_std` = 0.05 m contributes Q_rr = 2.5e-3 m^2 per step,
    roughly 300x the 8.3e-6 m^2 an actual 5e-4 m/s^2 disturbance produces, so
    the filter is made needlessly conservative and velocity is only weakly
    informed by position measurements.
    """
    if accel_std is None:
        diag = np.concatenate([
            np.full(3, pos_std**2  * dt),
            np.full(3, vel_std**2  * dt),
            np.full(3, att_std**2  * dt),
            np.full(3, rate_std**2 * dt),
        ])
        return np.diag(diag)

    Q = np.zeros((12, 12))
    I3 = np.eye(3)

    sa2 = float(accel_std) ** 2
    Q[0:3, 0:3] = sa2 * dt**3 / 3.0 * I3
    Q[0:3, 3:6] = sa2 * dt**2 / 2.0 * I3
    Q[3:6, 0:3] = sa2 * dt**2 / 2.0 * I3
    Q[3:6, 3:6] = sa2 * dt * I3

    sw2 = float(ang_accel_std if ang_accel_std is not None else rate_std) ** 2
    Q[6:9,  6:9 ] = sw2 * dt**3 / 3.0 * I3
    Q[6:9,  9:12] = sw2 * dt**2 / 2.0 * I3
    Q[9:12, 6:9 ] = sw2 * dt**2 / 2.0 * I3
    Q[9:12, 9:12] = sw2 * dt * I3
    return Q


def default_measurement_noise(pos_std=0.5, att_std=0.05):
    """
    Diagonal measurement noise R (6×6).
    pos_std [m] — position measurement uncertainty
    att_std [rad] — attitude measurement uncertainty (tangent space)
    """
    diag = np.concatenate([
        np.full(3, pos_std**2),
        np.full(3, att_std**2),
    ])
    return np.diag(diag)


def range_scaled_measurement_noise(range_m,
                                   pos_std_ref=0.25, att_std_ref=0.045,
                                   range_ref=20.0,
                                   range_min=1.0):
    """
    Measurement noise for a monocular pose fix, scaled with range.

    A pinhole pose solution degrades quadratically with depth: a fixed pixel
    error subtends a physical error proportional to z, and the depth component
    of the solution degrades faster still.  Measured on this pipeline at
    1.5 px keypoint noise, position RMSE runs

        5 m -> 0.007 m,  20 m -> 0.096 m,  30 m -> 0.249 m,  80 m -> 1.96 m

    which is very close to (z / z_ref)^2.  A single fixed R is therefore 70x
    too large at 5 m and 4x too small at 80 m: the filter throws away good
    close-range fixes and over-trusts poor distant ones.

    Parameters
    ----------
    range_m     : current target range [m]
    pos_std_ref : position 1-sigma at `range_ref` [m]
    att_std_ref : attitude 1-sigma at `range_ref` [rad]
    range_ref   : reference range [m]
    range_min   : floor on the range used for scaling, so R cannot collapse
                  to zero at contact

    Returns
    -------
    R : (6,6) diagonal measurement covariance
    """
    z = max(float(range_m), float(range_min))
    s = (z / float(range_ref)) ** 2
    pos_std = pos_std_ref * s
    att_std = att_std_ref * s
    return np.diag(np.concatenate([
        np.full(3, pos_std**2),
        np.full(3, att_std**2),
    ]))


# ── Filter consistency ─────────────────────────────────────────────────────────

def nis_statistics(nis, dof=6, alpha=0.05):
    """
    Summarise a NIS (normalised innovation squared) sequence.

    For a consistent filter the NIS of each update is chi-square distributed
    with `dof` degrees of freedom, so its mean should be `dof`.  The two-sided
    confidence interval on the SAMPLE MEAN of N draws is

        dof  +/-  z * sqrt(2 * dof / N)

    since Var[chi2_k] = 2k.  A mean above the interval means the filter is
    over-confident (its covariance is too small for the errors it actually
    makes); below means it is conservative.

    Parameters
    ----------
    nis   : sequence of per-update NIS values
    dof   : measurement dimension (6 here: 3 position + 3 attitude)
    alpha : significance level for the interval (default 0.05 -> 95 %)

    Returns
    -------
    dict with n, mean, dof, ci_low, ci_high, consistent, tail_fraction
    """
    nis = np.asarray([v for v in np.asarray(nis, dtype=float).ravel()
                      if np.isfinite(v)], dtype=float)
    n = int(nis.size)
    if n == 0:
        return {'n': 0, 'mean': float('nan'), 'dof': dof,
                'ci_low': float('nan'), 'ci_high': float('nan'),
                'consistent': False, 'tail_fraction': float('nan')}

    # Normal approximation to the mean of n chi-square draws.
    z = 1.959963984540054 if abs(alpha - 0.05) < 1e-12 else _norm_ppf(1 - alpha / 2)
    half = z * np.sqrt(2.0 * dof / n)
    mean = float(nis.mean())

    try:
        from scipy.stats import chi2
        thresh = float(chi2.ppf(0.99, dof))
    except Exception:                                    # pragma: no cover
        thresh = {6: 16.811893829770927}.get(dof, float('inf'))

    return {
        'n'            : n,
        'mean'         : mean,
        'dof'          : dof,
        'ci_low'       : dof - half,
        'ci_high'      : dof + half,
        'consistent'   : bool(dof - half <= mean <= dof + half),
        'tail_fraction': float((nis > thresh).mean()),
    }


def _norm_ppf(p):
    """Standard normal quantile (Acklam's rational approximation)."""
    from math import sqrt, log
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = sqrt(-2 * log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > ph:
        q = sqrt(-2 * log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)

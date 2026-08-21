"""
estimator package — Phase 5: EKF / UKF Pose Tracking
=====================================================
Modules:
    state   : state vector, dynamics, measurement models
    ekf     : Multiplicative Extended Kalman Filter (MEKF)
    ukf     : Unscented Kalman Filter (UKF)
"""
from .state import (
    N_ORBITAL, J_ARIANE,
    pack_state, unpack_state,
    quat_to_dcm, dcm_to_quat, quat_mult, quat_norm,
    rotvec_to_quat, quat_to_rotvec,
    f_continuous, propagate_rk4, process_jacobian,
    h_measurement, measurement_jacobian,
    default_process_noise, default_measurement_noise,
)
from .ekf import MultEKF, make_ekf
from .ukf import UnscentedKF, make_ukf

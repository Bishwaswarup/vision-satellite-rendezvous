"""
target package — Phase 2: Target Kinematics & Tumbling Model
=============================================================
Modules:
    quaternion  : Unit quaternion operations (multiply, DCM, Ξ matrix, Euler angles)
    attitude    : Rigid-body Euler equations + quaternion kinematic integrator
                  Debris body presets: ariane_upper_stage, cubesat_3u, custom_body
"""
from .quaternion import (
    qnormalize, qconjugate, qinverse, qmultiply,
    q_to_dcm, dcm_to_q,
    xi_matrix, qdot,
    q_to_euler321, euler321_to_q,
    attitude_error_deg, rotate_vector, rotate_vector_inv,
)
from .attitude import (
    RigidBodyAttitude,
    ariane_upper_stage, cubesat_3u, custom_body,
)

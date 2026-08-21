"""
dynamics package — Phase 1: Orbital Dynamics Engine
====================================================
Modules:
    constants   : Physical constants (mu, J2, R_E, ...)
    hcw         : Hill-Clohessy-Wiltshire propagator (analytical + numerical)
    ya_stm      : Yamanaka-Ankersen STM for elliptic orbits
    j2_perturb  : J2 perturbation model + chief ECI propagator
"""
from .constants   import *
from .hcw         import HCWPropagator
from .ya_stm      import YAStatTransition
from .j2_perturb  import (j2_accel_eci, differential_j2_lvlh,
                           J2PerturbedHCW, propagate_chief_eci,
                           keplerian_to_eci, lvlh_to_eci_rotation)

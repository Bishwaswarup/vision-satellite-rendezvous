"""
controller package — Phase 6: LQR / MPC Rendezvous Control
===========================================================
Modules:
    lqr  : Infinite-horizon discrete LQR via DARE
    mpc  : Receding-horizon MPC with thrust + corridor constraints
"""
from .lqr import LQRController, make_lqr, hcw_matrices, hcw_discrete, N_ORBITAL_DEFAULT
from .mpc import MPCController, make_mpc

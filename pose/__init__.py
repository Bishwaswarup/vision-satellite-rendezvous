"""
pose package — Phase 4: PnP Pose Estimation
============================================
Modules:
    epnp    : EPnP algorithm (Lepetit 2009) — O(n) closed-form PnP
    ransac  : RANSAC-EPnP robust estimator with local optimisation
    refine  : Gauss-Newton / Levenberg-Marquardt reprojection refinement
"""
from .epnp import (
    EPnPSolver,
    solve_epnp,
)
from .ransac import (
    RANSACSolver,
    solve_pnp_ransac,
)
from .refine import (
    GaussNewtonRefiner,
    refine_pose,
    _dcm_to_rodrigues,
    _rodrigues_to_dcm,
)

"""
vision package — Phase 3: Synthetic Vision Pipeline
====================================================
Modules:
    camera     : PinholeCamera model + preset cameras
    body_model : 3D debris geometry (Ariane, CubeSat, custom)
    renderer   : DebrisRenderer (projection, backface culling, Z-buffer, noise)
    dataset    : DatasetGenerator for synthetic pose datasets
"""
from .camera import (
    PinholeCamera,
    rendezvous_camera,
    wide_angle_camera,
)
from .body_model import (
    DebrisModel,
    ariane_model,
    cubesat_3u_model,
    custom_cylinder,
)
from .renderer import (
    RendererConfig,
    DebrisRenderer,
    random_rotation,
    look_at_rotation,
)
from .dataset import (
    DatasetConfig,
    DatasetGenerator,
    PoseAnnotation,
)

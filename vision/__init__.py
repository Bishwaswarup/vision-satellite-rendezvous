"""
vision package — Phase 3: Synthetic Vision Pipeline
====================================================
Modules:
    camera          : PinholeCamera model + preset cameras
    body_model      : 3D debris geometry (Ariane, CubeSat, custom)
    renderer        : DebrisRenderer (projection, backface culling, Z-buffer, noise)
    dataset         : DatasetGenerator for synthetic pose datasets
    vispy_renderer  : GPU-accelerated renderer (optional — requires vispy + pyopengl)
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

# Vispy GPU renderer is optional (requires: pip install "vispy>=0.14" pyopengl)
try:
    from .vispy_renderer import (
        VispyConfig,
        VispyRenderer,
        vispy_offscreen_renderer,
        vispy_interactive_renderer,
    )
    HAS_VISPY = True
except ImportError:
    HAS_VISPY = False

"""
simulation package — Phase 7: Closed-Loop Integration + Rendezvous Video
=========================================================================
Modules:
    runner : End-to-end simulation loop (vision → EKF → LQR → dynamics)
    video  : Wireframe renderer + figure-sequence → MP4 / GIF exporter
"""
from .runner import (
    SimConfig, SimResult, RendezvousSimulator,
    run_simulation, default_config,
)
from .video import VideoExporter, render_frame, frames_to_gif

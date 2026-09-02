"""
simulation package — Phase 7: Closed-Loop Integration + Rendezvous Video
=========================================================================
Modules:
    runner : End-to-end simulation loop (vision → EKF → LQR → dynamics)
    montecarlo : Dispersed Monte Carlo campaign driver
    video  : Wireframe renderer + figure-sequence → MP4 / GIF exporter
"""
from .runner import (
    SimConfig, SimResult, RendezvousSimulator,
    run_simulation, default_config,
)
from .video import VideoExporter, render_frame, frames_to_gif
from .montecarlo import (
    Dispersion, MonteCarloConfig, CampaignResult,
    run_trial, run_campaign, sweep, standard_configs, latex_table,
)

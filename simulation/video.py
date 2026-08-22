"""
video.py
========
Frame renderer and animation exporter for the rendezvous simulation.

Capabilities
------------
* render_frame(result, k, ax)  — draw one time-step of a SimResult onto a
  given matplotlib Axes (wireframe chaser view + trajectory trace).
* frames_to_gif(frames, path, fps) — encode a list of RGB numpy arrays as
  an animated GIF using matplotlib's PillowWriter.
* VideoExporter — high-level helper that renders all steps of a SimResult,
  builds an animated GIF, and optionally also writes a PNG summary sheet.

Dependencies
------------
  matplotlib, numpy, Pillow  (all in requirements.txt)
  No OpenCV or ffmpeg required — pure-Python rendering.
"""

import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 (side-effect import)

from vision.camera import rendezvous_camera
from vision.body_model import ariane_model
from vision.renderer import DebrisRenderer, RendererConfig
from estimator.state import quat_to_dcm

# ── Colour palette ────────────────────────────────────────────────────────────
BLUE   = '#4FC3F7'
AMBER  = '#FFB74D'
GREEN  = '#81C784'
RED    = '#EF9A9A'
GREY   = '#78909C'


# ── Single-frame renderer ─────────────────────────────────────────────────────

def render_frame(result, k: int, axes=None, figsize=(14, 5)):
    """
    Render time step `k` of `result` as a multi-panel figure.

    Panels
    ------
    Left   : synthetic wireframe image (chaser camera view)
    Centre : X-Y LVLH trajectory up to step k
    Right  : range and position error vs time up to step k

    Parameters
    ----------
    result : SimResult
    k      : int  time index (0 … n_steps)
    axes   : optional iterable of 3 Axes; if None a new figure is created
    figsize: (w, h) inches for new figure

    Returns
    -------
    fig : matplotlib Figure
    """
    from simulation.runner import SimResult
    assert isinstance(result, SimResult)

    cam   = rendezvous_camera()
    model = ariane_model()
    rend  = DebrisRenderer(cam)

    if axes is None:
        fig = plt.figure(figsize=figsize, facecolor='#1a1a2e')
        axes = [
            fig.add_subplot(1, 3, 1),
            fig.add_subplot(1, 3, 2),
            fig.add_subplot(1, 3, 3),
        ]
    else:
        fig = axes[0].figure

    ax_img, ax_xy, ax_err = axes

    # ── Panel 1: wireframe camera view ────────────────────────────────────────
    r_k = result.r_true[k]
    q_k = result.q_true[k]
    R_body = quat_to_dcm(q_k)

    img, _ = rend.render(model, R_body, r_k)
    ax_img.clear()
    ax_img.imshow(img, cmap='Blues_r', vmin=0, vmax=1)
    ax_img.set_title(f'Camera view  k={k}', color='white', fontsize=9)
    ax_img.axis('off')

    # ── Panel 2: X-Y trajectory ───────────────────────────────────────────────
    ax_xy.clear()
    ax_xy.set_facecolor('#1a1a2e')
    ax_xy.plot(result.r_true[:k+1, 0], result.r_true[:k+1, 1],
               color=BLUE, lw=1.5, label='True')
    ax_xy.plot(result.r_est[:k+1, 0],  result.r_est[:k+1, 1],
               color=AMBER, lw=1.2, ls='--', label='EKF est.')
    ax_xy.scatter(r_k[0], r_k[1], color=GREEN, s=50, zorder=5)
    ax_xy.scatter(0, 0, color=RED, s=80, marker='*', zorder=5, label='Target')
    ax_xy.set_xlabel('X radial [m]', color='white', fontsize=8)
    ax_xy.set_ylabel('Y along-track [m]', color='white', fontsize=8)
    ax_xy.set_title('LVLH trajectory', color='white', fontsize=9)
    ax_xy.legend(fontsize=7); ax_xy.grid(alpha=0.2)
    ax_xy.tick_params(colors='white', labelsize=7)

    # ── Panel 3: range + position error ──────────────────────────────────────
    ax_err.clear()
    ax_err.set_facecolor('#1a1a2e')
    t = np.arange(k + 1)
    rng  = np.linalg.norm(result.r_true[:k+1], axis=1)
    perr = result.pos_error[:k+1]
    ax_err.plot(t, rng,  color=BLUE,  lw=1.5, label='Range [m]')
    ax_err.plot(t, perr, color=AMBER, lw=1.5, ls='--', label='Est. error [m]')
    ax_err.set_xlabel('Step', color='white', fontsize=8)
    ax_err.set_ylabel('Distance [m]', color='white', fontsize=8)
    ax_err.set_title('Range & estimation error', color='white', fontsize=9)
    ax_err.legend(fontsize=7); ax_err.grid(alpha=0.2)
    ax_err.tick_params(colors='white', labelsize=7)

    fig.patch.set_facecolor('#1a1a2e')
    plt.tight_layout()
    return fig


def fig_to_rgb(fig) -> np.ndarray:
    """Convert a matplotlib figure to an (H, W, 3) uint8 RGB array."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=80, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0)
    import PIL.Image
    img = np.array(PIL.Image.open(buf).convert('RGB'))
    buf.close()
    return img


# ── GIF writer ────────────────────────────────────────────────────────────────

def frames_to_gif(frames: list, path: str, fps: int = 10) -> str:
    """
    Write a list of (H, W, 3) uint8 RGB arrays to an animated GIF.

    Parameters
    ----------
    frames : list of ndarray
    path   : output file path (should end in .gif)
    fps    : frames per second

    Returns
    -------
    path : str (same as input)
    """
    import PIL.Image
    pil_frames = [PIL.Image.fromarray(f) for f in frames]
    duration_ms = max(20, int(1000 / fps))
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        loop=0,
        duration=duration_ms,
        optimize=False,
    )
    return path


# ── High-level exporter ───────────────────────────────────────────────────────

class VideoExporter:
    """
    Renders selected frames of a SimResult and exports an animated GIF.

    Parameters
    ----------
    result  : SimResult
    fps     : target frame rate for the output GIF
    step    : render every `step`-th simulation step (default 5)
    figsize : figure size in inches
    """

    def __init__(self, result, fps: int = 8, step: int = 5,
                 figsize=(14, 5)):
        self.result  = result
        self.fps     = fps
        self.step    = step
        self.figsize = figsize

    def export_gif(self, path: str, verbose: bool = True) -> str:
        """
        Render frames and write animated GIF to `path`.

        Returns
        -------
        path : str
        """
        T = self.result.n_steps_run + 1
        indices = list(range(0, T, self.step))
        if (T - 1) not in indices:
            indices.append(T - 1)

        frames = []
        for i, k in enumerate(indices):
            if verbose and i % 10 == 0:
                print(f'  Rendering frame {i+1}/{len(indices)}  (step {k})')
            fig = render_frame(self.result, k, figsize=self.figsize)
            frames.append(fig_to_rgb(fig))
            plt.close(fig)

        frames_to_gif(frames, path, fps=self.fps)
        if verbose:
            print(f'  GIF saved → {path}  ({len(frames)} frames @ {self.fps} fps)')
        return path

    def export_summary_png(self, path: str) -> str:
        """
        Write a static summary figure (trajectory + errors over full run).

        Parameters
        ----------
        path : str  output .png path

        Returns
        -------
        path : str
        """
        res = self.result
        T   = res.n_steps_run + 1
        t   = np.arange(T)

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.patch.set_facecolor('#1a1a2e')
        plt.style.use('dark_background')

        # ── (0,0) X-Y trajectory ─────────────────────────────────────────
        ax = axes[0, 0]
        ax.plot(res.r_true[:T, 0], res.r_true[:T, 1],
                color=BLUE, lw=2, label='True')
        ax.plot(res.r_est[:T, 0],  res.r_est[:T, 1],
                color=AMBER, lw=1.5, ls='--', label='EKF est.')
        ax.scatter(0, 0, color=RED, s=100, marker='*', label='Target')
        ax.scatter(*res.r_true[0, :2], color=GREEN, s=80, label='Start')
        ax.set_xlabel('X radial [m]'); ax.set_ylabel('Y along-track [m]')
        ax.set_title('LVLH Trajectory (X-Y)', color='white')
        ax.legend(fontsize=8); ax.grid(alpha=0.2)

        # ── (0,1) Range vs time ───────────────────────────────────────────
        ax = axes[0, 1]
        rng = np.linalg.norm(res.r_true[:T], axis=1)
        ax.plot(t, rng, color=BLUE, lw=2)
        if res.dock_step:
            ax.axvline(res.dock_step, color=GREEN, ls='--', lw=1.5,
                       label=f'Docked @ step {res.dock_step}')
            ax.legend(fontsize=8)
        ax.set_xlabel('Step'); ax.set_ylabel('Range [m]')
        ax.set_title('Range vs time', color='white'); ax.grid(alpha=0.2)

        # ── (0,2) Position estimation error ───────────────────────────────
        ax = axes[0, 2]
        ax.plot(t, res.pos_error[:T], color=AMBER, lw=2)
        ax.set_xlabel('Step'); ax.set_ylabel('‖r_true − r_est‖  [m]')
        ax.set_title('Position estimation error', color='white')
        ax.grid(alpha=0.2)

        # ── (1,0) Velocity magnitude ──────────────────────────────────────
        ax = axes[1, 0]
        spd = np.linalg.norm(res.v_true[:T], axis=1)
        ax.plot(t, spd, color=GREEN, lw=2)
        ax.set_xlabel('Step'); ax.set_ylabel('Speed [m/s]')
        ax.set_title('Chaser speed', color='white'); ax.grid(alpha=0.2)

        # ── (1,1) Thrust profiles ─────────────────────────────────────────
        ax = axes[1, 1]
        t_ctrl = np.arange(res.n_steps_run)
        labels = ['$u_x$', '$u_y$', '$u_z$']
        cols   = [BLUE, AMBER, GREEN]
        for i, (lab, col) in enumerate(zip(labels, cols)):
            ax.plot(t_ctrl, res.controls[:res.n_steps_run, i],
                    color=col, lw=1.5, label=lab)
        ax.set_xlabel('Step'); ax.set_ylabel('Thrust [m/s²]')
        ax.set_title('Control thrust', color='white')
        ax.legend(fontsize=8); ax.grid(alpha=0.2)

        # ── (1,2) EPnP reprojection error ─────────────────────────────────
        ax = axes[1, 2]
        valid = np.isfinite(res.repr_errs[:res.n_steps_run])
        ax.plot(t_ctrl[valid], res.repr_errs[:res.n_steps_run][valid],
                color=RED, lw=1.5)
        ax.set_xlabel('Step'); ax.set_ylabel('Repr. error [px]')
        ax.set_title('EPnP reprojection error', color='white')
        ax.grid(alpha=0.2)

        fig.suptitle('Phase 7 — Full Closed-Loop Rendezvous Summary',
                     fontsize=14, color='white')
        plt.tight_layout()
        fig.savefig(path, dpi=120, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        return path

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

from viz import (apply_style, series_kw, style_ax, save_fig,
                 add_panel_label, GREY, IMAGE_CMAP)

# Print-safe monochrome: series are separated by grey level, line style and
# marker, never by hue, so the animation and its still frames survive a
# greyscale print and read correctly to colour-blind viewers.
apply_style()


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
        fig = plt.figure(figsize=figsize, facecolor=GREY['page'])
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
    # The renderer returns an RGB canvas, and imshow ignores `cmap` for 3-D
    # input, so fold it to luminance first (Rec. 709) to keep the figure
    # genuinely monochrome rather than merely looking that way.
    from simulation.dualview import _luminance
    ax_img.imshow(_luminance(img), cmap=IMAGE_CMAP, vmin=0.0, vmax=1.0)
    ax_img.set_title(f'Camera view   step {k}')
    ax_img.axis('off')

    n_steps = max(len(result.r_true) - 1, 1)
    rng_k = float(np.linalg.norm(r_k))

    # ── Panel 2: X-Y trajectory ───────────────────────────────────────────────
    ax_xy.clear()
    ax_xy.set_facecolor(GREY['panel'])
    ax_xy.plot(result.r_true[:k+1, 0], result.r_true[:k+1, 1],
               label='True', **series_kw(0))
    ax_xy.plot(result.r_est[:k+1, 0], result.r_est[:k+1, 1],
               label='Estimate', **series_kw(1))
    # Current position: an open marker reads clearly on any grey.
    ax_xy.plot(r_k[0], r_k[1], marker='o', markersize=7,
               markerfacecolor=GREY['page'], markeredgecolor=GREY['ink'],
               markeredgewidth=1.4, linestyle='none', zorder=5)
    # Target at the origin.
    ax_xy.plot(0, 0, marker='*', markersize=13, color=GREY['ink'],
               linestyle='none', zorder=5, label='Target')
    style_ax(ax_xy, xlabel='$x$  radial [m]', ylabel='$y$  along-track [m]',
             title='LVLH trajectory', legend=True, legend_loc='best')

    # ── Panel 3: range + estimation error ─────────────────────────────────────
    ax_err.clear()
    ax_err.set_facecolor(GREY['panel'])
    t = np.arange(k + 1)
    rng = np.linalg.norm(result.r_true[:k+1], axis=1)
    perr = result.pos_error[:k+1]
    ax_err.plot(t, rng, label='Range', **series_kw(0))
    ax_err.plot(t, perr, label='Estimation error', **series_kw(1))
    ax_err.set_xlim(0, n_steps)
    style_ax(ax_err, xlabel='Step', ylabel='Distance [m]',
             title='Range and estimation error', legend=True,
             legend_loc='upper right')

    fig.suptitle(f'Range {rng_k:6.2f} m', y=1.0, fontsize=10)
    fig.patch.set_facecolor(GREY['page'])
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

        fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
        fig.patch.set_facecolor(GREY['page'])

        # ── (0,0) X-Y trajectory ─────────────────────────────────────────
        ax = axes[0, 0]
        ax.plot(res.r_true[:T, 0], res.r_true[:T, 1],
                label='True', **series_kw(0))
        ax.plot(res.r_est[:T, 0], res.r_est[:T, 1],
                label='Estimate', **series_kw(1))
        ax.plot(0, 0, marker='*', markersize=13, color=GREY['ink'],
                linestyle='none', label='Target')
        ax.plot(res.r_true[0, 0], res.r_true[0, 1], marker='o', markersize=7,
                markerfacecolor=GREY['page'], markeredgecolor=GREY['ink'],
                markeredgewidth=1.4, linestyle='none', label='Start')
        style_ax(ax, xlabel='$x$  radial [m]', ylabel='$y$  along-track [m]',
                 title='LVLH trajectory', legend=True)
        add_panel_label(ax, '(a)')

        # ── (0,1) Range vs time ───────────────────────────────────────────
        ax = axes[0, 1]
        rng = np.linalg.norm(res.r_true[:T], axis=1)
        ax.plot(t, rng, **series_kw(0))
        if res.dock_step:
            ax.axvline(res.dock_step, color=GREY['mid'], linestyle=(0, (4, 2)),
                       linewidth=1.1, label=f'Docked, step {res.dock_step}')
            ax.legend()
        style_ax(ax, xlabel='Step', ylabel='Range [m]', title='Range')
        add_panel_label(ax, '(b)')

        # ── (0,2) Position estimation error ───────────────────────────────
        ax = axes[0, 2]
        ax.plot(t, res.pos_error[:T], **series_kw(0))
        style_ax(ax, xlabel='Step',
                 ylabel=r'$\|\mathbf{r}-\hat{\mathbf{r}}\|$  [m]',
                 title='Navigation error')
        add_panel_label(ax, '(c)')

        # ── (1,0) Speed ───────────────────────────────────────────────────
        ax = axes[1, 0]
        ax.plot(t, np.linalg.norm(res.v_true[:T], axis=1), **series_kw(0))
        style_ax(ax, xlabel='Step', ylabel='Speed [m/s]',
                 title='Closing speed')
        add_panel_label(ax, '(d)')

        # ── (1,1) Thrust profiles ─────────────────────────────────────────
        ax = axes[1, 1]
        t_ctrl = np.arange(res.n_steps_run)
        for i, lab in enumerate(('$u_x$', '$u_y$', '$u_z$')):
            ax.plot(t_ctrl, res.controls[:res.n_steps_run, i],
                    label=lab, **series_kw(i))
        style_ax(ax, xlabel='Step', ylabel=r'Thrust [m/s$^2$]',
                 title='Control', legend=True)
        add_panel_label(ax, '(e)')

        # ── (1,2) Measurement availability and reprojection error ─────────
        ax = axes[1, 2]
        valid = np.isfinite(res.repr_errs[:res.n_steps_run])
        if valid.any():
            ax.plot(t_ctrl[valid], res.repr_errs[:res.n_steps_run][valid],
                    linestyle='none', marker='.', markersize=3,
                    color=GREY['ink'])
        # Shade the steps where no pose fix was available.
        if res.meas_used is not None:
            gaps = ~res.meas_used[:res.n_steps_run]
            if gaps.any():
                ax.fill_between(t_ctrl, 0, 1, where=gaps,
                                transform=ax.get_xaxis_transform(),
                                facecolor=GREY['faint'], alpha=0.55,
                                linewidth=0, label='No pose fix')
                ax.legend(loc='upper left')
        style_ax(ax, xlabel='Step', ylabel='Reprojection error [px]',
                 title='Vision availability')
        add_panel_label(ax, '(f)')

        drop = (res.vision_dropout_rate * 100
                if res.cfg.use_vision else 0.0)
        fig.suptitle(
            f'Closed-loop rendezvous — {res.n_steps_run} steps, '
            f'$\\Delta v$ = {res.delta_v:.2f} m/s, '
            f'vision dropout {drop:.0f}%', fontsize=11, y=1.01)
        plt.tight_layout()
        save_fig(fig, path, dpi=300)
        return path

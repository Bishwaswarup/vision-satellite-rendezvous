"""
dualview.py
===========
Two-panel visualisation of a closed-loop rendezvous run.

    LEFT   what the chaser's camera sees: the shaded target, the projected
           keypoints, and which of them pass the visibility gate
    RIGHT  where the chaser is: the target as a solid body with its true
           attitude, the chaser and its camera frustum, and the flown path,
           all in the LVLH frame

Both panels are driven entirely by an already-computed `SimResult`. Nothing
here integrates anything: the trajectory, the attitude and the pose fixes come
from the validated propagators, and this module only draws them. That
separation is deliberate — a visualiser that re-simulates is a second, silently
different model.

Usage
-----
    from simulation.runner import RendezvousSimulator, SimConfig
    from simulation.dualview import export_dual_gif

    res = RendezvousSimulator(SimConfig(n_steps=600), rng_seed=42).run()
    export_dual_gif(res, 'outputs/dualview.gif')
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from vision.body_model import ariane_model
from vision.renderer import DebrisRenderer, RendererConfig
from estimator.state import quat_to_dcm
from viz import apply_style, series_kw, GREY, IMAGE_CMAP, save_fig

apply_style()


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _luminance(img: np.ndarray) -> np.ndarray:
    """
    Fold a rendered canvas to Rec. 709 luminance in [0, 1].

    `DebrisRenderer` returns an 0-255 canvas (its background is (5, 5, 10) and
    its faces are (180, 200, 220)), so the result must be normalised before
    display: `imshow(..., vmax=1.0)` on raw 0-255 data clips every lit pixel to
    white and the panel appears blank.
    """
    img = np.asarray(img, dtype=float)
    if img.ndim == 3:
        img = (0.2126 * img[..., 0] + 0.7152 * img[..., 1]
               + 0.0722 * img[..., 2])
    if img.size and img.max() > 1.5:          # 0-255 canvas
        img = img / 255.0
    return np.clip(img, 0.0, 1.0)


def _frustum_rays(cam, R_cl: np.ndarray, length: float) -> np.ndarray:
    """
    Four corner rays of the camera frustum, expressed in LVLH.

    The camera looks along its own +z; a corner of the image plane at unit
    depth sits at (±W/2f_x, ±H/2f_y, 1), so scaling by `length` and rotating
    by R_cl^T gives the ray in LVLH.
    """
    ax = 0.5 * cam.width / cam.K[0, 0]
    ay = 0.5 * cam.height / cam.K[1, 1]
    corners = np.array([[-ax, -ay, 1.0], [ax, -ay, 1.0],
                        [ax, ay, 1.0], [-ax, ay, 1.0]])
    corners = corners / np.linalg.norm(corners, axis=1, keepdims=True)
    return (R_cl.T @ (corners * length).T).T


def _shade(normals_lvlh: np.ndarray, light_dir: np.ndarray) -> np.ndarray:
    """
    Lambertian grey level per face, in [0.18, 0.95].

    A single directional light plus a fixed ambient term is enough to read the
    body's shape and its rotation; the ambient floor keeps back faces visible
    rather than black.
    """
    light_dir = light_dir / np.linalg.norm(light_dir)
    lam = np.clip(normals_lvlh @ light_dir, 0.0, 1.0)
    return 0.18 + 0.77 * lam


# ── Single frame ──────────────────────────────────────────────────────────────

def render_dual_frame(result, k: int, sim=None, model=None,
                      figsize=(12.5, 5.6), elev: float = 22.0,
                      azim: float = -58.0, view_model=None):
    """
    Draw step `k` of `result` as a camera panel and a 3-D panel.

    Parameters
    ----------
    result     : SimResult
    k          : time index into result.r_true
    sim        : RendezvousSimulator whose camera geometry produced `result`.
                 Rebuilt from result.cfg if omitted.
    model      : DebrisModel for the camera panel (full resolution)
    view_model : coarser DebrisModel for the 3-D panel; the external view does
                 not need 24 circumferential segments and matplotlib's 3-D
                 renderer is the bottleneck when animating.
    elev, azim : 3-D viewing angles [deg]

    Returns
    -------
    fig : matplotlib Figure
    """
    from simulation.runner import RendezvousSimulator

    if sim is None:
        sim = RendezvousSimulator(result.cfg, rng_seed=0)
    if model is None:
        model = sim.model
    if view_model is None:
        view_model = ariane_model(n_circ=12, n_long=3)

    cam = sim.cam
    r_k = np.asarray(result.r_true[k], dtype=float)
    q_k = np.asarray(result.q_true[k], dtype=float)
    R_bl = quat_to_dcm(q_k)                       # body -> LVLH

    # The chaser points its camera using its own ESTIMATE of the target, which
    # is what the closed loop does; using the truth here would draw a view the
    # spacecraft could not have produced.
    k_est = min(k, len(result.r_est) - 1)
    R_cl = sim.camera_attitude(result.r_est[k_est])

    rng_k = float(np.linalg.norm(r_k))
    n_steps = max(result.n_steps_run, 1)

    fig = plt.figure(figsize=figsize, facecolor=GREY['page'])
    ax_cam = fig.add_subplot(1, 2, 1)
    ax_3d = fig.add_subplot(1, 2, 2, projection='3d')

    # ── Panel 1: camera feed ─────────────────────────────────────────────────
    R_bc = R_cl @ R_bl                            # body -> camera
    t_bc = R_cl @ r_k                             # target origin in camera
    rend = DebrisRenderer(cam, RendererConfig(noise_sigma=0.0))
    img, _ = rend.render(model, R_bc, t_bc)

    ax_cam.imshow(_luminance(img), cmap=IMAGE_CMAP, vmin=0.0, vmax=1.0,
                  origin='upper')

    # Keypoints, split by the visibility gate the estimator actually applies.
    _, kp2d, visible, _ = sim._project_keypoints(r_k, q_k, R_cl)
    if visible.any():
        ax_cam.plot(kp2d[visible, 0], kp2d[visible, 1], linestyle='none',
                    marker='o', markersize=5.5, markerfacecolor='none',
                    markeredgecolor=GREY['ink'], markeredgewidth=1.3,
                    label=f'in frame ({int(visible.sum())})')
    culled = ~visible
    inside = ((kp2d[:, 0] >= 0) & (kp2d[:, 0] < cam.width)
              & (kp2d[:, 1] >= 0) & (kp2d[:, 1] < cam.height))
    show_x = culled & inside
    if show_x.any():
        ax_cam.plot(kp2d[show_x, 0], kp2d[show_x, 1], linestyle='none',
                    marker='x', markersize=5.5, color=GREY['light'],
                    markeredgewidth=1.3, label='behind camera')

    ax_cam.set_xlim(0, cam.width)
    ax_cam.set_ylim(cam.height, 0)
    ax_cam.set_xticks([]); ax_cam.set_yticks([])
    for side in ax_cam.spines.values():
        side.set_color(GREY['mid'])
        side.set_linewidth(1.0)

    n_vis = int(visible.sum())
    fixed = (result.meas_used[k - 1]
             if (result.meas_used is not None and 0 < k <= len(result.meas_used))
             else None)
    if fixed is None:
        status = ''
    elif fixed:
        status = 'pose fix: OK'
    else:
        status = 'pose fix: NO SOLUTION'

    ax_cam.set_title(f'Chaser camera   step {k}', pad=6)
    ax_cam.text(0.02, 0.975,
                f'range {rng_k:6.2f} m\nkeypoints in frame {n_vis:2d} / '
                f'{len(visible)}\n{status}',
                transform=ax_cam.transAxes, va='top', ha='left',
                fontsize=8.5, family='monospace', color=GREY['ink'],
                bbox=dict(boxstyle='square,pad=0.4', facecolor=GREY['page'],
                          edgecolor=GREY['grid'], linewidth=0.7))
    if n_vis or show_x.any():
        ax_cam.legend(loc='lower right', fontsize=7.5)

    # ── Panel 2: external 3-D view ───────────────────────────────────────────
    verts = (R_bl @ view_model.vertices.T).T + r_k          # LVLH
    norms = (R_bl @ view_model.face_normals.T).T
    light = np.array([0.4, -0.8, 0.45])
    greys = _shade(norms, light)

    polys, cols = [], []
    for f, g in zip(view_model.faces, greys):
        idx = [i for i in np.atleast_1d(f) if i >= 0]
        if len(idx) >= 3:
            polys.append(verts[idx])
            cols.append((g, g, g))
    body = Poly3DCollection(polys, facecolors=cols,
                            edgecolors=GREY['mid'], linewidths=0.25)
    body.set_zsort('average')
    ax_3d.add_collection3d(body)

    # Flown path, and the remainder faintly so the frame does not rescale.
    P = np.asarray(result.r_true[:n_steps + 1], dtype=float)
    ax_3d.plot(P[:k + 1, 0], P[:k + 1, 1], P[:k + 1, 2],
               **series_kw(0), zorder=4)
    if k + 1 < len(P):
        ax_3d.plot(P[k:, 0], P[k:, 1], P[k:, 2],
                   color=GREY['faint'], linestyle=(0, (1, 2)), linewidth=1.0)

    # Chaser at the LVLH origin, with its camera frustum.
    ax_3d.plot([0], [0], [0], marker='o', markersize=7,
               markerfacecolor=GREY['page'], markeredgecolor=GREY['ink'],
               markeredgewidth=1.5, linestyle='none', zorder=5)
    rays = _frustum_rays(cam, R_cl, max(rng_k, 1.0))
    for c in rays:
        ax_3d.plot([0, c[0]], [0, c[1]], [0, c[2]],
                   color=GREY['light'], linewidth=0.7,
                   linestyle=(0, (3, 2)))
    loop = np.vstack([rays, rays[:1]])
    ax_3d.plot(loop[:, 0], loop[:, 1], loop[:, 2],
               color=GREY['light'], linewidth=0.7)

    # Frame the scene on the initial separation so the animation is steady.
    span = float(np.abs(P).max()) * 1.15 + 2.0
    ax_3d.set_xlim(-span * 0.15, span)
    ax_3d.set_ylim(-span * 0.55, span * 0.55)
    ax_3d.set_zlim(-span * 0.55, span * 0.55)
    try:
        ax_3d.set_box_aspect((1.15, 1.1, 1.1))
    except Exception:                                    # older matplotlib
        pass
    ax_3d.view_init(elev=elev, azim=azim)

    ax_3d.set_xlabel('$x$ radial [m]', labelpad=2)
    ax_3d.set_ylabel('$y$ along-track [m]', labelpad=2)
    ax_3d.set_zlabel('$z$ cross-track [m]', labelpad=6)
    ax_3d.set_title('LVLH frame   (chaser at origin)', pad=6)
    for pane in (ax_3d.xaxis, ax_3d.yaxis, ax_3d.zaxis):
        pane.set_pane_color((1.0, 1.0, 1.0, 0.0))
        pane._axinfo['grid']['color'] = GREY['grid']
        pane._axinfo['grid']['linewidth'] = 0.5
    ax_3d.tick_params(labelsize=7.5)

    fig.subplots_adjust(left=0.02, right=0.94, top=0.90, bottom=0.06,
                        wspace=0.06)
    return fig


# ── Animation ─────────────────────────────────────────────────────────────────

def export_dual_gif(result, path: str, fps: int = 8, step: int = 1,
                    spin: float = 0.0, verbose: bool = True) -> str:
    """
    Render every `step`-th frame of `result` and encode an animated GIF.

    Parameters
    ----------
    result  : SimResult
    path    : output .gif path
    fps     : frame rate
    step    : render every Nth simulation step
    spin    : degrees of azimuth added per rendered frame, to slowly orbit the
              3-D view; 0 keeps it fixed
    """
    from simulation.runner import RendezvousSimulator
    from simulation.video import fig_to_rgb, frames_to_gif

    sim = RendezvousSimulator(result.cfg, rng_seed=0)
    view_model = ariane_model(n_circ=12, n_long=3)

    ks = list(range(0, result.n_steps_run + 1, max(1, step)))
    frames = []
    for i, k in enumerate(ks):
        fig = render_dual_frame(result, k, sim=sim, view_model=view_model,
                               azim=-58.0 + spin * i)
        frames.append(fig_to_rgb(fig))
        plt.close(fig)
        if verbose and (i % 10 == 0 or i == len(ks) - 1):
            print(f'    frame {i + 1:4d}/{len(ks)}  (step {k})', flush=True)

    return frames_to_gif(frames, path, fps=fps)


def export_dual_contact_sheet(result, path: str, n: int = 6) -> str:
    """
    A single still figure with `n` frames sampled across the approach.

    A contact sheet is often more useful than the animation for a paper or a
    slide: it shows the same geometry evolving without needing playback.
    """
    from simulation.runner import RendezvousSimulator

    sim = RendezvousSimulator(result.cfg, rng_seed=0)
    view_model = ariane_model(n_circ=12, n_long=3)
    ks = np.linspace(0, result.n_steps_run, n, dtype=int)

    fig, axes = plt.subplots(2, n, figsize=(2.5 * n, 5.6),
                             facecolor=GREY['page'])
    rend = DebrisRenderer(sim.cam, RendererConfig(noise_sigma=0.0))

    for j, k in enumerate(ks):
        r_k = np.asarray(result.r_true[k], dtype=float)
        q_k = np.asarray(result.q_true[k], dtype=float)
        R_cl = sim.camera_attitude(result.r_est[min(k, len(result.r_est) - 1)])
        R_bl = quat_to_dcm(q_k)

        img, _ = rend.render(view_model, R_cl @ R_bl, R_cl @ r_k)
        ax = axes[0, j]
        ax.imshow(_luminance(img), cmap=IMAGE_CMAP, vmin=0, vmax=1,
                  aspect='equal')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'{np.linalg.norm(r_k):.1f} m', fontsize=9, pad=4)

        _, kp2d, visible, _ = sim._project_keypoints(r_k, q_k, R_cl)
        ax = axes[1, j]
        ax.plot(kp2d[visible, 0], kp2d[visible, 1], linestyle='none',
                marker='o', markersize=3.5, markerfacecolor='none',
                markeredgecolor=GREY['ink'], markeredgewidth=1.0)
        ax.set_xlim(0, sim.cam.width); ax.set_ylim(sim.cam.height, 0)
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor(GREY['panel'])
        n_ok = int(visible.sum())
        ax.set_xlabel(f'{n_ok}/{len(visible)}'
                      + ('' if n_ok >= 6 else '   no solve'), fontsize=8)

    axes[0, 0].set_ylabel('camera', fontsize=9)
    axes[1, 0].set_ylabel('keypoints in frame', fontsize=9)
    fig.suptitle('Approach sequence: the target overflows the frame as range '
                 'closes.  Six keypoints are needed for a pose solve.',
                 fontsize=10, y=1.0)
    fig.tight_layout()
    return save_fig(fig, path)

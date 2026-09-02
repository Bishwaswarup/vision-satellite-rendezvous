"""
renderer.py
===========
Synthetic image renderer for space debris inspection.

Pipeline
--------
  1. Transform body vertices into camera frame   (R_cw, t_cw)
  2. Backface culling — discard faces whose normal points away from camera
  3. Project surviving vertices with PinholeCamera
  4. Painter's sort (depth of face centroid) — far-to-near drawing order
  5. Rasterise: draw filled quads/triangles (shaded) + wireframe edges
  6. Overlay keypoints
  7. Add Gaussian pixel noise + optional background stars

Returns a numpy uint8 image (H × W × 3) in BGR or RGB.

Usage
-----
  from vision import RendererConfig, DebrisRenderer, ariane_model, rendezvous_camera
  cam    = rendezvous_camera()
  model  = ariane_model()
  rend   = DebrisRenderer(cam)
  img, meta = rend.render(model, R_cw, t_cw)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict
from .camera import PinholeCamera
from .body_model import DebrisModel


@dataclass
class RendererConfig:
    """Rendering parameters."""
    ambient       : float = 0.25     # Ambient light fraction  [0, 1]
    diffuse       : float = 0.75     # Diffuse light fraction  [0, 1]
    light_dir_cam : np.ndarray = field(
        default_factory=lambda: np.array([-0.3, -0.5, 1.0]))
                                     # Light direction in camera frame (unnorm)
    face_color    : Tuple  = (180, 200, 220)   # Base RGB colour of body
    edge_color    : Tuple  = (255, 255,  80)   # Wireframe edge colour
    kpt_color     : Tuple  = (  0, 255,   0)   # Keypoint marker colour
    kpt_radius    : int    = 4
    draw_faces    : bool   = True
    draw_edges    : bool   = True
    draw_keypoints: bool   = True
    noise_sigma   : float  = 1.5     # Gaussian noise std [DN, 0-255]
    star_density  : float  = 0.0003  # Expected stars per pixel
    bg_color      : Tuple  = (  5,   5,  10)   # Background (space) RGB
    image_mode    : str    = 'RGB'   # 'RGB' or 'BGR'


class DebrisRenderer:
    """
    Software rasteriser for debris body visualisation.

    Parameters
    ----------
    camera : PinholeCamera
    config : RendererConfig  (optional, defaults constructed if None)
    """

    def __init__(self, camera: PinholeCamera,
                 config: RendererConfig = None):
        self.cam  = camera
        self.cfg  = config or RendererConfig()
        # Normalise light direction
        ld = np.asarray(self.cfg.light_dir_cam, dtype=float)
        self._light = ld / (np.linalg.norm(ld) + 1e-30)

    # ── Main render entry point ───────────────────────────────────────────────
    def render(self,
               model   : DebrisModel,
               R_cw    : np.ndarray,
               t_cw    : np.ndarray,
               seed    : int = None) -> Tuple[np.ndarray, dict]:
        """
        Render a debris model from a given camera pose.

        Parameters
        ----------
        model : DebrisModel
        R_cw  : (3,3) rotation world → camera
        t_cw  : (3,)  translation (camera origin in camera frame = -R_cw @ p_cam_world)
        seed  : int or None  RNG seed for reproducible noise

        Returns
        -------
        image : np.ndarray uint8 (H, W, 3)
        meta  : dict   {
                    'proj_keypoints': {name: (u,v)},
                    'keypoints_2d'  : (K, 2) array of visible kpt pixels,
                    'keypoint_names': [str],
                    'bbox'          : (u_min, v_min, u_max, v_max) or None,
                    'n_visible_faces': int,
                    'pose_R'        : R_cw,
                    'pose_t'        : t_cw,
                }
        """
        rng = np.random.default_rng(seed)
        R   = np.asarray(R_cw, dtype=float)
        t   = np.asarray(t_cw, dtype=float).ravel()

        # ── Canvas ────────────────────────────────────────────────────────
        H, W = self.cam.height, self.cam.width
        canvas = np.full((H, W, 3), self.cfg.bg_color, dtype=np.float32)
        zbuf   = np.full((H, W), np.inf)          # Z-buffer (depth per pixel)

        # ── Star field ────────────────────────────────────────────────────
        if self.cfg.star_density > 0:
            n_stars = int(self.cfg.star_density * H * W)
            su = rng.integers(0, W, n_stars)
            sv = rng.integers(0, H, n_stars)
            sb = rng.uniform(150, 255, n_stars).astype(np.float32)
            canvas[sv, su] = np.stack([sb, sb, sb], axis=-1)

        # ── Transform vertices to camera frame ────────────────────────────
        V_cam = (R @ model.vertices.T).T + t          # (N, 3)

        # ── Project all vertices ───────────────────────────────────────────
        uv_all, z_all, vis_all = self.cam.project(V_cam, clip=False)

        # ── Transform face normals to camera frame ────────────────────────
        N_cam = (R @ model.face_normals.T).T          # (F, 3)

        # ── Build render list: (depth, face_idx) sorted far→near ──────────
        render_faces = []
        for fi in range(model.n_faces):
            face = model.faces[fi]
            valid_verts = face[face >= 0]
            if len(valid_verts) < 3:
                continue

            # Backface culling: face normal dot (face_centroid direction)
            centroid_cam = V_cam[valid_verts].mean(axis=0)
            view_dir = centroid_cam / (np.linalg.norm(centroid_cam) + 1e-30)
            if np.dot(N_cam[fi], view_dir) > 0.0:
                continue   # facing away

            # Depth = z of centroid
            depth = centroid_cam[2]
            if depth <= 0:
                continue

            render_faces.append((depth, fi))

        render_faces.sort(key=lambda x: -x[0])   # far to near (painter's)

        # ── Rasterise faces ───────────────────────────────────────────────
        if self.cfg.draw_faces:
            for depth, fi in render_faces:
                face = model.faces[fi]
                valid_verts = face[face >= 0]

                # Check all vertices in front
                if not all(z_all[v] > 0 for v in valid_verts):
                    continue

                # Pixel coords of vertices
                pts = np.array([[int(round(uv_all[v, 0])),
                                 int(round(uv_all[v, 1]))]
                                for v in valid_verts], dtype=np.int32)

                # Diffuse shading
                cos_a = max(0.0, -float(np.dot(N_cam[fi], self._light)))
                shade = self.cfg.ambient + self.cfg.diffuse * cos_a
                color = tuple(min(255, int(c * shade)) for c in self.cfg.face_color)

                _fill_polygon(canvas, zbuf, pts, color, float(depth))

        # ── Rasterise edges (wireframe) ───────────────────────────────────
        if self.cfg.draw_edges:
            for (ia, ib) in model.edges:
                if not (vis_all[ia] or vis_all[ib]):
                    if z_all[ia] <= 0 and z_all[ib] <= 0:
                        continue
                ua, va = int(round(uv_all[ia, 0])), int(round(uv_all[ia, 1]))
                ub, vb = int(round(uv_all[ib, 0])), int(round(uv_all[ib, 1]))
                _draw_line(canvas, ua, va, ub, vb,
                           self.cfg.edge_color, W, H)

        # ── Keypoint projection & overlay ─────────────────────────────────
        kpt_2d  = {}
        proj_uv = {}
        if model.keypoints:
            kp_world = model.keypoint_array
            kp_uv, kp_z, kp_vis = self.cam.project_world(
                kp_world, R, t, clip=True)

            for j, name in enumerate(model.keypoint_names):
                if kp_vis[j]:
                    pu, pv = int(round(kp_uv[j, 0])), int(round(kp_uv[j, 1]))
                    proj_uv[name] = (float(kp_uv[j, 0]), float(kp_uv[j, 1]))
                    kpt_2d[name]  = (float(kp_uv[j, 0]), float(kp_uv[j, 1]))
                    if self.cfg.draw_keypoints:
                        r = self.cfg.kpt_radius
                        v0, v1 = max(0, pv-r), min(H-1, pv+r)
                        u0, u1 = max(0, pu-r), min(W-1, pu+r)
                        canvas[v0:v1+1, u0:u1+1] = self.cfg.kpt_color

        # ── Bounding box from all projected body vertices ─────────────────
        body_vis = vis_all & (z_all > 0)
        if body_vis.any():
            us = uv_all[body_vis, 0]
            vs = uv_all[body_vis, 1]
            bbox = (float(us.min()), float(vs.min()),
                    float(us.max()), float(vs.max()))
        else:
            bbox = None

        # ── Gaussian noise ────────────────────────────────────────────────
        if self.cfg.noise_sigma > 0:
            noise = rng.normal(0, self.cfg.noise_sigma,
                               canvas.shape).astype(np.float32)
            canvas = canvas + noise

        # ── Finalise ──────────────────────────────────────────────────────
        image = np.clip(canvas, 0, 255).astype(np.uint8)
        if self.cfg.image_mode == 'BGR':
            image = image[:, :, ::-1]

        kpt_names_vis  = list(kpt_2d.keys())
        kpt_arr_vis    = np.array(list(kpt_2d.values()), dtype=float) \
                         if kpt_2d else np.zeros((0, 2))

        meta = {
            'proj_keypoints'  : proj_uv,
            'keypoints_2d'    : kpt_arr_vis,
            'keypoint_names'  : kpt_names_vis,
            'bbox'            : bbox,
            'n_visible_faces' : len(render_faces),
            'pose_R'          : R_cw,
            'pose_t'          : t_cw,
        }
        return image, meta

    # ── Depth map ─────────────────────────────────────────────────────────────
    def render_depth(self,
                     model : DebrisModel,
                     R_cw  : np.ndarray,
                     t_cw  : np.ndarray) -> np.ndarray:
        """
        Render a float32 depth map (H, W).
        Pixels with no geometry hit are set to np.inf.
        """
        H, W = self.cam.height, self.cam.width
        R = np.asarray(R_cw, dtype=float)
        t = np.asarray(t_cw, dtype=float).ravel()

        V_cam  = (R @ model.vertices.T).T + t
        uv_all, z_all, _ = self.cam.project(V_cam, clip=False)
        N_cam  = (R @ model.face_normals.T).T
        zbuf   = np.full((H, W), np.inf, dtype=np.float32)

        for fi in range(model.n_faces):
            face = model.faces[fi]
            valid_verts = face[face >= 0]
            if len(valid_verts) < 3:
                continue
            centroid_cam = V_cam[valid_verts].mean(axis=0)
            view_dir = centroid_cam / (np.linalg.norm(centroid_cam) + 1e-30)
            if np.dot(N_cam[fi], view_dir) > 0.0:
                continue
            depth = centroid_cam[2]
            if depth <= 0:
                continue
            pts = np.array([[int(round(uv_all[v, 0])),
                             int(round(uv_all[v, 1]))]
                            for v in valid_verts], dtype=np.int32)
            _fill_polygon_depth(zbuf, pts, float(depth), W, H)

        return zbuf


# ── Rasterisation helpers (pure NumPy, no OpenCV dependency) ─────────────────

def _fill_polygon(canvas, zbuf, pts, color, depth):
    """
    Fill a convex polygon on the canvas using scanline rasterisation.
    Writes to zbuf; only overwrites pixels where depth < current zbuf.
    """
    H, W = canvas.shape[:2]
    if len(pts) < 3:
        return

    y_min = max(0, pts[:, 1].min())
    y_max = min(H - 1, pts[:, 1].max())
    if y_min > y_max:
        return

    n = len(pts)
    for y in range(y_min, y_max + 1):
        xs = []
        for i in range(n):
            x0, y0 = pts[i]
            x1, y1 = pts[(i + 1) % n]
            if (y0 <= y < y1) or (y1 <= y < y0):
                if y1 != y0:
                    # float division: the integer form overflows once a vertex
                    # near the camera plane projects to ~1e9 px
                    x = float(x0) + (y - float(y0)) * \
                        (float(x1) - float(x0)) / (float(y1) - float(y0))
                    if np.isfinite(x):
                        xs.append(x)
        if len(xs) < 2:
            continue
        x_left  = max(0,     int(np.floor(min(xs))))
        x_right = min(W - 1, int(np.ceil(max(xs))))
        if x_left > x_right:
            continue
        mask = zbuf[y, x_left:x_right+1] > depth
        canvas[y, x_left:x_right+1][mask] = color
        zbuf[y, x_left:x_right+1][mask]   = depth


def _fill_polygon_depth(zbuf, pts, depth, W, H):
    """Fill polygon into depth buffer only."""
    if len(pts) < 3:
        return
    y_min = max(0, pts[:, 1].min())
    y_max = min(H - 1, pts[:, 1].max())
    n = len(pts)
    for y in range(y_min, y_max + 1):
        xs = []
        for i in range(n):
            x0, y0 = pts[i]
            x1, y1 = pts[(i + 1) % n]
            if (y0 <= y < y1) or (y1 <= y < y0):
                if y1 != y0:
                    xs.append(x0 + (y - y0) * (x1 - x0) / (y1 - y0))
        if len(xs) < 2:
            continue
        x_left  = max(0,     int(np.floor(min(xs))))
        x_right = min(W - 1, int(np.ceil(max(xs))))
        mask = zbuf[y, x_left:x_right+1] > depth
        zbuf[y, x_left:x_right+1][mask] = depth


def _clip_segment(u0, v0, u1, v1, W, H, margin=4):
    """
    Cohen-Sutherland clip of a segment to a slightly enlarged image rectangle.

    A vertex close to the camera plane projects to a coordinate of order
    f * X / z, which grows without bound as z -> 0.  Rasterising such a
    segment costs O(|u|) Python iterations — at z = 1e-5 m that is ~1e8 steps,
    i.e. an apparent hang — and at z = 1e-9 the coordinate overflows int32
    outright.  Clipping first bounds the work by the image size.

    Returns (u0, v0, u1, v1) clipped, or None if the segment misses entirely.
    """
    xmin, ymin = -margin, -margin
    xmax, ymax = W - 1 + margin, H - 1 + margin

    def code(x, y):
        c = 0
        if x < xmin: c |= 1
        elif x > xmax: c |= 2
        if y < ymin: c |= 4
        elif y > ymax: c |= 8
        return c

    x0, y0, x1, y1 = float(u0), float(v0), float(u1), float(v1)
    if not all(np.isfinite(v) for v in (x0, y0, x1, y1)):
        return None

    c0, c1 = code(x0, y0), code(x1, y1)
    for _ in range(8):                     # converges in <= 4; guard anyway
        if not (c0 | c1):
            return int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
        if c0 & c1:
            return None                    # wholly outside
        c = c0 or c1
        if c & 8:      x, y = x0 + (x1-x0)*(ymax-y0)/(y1-y0), ymax
        elif c & 4:    x, y = x0 + (x1-x0)*(ymin-y0)/(y1-y0), ymin
        elif c & 2:    x, y = xmax, y0 + (y1-y0)*(xmax-x0)/(x1-x0)
        else:          x, y = xmin, y0 + (y1-y0)*(xmin-x0)/(x1-x0)
        if c == c0:
            x0, y0, c0 = x, y, code(x, y)
        else:
            x1, y1, c1 = x, y, code(x, y)
    return None


def _draw_line(canvas, u0, v0, u1, v1, color, W, H):
    """Bresenham line on canvas, clipped to the image bounds first."""
    seg = _clip_segment(u0, v0, u1, v1, W, H)
    if seg is None:
        return
    u0, v0, u1, v1 = seg

    du, dv = abs(u1 - u0), abs(v1 - v0)
    su = 1 if u1 > u0 else -1
    sv = 1 if v1 > v0 else -1
    u, v = u0, v0
    steps = max(du, dv) + 1
    if steps < 1:
        return
    steps = min(int(steps), 4 * (W + H))   # hard bound on rasterisation cost
    for _ in range(steps):
        if 0 <= v < H and 0 <= u < W:
            canvas[v, u] = color
        if du > dv:
            u += su
            dv += abs(v1 - v0)
            if 2 * dv >= du:
                v += sv
                dv -= du
        else:
            v += sv
            du += abs(u1 - u0)
            if 2 * du >= dv:
                u += su
                du -= dv


# ── Pose sampling utilities ───────────────────────────────────────────────────

def random_rotation(rng=None) -> np.ndarray:
    """
    Sample a uniformly random rotation matrix (Haar measure on SO(3)).
    Uses the Shoemake / quaternion method.
    """
    if rng is None:
        rng = np.random.default_rng()
    u1, u2, u3 = rng.uniform(size=3)
    q = np.array([
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1)     * np.sin(2 * np.pi * u3),
        np.sqrt(u1)     * np.cos(2 * np.pi * u3),
    ])
    # Convert to DCM (q0, q1, q2, q3) scalar-first
    q0, q1, q2, q3 = q
    R = np.array([
        [1-2*(q2**2+q3**2), 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)],
        [2*(q1*q2+q0*q3), 1-2*(q1**2+q3**2), 2*(q2*q3-q0*q1)],
        [2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), 1-2*(q1**2+q2**2)],
    ])
    return R


def look_at_rotation(eye: np.ndarray,
                     target: np.ndarray = None,
                     up: np.ndarray = None) -> np.ndarray:
    """
    Compute R_cw so that the camera at `eye` looks at `target`.

    Parameters
    ----------
    eye    : (3,) camera position in world frame
    target : (3,) point to look at (default origin)
    up     : (3,) world up vector (default [0,0,1])

    Returns
    -------
    R_cw : (3,3)  rotation matrix world → camera
    t_cw : (3,)   t_cw = -R_cw @ eye  (so P_cam = R_cw P_world + t_cw)
    """
    eye    = np.asarray(eye,    dtype=float)
    target = np.asarray(target if target is not None else [0., 0., 0.], float)
    up     = np.asarray(up     if up     is not None else [0., 0., 1.], float)

    fwd = target - eye
    fwd = fwd / (np.linalg.norm(fwd) + 1e-30)

    right = np.cross(fwd, up)
    if np.linalg.norm(right) < 1e-10:
        up = np.array([0., 1., 0.])
        right = np.cross(fwd, up)
    right = right / np.linalg.norm(right)

    up_cam = np.cross(right, fwd)

    # Camera frame: x=right, y=-up_cam (down), z=fwd (into scene)
    R_wc = np.stack([right, -up_cam, fwd], axis=1)   # world → camera cols
    R_cw = R_wc.T
    t_cw = -R_cw @ eye

    return R_cw, t_cw

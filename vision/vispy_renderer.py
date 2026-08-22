"""
vispy_renderer.py
=================
GPU-accelerated renderer for debris visualisation using Vispy / OpenGL.

Two modes
---------
  Offscreen (headless) — default for pipeline use
      rend = VispyRenderer(camera)
      img  = rend.render(model, R_body, t_body)   # (H, W, 3) uint8
      Drop-in replacement for DebrisRenderer; requires no display.

  Interactive window — orbit the debris with mouse / trackpad
      rend = VispyRenderer(camera, mode='interactive')
      rend.show(model, R_body, t_body)            # blocks until window closed

Scene elements
--------------
  * Space background — deep-navy solid colour + 2 000 procedural stars.
  * Earth limb       — translucent blue arc below / behind the target.
  * Debris mesh      — quad faces triangulated, Phong-shaded per triangle.
  * Keypoint markers — green dots at the 15 model keypoints.

Coordinate systems
------------------
  Body frame (model vertices)  → camera frame: P_cam = R_body @ P_body + t_body
  Camera frame convention      : x right, y down, z forward  (pinhole standard)
  OpenGL / Vispy world frame   : x right, y up,   z out       (right-hand)
  Conversion matrix C          : diag(1, -1, -1)  (flip y and z)

Dependencies
------------
  pip install "vispy>=0.14" pyopengl   (PyOpenGL for headless offscreen)
"""

from __future__ import annotations

import os
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional

# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class VispyConfig:
    """All tunable parameters for the Vispy renderer."""

    # Output resolution (offscreen mode)
    width:  int = 1280
    height: int = 720

    # Phong lighting
    ambient:          float = 0.18
    diffuse:          float = 0.82
    specular:         float = 0.40
    shininess:        float = 48.0
    light_dir_cam:    np.ndarray = field(
        default_factory=lambda: np.array([-0.3, -0.5, 1.0], dtype=float))

    # Debris body
    body_color:  Tuple[float, ...] = (0.55, 0.65, 0.72)  # metallic grey-blue
    edge_color:  Tuple[float, ...] = (1.00, 0.90, 0.30)  # gold highlight

    # Keypoints
    show_keypoints: bool           = True
    kpt_color:      Tuple[float, ...] = (0.0, 1.0, 0.25)
    kpt_size:       float          = 10.0

    # Star field
    n_stars:    int   = 2000
    star_dist:  float = 80_000.0    # [m] radius of star sphere
    star_size:  float = 2.0

    # Earth limb
    earth_limb:   bool             = True
    earth_color:  Tuple[float, ...] = (0.08, 0.35, 0.90, 0.55)
    earth_dist:   float            = 6_000.0   # [m] displacement below scene
    earth_radius: float            = 3_000.0   # [m] apparent arc radius

    # Background colour (RGB 0-1)
    bg_color: Tuple[float, ...] = (0.020, 0.020, 0.040)


# ─── Internal helpers ──────────────────────────────────────────────────────────

# Camera → OpenGL world: flip y and z axes
_C2GL = np.diag([1.0, -1.0, -1.0]).astype(np.float32)


def _triangulate(faces: np.ndarray) -> np.ndarray:
    """Convert (M, 3-or-4) face indices to (N, 3) int32 triangles."""
    f = np.asarray(faces, dtype=np.int32)
    if f.shape[1] == 3:
        return f
    # Quad → two triangles: (0,1,2) and (0,2,3)
    tri = np.empty((len(f) * 2, 3), dtype=np.int32)
    tri[0::2] = f[:, [0, 1, 2]]
    tri[1::2] = f[:, [0, 2, 3]]
    return tri


def _phong_face_colors(verts_cam: np.ndarray,
                       tris:      np.ndarray,
                       cfg:       VispyConfig) -> np.ndarray:
    """
    Compute Phong-shaded RGBA colour for every triangle.

    Parameters
    ----------
    verts_cam : (V, 3) float  — vertices in camera frame
    tris      : (M, 3) int32  — triangle face indices
    cfg       : VispyConfig

    Returns
    -------
    rgba : (M, 4) float32  — per-face RGBA in [0, 1]
    """
    v0, v1, v2 = verts_cam[tris[:, 0]], verts_cam[tris[:, 1]], verts_cam[tris[:, 2]]

    # Face normals (camera frame)
    raw = np.cross(v1 - v0, v2 - v0)
    nrm = np.linalg.norm(raw, axis=1, keepdims=True)
    N   = raw / (nrm + 1e-30)                          # (M, 3) unit normals

    # Normalised light direction (camera frame)
    L = np.asarray(cfg.light_dir_cam, dtype=float)
    L = L / (np.linalg.norm(L) + 1e-30)

    # View direction per face (face-centre → camera at origin)
    ctr   = (v0 + v1 + v2) / 3.0
    V_dir = -ctr / (np.linalg.norm(ctr, axis=1, keepdims=True) + 1e-30)

    # Back-face: normal pointing away from camera
    facing = np.einsum('ij,ij->i', N, V_dir) > 0

    # Diffuse
    NdotL = np.clip(np.dot(N, L), 0.0, 1.0)

    # Blinn-Phong specular
    H_vec = (V_dir + L)
    H_vec /= (np.linalg.norm(H_vec, axis=1, keepdims=True) + 1e-30)
    NdotH = np.clip(np.einsum('ij,ij->i', N, H_vec), 0.0, 1.0)
    spec  = cfg.specular * NdotH ** cfg.shininess

    base  = np.asarray(cfg.body_color, dtype=np.float32)
    bright = (cfg.ambient + cfg.diffuse * NdotL + spec)[:, None]
    rgba  = np.ones((len(tris), 4), dtype=np.float32)
    rgba[:, :3] = np.clip(bright * base, 0.0, 1.0)
    rgba[~facing, :3] *= 0.06   # darken back faces

    return rgba


def _star_positions(cfg: VispyConfig, seed: int = 7) -> np.ndarray:
    """(N, 3) float32 star positions in OpenGL world frame."""
    rng  = np.random.default_rng(seed)
    phi  = rng.uniform(0, 2 * np.pi, cfg.n_stars)
    cosT = rng.uniform(-1.0, 1.0, cfg.n_stars)
    sinT = np.sqrt(np.maximum(0.0, 1.0 - cosT ** 2))
    r    = cfg.star_dist
    return (r * np.column_stack([sinT * np.cos(phi),
                                  sinT * np.sin(phi),
                                  cosT])).astype(np.float32)


def _earth_limb_arc(t_gl: np.ndarray, cfg: VispyConfig) -> np.ndarray:
    """
    Blue arc in OpenGL world frame representing the Earth limb.
    Centred below-and-behind the target; concentric multi-ring for a glow.
    Returns list of (K, 3) float32 arrays (one per ring).
    """
    # Earth centre in GL world: below the scene (-y), behind target (+z)
    ec = np.array([
        t_gl[0] * 0.05,               # slight x offset
        -cfg.earth_dist,               # below in GL frame
        t_gl[2] + cfg.earth_dist * 0.5,  # behind target
    ], dtype=np.float32)

    rings = []
    for scale in [1.0, 1.15, 1.30]:   # concentric for soft glow
        theta = np.linspace(-np.pi * 0.60, np.pi * 0.60, 140)
        pts   = ec + (cfg.earth_radius * scale) * np.column_stack([
            np.cos(theta), np.sin(theta), np.zeros_like(theta)
        ])
        rings.append(pts.astype(np.float32))
    return rings


# ─── Backend management ───────────────────────────────────────────────────────

_VISPY_BACKEND: Optional[str] = None


def _init_vispy(mode: str) -> None:
    """
    Call vispy.use() once before any scene import.
    Must be done before the first SceneCanvas is created.

    Note: vispy >= 0.14 removed 'offscreen' as a backend name.
    For headless rendering the strategy is platform-dependent:

    * Linux  — try osmesa (pure software GL, no display) then egl.
    * macOS  — osmesa/.so is not available; skip straight to auto-detect.
               SceneCanvas(show=False) renders headlessly with any Qt/glfw
               backend without opening a window.

    On any platform, if no explicit headless backend is found, vispy
    auto-detects (pyqt6 / pyqt5 / pyside6 / glfw / …).  The window is
    kept invisible via show=False in SceneCanvas; canvas.render() still
    returns a full RGBA array.
    """
    global _VISPY_BACKEND
    if _VISPY_BACKEND is not None:
        return  # already initialised; can't change mid-session
    import sys
    import vispy
    if mode == 'offscreen':
        # osmesa / egl only exist on Linux — don't even try on macOS.
        if sys.platform != 'darwin':
            for _backend in ('osmesa', 'egl'):
                try:
                    vispy.use(_backend)
                    _VISPY_BACKEND = _backend
                    return
                except Exception:
                    continue
        # Auto-detect: vispy picks the best available backend (Qt, glfw, …).
        # On macOS this is typically pyqt6 / pyside6 / glfw.
        # No vispy.use() call needed — SceneCanvas(show=False) is enough.
    _VISPY_BACKEND = mode


# ─── Renderer ─────────────────────────────────────────────────────────────────

class VispyRenderer:
    """
    GPU-accelerated Vispy renderer for debris visualisation.

    Parameters
    ----------
    camera : PinholeCamera
        Intrinsics used to set FOV and aspect ratio.
    config : VispyConfig, optional
        Rendering parameters.
    mode : str
        'offscreen' (default) — headless, returns numpy array.
        'interactive' — opens a live window with orbit controls.

    Notes
    -----
    * Call ``VispyRenderer(camera, mode='offscreen')`` before the first
      ``VispyRenderer(camera, mode='interactive')`` in the same process
      if you need both — the backend is set on the first call.
    * Offscreen rendering requires PyOpenGL: ``pip install pyopengl``.
    """

    def __init__(self,
                 camera,
                 config: Optional[VispyConfig] = None,
                 mode:   str = 'offscreen'):
        self.cam  = camera
        self.cfg  = config or VispyConfig()
        self.mode = mode

        _init_vispy(mode)

        from vispy import scene as _scene
        self._scene = _scene

        # FOV from camera intrinsics
        fy  = camera.K[1, 1]
        H   = camera.height if hasattr(camera, 'height') else self.cfg.height
        W   = camera.width  if hasattr(camera, 'width')  else self.cfg.width
        self._fov_y = float(np.degrees(2.0 * np.arctan(H / (2.0 * fy))))
        self._W, self._H = W, H

        # Pre-build star positions (constant across frames)
        self._stars_gl = _star_positions(self.cfg)

    # ── Public API ─────────────────────────────────────────────────────────

    def render(self, model, R_body: np.ndarray,
               t_body: np.ndarray) -> np.ndarray:
        """
        Render one frame offscreen.

        Parameters
        ----------
        model  : DebrisModel
        R_body : (3, 3) body-to-camera rotation
        t_body : (3,) body position in camera frame

        Returns
        -------
        img : (H, W, 3) uint8 RGB array
        """
        if self.mode != 'offscreen':
            raise RuntimeError("render() is only for offscreen mode; "
                               "use show() for interactive mode.")

        canvas, view = self._build_canvas(offscreen=True)
        self._populate_scene(view, model, R_body, t_body)

        # Correct perspective camera transform
        t_gl = (_C2GL @ np.asarray(t_body, dtype=np.float32)).flatten()
        dist = float(np.linalg.norm(t_gl))
        view.camera.center = (float(t_gl[0]), float(t_gl[1]), float(t_gl[2]))
        view.camera.distance = dist

        img_raw = canvas.render()   # (H_phys, W_phys, 4) RGBA uint8
        canvas.close()
        img = img_raw[:, :, :3]     # drop alpha → (H_phys, W_phys, 3)

        # On macOS Retina / HiDPI displays the canvas renders at 2× (or higher)
        # device pixel ratio.  Subsample back to the requested logical size.
        H_want, W_want = self.cfg.height, self.cfg.width
        if img.shape[0] != H_want or img.shape[1] != W_want:
            step_h = max(img.shape[0] // H_want, 1)
            step_w = max(img.shape[1] // W_want, 1)
            img = img[::step_h, ::step_w][:H_want, :W_want]

        return img

    def show(self, model, R_body: np.ndarray,
             t_body: np.ndarray,
             title: str = 'Debris Viewer') -> None:
        """
        Open an interactive 3-D window.

        Controls: left-drag orbit, scroll zoom, right-drag pan.
        Close the window to continue.
        """
        if self.mode != 'interactive':
            raise RuntimeError("show() is only for interactive mode; "
                               "create VispyRenderer(..., mode='interactive').")

        from vispy import app as _vapp

        canvas, view = self._build_canvas(offscreen=False, title=title)
        self._populate_scene(view, model, R_body, t_body)

        t_gl = (_C2GL @ np.asarray(t_body, dtype=np.float32)).flatten()
        dist = float(np.linalg.norm(t_gl))
        view.camera.set_range()
        if hasattr(view.camera, 'distance'):
            view.camera.distance = dist * 1.5

        canvas.show()
        _vapp.run()

    # ── Scene construction ─────────────────────────────────────────────────

    def _build_canvas(self, offscreen: bool, title: str = ''):
        sc = self._scene
        W, H = self.cfg.width, self.cfg.height
        bg   = self.cfg.bg_color

        canvas = sc.SceneCanvas(
            size=(W, H),
            bgcolor=bg,
            show=not offscreen,
            title=title,
        )
        view = canvas.central_widget.add_view()

        if offscreen:
            # TurntableCamera is the standard perspective camera in vispy >= 0.14.
            # (PerspectiveCamera was removed as a public name.)
            # azimuth=0, elevation=0 → looks along the scene -z axis from +z,
            # which maps to our camera looking toward the target.
            cam = sc.cameras.TurntableCamera(
                fov=self._fov_y,
                up='+y',
                azimuth=0.0,
                elevation=0.0,
            )
            view.camera = cam
        else:
            # Interactive orbit camera
            cam = sc.cameras.TurntableCamera(
                fov=45.0,
                up='+y',
                azimuth=30.0,
                elevation=20.0,
            )
            view.camera = cam

        return canvas, view

    def _populate_scene(self, view, model, R_body, t_body):
        """Add all visuals (stars, Earth limb, debris mesh, keypoints) to view."""
        sc = self._scene

        R  = np.asarray(R_body, dtype=np.float32)
        t  = np.asarray(t_body, dtype=np.float32).flatten()

        # ── 1. Transform body vertices → camera frame → GL world ───────────
        verts_body = np.asarray(model.vertices, dtype=np.float32)
        verts_cam  = (R @ verts_body.T).T + t          # (V, 3) camera frame
        verts_gl   = (verts_cam @ _C2GL.T)             # (V, 3) GL world frame
        t_gl       = _C2GL @ t

        tris = _triangulate(np.asarray(model.faces))   # (M, 3) int32
        rgba = _phong_face_colors(verts_cam, tris, self.cfg)

        # ── 2. Stars ────────────────────────────────────────────────────────
        sc.visuals.Markers(
            pos=self._stars_gl,
            size=self.cfg.star_size,
            face_color=(1.0, 1.0, 1.0, 0.85),
            edge_width=0,
            parent=view.scene,
        )

        # ── 3. Earth limb ───────────────────────────────────────────────────
        if self.cfg.earth_limb:
            rings = _earth_limb_arc(t_gl, self.cfg)
            alphas = [0.55, 0.35, 0.18]
            er, eg, eb = self.cfg.earth_color[:3]
            for ring, alpha in zip(rings, alphas):
                sc.visuals.Line(
                    pos=ring,
                    color=(er, eg, eb, alpha),
                    width=2.5,
                    parent=view.scene,
                    connect='strip',
                )

        # ── 4. Debris mesh ──────────────────────────────────────────────────
        mesh = sc.visuals.Mesh(
            vertices=verts_gl,
            faces=tris,
            face_colors=rgba,
            parent=view.scene,
        )

        # ── 5. Keypoints ────────────────────────────────────────────────────
        if self.cfg.show_keypoints:
            kp3d  = np.asarray(model.keypoint_array, dtype=np.float32)
            kp_cam = (R @ kp3d.T).T + t
            kp_gl  = kp_cam @ _C2GL.T
            sc.visuals.Markers(
                pos=kp_gl,
                size=self.cfg.kpt_size,
                face_color=(*self.cfg.kpt_color, 1.0),
                edge_width=0,
                parent=view.scene,
            )


# ── Convenience factory functions ─────────────────────────────────────────────

def vispy_offscreen_renderer(camera=None,
                              config: Optional[VispyConfig] = None
                              ) -> VispyRenderer:
    """
    Return a VispyRenderer in offscreen (headless) mode.

    Parameters
    ----------
    camera : PinholeCamera, optional
        Defaults to the standard rendezvous camera (800 mm equiv. lens).
    config : VispyConfig, optional
    """
    if camera is None:
        from vision.camera import rendezvous_camera
        camera = rendezvous_camera()
    return VispyRenderer(camera, config=config, mode='offscreen')


def vispy_interactive_renderer(camera=None,
                                config: Optional[VispyConfig] = None
                                ) -> VispyRenderer:
    """
    Return a VispyRenderer in interactive mode (opens a window).

    Parameters
    ----------
    camera : PinholeCamera, optional
    config : VispyConfig, optional
    """
    if camera is None:
        from vision.camera import rendezvous_camera
        camera = rendezvous_camera()
    return VispyRenderer(camera, config=config, mode='interactive')

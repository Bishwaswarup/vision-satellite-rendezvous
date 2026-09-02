"""
camera.py
=========
Pinhole camera model for synthetic vision generation.

Convention
----------
  Camera frame  : x right, y down, z into scene (OpenCV convention)
  Image frame   : (u, v) pixels, origin top-left
  World / body  : arbitrary right-hand frame

Projection pipeline
-------------------
  P_cam  = R_cw @ P_world + t_cw          (world → camera)
  p_norm = P_cam[:2] / P_cam[2]           (perspective divide)
  p_dist = distort(p_norm)                (optional radial distortion)
  p_px   = K[:2,:2] @ p_dist + [cx, cy]  (→ pixel)

Reference
---------
  Hartley & Zisserman, "Multiple View Geometry", Cambridge, 2003.
  OpenCV camera model: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
"""

import numpy as np


class PinholeCamera:
    """
    Pinhole camera with optional Brown-Conrady radial distortion.

    Parameters
    ----------
    fx, fy : float   Focal lengths [pixels]
    cx, cy : float   Principal point [pixels]
    width, height : int  Image resolution [pixels]
    k1, k2 : float   Radial distortion coefficients (default 0 = no distortion)
    p1, p2 : float   Tangential distortion coefficients (default 0)
    """

    def __init__(self,
                 fx: float, fy: float,
                 cx: float, cy: float,
                 width: int, height: int,
                 k1: float = 0.0, k2: float = 0.0,
                 p1: float = 0.0, p2: float = 0.0):
        self.fx, self.fy = float(fx), float(fy)
        self.cx, self.cy = float(cx), float(cy)
        self.width  = int(width)
        self.height = int(height)
        self.k1, self.k2 = float(k1), float(k2)
        self.p1, self.p2 = float(p1), float(p2)

        # Intrinsics matrix  K  (3×3)
        self.K = np.array([
            [fx,  0., cx],
            [ 0., fy, cy],
            [ 0.,  0., 1.],
        ], dtype=float)

        self.K_inv = np.linalg.inv(self.K)

    # ── Distortion ────────────────────────────────────────────────────────────
    def distort(self, p_norm: np.ndarray) -> np.ndarray:
        """
        Apply Brown-Conrady radial + tangential distortion.

        Parameters
        ----------
        p_norm : array (..., 2)   Normalised image coordinates (before K)

        Returns
        -------
        p_dist : array (..., 2)   Distorted normalised coordinates
        """
        p = np.asarray(p_norm, dtype=float)
        x, y = p[..., 0], p[..., 1]
        r2 = x**2 + y**2
        r4 = r2**2
        radial = 1.0 + self.k1 * r2 + self.k2 * r4
        xd = x * radial + 2*self.p1*x*y + self.p2*(r2 + 2*x**2)
        yd = y * radial + self.p1*(r2 + 2*y**2) + 2*self.p2*x*y
        return np.stack([xd, yd], axis=-1)

    # ── Core projection ───────────────────────────────────────────────────────
    def project(self,
                P_cam: np.ndarray,
                apply_distortion: bool = True,
                clip: bool = True) -> tuple:
        """
        Project 3D points in the camera frame to pixel coordinates.

        Parameters
        ----------
        P_cam  : array (N, 3)   3D points in camera frame [m]
        apply_distortion : bool
        clip   : bool           Return mask of points in front of camera & in frame

        Returns
        -------
        uv     : array (N, 2)   Pixel coordinates (u=col, v=row)
        depth  : array (N,)     Depth values (z in camera frame) [m]
        valid  : array (N,) bool  True if point is visible
        """
        P = np.atleast_2d(np.asarray(P_cam, dtype=float))
        z = P[:, 2]

        # Behind camera
        in_front = z > 0.0

        # Perspective divide
        p_norm = np.zeros((len(P), 2))
        mask   = in_front
        p_norm[mask, 0] = P[mask, 0] / z[mask]
        p_norm[mask, 1] = P[mask, 1] / z[mask]

        # Distortion
        if apply_distortion and (self.k1 != 0 or self.k2 != 0 or
                                  self.p1 != 0 or self.p2 != 0):
            p_norm[mask] = self.distort(p_norm[mask])

        # Apply K
        u = self.fx * p_norm[:, 0] + self.cx
        v = self.fy * p_norm[:, 1] + self.cy
        uv = np.stack([u, v], axis=-1)

        if clip:
            in_frame = (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
            valid = in_front & in_frame
        else:
            valid = in_front

        return uv, z, valid

    def project_world(self,
                      P_world: np.ndarray,
                      R_cw: np.ndarray,
                      t_cw: np.ndarray,
                      apply_distortion: bool = True,
                      clip: bool = True) -> tuple:
        """
        Project 3D world-frame points to pixel coordinates.

        Parameters
        ----------
        P_world : array (N, 3)  Points in world / body frame [m]
        R_cw    : array (3, 3)  Rotation: world → camera
        t_cw    : array (3,)    Translation: camera origin in world frame,
                                expressed as t_cw = -R_cw @ t_world_cam

        Returns
        -------
        uv, depth, valid  (same as project())
        """
        P = np.atleast_2d(np.asarray(P_world, dtype=float))
        R = np.asarray(R_cw, dtype=float)
        t = np.asarray(t_cw, dtype=float).ravel()
        P_cam = (R @ P.T).T + t          # (N, 3)
        return self.project(P_cam, apply_distortion=apply_distortion, clip=clip)

    # ── Back-projection ───────────────────────────────────────────────────────
    def backproject(self, uv: np.ndarray, depth: np.ndarray,
                    undistort: bool = True) -> np.ndarray:
        """
        Back-project pixel coordinates + depth to 3D camera-frame points.

        Parameters
        ----------
        uv        : array (N, 2)   Pixel coordinates (u, v)
        depth     : array (N,)     Depth [m]
        undistort : bool  invert the lens distortion first (default True).
                    project() applies distortion, so without this the round
                    trip is not the identity — up to 29 px of error at the
                    corners of `wide_angle_camera`.

        Returns
        -------
        P_cam : array (N, 3)
        """
        uv = np.atleast_2d(np.asarray(uv, dtype=float))
        d  = np.asarray(depth, dtype=float).ravel()
        if undistort and self.has_distortion:
            uv = self.undistort(uv)
        x_norm = (uv[:, 0] - self.cx) / self.fx
        y_norm = (uv[:, 1] - self.cy) / self.fy
        return np.stack([x_norm * d, y_norm * d, d], axis=-1)

    # ── Undistortion ──────────────────────────────────────────────────────────
    @property
    def has_distortion(self) -> bool:
        """True if any Brown-Conrady coefficient is non-zero."""
        return any(abs(c) > 0.0 for c in (self.k1, self.k2, self.p1, self.p2))

    def undistort(self, uv: np.ndarray, iters: int = 12) -> np.ndarray:
        """
        Invert the Brown-Conrady distortion: observed pixels -> ideal pixels.

        `distort()` was applied on projection but never inverted anywhere, and
        no pose solver took distortion coefficients, so anything rendered with
        a distorting preset was solved with an unmodelled systematic error of
        up to 29 px at the corners of `wide_angle_camera`.

        The forward model has no closed-form inverse, so this is the standard
        fixed-point iteration on the normalised coordinates: start from the
        distorted point and repeatedly subtract the distortion evaluated at
        the current estimate.  It converges in a few iterations for the
        moderate coefficients used here.
        """
        uv = np.atleast_2d(np.asarray(uv, dtype=float))
        if not self.has_distortion:
            return uv.copy()

        xd = (uv[:, 0] - self.cx) / self.fx
        yd = (uv[:, 1] - self.cy) / self.fy
        x, y = xd.copy(), yd.copy()

        for _ in range(iters):
            r2 = x*x + y*y
            radial = 1.0 + self.k1 * r2 + self.k2 * r2 * r2
            dx = 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x)
            dy = self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y
            x = (xd - dx) / radial
            y = (yd - dy) / radial

        return np.stack([x * self.fx + self.cx, y * self.fy + self.cy], axis=-1)

    # ── Helpers ───────────────────────────────────────────────────────────────
    @property
    def fov_x_deg(self) -> float:
        """Horizontal field-of-view [degrees]."""
        return float(np.degrees(2 * np.arctan(self.width  / (2 * self.fx))))

    @property
    def fov_y_deg(self) -> float:
        """Vertical field-of-view [degrees]."""
        return float(np.degrees(2 * np.arctan(self.height / (2 * self.fy))))

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height

    def __repr__(self):
        return (f"PinholeCamera(fx={self.fx:.1f}, fy={self.fy:.1f}, "
                f"cx={self.cx:.1f}, cy={self.cy:.1f}, "
                f"{self.width}×{self.height}px, "
                f"FoV {self.fov_x_deg:.1f}°×{self.fov_y_deg:.1f}°)")


# ── Preset cameras ────────────────────────────────────────────────────────────

def rendezvous_camera() -> PinholeCamera:
    """
    Representative near-range inspection camera (chaser spacecraft).

    Specs modelled loosely on ESA LIRIS / ClearSpace inspection imager:
      - Resolution : 1024 × 1024 pixels
      - Focal length: ~800 px  (~f = 12 mm on 1/2" sensor → ~800 px at 15 µm pitch)
      - FoV        : 65.2 deg x 65.2 deg = 2 atan(W / 2 fx), W=1024, fx=800
      - No distortion (corrected optics)
    """
    return PinholeCamera(
        fx=800.0, fy=800.0,
        cx=512.0, cy=512.0,
        width=1024, height=1024,
    )


def wide_angle_camera() -> PinholeCamera:
    """
    Wide-angle context camera for far-range acquisition.
      - Resolution : 640 × 480
      - FoV        : ~90° × 70°
      - Mild radial distortion
    """
    return PinholeCamera(
        fx=320.0, fy=320.0,
        cx=320.0, cy=240.0,
        width=640, height=480,
        k1=-0.05, k2=0.002,
    )

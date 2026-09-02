"""
dataset.py
==========
Synthetic pose dataset generator for debris pose estimation.

Generates a dataset of (image, ground-truth pose) pairs by:
  1. Sampling random camera positions on a hemisphere around the debris body
  2. Orienting the camera to look at the body centre (with random roll)
  3. Sampling a random body attitude from the tumbling model (Phase 2)
  4. Rendering the image with DebrisRenderer
  5. Saving metadata: quaternion, translation, 2D keypoints, bounding box

Dataset structure (on disk)
----------------------------
  dataset_dir/
      images/
          frame_000000.png
          frame_000001.png
          ...
      annotations.json   — list of per-frame dicts (see PoseAnnotation)
      camera.json        — camera intrinsics
      meta.json          — generation parameters

Usage
-----
  from vision.dataset import DatasetGenerator, DatasetConfig
  from vision import ariane_model, rendezvous_camera

  cfg = DatasetConfig(n_frames=500, range_min=15.0, range_max=40.0)
  gen = DatasetGenerator(rendezvous_camera(), ariane_model(), cfg)
  gen.generate('outputs/dataset_ariane')
"""

import os
import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict

from .camera   import PinholeCamera
from .body_model import DebrisModel
from .renderer import DebrisRenderer, RendererConfig, random_rotation, look_at_rotation


@dataclass
class DatasetConfig:
    """Configuration for synthetic dataset generation."""
    n_frames      : int   = 200        # Total number of frames
    range_min     : float = 10.0       # Min camera range [m]
    range_max     : float = 50.0       # Max camera range [m]
    elevation_min : float = -60.0      # Min elevation angle [deg]
    elevation_max : float =  60.0      # Max elevation angle [deg]
    roll_max_deg  : float = 180.0      # Max camera roll angle [deg]
    random_body_attitude : bool = True # Randomise body attitude (SO3 uniform)
    noise_sigma   : float = 1.5        # Image noise std [DN]
    star_density  : float = 0.0003     # Stars per pixel
    image_format  : str  = 'PNG'       # 'PNG' or 'JPEG'
    seed          : int  = 42          # Global RNG seed


@dataclass
class PoseAnnotation:
    """Ground-truth annotation for one rendered frame."""
    frame_id      : int
    filename      : str
    # Pose of the MODEL KEYPOINTS as shipped in `model.keypoint_array`
    # (unrotated body frame), i.e. P_cam = R_cw @ P_body + t_cw.  This is the
    # pose a PnP solver fed `keypoints_2d` recovers.
    R_cw          : List[List[float]]   # 3×3 rotation (nested list for JSON)
    t_cw          : List[float]         # [tx, ty, tz] [m]
    q_body        : List[float]         # body attitude quaternion [q0,q1,q2,q3]
    # 2D observations
    keypoints_2d  : Dict[str, List[float]]  # name → [u, v]
    bbox_xyxy     : Optional[List[float]]   # [u_min, v_min, u_max, v_max]
    # Scalars
    range_m       : float
    n_visible_kpts: int
    n_visible_faces: int
    # Diagnostic: the camera pose alone, before the body attitude is folded in.
    R_camera_world: Optional[List[List[float]]] = None


class DatasetGenerator:
    """
    Generates a synthetic labelled dataset for debris pose estimation.

    Parameters
    ----------
    camera  : PinholeCamera
    model   : DebrisModel
    config  : DatasetConfig
    """

    def __init__(self,
                 camera : PinholeCamera,
                 model  : DebrisModel,
                 config : DatasetConfig = None):
        self.cam    = camera
        self.model  = model
        self.cfg    = config or DatasetConfig()
        self.rend   = DebrisRenderer(
            camera,
            RendererConfig(
                noise_sigma=self.cfg.noise_sigma,
                star_density=self.cfg.star_density,
            )
        )

    def _sample_camera_pose(self, rng) -> tuple:
        """
        Sample a random camera position on a hemisphere and compute look-at pose.

        Returns (R_cw, t_cw, range_m)
        """
        # Random range
        r = rng.uniform(self.cfg.range_min, self.cfg.range_max)

        # Random azimuth + elevation
        az  = rng.uniform(0.0, 2 * np.pi)
        el  = np.radians(rng.uniform(self.cfg.elevation_min, self.cfg.elevation_max))

        # Camera position in world frame (body at origin)
        x = r * np.cos(el) * np.cos(az)
        y = r * np.cos(el) * np.sin(az)
        z = r * np.sin(el)
        eye = np.array([x, y, z])

        R_cw, t_cw = look_at_rotation(eye, target=np.zeros(3),
                                       up=np.array([0., 0., 1.]))

        # Random roll around camera z-axis
        roll_max = np.radians(self.cfg.roll_max_deg)
        roll = rng.uniform(-roll_max, roll_max)
        Rz = np.array([
            [ np.cos(roll), np.sin(roll), 0.],
            [-np.sin(roll), np.cos(roll), 0.],
            [ 0.,           0.,           1.],
        ])
        R_cw = Rz @ R_cw
        t_cw = Rz @ t_cw

        return R_cw, t_cw, r

    def generate_frame(self, frame_id: int, rng) -> tuple:
        """
        Generate a single (image, annotation) pair.

        Returns
        -------
        image      : np.ndarray uint8 (H, W, 3)
        annotation : PoseAnnotation
        """
        # Camera pose
        R_cw, t_cw, range_m = self._sample_camera_pose(rng)

        # Body attitude (or identity)
        if self.cfg.random_body_attitude:
            R_body = random_rotation(rng)
            q_body = _dcm_to_q(R_body).tolist()
        else:
            R_body = np.eye(3)
            q_body = [1., 0., 0., 0.]

        # Transform model by body attitude, then render
        model_rotated = self.model.transform(R_body)
        img, meta = self.rend.render(
            model_rotated, R_cw, t_cw,
            seed=int(rng.integers(0, 2**31))
        )

        filename = f'frame_{frame_id:06d}.png'
        # POSE LABEL.  `keypoints_2d` is keyed by name, so the natural consumer
        # pairs it with `model.keypoint_array` — the UNROTATED body frame.  The
        # pose of those points is therefore R_cw @ R_body, not R_cw: storing
        # R_cw alone leaves the rotation label wrong by the body attitude
        # (measured 42-150 deg on the shipped datasets, while the translation
        # is correct).  R_cw and q_body are both kept as diagnostics.
        R_cb = R_cw @ R_body
        ann = PoseAnnotation(
            frame_id       = frame_id,
            filename       = filename,
            R_cw           = R_cb.tolist(),
            t_cw           = t_cw.tolist(),
            R_camera_world = R_cw.tolist(),
            q_body         = q_body,
            keypoints_2d   = {k: list(v) for k, v in meta['proj_keypoints'].items()},
            bbox_xyxy      = list(meta['bbox']) if meta['bbox'] else None,
            range_m        = float(range_m),
            n_visible_kpts = len(meta['keypoints_2d']),
            n_visible_faces= meta['n_visible_faces'],
        )
        return img, ann

    def generate(self, output_dir: str,
                 verbose: bool = True) -> List[PoseAnnotation]:
        """
        Generate the full dataset and write to output_dir.

        Parameters
        ----------
        output_dir : str   Directory (created if not exists)
        verbose    : bool  Print progress

        Returns
        -------
        annotations : list of PoseAnnotation
        """
        os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
        rng = np.random.default_rng(self.cfg.seed)

        annotations = []
        for i in range(self.cfg.n_frames):
            img, ann = self.generate_frame(i, rng)

            # Save image (PNG via matplotlib to avoid OpenCV dep)
            img_path = os.path.join(output_dir, 'images', ann.filename)
            _save_png(img, img_path)

            annotations.append(ann)
            if verbose and (i % max(1, self.cfg.n_frames // 10) == 0):
                print(f"  [{i+1:4d}/{self.cfg.n_frames}] {ann.filename}  "
                      f"range={ann.range_m:.1f}m  "
                      f"kpts={ann.n_visible_kpts}  "
                      f"faces={ann.n_visible_faces}")

        # Save annotations JSON
        ann_list = [asdict(a) for a in annotations]
        with open(os.path.join(output_dir, 'annotations.json'), 'w') as f:
            json.dump(ann_list, f, indent=2)

        # Save camera JSON
        cam_dict = {
            'fx': self.cam.fx, 'fy': self.cam.fy,
            'cx': self.cam.cx, 'cy': self.cam.cy,
            'width': self.cam.width, 'height': self.cam.height,
            'k1': self.cam.k1, 'k2': self.cam.k2,
            'p1': self.cam.p1, 'p2': self.cam.p2,
        }
        with open(os.path.join(output_dir, 'camera.json'), 'w') as f:
            json.dump(cam_dict, f, indent=2)

        # Save generation meta
        meta = {
            'model_name'   : self.model.name,
            'n_frames'     : self.cfg.n_frames,
            'range_min'    : self.cfg.range_min,
            'range_max'    : self.cfg.range_max,
            'elevation_min': self.cfg.elevation_min,
            'elevation_max': self.cfg.elevation_max,
            'seed'         : self.cfg.seed,
            'n_keypoints'  : len(self.model.keypoints),
            'keypoint_names': self.model.keypoint_names,
        }
        with open(os.path.join(output_dir, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)

        if verbose:
            print(f"Dataset saved to {output_dir}/  "
                  f"({self.cfg.n_frames} frames)")
        return annotations


# ── Private helpers ───────────────────────────────────────────────────────────

def _dcm_to_q(R: np.ndarray) -> np.ndarray:
    """DCM to quaternion (scalar-first) via Shepperd's method."""
    trace = np.trace(R)
    if trace > 0:
        s  = 0.5 / np.sqrt(trace + 1.0)
        q0 = 0.25 / s
        q1 = (R[2,1] - R[1,2]) * s
        q2 = (R[0,2] - R[2,0]) * s
        q3 = (R[1,0] - R[0,1]) * s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s  = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        q0 = (R[2,1] - R[1,2]) / s
        q1 = 0.25 * s
        q2 = (R[0,1] + R[1,0]) / s
        q3 = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s  = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        q0 = (R[0,2] - R[2,0]) / s
        q1 = (R[0,1] + R[1,0]) / s
        q2 = 0.25 * s
        q3 = (R[1,2] + R[2,1]) / s
    else:
        s  = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        q0 = (R[1,0] - R[0,1]) / s
        q1 = (R[0,2] + R[2,0]) / s
        q2 = (R[1,2] + R[2,1]) / s
        q3 = 0.25 * s
    q = np.array([q0, q1, q2, q3])
    return q / np.linalg.norm(q)


def _save_png(img: np.ndarray, path: str):
    """Save uint8 RGB array as PNG without OpenCV."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(img.shape[1]/100, img.shape[0]/100), dpi=100)
    plt.imshow(img)
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

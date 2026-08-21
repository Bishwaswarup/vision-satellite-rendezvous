"""
test_phase3.py
==============
Unit tests for Phase 3 — Synthetic Vision Pipeline.

Tests:
    1.  Projection of origin → principal point when body at camera axis
    2.  Point behind camera → invalid (not projected)
    3.  Projection + backprojection round-trip (no distortion)
    4.  Backface culling: forward-facing face is kept, backward is culled
    5.  Ariane model: vertex count, edge count, keypoint count
    6.  CubeSat model: 8 corners present, face normals unit length
    7.  render() returns correct image shape and dtype
    8.  Rendered image is non-trivial (not all background)
    9.  Keypoints within image bounds for frontal view
    10. Bounding box is valid (min < max) for visible body
    11. random_rotation() produces a valid SO(3) matrix
    12. look_at_rotation: camera z-axis points toward target
    13. Depth map: closest point has smallest depth value
    14. Dataset generator produces correct number of annotations
    15. Annotation keypoints are within image bounds

Run:
    cd /home/user/phase1
    python -m pytest tests/test_phase3.py -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from vision import (
    PinholeCamera, rendezvous_camera,
    DebrisModel, ariane_model, cubesat_3u_model,
    RendererConfig, DebrisRenderer,
    random_rotation, look_at_rotation,
    DatasetConfig, DatasetGenerator,
)

# ── Shared fixtures ────────────────────────────────────────────────────────────
CAM   = rendezvous_camera()      # 1024×1024, fx=fy=800, cx=cy=512
ARIANE = ariane_model()
CUBE   = cubesat_3u_model()

# Standard frontal view: camera 30 m along +z, looking at origin
EYE    = np.array([0., 0., 30.])
R_CW, T_CW = look_at_rotation(EYE, target=np.zeros(3))


# ── Test 1: Origin projects to principal point ────────────────────────────────
def test_project_origin_to_principal_point():
    P_cam = np.array([[0., 0., 10.]])     # on camera axis
    uv, depth, valid = CAM.project(P_cam)
    np.testing.assert_allclose(uv[0], [CAM.cx, CAM.cy], atol=1e-10)
    assert valid[0]
    assert abs(depth[0] - 10.0) < 1e-10


# ── Test 2: Behind-camera point is invalid ────────────────────────────────────
def test_behind_camera_invalid():
    P_cam = np.array([[0., 0., -5.]])    # negative z → behind camera
    uv, depth, valid = CAM.project(P_cam)
    assert not valid[0], "Point behind camera must be marked invalid"


# ── Test 3: Project + backproject round-trip ─────────────────────────────────
def test_project_backproject_roundtrip():
    rng = np.random.default_rng(0)
    # Random 3D points in front of camera
    P = rng.uniform(-2, 2, (50, 3))
    P[:, 2] = rng.uniform(5, 30, 50)    # positive z
    uv, depth, valid = CAM.project(P, apply_distortion=False)
    P_back = CAM.backproject(uv[valid], depth[valid])
    np.testing.assert_allclose(P_back, P[valid], atol=1e-8,
        err_msg="Project → backproject must recover original 3D points")


# ── Test 4: Backface culling ──────────────────────────────────────────────────
def test_backface_culling():
    """
    Face with normal pointing toward camera (into -z of camera frame)
    should be visible; face with normal pointing away should be culled.
    """
    rend = DebrisRenderer(CAM)

    # A single-face quad in the xy-plane at z=5, normal = [0,0,-1] (toward cam)
    verts_front = np.array([
        [-1., -1., 5.],
        [ 1., -1., 5.],
        [ 1.,  1., 5.],
        [-1.,  1., 5.],
    ])
    # Camera is at origin looking along +z, so face normal pointing to camera = -z
    # In camera frame P_cam = P_world (identity pose)
    R_id = np.eye(3)
    t_id = np.zeros(3)

    # Count visible faces for front-facing model
    model_front = DebrisModel(
        name='test_front',
        vertices=verts_front,
        faces=np.array([[0,1,2,3]]),
        edges=np.array([[0,1],[1,2],[2,3],[3,0]]),
        face_normals=np.array([[0., 0., -1.]]),   # pointing toward camera
    )
    _, meta_front = rend.render(model_front, R_id, t_id)
    assert meta_front['n_visible_faces'] == 1, "Front-facing face must be visible"

    # Back-facing version: normal = [0,0,+1] (away from camera)
    model_back = DebrisModel(
        name='test_back',
        vertices=verts_front,
        faces=np.array([[0,1,2,3]]),
        edges=np.array([[0,1],[1,2],[2,3],[3,0]]),
        face_normals=np.array([[0., 0., 1.]]),    # pointing away from camera
    )
    _, meta_back = rend.render(model_back, R_id, t_id)
    assert meta_back['n_visible_faces'] == 0, "Back-facing face must be culled"


# ── Test 5: Ariane model geometry counts ─────────────────────────────────────
def test_ariane_geometry_counts():
    assert ARIANE.n_verts > 100, "Ariane model must have many vertices"
    assert ARIANE.n_edges > 100, "Ariane model must have many edges"
    assert len(ARIANE.keypoints) >= 10, \
        "Ariane model must have at least 10 keypoints"
    assert 'nose' in ARIANE.keypoints
    assert 'nozzle_tip' in ARIANE.keypoints
    assert 'body_centre' in ARIANE.keypoints


# ── Test 6: CubeSat face normals are unit vectors ─────────────────────────────
def test_cubesat_face_normals_unit():
    norms = np.linalg.norm(CUBE.face_normals, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-14,
        err_msg="All face normals must be unit vectors")
    assert len(CUBE.keypoints) >= 8, "CubeSat must have at least 8 corner keypoints"


# ── Test 7: render() returns correct shape and dtype ─────────────────────────
def test_render_output_shape():
    rend = DebrisRenderer(CAM)
    img, _ = rend.render(ARIANE, R_CW, T_CW, seed=0)
    assert img.shape == (CAM.height, CAM.width, 3), \
        f"Expected {(CAM.height, CAM.width, 3)}, got {img.shape}"
    assert img.dtype == np.uint8, "Image must be uint8"


# ── Test 8: Rendered image is not all background ──────────────────────────────
def test_render_nontrivial():
    rend = DebrisRenderer(CAM, RendererConfig(star_density=0.0, noise_sigma=0.0))
    img, meta = rend.render(ARIANE, R_CW, T_CW, seed=0)
    bg = np.array(RendererConfig().bg_color)
    # At least some pixels should differ from the background
    diff = np.abs(img.astype(int) - bg).sum(axis=-1)
    assert (diff > 10).sum() > 100, \
        "Rendered image must have non-background pixels (body must be visible)"


# ── Test 9: Keypoints within image bounds ─────────────────────────────────────
def test_keypoints_in_image_bounds():
    rend = DebrisRenderer(CAM)
    _, meta = rend.render(ARIANE, R_CW, T_CW, seed=0)
    kpts = meta['keypoints_2d']
    if len(kpts) == 0:
        pytest.skip("No visible keypoints in this view")
    us = kpts[:, 0]
    vs = kpts[:, 1]
    assert (us >= 0).all() and (us < CAM.width).all(),  "u coords out of bounds"
    assert (vs >= 0).all() and (vs < CAM.height).all(), "v coords out of bounds"


# ── Test 10: Bounding box valid ───────────────────────────────────────────────
def test_bounding_box_valid():
    rend = DebrisRenderer(CAM)
    _, meta = rend.render(ARIANE, R_CW, T_CW, seed=0)
    bbox = meta['bbox']
    assert bbox is not None, "Bounding box must not be None for visible body"
    u_min, v_min, u_max, v_max = bbox
    assert u_min < u_max, "bbox u_min must be < u_max"
    assert v_min < v_max, "bbox v_min must be < v_max"


# ── Test 11: random_rotation() is proper SO(3) ───────────────────────────────
def test_random_rotation_so3():
    rng = np.random.default_rng(7)
    for _ in range(20):
        R = random_rotation(rng)
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12,
            err_msg="random_rotation must be orthogonal")
        assert abs(np.linalg.det(R) - 1.0) < 1e-12, \
            "random_rotation determinant must be +1"


# ── Test 12: look_at_rotation points z-axis toward target ────────────────────
def test_look_at_rotation_z_toward_target():
    eye    = np.array([10., 5., 20.])
    target = np.zeros(3)
    R_cw, t_cw = look_at_rotation(eye, target)

    # Camera z-axis in world frame = R_cw.T @ [0,0,1]
    # Direction from eye to target in world
    expected_dir = (target - eye)
    expected_dir /= np.linalg.norm(expected_dir)

    cam_z_world = R_cw.T @ np.array([0., 0., 1.])
    np.testing.assert_allclose(cam_z_world, expected_dir, atol=1e-10,
        err_msg="Camera z-axis must point toward target")


# ── Test 13: Depth map minimum at closest point ───────────────────────────────
def test_depth_map_closest_point():
    rend = DebrisRenderer(CAM)
    depth = rend.render_depth(ARIANE, R_CW, T_CW)
    finite = depth[np.isfinite(depth)]
    if len(finite) == 0:
        pytest.skip("No visible geometry in depth map")
    # Camera is 30 m away; body is 8 m long, so min depth ≈ 26 m
    assert finite.min() > 0, "Depth values must be positive"
    assert finite.min() < 35.0, "Minimum depth must be less than 35 m for 30 m range"
    assert finite.max() > finite.min(), "Depth map must have variation"


# ── Test 14: Dataset generator produces correct frame count ──────────────────
def test_dataset_generator_frame_count():
    cfg = DatasetConfig(n_frames=5, seed=0)
    gen = DatasetGenerator(CAM, CUBE, cfg)
    annotations = gen.generate('outputs/test_dataset', verbose=False)
    assert len(annotations) == 5, \
        f"Expected 5 annotations, got {len(annotations)}"


# ── Test 15: Annotation keypoints are within image bounds ─────────────────────
def test_dataset_annotation_keypoints_in_bounds():
    cfg = DatasetConfig(n_frames=3, seed=99)
    gen = DatasetGenerator(CAM, ARIANE, cfg)
    anns = gen.generate('outputs/test_dataset2', verbose=False)
    for ann in anns:
        for name, (u, v) in ann.keypoints_2d.items():
            assert 0 <= u < CAM.width,  \
                f"Frame {ann.frame_id}: keypoint {name} u={u:.1f} out of bounds"
            assert 0 <= v < CAM.height, \
                f"Frame {ann.frame_id}: keypoint {name} v={v:.1f} out of bounds"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

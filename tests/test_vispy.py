"""
tests/test_vispy.py
===================
Unit tests for the Vispy GPU renderer (vision/vispy_renderer.py).

Requires: pip install "vispy>=0.14" pyopengl
Tests are skipped automatically when vispy is not installed.
"""

import pytest
import numpy as np

vispy_available = pytest.importorskip("vispy", reason="vispy not installed")

from vision.vispy_renderer import (
    VispyConfig,
    VispyRenderer,
    _triangulate,
    _phong_face_colors,
    _star_positions,
    _earth_limb_arc,
    vispy_offscreen_renderer,
)
from vision.camera import rendezvous_camera
from vision.body_model import ariane_model, cubesat_3u_model
from vision.renderer import look_at_rotation, random_rotation


@pytest.fixture(scope='module')
def camera():
    return rendezvous_camera()

@pytest.fixture(scope='module')
def model():
    return ariane_model()

@pytest.fixture(scope='module')
def cfg_small():
    return VispyConfig(width=128, height=96, n_stars=50, earth_limb=True)

@pytest.fixture(scope='module')
def renderer(camera, cfg_small):
    return VispyRenderer(camera, config=cfg_small, mode='offscreen')

@pytest.fixture(scope='module')
def pose():
    # look_at_rotation returns (R, look_direction) — unpack R only
    R, _ = look_at_rotation([0, 0, 1])
    t = np.array([0.0, 0.0, 20.0])
    return R, t


class TestTriangulate:
    def test_pass_through_triangles(self):
        tris = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32)
        out  = _triangulate(tris)
        assert out.shape == (2, 3)
        np.testing.assert_array_equal(out, tris)

    def test_quad_to_two_tris(self):
        quads = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)
        out   = _triangulate(quads)
        assert out.shape == (4, 3)
        np.testing.assert_array_equal(out[0], [0, 1, 2])
        np.testing.assert_array_equal(out[1], [0, 2, 3])

    def test_output_dtype(self):
        quads = np.array([[0, 1, 2, 3]], dtype=np.int64)
        out   = _triangulate(quads)
        assert out.dtype == np.int32


class TestPhongColors:
    def test_output_shape(self):
        verts = np.random.randn(10, 3).astype(np.float32)
        verts[:, 2] += 20.0
        tris  = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        rgba  = _phong_face_colors(verts, tris, VispyConfig())
        assert rgba.shape == (2, 4)
        assert rgba.dtype == np.float32

    def test_rgba_in_range(self):
        verts = np.array([[0, 0, 20], [1, 0, 20], [0, 1, 20],
                          [1, 1, 20]], dtype=np.float32)
        tris  = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
        rgba  = _phong_face_colors(verts, tris, VispyConfig())
        assert np.all(rgba >= 0.0) and np.all(rgba <= 1.0)

    def test_alpha_is_one(self):
        verts = np.random.randn(6, 3).astype(np.float32)
        verts[:, 2] += 25.0
        tris  = np.arange(6).reshape(2, 3).astype(np.int32)
        rgba  = _phong_face_colors(verts, tris, VispyConfig())
        np.testing.assert_array_equal(rgba[:, 3], 1.0)


class TestStarPositions:
    def test_count(self):
        pos = _star_positions(VispyConfig(n_stars=500))
        assert pos.shape == (500, 3)

    def test_radius(self):
        cfg  = VispyConfig(n_stars=200, star_dist=50_000.0)
        pos  = _star_positions(cfg)
        radii = np.linalg.norm(pos, axis=1)
        np.testing.assert_allclose(radii, 50_000.0, rtol=1e-4)

    def test_reproducible(self):
        cfg = VispyConfig(n_stars=100)
        np.testing.assert_array_equal(_star_positions(cfg, seed=42),
                                      _star_positions(cfg, seed=42))


class TestEarthLimb:
    def test_returns_three_rings(self):
        rings = _earth_limb_arc(np.array([0., 0., -20.], dtype=np.float32),
                                VispyConfig())
        assert len(rings) == 3

    def test_ring_shapes(self):
        rings = _earth_limb_arc(np.array([0., 0., -20.], dtype=np.float32),
                                VispyConfig())
        for ring in rings:
            assert ring.ndim == 2 and ring.shape[1] == 3


class TestVispyRenderer:
    def test_render_output_shape(self, renderer, model, pose):
        img = renderer.render(model, *pose)
        assert img.shape == (renderer.cfg.height, renderer.cfg.width, 3)

    def test_render_output_dtype(self, renderer, model, pose):
        assert renderer.render(model, *pose).dtype == np.uint8

    def test_render_not_all_black(self, renderer, model, pose):
        assert renderer.render(model, *pose).max() > 20

    def test_render_not_all_white(self, renderer, model, pose):
        assert renderer.render(model, *pose).min() < 250

    def test_render_different_poses_differ(self, renderer, model):
        R1, _ = look_at_rotation([0, 0, 1])
        Rz90  = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        R2    = R1 @ Rz90
        t     = np.array([0., 0., 20.])
        assert not np.array_equal(renderer.render(model, R1, t),
                                  renderer.render(model, R2, t))

    def test_render_cubesat(self, renderer, pose):
        img = renderer.render(cubesat_3u_model(), *pose)
        assert img.shape[2] == 3

    def test_render_close_range(self, camera, cfg_small, model):
        rend  = VispyRenderer(camera, config=cfg_small, mode='offscreen')
        R, _  = look_at_rotation([0, 0, 1])
        img   = rend.render(model, R, np.array([0., 0., 2.]))
        assert img.shape == (cfg_small.height, cfg_small.width, 3)

    def test_render_far_range(self, camera, cfg_small, model):
        rend  = VispyRenderer(camera, config=cfg_small, mode='offscreen')
        R, _  = look_at_rotation([0, 0, 1])
        img   = rend.render(model, R, np.array([0., 0., 200.]))
        assert img.shape == (cfg_small.height, cfg_small.width, 3)

    def test_factory_offscreen(self, camera, cfg_small, model, pose):
        img = vispy_offscreen_renderer(camera, cfg_small).render(model, *pose)
        assert img.shape == (cfg_small.height, cfg_small.width, 3)

    def test_wrong_mode_raises(self, camera, cfg_small, model, pose):
        rend = VispyRenderer(camera, config=cfg_small, mode='offscreen')
        rend.mode = 'interactive'
        with pytest.raises(RuntimeError):
            rend.render(model, *pose)


class TestVispyConfig:
    def test_defaults(self):
        cfg = VispyConfig()
        assert cfg.width == 1280 and cfg.height == 720
        assert cfg.earth_limb is True and cfg.n_stars == 2000

    def test_custom_values(self):
        cfg = VispyConfig(width=320, height=240, n_stars=100, earth_limb=False)
        assert cfg.width == 320 and cfg.n_stars == 100
        assert cfg.earth_limb is False

    def test_light_dir_is_numpy(self):
        cfg = VispyConfig()
        assert isinstance(cfg.light_dir_cam, np.ndarray)
        assert cfg.light_dir_cam.shape == (3,)

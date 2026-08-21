"""
phase3_demo.py
==============
Visualisation demo for Phase 3 — Synthetic Vision Pipeline.

Generates:
  Fig 10  — Ariane wireframe from 3 viewpoints (frontal, side, 3/4)
  Fig 11  — Rendered synthetic image (full shading + noise + stars)
  Fig 12  — Depth map heat-map
  Fig 13  — Keypoint & bounding-box overlay
  Fig 14  — Dataset sample grid (16 random poses, CubeSat)

Run from phase1/ directory:
    python notebooks/phase3_demo.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from vision import (
    rendezvous_camera, wide_angle_camera,
    ariane_model, cubesat_3u_model,
    RendererConfig, DebrisRenderer,
    random_rotation, look_at_rotation,
    DatasetConfig, DatasetGenerator,
)

os.makedirs('outputs', exist_ok=True)

print("Phase 3 Demo — Synthetic Vision Pipeline")
print("=" * 55)

CAM    = rendezvous_camera()
ARIANE = ariane_model()
CUBE   = cubesat_3u_model()

plt.rcParams.update({'font.family': 'serif', 'font.size': 10,
                     'figure.dpi': 150})

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 10 — Ariane wireframe from 3 viewpoints
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Fig 10: Ariane 3-view wireframe...")

cfg_wire = RendererConfig(
    draw_faces=False, draw_edges=True, draw_keypoints=True,
    star_density=0.0, noise_sigma=0.0,
    bg_color=(15, 15, 25),
    edge_color=(100, 200, 255),
    kpt_color=(255, 80, 80),
    kpt_radius=3,
)
rend_wire = DebrisRenderer(CAM, cfg_wire)

viewpoints = [
    ('Frontal',   np.array([ 0., 0., 30.])),
    ('Side',      np.array([30., 0.,  0.])),
    ('3/4 view',  np.array([20., 20., 15.])),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (label, eye) in zip(axes, viewpoints):
    R_cw, t_cw = look_at_rotation(eye, target=np.zeros(3))
    img, _ = rend_wire.render(ARIANE, R_cw, t_cw, seed=0)
    ax.imshow(img)
    ax.set_title(f'{label}\neye = {eye}', fontsize=9)
    ax.axis('off')

plt.suptitle('Ariane 44L — Wireframe + Keypoints (3 Viewpoints)', fontsize=12,
             fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/fig10_wireframe_views.png', dpi=150, bbox_inches='tight')
plt.close()
print("  -> outputs/fig10_wireframe_views.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 11 — Full shaded render with noise + stars
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Fig 11: Full shaded render...")

cfg_full = RendererConfig(
    draw_faces=True, draw_edges=True, draw_keypoints=False,
    ambient=0.20, diffuse=0.80,
    face_color=(160, 185, 210),
    edge_color=(220, 220, 80),
    noise_sigma=2.0,
    star_density=0.0004,
    bg_color=(4, 4, 10),
)
rend_full = DebrisRenderer(CAM, cfg_full)
eye_3q = np.array([18., 12., 15.])
R_cw, t_cw = look_at_rotation(eye_3q, np.zeros(3))

# Add a small body rotation for visual interest
R_body = random_rotation(np.random.default_rng(7))
img_full, meta_full = rend_full.render(
    ARIANE.transform(R_body), R_cw, t_cw, seed=42)

fig, ax = plt.subplots(1, 1, figsize=(7, 7))
ax.imshow(img_full)
ax.set_title(f'Ariane 44L — Synthetic Image\n'
             f'range≈{np.linalg.norm(eye_3q):.1f} m  |  '
             f'{meta_full["n_visible_faces"]} visible faces  |  '
             f'noise σ=2 DN  |  stars', fontsize=10)
ax.axis('off')
plt.tight_layout()
plt.savefig('outputs/fig11_synthetic_render.png', dpi=150, bbox_inches='tight')
plt.close()
print("  -> outputs/fig11_synthetic_render.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 12 — Depth map
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Fig 12: Depth map...")

rend_depth = DebrisRenderer(CAM)
depth_map  = rend_depth.render_depth(ARIANE.transform(R_body), R_cw, t_cw)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Depth heat-map (mask inf)
d_vis = np.where(np.isfinite(depth_map), depth_map, np.nan)
im = axes[0].imshow(d_vis, cmap='plasma', origin='upper')
plt.colorbar(im, ax=axes[0], label='Depth [m]')
axes[0].set_title('Depth Map (Z-buffer)', fontsize=10)
axes[0].axis('off')

# Depth histogram
finite_d = depth_map[np.isfinite(depth_map)]
axes[1].hist(finite_d, bins=80, color='#1A6FBF', edgecolor='none', alpha=0.8)
axes[1].set_xlabel('Depth [m]')
axes[1].set_ylabel('Pixel count')
axes[1].set_title(f'Depth Distribution\nmin={finite_d.min():.2f} m  '
                  f'max={finite_d.max():.2f} m', fontsize=10)
axes[1].grid(alpha=0.3, linestyle='--')

plt.suptitle('Ariane 44L — Depth Map', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/fig12_depth_map.png', dpi=150, bbox_inches='tight')
plt.close()
print("  -> outputs/fig12_depth_map.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 13 — Keypoint & bounding-box overlay
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Fig 13: Keypoint + bounding-box overlay...")

cfg_kpt = RendererConfig(
    draw_faces=True, draw_edges=False, draw_keypoints=False,
    ambient=0.25, diffuse=0.75,
    face_color=(160, 185, 210),
    noise_sigma=1.0, star_density=0.0003,
    bg_color=(4, 4, 10),
)
rend_kpt  = DebrisRenderer(CAM, cfg_kpt)
img_kpt, meta_kpt = rend_kpt.render(ARIANE.transform(R_body), R_cw, t_cw, seed=5)

fig, ax = plt.subplots(1, 1, figsize=(7, 7))
ax.imshow(img_kpt)

# Overlay keypoints
colors_kpt = plt.cm.Set1(np.linspace(0, 1, len(meta_kpt['keypoint_names'])))
for j, name in enumerate(meta_kpt['keypoint_names']):
    if name in meta_kpt['proj_keypoints']:
        u, v = meta_kpt['proj_keypoints'][name]
        ax.plot(u, v, 'o', color=colors_kpt[j % len(colors_kpt)],
                ms=6, markeredgewidth=0.5, markeredgecolor='w',
                label=name if j < 8 else '_')

# Bounding box
if meta_kpt['bbox']:
    u0, v0, u1, v1 = meta_kpt['bbox']
    rect = patches.Rectangle((u0, v0), u1-u0, v1-v0,
                               linewidth=2, edgecolor='#FFD700',
                               facecolor='none', linestyle='--')
    ax.add_patch(rect)

ax.legend(fontsize=7, ncol=2, loc='lower right',
          framealpha=0.7, markerscale=1.0)
ax.set_title(f'Ariane 44L — Keypoints ({len(meta_kpt["keypoint_names"])} visible) + '
             f'Bounding Box', fontsize=10)
ax.axis('off')
plt.tight_layout()
plt.savefig('outputs/fig13_keypoints_bbox.png', dpi=150, bbox_inches='tight')
plt.close()
print("  -> outputs/fig13_keypoints_bbox.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 14 — Dataset sample grid (16 random poses, CubeSat)
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Fig 14: Dataset sample grid (CubeSat)...")

cfg_ds = DatasetConfig(
    n_frames=16,
    range_min=2.0, range_max=8.0,
    noise_sigma=1.5, star_density=0.0003,
    seed=123,
)
gen_ds = DatasetGenerator(CAM, CUBE, cfg_ds)

rng_grid = np.random.default_rng(123)
frames   = []
cfg_rend = RendererConfig(
    draw_faces=True, draw_edges=True, draw_keypoints=True,
    face_color=(200, 160, 100),
    edge_color=(255, 220, 60),
    kpt_color=(0, 255, 120),
    kpt_radius=3,
    noise_sigma=1.5, star_density=0.0003,
    bg_color=(4, 4, 10),
)
rend_grid = DebrisRenderer(CAM, cfg_rend)

for i in range(16):
    R_b  = random_rotation(rng_grid)
    az   = rng_grid.uniform(0, 2*np.pi)
    el   = rng_grid.uniform(-1.0, 1.0)
    r    = rng_grid.uniform(2.0, 8.0)
    eye  = r * np.array([np.cos(el)*np.cos(az),
                          np.cos(el)*np.sin(az),
                          np.sin(el)])
    R_c, t_c = look_at_rotation(eye, np.zeros(3))
    img_i, _ = rend_grid.render(CUBE.transform(R_b), R_c, t_c, seed=i)
    frames.append(img_i)

fig, axes = plt.subplots(4, 4, figsize=(12, 12))
for i, (ax, img) in enumerate(zip(axes.flat, frames)):
    ax.imshow(img)
    ax.axis('off')
plt.suptitle('3U CubeSat — Synthetic Dataset Sample (16 Random Poses)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/fig14_dataset_grid.png', dpi=120, bbox_inches='tight')
plt.close()
print("  -> outputs/fig14_dataset_grid.png")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 55)
print("Phase 3 Validation Summary")
print("=" * 55)
print(f"  Camera (rendezvous)    : {CAM}")
print(f"  Ariane vertices        : {ARIANE.n_verts}")
print(f"  Ariane edges           : {ARIANE.n_edges}")
print(f"  Ariane faces           : {ARIANE.n_faces}")
print(f"  Ariane keypoints       : {len(ARIANE.keypoints)}")
print(f"  CubeSat vertices       : {CUBE.n_verts}")
print(f"  CubeSat keypoints      : {len(CUBE.keypoints)}")
print(f"  Visible faces (Fig 11) : {meta_full['n_visible_faces']}")
print(f"  Visible keypoints (13) : {len(meta_kpt['keypoint_names'])}")
if meta_kpt['bbox']:
    u0, v0, u1, v1 = meta_kpt['bbox']
    print(f"  Bounding box (px)      : [{u0:.0f},{v0:.0f}] → [{u1:.0f},{v1:.0f}]")
d_range = finite_d.max() - finite_d.min() if len(finite_d) else 0
print(f"  Depth range on body    : {d_range:.2f} m")
print()
print("Figures saved to outputs/  (fig10 through fig14)")
print("Run tests: python -m pytest tests/test_phase3.py -v")

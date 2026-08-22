"""
vispy_demo.py
=============
Demonstrate the GPU-accelerated Vispy renderer.

Produces
--------
  figures/fig_vispy_six_views.png      — Six rendered views at different poses
  figures/fig_vispy_compare.png        — Side-by-side: CPU wireframe vs Vispy GPU
  figures/fig_vispy_rotation_strip.png — 12-frame yaw rotation strip

Run
---
  cd vision-satellite-rendezvous
  python notebooks/vispy_demo.py

Requirements: pip install "vispy>=0.14" pyopengl
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from vision import (
    rendezvous_camera, ariane_model,
    DebrisRenderer, random_rotation, look_at_rotation, HAS_VISPY,
)

if not HAS_VISPY:
    print("[vispy_demo] vispy not installed — run:\n"
          "  pip install 'vispy>=0.14' pyopengl")
    sys.exit(0)

from vision import VispyConfig, VispyRenderer

FIG_DIR = Path(__file__).parent.parent / 'figures'
FIG_DIR.mkdir(exist_ok=True)

camera = rendezvous_camera()
model  = ariane_model()


def _Rx(deg):
    a = np.radians(deg)
    return np.array([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]])

def _Rz(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])


vcfg = VispyConfig(width=640, height=480, n_stars=1200, earth_limb=True)
rend_gpu = VispyRenderer(camera, config=vcfg, mode='offscreen')
rend_cpu = DebrisRenderer(camera)

# look_at_rotation returns (R_matrix, look_direction) — unpack with [0]
POSES = [
    dict(label='Nominal (20 m)',       R=look_at_rotation([0,0,1])[0],              t=np.array([0., 0.,20.])),
    dict(label='Roll 45°',             R=look_at_rotation([0,0,1])[0] @ _Rx(45),   t=np.array([0., 0.,20.])),
    dict(label='Yaw 60°',              R=look_at_rotation([0,0,1])[0] @ _Rz(60),   t=np.array([0., 0.,20.])),
    dict(label='Close approach (8 m)', R=look_at_rotation([0,0,1])[0],              t=np.array([0., 0., 8.])),
    dict(label='Off-axis (−5 m x)',    R=look_at_rotation([-5,0,20])[0],            t=np.array([-5.,0.,20.])),
    dict(label='Random tumble',        R=random_rotation(),                           t=np.array([2.,-1.,18.])),
]

# ── 1. Six-panel ──────────────────────────────────────────────────────────────
print("Rendering 6 GPU views …")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.patch.set_facecolor('#050510')
for ax, p in zip(axes.flat, POSES):
    ax.imshow(rend_gpu.render(model, p['R'], p['t']))
    ax.set_title(p['label'], color='white', fontsize=9)
    ax.axis('off')
fig.suptitle('GPU Renderer — Six Orbital Views', color='white', fontsize=13,
             fontweight='bold', y=1.01)
plt.tight_layout()
out = FIG_DIR / 'fig_vispy_six_views.png'
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"  → {out}")
plt.close(fig)

# ── 2. CPU vs GPU comparison ──────────────────────────────────────────────────
print("Rendering comparison …")
p0 = POSES[0]
img_cpu, _ = rend_cpu.render(model, p0['R'], p0['t'])
img_gpu    = rend_gpu.render(model, p0['R'], p0['t'])

fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig2.patch.set_facecolor('#050510')
ax1.imshow(img_cpu); ax1.set_title('CPU Renderer (Z-buffer wireframe)', color='white'); ax1.axis('off')
ax2.imshow(img_gpu); ax2.set_title('GPU Renderer (Vispy Phong)',         color='white'); ax2.axis('off')
fig2.suptitle('Renderer Comparison — Ariane 44L at 20 m', color='white', fontsize=12, fontweight='bold')
plt.tight_layout()
out2 = FIG_DIR / 'fig_vispy_compare.png'
fig2.savefig(out2, dpi=150, bbox_inches='tight', facecolor=fig2.get_facecolor())
print(f"  → {out2}")
plt.close(fig2)

# ── 3. Rotation strip ─────────────────────────────────────────────────────────
print("Rendering rotation strip …")
angles = np.linspace(0, 330, 12)
R_nom, _ = look_at_rotation([0, 0, 1])
imgs   = [rend_gpu.render(model, R_nom @ _Rz(a), np.array([0.,0.,20.])) for a in angles]

fig3, axes3 = plt.subplots(2, 6, figsize=(18, 6))
fig3.patch.set_facecolor('#050510')
for ax, img, ang in zip(axes3.flat, imgs, angles):
    ax.imshow(img); ax.set_title(f'{ang:.0f}°', color='white', fontsize=8); ax.axis('off')
fig3.suptitle('Yaw Rotation Strip (0° → 330°)', color='white', fontsize=12, fontweight='bold')
plt.tight_layout()
out3 = FIG_DIR / 'fig_vispy_rotation_strip.png'
fig3.savefig(out3, dpi=130, bbox_inches='tight', facecolor=fig3.get_facecolor())
print(f"  → {out3}")
plt.close(fig3)

print("\nDone. To launch the interactive viewer:")
print("  from vision import vispy_interactive_renderer, ariane_model, look_at_rotation")
print("  import numpy as np")
print("  rend = vispy_interactive_renderer()")
print("  R, _ = look_at_rotation([0,0,1])")
print("  rend.show(ariane_model(), R, np.array([0,0,20.]))")

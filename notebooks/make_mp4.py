"""
make_mp4.py — raw-pipe approach (fast)
=======================================
Produces two MP4s in outputs/:
  phase7_full_pipeline.mp4  — 3-panel: camera view | LVLH trajectory | range+error
  phase7_camera_feed.mp4    — immersive chaser-camera with telemetry HUD

Pipeline:
  1. Pre-render all wireframe images (numpy, fast).
  2. Build each frame as a composited numpy array using PIL + a tiny
     matplotlib sub-figure for the trajectory/range panels.
  3. Pipe raw RGB bytes directly to ffmpeg — no PNG encode per frame.

Run:
    cd /home/user/phase1
    python notebooks/make_mp4.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subprocess, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

from simulation import default_config, run_simulation
from vision.camera import rendezvous_camera
from vision.body_model import ariane_model
from vision.renderer import DebrisRenderer
from estimator.state import quat_to_dcm

OUTDIR = Path(__file__).parent.parent / 'outputs'
OUTDIR.mkdir(exist_ok=True)

FPS   = 24
# Output frame dimensions (must be even)
VW, VH = 1280, 480

BLUE  = (79,  195, 247)
AMBER = (255, 183, 77)
GREEN = (129, 199, 132)
RED   = (239, 154, 154)
GREY  = (120, 144, 156)
BG    = (13,  17,  23)

# ─────────────────────────────────────────────────────────────────────────────
def open_ffmpeg(path, w, h, fps=FPS):
    cmd = ['ffmpeg', '-y',
           '-f', 'rawvideo', '-vcodec', 'rawvideo',
           '-s', f'{w}x{h}', '-pix_fmt', 'rgb24',
           '-r', str(fps), '-i', '-',
           '-vcodec', 'libx264', '-preset', 'fast', '-crf', '20',
           '-pix_fmt', 'yuv420p', str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)


def canvas_to_rgb(fig):
    """Read a drawn matplotlib figure as (H, W, 3) uint8 — works on all mpl versions."""
    fig.canvas.draw()
    fw, fh = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(fh, fw, 4)
    return buf[:, :, :3].copy()   # RGBA → RGB


# ══════════════════════════════════════════════════════════════════════════════
# 1. Simulate
# ══════════════════════════════════════════════════════════════════════════════
print("Running simulation …")
cfg = default_config(
    n_steps=250, use_vision=True, use_ekf=True,
    u_max=0.3, pos_weight=15.0, vel_weight=1.5,
    pixel_noise_std=1.5, pos_meas_std=0.5,
    r0=np.array([30., 3., -1.5]),
    v0=np.array([-0.08, 0., 0.]),
    w0=np.array([0.02, 0.05, 0.01]),
    r0_err=np.array([2.5, -0.8, 0.5]),
)
res = run_simulation(cfg, rng_seed=42)
T   = res.n_steps_run + 1
print(f"  {T} steps | range {res.range_m[-1]:.2f} m | dock_step={res.dock_step}")

range_all = np.linalg.norm(res.r_true, axis=1)
perr_all  = res.pos_error


# ══════════════════════════════════════════════════════════════════════════════
# 2. Pre-render wireframes as PIL Images (uint8, exact panel size)
# ══════════════════════════════════════════════════════════════════════════════
CAM_W, CAM_H = VW // 2, VH   # camera panel occupies left half

print(f"Pre-rendering {T} wireframes …")
# Use a small camera (480×480) — same intrinsics scaled down, ~10× faster render
from vision.camera import PinholeCamera
_rc = rendezvous_camera()
_s  = 480 / _rc.width        # scale factor
cam_small = PinholeCamera(480, 480,
                          _rc.K[0,0]*_s, _rc.K[1,1]*_s,
                          _rc.K[0,2]*_s, _rc.K[1,2]*_s)
model   = ariane_model()
rend    = DebrisRenderer(cam_small)

# Try to load a monospace font; fall back to default
try:
    FONT_HUD  = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', 13)
    FONT_TINY = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', 11)
except Exception:
    FONT_HUD  = ImageFont.load_default()
    FONT_TINY = FONT_HUD

wire_pil = []
t0 = time.time()
for k in range(T):
    R_body  = quat_to_dcm(res.q_true[k])
    img_arr, _ = rend.render(model, R_body, res.r_true[k])
    # img_arr is already (H, W, 3) uint8 — resize to panel size
    pil_img = Image.fromarray(img_arr).resize((CAM_W, CAM_H), Image.BILINEAR)
    wire_pil.append(pil_img)
print(f"  Done in {time.time()-t0:.1f}s")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Build persistent matplotlib sub-figure for right-side panels
#    (trajectory + range/error)  — created once, artists updated per frame.
# ══════════════════════════════════════════════════════════════════════════════
SIDE_W, SIDE_H = VW - CAM_W, VH
DPI = 80
fig_s = plt.figure(figsize=(SIDE_W/DPI, SIDE_H/DPI), dpi=DPI, facecolor=(BG[0]/255, BG[1]/255, BG[2]/255))

ax_xy = fig_s.add_subplot(2, 1, 1); ax_xy.set_facecolor((0.07, 0.09, 0.12))
ax_er = fig_s.add_subplot(2, 1, 2); ax_er.set_facecolor((0.07, 0.09, 0.12))

def c1(t): return tuple(v/255 for v in t)  # (R,G,B) int → float for mpl

line_tr, = ax_xy.plot([], [], color=c1(BLUE),  lw=1.4, label='True')
line_es, = ax_xy.plot([], [], color=c1(AMBER), lw=1.1, ls='--', alpha=0.85, label='EKF')
dot_cur, = ax_xy.plot([], [], 'o', color=c1(GREEN), ms=5, zorder=5)
ax_xy.plot(0, 0, '*', color=c1(RED), ms=8, zorder=5)
ax_xy.set_xlim(res.r_true[:, 0].min()-1, res.r_true[:, 0].max()+1)
ax_xy.set_ylim(res.r_true[:, 1].min()-1, res.r_true[:, 1].max()+1)
ax_xy.set_title('LVLH trajectory', color='white', fontsize=8, pad=2)
ax_xy.tick_params(colors='white', labelsize=6)
ax_xy.grid(alpha=0.15)
ax_xy.legend(fontsize=6, loc='upper right')
step_txt = ax_xy.text(0.02, 0.95, '', transform=ax_xy.transAxes,
                      color=c1(GREEN), fontsize=7.5, va='top', fontfamily='monospace')

line_rng, = ax_er.plot([], [], color=c1(BLUE),  lw=1.4, label='Range [m]')
line_err, = ax_er.plot([], [], color=c1(AMBER), lw=1.1, ls='--', label='EKF err [m]')
if res.dock_step:
    ax_er.axvline(res.dock_step, color=c1(GREEN), lw=0.8, ls=':', alpha=0.7)
ax_er.set_xlim(0, T - 1)
ax_er.set_ylim(-0.5, max(range_all.max(), perr_all.max()) * 1.05)
ax_er.set_title('Range & estimation error', color='white', fontsize=8, pad=2)
ax_er.tick_params(colors='white', labelsize=6)
ax_er.grid(alpha=0.15)
ax_er.legend(fontsize=6, loc='upper right')

for ax in [ax_xy, ax_er]:
    for sp in ax.spines.values():
        sp.set_color('#2a3040')

fig_s.tight_layout(pad=0.5)

# ══════════════════════════════════════════════════════════════════════════════
# Helpers: draw HUD on PIL image
# ══════════════════════════════════════════════════════════════════════════════

def draw_reticle(pil_img):
    """Draw targeting brackets + crosshair on the camera image (in place)."""
    d  = ImageDraw.Draw(pil_img)
    cx, cy = CAM_W // 2, CAM_H // 2
    arm = min(CAM_W, CAM_H) // 8
    pad = arm // 2
    gn  = tuple(GREEN)
    lw  = 2
    for sx, sy in [(-1,-1),(-1,1),(1,-1),(1,1)]:
        x0 = cx + sx * pad; y0 = cy + sy * pad
        d.line([(x0, y0), (x0 + sx*arm, y0)], fill=gn, width=lw)
        d.line([(x0, y0), (x0, y0 + sy*arm)], fill=gn, width=lw)
    d.line([(cx-8, cy),(cx+8, cy)], fill=gn, width=1)
    d.line([(cx, cy-8),(cx, cy+8)], fill=gn, width=1)


def draw_hud(pil_img, k, blink):
    """Write telemetry text onto camera panel."""
    d = ImageDraw.Draw(pil_img)
    range_k = range_all[k]
    vel_k   = float(np.linalg.norm(res.v_true[k]))
    err_k   = perr_all[k]
    lines = [
        f'STEP  {k:>3d}/{T-1}',
        f'RANGE {range_k:>7.2f} m',
        f'SPEED {vel_k:>7.4f} m/s',
        f'EKF Δ {err_k:>7.3f} m',
    ]
    if res.dock_step and k >= res.dock_step:
        lines.append('■ DOCKED')
    y = 10
    for ln in lines:
        d.text((10, y), ln, fill=GREEN, font=FONT_HUD)
        y += 18

    if blink:
        d.ellipse([CAM_W-22, 10, CAM_W-10, 22], fill=tuple(RED))
        d.text((CAM_W-50, 10), 'REC', fill=tuple(RED), font=FONT_HUD)

    d.text((CAM_W//2 - 80, CAM_H - 20),
           'CHASER CAM  ·  ARIANE 44L',
           fill=tuple(GREY), font=FONT_TINY)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Camera-feed MP4  (immersive single-camera view)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nEncoding camera-feed MP4 ({T} frames) …")
out_cam = OUTDIR / 'phase7_camera_feed.mp4'
proc_cam = open_ffmpeg(out_cam, VW, VH)

t0 = time.time()
for k in range(T):
    # ── left: camera panel ──────────────────────────────────────────────
    cam_pil = wire_pil[k].copy()
    draw_reticle(cam_pil)
    draw_hud(cam_pil, k, blink=(k % 20 < 10))

    # ── right: side panels ─────────────────────────────────────────────
    t_arr = np.arange(k + 1)
    line_tr.set_data(res.r_true[:k+1, 0], res.r_true[:k+1, 1])
    line_es.set_data(res.r_est[:k+1, 0],  res.r_est[:k+1, 1])
    dot_cur.set_data([res.r_true[k, 0]], [res.r_true[k, 1]])
    step_txt.set_text(f'step {k}  {range_all[k]:.1f} m')
    line_rng.set_data(t_arr, range_all[:k+1])
    line_err.set_data(t_arr, perr_all[:k+1])
    fig_s.canvas.draw()
    side_arr = canvas_to_rgb(fig_s)
    side_pil = Image.fromarray(side_arr).resize((SIDE_W, SIDE_H), Image.BILINEAR)

    # ── composite ──────────────────────────────────────────────────────
    frame_pil = Image.new('RGB', (VW, VH), BG)
    frame_pil.paste(cam_pil,  (0,      0))
    frame_pil.paste(side_pil, (CAM_W,  0))

    proc_cam.stdin.write(np.array(frame_pil).tobytes())

    if k % 50 == 0:
        print(f'  cam-feed frame {k}/{T-1}  elapsed {time.time()-t0:.0f}s')

proc_cam.stdin.close()
proc_cam.wait()
size_cam = out_cam.stat().st_size / 1e6
print(f"  ✓  {len(wire_pil)} frames  {size_cam:.1f} MB  ({time.time()-t0:.0f}s)")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Full 3-panel pipeline MP4
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nEncoding full-pipeline MP4 ({T} frames) …")

# Panel layout: [cam | traj | range+err]  each 1/3 width
PW = VW // 3
fig_p = plt.figure(figsize=(VW/DPI, VH/DPI), dpi=DPI,
                   facecolor=(BG[0]/255, BG[1]/255, BG[2]/255))
ax_im = fig_p.add_subplot(1, 3, 1); ax_im.set_facecolor((0.07, 0.09, 0.12))
ax_p2 = fig_p.add_subplot(1, 3, 2); ax_p2.set_facecolor((0.07, 0.09, 0.12))
ax_p3 = fig_p.add_subplot(1, 3, 3); ax_p3.set_facecolor((0.07, 0.09, 0.12))

im_art = ax_im.imshow(np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8),
                      interpolation='nearest', aspect='auto')
ax_im.set_title('Chaser camera', color='white', fontsize=9)
ax_im.axis('off')

p2_tr,  = ax_p2.plot([], [], color=c1(BLUE),  lw=1.4, label='True')
p2_es,  = ax_p2.plot([], [], color=c1(AMBER), lw=1.1, ls='--', label='EKF')
p2_dot, = ax_p2.plot([], [], 'o', color=c1(GREEN), ms=5, zorder=5)
ax_p2.plot(0, 0, '*', color=c1(RED), ms=8, zorder=5)
ax_p2.set_xlim(res.r_true[:, 0].min()-1, res.r_true[:, 0].max()+1)
ax_p2.set_ylim(res.r_true[:, 1].min()-1, res.r_true[:, 1].max()+1)
ax_p2.set_title('LVLH trajectory', color='white', fontsize=9)
ax_p2.tick_params(colors='white', labelsize=6)
ax_p2.grid(alpha=0.2); ax_p2.legend(fontsize=7)
p2_txt = ax_p2.text(0.02, 0.96, '', transform=ax_p2.transAxes,
                    color=c1(GREEN), fontsize=7.5, va='top', fontfamily='monospace')

p3_rng, = ax_p3.plot([], [], color=c1(BLUE),  lw=1.4, label='Range [m]')
p3_err, = ax_p3.plot([], [], color=c1(AMBER), lw=1.1, ls='--', label='EKF err')
if res.dock_step:
    ax_p3.axvline(res.dock_step, color=c1(GREEN), lw=0.8, ls=':', alpha=0.7)
ax_p3.set_xlim(0, T - 1)
ax_p3.set_ylim(-0.5, max(range_all.max(), perr_all.max()) * 1.05)
ax_p3.set_title('Range & estimation error', color='white', fontsize=9)
ax_p3.tick_params(colors='white', labelsize=6)
ax_p3.grid(alpha=0.2); ax_p3.legend(fontsize=7)
for ax in [ax_p2, ax_p3]:
    for sp in ax.spines.values(): sp.set_color('#2a3040')

fig_p.suptitle('Phase 7 — Closed-Loop Rendezvous  (EKF + LQR + Vision)',
               color='white', fontsize=10)
fig_p.tight_layout(pad=0.4)

out_full = OUTDIR / 'phase7_full_pipeline.mp4'
proc_full = open_ffmpeg(out_full, VW, VH)

t0 = time.time()
for k in range(T):
    # update wireframe (convert PIL → array for imshow)
    cam_arr = np.array(wire_pil[k])
    im_art.set_data(cam_arr)

    # update trajectory lines
    t_arr = np.arange(k + 1)
    p2_tr.set_data(res.r_true[:k+1, 0], res.r_true[:k+1, 1])
    p2_es.set_data(res.r_est[:k+1, 0],  res.r_est[:k+1, 1])
    p2_dot.set_data([res.r_true[k, 0]], [res.r_true[k, 1]])
    p2_txt.set_text(f'step {k}  {range_all[k]:.1f} m')
    p3_rng.set_data(t_arr, range_all[:k+1])
    p3_err.set_data(t_arr, perr_all[:k+1])

    arr = canvas_to_rgb(fig_p)
    if arr.shape[:2] != (VH, VW):
        arr = np.array(Image.fromarray(arr).resize((VW, VH), Image.BILINEAR))
    proc_full.stdin.write(arr.tobytes())

    if k % 50 == 0:
        print(f'  pipeline frame {k}/{T-1}  elapsed {time.time()-t0:.0f}s')

proc_full.stdin.close()
proc_full.wait()
size_full = out_full.stat().st_size / 1e6
print(f"  ✓  {T} frames  {size_full:.1f} MB  ({time.time()-t0:.0f}s)")

print("\n=== Complete ===")
print(f"  {out_cam.name:<36}  {out_cam.stat().st_size/1e6:.1f} MB")
print(f"  {out_full.name:<36}  {out_full.stat().st_size/1e6:.1f} MB")

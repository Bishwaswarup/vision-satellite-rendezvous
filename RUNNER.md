# RUNNER — Terminal Guide
## Vision-Based Satellite Rendezvous & Debris Tracking Simulator

All commands are run from the **project root** (`vision-satellite-rendezvous/`).

---

## Quick start — run everything

```bash
# Install dependencies (once)
pip install -r requirements.txt

# Run all 129 tests (105 core + 24 GPU renderer)
python -m pytest tests/ -v

# Run all demos (generates fig1–fig34 + GIF + vispy figures in outputs/)
python notebooks/phase1_demo.py
python notebooks/phase2_demo.py
python notebooks/phase3_demo.py
python notebooks/phase4_demo.py
python notebooks/phase5_demo.py
python notebooks/phase6_demo.py
python notebooks/phase7_demo.py
python notebooks/vispy_demo.py
```

---

## Phase 1 — Orbital Dynamics (HCW + J2)

**What it does:** Clohessy-Wiltshire propagator, Yamanaka-Ankersen STM,
J2 perturbation, periodic orbit design.

```bash
# Tests (15)
python -m pytest tests/test_phase1.py -v

# Demo → outputs/fig1_periodic_orbit.png … fig4_j2_vs_hcw.png
python notebooks/phase1_demo.py
```

**Key modules:**
```
dynamics/hcw.py          HCWPropagator, solve_hcw()
dynamics/ya_stm.py       YAStatTransition
dynamics/j2_perturb.py   J2PerturbedHCW
dynamics/constants.py    MU, R_EARTH, J2, …
```

**Quick import check:**
```bash
python -c "from dynamics import HCWPropagator; print('Phase 1 OK')"
```

---

## Phase 2 — Target Kinematics (Rigid Body Attitude)

**What it does:** Torque-free Euler dynamics, DOP853 integrator, quaternion
library, Dzhanibekov (tumbling) effect for Ariane 44L.

```bash
# Tests (15)
python -m pytest tests/test_phase2.py -v

# Demo → outputs/fig5_angular_velocity.png … fig9_body_comparison.png
python notebooks/phase2_demo.py
```

**Key modules:**
```
target/attitude.py       RigidBodyAttitude, J_ARIANE_44L
target/quaternion.py     quat_mult, quat_to_dcm, dcm_to_quat, …
```

**Quick import check:**
```bash
python -c "from target import RigidBodyAttitude; print('Phase 2 OK')"
```

---

## Phase 3 — Synthetic Vision Pipeline

**What it does:** Pinhole camera model, Ariane / CubeSat 3-D mesh models,
OpenGL-free wireframe renderer, keypoint projection, dataset generator.

```bash
# Tests (15)
python -m pytest tests/test_phase3.py -v

# Demo → outputs/fig10_wireframe_views.png … fig14_dataset_grid.png
python notebooks/phase3_demo.py
```

**Key modules:**
```
vision/camera.py         PinholeCamera, rendezvous_camera(), wide_angle_camera()
vision/body_model.py     DebrisModel, ariane_model(), cubesat_3u_model()
vision/renderer.py       DebrisRenderer, look_at_rotation(), random_rotation()
vision/dataset.py        DatasetGenerator, DatasetConfig, PoseAnnotation
```

**Quick import check:**
```bash
python -c "from vision import rendezvous_camera, ariane_model; print('Phase 3 OK')"
```

---

## Phase 4 — PnP Pose Estimation (EPnP + RANSAC + GN Refinement)

**What it does:** EPnP (Lepetit 2009) closed-form solver, RANSAC outlier
rejection, Gauss-Newton / Levenberg-Marquardt reprojection refinement,
Rodrigues parameterisation.

```bash
# Tests (15)
python -m pytest tests/test_phase4.py -v

# Demo → outputs/fig15_reprojection_overlay.png … fig19_ariane_pose.png
python notebooks/phase4_demo.py
```

**Key modules:**
```
pose/epnp.py             EPnPSolver, solve_epnp()
pose/ransac.py           RANSACSolver, solve_pnp_ransac()
pose/refine.py           GaussNewtonRefiner, refine_pose()
```

**One-liner pose solve:**
```bash
python -c "
import numpy as np
from vision import rendezvous_camera, ariane_model, look_at_rotation
from pose import solve_epnp

cam   = rendezvous_camera()
model = ariane_model()
eye   = np.array([20., 10., 15.])
R_gt, _ = look_at_rotation(eye)
kp3d  = model.keypoint_array
P_cam = (R_gt @ kp3d.T).T + np.array([0., 0., 20.])
u = cam.K[0,0] * P_cam[:,0] / P_cam[:,2] + cam.K[0,2]
v = cam.K[1,1] * P_cam[:,1] / P_cam[:,2] + cam.K[1,2]
kp2d  = np.stack([u,v], axis=-1)
R, t, errs, ok = solve_epnp(kp3d, kp2d, cam.K)
print(f'EPnP: ok={ok}  mean_repr_err={errs.mean():.2e} px')
"
```

---

## Phase 5 — EKF / UKF State Estimator

**What it does:** Multiplicative EKF (MEKF) and Unscented KF for relative
pose tracking over time. HCW + quaternion kinematics process model.
Mahalanobis outlier gating. NIS covariance consistency.

```bash
# Tests (15)
python -m pytest tests/test_phase5.py -v

# Demo → outputs/fig20_ekf_error_vs_time.png … fig24_ariane_ekf_pipeline.png
python notebooks/phase5_demo.py
```

**Key modules:**
```
estimator/state.py       pack_state(), unpack_state(), propagate_rk4(),
                         h_measurement(), measurement_jacobian()
estimator/ekf.py         MultEKF, make_ekf()
estimator/ukf.py         UnscentedKF, make_ukf()
```

**Quick filter run:**
```bash
python -c "
import numpy as np
from estimator import make_ekf, pack_state, rotvec_to_quat, propagate_rk4, h_measurement, unpack_state

x0  = pack_state([25,0,0], [-0.05,0,0], rotvec_to_quat([0,0,0.2]), [0,0,0.02])
ekf = make_ekf(x0, dt=1.0, pos0_std=2.0)
for _ in range(10):
    propagate_rk4(x0, 1.0)
    ekf.predict(1.0)
    ekf.update(h_measurement(x0))
r = unpack_state(ekf.state)[0]
print(f'EKF position estimate: {r}')
"
```

---

## Phase 6 — LQR / MPC Controller

**What it does:** Infinite-horizon discrete LQR via DARE, receding-horizon
MPC with condensed QP (SLSQP), thrust saturation, approach-cone constraint,
Δv accounting, EKF-in-the-loop rendezvous.

```bash
# Tests (15)
python -m pytest tests/test_phase6.py -v

# Demo → outputs/fig25_lqr_trajectory.png … fig29_ekf_lqr_pipeline.png
python notebooks/phase6_demo.py
```

**Key modules:**
```
controller/lqr.py        LQRController, make_lqr(), hcw_discrete()
controller/mpc.py        MPCController, make_mpc()
```

**Quick control run:**
```bash
python -c "
import numpy as np
from controller import make_lqr

lqr = make_lqr(u_max=0.2)
print(f'Stable: {lqr.is_stable()}')
print(f'Gain K:\n{lqr.K.round(4)}')
sim = lqr.simulate([20, 3, -1, -0.1, 0, 0], n_steps=200)
print(f'Final range: {np.linalg.norm(sim[\"states\"][-1,:3]):.4f} m')
print(f'Total Δv:    {sim[\"delta_v\"]:.3f} m/s')
"
```

---

## Phase 7 — Closed-Loop Integration + Rendezvous Video

**What it does:** End-to-end closed-loop simulation integrating all phases:
vision pipeline → RANSAC+EPnP pose → Multiplicative EKF → LQR control →
HCW+attitude dynamics. Animated GIF of the wireframe rendezvous. SimConfig
dataclass for all parameters, SimResult for logged trajectories.

```bash
# Tests (15)
python -m pytest tests/test_phase7.py -v

# Demo → outputs/fig30–fig34 + outputs/phase7_rendezvous.gif
python notebooks/phase7_demo.py
```

**Key modules:**
```
simulation/runner.py     RendezvousSimulator, SimConfig, SimResult, run_simulation()
simulation/video.py      VideoExporter, render_frame(), frames_to_gif()
```

**Quick full-pipeline run:**
```bash
python -c "
import numpy as np
from simulation import run_simulation, default_config

cfg = default_config(n_steps=150, use_vision=False, use_ekf=True,
                     u_max=0.3, r0=np.array([20., 2., -1.]))
res = run_simulation(cfg)
print(f'Final range:   {res.range_m[-1]:.3f} m')
print(f'Est. error:    {res.pos_error[-1]:.3f} m')
print(f'Total Δv:      {res.delta_v:.2f} m/s')
print(f'Docked:        {\"Yes @ step \" + str(res.dock_step) if res.dock_step else \"No\"}')
"
```

---

## GPU Renderer (Vispy) — Optional

**What it does:** Hardware-accelerated offscreen rendering with Blinn-Phong
shading, 2 000-point star field, Earth limb glow, and full HiDPI / Retina
support. Platform-aware backend: osmesa/egl on Linux, auto-detect on macOS.
Compatible with vispy ≥ 0.14 (uses TurntableCamera; PerspectiveCamera removed).

```bash
# Tests (24)
python -m pytest tests/test_vispy.py -v

# Demo → outputs/fig_vispy_six_views.png
#         outputs/fig_vispy_compare.png
#         outputs/fig_vispy_rotation_strip.png
python notebooks/vispy_demo.py
```

**Key module:**
```
vision/vispy_renderer.py   VispyRenderer, VispyConfig, RendererMode
```

**Quick render check:**
```bash
python -c "
import numpy as np
from vision import rendezvous_camera, ariane_model, look_at_rotation
from vision.vispy_renderer import VispyRenderer

cam   = rendezvous_camera()
rend  = VispyRenderer(cam, mode='offscreen')
model = ariane_model()
R, _  = look_at_rotation([0, 0, 1])
img   = rend.render(model, R, np.array([0., 0., 20.]))
print(f'GPU render OK — shape: {img.shape}  dtype: {img.dtype}')
"
```

**Import check:**
```bash
python -c "from vision.vispy_renderer import VispyRenderer; print('Vispy renderer OK')"
```

---

## Running specific test groups

```bash
# Single test by name
python -m pytest tests/test_phase4.py::test_epnp_noiseless_reprojection -v

# All RANSAC tests
python -m pytest tests/test_phase4.py -k "ransac" -v

# All EKF tests across phases
python -m pytest tests/ -k "ekf" -v

# All GPU renderer tests
python -m pytest tests/test_vispy.py -v

# Core phases only (no GPU)
python -m pytest tests/ --ignore=tests/test_vispy.py -v

# Stop on first failure
python -m pytest tests/ -x -v

# Short traceback
python -m pytest tests/ --tb=short

# With timing info
python -m pytest tests/ -v --durations=10
```

---

## Output figures reference

| Figure | File | Phase | Description |
|--------|------|-------|-------------|
| fig1  | `fig1_periodic_orbit.png`          | 1   | HCW periodic orbit |
| fig2  | `fig2_analytical_vs_numerical.png` | 1   | Analytical vs RK45 |
| fig3  | `fig3_j2_disturbance.png`          | 1   | J2 perturbation drift |
| fig4  | `fig4_j2_vs_hcw.png`               | 1   | J2 vs ideal HCW |
| fig5  | `fig5_angular_velocity.png`        | 2   | ω(t) tumbling Ariane |
| fig6  | `fig6_polhode.png`                 | 2   | Polhode on energy ellipsoid |
| fig7  | `fig7_euler_angles.png`            | 2   | Euler angle evolution |
| fig8  | `fig8_conservation.png`            | 2   | Energy / momentum conservation |
| fig9  | `fig9_body_comparison.png`         | 2   | Ariane vs CubeSat dynamics |
| fig10 | `fig10_wireframe_views.png`        | 3   | Multi-view wireframe |
| fig11 | `fig11_synthetic_render.png`       | 3   | Synthetic depth render |
| fig12 | `fig12_depth_map.png`              | 3   | Depth map |
| fig13 | `fig13_keypoints_bbox.png`         | 3   | Keypoints + bounding box |
| fig14 | `fig14_dataset_grid.png`           | 3   | Dataset sample grid |
| fig15 | `fig15_reprojection_overlay.png`   | 4   | EPnP reprojection overlay |
| fig16 | `fig16_error_vs_noise.png`         | 4   | Pose error vs noise level |
| fig17 | `fig17_ransac_convergence.png`     | 4   | RANSAC best-inlier count |
| fig18 | `fig18_repr_error_cdf.png`         | 4   | Reprojection error CDF |
| fig19 | `fig19_ariane_pose.png`            | 4   | Full pipeline on Ariane |
| fig20 | `fig20_ekf_error_vs_time.png`      | 5   | EKF position & attitude error |
| fig21 | `fig21_ukf_error_vs_time.png`      | 5   | UKF position & attitude error |
| fig22 | `fig22_ekf_vs_ukf_rmse.png`        | 5   | EKF vs UKF MC RMSE |
| fig23 | `fig23_nis_consistency.png`        | 5   | NIS covariance consistency |
| fig24 | `fig24_ariane_ekf_pipeline.png`    | 5   | RANSAC+EPnP → EKF pipeline |
| fig25 | `fig25_lqr_trajectory.png`         | 6   | LQR rendezvous trajectory |
| fig26 | `fig26_lqr_vs_mpc.png`             | 6   | LQR vs MPC comparison |
| fig27 | `fig27_thrust_profiles.png`        | 6   | Per-axis thrust profiles |
| fig28 | `fig28_approach_cone.png`          | 6   | MPC approach cone constraint |
| fig29 | `fig29_ekf_lqr_pipeline.png`       | 6   | Full EKF + LQR pipeline |
| fig30 | `fig30_fullpipeline_trajectory.png`| 7   | Full pipeline 3-D trajectory |
| fig31 | `fig31_range_and_error.png`        | 7   | Range + estimation error |
| fig32 | `fig32_thrust_profiles.png`        | 7   | Thrust + cumulative Δv |
| fig33 | `fig33_reprojection_error.png`     | 7   | EPnP reprojection error vs range |
| fig34 | `fig34_ekf_convergence.png`        | 7   | EKF covariance convergence |
| GIF   | `phase7_rendezvous.gif`            | 7   | Animated wireframe rendezvous |
| GPU-1 | `fig_vispy_six_views.png`         | GPU | Six-pose Blinn-Phong render grid |
| GPU-2 | `fig_vispy_compare.png`           | GPU | CPU wireframe vs GPU shaded |
| GPU-3 | `fig_vispy_rotation_strip.png`    | GPU | Rotation strip across yaw angles |

---

## Project structure

```
vision-satellite-rendezvous/
├── dynamics/           Phase 1 — Orbital mechanics
│   ├── hcw.py
│   ├── ya_stm.py
│   ├── j2_perturb.py
│   └── constants.py
├── target/             Phase 2 — Rigid body attitude
│   ├── attitude.py
│   └── quaternion.py
├── vision/             Phase 3 — Synthetic vision + GPU renderer
│   ├── camera.py
│   ├── body_model.py
│   ├── renderer.py
│   ├── dataset.py
│   └── vispy_renderer.py
├── pose/               Phase 4 — PnP estimation
│   ├── epnp.py
│   ├── ransac.py
│   └── refine.py
├── estimator/          Phase 5 — EKF / UKF filter
│   ├── state.py
│   ├── ekf.py
│   └── ukf.py
├── controller/         Phase 6 — LQR / MPC control
│   ├── lqr.py
│   └── mpc.py
├── simulation/         Phase 7 — Closed-loop integration
│   ├── runner.py
│   └── video.py
├── tests/              129 unit tests (15 × 7 phases + 24 GPU)
├── notebooks/          Demo scripts → outputs/
├── outputs/            Generated figures (fig1–fig34, vispy figs, GIF)
└── requirements.txt
```

---

## Environment notes

- **Python:** 3.10 or later
- **Core dependencies:** `numpy`, `scipy`, `matplotlib`, `pytest`
- **GPU renderer:** `vispy >= 0.14` (optional — core phases work without it)
- **Install:** `pip install -r requirements.txt`
- **Working directory:** always run from the project root (the folder containing `requirements.txt`)
- **No GPU required for core phases:** all core rendering is CPU-based wireframe
- **GPU renderer notes:** uses osmesa/egl on Linux; auto-detects backend on macOS; HiDPI-aware on Retina displays

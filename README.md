# Vision-Based Satellite Rendezvous & Debris Tracking Simulator

A closed-loop GNC simulation framework for autonomous space debris rendezvous using monocular vision, nonlinear Kalman filtering, and predictive control.

---

## Project Structure

```
vision-satellite-rendezvous/
├── dynamics/
│   ├── constants.py       # Physical constants (mu, J2, R_E, LEO defaults)
│   ├── hcw.py             # HCW propagator — analytical STM + DOP853 numerical
│   ├── ya_stm.py          # Yamanaka-Ankersen STM for eccentric orbits
│   ├── j2_perturb.py      # J2 perturbation + chief ECI propagator
│   └── __init__.py
├── target/
│   ├── attitude.py        # Torque-free Euler dynamics, Dzhanibekov effect
│   ├── quaternion.py      # Quaternion library (mult, DCM, integration)
│   └── __init__.py
├── vision/
│   ├── camera.py          # PinholeCamera, rendezvous_camera(), wide_angle_camera()
│   ├── body_model.py      # DebrisModel, ariane_model(), cubesat_3u_model()
│   ├── renderer.py        # CPU wireframe renderer
│   ├── dataset.py         # DatasetGenerator, DatasetConfig, PoseAnnotation
│   ├── vispy_renderer.py  # GPU renderer — Blinn-Phong, star field, Earth limb
│   └── __init__.py
├── pose/
│   ├── epnp.py            # EPnP (Lepetit 2009) closed-form solver
│   ├── ransac.py          # RANSAC outlier rejection
│   ├── refine.py          # Gauss-Newton / Levenberg-Marquardt refinement
│   └── __init__.py
├── estimator/
│   ├── state.py           # State packing, RK4 propagation, measurement model
│   ├── ekf.py             # Multiplicative EKF (MEKF)
│   ├── ukf.py             # Unscented Kalman Filter (Merwe sigma points)
│   └── __init__.py
├── controller/
│   ├── lqr.py             # Infinite-horizon LQR via DARE
│   ├── mpc.py             # Receding-horizon MPC (SLSQP, approach-cone constraint)
│   └── __init__.py
├── simulation/
│   ├── runner.py          # RendezvousSimulator, SimConfig, SimResult
│   ├── video.py           # VideoExporter, frames_to_gif()
│   └── __init__.py
├── tests/
│   ├── test_phase1.py     # 15 unit tests — orbital dynamics
│   ├── test_phase2.py     # 15 unit tests — rigid body attitude
│   ├── test_phase3.py     # 15 unit tests — synthetic vision
│   ├── test_phase4.py     # 15 unit tests — pose estimation
│   ├── test_phase5.py     # 15 unit tests — EKF / UKF
│   ├── test_phase6.py     # 15 unit tests — LQR / MPC
│   ├── test_phase7.py     # 15 unit tests — closed-loop integration
│   └── test_vispy.py      # 24 unit tests — GPU renderer (optional)
├── notebooks/
│   ├── phase1_demo.py     # → fig1–fig4
│   ├── phase2_demo.py     # → fig5–fig9
│   ├── phase3_demo.py     # → fig10–fig14
│   ├── phase4_demo.py     # → fig15–fig19
│   ├── phase5_demo.py     # → fig20–fig24
│   ├── phase6_demo.py     # → fig25–fig29
│   ├── phase7_demo.py     # → fig30–fig34 + rendezvous GIF
│   └── vispy_demo.py      # → fig_vispy_*.png (GPU renderer showcase)
├── outputs/               # Generated figures and GIF
├── config/                # YAML config files
├── requirements.txt
├── README.md
└── RUNNER.md              # Terminal quick-reference guide
```

---

## Phase Roadmap

| Phase | Module | Description | Status |
|-------|--------|-------------|--------|
| 1 | Orbital Dynamics Engine | HCW propagator, Yamanaka-Ankersen STM, J2 perturbation | ✅ Complete |
| 2 | Target Kinematics & Tumbling | Torque-free Euler dynamics, Dzhanibekov effect, quaternion library | ✅ Complete |
| 3 | Synthetic Vision Pipeline | Pinhole camera, wireframe mesh models, CPU renderer, dataset generator | ✅ Complete |
| 4 | Pose Estimation (EPnP) | EPnP closed-form solver, RANSAC outlier rejection, GN/LM refinement | ✅ Complete |
| 5 | State Estimation — EKF & UKF | Multiplicative EKF, Unscented KF, NIS covariance consistency | ✅ Complete |
| 6 | GNC — LQR & MPC | Discrete LQR via DARE, receding-horizon MPC, approach-cone constraint | ✅ Complete |
| 7 | Closed-Loop Integration | Full pipeline, Monte Carlo, animated rendezvous GIF | ✅ Complete |
| GPU | Vispy GPU Renderer | Blinn-Phong shading, star field, Earth limb, HiDPI-aware offscreen render | ✅ Complete |

**Total: 129 / 129 tests passing** (105 core + 24 GPU renderer)

---

## Quick Start

```bash
pip install -r requirements.txt

# Run all 105 core tests
python -m pytest tests/test_phase1.py tests/test_phase2.py tests/test_phase3.py \
                 tests/test_phase4.py tests/test_phase5.py tests/test_phase6.py \
                 tests/test_phase7.py -v

# Run GPU renderer tests (requires vispy)
python -m pytest tests/test_vispy.py -v

# Run all 129 tests
python -m pytest tests/ -v

# Generate all validation figures (fig1–fig34 + GIF)
python notebooks/phase1_demo.py
python notebooks/phase2_demo.py
python notebooks/phase3_demo.py
python notebooks/phase4_demo.py
python notebooks/phase5_demo.py
python notebooks/phase6_demo.py
python notebooks/phase7_demo.py

# Generate GPU renderer showcase figures
python notebooks/vispy_demo.py
```

---

## Validation Results

### Core Phases (105 tests)

| Metric | Value | Target |
|--------|-------|--------|
| Periodic orbit closure (position) | 2.3 × 10⁻¹³ m | < 1 μm |
| Periodic orbit closure (velocity) | 1.4 × 10⁻¹⁷ m/s | < 1 nm/s |
| Analytical vs DOP853 max error | 25 nm | < 0.1 mm |
| J₂ differential disturbance (100 m offset) | 0.885 μm/s² | — |
| J₂ trajectory divergence over 5 orbits | 9.4 m | — |
| EPnP mean reprojection error (noiseless) | < 0.01 px | — |
| EKF position RMSE convergence | < 0.5 m | — |
| LQR final range | < 0.01 m | — |
| LQR total Δv (20 m approach) | ~0.08 m/s | — |
| Full pipeline docking success | Yes | Yes |
| Core tests passed | **105 / 105** | 105 / 105 |

### GPU Renderer (24 tests)

| Feature | Detail |
|---------|--------|
| Shading | Blinn-Phong (ambient 0.18, diffuse 0.82, specular 0.40, shininess 48) |
| Star field | 2 000 points, reproducible from seed |
| Earth limb | 3 concentric translucent line strips |
| HiDPI support | macOS Retina 2× stride subsampling |
| Backend | osmesa / egl on Linux; auto-detect on macOS (vispy ≥ 0.14) |
| GPU tests passed | **24 / 24** | 24 / 24 |

---

## Key Equations

**HCW relative motion (LVLH frame):**
```
x'' - 2n·y' - 3n²x = fx
y'' + 2n·x'         = fy
z'' + n²z           = fz
```

**J₂ perturbing acceleration (ECI):**
```
a_J2 = (3μJ₂Re²)/(2r⁵) · [x(5z²/r²-1), y(5z²/r²-1), z(5z²/r²-3)]
```

**Torque-free Euler dynamics:**
```
J·ω' = -ω × (J·ω)
q'   = ½ · q ⊗ [0, ωx, ωy, ωz]ᵀ
```

**EPnP reprojection:**
```
[u, v, 1]ᵀ ~ K · (R·X + t)
```

**Discrete LQR (DARE):**
```
P  = AᵀPA - AᵀPB(BᵀPB + R)⁻¹BᵀPA + Q
K  = (BᵀPB + R)⁻¹BᵀPA
u* = -K·x
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| numpy | Linear algebra, state vectors |
| scipy | DOP853 integrator, DARE solver, SLSQP optimizer |
| matplotlib | All output figures and animated GIF |
| pytest | 129 unit tests |
| vispy ≥ 0.14 | GPU renderer (optional) |

Install: `pip install -r requirements.txt`

---

## Author

**Bishwaswarup Nayak**  
OUSUMS Lab, Department of Physics  
Indian Institute of Science, Bengaluru 560012, India

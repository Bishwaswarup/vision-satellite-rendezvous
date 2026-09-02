# Vision-Based Satellite Rendezvous Simulator

A closed-loop GNC simulation of autonomous rendezvous with a tumbling, non-cooperative
target, built from first principles in Python: relative orbital dynamics, torque-free
attitude propagation, a synthetic monocular camera, PnP pose estimation, multiplicative
Kalman filtering, and constrained optimal control.

**Bishwaswarup Nayak** — Indian Institute of Science, Bengaluru 560012, India

---

## Setup

```bash
git clone <this repo>
cd vision-satellite-rendezvous
bash setup.sh                    # creates .venv/ and installs everything
source .venv/bin/activate
python main.py check
```

`setup.sh` picks a Python 3.10+ interpreter, creates `.venv/` **inside the
repository**, installs the pinned requirements, and then verifies the install by
importing each package. Options: `--gpu` adds the optional vispy renderer, `--dev`
adds OpenCV (used only as a reference by a few tests), `--recreate` starts clean.

### PyCharm

PyCharm looks for a virtual environment in the **project root**. If it reports no
interpreter:

1. **Settings → Project: vision-satellite-rendezvous → Python Interpreter**
2. Gear icon → **Add Local Interpreter… → Existing**
3. Interpreter: `<repo>/.venv/bin/python`

Run `bash setup.sh` first — PyCharm cannot use an environment that does not exist
yet, and it will not install the dependencies for you.

Two things to know if you inherited an older checkout:

- A virtualenv one level **above** the repo (`VISION-BASED/.venv`) is not detected,
  because PyCharm only auto-detects `.venv` in the project root. It also contains
  nothing but `pip` and `setuptools`, so it would fail on the first `import numpy`
  even if it were found. It is unused and safe to delete.
- A virtual environment is **never** portable. Its `pyvenv.cfg` hardcodes absolute
  paths and its `site-packages` holds wheels compiled for one OS and CPU
  architecture, so it cannot be copied between machines or committed to git — hence
  the `.gitignore` entry. Always recreate it locally with `setup.sh`.

`vispy`, `PyOpenGL` and `opencv-python-headless` are optional; the tests that need
them skip when they are absent.

## Quick start

```bash
python main.py check        # 10-second smoke test of every subsystem
python main.py test         # 141 tests (a few skip without the extras)
python main.py figures      # experiments A-F  ->  outputs/*.png
python main.py animate      # closed-loop GIF + summary sheet
python main.py all          # everything, in order
```

Every command takes `--help`. `main.py figures --only D,E` runs a subset;
`main.py animate --no-vision --full-run` renders the estimator-isolation case.

All figures are written to `outputs/` in **print-safe monochrome** — series are
separated by grey level, line style and marker rather than by hue, so they survive a
greyscale journal printer and read correctly with a colour-vision deficiency. The
style lives in `viz/style.py`: call `apply_style()` once, then `series_kw(i)` per
series.

---

## What this actually simulates

Being precise about this matters, because it is easy to over-claim.

**The measurement chain is analytic, not photometric.** Each step projects 15 known
body-frame keypoints of a parametric Ariane 44L upper stage through a pinhole camera,
applies the visibility gate (in front of the camera, inside the image), adds Gaussian
pixel noise, and solves for pose with RANSAC + EPnP + Gauss-Newton refinement.
Correspondences are known by construction. **No image is rendered in the control
loop**, there is no feature detector, no data association, and no occlusion reasoning.

The renderer in `vision/renderer.py` is real and produces the camera views used in the
animation and in the dataset generator — it is simply not in the estimation path.

**What is modelled:** Hill-Clohessy-Wiltshire relative translation, torque-free Euler
attitude with quaternion kinematics, a calibrated pinhole camera with Brown-Conrady
distortion, keypoint visibility, pixel noise, pose-solver failure, an unmodelled
disturbance acceleration on the truth, thrust saturation, and an approach corridor.

**What is not:** spacecraft mass or thruster dynamics (control is a commanded
acceleration), J2 in the relative dynamics (the model exists and is validated, but no
reported result uses it), sensor radiometry, eclipse, or actuator lag.

---

## Layout

```
main.py                  single entry point - test / figures / animate / check
setup.sh                 creates and populates .venv/
requirements.txt         core dependencies (optional extras are commented)
viz/style.py             monochrome figure style shared by every plot

dynamics/
  constants.py           physical constants (mu, J2, R_E, LEO defaults)
  hcw.py                 HCW propagator: analytical STM + DOP853 reference
  j2_perturb.py          J2 acceleration, differential J2, ECI chief propagator
  ya_stm.py              Yamanaka-Ankersen STM - NOT VALIDATED, see below

target/
  attitude.py            torque-free Euler dynamics, inertia presets
  quaternion.py          quaternion algebra (scalar-first, Hamilton, active)

vision/
  camera.py              pinhole model, distortion and its inverse
  body_model.py          Ariane 44L and 3U CubeSat wireframe models
  renderer.py            CPU wireframe/solid renderer with z-buffer
  dataset.py             labelled synthetic dataset generator
  vispy_renderer.py      optional GPU renderer (Blinn-Phong, star field)

pose/
  epnp.py                EPnP (Lepetit 2009), including the planar case
  ransac.py              RANSAC with adaptive termination and local optimisation
  refine.py              Levenberg-Marquardt refinement on the SO(3) manifold

estimator/
  state.py               state packing, RK4, measurement model, NIS statistics
  ekf.py                 multiplicative EKF
  ukf.py                 unscented KF (van der Merwe sigma points)

controller/
  lqr.py                 infinite-horizon LQR via the DARE
  mpc.py                 receding-horizon MPC with a soft second-order approach cone

simulation/
  runner.py              closed-loop simulator
  video.py               animation and summary-sheet export

experiments/             experiment_A ... experiment_F - the paper's figures
tests/                   141 unit tests
notebooks/phase7_demo.py Phase-7 walkthrough figures
```

---

## Validation

Numbers below are reproduced by `python main.py test` and the experiment scripts.
Where a component has an independent reference implementation, it is checked against
that rather than against itself.

| Component | Check | Result |
|---|---|---|
| HCW analytical STM | vs DOP853 (rtol 1e-13), 5 orbits, drifting IC | max error 6e-11 m |
| HCW STM | det, Phi(0) = I, dPhi/dt = A Phi | 1.000000000000, 0.0, 4e-9 |
| J2 acceleration | vs grad R by central differences, off-equatorial | 7e-12 m/s^2 |
| J2 chief propagator | total energy and L_z conserved, 3 orbits | 3.9e-11 |
| Quaternion library | vs `scipy.spatial.transform`, 200 random | 6.7e-16 |
| Torque-free attitude | inertial **H** *vector* constant, 600 s | 2.6e-10 |
| EPnP | vs `cv2.SOLVEPNP_EPNP`, non-planar, sigma = 1 px | within 1.2x |
| EPnP planar | coplanar point sets, noiseless | 0.0000 deg |
| LM refinement | vs `cv2.solvePnPRefineLM` | identical to 4 dp |
| LM Jacobian | vs finite differences | 5e-9 relative |
| MEKF Jacobian | analytic vs finite differences | 1.4e-8 |
| **MEKF consistency** | **mean NIS over 3600 updates (n_z = 6)** | **6.044, CI [5.887, 6.113]** |
| LQR | DARE residual; J(u) = x0' P x0 | 2e-14; exact |
| MPC | unconstrained MPC vs infinite-horizon LQR | 4.8e-7 |
| MPC cone | constraint Jacobian vs finite differences | 1e-9 |

### Closed-loop performance

Default configuration: 30 m initial range, 1.5 px keypoint noise, tumbling target,
0.3 m/s^2 thrust limit, 1 s control interval.

| Metric | Value |
|---|---|
| Docking | step 42 (range < 1 m, speed < 0.05 m/s) |
| Delta-v | 2.61 m/s |
| Peak closing speed | 1.36 m/s |
| Thrust-saturated steps | 0 |
| Navigation RMSE | 0.33 m |
| Vision dropout | 35.7 % of steps |

### The dropout result

The most interesting behaviour in the simulator is that **measurement availability,
not pose accuracy, is what limits the loop.** Pose error degrades gracefully with
pixel noise; the solver's *failure rate* does not, because it is dominated by
geometry:

| Range | Mean visible keypoints | Pose-solve failure |
|---|---|---|
| > 12 m | 15.0 | 0 % |
| 6-12 m | 13.1 | 3 % |
| 3-6 m | 8.4 | 33 % |
| 1-3 m | 3.6 | 77 % |
| < 1 m | 1.6 | 99 % |

The knee sits at 6 m, and an 8 m target fills a 1024 px frame at f L / W =
800 x 8 / 1024 = **6.25 m**. The measured breakdown matches the closed-form
prediction. With a 65 degree field of view you cannot see an Ariane upper stage inside
about 6 m; terminal approach needs a second sensor, a wider lens, or a close-range
keypoint subset.

---

## Known limitations

Stated plainly, because a simulator's credibility rests on what it admits.

- **No rendered imagery in the loop** (see above). The pose problem the filter solves
  is easier than the real one: perfect correspondence, no occlusion, no detector.
- **`dynamics/ya_stm.py` is not validated.** Its in-plane block does not satisfy the
  Tschauner-Hempel equations at any eccentricity, and it does not reduce to HCW as
  e -> 0 (det Phi = 0.73 at e = 0.05, where it must be exactly 1). Nothing imports it
  and no result depends on it; it is retained with a warning rather than silently
  deleted. Fix it against a TH reference before claiming eccentric-orbit capability.
- **RANSAC sees no outliers in the closed loop.** Correspondences are index-aligned,
  so end-to-end the outlier rejection has nothing to reject. Experiment C exercises it
  properly with injected outliers.
- **Truth and filter share the propagator**, differing only by the injected
  disturbance acceleration (`SimConfig.disturb_accel_std`). There is no structural
  model error. Set it to 0 to recover the fully idealised case.
- **A ~0.5 m radial undershoot** past the docking hold point remains. It is a
  transient, not a limit cycle; removing it wants a glideslope guidance law rather
  than a re-tuned LQR.
- **MPC is not in the closed loop** — the runner uses LQR. Experiment E compares the
  two open-loop, and the comparison is *MPC against a saturated LQR*: with shared
  weights and the LQR terminal cost an unconstrained MPC reproduces the LQR exactly,
  so the difference is anti-windup, not prediction.
- **Timings are on unpinned hardware.** Quote them with a machine description or not
  at all.

---

## Reproducibility

| Parameter | Value | Where |
|---|---|---|
| Reference orbit | 400 km circular, n = 1.1368e-3 rad/s | `estimator/state.py`, `controller/lqr.py` |
| Experiment A orbit | 500 km, n = 1.10678e-3 rad/s | `dynamics/constants.py` |
| Control interval | 1.0 s (closed loop), 10 s (Exp. D) | `SimConfig.dt` |
| Camera | 1024^2, f = 800 px, c = (512, 512), no distortion | `vision/camera.py` |
| Target inertia | diag(1176, 6988, 6988) kg m^2, x = symmetry axis | `estimator/state.py` |
| Keypoint noise | 1.5 px 1-sigma | `SimConfig.pixel_noise_std` |
| Disturbance | 5e-4 m/s^2 1-sigma per axis | `SimConfig.disturb_accel_std` |
| Seeds | 42 for A-E; 0...24 and 0...19 for F | experiment scripts |

Note the two reference orbits: Experiment A validates the dynamics at 500 km while
every other experiment runs at 400 km. Unify them before publication.

Software: numpy >= 1.26, scipy >= 1.12, matplotlib >= 3.8, pytest >= 8.0.
`vispy` and `PyOpenGL` are optional, used only by the GPU renderer and its tests,
which skip when vispy is absent. OpenCV is not a runtime dependency; a few tests use
it as a reference and skip without it.

---

## Testing

```bash
python main.py test              # all 141
python main.py test -k epnp      # a subset
```

Two or three tests skip on a minimal install: the ones that cross-check EPnP and the
LM refiner against OpenCV, and the GPU-renderer suite. `bash setup.sh --gpu --dev`
enables them.

The suite is written to be *sensitive*, not merely green. Tests assert against
independent references (`scipy.spatial.transform`, `cv2`) and against invariants that
a plausible bug would break — the inertial angular-momentum **vector** rather than its
norm, det Phi = 1, NIS against its chi-square interval, analytic Jacobians against
finite differences, and the approach corridor against the achieved trajectory rather
than the solver's own report. Each regression test was verified by reintroducing the
defect it guards and confirming that it fails.

---

## License and citation

Research code accompanying work in preparation. If you use it, please cite the
repository and open an issue describing what you needed — the interfaces are still
moving.

# Vision-Based Satellite Rendezvous & Debris Tracking Simulator

**Phase 1 — Orbital Dynamics Engine**

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
├── tests/
│   └── test_phase1.py     # 15 unit tests (pytest)
├── notebooks/
│   └── phase1_demo.py     # Validation demo — generates 4 plots
├── outputs/               # Generated figures
├── config/                # YAML config files (populated in later phases)
├── requirements.txt
└── README.md
```

## Phase Roadmap

| Phase | Module | Status |
|-------|--------|--------|
| 1 | Orbital Dynamics Engine | ✅ Complete |
| 2 | Target Kinematics & Tumbling | 🔲 Next |
| 3 | Synthetic Vision Pipeline | 🔲 |
| 4 | Pose Estimation (EPnP) | 🔲 |
| 5 | State Estimation — EKF & UKF | 🔲 |
| 6 | GNC — LQR & MPC | 🔲 |
| 7 | Closed-Loop Integration & Monte Carlo | 🔲 |

## Quick Start

```bash
pip install -r requirements.txt

# Run all unit tests
python -m pytest tests/test_phase1.py -v

# Generate validation plots
python notebooks/phase1_demo.py
```

## Phase 1 Validation Results

| Metric | Value | Target |
|--------|-------|--------|
| Periodic orbit closure (position) | 2.3 × 10⁻¹³ m | < 1 μm |
| Periodic orbit closure (velocity) | 1.4 × 10⁻¹⁷ m/s | < 1 nm/s |
| Analytical vs DOP853 max error | 25 nm | < 0.1 mm |
| J₂ differential disturbance (100 m offset) | 0.885 μm/s² | — |
| J₂ trajectory divergence over 5 orbits | 9.4 m | — |
| Tests passed | 15 / 15 | 15 / 15 |

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

## Author
Bishwaswarup Nayak

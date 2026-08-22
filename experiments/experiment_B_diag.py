"""
Experiment B — Diagnostic: EPnP chirality at σ=0 and transition σ
=================================================================
Fine-grained sweep: σ = 0, 1e-6, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.25, 0.5
Reports BOTH raw EPnP and EPnP+GN errors so we can pinpoint whether the
failure is in the closed-form solver or the refinement step.

Run from the project root:
    python experiments/experiment_B_diag.py
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
from vision import rendezvous_camera, ariane_model, look_at_rotation
from pose import solve_epnp
from pose.refine import refine_pose

RNG      = np.random.default_rng(42)
N_TRIALS = 200
SIGMAS   = [0.0, 1e-6, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.25, 0.5]

cam   = rendezvous_camera()
model = ariane_model()
K     = cam.K
kp3d  = model.keypoint_array

eye = np.array([0., 0., 20.])
R_gt, t_gt = look_at_rotation(eye, target=np.zeros(3), up=np.array([0., 1., 0.]))

P_cam    = (R_gt @ kp3d.T).T + t_gt
u_true   = K[0,0] * P_cam[:,0] / P_cam[:,2] + K[0,2]
v_true   = K[1,1] * P_cam[:,1] / P_cam[:,2] + K[1,2]
kp2d_gt  = np.stack([u_true, v_true], axis=1)

def t_err(t_est):
    return np.linalg.norm(t_est - t_gt)

def r_err_deg(R_est):
    c = np.clip((np.trace(R_est @ R_gt.T) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(c))

def reproj(R, t):
    Pc = (R @ kp3d.T).T + t
    u  = K[0,0]*Pc[:,0]/Pc[:,2] + K[0,2]
    v  = K[1,1]*Pc[:,1]/Pc[:,2] + K[1,2]
    return np.sqrt(((np.stack([u,v],1) - kp2d_gt)**2).sum(1)).mean()

header = (f"{'sigma':>10}  {'EPnP t[cm]':>12}  {'EPnP r[°]':>10}  "
          f"{'GN t[cm]':>10}  {'GN r[°]':>8}  {'GN reproj[px]':>14}  "
          f"{'ok/N':>6}")
print(header)
print('-' * len(header))

for sigma in SIGMAS:
    epnp_t, epnp_r = [], []
    gn_t,   gn_r   = [], []
    gn_rp          = []
    ok_count       = 0

    for _ in range(N_TRIALS):
        noise  = RNG.normal(0.0, sigma, kp2d_gt.shape) if sigma > 0 else np.zeros_like(kp2d_gt)
        kp2d_n = kp2d_gt + noise

        R_e, t_e, _, ok = solve_epnp(kp3d, kp2d_n, K)
        if not ok:
            continue
        ok_count += 1

        # raw EPnP
        epnp_t.append(t_err(t_e))
        epnp_r.append(r_err_deg(R_e))

        # after GN refinement
        R_gn, t_gn, _ = refine_pose(R_e, t_e, kp3d, kp2d_n, K)
        gn_t.append(t_err(t_gn))
        gn_r.append(r_err_deg(R_gn))
        gn_rp.append(reproj(R_gn, t_gn))

    def fmt(vals):
        return np.sqrt(np.mean(np.array(vals)**2)) if vals else float('nan')

    print(f"{sigma:>10.2e}  "
          f"{fmt(epnp_t)*100:>12.4f}  "
          f"{np.mean(epnp_r) if epnp_r else float('nan'):>10.4f}  "
          f"{fmt(gn_t)*100:>10.4f}  "
          f"{np.mean(gn_r) if gn_r else float('nan'):>8.4f}  "
          f"{np.mean(gn_rp) if gn_rp else float('nan'):>14.4f}  "
          f"{ok_count}/{N_TRIALS}")

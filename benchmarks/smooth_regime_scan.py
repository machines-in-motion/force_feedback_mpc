"""
Step 1 of the anchor-vs-state convergence comparison: find an operating point at which
the multi-contact OCP is actually solvable, i.e. where the friction-cone constraint stays
away from its nonsmooth apex.

Lowering the end-effector force target unloads the front feet less, so the predicted
normal forces stay away from lambda_N = 0. For each target we report whether the
force-state formulation reaches the tolerance and how close the prediction comes to the
apex, so the choice of regime is a measurement rather than an assertion.

Usage:  python smooth_regime_scan.py [reg_min]
"""
import json
import os
import sys
import time

import numpy as np
import mim_solvers

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anchor_vs_state_common as C  # noqa: E402

REG = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-3
OUT = "/home/skleff/Desktop/PUBLICATIONS/TRO-Soft/AURO/REVISION/SMOOTH_REGIME"
F_TARGETS = [20.0, 40.0, 60.0, 80.0]
SAMPLES = [80, 210, 330]
NF, N_FEET, MU, MAXIT = 15, 4, 0.75, 1000


def set_f_des(mpc, f):
    """end-effector force target, as the demo sets it (pushing along -x)"""
    des = np.array([-f, 0.0, 0.0])
    models = list(mpc.ocp.runningModels) + [mpc.ocp.terminalModel]
    for m in models:
        fc = getattr(m.differential, "forceCosts", None)
        if fc is not None:
            for c in fc.forceCosts:
                c.f_des = des
    return des


def predicted(sol):
    lam = np.array([np.array(x)[-NF:] for x in sol.xs])[1:, :N_FEET * 3]
    lam = lam.reshape(len(lam), N_FEET, 3)
    fn = lam[..., 2]
    margin = MU * fn - np.linalg.norm(lam[..., :2], axis=-1)
    return fn, margin


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    rows = []
    print(f"reg_min = {REG:.0e}, tolerance 1e-4, max {MAXIT} SQP iterations\n")
    print(f"{'f_des':>6s} {'OCP':>5s} {'KKT':>9s} {'conv':>5s} {'SQPit':>6s} {'cone viol [N]':>13s} "
          f"{'min lam_N [N]':>13s} {'nodes<1N':>9s} {'active cone':>11s} {'ms':>7s}")
    for f in F_TARGETS:
        for smp in SAMPLES:
            st = C.load_b_final_states([smp])[0]
            mpc = C.build_state_mpc()
            set_f_des(mpc, f)
            anchors = C.make_anchors(mpc, st["y0"])
            _, xs, us = C.make_guess(mpc, anchors, st["y0"], st["u0"],
                                     traj=(st["xs_traj"], st["us_traj"]))
            mpc.ocp.x0 = st["y0"]
            s = mim_solvers.SolverCSQP(mpc.ocp)
            for k, v in C.SOLVER.items():
                setattr(s, k, v)
            s.reg_min = REG
            t0 = time.perf_counter()
            C.capture_stdout(lambda: s.solve(list(xs), list(us), MAXIT))
            ms = (time.perf_counter() - t0) * 1e3
            fn, margin = predicted(s)
            viol = float(np.maximum(-margin, 0).max())
            conv = bool(s.KKT <= 1e-4)
            row = dict(f_des=f, sample=smp, kkt=float(s.KKT), converged=conv,
                       iters=int(s.iter), cone_viol=viol, min_fn=float(fn.min()),
                       nodes_below_1N=int((fn < 1.0).sum()), n_nodes=int(fn.size),
                       active_cone=int((margin < 1e-3).sum()), ms=ms,
                       capped=bool(s.iter >= MAXIT))
            rows.append(row)
            print(f"{f:6.0f} {smp:5d} {s.KKT:9.2e} {str(conv):>5s} {s.iter:6d} {viol:13.2e} "
                  f"{fn.min():13.3f} {row['nodes_below_1N']:4d}/{fn.size:<4d} "
                  f"{row['active_cone']:11d} {ms:7.0f}", flush=True)
    json.dump(rows, open(os.path.join(OUT, "smooth_regime_scan.json"), "w"), indent=1)

    print("\nsummary per force target (across the three OCPs):")
    print(f"{'f_des':>6s} {'converged':>10s} {'KKT max':>9s} {'min lam_N':>10s} {'nodes<1N':>9s}")
    for f in F_TARGETS:
        g = [r for r in rows if r["f_des"] == f]
        print(f"{f:6.0f} {sum(r['converged'] for r in g):4d}/{len(g):<5d} "
              f"{max(r['kkt'] for r in g):9.2e} {min(r['min_fn'] for r in g):10.3f} "
              f"{sum(r['nodes_below_1N'] for r in g):4d}/{sum(r['n_nodes'] for r in g):<4d}")
    print("\nwrote", os.path.join(OUT, "smooth_regime_scan.json"))

"""
Sensitivity of each transcription to the solver's Riccati regularization, at one smooth
operating point and one apex operating point.

The question is not which transcription is "better conditioned" in the abstract, but
whether the regularization that the SQP solver applies is able to recover a usable
solution near the nonsmooth Coulomb-cone apex, and for which transcription. Both are run
over the same grid of regularization floors, with everything else identical.

Metric note: KKT residuals of the two transcriptions are residuals of different nonlinear
programs; the maximum friction-cone violation, in newtons, is comparable across them and
is reported alongside.

Usage:  python reg_sweep.py
"""
import json
import os
import sys
import time

import numpy as np
import crocoddyl
import mim_solvers

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anchor_vs_state_common as C  # noqa: E402

OUT = "/home/skleff/Desktop/PUBLICATIONS/TRO-Soft/AURO/REVISION/SMOOTH_REGIME"
TARGETS = {"smooth (40 N)": 40.0, "apex (80 N)": 80.0}
REGS = [1e-9, 1e-6, 1e-3, 1e-2, 1e-1]
SAMPLES = [80, 210, 330]
NF, N_FEET, MU, MAXIT, TOL = 15, 4, 0.75, 1000, 1e-4


def set_f_des(mpc, f):
    des = np.array([-f, 0.0, 0.0])
    for m in list(mpc.ocp.runningModels) + [mpc.ocp.terminalModel]:
        fc = getattr(m.differential, "forceCosts", None)
        if fc is not None:
            for c in fc.forceCosts:
                c.f_des = des


def cone_stats(lams):
    lam = np.asarray(lams)[:, :N_FEET * 3].reshape(len(lams), N_FEET, 3)
    fn = lam[..., 2]
    margin = MU * fn - np.linalg.norm(lam[..., :2], axis=-1)
    return float(np.maximum(-margin, 0).max()), float(fn.min())


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    rows = []
    print(f"{'regime':14s} {'rho':>7s} {'OCP':>5s} {'form':>12s} {'KKT':>9s} {'conv':>5s} "
          f"{'it':>5s} {'cone viol':>10s} {'min lam_N':>10s} {'cap':>5s} {'ms':>8s}")
    for label, f in TARGETS.items():
        for reg in REGS:
            for smp in SAMPLES:
                st = C.load_b_final_states([smp])[0]
                mpc = C.build_state_mpc()
                set_f_des(mpc, f)
                anchors = C.make_anchors(mpc, st["y0"])
                xs_r, xs_s, us = C.make_guess(mpc, anchors, st["y0"], st["u0"],
                                              traj=(st["xs_traj"], st["us_traj"]))
                an = C.build_anchor_models(mpc, anchors)
                p_a = crocoddyl.ShootingProblem(np.array(st["y0"])[:-NF].copy(),
                                                an[:-1], an[-1])
                mpc.ocp.x0 = st["y0"]
                state = mpc.running_models[0].differential.state
                pdata = state.pinocchio.createData()
                for tag, prob, xs in (("force state", mpc.ocp, xs_s),
                                      ("anchor", p_a, xs_r)):
                    s = mim_solvers.SolverCSQP(prob)
                    for k, v in C.SOLVER.items():
                        setattr(s, k, v)
                    s.reg_min, s.termination_tolerance = reg, TOL
                    t0 = time.perf_counter()
                    C.capture_stdout(lambda: s.solve(list(xs), list(us), MAXIT))
                    ms = (time.perf_counter() - t0) * 1e3
                    lam = ([np.array(x)[-NF:] for x in s.xs][1:] if tag == "force state"
                           else [C.lam_of(anchors, state, pdata, np.array(x))
                                 for x in s.xs][1:])
                    cv, minfn = cone_stats(lam)
                    rows.append(dict(regime=label, f_des=f, reg=reg, sample=smp, form=tag,
                                     kkt=float(s.KKT), converged=bool(s.KKT <= TOL),
                                     iters=int(s.iter), cone_viol=cv, min_fn=minfn,
                                     capped=bool(s.iter >= MAXIT), ms=ms))
                    r = rows[-1]
                    print(f"{label:14s} {reg:7.0e} {smp:5d} {tag:>12s} {s.KKT:9.2e} "
                          f"{str(r['converged']):>5s} {s.iter:5d} {cv:10.2e} {minfn:10.3f} "
                          f"{str(r['capped']):>5s} {ms:8.0f}", flush=True)
                json.dump(rows, open(os.path.join(OUT, "reg_sweep.json"), "w"), indent=1)

    print("\nmedian over the three OCPs")
    print(f"{'regime':14s} {'rho':>7s} {'form':>12s} {'KKT':>9s} {'cone viol [N]':>13s} {'conv':>6s}")
    for label in TARGETS:
        for reg in REGS:
            for tag in ("force state", "anchor"):
                g = [r for r in rows if r["regime"] == label and r["reg"] == reg
                     and r["form"] == tag]
                print(f"{label:14s} {reg:7.0e} {tag:>12s} "
                      f"{np.median([r['kkt'] for r in g]):9.2e} "
                      f"{np.median([r['cone_viol'] for r in g]):13.2e} "
                      f"{sum(r['converged'] for r in g):3d}/{len(g):<2d}")
    print("\nwrote", os.path.join(OUT, "reg_sweep.json"))

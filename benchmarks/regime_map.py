"""
Regime map: how do the two transcriptions behave as the prediction approaches the
nonsmooth apex of the Coulomb cone?

The end-effector force target controls how much the front feet unload, hence how close
the active friction constraint comes to its apex (lambda_N = 0). We sweep it and record,
for both transcriptions under identical settings, the KKT residual reached, the friction
cone violation, the minimum predicted normal force (the regime coordinate), the number of
SQP iterations and the solver exit path.

Default regularization (reg_min = 1e-9): in the smooth part of the sweep no damping is
needed, and damping is not neutral (it pushes the anchor-update transcription onto a
different branch), so the comparison is made with the solver's own defaults.

Usage:  python regime_map.py [reg_min]
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

REG = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-9
OUT = "/home/skleff/Desktop/PUBLICATIONS/TRO-Soft/AURO/REVISION/SMOOTH_REGIME"
F_TARGETS = ([float(v) for v in sys.argv[2].split(",")] if len(sys.argv) > 2
             else [20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
SAMPLES = [80, 210, 330]
NF, N_FEET, MU, MAXIT, TOL = 15, 4, 0.75, 1000, 1e-4


def set_f_des(mpc, f):
    des = np.array([-f, 0.0, 0.0])
    for m in list(mpc.ocp.runningModels) + [mpc.ocp.terminalModel]:
        fc = getattr(m.differential, "forceCosts", None)
        if fc is not None:
            for c in fc.forceCosts:
                c.f_des = des


def steps(log):
    out = []
    for line in log.splitlines():
        m = C.ROW.match(line)
        if m:
            try:
                out.append(float(m.group(7)))
            except ValueError:
                pass
    return out


def cone_stats(lams):
    lam = np.asarray(lams)[:, :N_FEET * 3].reshape(len(lams), N_FEET, 3)
    fn = lam[..., 2]
    margin = MU * fn - np.linalg.norm(lam[..., :2], axis=-1)
    return float(np.maximum(-margin, 0).max()), float(fn.min()), int((margin < 1e-3).sum())


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    rows = []
    print(f"reg_min={REG:.0e}  tol={TOL:.0e}  max {MAXIT} iterations\n")
    print(f"{'f_des':>6s} {'OCP':>5s} {'form':>12s} {'KKT':>9s} {'conv':>5s} {'it':>5s} "
          f"{'cone viol':>10s} {'min lam_N':>10s} {'active':>7s} {'bt':>4s} {'exit':>12s} {'ms':>8s}")
    for f in F_TARGETS:
        for smp in SAMPLES:
            st = C.load_b_final_states([smp])[0]
            mpc = C.build_state_mpc()
            set_f_des(mpc, f)
            anchors = C.make_anchors(mpc, st["y0"])
            xs_r, xs_s, us = C.make_guess(mpc, anchors, st["y0"], st["u0"],
                                          traj=(st["xs_traj"], st["us_traj"]))
            an = C.build_anchor_models(mpc, anchors)
            p_a = crocoddyl.ShootingProblem(np.array(st["y0"])[:-NF].copy(), an[:-1], an[-1])
            mpc.ocp.x0 = st["y0"]
            state = mpc.running_models[0].differential.state
            pdata = state.pinocchio.createData()
            for tag, prob, xs in (("force state", mpc.ocp, xs_s), ("anchor", p_a, xs_r)):
                s = mim_solvers.SolverCSQP(prob)
                for k, v in C.SOLVER.items():
                    setattr(s, k, v)
                s.reg_min, s.termination_tolerance = REG, TOL
                s.setCallbacks([mim_solvers.CallbackVerbose()])
                t0 = time.perf_counter()
                _, log = C.capture_stdout(lambda: s.solve(list(xs), list(us), MAXIT))
                ms = (time.perf_counter() - t0) * 1e3
                lam = ([np.array(x)[-NF:] for x in s.xs][1:] if tag == "force state"
                       else [C.lam_of(anchors, state, pdata, np.array(x)) for x in s.xs][1:])
                cv, minfn, act = cone_stats(lam)
                stp = steps(log)
                conv = bool(s.KKT <= TOL)
                capped = bool(s.iter >= MAXIT)
                exit_ = ("tolerance" if conv else "iter cap" if capped
                         else "reg max" if s.preg >= s.reg_max else "line search")
                row = dict(f_des=f, sample=smp, form=tag, reg=REG, kkt=float(s.KKT),
                           converged=conv, iters=int(s.iter), cone_viol=cv, min_fn=minfn,
                           active=act, backtracks=int(sum(1 for x in stp if x < 1.0)),
                           exit=exit_, ms=ms, cost=float(s.cost))
                rows.append(row)
                print(f"{f:6.0f} {smp:5d} {tag:>12s} {s.KKT:9.2e} {str(conv):>5s} {s.iter:5d} "
                      f"{cv:10.2e} {minfn:10.3f} {act:7d} {row['backtracks']:4d} "
                      f"{exit_:>12s} {ms:8.0f}", flush=True)
            json.dump(rows, open(os.path.join(OUT, f"regime_map_reg{REG:.0e}.json"), "w"), indent=1)

    print("\nmedian over the three OCPs")
    print(f"{'f_des':>6s} {'form':>12s} {'conv':>7s} {'KKT':>9s} {'cone viol':>10s} "
          f"{'min lam_N':>10s} {'it':>5s}")
    for f in F_TARGETS:
        for tag in ("force state", "anchor"):
            g = [r for r in rows if r["f_des"] == f and r["form"] == tag]
            print(f"{f:6.0f} {tag:>12s} {sum(r['converged'] for r in g):3d}/{len(g):<3d} "
                  f"{np.median([r['kkt'] for r in g]):9.2e} "
                  f"{np.median([r['cone_viol'] for r in g]):10.2e} "
                  f"{np.median([r['min_fn'] for r in g]):10.3f} "
                  f"{np.median([r['iters'] for r in g]):5.0f}")
    print("\nwrote", os.path.join(OUT, f"regime_map_reg{REG:.0e}.json"))

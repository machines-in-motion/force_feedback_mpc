"""
Step 2: convergence comparison of the anchor-update and force-state transcriptions on a
SMOOTH instance of the multi-contact OCP, i.e. an end-effector force target at which the
friction cone is active but the predicted normal forces stay away from the cone apex.

Unlike the comparison at the nominal target, both formulations can reach the requested
tolerance here, so "iterations (and time) to reach KKT <= tol" is well defined and the
comparison does not depend on where each solve happens to stagnate.

Usage:  python anchor_vs_state_smooth.py [f_des]
"""
import json
import os
import sys
import time

import numpy as np
import crocoddyl
import mim_solvers
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anchor_vs_state_common as C  # noqa: E402

F_DES = float(sys.argv[1]) if len(sys.argv) > 1 else 40.0
OUT = "/home/skleff/Desktop/PUBLICATIONS/TRO-Soft/AURO/REVISION/SMOOTH_REGIME"
SAMPLES = [80, 210, 330]
REGS = [1e-9, 1e-3]
NF, N_FEET, MU, MAXIT, TOL = 15, 4, 0.75, 2000, 1e-4
C_STATE, C_ANCHOR = "#1b6ca8", "#c1492b"

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True, "legend.frameon": False,
                     "figure.dpi": 150})


def set_f_des(mpc, f):
    des = np.array([-f, 0.0, 0.0])
    for m in list(mpc.ocp.runningModels) + [mpc.ocp.terminalModel]:
        fc = getattr(m.differential, "forceCosts", None)
        if fc is not None:
            for c in fc.forceCosts:
                c.f_des = des


def traces(log):
    kkt = []
    for line in log.splitlines():
        m = C.ROW.match(line)
        if m:
            try:
                kkt.append(float(m.group(8)))
            except ValueError:
                pass
    return kkt


def cone_stats(lams):
    lam = np.asarray(lams)[:, :N_FEET * 3].reshape(len(lams), N_FEET, 3)
    fn = lam[..., 2]
    margin = MU * fn - np.linalg.norm(lam[..., :2], axis=-1)
    return float(np.maximum(-margin, 0).max()), float(fn.min()), int((margin < 1e-3).sum())


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, tr = [], {}
    print(f"f_des = {F_DES:.0f} N, tolerance {TOL:.0e}, max {MAXIT} SQP iterations\n")
    print(f"{'reg':>6s} {'OCP':>5s} {'form':>12s} {'KKT':>9s} {'conv':>5s} {'it':>5s} "
          f"{'ms':>8s} {'ms/it':>7s} {'cone viol':>10s} {'min lam_N':>10s} {'active':>7s} {'cost':>8s}")
    for reg in REGS:
        for smp in SAMPLES:
            st = C.load_b_final_states([smp])[0]
            mpc = C.build_state_mpc()
            set_f_des(mpc, F_DES)
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
                s.reg_min, s.termination_tolerance = reg, TOL
                s.setCallbacks([mim_solvers.CallbackVerbose()])
                t0 = time.perf_counter()
                _, log = C.capture_stdout(lambda: s.solve(list(xs), list(us), MAXIT))
                ms = (time.perf_counter() - t0) * 1e3
                lam = ([np.array(x)[-NF:] for x in s.xs][1:] if tag == "force state"
                       else [C.lam_of(anchors, state, pdata, np.array(x)) for x in s.xs][1:])
                cv, minfn, act = cone_stats(lam)
                k = traces(log)
                tr[(reg, smp, tag)] = k
                row = dict(reg=reg, sample=smp, form=tag, kkt=float(s.KKT),
                           converged=bool(s.KKT <= TOL), iters=int(s.iter), ms=ms,
                           ms_per_it=ms / max(s.iter, 1), cone_viol=cv, min_fn=minfn,
                           active=act, cost=float(s.cost), capped=bool(s.iter >= MAXIT))
                rows.append(row)
                print(f"{reg:6.0e} {smp:5d} {tag:>12s} {s.KKT:9.2e} "
                      f"{str(row['converged']):>5s} {s.iter:5d} {ms:8.0f} {row['ms_per_it']:7.1f} "
                      f"{cv:10.2e} {minfn:10.3f} {act:7d} {s.cost:8.4f}", flush=True)

    json.dump(rows, open(os.path.join(OUT, f"smooth_cmp_f{F_DES:.0f}.json"), "w"), indent=1)

    # ----------------------------------------------------------------- figure
    fig, ax = plt.subplots(1, len(REGS), figsize=(7.0, 2.8), sharey=True)
    for a, reg in zip(np.atleast_1d(ax), REGS):
        for tag, c in (("force state", C_STATE), ("anchor", C_ANCHOR)):
            for i, smp in enumerate(SAMPLES):
                k = tr[(reg, smp, tag)]
                a.semilogy(np.arange(1, len(k) + 1), k, color=c, lw=1.1,
                           alpha=0.9 if i == 0 else 0.45,
                           label=(tag if i == 0 else None))
        a.axhline(TOL, color="0.35", ls=":", lw=1)
        a.set_xscale("log")
        a.set_xlabel("SQP iteration")
        a.set_title(rf"$\rho=10^{{{int(np.log10(reg))}}}$", fontsize=9, loc="left")
    np.atleast_1d(ax)[0].set_ylabel("KKT residual")
    np.atleast_1d(ax)[0].legend(loc="lower left", fontsize=8)
    fig.suptitle(f"smooth regime: end-effector force target {F_DES:.0f} N", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_smooth_convergence_f{F_DES:.0f}.{ext}"))

    print("\nsummary (median over the three OCPs)")
    print(f"{'reg':>6s} {'form':>12s} {'converged':>10s} {'it':>6s} {'ms':>8s} {'ms/it':>7s}")
    for reg in REGS:
        for tag in ("force state", "anchor"):
            g = [r for r in rows if r["reg"] == reg and r["form"] == tag]
            print(f"{reg:6.0e} {tag:>12s} {sum(r['converged'] for r in g):4d}/{len(g):<5d} "
                  f"{np.median([r['iters'] for r in g]):6.0f} "
                  f"{np.median([r['ms'] for r in g]):8.0f} "
                  f"{np.median([r['ms_per_it'] for r in g]):7.1f}")
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()

"""
Is the residual friction-cone violation of the force-feedback OCP concentrated at the
nonsmooth apex of the Coulomb cone (lambda_N -> 0)?

Corrected force-cost Hessian, reg_min as used in the closed-loop experiment. Reports
(i) where the violation sits relative to the normal force, and (ii) the counterfactual
in which a small positive lower bound lambda_N >= eps is imposed on the decision nodes.

Usage:  python apex_check.py [reg_min] [eps]
"""
import os
import sys

import numpy as np
import mim_solvers

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anchor_vs_state_common as C  # noqa: E402

REG = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-3
EPS = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
NF, N_FEET, MU = 15, 4, 0.75
SAMPLES = [80, 210, 330]


def solve(problem, xs, us, reg, eps=None):
    if eps is not None:                       # lambda_N >= eps on the decision nodes
        for m in list(problem.runningModels):
            if m.ng:
                lb = np.array(m.g_lb)
                lb[1::2] = eps                # rows are [cone, unilaterality] per foot
                m.g_lb = lb
    s = mim_solvers.SolverCSQP(problem)
    for k, v in C.SOLVER.items():
        setattr(s, k, v)
    s.reg_min = reg
    C.capture_stdout(lambda: s.solve(list(xs), list(us), C.MAXIT))
    return s


def viol_field(s):
    """per (node, foot) cone violation and normal force of the returned solution"""
    lam = np.array([np.array(x)[-NF:] for x in s.xs])[1:, :N_FEET * 3]
    lam = lam.reshape(len(lam), N_FEET, 3)
    cone = np.maximum(np.linalg.norm(lam[..., :2], axis=-1) - MU * lam[..., 2], 0.0)
    return cone, lam[..., 2]


if __name__ == "__main__":
    print(f"reg_min = {REG:.0e},  apex counterfactual eps = {EPS} N\n")
    print(f"{'OCP':>5s} {'case':>12s} {'KKT':>9s} {'max cone [N]':>12s} {'lam_N at argmax':>15s} "
          f"{'min lam_N':>10s} {'SQP it':>6s}")
    allv, alln = [], []
    for st in C.load_b_final_states(SAMPLES):
        mpc = C.build_state_mpc()
        anchors = C.make_anchors(mpc, st["y0"])
        _, xs_s, us = C.make_guess(mpc, anchors, st["y0"], st["u0"],
                                   traj=(st["xs_traj"], st["us_traj"]))
        mpc.ocp.x0 = st["y0"]
        s = solve(mpc.ocp, xs_s, us, REG)
        cone, fn = viol_field(s)
        i, j = np.unravel_index(np.argmax(cone), cone.shape)
        allv.append(cone.ravel()); alln.append(fn.ravel())
        print(f"{st['sample']:5d} {'baseline':>12s} {s.KKT:9.2e} {cone.max():12.2e} "
              f"{fn[i, j]:15.3f} {fn.min():10.3f} {s.iter:6d}")
        # counterfactual on a freshly built problem so the bounds change nothing else
        mpc2 = C.build_state_mpc()
        anchors2 = C.make_anchors(mpc2, st["y0"])
        _, xs2, us2 = C.make_guess(mpc2, anchors2, st["y0"], st["u0"],
                                   traj=(st["xs_traj"], st["us_traj"]))
        mpc2.ocp.x0 = st["y0"]
        s2 = solve(mpc2.ocp, xs2, us2, REG, eps=EPS)
        cone2, fn2 = viol_field(s2)
        i2, j2 = np.unravel_index(np.argmax(cone2), cone2.shape)
        print(f"{st['sample']:5d} {'lam_N>=eps':>12s} {s2.KKT:9.2e} {cone2.max():12.2e} "
              f"{fn2[i2, j2]:15.3f} {fn2.min():10.3f} {s2.iter:6d}")

    v = np.concatenate(allv); n = np.concatenate(alln)
    print("\nwhere does the violation sit? (baseline solutions, all nodes x feet)")
    act = v > 1e-9
    print(f"  nodes with violation > 1e-9 N : {int(act.sum())} of {len(v)}")
    if act.any():
        print(f"  their normal force: median {np.median(n[act]):.3f} N, "
              f"max {n[act].max():.3f} N, min {n[act].min():.3f} N")
        print(f"  normal force elsewhere    : median {np.median(n[~act]):.3f} N")
    for thr in (1.0, 5.0, 20.0):
        m = n < thr
        if m.any():
            print(f"  lambda_N < {thr:5.1f} N : {int(m.sum()):4d} nodes, "
                  f"max violation {v[m].max():.2e} N")
        if (~m).any():
            print(f"  lambda_N >= {thr:5.1f} N: {int((~m).sum()):4d} nodes, "
                  f"max violation {v[~m].max():.2e} N")

"""
Analytical vs numerical derivatives of the Go2+arm soft-contact model: timing over
RANDOMIZED operating points, and the resulting speedup on a full OCP solve.

  * per-node timing at K operating points sampled along the closed-loop experiment
    (each perturbed), R repetitions each; we report the mean and standard deviation
    of the per-point medians, so the spread reflects the operating point rather than
    a single noisy measurement;
  * the same for one full horizon of N nodes;
  * a full OCP solve with the analytical models and with every model wrapped in
    crocoddyl's ActionModelNumDiff, from the same warm start with the same solver
    settings. Total solve time conflates derivative cost with the iteration count,
    so the time per SQP iteration is reported alongside it.

Usage:
    python deriv_timing_go2.py [out_dir]
"""
import glob
import json
import os
import sys
import time

import numpy as np
import crocoddyl
import mim_solvers

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anchor_vs_state_common as C  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/skleff/Desktop/PUBLICATIONS/TRO-Soft/AURO/REVISION/DERIVATIVES"
RUN = ("/home/skleff/Desktop/PUBLICATIONS/TRO-Soft/AURO/REVISION/GO2_RERUN/"
       "C_hessfix_reg1em3/go2_soft*.npz")
K_POINTS, REPS, WARMUP = 12, 25, 5
H_FWD, H_CEN = 1e-7, 1e-5
REG_MIN, SQP_TOL, MAXIT_SOLVE = 1e-3, 1e-4, 10   # same cap both sides:
# a NumDiff solve to convergence would take hours, and the comparable quantity
# is the cost per SQP iteration, which a fixed iteration budget isolates
SEED = 0


def timeit(fn, reps=REPS, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    t = np.empty(reps)
    for i in range(reps):
        t0 = time.perf_counter_ns()
        fn()
        t[i] = (time.perf_counter_ns() - t0) * 1e-6      # ms
    return float(np.median(t))


def fd_first_order(iam, dp, dm, x, u, h, central=True):
    """manifold-aware finite differences of the first-order quantities only"""
    st = iam.state
    ndx, nu = st.ndx, iam.nu
    Fx = np.zeros((ndx, ndx))
    Lx = np.zeros(ndx)
    if not central:
        iam.calc(dm, x, u)
    dx = np.zeros(ndx)
    den = 2.0 * h if central else h
    for i in range(ndx):
        dx[i] = h
        iam.calc(dp, st.integrate(x, dx), u)
        if central:
            iam.calc(dm, st.integrate(x, -dx), u)
        Fx[:, i] = st.diff(np.array(dm.xnext), np.array(dp.xnext)) / den
        Lx[i] = (dp.cost - dm.cost) / den
        dx[i] = 0.0
    Fu = np.zeros((ndx, nu))
    du = np.zeros(nu)
    for j in range(nu):
        du[j] = h
        iam.calc(dp, x, u + du)
        if central:
            iam.calc(dm, x, u - du)
        Fu[:, j] = st.diff(np.array(dm.xnext), np.array(dp.xnext)) / den
        du[j] = 0.0
    return Fx, Fu, Lx


def sample_points(mpc, k, rng):
    """operating points along the reported closed-loop run, randomly perturbed"""
    d = np.load(glob.glob(RUN)[0], allow_pickle=True)
    q = np.asarray(d["jointPos"], float)
    v = np.asarray(d["jointVel"], float)
    tau = np.asarray(d["joint_torques"], float)
    ocp_f = np.asarray(d["ocp_forces"], float)          # (cycles, nodes, ee, 3)
    st = mpc.running_models[0].state
    nv = mpc.running_models[0].differential.state.nv
    pts = []
    idx = rng.choice(len(q) - 1, size=k, replace=False)
    for i in idx:
        qi = q[i].copy()
        qi[3:7] /= np.linalg.norm(qi[3:7])
        f = ocp_f[min(i // 10, len(ocp_f) - 1), 1].reshape(-1)
        x = np.concatenate([qi, v[i], f])
        dx = np.zeros(st.ndx)
        dx[:2 * nv] = 0.01 * rng.standard_normal(2 * nv)
        pts.append((np.array(st.integrate(x, dx)),
                    tau[i] + 0.1 * rng.standard_normal(tau.shape[1])))
    return pts


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    mpc = C.build_state_mpc()
    iam = mpc.running_models[1]
    data, dp, dm = iam.createData(), iam.createData(), iam.createData()
    nd = crocoddyl.ActionModelNumDiff(iam)
    nd_data = nd.createData()
    pts = sample_points(mpc, K_POINTS, rng)
    print(f"model: ndx={iam.state.ndx} nu={iam.nu} ng={iam.ng};  "
          f"{K_POINTS} operating points x {REPS} repetitions")

    per = {"analytical": [], "forward FD": [], "central FD": [], "crocoddyl NumDiff": []}
    for n, (x, u) in enumerate(pts):
        per["analytical"].append(timeit(lambda: (iam.calc(data, x, u), iam.calcDiff(data, x, u))))
        per["forward FD"].append(timeit(lambda: fd_first_order(iam, dp, dm, x, u, H_FWD, False)))
        per["central FD"].append(timeit(lambda: fd_first_order(iam, dp, dm, x, u, H_CEN, True)))
        per["crocoddyl NumDiff"].append(
            timeit(lambda: (nd.calc(nd_data, x, u), nd.calcDiff(nd_data, x, u)),
                   reps=3, warmup=1))
        print(f"  point {n + 1:2d}/{K_POINTS}", end="\r", flush=True)
    print()

    json.dump({"per_node_ms": per}, open(os.path.join(OUT, "_partial.json"), "w"))
    res = {"per_node_ms": per, "n_points": K_POINTS, "reps": REPS,
           "ndx": int(iam.state.ndx), "nu": int(iam.nu)}
    base = np.array(per["analytical"])
    print(f"\n{'method':22s} {'mean [ms]':>10s} {'std':>8s} {'min':>8s} {'max':>8s} {'speedup':>9s}")
    for k, v in per.items():
        a = np.array(v)
        print(f"{k:22s} {a.mean():10.3f} {a.std():8.3f} {a.min():8.3f} {a.max():8.3f} "
              f"{a.mean() / base.mean():9.1f}x")

    # ------------------------------------------------ full horizon (derivatives only)
    models = list(mpc.ocp.runningModels)
    datas = [m.createData() for m in models]
    dps, dms = [m.createData() for m in models], [m.createData() for m in models]
    nds = [crocoddyl.ActionModelNumDiff(m) for m in models]
    nddatas = [m.createData() for m in nds]
    st0 = C.load_b_final_states([80])[0]
    anchors = C.make_anchors(mpc, st0["y0"])
    _, xs, us = C.make_guess(mpc, anchors, st0["y0"], st0["u0"],
                             traj=(st0["xs_traj"], st0["us_traj"]))
    hz = {}
    hz["analytical"] = timeit(lambda: [(m.calc(d, x, u), m.calcDiff(d, x, u))
                                       for m, d, x, u in zip(models, datas, xs, us)], reps=10)
    hz["forward FD"] = timeit(lambda: [fd_first_order(m, a, b, x, u, H_FWD, False)
                                       for m, a, b, x, u in zip(models, dps, dms, xs, us)], reps=5, warmup=2)
    hz["central FD"] = timeit(lambda: [fd_first_order(m, a, b, x, u, H_CEN, True)
                                       for m, a, b, x, u in zip(models, dps, dms, xs, us)], reps=5, warmup=2)
    hz["crocoddyl NumDiff"] = timeit(lambda: [(m.calc(d, x, u), m.calcDiff(d, x, u))
                                              for m, d, x, u in zip(nds, nddatas, xs, us)],
                                     reps=3, warmup=1)
    res["horizon_ms"] = hz
    print(f"\nfull horizon N={len(models)} (derivative evaluation only)")
    for k, v in hz.items():
        print(f"  {k:22s} {v:10.1f} ms   {v / hz['analytical']:6.1f}x")

    # ------------------------------------------------------------ full OCP solve
    # A solve with crocoddyl's NumDiff models segfaults inside CSQP (the wrapper does
    # not expose the constraint dimension), so the analytical solve is measured and the
    # numerical-derivative cost is projected by substituting the measured horizon
    # derivative time, keeping the rest of the SQP iteration unchanged.
    print("\nfull OCP solve (analytical models, same warm start as above)")
    mpc.ocp.x0 = st0["y0"]
    s_ = mim_solvers.SolverCSQP(mpc.ocp)
    for k, v in C.SOLVER.items():
        setattr(s_, k, v)
    s_.reg_min, s_.termination_tolerance = REG_MIN, SQP_TOL
    t0 = time.perf_counter()
    C.capture_stdout(lambda: s_.solve(list(xs), list(us), MAXIT_SOLVE))
    el = (time.perf_counter() - t0) * 1e3
    per_it = el / max(s_.iter, 1)
    overhead = per_it - hz["analytical"]          # QP + line search, derivative-agnostic
    res["solve"] = dict(ms=el, iters=int(s_.iter), ms_per_iter=per_it,
                        deriv_ms=hz["analytical"], other_ms=overhead, kkt=float(s_.KKT))
    print(f"  analytical: {el:.0f} ms, {s_.iter} it, {per_it:.1f} ms/it "
          f"(derivatives {hz['analytical']:.1f} ms + solver {overhead:.1f} ms), KKT {s_.KKT:.2e}")
    print("  projected cost per SQP iteration if derivatives were numerical:")
    for k, v in hz.items():
        if k == "analytical":
            continue
        proj = v + overhead
        res["solve"][f"proj_{k}"] = dict(ms_per_iter=proj, speedup=proj / per_it)
        print(f"    {k:22s} {proj:9.1f} ms/it   {proj / per_it:6.1f}x slower")

    json.dump(res, open(os.path.join(OUT, "deriv_timing_go2.json"), "w"), indent=1)
    print("\nwrote", os.path.join(OUT, "deriv_timing_go2.json"))


if __name__ == "__main__":
    main()

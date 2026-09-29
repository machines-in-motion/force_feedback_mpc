"""
Convergence-basin sweep, matched settings. Both formulations get the SAME perturbed
physical (q, v, u) guess; the lifted one takes its force guess from the anchor map along
that same perturbed trajectory, so both start from the same physical force prediction.
Thresholds fixed in advance: 1e-4, 1e-3, 1e-2. Primary cross-formulation metric is the
physical cone violation in newtons (KKT norms of the two NLPs are not commensurable).
"""
import sys, time, numpy as np, crocoddyl, mim_solvers
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anchor_vs_state_common as C

NF, MU = 15, 0.75
SAMPLES = [80, 210]
AMPS = [0.0, 0.01, 0.05, 0.2]
SEEDS = [0, 1, 2, 3, 4]
REGS = [1e-2, 1e-1]
THRESH = (1e-4, 1e-3, 1e-2)
MAXIT = 1000

def cone_viol(lams):
    c = u = 0.0
    for lam in lams:
        for k in range(4):
            f = np.asarray(lam[3*k:3*k+3], float)
            c = max(c, -(MU*f[2] - np.linalg.norm(f[:2]))); u = max(u, -f[2])
    return max(c, 0.0), max(u, 0.0)

def solve(problem, xs, us, rm):
    s = mim_solvers.SolverCSQP(problem)
    for k, v in C.SOLVER.items(): setattr(s, k, v)
    s.reg_min = rm
    t0 = time.perf_counter()
    C.capture_stdout(lambda: s.solve(list(xs), list(us), MAXIT))
    return s, (time.perf_counter() - t0) * 1e3

res = []
for st in C.load_b_final_states(SAMPLES):
    mpc = C.build_state_mpc()
    anchors = C.make_anchors(mpc, st["y0"])
    state = mpc.running_models[0].differential.state
    pdata = state.pinocchio.createData()
    an = C.build_anchor_models(mpc, anchors)
    p_a = crocoddyl.ShootingProblem(np.array(st["y0"])[:-NF].copy(), an[:-1], an[-1])
    mpc.ocp.x0 = st["y0"]
    for amp in AMPS:
        seeds = [0] if amp == 0.0 else SEEDS
        for seed in seeds:
            rng = np.random.default_rng(1000 * st["sample"] + seed)
            xs_r, xs_s, us = C.make_guess(mpc, anchors, st["y0"], st["u0"], rng=rng, amp=amp,
                                          traj=(st["xs_traj"], st["us_traj"]))
            for rm in REGS:
                for tag, prob, xs in (("lifted", mpc.ocp, xs_s), ("anchor", p_a, xs_r)):
                    s, ms = solve(prob, xs, us, rm)
                    lam = ([np.array(x)[-NF:] for x in s.xs] if tag == "lifted"
                           else [C.lam_of(anchors, state, pdata, np.array(x)) for x in s.xs])
                    cv, uv = cone_viol(lam)
                    res.append(dict(sample=st["sample"], amp=amp, seed=seed, rm=rm, form=tag,
                                    kkt=s.KKT, cstr=s.constraint_norm, cv=cv, uv=uv,
                                    it=s.iter, ms=ms, capped=int(s.iter >= MAXIT)))
            print(f"done {st['sample']} amp={amp} seed={seed}", flush=True)

import collections
print(f"\n{'reg':>6s} {'sample':>6s} {'amp':>5s} {'form':>7s} {'n':>3s} " +
      " ".join(f"{'<'+f'{t:.0e}':>9s}" for t in THRESH) +
      f" {'KKT med':>9s} {'coneN med':>9s} {'coneN max':>9s} {'it med':>6s} {'capped':>6s} {'ms med':>7s}")
for rm in REGS:
    for smp in SAMPLES:
        for amp in AMPS:
            for form in ("lifted", "anchor"):
                g = [r for r in res if r["rm"] == rm and r["sample"] == smp
                     and r["amp"] == amp and r["form"] == form]
                if not g: continue
                k = np.array([r["kkt"] for r in g]); cv = np.array([r["cv"] for r in g])
                it = np.array([r["it"] for r in g]); ms = np.array([r["ms"] for r in g])
                cap = sum(r["capped"] for r in g)
                fr = " ".join(f"{int((k <= t).sum()):4d}/{len(k):<4d}" for t in THRESH)
                print(f"{rm:6.0e} {smp:6d} {amp:5.2f} {form:>7s} {len(g):3d} {fr} "
                      f"{np.median(k):9.2e} {np.median(cv):9.4f} {cv.max():9.4f} "
                      f"{np.median(it):6.0f} {cap:6d} {np.median(ms):7.0f}")

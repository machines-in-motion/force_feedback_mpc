"""
Single consistent dataset for the Reviewer-1 Major-2 answer: both transcriptions, the
same three frozen OCPs, the same warm start, the same solver settings, at two matched
regularizations. Saves KKT traces, per-node cone violations, timings and dimensions.
"""
import sys, time, json, numpy as np, pinocchio as pin, crocoddyl, mim_solvers
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anchor_vs_state_common as C

OUT = "/home/skleff/Desktop/PUBLICATIONS/TRO-Soft/AURO/REVISION/ANCHOR_VS_STATE/"
NF, MU = 15, 0.75
SAMPLES, REGS, MAXIT = [80, 210, 330], [1e-3, 1e-1], 5000

def traces(log):
    kkt, cstr = [], []
    for line in log.splitlines():
        m = C.ROW.match(line)
        if m:
            try: cstr.append(float(m.group(5))); kkt.append(float(m.group(8)))
            except ValueError: pass
    return kkt, cstr

def per_node_viol(lams):
    """max over feet of the cone and unilaterality violation, per horizon node, in N"""
    cone, uni = [], []
    for lam in lams:
        c = u = 0.0
        for k in range(4):
            f = np.asarray(lam[3*k:3*k+3], float)
            c = max(c, -(MU*f[2] - np.linalg.norm(f[:2]))); u = max(u, -f[2])
        cone.append(max(c, 0.0)); uni.append(max(u, 0.0))
    return np.array(cone), np.array(uni)

data = {"runs": [], "jac": {}}
for st in C.load_b_final_states(SAMPLES):
    mpc = C.build_state_mpc()
    anchors = C.make_anchors(mpc, st["y0"])
    state = mpc.running_models[0].differential.state
    nq, nv = state.nq, state.nv
    pdata = state.pinocchio.createData()
    xs_r, xs_s, us = C.make_guess(mpc, anchors, st["y0"], st["u0"],
                                  traj=(st["xs_traj"], st["us_traj"]))
    an = C.build_anchor_models(mpc, anchors)
    p_a = crocoddyl.ShootingProblem(np.array(st["y0"])[:-NF].copy(), an[:-1], an[-1])
    mpc.ocp.x0 = st["y0"]
    # constraint-Jacobian scaling at this state
    x0 = np.array(st["y0"])[:nq+nv]
    pin.forwardKinematics(state.pinocchio, pdata, x0[:nq], x0[nq:])
    pin.updateFramePlacements(state.pinocchio, pdata)
    pin.computeForwardKinematicsDerivatives(state.pinocchio, pdata, x0[:nq], x0[nq:], np.zeros(nv))
    pin.updateFramePlacements(state.pinocchio, pdata)
    dq = np.vstack([a.calcDiff(pdata)[0] for a in anchors])
    dv = np.vstack([a.calcDiff(pdata)[1] for a in anchors])
    g_lam = mpc.running_models[1].forceConstraints.calcDiff(np.array(st["y0"])[-NF:])
    data["jac"][str(st["sample"])] = dict(
        g_lambda=float(np.linalg.norm(g_lam, 2)),
        g_x=float(np.linalg.norm(g_lam @ np.hstack([dq, dv]), 2)),
        dlam_dq=float(np.linalg.norm(dq, 2)), dlam_dv=float(np.linalg.norm(dv, 2)))
    for rm in REGS:
        for tag, prob, xs in (("force-state", mpc.ocp, xs_s), ("anchor", p_a, xs_r)):
            s = mim_solvers.SolverCSQP(prob)
            for k, v in C.SOLVER.items(): setattr(s, k, v)
            s.reg_min = rm
            s.setCallbacks([mim_solvers.CallbackVerbose()])
            t0 = time.perf_counter()
            _, log = C.capture_stdout(lambda: s.solve(list(xs), list(us), MAXIT))
            ms = (time.perf_counter() - t0) * 1e3
            k_, c_ = traces(log)
            lam = ([np.array(x)[-NF:] for x in s.xs] if tag == "force-state"
                   else [C.lam_of(anchors, state, pdata, np.array(x)) for x in s.xs])
            cone, uni = per_node_viol(lam)
            m = prob.runningModels[0]
            data["runs"].append(dict(
                sample=int(st["sample"]), form=tag, reg=rm, kkt=float(s.KKT),
                cstr=float(s.constraint_norm), gap=float(s.gap_norm), cost=float(s.cost),
                iters=int(s.iter), ms=float(ms), ms_per_it=float(ms/max(s.iter, 1)),
                cone_max=float(cone.max()), uni_max=float(uni.max()),
                cone=cone.tolist(), uni=uni.tolist(), kkt_trace=k_, cstr_trace=c_,
                ndx=int(m.state.ndx), nx=int(m.state.nx), nu=int(m.nu), ng=int(m.ng),
                capped=bool(s.iter >= MAXIT)))
            print(f"done {st['sample']} {tag} reg={rm:.0e} KKT={s.KKT:.2e} "
                  f"cone={cone.max():.2e}N it={s.iter}", flush=True)

json.dump(data, open(OUT + "letter_data.json", "w"))
print("saved", OUT + "letter_data.json")

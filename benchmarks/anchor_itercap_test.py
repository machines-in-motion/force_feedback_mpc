"""Does raising the 1000-iteration cap get below 1e-4, or does the KKT plateau?"""
import sys, time, numpy as np, crocoddyl, mim_solvers
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anchor_vs_state_common as C

NF, BIG = 15, 5000
MARKS = (50, 100, 250, 500, 1000, 2000, 3000, 4000, 5000)

def traces(log):
    kkt, cstr = [], []
    for line in log.splitlines():
        m = C.ROW.match(line)
        if m:
            try: cstr.append(float(m.group(5))); kkt.append(float(m.group(8)))
            except ValueError: pass
    return kkt, cstr

print(f"{'sample':>6s} {'form':>7s} {'reg':>6s} {'iters':>6s} {'early':>5s} {'final KKT':>10s} "
      f"{'min KKT':>10s} {'final cstr':>10s} " + " ".join(f"{'@'+str(m):>9s}" for m in MARKS))
for st in C.load_b_final_states([80, 210]):
    mpc = C.build_state_mpc()
    anchors = C.make_anchors(mpc, st["y0"])
    xs_r, xs_s, us = C.make_guess(mpc, anchors, st["y0"], st["u0"],
                                  traj=(st["xs_traj"], st["us_traj"]))
    an = C.build_anchor_models(mpc, anchors)
    p_a = crocoddyl.ShootingProblem(np.array(st["y0"])[:-NF].copy(), an[:-1], an[-1])
    mpc.ocp.x0 = st["y0"]
    for rm in (1e-2, 1e-1):
        for tag, prob, xs in (("lifted", mpc.ocp, xs_s), ("anchor", p_a, xs_r)):
            s = mim_solvers.SolverCSQP(prob)
            for k, v in C.SOLVER.items(): setattr(s, k, v)
            s.reg_min = rm
            s.setCallbacks([mim_solvers.CallbackVerbose()])
            _, log = C.capture_stdout(lambda: s.solve(list(xs), list(us), BIG))
            k_, c_ = traces(log)
            at = " ".join(f"{(k_[m-1] if len(k_) >= m else np.nan):9.2e}" for m in MARKS)
            print(f"{st['sample']:6d} {tag:>7s} {rm:6.0e} {s.iter:6d} {str(s.iter < BIG):>5s} "
                  f"{s.KKT:10.2e} {min(k_) if k_ else np.nan:10.2e} {s.constraint_norm:10.2e} {at}",
                  flush=True)

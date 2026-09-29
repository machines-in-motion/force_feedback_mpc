"""
Sections 5 + 8: nominal apples-to-apples comparison of the force-state (lifted) and
anchor-output transcriptions on the corrected Go2 setup (B_final parameters).

Receding sequence of MPC cycles: x0 is taken from the realized B_final closed loop
(identical for both formulations), each formulation is warm-started from ITS OWN
previous solution (shifted), and the anchor p_c is re-reconstructed at every cycle
from (q0, v0, lambda_measured) exactly as an anchor-update controller would.

Run:  python benchmarks/anchor_vs_state_nominal.py [n_cycles]
"""
import sys, os, time
import numpy as np
import crocoddyl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anchor_vs_state_common as C   # noqa: E402

N_CYCLES = int(sys.argv[1]) if len(sys.argv) > 1 else 20
START = 80          # first simulation sample (t = 80 ms, ordinary loaded contact)
STEP = 10           # 10 ms between MPC cycles


def shift(xs, us):
    return [*xs[1:], xs[-1].copy()], [*us[1:], us[-1].copy()]


def main():
    samples = [START + STEP * k for k in range(N_CYCLES)]
    states = C.load_b_final_states(samples)
    mpc = C.build_state_mpc()
    st0 = mpc.running_models[0].differential.state
    nq, nv = st0.nq, st0.nv

    # initial common warm start: the realized closed-loop trajectory
    anchors = C.make_anchors(mpc, states[0]["y0"])
    an_iams = C.build_anchor_models(mpc, anchors)
    p_anchor = crocoddyl.ShootingProblem(states[0]["y0"][:nq + nv].copy(),
                                         an_iams[:-1], an_iams[-1])
    xs_r, xs_s, us0 = C.make_guess(mpc, anchors, states[0]["y0"], states[0]["u0"],
                                   traj=(states[0]["xs_traj"], states[0]["us_traj"]))
    ws = {"state": (xs_s, us0), "anchor": (xs_r, list(us0))}
    rows = []
    for k, st in enumerate(states):
        y0 = st["y0"]; x0 = y0[:nq + nv]
        # anchor update from the measured force at this cycle
        for i, a in enumerate(anchors):
            pass
        anchors_k = C.make_anchors(mpc, y0)
        for a_old, a_new in zip(anchors, anchors_k):
            a_old.oPc = a_new.oPc            # same anchor objects are shared by the models
        mpc.ocp.x0 = y0
        p_anchor.x0 = x0
        out = {}
        for tag, problem in (("state", mpc.ocp), ("anchor", p_anchor)):
            xs, us = ws[tag]
            xs = [x.copy() for x in xs]
            xs[0] = y0.copy() if tag == "state" else x0.copy()
            r = C.solve_record(problem, xs, us, tag)
            ws[tag] = shift(r["xs"], r["us"])
            out[tag] = r
        # physical comparison at the first node of the solution
        lam_state = out["state"]["xs"][1][nq + nv:]
        pdata = st0.pinocchio.createData()
        lam_anchor = C.lam_of(anchors, st0, pdata, out["anchor"]["xs"][1])
        rows.append((st["sample"], out, float(np.abs(lam_state - lam_anchor).max())))
        print(f"cycle {k:3d} (t={st['sample']*1e-3:.2f}s)  " + "  ".join(
            f"{t}: KKT={out[t]['kkt']:.2e} it={out[t]['iters']:4d} qp={out[t]['qp_total']:6d} "
            f"bt={out[t]['backtracks']:3d} viol={out[t]['viol']:.1e} {out[t]['time_ms']:6.0f}ms"
            for t in ("state", "anchor"))
            + f"  |dlam|={np.abs(lam_state - lam_anchor).max():.2f}N", flush=True)

    print("\n================ summary over %d cycles ================" % len(rows))
    hdr = f"{'':8s} {'KKT med':>9s} {'KKT max':>9s} {'conv<1e-4':>9s} {'<1e-3':>6s} {'<1e-2':>6s} " \
          f"{'SQP it med':>10s} {'QP it med':>10s} {'backtracks':>10s} {'viol max':>9s} {'time med':>9s} {'cost med':>9s}"
    print(hdr)
    for t in ("state", "anchor"):
        kkt = np.array([r[1][t]["kkt"] for r in rows])
        it = np.array([r[1][t]["iters"] for r in rows])
        qp = np.array([r[1][t]["qp_total"] for r in rows])
        bt = np.array([r[1][t]["backtracks"] for r in rows])
        vi = np.array([r[1][t]["viol"] for r in rows])
        tm = np.array([r[1][t]["time_ms"] for r in rows])
        co = np.array([r[1][t]["cost"] for r in rows])
        print(f"{t:8s} {np.median(kkt):9.2e} {kkt.max():9.2e} {int((kkt<=1e-4).sum()):9d} "
              f"{int((kkt<=1e-3).sum()):6d} {int((kkt<=1e-2).sum()):6d} {np.median(it):10.0f} "
              f"{np.median(qp):10.0f} {int(bt.sum()):10d} {vi.max():9.2e} {np.median(tm):9.0f} {np.median(co):9.4f}")
    dl = np.array([r[2] for r in rows])
    print(f"\nnode-1 force prediction difference |lambda_state - lambda_anchor|: "
          f"median {np.median(dl):.2f} N, max {dl.max():.2f} N")


if __name__ == "__main__":
    main()

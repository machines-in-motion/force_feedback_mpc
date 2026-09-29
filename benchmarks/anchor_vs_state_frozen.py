"""
Sections 5-8: frozen-OCP comparison of the force-state (lifted) and anchor-output
transcriptions on the corrected Go2 setup.

Warm starts come from a replay of the closed-loop force-state run: at each selected
MPC cycle the warm start is the shifted previous SOLUTION (what the controller
actually uses).  The SAME physical (q, v, u) content is given to both formulations;
the state formulation's force guess is taken from the anchor map along that
trajectory, i.e. it starts ON the anchor manifold (section 7.1).

  python anchor_vs_state_frozen.py nominal   <dump.npz>
  python anchor_vs_state_frozen.py stiffness <dump.npz>
  python anchor_vs_state_frozen.py basin     <dump.npz>
"""
import sys, os
import numpy as np
import crocoddyl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anchor_vs_state_common as C   # noqa: E402

CYCLES = [8, 15, 21, 27]        # ordinary loaded-contact cycles of the B_final run
APEX_CYCLE = 33                 # near-apex cycle, reported separately
ALPHAS = [0.25, 0.5, 1.0, 2.0, 4.0]
AMPS = [0.0, 0.01, 0.03, 0.1, 0.3]
SEEDS = range(10)
THRESH = (1e-4, 1e-3, 1e-2)     # fixed in advance


def setup(dump, cycle, alpha=1.0):
    """state + anchor problems at one cycle, with the matched warm start"""
    y0 = np.array(dump["x0"][cycle], float)
    mpc = C.build_state_mpc(alpha=alpha)
    st = mpc.running_models[0].differential.state
    nq, nv = st.nq, st.nv
    anchors = C.make_anchors(mpc, y0)        # p_c from (q0, v0, lambda_meas)
    an_iams = C.build_anchor_models(mpc, anchors)
    x0 = y0[:nq + nv]
    p_anchor = crocoddyl.ShootingProblem(x0.copy(), an_iams[:-1], an_iams[-1])
    mpc.ocp.x0 = y0
    # shifted previous solution -> common physical (q, v, u) guess
    ws_xs = np.array(dump["ws_xs"][cycle], float)
    ws_us = np.array(dump["ws_us"][cycle], float)
    xs_r = [ws_xs[t][:nq + nv].copy() for t in range(len(ws_xs))]
    xs_r[0] = x0.copy()
    us = [ws_us[t].copy() for t in range(len(ws_us))]
    return mpc, anchors, p_anchor, x0, y0, xs_r, us, st


def guesses(anchors, st, xs_r, y0, rng=None, amp=0.0):
    """perturb the common physical guess; state force guess from the anchor map"""
    nq, nv, ndx = st.nq, st.nv, st.ndx
    pdata = st.pinocchio.createData()
    xs = [x.copy() for x in xs_r]
    if amp > 0:
        for t in range(1, len(xs)):
            dx = np.zeros(ndx)
            dx[:nv] = amp * rng.standard_normal(nv)
            dx[nv:] = amp * rng.standard_normal(nv)
            xs[t] = st.integrate(xs[t], dx)
    xs_state = [np.concatenate([x, C.lam_of(anchors, st, pdata, x)]) for x in xs]
    xs_state[0] = y0.copy()
    return xs, xs_state


def jac_metrics(anchors, st, mpc, x, lam):
    """structural scaling metrics: anchor pullback vs direct force Jacobians"""
    import pinocchio as pin
    pdata = st.pinocchio.createData()
    pin.forwardKinematics(st.pinocchio, pdata, x[:st.nq], x[st.nq:])
    pin.updateFramePlacements(st.pinocchio, pdata)
    pin.computeForwardKinematicsDerivatives(st.pinocchio, pdata, x[:st.nq], x[st.nq:],
                                            np.zeros(st.nv))
    pin.updateFramePlacements(st.pinocchio, pdata)
    dq = np.vstack([a.calcDiff(pdata)[0] for a in anchors])
    dv = np.vstack([a.calcDiff(pdata)[1] for a in anchors])
    # force-constraint Jacobian: anchor = g_lambda * dlambda/dx ; state = g_lambda
    fcm = mpc.running_models[1].forceConstraints
    g_lam = fcm.calcDiff(lam)
    return dict(dlam_dq=np.linalg.norm(dq, 2), dlam_dv=np.linalg.norm(dv, 2),
                g_state=np.linalg.norm(g_lam, 2),
                g_anchor=np.linalg.norm(g_lam @ np.hstack([dq, dv]), 2))


def run_nominal(dump):
    print(f"{'cycle':>6s} {'form':7s} {'KKT':>9s} {'cost':>9s} {'SQPit':>6s} {'QPit':>8s} "
          f"{'bt':>4s} {'viol':>9s} {'time_ms':>8s}")
    for c in CYCLES + [APEX_CYCLE]:
        mpc, anchors, p_anchor, x0, y0, xs_r, us, st = setup(dump, c)
        xs, xs_state = guesses(anchors, st, xs_r, y0)
        rs = C.solve_record(mpc.ocp, xs_state, us, "state")
        ra = C.solve_record(p_anchor, xs, us, "anchor")
        tag = " (apex)" if c == APEX_CYCLE else ""
        for r in (rs, ra):
            print(f"{c:6d}{tag} {r['tag']:7s} {r['kkt']:9.2e} {r['cost']:9.4f} {r['iters']:6d} "
                  f"{r['qp_total']:8d} {r['backtracks']:4d} {r['viol']:9.2e} {r['time_ms']:8.0f}")
        nq, nv = st.nq, st.nv
        pdata = st.pinocchio.createData()
        lam_s = rs["xs"][1][nq + nv:]
        lam_a = C.lam_of(anchors, st, pdata, ra["xs"][1])
        print(f"       -> |lambda_state - lambda_anchor| at node 1: {np.abs(lam_s - lam_a).max():.3f} N, "
              f"cost difference {abs(rs['cost'] - ra['cost']):.2e}")


def run_stiffness(dump):
    print(f"{'cycle':>6s} {'alpha':>6s} {'form':7s} {'KKT':>9s} {'SQPit':>6s} {'QPit':>8s} "
          f"{'bt':>4s} {'viol':>9s} {'|dlam/dq|':>10s} {'|g_x|':>10s}")
    for c in CYCLES[:3]:
        for a in ALPHAS:
            mpc, anchors, p_anchor, x0, y0, xs_r, us, st = setup(dump, c, alpha=a)
            xs, xs_state = guesses(anchors, st, xs_r, y0)
            m = jac_metrics(anchors, st, mpc, x0, y0[st.nq + st.nv:])
            rs = C.solve_record(mpc.ocp, xs_state, us, "state")
            ra = C.solve_record(p_anchor, xs, us, "anchor")
            for r, gn in ((rs, m["g_state"]), (ra, m["g_anchor"])):
                print(f"{c:6d} {a:6.2f} {r['tag']:7s} {r['kkt']:9.2e} {r['iters']:6d} "
                      f"{r['qp_total']:8d} {r['backtracks']:4d} {r['viol']:9.2e} "
                      f"{m['dlam_dq']:10.1f} {gn:10.3f}", flush=True)


def run_basin(dump):
    print(f"{'cycle':>6s} {'amp':>6s} {'form':7s} " +
          " ".join(f"{'<'+f'{t:.0e}':>8s}" for t in THRESH) +
          f" {'KKT med':>9s} {'SQPit med':>9s} {'bt tot':>7s} {'time med':>9s}")
    for c in CYCLES[:2]:
        for amp in AMPS:
            acc = {"state": [], "anchor": []}
            for seed in SEEDS:
                rng = np.random.default_rng(1000 * c + seed)
                mpc, anchors, p_anchor, x0, y0, xs_r, us, st = setup(dump, c)
                xs, xs_state = guesses(anchors, st, xs_r, y0, rng=rng, amp=amp)
                acc["state"].append(C.solve_record(mpc.ocp, xs_state, us, "state"))
                acc["anchor"].append(C.solve_record(p_anchor, xs, us, "anchor"))
                if amp == 0.0:
                    break     # deterministic
            for t in ("state", "anchor"):
                k = np.array([r["kkt"] for r in acc[t]])
                it = np.array([r["iters"] for r in acc[t]])
                bt = np.array([r["backtracks"] for r in acc[t]])
                tm = np.array([r["time_ms"] for r in acc[t]])
                fr = " ".join(f"{int((k <= th).sum()):3d}/{len(k):<4d}" for th in THRESH)
                print(f"{c:6d} {amp:6.2f} {t:7s} {fr} {np.median(k):9.2e} "
                      f"{np.median(it):9.0f} {int(bt.sum()):7d} {np.median(tm):9.0f}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "nominal"
    dump = np.load(sys.argv[2], allow_pickle=True)
    print(f"# mode={mode} dump={sys.argv[2]} cycles={len(dump['x0'])}")
    {"nominal": run_nominal, "stiffness": run_stiffness, "basin": run_basin}[mode](dump)

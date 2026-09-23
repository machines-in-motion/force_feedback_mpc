"""
Benchmark (AuRo revision, reviewer 1 major comment 4): analytical vs numerical
derivatives of the Go2+arm multicontact soft-contact action model (Python API).

Accuracy : analytical calcDiff vs CENTRAL finite differences (manifold-aware),
           with a finite-difference step sweep, at real MPC states of the final
           B_final experiment (ordinary loaded contact and near-apex nodes).
Timing   : analytical calc+calcDiff, forward and central finite differences of the
           same action model, and crocoddyl's ActionModelNumDiff, per node and over
           a full MPC horizon at a fixed trajectory.

Run (in the force_feedback_go2 environment):
    python benchmarks/soft_derivatives_go2.py [path_to_states.npz]

This file only benchmarks the production models; it does not modify them.
"""

import sys, time, os
import numpy as np
import crocoddyl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "demos", "go2arm"))
from Go2MPC_wrapper_soft import Go2MPCSoft  # noqa: E402

# ------------------------------------------------------------------ settings
STATES = sys.argv[1] if len(sys.argv) > 1 else (
    "/tmp/claude-1000/-home-skleff-CODE-force-feedback-mpc/"
    "b542be30-66c4-4751-9895-019002b73177/scratchpad/replay/final_soft_dump.npz"
)
MU, FWEIGHT, DT, HORIZON, FMIN = 0.75, 0.002, 0.01, 20, 80   # final Go2 experiment (B_final)
H_SWEEP = [1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7]
H_REF = 1e-5
REPS, HORIZON_REPS = 200, 20
APEX_TOL = 0.5  # [N] a node is "near apex" when a foot normal force is below this


# ------------------------------------------------------- finite differences
def fd_derivatives(iam, d_plus, d_minus, y, u, h, central=True):
    """First-order derivatives of the action model by finite differences.

    Manifold-aware: the state is perturbed in the tangent space and the output
    difference is taken with state.diff, as crocoddyl's NumDiff does.
    Returns Fx, Fu, Lx, Lu, Gx, Gu (Gx/Gu only when the model has constraints).
    """
    st = iam.state
    ndx, nu, ng = st.ndx, iam.nu, iam.ng
    Fx, Fu = np.zeros((ndx, ndx)), np.zeros((ndx, nu))
    Lx, Lu = np.zeros(ndx), np.zeros(nu)
    Gx, Gu = np.zeros((ng, ndx)), np.zeros((ng, nu))
    den = 2.0 * h if central else h
    if not central:
        iam.calc(d_minus, y, u)  # baseline
    dx = np.zeros(ndx)
    for i in range(ndx):
        dx[i] = h
        iam.calc(d_plus, st.integrate(y, dx), u)
        if central:
            iam.calc(d_minus, st.integrate(y, -dx), u)
        Fx[:, i] = st.diff(np.array(d_minus.xnext), np.array(d_plus.xnext)) / den
        Lx[i] = (d_plus.cost - d_minus.cost) / den
        if ng:
            Gx[:, i] = (np.array(d_plus.g) - np.array(d_minus.g)) / den
        dx[i] = 0.0
    du = np.zeros(nu)
    for j in range(nu):
        du[j] = h
        iam.calc(d_plus, y, u + du)
        if central:
            iam.calc(d_minus, y, u - du)
        Fu[:, j] = st.diff(np.array(d_minus.xnext), np.array(d_plus.xnext)) / den
        Lu[j] = (d_plus.cost - d_minus.cost) / den
        if ng:
            Gu[:, j] = (np.array(d_plus.g) - np.array(d_minus.g)) / den
        du[j] = 0.0
    return Fx, Fu, Lx, Lu, Gx, Gu


def err(a, b):
    """max absolute error and relative Frobenius error with a robust denominator"""
    a, b = np.atleast_2d(a), np.atleast_2d(b)
    return np.abs(a - b).max(), np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-9)


def timeit(fn, reps, warmup=10):
    for _ in range(warmup):
        fn()
    t = np.empty(reps)
    for i in range(reps):
        t0 = time.perf_counter_ns()
        fn()
        t[i] = (time.perf_counter_ns() - t0) * 1e-3  # us
    return dict(median=np.median(t), mean=t.mean(), p05=np.percentile(t, 5),
                p95=np.percentile(t, 95), n=reps)


def show(name, s):
    print(f"  {name:36s} {s['median']:10.1f} us (median) {s['mean']:10.1f} (mean) "
          f"{s['p05']:9.1f} (p05) {s['p95']:9.1f} (p95)  n={s['n']}")


# ----------------------------------------------------------------- main
if __name__ == "__main__":
    mpc = Go2MPCSoft(HORIZON=HORIZON, friction_mu=MU, dt=DT, USE_MUJOCO=False)
    mpc.initialize(FMIN=FMIN, FWEIGHT=FWEIGHT, FDOTWEIGHT=0.0)
    iam = mpc.running_models[1]
    data = iam.createData()
    d_plus, d_minus = iam.createData(), iam.createData()
    nq, nv = mpc.rmodel.nq, mpc.rmodel.nv
    nc = 5 * 3
    st = iam.state
    print(f"Go2+arm soft-contact multicontact model: nq={nq} nv={nv} contacts=5 (nc={nc}) "
          f"-> nx={st.nx} ndx={st.ndx} nu={iam.nu} ng={iam.ng} dt={DT} N={HORIZON}")

    # representative states: real MPC solutions of the final experiment
    D = np.load(STATES, allow_pickle=True)
    picks = []
    for c in (2, 21, 33):
        xs, us = D["xs"][c], D["us"][c]
        FN = np.array([xs[t][nq + nv:].reshape(5, 3)[:4, 2].min() for t in range(len(us))])
        picks.append((np.array(xs[int(np.argmax(FN))]), np.array(us[int(np.argmax(FN))]),
                      f"cycle {c} ordinary (min F_N={FN.max():.2f} N)"))
        picks.append((np.array(xs[int(np.argmin(FN))]), np.array(us[int(np.argmin(FN))]),
                      f"cycle {c} near apex (min F_N={FN.min():.3f} N)"))
    y0, u0 = picks[0][0], picks[0][1]

    # ---------------- finite-difference step sweep
    print(f"\n-- central finite-difference step sweep ({picks[0][2]})")
    print("     h        max|Fx err|   relFro(Fx)   max|Fu err|   relFro(Fu)")
    iam.calc(data, y0, u0); iam.calcDiff(data, y0, u0)
    A_Fx, A_Fu = np.array(data.Fx).copy(), np.array(data.Fu).copy()
    for h in H_SWEEP:
        Fx, Fu = fd_derivatives(iam, d_plus, d_minus, y0, u0, h)[:2]
        ex, eu = err(A_Fx, Fx), err(A_Fu, Fu)
        print(f"  {h:8.0e}   {ex[0]:11.2e}   {ex[1]:11.2e}   {eu[0]:11.2e}   {eu[1]:11.2e}")

    # ---------------- accuracy at the representative states
    print(f"\n-- accuracy vs central finite differences (h={H_REF:.0e})")
    print(f"  {'state':38s} {'block':32s} {'max|err|':>11s} {'relFro':>11s}")
    for y, u, tag in picks:
        iam.calc(data, y, u); iam.calcDiff(data, y, u)
        A = dict(Fx=np.array(data.Fx).copy(), Fu=np.array(data.Fu).copy(),
                 Lx=np.array(data.Lx).copy(), Lu=np.array(data.Lu).copy(),
                 Gx=np.array(data.Gx).copy(), Gu=np.array(data.Gu).copy())
        Fx, Fu, Lx, Lu, Gx, Gu = fd_derivatives(iam, d_plus, d_minus, y, u, H_REF)
        nf = nc  # force-state rows are the last nc rows of the tangent space
        blocks = [
            ("Fx (all)", A["Fx"], Fx),
            ("Fx robot-state rows", A["Fx"][:-nf], Fx[:-nf]),
            ("Fx force-state rows", A["Fx"][-nf:], Fx[-nf:]),
            ("  dlam_next/dq", A["Fx"][-nf:, :nv], Fx[-nf:, :nv]),
            ("  dlam_next/dv", A["Fx"][-nf:, nv:2 * nv], Fx[-nf:, nv:2 * nv]),
            ("  dlam_next/dlam", A["Fx"][-nf:, -nf:], Fx[-nf:, -nf:]),
            ("Fu (all)", A["Fu"], Fu),
            ("  dlam_next/dtau", A["Fu"][-nf:], Fu[-nf:]),
            ("Lx", A["Lx"], Lx), ("Lu", A["Lu"], Lu),
            ("Gx (friction/unilaterality)", A["Gx"], Gx), ("Gu", A["Gu"], Gu),
        ]
        for name, a, b in blocks:
            e = err(a, b)
            print(f"  {tag:38s} {name:32s} {e[0]:11.2e} {e[1]:11.2e}")

    # ---------------- timing per node
    print(f"\n-- timing per node ({REPS} repetitions, {picks[0][2]})")
    nd = crocoddyl.ActionModelNumDiff(iam)
    nd_data = nd.createData()

    def f_ana():
        iam.calc(data, y0, u0); iam.calcDiff(data, y0, u0)

    def f_fwd():
        fd_derivatives(iam, d_plus, d_minus, y0, u0, 1e-7, central=False)

    def f_cen():
        fd_derivatives(iam, d_plus, d_minus, y0, u0, H_REF, central=True)

    def f_nd():
        nd.calc(nd_data, y0, u0); nd.calcDiff(nd_data, y0, u0)

    s_ana, s_fwd, s_cen = timeit(f_ana, REPS), timeit(f_fwd, REPS), timeit(f_cen, REPS)
    s_nd = timeit(f_nd, max(REPS // 10, 5))
    show("analytical (calc+calcDiff)", s_ana)
    show("forward FD (1st order only)", s_fwd)
    show("central FD (1st order only)", s_cen)
    show("crocoddyl NumDiff (+num Hessians)", s_nd)
    print(f"  speedup vs forward FD: {s_fwd['median']/s_ana['median']:.1f}x   "
          f"vs central FD: {s_cen['median']/s_ana['median']:.1f}x   "
          f"vs crocoddyl NumDiff: {s_nd['median']/s_ana['median']:.1f}x")

    # ---------------- timing over a full horizon at a fixed trajectory
    print(f"\n-- timing full horizon N={HORIZON} ({HORIZON_REPS} repetitions, fixed trajectory)")
    models = list(mpc.ocp.runningModels)
    datas = [m.createData() for m in models]
    dps, dms = [m.createData() for m in models], [m.createData() for m in models]
    xs = [np.array(x) for x in D["xs"][21][:HORIZON]]
    us = [np.array(u) for u in D["us"][21][:HORIZON]]

    def h_ana():
        for m, dd, x, u in zip(models, datas, xs, us):
            m.calc(dd, x, u); m.calcDiff(dd, x, u)

    def h_cen():
        for m, dp, dm, x, u in zip(models, dps, dms, xs, us):
            fd_derivatives(m, dp, dm, x, u, H_REF, central=True)

    def h_fwd():
        for m, dp, dm, x, u in zip(models, dps, dms, xs, us):
            fd_derivatives(m, dp, dm, x, u, 1e-7, central=False)

    ha, hf, hc = timeit(h_ana, HORIZON_REPS, 3), timeit(h_fwd, HORIZON_REPS, 3), timeit(h_cen, HORIZON_REPS, 3)
    show("analytical (all nodes)", ha)
    show("forward FD (all nodes)", hf)
    show("central FD (all nodes)", hc)
    print(f"  speedup vs forward FD: {hf['median']/ha['median']:.1f}x   "
          f"vs central FD: {hc['median']/ha['median']:.1f}x")

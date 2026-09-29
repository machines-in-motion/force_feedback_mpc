"""
Shared harness for the state (lifted) vs anchor-output comparison on Go2+arm.

Fairness by construction:
  - the anchor models reuse the production cost / force-cost / force-constraint
    objects, so costs, weights, references, bounds are identical;
  - both problems are solved by the SAME solver class with identical settings;
  - warm starts are built from the SAME physical (q, v, u) trajectory, with the
    state formulation's force initialized ON the anchor manifold.
"""
import os, re, sys, tempfile, time
import numpy as np
import pinocchio as pin
import crocoddyl
import mim_solvers

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "demos", "go2arm"))
sys.path.insert(0, HERE)
from Go2MPC_wrapper_soft import Go2MPCSoft                       # noqa: E402
from anchor_output_go2 import (                                   # noqa: E402
    AnchorContact3D, DAMAnchorOutput3D_Go2, IAMAnchorOutput_Go2,
)

B_FINAL_GLOB = ("/home/skleff/Desktop/PUBLICATIONS/TRO-Soft/AURO/REVISION/"
                "GO2_RERUN/B_final/go2_soft*.npz")
MU, FWEIGHT, DT, HORIZON, FMIN = 0.75, 0.002, 0.01, 20, 80
SOLVER = dict(max_qp_iters=10000, termination_tolerance=1e-4, eps_abs=1e-8,
              eps_rel=1e-8, use_filter_line_search=False, mu_constraint=-1.0)
MAXIT = 1000

ROW = re.compile(r"\s*(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\d+)\s*$")


# ------------------------------------------------------------------- utilities
def capture_stdout(fn):
    """capture C-level stdout (the solver callbacks write there)"""
    saved = os.dup(1)
    with tempfile.TemporaryFile(mode="w+") as tmp:
        os.dup2(tmp.fileno(), 1)
        try:
            out = fn()
        finally:
            os.dup2(saved, 1)
            os.close(saved)
        tmp.seek(0)
        return out, tmp.read()


def parse_log(text):
    """iteration count, total QP iterations, line-search backtracks, KKT trace"""
    kkt, qp, steps = [], [], []
    for line in text.splitlines():
        m = ROW.match(line)
        if m:
            try:
                steps.append(float(m.group(7)))
                kkt.append(float(m.group(8)))
                qp.append(int(m.group(9)))
            except ValueError:
                pass
    return dict(n_iter=len(kkt), qp_total=int(np.sum(qp)) if qp else 0,
                backtracks=int(np.sum(np.array(steps) < 1.0)) if steps else 0,
                kkt_trace=kkt, step_trace=steps)


def load_b_final_states(samples):
    """(y0, u0) at the given simulation samples, plus the realized closed-loop
    trajectory over the following horizon (MPC nodes are 10 simulation steps apart),
    which serves as the common physical warm start for both formulations."""
    import glob
    d = np.load(glob.glob(B_FINAL_GLOB)[0], allow_pickle=True)
    q = np.asarray(d["jointPos"], float); v = np.asarray(d["jointVel"], float)
    tau = np.asarray(d["joint_torques"], float)
    M = d["measured_forces"].item(); names = list(d["ee_frame_names"])
    step = int(round(DT / 1e-3))       # simulation step is 1 ms
    out = []
    for i in samples:
        def state_at(j):
            j = min(j, len(q) - 1)
            qi = q[j].copy(); qi[3:7] /= np.linalg.norm(qi[3:7])
            return np.concatenate([qi, v[j]])
        f = np.concatenate([np.asarray(M[n], float)[i] for n in names])
        xs_traj = [state_at(i + step * t) for t in range(HORIZON + 1)]
        us_traj = [tau[min(i + step * t, len(tau) - 1)].copy() for t in range(HORIZON)]
        out.append(dict(y0=np.concatenate([xs_traj[0], f]), u0=tau[i].copy(),
                        sample=i, xs_traj=xs_traj, us_traj=us_traj))
    return out


# ----------------------------------------------------------------- model setup
def build_state_mpc(alpha=1.0):
    """production force-state MPC; alpha scales the contact stiffness Kp"""
    mpc = Go2MPCSoft(HORIZON=HORIZON, friction_mu=MU, dt=DT, USE_MUJOCO=False)
    mpc.initialize(FMIN=FMIN, FWEIGHT=FWEIGHT, FDOTWEIGHT=0.0)
    if alpha != 1.0:
        for iam in mpc.running_models:
            for ct in iam.differential.contacts.contacts:
                ct.Kp = ct.Kp * alpha
    return mpc


def make_anchors(mpc, y0):
    """anchors with p_c reconstructed so that h(q0,v0;p_c) == lambda_measured"""
    dam0 = mpc.running_models[0].differential
    state = dam0.state
    nq, nv = state.nq, state.nv
    pdata = state.pinocchio.createData()
    pin.forwardKinematics(state.pinocchio, pdata, y0[:nq], y0[nq:nq + nv])
    pin.updateFramePlacements(state.pinocchio, pdata)
    anchors = []
    for k, ct in enumerate(dam0.contacts.contacts):
        a = AnchorContact3D(state, ct.frameId, ct.Kp, ct.Kv, ct.pinRef)
        a.set_anchor_from_measurement(pdata, y0[nq + nv:][3 * k: 3 * k + 3])
        anchors.append(a)
    return anchors


def build_anchor_models(mpc, anchors):
    iams = []
    for iam_s in mpc.running_models:
        dam_s = iam_s.differential
        dam = DAMAnchorOutput3D_Go2(
            dam_s.state, dam_s.actuation, dam_s.costs, anchors,
            forceCosts=dam_s.forceCosts if dam_s.with_force_cost else None,
            forceConstraints=iam_s.forceConstraints,
        )
        iams.append(IAMAnchorOutput_Go2(dam, dt=iam_s.dt, withCostResidual=True))
    return iams


def make_solver(problem):
    s = mim_solvers.SolverCSQP(problem)
    for k, v in SOLVER.items():
        setattr(s, k, v)
    return s


def lam_of(anchors, state, pdata, x):
    nq = state.nq
    pin.forwardKinematics(state.pinocchio, pdata, x[:nq], x[nq:])
    pin.updateFramePlacements(state.pinocchio, pdata)
    return np.concatenate([c.calc(pdata) for c in anchors])


# ------------------------------------------------------------------ warm starts
def make_guess(mpc, anchors, y0, u0, rng=None, amp=0.0, traj=None):
    """
    Common physical guess (q, v, u) for both formulations; the state guess gets its
    force from the anchor map along that same trajectory (i.e. ON the anchor manifold).
    amp > 0 perturbs the guess (tangent-space configuration, velocity, controls).
    """
    state = mpc.running_models[0].differential.state
    nq, nv, ndx = state.nq, state.nv, state.ndx
    x0 = y0[:nq + nv]
    pdata = state.pinocchio.createData()
    if traj is not None:                      # realized closed-loop trajectory
        xs_r = [x.copy() for x in traj[0]]
        us = [u.copy() for u in traj[1]]
    else:
        xs_r = [x0.copy() for _ in range(HORIZON + 1)]
        us = [u0.copy() for _ in range(HORIZON)]
    if amp > 0 and rng is not None:
        us = [u + amp * 10.0 * rng.standard_normal(len(u0)) for u in us]
    if amp > 0 and rng is not None:          # perturb the shooting nodes (not x0)
        for t in range(1, HORIZON + 1):
            dx = np.zeros(ndx)
            dx[:nv] = amp * rng.standard_normal(nv)
            dx[nv:] = amp * rng.standard_normal(nv)
            xs_r[t] = state.integrate(xs_r[t], dx)
    xs_r[0] = x0.copy()                      # x0 is fixed by the OCP
    xs_state = [np.concatenate([x, lam_of(anchors, state, pdata, x)]) for x in xs_r]
    xs_state[0] = y0.copy()                  # measured force at node 0
    return xs_r, xs_state, us


# --------------------------------------------------------------------- solving
def solve_record(problem, xs, us, tag):
    s = make_solver(problem)
    s.setCallbacks([mim_solvers.CallbackVerbose()])
    t0 = time.perf_counter_ns()
    _, log = capture_stdout(lambda: s.solve(list(xs), list(us), MAXIT))
    dt_ms = (time.perf_counter_ns() - t0) * 1e-6
    st = parse_log(log)
    # max constraint violation over the horizon at the returned solution
    viol = 0.0
    models = list(s.problem.runningModels); datas = list(s.problem.runningDatas)
    xs_sol = [np.array(x) for x in s.xs]; us_sol = [np.array(u) for u in s.us]
    s.problem.calc(xs_sol, us_sol)
    for m, d in zip(models, datas):
        if m.ng:
            g = np.array(d.g)
            viol = max(viol, np.max(np.maximum(np.array(m.g_lb) - g, 0)),
                       np.max(np.maximum(g - np.array(m.g_ub), 0)))
    return dict(tag=tag, kkt=s.KKT, cost=s.cost, iters=s.iter, time_ms=dt_ms,
                constraint_norm=s.constraint_norm, gap_norm=s.gap_norm,
                viol=viol, xs=xs_sol, us=us_sol, **st)

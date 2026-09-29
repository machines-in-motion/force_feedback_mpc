"""
Build the anchor-output Go2 OCP by REUSING the production force-state objects
(cost sum, force-cost manager, force-constraint manager, contact parameters), and
validate its analytical derivatives against manifold-consistent CENTRAL finite
differences at representative B_final states.

Usage (force_feedback_go2 environment):
    python benchmarks/anchor_build_and_validate.py
"""
import sys, os, glob
import numpy as np
import pinocchio as pin

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "demos", "go2arm"))
sys.path.insert(0, HERE)
from Go2MPC_wrapper_soft import Go2MPCSoft                      # noqa: E402
from anchor_output_go2 import (                                  # noqa: E402
    AnchorContact3D, DAMAnchorOutput3D_Go2, IAMAnchorOutput_Go2,
)

B_FINAL = glob.glob(
    "/home/skleff/Desktop/PUBLICATIONS/TRO-Soft/AURO/REVISION/GO2_RERUN/"
    "B_final/go2_soft*.npz"
)[0]
MU, FWEIGHT, DT, HORIZON, FMIN = 0.75, 0.002, 0.01, 20, 80


# ------------------------------------------------------------------ B_final states
def load_states(samples):
    """(y_state, u) at the given simulation samples of the B_final run"""
    d = np.load(B_FINAL, allow_pickle=True)
    q = np.asarray(d["jointPos"], float); v = np.asarray(d["jointVel"], float)
    tau = np.asarray(d["joint_torques"], float)
    M = d["measured_forces"].item(); names = list(d["ee_frame_names"])
    out = []
    for i in samples:
        qi = q[i].copy()
        qi[3:7] /= np.linalg.norm(qi[3:7])   # logged quaternion is only ~1e-8 normalized
        f = np.concatenate([np.asarray(M[n], float)[i] for n in names])
        out.append((np.concatenate([qi, v[i], f]), tau[i].copy(), i))
    return out


# --------------------------------------------------------------- anchor problem
def build_state_mpc():
    mpc = Go2MPCSoft(HORIZON=HORIZON, friction_mu=MU, dt=DT, USE_MUJOCO=False)
    mpc.initialize(FMIN=FMIN, FWEIGHT=FWEIGHT, FDOTWEIGHT=0.0)
    return mpc


def make_anchors(mpc, y0):
    """One AnchorContact3D per production contact, with p_c reconstructed from
    (q0, v0, lambda_measured) so that h(q0, v0; p_c) == lambda_measured exactly."""
    iam0 = mpc.running_models[0]
    dam0 = iam0.differential
    state = dam0.state
    nq, nv = state.nq, state.nv
    q0, v0 = y0[:nq], y0[nq:nq + nv]
    f0 = y0[nq + nv:]
    pdata = state.pinocchio.createData()
    pin.forwardKinematics(state.pinocchio, pdata, q0, v0)
    pin.updateFramePlacements(state.pinocchio, pdata)
    anchors = []
    for k, ct in enumerate(dam0.contacts.contacts):
        a = AnchorContact3D(state, ct.frameId, ct.Kp, ct.Kv, ct.pinRef)
        a.set_anchor_from_measurement(pdata, f0[3 * k: 3 * k + 3])
        anchors.append(a)
    return anchors


def build_anchor_models(mpc, anchors):
    """Anchor IAMs reusing the production cost/force-cost/constraint objects."""
    iams = []
    for t, iam_s in enumerate(mpc.running_models):
        dam_s = iam_s.differential
        dam = DAMAnchorOutput3D_Go2(
            dam_s.state, dam_s.actuation, dam_s.costs, anchors,
            forceCosts=dam_s.forceCosts if dam_s.with_force_cost else None,
            forceConstraints=iam_s.forceConstraints,
        )
        iams.append(IAMAnchorOutput_Go2(dam, dt=iam_s.dt, withCostResidual=True))
    return iams


# ------------------------------------------------------------------ FD utilities
def fd_jac_state(fun, state, x, h, nout):
    """central FD of fun(x) w.r.t. the state, in tangent coordinates"""
    J = np.zeros((nout, state.ndx))
    dx = np.zeros(state.ndx)
    for i in range(state.ndx):
        dx[i] = h
        J[:, i] = (np.asarray(fun(state.integrate(x, dx))) -
                   np.asarray(fun(state.integrate(x, -dx)))) / (2 * h)
        dx[i] = 0.0
    return J


def err(a, b):
    a, b = np.atleast_2d(np.asarray(a, float)), np.atleast_2d(np.asarray(b, float))
    return np.abs(a - b).max(), np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-9)


# ----------------------------------------------------------------------- main
if __name__ == "__main__":
    mpc = build_state_mpc()
    states = load_states([80, 210, 330])
    y0, u0, _ = states[0]
    anchors = make_anchors(mpc, y0)
    iams = build_anchor_models(mpc, anchors)
    iam = iams[1]
    dam = iam.differential
    state = dam.state
    nq, nv, ndx = state.nq, state.nv, state.ndx
    print(f"anchor model: nx={state.nx} ndx={ndx} nu={iam.nu} ng={iam.ng} "
          f"(state formulation: nx={mpc.running_models[1].state.nx} "
          f"ndx={mpc.running_models[1].state.ndx})")

    # ---- 0. anchor reconstruction reproduces the measured force at node 0
    pdata = state.pinocchio.createData()
    pin.forwardKinematics(state.pinocchio, pdata, y0[:nq], y0[nq:nq + nv])
    pin.updateFramePlacements(state.pinocchio, pdata)
    lam0 = np.concatenate([a.calc(pdata) for a in anchors])
    print(f"\n[0] anchor reconstruction: max|h(q0,v0;p_c) - lambda_meas| = "
          f"{np.abs(lam0 - y0[nq + nv:]).max():.3e}")

    # ---- 1. step sweep for dlambda/dx, then per-state accuracy
    x0 = y0[:nq + nv]
    d = iam.createData()
    print("\n[1] central-FD step sweep on dlambda/dx (state at sample 80)")
    print("      h        max|err|      relFro")
    def lam_of(x):
        pin.forwardKinematics(state.pinocchio, pdata, x[:nq], x[nq:])
        pin.updateFramePlacements(state.pinocchio, pdata)
        return np.concatenate([a.calc(pdata) for a in anchors])
    iam.calc(d, x0, u0); iam.calcDiff(d, x0, u0)
    A = d.differential.df_dx.copy()
    for h in (1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6):
        N = fd_jac_state(lam_of, state, x0, h, 15)
        e = err(A, N)
        print(f"   {h:8.0e}  {e[0]:11.2e}  {e[1]:11.2e}")

    H = 1e-5
    print(f"\n[2] analytical vs central FD at B_final states (h={H:.0e})")
    print(f"   {'state':10s} {'block':28s} {'max|err|':>11s} {'relFro':>11s}")
    for y, u, idx in states:
        a_ = make_anchors(mpc, y)
        iams_ = build_anchor_models(mpc, a_)
        it = iams_[1]; dd = it.createData()
        x = y[:nq + nv]
        it.calc(dd, x, u); it.calcDiff(dd, x, u)
        def lam_of2(z, anch=a_):
            pin.forwardKinematics(state.pinocchio, pdata, z[:nq], z[nq:])
            pin.updateFramePlacements(state.pinocchio, pdata)
            return np.concatenate([c.calc(pdata) for c in anch])
        def xnext_of(z):
            dtmp = it.createData(); it.calc(dtmp, z, u); return np.asarray(dtmp.xnext)
        def g_of(z):
            dtmp = it.createData(); it.calc(dtmp, z, u); return np.asarray(dtmp.g)
        dlam = fd_jac_state(lam_of2, state, x, H, 15)
        Fx_fd = np.zeros((ndx, ndx)); dx = np.zeros(ndx)
        for i in range(ndx):
            dx[i] = H
            xp, xm = state.integrate(x, dx), state.integrate(x, -dx)
            Fx_fd[:, i] = state.diff(xnext_of(xm), xnext_of(xp)) / (2 * H)
            dx[i] = 0.0
        Fu_fd = np.zeros((ndx, it.nu)); du = np.zeros(it.nu)
        for j in range(it.nu):
            du[j] = H
            dp_, dm_ = it.createData(), it.createData()
            it.calc(dp_, x, u + du); it.calc(dm_, x, u - du)
            Fu_fd[:, j] = state.diff(np.asarray(dm_.xnext), np.asarray(dp_.xnext)) / (2 * H)
            du[j] = 0.0
        Gx_fd = fd_jac_state(g_of, state, x, H, it.ng)
        blocks = [
            ("dlambda/dq", dd.differential.df_dx[:, :nv], dlam[:, :nv]),
            ("dlambda/dv", dd.differential.df_dx[:, nv:], dlam[:, nv:]),
            ("Fx", dd.Fx, Fx_fd),
            ("Fu", dd.Fu, Fu_fd),
            ("Gx (friction/unilaterality)", dd.Gx, Gx_fd),
            ("Lx", dd.Lx, fd_jac_state(
                lambda z: (lambda t_: (it.calc(t_, z, u), t_.cost)[1])(it.createData()),
                state, x, H, 1).ravel()),
        ]
        for name, an, nd in blocks:
            e = err(an, nd)
            print(f"   t={idx*1e-3:<8.2f} {name:28s} {e[0]:11.2e} {e[1]:11.2e}")

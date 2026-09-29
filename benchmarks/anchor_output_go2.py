"""
Anchor-output (algebraic force) transcription of the Go2+arm soft-contact MPC,
for the AuRo revision (reviewer 1, major comment 2).

Benchmark-only module: it does NOT modify the production force-state models.

Two transcriptions of the SAME compliant contact predictor:

  (A) force-state / lifted (production, python/soft_mpc/soft_multicontact_api.py)
        y = (q, v, lambda),  lambda integrated with  lambda_dot = -Kp v_c - Kv a_c
      the force is a decision variable; the force dynamics are a defect constraint.

  (B) anchor-output (this file)
        x = (q, v),  lambda = h(q, v; p_c) = -Kp (p(q) - p_c) - Kv v_c(q, v)
      the force is an algebraic output of the state; the anchor p_c is
      reconstructed once per MPC cycle and held fixed over the horizon.

Conventions taken from the production model (verified, see report):
  - lambda is expressed in LOCAL_WORLD_ALIGNED at the contact frame;
  - it is applied to the robot as fext = jMf.act(Force(oRf^T lambda));
  - p(q) is the contact-frame origin in WORLD, v_c the LWA linear velocity;
  - Kp, Kv are isotropic (diagonal, equal entries) in the Go2 setup.

Anchor reconstruction (exact, per contact), so that h reproduces the measured
force at the first node:
        p_c = p(q_0) + Kp^-1 ( lambda_meas + Kv v_c(q_0, v_0) )
"""

import numpy as np
import pinocchio as pin
import crocoddyl


# --------------------------------------------------------------------- anchor map
class AnchorContact3D:
    """Algebraic contact-force output lambda = -Kp (p - p_c) - Kv v_c (LWA)."""

    def __init__(self, state, frameId, Kp, Kv, pinRef=pin.LOCAL_WORLD_ALIGNED):
        self.state = state
        self.pinocchio = state.pinocchio
        self.frameId = frameId
        self.Kp = float(Kp)
        self.Kv = float(Kv)
        self.pinRef = pinRef
        self.nc = 3
        self.parentId = self.pinocchio.frames[frameId].parentJoint
        self.jMf = self.pinocchio.frames[frameId].placement
        self.oPc = np.zeros(3)
        # buffers
        self.f = np.zeros(3)
        self.df_dq = np.zeros((3, state.nv))
        self.df_dv = np.zeros((3, state.nv))
        self.dABA_df = np.zeros((state.nv, 3))

    # ---- anchor reconstruction from a measured force at (q, v)
    def set_anchor_from_measurement(self, pin_data, f_meas):
        """p_c such that h(q, v; p_c) = f_meas exactly (pin_data must hold FK at (q,v))"""
        p = pin_data.oMf[self.frameId].translation
        vc = pin.getFrameVelocity(
            self.pinocchio, pin_data, self.frameId, self.pinRef
        ).linear
        self.oPc = p + (np.asarray(f_meas) + self.Kv * vc) / self.Kp
        return self.oPc

    # ---- force output
    def calc(self, pin_data):
        p = pin_data.oMf[self.frameId].translation
        vc = pin.getFrameVelocity(
            self.pinocchio, pin_data, self.frameId, self.pinRef
        ).linear
        self.f = -self.Kp * (p - self.oPc) - self.Kv * vc
        return self.f

    def calcDiff(self, pin_data):
        """d lambda / d(q, v) in the tangent space (LWA)

        d p / dq is the linear part of the LWA jacobian.
        For d v_c / dq the LWA derivative returned by pinocchio does not contain the
        contribution of the rotating frame axes, so it is rebuilt from the LOCAL
        derivative exactly as the production force-rate derivative does:
            d(oRf v_loc)/dq = oRf dv_loc/dq - skew(oRf v_loc) oJ_angular
        (validated against central finite differences: 7e-13 vs 1.9e-4).
        """
        oJ = pin.getFrameJacobian(self.pinocchio, pin_data, self.frameId, self.pinRef)
        loc_dq, _ = pin.getFrameVelocityDerivatives(
            self.pinocchio, pin_data, self.frameId, pin.LOCAL
        )
        oRf = pin_data.oMf[self.frameId].rotation
        v_loc = pin.getFrameVelocity(
            self.pinocchio, pin_data, self.frameId, pin.LOCAL
        ).linear
        dvc_dq = oRf @ loc_dq[:3] - pin.skew(oRf @ v_loc) @ oJ[3:]
        self.df_dq = -self.Kp * oJ[:3] - self.Kv * dvc_dq
        self.df_dv = -self.Kv * oJ[:3]
        return self.df_dq, self.df_dv

    # ---- contribution to the robot dynamics (identical to the production model)
    def fext(self, pin_data):
        oRf = pin_data.oMf[self.frameId].rotation
        return self.jMf.act(pin.Force(oRf.T @ self.f, np.zeros(3)))

    def update_ABAderivatives(self, pin_data, Fx_q):
        """dABA/dlambda, plus the LWA skew term added to dABA/dq (as in the production model)"""
        lJ = pin.getFrameJacobian(self.pinocchio, pin_data, self.frameId, pin.LOCAL)
        oRf = pin_data.oMf[self.frameId].rotation
        self.dABA_df = pin_data.Minv @ lJ[:3].T
        if self.pinRef != pin.LOCAL:
            Fx_q += pin_data.Minv @ lJ[:3].T @ pin.skew(oRf.T @ self.f) @ lJ[3:]
            self.dABA_df = self.dABA_df @ oRf.T
        return self.dABA_df


# ------------------------------------------------------- differential action model
class DAMAnchorOutput3D_Go2(crocoddyl.DifferentialActionModelAbstract):
    """
    Robot dynamics with algebraic contact forces lambda = h(q, v; p_c).

    Reuses the SAME crocoddyl cost sum, force-cost manager and force-constraint
    manager objects as the force-state model, so costs, weights, references,
    constraint definitions and bounds are identical by construction.
    """

    def __init__(self, state, actuation, costModelSum, anchors, forceCosts=None,
                 forceConstraints=None):
        ng = 0 if forceConstraints is None else int(forceConstraints.nr)
        crocoddyl.DifferentialActionModelAbstract.__init__(
            self, state, actuation.nu, costModelSum.nr, ng, 0
        )
        self.actuation = actuation
        self.costs = costModelSum
        self.anchors = anchors                    # list of AnchorContact3D
        self.forceCosts = forceCosts              # ForceCostManager (production class)
        self.forceConstraints = forceConstraints  # ForceConstraintManager (production class)
        self.pinocchio = state.pinocchio
        self.nc_tot = 3 * len(anchors)
        if forceConstraints is not None:
            self.g_lb = forceConstraints.lb
            self.g_ub = forceConstraints.ub

    def createData(self):
        return DADAnchorOutput3D_Go2(self)

    # -- force stack and its derivatives at the current (q, v)
    def _forces(self, data):
        f = np.zeros(self.nc_tot)
        for k, c in enumerate(self.anchors):
            f[3 * k: 3 * k + 3] = c.calc(data.pinocchio)
        return f

    def _force_jacobians(self, data):
        df_dx = np.zeros((self.nc_tot, self.state.ndx))
        nv = self.state.nv
        for k, c in enumerate(self.anchors):
            dq, dv = c.calcDiff(data.pinocchio)
            df_dx[3 * k: 3 * k + 3, :nv] = dq
            df_dx[3 * k: 3 * k + 3, nv:] = dv
        return df_dx

    def calc(self, data, x, u=None):
        nq, nv = self.state.nq, self.state.nv
        q, v = x[:nq], x[nq:]
        pin.computeAllTerms(self.pinocchio, data.pinocchio, q, v)
        pin.forwardKinematics(self.pinocchio, data.pinocchio, q, v, np.zeros(nv))
        pin.updateFramePlacements(self.pinocchio, data.pinocchio)
        data.f = self._forces(data)
        if u is not None:
            self.actuation.calc(data.multibody.actuation, x, u)
            data.fext = [pin.Force.Zero() for _ in range(self.pinocchio.njoints)]
            for c in self.anchors:
                data.fext[c.parentId] += c.fext(data.pinocchio)
            data.xout = pin.aba(
                self.pinocchio, data.pinocchio, q, v,
                data.multibody.actuation.tau, data.fext,
            )
            self.costs.calc(data.costs, x, u)
        else:
            data.xout = np.zeros(nv)
            self.costs.calc(data.costs, x)
        data.cost = data.costs.cost
        if self.forceCosts is not None:
            data.cost += self.forceCosts.calc(data, data.f)
        if self.forceConstraints is not None:
            data.g = self.forceConstraints.calc(data.f).copy()
        return data.xout, data.cost

    def calcDiff(self, data, x, u=None):
        nq, nv = self.state.nq, self.state.nv
        q, v = x[:nq], x[nq:]
        if u is not None:
            self.actuation.calcDiff(data.multibody.actuation, x, u)
            # ABA derivatives at fixed external forces (same call as the production model)
            aba_dq, aba_dv, aba_dtau = pin.computeABADerivatives(
                self.pinocchio, data.pinocchio, q, v,
                data.multibody.actuation.tau, data.fext,
            )
            Minv = np.array(data.pinocchio.Minv, copy=True)
            data.Fx[:, :nv] = aba_dq
            data.Fx[:, nv:] = aba_dv
            data.Fx += Minv @ data.multibody.actuation.dtau_dx
            data.Fu = aba_dtau @ data.multibody.actuation.dtau_du
        else:
            Minv = np.array(data.pinocchio.Minv, copy=True)
        # kinematics derivatives needed by the anchor map (velocity partials)
        pin.computeForwardKinematicsDerivatives(
            self.pinocchio, data.pinocchio, q, v, np.zeros(nv)
        )
        pin.updateFramePlacements(self.pinocchio, data.pinocchio)
        df_dx = self._force_jacobians(data)
        data.df_dx = df_dx
        if u is not None:
            # contact-force contribution: dABA/dlambda * dlambda/dx (+ LWA skew term)
            for k, c in enumerate(self.anchors):
                lJ = pin.getFrameJacobian(self.pinocchio, data.pinocchio, c.frameId, pin.LOCAL)
                oRf = data.pinocchio.oMf[c.frameId].rotation
                dABA_df = Minv @ lJ[:3].T
                if c.pinRef != pin.LOCAL:
                    data.Fx[:, :nv] += Minv @ lJ[:3].T @ pin.skew(oRf.T @ c.f) @ lJ[3:]
                    dABA_df = dABA_df @ oRf.T
                c.dABA_df = dABA_df
                data.Fx += dABA_df @ df_dx[3 * k: 3 * k + 3]
            self.costs.calcDiff(data.costs, x, u)
            data.Lu = data.costs.Lu
            data.Luu = data.costs.Luu
            data.Lxu = data.costs.Lxu
        else:
            self.costs.calcDiff(data.costs, x)
        data.Lx = np.array(data.costs.Lx, copy=True)
        data.Lxx = np.array(data.costs.Lxx, copy=True)
        # force cost pulled back through the anchor map: l(lambda(x))
        if self.forceCosts is not None:
            Lf, Lff = self.forceCosts.calcDiff(data, data.f)
            data.Lx += np.asarray(Lf).ravel() @ df_dx
            data.Lxx += df_dx.T @ np.asarray(Lff) @ df_dx     # Gauss-Newton
        # force constraints pulled back: g(lambda(x))
        if self.forceConstraints is not None:
            data.Gx = self.forceConstraints.calcDiff(data.f) @ df_dx
            if u is not None:
                data.Gu = np.zeros((self.ng, self.nu))
        return


class DADAnchorOutput3D_Go2(crocoddyl.DifferentialActionDataAbstract):
    def __init__(self, am):
        crocoddyl.DifferentialActionDataAbstract.__init__(self, am)
        nv, ndx, nu = am.state.nv, am.state.ndx, am.nu
        self.pinocchio = am.pinocchio.createData()
        self.actuation_data = am.actuation.createData()
        self.multibody = crocoddyl.DataCollectorActMultibody(
            self.pinocchio, self.actuation_data
        )
        self.costs = am.costs.createData(self.multibody)
        self.f = np.zeros(am.nc_tot)
        self.df_dx = np.zeros((am.nc_tot, ndx))
        self.fext = [pin.Force.Zero() for _ in range(am.pinocchio.njoints)]
        self.Fx = np.zeros((nv, ndx))
        self.Fu = np.zeros((nv, nu))
        self.Lx = np.zeros(ndx)
        self.Lu = np.zeros(nu)
        self.Lxx = np.zeros((ndx, ndx))
        self.Lxu = np.zeros((ndx, nu))
        self.Luu = np.zeros((nu, nu))
        self.Gx = np.zeros((am.ng, ndx))
        self.Gu = np.zeros((am.ng, nu))
        self.g = np.zeros(am.ng)
        # force-cost manager scratch (same attribute names as the production data)
        self.Lf = np.zeros(3)
        self.Lff = np.zeros((3, 3))
        self.f_residual = np.zeros(3)


# ------------------------------------------------------------- integrated model
class IAMAnchorOutput_Go2(crocoddyl.ActionModelAbstract):
    """Explicit Euler integration, mirroring IAMSoftContactDynamics3D_Go2 exactly
    (dx = [v dt + a dt^2, a dt], cost = dt * differential cost)."""

    def __init__(self, dam, dt=1e-2, withCostResidual=True):
        crocoddyl.ActionModelAbstract.__init__(
            self, dam.state, dam.nu, dam.nr, dam.ng, 0
        )
        self.differential = dam
        self.dt = dt
        self.withCostResidual = withCostResidual
        if dam.ng > 0:
            self.g_lb = dam.g_lb
            self.g_ub = dam.g_ub

    def createData(self):
        return IADAnchorOutput_Go2(self)

    def calc(self, data, x, u=None):
        nv = self.state.nv
        v = x[self.state.nq:]
        if u is not None:
            self.differential.calc(data.differential, x, u)
            a = data.differential.xout
            data.dx[:nv] = v * self.dt + a * self.dt ** 2
            data.dx[nv:] = a * self.dt
            data.xnext = self.state.integrate(x, data.dx)
            data.cost = self.dt * data.differential.cost
        else:
            self.differential.calc(data.differential, x)
            data.dx = np.zeros(self.state.ndx)
            data.xnext = x.copy()
            data.cost = data.differential.cost
        if self.ng > 0:
            data.g = data.differential.g.copy()
        return data.xnext, data.cost

    def calcDiff(self, data, x, u=None):
        ndx, nv, nu = self.state.ndx, self.state.nv, self.nu
        if u is not None:
            self.differential.calcDiff(data.differential, x, u)
            da_dx, da_du = data.differential.Fx, data.differential.Fu
            data.Fx[:nv, :] = da_dx * self.dt ** 2
            data.Fx[nv:, :] = da_dx * self.dt
            data.Fx[:nv, nv:] += self.dt * np.eye(nv)
            data.Fu[:nv, :] = da_du * self.dt ** 2
            data.Fu[nv:, :] = da_du * self.dt
            self.state.JintegrateTransport(x, data.dx, data.Fx, crocoddyl.Jcomponent.second)
            data.Fx += self.state.Jintegrate(x, data.dx, crocoddyl.Jcomponent.first).tolist()[0]
            self.state.JintegrateTransport(x, data.dx, data.Fu, crocoddyl.Jcomponent.second)
            data.Lx = data.differential.Lx * self.dt
            data.Lxx = data.differential.Lxx * self.dt
            data.Lxu = data.differential.Lxu * self.dt
            data.Lu = data.differential.Lu * self.dt
            data.Luu = data.differential.Luu * self.dt
        else:
            self.differential.calcDiff(data.differential, x)
            data.Fx = self.state.Jintegrate(x, data.dx, crocoddyl.Jcomponent.first).tolist()[0]
            data.Lx = data.differential.Lx.copy()
            data.Lxx = data.differential.Lxx.copy()
        if self.ng > 0:
            data.Gx = data.differential.Gx.copy()
            if u is not None:
                data.Gu = np.zeros((self.ng, nu))
        return


class IADAnchorOutput_Go2(crocoddyl.ActionDataAbstract):
    def __init__(self, am):
        crocoddyl.ActionDataAbstract.__init__(self, am)
        self.differential = am.differential.createData()
        self.dx = np.zeros(am.state.ndx)
        self.xnext = np.zeros(am.state.nx)
        self.Fx = np.zeros((am.state.ndx, am.state.ndx))
        self.Fu = np.zeros((am.state.ndx, am.nu))
        self.Gx = np.zeros((am.ng, am.state.ndx))
        self.Gu = np.zeros((am.ng, am.nu))
        self.g = np.zeros(am.ng)

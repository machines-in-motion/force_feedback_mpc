"""
Closed-loop anchor-update MPC for the Go2 multi-contact task (Han et al. baseline).

Subclasses the production force-state wrapper so that the robot model, costs, force
costs, hard force constraints, horizon, time step and solver settings are IDENTICAL:
the only difference is the OCP transcription. The contact force is the algebraic output

    lambda = -K (p(q) - p_c) - B pdot(q, qdot),

with the anchor p_c reconstructed once per MPC cycle from the measured force, and the
optimization state is the classical (q, qdot).
"""
import os
import sys

import numpy as np
import crocoddyl
import mim_solvers
import pinocchio as pin

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "benchmarks"))
from anchor_output_go2 import (  # noqa: E402
    AnchorContact3D, DAMAnchorOutput3D_Go2, IAMAnchorOutput_Go2,
)
from Go2MPC_wrapper_soft import Go2MPCSoft  # noqa: E402


class Go2MPCAnchor(Go2MPCSoft):
    """Anchor-update transcription of the same OCP."""

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)
        self.nf = 3 * len(self.ee_frame_names)
        dam0 = self.running_models[0].differential
        self.pin_state = dam0.state
        self._pdata = self.pin_state.pinocchio.createData()
        # one anchor per production contact, sharing its parameters
        self.anchors = [AnchorContact3D(self.pin_state, ct.frameId, ct.Kp, ct.Kv, ct.pinRef)
                        for ct in dam0.contacts.contacts]
        # anchor models reusing the production cost / force-cost / constraint objects
        iams = []
        for iam_s in self.running_models:
            dam_s = iam_s.differential
            dam = DAMAnchorOutput3D_Go2(
                dam_s.state, dam_s.actuation, dam_s.costs, self.anchors,
                forceCosts=dam_s.forceCosts if dam_s.with_force_cost else None,
                forceConstraints=iam_s.forceConstraints,
            )
            iams.append(IAMAnchorOutput_Go2(dam, dt=iam_s.dt, withCostResidual=True))
        self.anchor_models = iams
        nq, nv = self.pin_state.nq, self.pin_state.nv
        y0 = np.array(self.ocp.x0)
        x0 = y0[:nq + nv]
        # the anchors must reproduce the initial contact force: with the default
        # p_c = 0 the map would predict forces of order K * ||p(q)||
        self.update_anchors(x0[:nq], x0[nq:], y0[nq + nv:])
        self.ocp = crocoddyl.ShootingProblem(x0.copy(), iams[:-1], iams[-1])
        self.createSolver()                       # same settings as the force-state MPC
        # same initial guess as the force-state MPC: replicated x0 and quasi-static torques
        self.xs = [x0.copy() for _ in range(self.HORIZON + 1)]
        self.us = list(self.ocp.quasiStatic([x0.copy() for _ in range(self.HORIZON)]))

    # -------------------------------------------------------------- anchor update
    def update_anchors(self, q, dq, f):
        """reconstruct p_c so that the model reproduces the measured force at (q, dq)"""
        pin.forwardKinematics(self.pin_state.pinocchio, self._pdata, q, dq)
        pin.updateFramePlacements(self.pin_state.pinocchio, self._pdata)
        for k, a in enumerate(self.anchors):
            a.set_anchor_from_measurement(self._pdata, f[3 * k: 3 * (k + 1)])

    def horizon_forces(self):
        """forces predicted by the anchor map along the returned trajectory"""
        nq = self.pin_state.nq
        out = []
        for x in self.solver.xs:
            x = np.array(x)
            pin.forwardKinematics(self.pin_state.pinocchio, self._pdata, x[:nq], x[nq:])
            pin.updateFramePlacements(self.pin_state.pinocchio, self._pdata)
            out.append(np.concatenate([a.calc(self._pdata) for a in self.anchors]))
        return np.array(out).reshape(-1, len(self.ee_frame_names), 3)

    # ------------------------------------------------------------------- solving
    def updateAndSolve2(self, q, dq, f):
        self.update_anchors(q, dq, f)
        x = np.hstack([q, dq])
        self.solver.problem.x0 = x
        xs_list = list(self.solver.xs)
        self.xs = xs_list[1:] + [xs_list[-1]]
        self.xs[0] = x
        us_list = list(self.us)
        self.us = us_list[1:] + [us_list[-1]]
        self.solver.solve(self.xs, self.us, self.max_iterations)
        self.xs, self.us = self.solver.xs, self.solver.us
        return self.getSolution()

    def solve(self):
        self.solver.solve(self.xs, self.us, self.max_iterations)
        self.xs, self.us = self.solver.xs, self.solver.us
        return self.getSolution()

    def getSolution(self, k=None):
        x_idx = 1 if k is None else k
        u_idx = 0 if k is None else k
        x = np.array(self.xs[x_idx])
        nq = self.pin_state.nq
        pin.forwardKinematics(self.pin_state.pinocchio, self._pdata, x[:nq], x[nq:])
        pin.updateFramePlacements(self.pin_state.pinocchio, self._pdata)
        f = np.concatenate([a.calc(self._pdata) for a in self.anchors])
        return dict(
            position=x[:3],
            orientation=np.array([x[6], x[3], x[4], x[5]]),
            velocity=x[nq:nq + 3], omega=x[nq + 3:nq + 6],
            q=x[7:25], dq=x[nq + 6:nq + 24],
            f_lf=f[:3], f_rf=f[3:6], f_lh=f[6:9], f_rh=f[9:12], f_ee=f[12:],
            tau=self.us[u_idx],
            constraint_norm=self.solver.constraint_norm,
        )

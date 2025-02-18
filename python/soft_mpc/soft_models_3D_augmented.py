'''
Prototype Differential Action Model for augmented state space model 
This formulation supposedly allows force feedback in MPC
The derivatives of ABA in the DAM are unchanged, except we need to implement
as well d(ABA)/df . Also df/dt and its derivatives are implemented

In the IAM , a simple Euler integration (explicit) is used for the contact force
and the partials are aggregated using DAM partials.
'''
from croco_mpc_utils.utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger

import numpy as np
np.set_printoptions(precision=6, linewidth=180, suppress=True)
np.random.seed(1)

import crocoddyl
import pinocchio as pin

# # Custom state model 
# class StateSoftContact3D(crocoddyl.StateAbstract):
#     def __init__(self, rmodel, nc):
#         crocoddyl.StateAbstract.__init__(self, rmodel.nq + rmodel.nv + nc, 2*rmodel.nv + nc)
#         self.pinocchio = rmodel
#         self.nc = nc
#         self.nv = (self.ndx - self.nc)//2
#         self.nq = self.nx - self.nc - self.nv
#         self.ny = self.nq + self.nv + self.nc
#         self.ndy = 2*self.nv + self.nc
#         # print("Augmented state ny = ", self.ny)
#         # print("Augmented state ndy = ", self.ndy)

#     def diff(self, y0, y1):
#         yout = np.zeros(self.ny)
#         nq = self.pinocchio.nq
#         # nv = self.pinocchio.nv
#         yout[:nq] = pin.difference(self.pinocchio, y0[:nq], y1[:nq])
#         yout[nq:] = y1[nq:] - y0[nq:]
#         return yout

#     def integrate(self, y, dy):
#         yout = np.zeros(self.ndy)
#         nq = self.pinocchio.nq
#         # nv = self.pinocchio.nv
#         yout[:nq] = pin.integrate(self.pinocchio, y[:nq], dy[:nq])
#         yout[nq:] = y[nq:] + dy[nq:]
#         return yout

#     def Jintegrate(self, y, dy, Jfirst):
#         '''
#         Default values :
#          firstsecond = crocoddyl.Jcomponent.first 
#          op = crocoddyl.addto
#         '''
#         Jfirst[:self.nv, :self.nv] = pin.dIntegrate(self.pinocchio, y[:self.nq], dy[:self.nv], pin.ARG0) #, crocoddyl.addto)
#         Jfirst[self.nv:2*self.nv, self.nv:2*self.nv] += np.eye(self.nv)
#         Jfirst[-self.nc:, -self.nc:] += np.eye(self.nc)
    
#     def JintegrateTransport(self, y, dy, Jin, firstsecond):
#         if(firstsecond == crocoddyl.Jcomponent.first):
#             pin.dIntegrateTransport(self.pinocchio, y[:self.nq], dy[:self.nv], Jin[:self.nv], pin.ARG0)
#         elif(firstsecond == crocoddyl.Jcomponent.second):
#             pin.dIntegrateTransport(self.pinocchio, y[:self.nq], dy[:self.nv], Jin[:self.nv], pin.ARG1)
#         else:
#             logger.error("wrong arg firstsecond")

import force_feedback_mpc
# Integrated action model and data 
class IAMSoftContactDynamics3D(crocoddyl.ActionModelAbstract): #IntegratedActionModelAbstract
    def __init__(self, dam, dt=1e-3, withCostResidual=True, frictionConeConstaints = []):
        # crocoddyl.ActionModelAbstract.__init__(self, dam.state, dam.nu, dam.costs.nr + 3)
        ng = int(dam.ng + 3 + len(frictionConeConstaints))
        crocoddyl.ActionModelAbstract.__init__(self, force_feedback_mpc.StateSoftContact(dam.pinocchio, 3), dam.nu, dam.costs.nr + 3, ng, 0)
        # crocoddyl.ActionModelAbstract.__init__(self, crocoddyl.StateVector(dam.state.nq + dam.state.nv + 3), dam.nu, dam.costs.nr + 3)
        self.differential = dam
        self.stateSoft = force_feedback_mpc.StateSoftContact(dam.pinocchio, 3)
        self.dt = dt
        self.nc = 3
        self.ny = self.stateSoft.nx 
        self.ndy = self.stateSoft.ndx 
        self.withCostResidual = withCostResidual
        self.withForceConstraint = False
        self.force_lb = np.array([-np.inf]*self.differential.nc)
        self.force_ub = np.array([np.inf]*self.differential.nc)
        if(len(frictionConeConstaints)==0):
            self.withFrictionConeConstraint = False

    def createData(self):
        data = IADSoftContactDynamics3D(self)
        return data
    
    def calc(self, data, y, u=None):
        nx = self.differential.state.nx
        nv = self.differential.state.nv
        nq = self.differential.state.nq
        # logger.debug(nx)
        # logger.debug(data.differential.xout)
        nc = self.nc
        x = y[:nx]
        f = y[-nc:]
        # q = x[:self.state.nq]
        v = x[-nv:]

        if(u is not None):
            # self.control.calc(data.control, 0., u)
            self.differential.calc(data.differential, x, f, u) 
            a = data.differential.xout
            fdot = data.differential.fout
            data.dx[:nv] = v*self.dt + a*self.dt**2
            data.dx[nv:2*nv] = a*self.dt
            data.dx[-nc:] = fdot*self.dt
            data.xnext = self.stateSoft.integrate(y, data.dx)
            data.cost = self.dt*data.differential.cost
            if(self.withCostResidual):
                data.r = data.differential.r
            # # compute constraint residual
            # if(self.with_force_constraint):
            #     data.g[self.differential.ng: self.differential.ng+nc] = f
            # if(self.withFrictionConeConstraint){
            #     # Resize the constraint matrices of IAM 
            #     data.friction_cone_residual[0] = friction_coef_ * f(2) - sqrt(f(0)*f(0) + f(1)*f(1));
            #     data.g.tail(1) << d->friction_cone_residual[0];
            #     // std::cout << "friction cone residual = " << d->g.tail(1) << std::endl;

            #     # std::cout << " residual = " << d->friction_cone_residual[0] << std::endl;
            #     # // std::cout << "resize IAM for nc=" << nf_ << " friction constraints" << std::endl;
            #     # d->resizeIneqConstraint(this);
            #     # // std::cout << "g.tail(nf_) = " << d->g.tail(nf_) << std::endl;
            #     # // Iterate over friction models
            #     # // std::cout << "Loop over constraint models " << std::endl;
            #     # for(std::size_t i=0; i<friction_constraints_.size(); i++){
            #     #   // std::cout << "constraint model " << i << std::endl;
            #     #   // calc if constraint is active and data is well defined
            #     #   if(friction_constraints_[i]->get_active() && friction_datas_[i] != nullptr){
            #     #      friction_constraints_[i]->calc(friction_datas_[i], f);
            #     #     //  std::cout << " fill out residual g from index " << differential_->get_ng() + nc_ + i << " to " << differential_->get_ng() + nc_ + i +1 << std::endl;
            #     #      d->g.segment(differential_->get_ng() + nc_ + i, 1) << friction_datas_[i]->residual;
            #     #   }
            #     #   // fill out partial derivatives of the IAM
            #     # }
            #     # std::cout << "Finished " << std::endl;
            #     # std::cout << "g.tail(nf_) = " << d->g.tail(nf_) << std::endl;
            # }
            # # // compute cost residual
            # # if (with_cost_residual_) {
            # #     d->r.head(differential_->get_nr()) = diff_data_soft->r;
            # #     d->r.tail(nc_) = diff_data_soft->f_residual;
            # # }
        else:
            pass
                

    def calcDiff(self, data, y, u=None):
        nx = self.differential.state.nx
        ndx = self.differential.state.ndx
        nv = self.differential.state.nv
        nu = self.differential.nu
        nc = self.nc
        x = y[:nx]
        f = y[-nc:]
        if(u is not None):
            # Calc forward dyn derivatives
            self.differential.calcDiff(data.differential, x, f, u)
            da_dx = data.differential.Fx 
            da_du = data.differential.Fu

            # Fill out blocks
            data.Fx[:nv,:ndx] = da_dx*self.dt**2
            data.Fx[nv:ndx, :ndx] = da_dx*self.dt
            data.Fx[:nv, nv:ndx] += self.dt * np.eye(nv)
            data.Fu[:nv, :] = da_du * self.dt**2
            data.Fu[nv:ndx, :] = da_du * self.dt
            # New block from augmented dynamics (top right corner)
            data.Fx[:nv, -nc:] = data.differential.dABA_df * self.dt**2
            data.Fx[nv:ndx, -nc:] = data.differential.dABA_df * self.dt
            # New block from augmented dynamics (bottom right corner)
            data.Fx[-nc:,-nc:] = np.eye(3) + data.differential.dfdt_df*self.dt
            # New block from augmented dynamics (bottom left corner)
            data.Fx[-nc:, :ndx] = data.differential.dfdt_dx * self.dt
            
            data.Fu[-nc:, :] = data.differential.dfdt_du * self.dt

            self.stateSoft.JintegrateTransport(y, data.dx, data.Fx, crocoddyl.Jcomponent.second)
            data.Fx += self.stateSoft.Jintegrate(y, data.dx, crocoddyl.Jcomponent.first).tolist()[0]  # add identity to Fx = d(x+dx)/dx = d(q,v)/d(q,v)
            data.Fx[-nc:, -nc:] -= np.eye(nc)
            self.stateSoft.JintegrateTransport(y, data.dx, data.Fu, crocoddyl.Jcomponent.second)

            # d->Lx.noalias() = time_step_ * d->differential->Lx;
            data.Lx[:ndx] = data.differential.Lx*self.dt
            data.Lx[-nc:] = data.differential.Lf*self.dt
            data.Lxx[:ndx,:ndx] = data.differential.Lxx*self.dt
            data.Lxx[-nc:,-nc:] = data.differential.Lff*self.dt
            data.Lxu[:ndx, :nu] = data.differential.Lxu*self.dt
            data.Lu = data.differential.Lu*self.dt
            data.Luu = data.differential.Luu*self.dt
        else:
            pass

class IADSoftContactDynamics3D(crocoddyl.ActionDataAbstract): #IntegratedActionDataAbstract
    '''
    Creates a data class for IAM
    '''
    def __init__(self, am):
        # super().__init__(am)
        crocoddyl.ActionDataAbstract.__init__(self, am)
        self.differential = am.differential.createData()
        self.xnext = np.zeros(am.ny)
        self.dx = np.zeros(am.ndy)
        self.Fx = np.zeros((am.ndy, am.ndy))
        self.Fu = np.zeros((am.ndy, am.nu))
        # self.r = am.differential.costs.nr
        self.Lx = np.zeros(am.ny)
        self.Lu = np.zeros(am.differential.actuation.nu)
        self.Lxx = np.zeros((am.ny, am.ny))
        self.Lxu = np.zeros((am.ny, am.differential.actuation.nu))
        self.Luu = np.zeros((am.differential.actuation.nu, am.differential.actuation.nu))



# Differential action model and data
class DAMSoftContactDynamics3D(crocoddyl.DifferentialActionModelAbstract):
    '''
    Computes the forward dynamics under visco-elastic (spring damper) force
    '''
    def __init__(self, stateMultibody, actuationModel, costModelSum, frameId, Kp=1e3, Kv=60, oPc=np.zeros(3), pinRefFrame=pin.LOCAL, constraintModelManager=None):
        # super(DAMSoftContactDynamics, self).__init__(stateMultibody, actuationModel.nu, costModelSum.nr)
        if(constraintModelManager is None):
            ng = 0
            nh = 0
        else:
            ng = constraintModelManager.ng
            nh = constraintModelManager.nh
        crocoddyl.DifferentialActionModelAbstract.__init__(self, stateMultibody, actuationModel.nu, costModelSum.nr, ng, nh)
        self.Kp = Kp 
        self.Kv = Kv
        self.pinRef = pinRefFrame
        self.frameId = frameId
        self.with_armature = False
        self.armature = np.zeros(self.state.nq)
        self.oPc = oPc
        # To complete DAMAbstract into sth like DAMFwdDyn
        self.actuation = actuationModel
        self.costs = costModelSum
        if(constraintModelManager is not None):
            self.constraints = constraintModelManager
        self.pinocchio = stateMultibody.pinocchio
        # hard coded costs 
        self.with_force_cost = False
        self.active_contact = True
        self.nc = 3

        self.parentId = self.pinocchio.frames[self.frameId].parent
        self.jMf = self.pinocchio.frames[self.frameId].placement

        # Hard-coded cost on force and gravity reg
        self.with_force_cost = False
        self.force_weight = np.zeros(self.nc)
        self.force_rate_reg_weight = np.zeros(self.nc)
        self.force_des = np.zeros(self.nc)
        self.with_gravity_torque_reg = False
        self.tau_grav_weight = 0.
        self.with_force_rate_reg_cost = False

    def set_active_contact(self, active):
        self.active_contact = active

    def createData(self):
        '''
            The data is created with a custom data class that contains the filtered torque tau_plus and the activation
        '''
        data = DADSoftContactDynamics(self)
        return data

    def set_force_cost(self, f_des, f_weight):
        assert(len(f_des) == self.nc)
        self.with_force_cost = True
        self.f_des = f_des
        self.f_weight = f_weight

    def calc(self, data, x, f, u=None):
        '''
        Compute joint acceleration based on state, force and torques
        '''
        # logger.debug("CALC")
        q = x[:self.state.nq]
        v = x[self.state.nq:]
        pin.computeAllTerms(self.pinocchio, data.pinocchio, q, v)
        pin.forwardKinematics(self.pinocchio, data.pinocchio, q, v, np.zeros(self.state.nv))
        pin.updateFramePlacements(self.pinocchio, data.pinocchio)
        oRf = data.pinocchio.oMf[self.frameId].rotation
        
        if(u is not None):
            # Actuation calc
            self.actuation.calc(data.multibody.actuation, x, u)

            # Compute forward dynamics ddq = ABA(q,dq,tau,fext)
            if(self.active_contact):
                # Compute external wrench for LOCAL f
                data.fext[self.parentId] = self.jMf.act(pin.Force(f, np.zeros(3)))
                data.fext_copy = data.fext.copy()
                # Rotate if not f not in LOCAL
                if(self.pinRef != pin.LOCAL):
                    data.fext[self.parentId] = self.jMf.act(pin.Force(oRf.T @ f, np.zeros(3)))
                
                data.xout = pin.aba(self.pinocchio, data.pinocchio, q, v, data.multibody.actuation.tau, data.fext) 

                # Compute time derivative of contact force : need to forward kin with current acc
                pin.forwardKinematics(self.pinocchio, data.pinocchio, q, v, data.xout)
                la = pin.getFrameAcceleration(self.pinocchio, data.pinocchio, self.frameId, pin.LOCAL).linear         
                lv = pin.getFrameVelocity(self.pinocchio, data.pinocchio, self.frameId, pin.LOCAL).linear
                data.fout = -self.Kp * lv - self.Kv * la
                data.fout_copy = data.fout.copy()
                # Rotate if not f not in LOCAL
                if(self.pinRef != pin.LOCAL):
                    oa = pin.getFrameAcceleration(self.pinocchio, data.pinocchio, self.frameId, pin.LOCAL_WORLD_ALIGNED).linear         
                    ov = pin.getFrameVelocity(self.pinocchio, data.pinocchio, self.frameId, pin.LOCAL_WORLD_ALIGNED).linear
                    data.fout = -self.Kp * ov - self.Kv * oa
                    assert(np.linalg.norm(data.fout - oRf @ data.fout_copy) < 1e-3)
            else:
                data.xout = pin.aba(self.pinocchio, data.pinocchio, q, v, data.multibody.actuation.tau)
                # data.fout = np.zeros(3)

            pin.updateGlobalPlacements(self.pinocchio, data.pinocchio)
            
            # Computing the cost value and residuals
            self.costs.calc(data.costs, x, u) 
            data.cost = data.costs.cost
            # d->residual.head(this->get_costs()->get_nr()) = d->r;

            # Add hard-coded cost
            if(self.with_force_cost):
                # Compute force residual and add force cost to total cost
                data.f_residual = f - self.f_des
                data.cost += 0.5 * self.f_weight * data.f_residual.T @ data.f_residual
            #TODO : gravity torque reg cost ( in contact and not in contact)
            #TODO : force rate reg cost
            
            # Constraints (on multibody state x=(q,v))
            if (self.constraints is not None):
                # data.constraints.resize(self, data)
                self.constraints.calc(data.constraints, x, u)

        else:
            pass

    def calcDiff(self, data, x, f, u=None):
        '''
        Compute derivatives 
        '''
        q = x[:self.state.nq]
        v = x[self.state.nq:]
        oRf = data.pinocchio.oMf[self.frameId].rotation
        if(u is not None):
            # Actuation calcDiff
            self.actuation.calcDiff(data.multibody.actuation, x, u)

            if(self.active_contact):
                # Compute Jacobian
                lJ = pin.getFrameJacobian(self.pinocchio, data.pinocchio, self.frameId, pin.LOCAL)            
                
                # Derivatives of data.xout (ABA) w.r.t. x and u in LOCAL (same in WORLD)
                aba_dq, aba_dv, aba_dtau = pin.computeABADerivatives(self.pinocchio, data.pinocchio, q, v, data.multibody.actuation.tau, data.fext)
                data.Fx[:,:self.state.nq] = aba_dq 
                data.Fx[:,self.state.nq:] = aba_dv 
                data.Fx += data.pinocchio.Minv @ data.multibody.actuation.dtau_dx
                data.Fu = aba_dtau @ data.multibody.actuation.dtau_du
                # Compute derivatives of data.xout (ABA) w.r.t. f in LOCAL 
                data.dABA_df = data.pinocchio.Minv @ lJ[:3].T @ self.pinocchio.frames[self.frameId].placement.rotation @ np.eye(3) 
            
                # Skew term added to RNEA derivatives when force is expressed in LWA
                if(self.pinRef != pin.LOCAL):
                    # logger.debug("corrective term aba LWA : \n"+str(data.pinocchio.Minv @ lJ[:3].T @ pin.skew(oRf.T @ f) @ lJ[3:]))
                    data.Fx[:,:self.state.nq] += data.pinocchio.Minv @ lJ[:3].T @ pin.skew(oRf.T @ f) @ lJ[3:]
                    # Rotate dABA/df
                    data.dABA_df = data.dABA_df @ oRf.T 

                # Derivatives of data.fout in LOCAL : important >> UPDATE FORWARD KINEMATICS with data.xout
                pin.computeAllTerms(self.pinocchio, data.pinocchio, q, v)
                pin.forwardKinematics(self.pinocchio, data.pinocchio, q, v, data.xout)
                pin.updateFramePlacements(self.pinocchio, data.pinocchio)
                lv_dq, lv_dv = pin.getFrameVelocityDerivatives(self.pinocchio, data.pinocchio, self.frameId, pin.LOCAL)
                lv_dx = np.hstack([lv_dq, lv_dv])
                _, a_dq, a_dv, a_da = pin.getFrameAccelerationDerivatives(self.pinocchio, data.pinocchio, self.frameId, pin.LOCAL)
                da_dx = np.zeros((3,self.state.nx))
                da_dx[:,:self.state.nq] = a_dq[:3] + a_da[:3] @ data.Fx[:,:self.state.nq] # same as aba_dq here
                da_dx[:,self.state.nq:] = a_dv[:3] + a_da[:3] @ data.Fx[:,self.state.nq:] # same as aba_dv here
                da_du = a_da[:3] @ data.Fu
                da_df = a_da[:3] @ data.dABA_df
                # Deriv of lambda dot
                data.dfdt_dx = -self.Kp*lv_dx[:3] - self.Kv*da_dx[:3]
                data.dfdt_du = -self.Kv*da_du
                data.dfdt_df = -self.Kv*da_df
                ldfdt_dx_copy = data.dfdt_dx.copy()
                ldfdt_du_copy = data.dfdt_du.copy()
                ldfdt_df_copy = data.dfdt_df.copy()
                # Rotate dfout_dx if not LOCAL 
                if(self.pinRef != pin.LOCAL):
                    oJ = pin.getFrameJacobian(self.pinocchio, data.pinocchio, self.frameId, pin.LOCAL_WORLD_ALIGNED)
                    data.dfdt_dx[:,:self.state.nq] = oRf @ ldfdt_dx_copy[:,:self.state.nq] - pin.skew(oRf @ data.fout_copy) @ oJ[3:]
                    data.dfdt_dx[:,self.state.nq:] = oRf @ ldfdt_dx_copy[:,self.state.nq:] 
                    data.dfdt_du = oRf @ ldfdt_du_copy
                    data.dfdt_df = oRf @ ldfdt_df_copy
            else:
                # Computing the free forward dynamics with ABA derivatives
                aba_dq, aba_dv, aba_dtau = pin.computeABADerivatives(self.pinocchio, data.pinocchio, q, v, data.multibody.actuation.tau)
                data.Fx[:,:self.state.nq] = aba_dq 
                data.Fx[:,self.state.nq:] = aba_dv 
                data.Fx += data.pinocchio.Minv @ data.multibody.actuation.dtau_dx
                data.Fu = aba_dtau @ data.multibody.actuation.dtau_du
            assert(np.linalg.norm(aba_dtau - data.pinocchio.Minv) <1e-4)
            self.costs.calcDiff(data.costs, x, u)
            data.Lx = data.costs.Lx
            data.Lu = data.costs.Lu
            data.Lxx = data.costs.Lxx
            data.Lxu = data.costs.Lxu
            data.Luu = data.costs.Luu
            # add hard-coded cost
            if(self.active_contact and self.with_force_cost):
                data.f_residual = f - self.f_des
                data.Lf = self.f_weight * data.f_residual.T
                data.Lff = self.f_weight * np.eye(3)
            # TODO: gravity torque reg cost 
            # TODO: force rate reg cost
            
            # Constraints (on multibody state x=(q,v))
            if (self.constraints is not None):
                # data.constraints.resize(self, data)
                self.constraints.calcDiff(data.constraints, x, u)
        else:
            pass

class DADSoftContactDynamics(crocoddyl.DifferentialActionDataAbstract):
    '''
    Creates a data class with differential and augmented matrices from IAM (initialized with stateVector)
    '''
    def __init__(self, am):
        # super().__init__(am)
        crocoddyl.DifferentialActionDataAbstract.__init__(self, am)
        # Force model + derivatives
        self.fout = np.zeros(am.nc)
        self.fout_copy = np.zeros(am.nc)
        self.dfdt_dx = np.zeros((3,am.state.nx))
        self.dfdt_du = np.zeros((3,am.nu))
        self.dfdt_df = np.zeros((3,3))  
        # ABA model derivatives
        self.Fx = np.zeros((am.state.nq, am.state.nx))
        self.Fu = np.zeros((am.state.nq, am.nu))
        self.dABA_df = np.zeros((am.state.nq, am.nc))
        # Cost derivatives
        self.Lx = np.zeros(am.state.nx)
        self.Lu = np.zeros(am.actuation.nu)
        self.Lxx = np.zeros((am.state.nx, am.state.nx))
        self.Lxu = np.zeros((am.state.nx, am.actuation.nu))
        self.Luu = np.zeros((am.actuation.nu, am.actuation.nu))
        self.Lf = np.zeros(am.nc)
        self.Lff = np.zeros((am.nc, am.nc))
        self.f_residual = np.zeros(am.nc)
        # External wrench
        self.fext = [pin.Force.Zero() for _ in range(am.pinocchio.njoints)]
        self.fext_copy = [pin.Force.Zero() for _ in range(am.pinocchio.njoints)]
        # Data containers
        self.pinocchio  = am.pinocchio.createData()
        self.actuation_data = am.actuation.createData()
        self.multibody = crocoddyl.DataCollectorActMultibody(self.pinocchio, self.actuation_data)
        # self.costs = am.costs.createData(crocoddyl.DataCollectorMultibody(self.pinocchio))
        self.costs = am.costs.createData(self.multibody)
        if(am.constraints is not None):
            self.constraints = am.constraints.createData(self.multibody)    


from mim_robots.robot_loader import load_pinocchio_wrapper
robot = load_pinocchio_wrapper('iiwa')
state = crocoddyl.StateMultibody(robot.model)
actuation = crocoddyl.ActuationModelFull(state)
costs = crocoddyl.CostModelSum(state, actuation.nu)
constraintModelManager = crocoddyl.ConstraintModelManager(state, actuation.nu)
frameId = robot.model.getFrameId('contact')
Kp = 0 ; Kv = 0 ; oPc = np.zeros(3)

# Test create data 
dam = DAMSoftContactDynamics3D(state, actuation, costs, frameId, Kp, Kv, oPc, pin.LOCAL_WORLD_ALIGNED, constraintModelManager)
dad = dam.createData()

# Test calc and calcDiff
q = pin.randomConfiguration(robot.model)
v = np.zeros(robot.model.nv)
x = np.concatenate([q, v])
u = np.random.rand(actuation.nu)
f = np.random.rand(3)
dam.calc(dad, x, f, u)
dam.calcDiff(dad, x, f, u)

y = np.concatenate([x,f])
iam = IAMSoftContactDynamics3D(dam)
iad = iam.createData()
iam.calc(iad, y, u)
iam.calcDiff(iad, y, u)

# Create shooting problem
ocp = crocoddyl.ShootingProblem(y, [iam]*10, iam)
import mim_solvers
solver = mim_solvers.SolverCSQP(ocp)
solver.max_qp_iters = 1000
max_iter = 500
solver.with_callbacks = True
solver.use_filter_line_search = True
solver.termination_tolerance = 1e-4
solver.eps_abs = 1e-6
solver.eps_rel = 1e-6

xs = [y]*(solver.problem.T + 1)
us = [u]*solver.problem.T
# us = solver.problem.quasiStatic([x0]*solver.problem.T) 
solver.setCallbacks([mim_solvers.CallbackVerbose(), mim_solvers.CallbackLogger()])
solver.solve(xs, us, max_iter)   

print(" >>>> TEST iiwa PASSED. ")

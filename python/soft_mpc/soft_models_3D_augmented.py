'''
Prototype Differential Action Model for augmented state space model 
This formulation supposedly allows force feedback in MPC
The derivatives of ABA in the DAM are unchanged, except we need to implement
as well d(ABA)/df . Also df/dt and its derivatives are implemented

In the IAM , a simple Euler integration (explicit) is used for the contact force
and the partials are aggregated using DAM partials.
'''
from core_mpc_utils.misc_utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger

import numpy as np
np.set_printoptions(precision=6, linewidth=180, suppress=True)
np.random.seed(1)

import crocoddyl
import pinocchio as pin

# Custom state model 
class StateSoftContact3D(crocoddyl.StateAbstract):
    def __init__(self, rmodel, nc):
        crocoddyl.StateAbstract.__init__(self, rmodel.nq + rmodel.nv + nc, 2*rmodel.nv + nc)
        self.pinocchio = rmodel
        self.nc = nc
        self.nv = (self.ndx - self.nc)//2
        self.nq = self.nx - self.nc - self.nv
        self.ny = self.nq + self.nv + self.nc
        self.ndy = 2*self.nv + self.nc
        # print("Augmented state ny = ", self.ny)
        # print("Augmented state ndy = ", self.ndy)

    def diff(self, y0, y1):
        yout = np.zeros(self.ny)
        nq = self.pinocchio.nq
        # nv = self.pinocchio.nv
        yout[:nq] = pin.difference(self.pinocchio, y0[:nq], y1[:nq])
        yout[nq:] = y1[nq:] - y0[nq:]
        return yout

    def integrate(self, y, dy):
        yout = np.zeros(self.ndy)
        nq = self.pinocchio.nq
        # nv = self.pinocchio.nv
        yout[:nq] = pin.integrate(self.pinocchio, y[:nq], dy[:nq])
        yout[nq:] = y[nq:] + dy[nq:]
        return yout

    def Jintegrate(self, y, dy, Jfirst):
        '''
        Default values :
         firstsecond = crocoddyl.Jcomponent.first 
         op = crocoddyl.addto
        '''
        Jfirst[:self.nv, :self.nv] = pin.dIntegrate(self.pinocchio, y[:self.nq], dy[:self.nv], pin.ARG0) #, crocoddyl.addto)
        Jfirst[self.nv:2*self.nv, self.nv:2*self.nv] += np.eye(self.nv)
        Jfirst[-self.nc:, -self.nc:] += np.eye(self.nc)
    
    def JintegrateTransport(self, y, dy, Jin, firstsecond):
        if(firstsecond == crocoddyl.Jcomponent.first):
            pin.dIntegrateTransport(self.pinocchio, y[:self.nq], dy[:self.nv], Jin[:self.nv], pin.ARG0)
        elif(firstsecond == crocoddyl.Jcomponent.second):
            pin.dIntegrateTransport(self.pinocchio, y[:self.nq], dy[:self.nv], Jin[:self.nv], pin.ARG1)
        else:
            logger.error("wrong arg firstsecond")


# Integrated action model and data 
class IAMSoftContactDynamics3D(crocoddyl.ActionModelAbstract): #IntegratedActionModelAbstract
    def __init__(self, dam, dt=1e-3, withCostResidual=True):
        # crocoddyl.ActionModelAbstract.__init__(self, dam.state, dam.nu, dam.costs.nr + 3)
        crocoddyl.ActionModelAbstract.__init__(self, crocoddyl.StateVector(dam.state.nq + dam.state.nv + 3), dam.nu, dam.costs.nr + 3)
        self.differential = dam
        # self.state = StateSoftContact3D(dam.pinocchio, 3)
        self.stateSoft = StateSoftContact3D(dam.pinocchio, 3)
        self.dt = dt
        self.withCostResidual = withCostResidual

    def createData(self):
        data = IADSoftContactDynamics3D(self)
        return data
    
    def calc(self, data, y, u):
        nx = self.differential.state.nx
        nv = self.differential.state.nv
        nq = self.differential.state.nq
        # logger.debug(nx)
        # logger.debug(data.differential.xout)
        nc = self.stateSoft.nc
        x = y[:nx]
        f = y[-nc:]
        # q = x[:self.state.nq]
        v = x[-nv:]
        # self.control.calc(data.control, 0., u)
        self.differential.calc(data.differential, x, f, u) #data.control.w)
        a = data.differential.xout
        fdot = data.differential.fout
        data.dx[:nv] = v*self.dt + a*self.dt**2
        data.dx[nv:2*nv] = a*self.dt
        data.dx[-nc:] = fdot*self.dt
        data.xnext = self.stateSoft.integrate(y, data.dx)
        data.cost = self.dt*data.differential.cost
        if(self.withCostResidual):
            data.r = data.differential.r


    def calcDiff(self, data, y, u):
        nx = self.differential.state.nx
        ndx = self.differential.state.ndx
        nv = self.differential.state.nv
        nu = self.differential.nu
        nc = self.stateSoft.nc
        x = y[:nx]
        f = y[-nc:]

        # Calc forward dyn derivatives
        self.differential.calcDiff(data.differential, x, f, u)
        da_dx = data.differential.Fx 
        da_du = data.differential.Fu

        # Fill out blocks
        # d->Fx.topRows(nv).noalias() = da_dx * time_step2_;
        data.Fx[:nv,:ndx] = da_dx*self.dt**2
       
        # d->Fx.bottomRows(nv).noalias() = da_dx * time_step_;
        data.Fx[nv:ndx, :ndx] = da_dx*self.dt
        
        # d->Fx.topRightCorner(nv, nv).diagonal().array() += Scalar(time_step_);
        data.Fx[:nv, nv:ndx] += self.dt * np.eye(nv)
        
        # d->Fu.topRows(nv).noalias() = time_step2_ * d->da_du;
        data.Fu[:nv, :] = da_du * self.dt**2
        
        # d->Fu.bottomRows(nv).noalias() = time_step_ * d->da_du;
        data.Fu[nv:ndx, :] = da_du * self.dt
        
        # New block from augmented dynamics (top right corner)
        data.Fx[:nv, -nc:] = data.differential.dABA_df * self.dt**2
        data.Fx[nv:ndx, -nc:] = data.differential.dABA_df * self.dt
        # New block from augmented dynamics (bottom right corner)
        data.Fx[-nc:,-nc:] = np.eye(3) + data.differential.dfdt_df*self.dt
        # New block from augmented dynamics (bottom left corner)
        data.Fx[-nc:, :ndx] = data.differential.dfdt_dx * self.dt
        
        data.Fu[-nc:, :] = data.differential.dfdt_du * self.dt

        # state_->JintegrateTransport(x, d->dx, d->Fx, second);
        self.stateSoft.JintegrateTransport(y, data.dx, data.Fx, crocoddyl.Jcomponent.second)
        
        # state_->Jintegrate(x, d->dx, d->Fx, d->Fx, first, addto);
        self.stateSoft.Jintegrate(y, data.dx, data.Fx)  # add identity to Fx = d(x+dx)/dx = d(q,v)/d(q,v)
        # data.Fx (nu, nu).diagonal().array() -=
        #     Scalar(1.);  // remove identity from Ftau (due to stateLPF.Jintegrate)
        data.Fx[-nc:, -nc:] -= np.eye(nc)

        # state_->JintegrateTransport(x, d->dx, d->Fu, second);
        self.stateSoft.JintegrateTransport(y, data.dx, data.Fu, crocoddyl.Jcomponent.second)

        # d->Lx.noalias() = time_step_ * d->differential->Lx;
        data.Lx[:ndx] = data.differential.Lx*self.dt
        data.Lx[-nc:] = data.differential.Lf*self.dt
        data.Lxx[:ndx,:ndx] = data.differential.Lxx*self.dt
        data.Lxx[-nc:,-nc:] = data.differential.Lff*self.dt
        data.Lxu[:ndx, :nu] = data.differential.Lxu*self.dt
        data.Lu = data.differential.Lu*self.dt
        data.Luu = data.differential.Luu*self.dt


class IADSoftContactDynamics3D(crocoddyl.ActionDataAbstract): #IntegratedActionDataAbstract
    '''
    Creates a data class for IAM
    '''
    def __init__(self, am):
        # super().__init__(am)
        crocoddyl.ActionDataAbstract.__init__(self, am)
        self.differential = am.differential.createData()
        self.xnext = np.zeros(am.stateSoft.ny)
        self.dx = np.zeros(am.stateSoft.ndy)
        self.Fx = np.zeros((am.stateSoft.ndy, am.stateSoft.ndy))
        self.Fu = np.zeros((am.stateSoft.ndy, am.nu))
        # self.r = am.differential.costs.nr
        self.Lx = np.zeros(am.stateSoft.ny)
        self.Lu = np.zeros(am.differential.actuation.nu)
        self.Lxx = np.zeros((am.stateSoft.ny, am.stateSoft.ny))
        self.Lxu = np.zeros((am.stateSoft.ny, am.differential.actuation.nu))
        self.Luu = np.zeros((am.differential.actuation.nu, am.differential.actuation.nu))



# Differential action model and data
class DAMSoftContactDynamics3D(crocoddyl.DifferentialActionModelAbstract):
    '''
    Computes the forward dynamics under visco-elastic (spring damper) force
    '''
    def __init__(self, stateMultibody, actuationModel, costModelSum, frameId, Kp=1e3, Kv=60, oPc=np.zeros(3), pinRefFrame=pin.LOCAL):
        # super(DAMSoftContactDynamics, self).__init__(stateMultibody, actuationModel.nu, costModelSum.nr)
        crocoddyl.DifferentialActionModelAbstract.__init__(self, stateMultibody, actuationModel.nu, costModelSum.nr)
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
        self.pinocchio = stateMultibody.pinocchio
        # hard coded costs 
        self.with_force_cost = False
        self.active_contact = True
        self.nc = 3

        self.parentId = self.pinocchio.frames[self.frameId].parent
        self.jMf = self.pinocchio.frames[self.frameId].placement

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

    def calc(self, data, x, f, u, ):
        '''
        Compute joint acceleration based on state, force and torques
        '''
        # logger.debug("CALC")
        q = x[:self.state.nq]
        v = x[self.state.nq:]
        pin.computeAllTerms(self.pinocchio, data.pinocchio, q, v)
        pin.forwardKinematics(self.pinocchio, data.pinocchio, q, v, np.zeros(self.state.nq))
        pin.updateFramePlacements(self.pinocchio, data.pinocchio)
        oRf = data.pinocchio.oMf[self.frameId].rotation
        # Actuation calc
        self.actuation.calc(data.multibody.actuation, x, u)

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
        # Cost calc 
        self.costs.calc(data.costs, x, u) 
        data.cost = data.costs.cost
        # Add hard-coded cost
        if(self.with_force_cost):
            # Compute force residual and add force cost to total cost
            data.f_residual = f - self.f_des
            data.cost += 0.5 * self.f_weight * data.f_residual.T @ data.f_residual
        return data.xout, data.fout, data.cost

    def calcDiff(self, data, x, f, u):
        '''
        Compute derivatives 
        '''
        q = x[:self.state.nq]
        v = x[self.state.nq:]
        oRf = data.pinocchio.oMf[self.frameId].rotation
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
        self.costs = am.costs.createData(crocoddyl.DataCollectorMultibody(self.pinocchio))
        
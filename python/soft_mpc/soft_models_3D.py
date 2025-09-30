'''
Prototype Differential Action Model using visco-elastic contact force
The force is a function of the current state (linear spring damper)
The derivatives of ABA (forward dynamics) need to be updated because now the 
force depend on the state . This can be used with standard Euler integrator 
to solve an OCP and in MPC with the classical state feedback. 

This formulation doesn"t seem to allow to feebdack measured forces in MPC 
We need to augment the state . This is done in the other DAM 
'''
from croco_mpc_utils.utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger

import numpy as np
np.set_printoptions(precision=6, linewidth=180, suppress=True)
np.random.seed(1)

import crocoddyl
import pinocchio as pin



# # when DAM inherits from FreeFwdDyn  : error in the computation
# # when DAM inherits from DAMAbstract : ok 
# # Can be a binding problem because check point inside calc not reached in first case, reached in second case
# # NumDiff uses base.calc() . Is it also true in C++? 

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

    def calc(self, data, x, u):
        '''
        Compute joint acceleration and costs 
         using visco-elastic force
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
            # Compute spring damper force + express at joint level 
            lv = pin.getFrameVelocity(self.pinocchio, data.pinocchio, self.frameId, pin.LOCAL).linear
            data.f = -self.Kp * oRf.T @ ( data.pinocchio.oMf[self.frameId].translation - self.oPc ) - self.Kv*lv
            data.fext[self.parentId] = self.jMf.act(pin.Force(data.f, np.zeros(3)))
            # Copy for later
            data.f_copy = data.f.copy()
            data.fext_copy = data.fext.copy()
            # rotate if not local
            if(self.pinRef != pin.LOCAL):
                # ov = pin.getFrameVelocity(self.pinocchio, data.pinocchio, self.frameId, pin.LOCAL_WORLD_ALIGNED).linear
                # assert(np.linalg.norm(ov - oRf @ lv ) < 1e-4)
                data.f = -self.Kp * ( data.pinocchio.oMf[self.frameId].translation - self.oPc ) - self.Kv * oRf @ lv
                # print('local = ', oRf @ data.f_copy)
                # print('world = ', data.f)
                assert(np.linalg.norm(data.f - oRf @ data.f_copy) < 1e-4)
                assert(np.linalg.norm(oRf.T @ data.f - data.f_copy) < 1e-4)
                data.fext[self.parentId] = self.jMf.act(pin.Force(oRf.T @ data.f, np.zeros(3)))
            data.xout = pin.aba(self.pinocchio, data.pinocchio, q, v, data.multibody.actuation.tau, data.fext) 
        else:
            data.xout = pin.aba(self.pinocchio, data.pinocchio, q, v, data.multibody.actuation.tau)
        
        pin.updateGlobalPlacements(self.pinocchio, data.pinocchio)
        # Cost calc 
        self.costs.calc(data.costs, x, u) 
        data.cost = data.costs.cost
        # Add hard-coded cost
        if(self.with_force_cost):
            self.f_residual = data.f - self.f_des
            data.cost += 0.5 * self.f_weight * self.f_residual.T @ self.f_residual
        return data.xout, data.cost

    def calcDiff(self, data, x, u):
        '''
        Compute derivatives 
        '''
        # logger.debug("CALCDIFF")
        # First call calc
        q = x[:self.state.nq]
        v = x[self.state.nq:]
        oRf = data.pinocchio.oMf[self.frameId].rotation
        # Actuation calcDiff
        self.actuation.calcDiff(data.multibody.actuation, x, u)

        if(self.active_contact):
            # Compute spring damper force derivatives in LOCAL
            lJ = pin.getFrameJacobian(self.pinocchio, data.pinocchio, self.frameId, pin.LOCAL)
            # logger.debug(lJ)
            oJ = pin.getFrameJacobian(self.pinocchio, data.pinocchio, self.frameId, pin.LOCAL_WORLD_ALIGNED)
            lv_partial_dq, lv_partial_dv = pin.getFrameVelocityDerivatives(self.pinocchio, data.pinocchio, self.frameId, pin.LOCAL) 
            data.df_dx[:,:self.state.nq] = \
                - self.Kp * (lJ[:3] + pin.skew(oRf.T @ (data.pinocchio.oMf[self.frameId].translation - self.oPc)) @ lJ[3:]) \
                - self.Kv*lv_partial_dq[:3]
            data.df_dx[:,self.state.nq:] = \
                - self.Kv*lv_partial_dv[:3]
            # copy for later
            data.df_dx_copy = data.df_dx.copy()
            # rotate if not LOCAL 
            if(self.pinRef != pin.LOCAL):
                data.df_dx[:,:self.state.nq] = oRf @ data.df_dx_copy[:,:self.state.nq] - pin.skew(oRf @ data.f_copy) @ oJ[3:]
                data.df_dx[:,self.state.nq:] = oRf @ data.df_dx_copy[:,self.state.nq:]
            # Computing the dynamics using ABA or manually if armature
            aba_dq, ada_dv, aba_dtau = pin.computeABADerivatives(self.pinocchio, data.pinocchio, q, v, data.multibody.actuation.tau, data.fext_copy)
            # print("classical : \n", aba_dq)
            data.Fx[:,:self.state.nq] = aba_dq + data.pinocchio.Minv @ lJ[:3].T @ data.df_dx_copy[:,:self.state.nq]
            data.Fx[:,self.state.nq:] = ada_dv + data.pinocchio.Minv @ lJ[:3].T @ data.df_dx_copy[:,self.state.nq:]
            data.Fx += data.pinocchio.Minv @ data.multibody.actuation.dtau_dx
            data.Fu = aba_dtau @ data.multibody.actuation.dtau_du

        else:
            # Computing the free forward dynamics with ABA derivatives
            aba_dq, ada_dv, aba_dtau = pin.computeABADerivatives(self.pinocchio, data.pinocchio, q, v, data.multibody.actuation.tau)
            data.Fx[:,:self.state.nq] = aba_dq 
            data.Fx[:,self.state.nq:] = ada_dv 
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
            self.f_residual = data.f - self.f_des
            data.Lx += self.f_weight * self.f_residual.T @ data.df_dx 
            data.Lxx += self.f_weight * data.df_dx.T @ data.df_dx 



class DADSoftContactDynamics(crocoddyl.DifferentialActionDataAbstract):
    '''
    Creates a data class with differential and augmented matrices from IAM (initialized with stateVector)
    '''
    def __init__(self, am):
        # super().__init__(am)
        crocoddyl.DifferentialActionDataAbstract.__init__(self, am)
        self.Fx = np.zeros((am.state.nq, am.state.nx))
        self.Fu = np.zeros((am.state.nq, am.nu))
        self.Lx = np.zeros(am.state.nx)
        self.Lu = np.zeros(am.actuation.nu)
        self.Lxx = np.zeros((am.state.nx, am.state.nx))
        self.Lxu = np.zeros((am.state.nx, am.actuation.nu))
        self.Luu = np.zeros((am.actuation.nu, am.actuation.nu))

        self.df_dx = np.zeros((am.nc, am.state.nx))   
        self.df_dx_copy = np.zeros((am.nc, am.state.nx))   
        self.f = np.zeros(am.nc)    
        self.f_copy = np.zeros(am.nc)   
        self.fext = [pin.Force.Zero() for _ in range(am.pinocchio.njoints)]
        self.fext_copy = [pin.Force.Zero() for _ in range(am.pinocchio.njoints)]

        self.pinocchio  = am.pinocchio.createData()
        self.actuation_data = am.actuation.createData()
        self.multibody = crocoddyl.DataCollectorActMultibody(self.pinocchio, self.actuation_data)
        self.costs = am.costs.createData(crocoddyl.DataCollectorMultibody(self.pinocchio))
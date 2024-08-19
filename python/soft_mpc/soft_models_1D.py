from core_mpc_utils.misc_utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger

import numpy as np
np.set_printoptions(precision=6, linewidth=180, suppress=True)
np.random.seed(1)

import crocoddyl
import pinocchio as pin



class DAMSoftContactDynamics1D(crocoddyl.DifferentialActionModelAbstract):
    '''
    Computes the forward dynamics under visco-elastic (spring damper) force 1D
    '''
    def __init__(self, stateMultibody, actuationModel, costModelSum, frameId, contactType, Kp=1e3, Kv=60, oPc=np.zeros(3), pinRefFrame=pin.LOCAL):
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

        self.parentId = self.pinocchio.frames[self.frameId].parent
        self.jMf = self.pinocchio.frames[self.frameId].placement

        self.set_contactType(contactType)


    def set_contactType(self, contactType):
        assert(contactType in ['1Dx', '1Dy', '1Dz'])
        self.contact_type = contactType
        self.nc = 1
        if(contactType == '1Dx'):
            self.mask = [0]
        if(contactType == '1Dy'):
            self.mask = [1]
        if(contactType == '1Dz'):
            self.mask = [2]

    def set_active_contact(self, active):
        self.active_contact = active


    def createData(self):
        '''
            The data is created with a custom data class that contains the filtered torque tau_plus and the activation
        '''
        data = DADSoftContactDynamics(self)
        return data


    def set_force_cost(self, f_des, f_weight):
        if(self.contact_type == '3D'):
            assert(len(f_des) == 3)
        else:
            assert(len(f_des) == 1)
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

        # If contact is active, compute aq = ABA(q,v,tau,fext)
        if(self.active_contact):
            # Compute spring damper force + express at joint level 
            lv = pin.getFrameVelocity(self.pinocchio, data.pinocchio, self.frameId, pin.LOCAL).linear
            # Compute and project local 3D force --> 1D + get external wrench at joint level
            data.f3d = -self.Kp * oRf.T @ ( data.pinocchio.oMf[self.frameId].translation - self.oPc ) - self.Kv*lv
            data.f = data.f3d[self.mask] 
            fLOCAL = np.zeros(3) ; fLOCAL[self.mask] = data.f
            lwrench = pin.Force(fLOCAL, np.zeros(3))
            data.fext[self.parentId] = self.jMf.act(lwrench)
            # Save local force and external wrench for later
            data.f3d_copy = data.f3d.copy()
            data.f_copy = data.f.copy()
            data.fext_copy = data.fext.copy()
            # If LWA, rotate LOCAL quantities to get force and external wrench
            if(self.pinRef != pin.LOCAL):
                data.f3d = -self.Kp * ( data.pinocchio.oMf[self.frameId].translation - self.oPc ) - self.Kv * oRf @ lv
                data.f = data.f3d[self.mask]
                assert(np.linalg.norm(data.f3d - oRf @ data.f3d_copy) < 1e-4)
                assert(np.linalg.norm(oRf.T @ data.f3d - data.f3d_copy) < 1e-4)
                data.fWORLD = np.zeros(3) ; data.fWORLD[self.mask] = data.f
                owrench = pin.Force(data.fWORLD, np.zeros(3))
                lwaXf = pin.SE3.Identity() ; lwaXf.rotation = oRf ; lwaXf.translation = np.zeros(3)
                data.fext[self.parentId] = self.jMf.act(lwaXf.actInv(owrench))
            # Compute joint acceleration by calling ABA
            data.xout = pin.aba(self.pinocchio, data.pinocchio, q, v, data.multibody.actuation.tau, data.fext) 
        
        # If contact NOT active : compute aq = ABA(q,v,tau)
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

        # If contact is active, compute ABA derivatives 
        if(self.active_contact):
            # Compute spring damper force derivatives in LOCAL
            lJ = pin.getFrameJacobian(self.pinocchio, data.pinocchio, self.frameId, pin.LOCAL)
            oJ = pin.getFrameJacobian(self.pinocchio, data.pinocchio, self.frameId, pin.LOCAL_WORLD_ALIGNED)
            lv_partial_dq, lv_partial_dv = pin.getFrameVelocityDerivatives(self.pinocchio, data.pinocchio, self.frameId, pin.LOCAL) 
            data.df3d_dx[:,:self.state.nq] = \
                - self.Kp * (lJ[:3] + pin.skew(oRf.T @ (data.pinocchio.oMf[self.frameId].translation - self.oPc)) @ lJ[3:]) \
                - self.Kv*lv_partial_dq[:3]
            data.df3d_dx[:,self.state.nq:] = \
                - self.Kv*lv_partial_dv[:3]
            # copy local 3D derivatives for later
            data.df3d_dx_copy = data.df3d_dx.copy()
            # Project 3D --> 1D + record local 1D for later 
            data.df_dx = data.df3d_dx[self.mask,:]
            data.df_dx_copy = data.df_dx.copy()
            # rotate if not LOCAL 
            if(self.pinRef != pin.LOCAL):
                data.df3d_dx[:,:self.state.nq] = oRf @ data.df3d_dx_copy[:,:self.state.nq] - pin.skew(oRf @ data.f3d_copy) @ oJ[3:]
                data.df3d_dx[:,self.state.nq:] = oRf @ data.df3d_dx_copy[:,self.state.nq:]
                data.df_dx = data.df3d_dx[self.mask,:]
            # Computing the dynamics using ABA (or manually if armature)
              # LOCAL case
            aba_dq, aba_dv, aba_dtau = pin.computeABADerivatives(self.pinocchio, data.pinocchio, q, v, data.multibody.actuation.tau, data.fext)
            if(self.pinRef == pin.LOCAL):
                data.Fx[:,:self.state.nq] = \
                    aba_dq + \
                    data.pinocchio.Minv @ lJ[:3].T[:,self.mask] @ data.df_dx_copy[:,:self.state.nq]
                data.Fx[:,self.state.nq:] = \
                    aba_dv + \
                    data.pinocchio.Minv @ lJ[:3].T[:,self.mask] @ data.df_dx_copy[:,self.state.nq:]
              # LWA case
            else:
                # aba_dq, aba_dv, aba_dtau = pin.computeABADerivatives(self.pinocchio, data.pinocchio, q, v, data.multibody.actuation.tau, data.fext)
                data.Fx[:,:self.state.nq] = \
                    aba_dq + \
                    data.pinocchio.Minv @ lJ[:3].T @ (oRf.T[:,self.mask] @ data.df_dx[:,:self.state.nq] + pin.skew(oRf.T @ data.fWORLD) @ lJ[3:])
                data.Fx[:,self.state.nq:] = \
                    aba_dv + \
                    data.pinocchio.Minv @ lJ[:3].T @ (oRf.T[:,self.mask] @ data.df_dx[:,self.state.nq:])
            data.Fx += data.pinocchio.Minv @ data.multibody.actuation.dtau_dx
            data.Fu = aba_dtau @ data.multibody.actuation.dtau_du
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
        # need to allocate 3D data as well
        self.f3d = np.zeros(3)
        self.f3d_copy = np.zeros(3)   
        self.df3d_dx = np.zeros((3, am.state.nx))
        self.df3d_dx_copy = np.zeros((3, am.state.nx))   
        self.fWORLD = np.zeros(3)
        self.fext = [pin.Force.Zero() for _ in range(am.pinocchio.njoints)]
        self.fext_copy = [pin.Force.Zero() for _ in range(am.pinocchio.njoints)]

        self.pinocchio  = am.pinocchio.createData()
        self.actuation_data = am.actuation.createData()
        self.multibody = crocoddyl.DataCollectorActMultibody(self.pinocchio, self.actuation_data)
        self.costs = am.costs.createData(crocoddyl.DataCollectorMultibody(self.pinocchio))
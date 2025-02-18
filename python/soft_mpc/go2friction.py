'''
Action Model for force feedback MPC on GO2 with 5 contacts 
'''
from croco_mpc_utils.utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger

import numpy as np
np.set_printoptions(precision=6, linewidth=180, suppress=True)
np.random.seed(1)

import crocoddyl
import pinocchio as pin


# 3D soft contact model 
class ViscoElasticContact3D:
    def __init__(self, state, actuation, frameId, oPc=np.zeros(3), Kp=10, Kv=0, pinRef=pin.LOCAL):
        self.active = True
        self.nc = 3
        self.name = ""
        self.state = state
        self.actuation = actuation
        self.pinocchio = self.state.pinocchio
        self.frameId = frameId
        self.oPc = oPc
        self.Kp = Kp
        self.Kv = Kv
        self.pinRef = pinRef
        # Parent joint id of contact frame
        self.parentId = self.pinocchio.frames[self.frameId].parent
        # Local placement of contact frame w.r.t. parent joint
        self.jMf = self.pinocchio.frames[self.frameId].placement
        # Spatial wrench in body coordinates 
        self.fext = pin.Force.Zero()
        # Contact force time derivative
        self.fout      = np.zeros(self.nc)
        self.fout_copy = np.zeros(self.nc)
        # Partial derivatives
        self.dABA_df = np.zeros((self.state.nv, self.nc))
        self.dfdt_dx   = np.zeros((self.nc, self.state.ndx))
        self.dfdt_du   = np.zeros((self.nc, self.actuation.nu))
        self.dfdt_df   = np.zeros((self.nc, self.nc))  

    def calc(self, dad, f):
        '''
        Calculate the contact force spatial wrench 
            f : 3d force
        '''
        # Placement of contact frame in WORLD
        oRf = dad.pinocchio.oMf[self.frameId].rotation
        # Compute external wrench given LOCAL 3d force
        self.fext = self.jMf.act(pin.Force(f, np.zeros(3)))
        # Rotate if f is not expressed in LOCAL
        if(self.pinRef != pin.LOCAL):
            self.fext = self.jMf.act(pin.Force(oRf.T @ f, np.zeros(3)))
        return self.fext

    def calc_fdot(self, dad):
        '''
        Compute the time derivative of the soft contact force
        '''
        la = pin.getFrameAcceleration(self.pinocchio, dad.pinocchio, self.frameId, pin.LOCAL).linear         
        lv = pin.getFrameVelocity(self.pinocchio, dad.pinocchio, self.frameId, pin.LOCAL).linear
        self.fout = -self.Kp * lv - self.Kv * la
        self.fout_copy = self.fout.copy()
        # Rotate if not f not in LOCAL
        if(self.pinRef != pin.LOCAL):
            oa = pin.getFrameAcceleration(self.pinocchio, dad.pinocchio, self.frameId, pin.LOCAL_WORLD_ALIGNED).linear         
            ov = pin.getFrameVelocity(self.pinocchio, dad.pinocchio, self.frameId, pin.LOCAL_WORLD_ALIGNED).linear
            self.fout = -self.Kp * ov - self.Kv * oa
    
    def update_ABAderivatives(self, dad, f):  
        '''
        Add the contribution of the soft contact force model to the ABA derivatives
        '''
        # Compute derivatives of data.xout (ABA) w.r.t. f in LOCAL 
        lJ = pin.getFrameJacobian(self.pinocchio, dad.pinocchio, self.frameId, pin.LOCAL)  
        oRf = dad.pinocchio.oMf[self.frameId].rotation
        self.dABA_df = dad.pinocchio.Minv @ lJ[:3].T @ self.pinocchio.frames[self.frameId].placement.rotation @ np.eye(3) 
        # Skew term added to RNEA derivatives when force is expressed in LWA
        if(self.pinRef != pin.LOCAL):
            # logger.debug("corrective term aba LWA : \n"+str(data.pinocchio.Minv @ lJ[:3].T @ pin.skew(oRf.T @ f) @ lJ[3:]))
            dad.Fx[:,:self.state.nv] += dad.pinocchio.Minv @ lJ[:3].T @ pin.skew(oRf.T @ f) @ lJ[3:]
            # Rotate dABA/df
            self.dABA_df = self.dABA_df @ oRf.T 

    def calcDiff(self, dad):  
        '''
        Compute partial derivatives of the soft contact force time derivative w.r.t. state and input
        '''
        oRf = dad.pinocchio.oMf[self.frameId].rotation
        # Derivatives of data.fout in LOCAL : important >> UPDATE FORWARD KINEMATICS with data.xout
        lv_dq, lv_dv = pin.getFrameVelocityDerivatives(self.pinocchio, dad.pinocchio, self.frameId, pin.LOCAL)
        lv_dx = np.hstack([lv_dq, lv_dv])
        _, a_dq, a_dv, a_da = pin.getFrameAccelerationDerivatives(self.pinocchio, dad.pinocchio, self.frameId, pin.LOCAL)
        da_dx = np.zeros((3,self.state.ndx))
        da_dx[:,:self.state.nv] = a_dq[:3] + a_da[:3] @ dad.Fx[:,:self.state.nv] # same as aba_dq here
        da_dx[:,self.state.nv:] = a_dv[:3] + a_da[:3] @ dad.Fx[:,self.state.nv:] # same as aba_dv here
        da_du = a_da[:3] @ dad.Fu
        da_df = a_da[:3] @ dad.dABA_df
        # Deriv of lambda dot
        self.dfdt_dx = -self.Kp*lv_dx[:3] - self.Kv*da_dx[:3]
        self.dfdt_du = -self.Kv*da_du
        self.dfdt_df = -self.Kv*da_df
        self.ldfdt_dx_copy = self.dfdt_dx.copy()
        self.ldfdt_du_copy = self.dfdt_du.copy()
        self.ldfdt_df_copy = self.dfdt_df.copy()
        # Rotate dfout_dx if not LOCAL 
        if(self.pinRef != pin.LOCAL):
            oJ = pin.getFrameJacobian(self.pinocchio, dad.pinocchio, self.frameId, pin.LOCAL_WORLD_ALIGNED)
            self.dfdt_dx[:,:self.state.nv] = oRf @ self.ldfdt_dx_copy[:,:self.state.nv] - pin.skew(oRf @ self.fout_copy) @ oJ[3:]
            self.dfdt_dx[:,self.state.nv:] = oRf @ self.ldfdt_dx_copy[:,self.state.nv:] 
            self.dfdt_du = oRf @ self.ldfdt_du_copy
            self.dfdt_df = oRf @ self.ldfdt_df_copy
            

# 3D Soft contact models stack
class ViscoElasticContact3d_Multiple:
    def __init__(self, state, actuation, contacts):
        '''
            state   : stateMultibody
            contact : ViscoElasticContact3D
        '''
        self.state     = state
        self.actuation = actuation
        self.pinocchio = state.pinocchio
        self.contacts  = contacts
        self.nv        = self.state.nv
        # number of contact models
        self.nc        = len(self.contacts) 
        # total contact space dimension
        self.nc_tot    = np.sum(np.array([ct.nc for ct in self.contacts])) 
        self.Jc        = np.zeros((self.nc_tot, self.nv))
        self.fext      = [pin.Force.Zero() for _ in range(self.pinocchio.njoints)]
        self.fext_copy = [pin.Force.Zero() for _ in range(self.pinocchio.njoints)]
        self.fout      = np.zeros(self.nc_tot)
        self.fout_copy = np.zeros(self.nc_tot)
        # Active if at least 1 contact is active
        self.active = bool(np.max([ct.active for ct in self.contacts]))
        # # Partial derivatives
        # self.dABA_df = np.zeros((self.state.nv, self.nc_tot))
        # self.dfdt_dx = np.zeros((self.nc_tot, self.state.nx))
        # self.dfdt_du = np.zeros((self.nc_tot, self.actuation.nu))
        # self.dfdt_df = np.zeros((self.nc_tot, self.nc_tot)) 

    def calc(self, dad, f):
        '''
            f : stack of 3d forces
        '''
        nc_i = 0
        # Compute & stack spatial forces
        for ct in self.contacts:
            ct.calc(dad, f[nc_i:nc_i+ct.nc])
            if(ct.active):
                self.fext[ct.parentId] = ct.fext
            nc_i += ct.nc
        return self.fext
    
    def calc_fdot(self, dad):
        '''
        Stack fdots from contact models
        '''
        nc_i = 0
        # Compute & stack force time derivatives
        for ct in self.contacts:
            ct.calc_fdot(dad)
            if(ct.active):
                self.fout[nc_i:nc_i+ct.nc] = ct.fout
            nc_i += ct.nc
        self.fout_copy = self.fout.copy()
        return self.fout

    def update_ABAderivatives(self, dad, f):
        '''
        Add the contribution of the soft contact force model to the ABA derivatives
        '''
        nc_i = 0
        # Compute & stack ABA corrective terms
        for ct in self.contacts:
            if(ct.active):
                ct.update_ABAderivatives(dad, f[nc_i:nc_i+ct.nc])
                dad.dABA_df[0:self.nv, nc_i:nc_i+ct.nc] = ct.dABA_df
            nc_i += ct.nc
    
    def calcDiff(self, dad):
        '''
        Compute partial derivatives of the soft contact force time derivative w.r.t. state and input
        '''
        nc_i = 0
        # Compute & stack partials of fdot
        for ct in self.contacts:
            if(ct.active):
                ct.calcDiff(dad)
                dad.dfdt_dx[nc_i:nc_i+ct.nc, :] = ct.dfdt_dx
                dad.dfdt_du[nc_i:nc_i+ct.nc, :] = ct.dfdt_du
                dad.dfdt_df[nc_i:nc_i+ct.nc, :] = ct.dfdt_df
            nc_i += ct.nc


class FrictionConeConstraint:
    def __init__(self, state, frameId, coef):
        self.state = state
        self.frameId = frameId
        self.coef = coef
        self.residual = 0
        self.dcone_df = np.zeros(3)

    def calc(self, f):
        self.residual = self.coef * f[2] - np.sqrt(f[0]*f[0] + f[1]*f[1])
        return self.residual
    
    def calcDiff(self, f):
        self.dcone_df[0] = -f[0] / np.sqrt(f[0]*f[0] + f[1]*f[1])
        self.dcone_df[1] = -f[1] / np.sqrt(f[0]*f[0] + f[1]*f[1])
        self.dcone_df[2] = self.coef
        self.dcone_df
        return self.dcone_df

# # Custom state model that includes the soft contact force
# class StateSoftContact3D(crocoddyl.StateAbstract):
#     def __init__(self, rmodel, nc):
#         # crocoddyl.StateAbstract.__init__(self, rmodel.nq + rmodel.nv + nc, 2*rmodel.nv + nc)
#         super().__init__(rmodel.nq + rmodel.nv + nc, 2*rmodel.nv + nc)
#         self.pinocchio = rmodel
#         self.nc = nc
#         self.nv = (self.ndx - self.nc)//2
#         self.nq = self.nx - self.nc - self.nv
#         self.ny = self.nq + self.nv + self.nc
#         self.ndy = 2*self.nv + self.nc

#     def diff(self, y0, y1):
#         yout = np.zeros(self.ny)
#         nq = self.pinocchio.nq
#         nv = self.pinocchio.nv
#         nc = self.nc
#         yout[:nv] = pin.difference(self.pinocchio, y0[:nq], y1[:nq])
#         yout[-nc:] = y1[-nc:] - y0[-nc:]
#         return yout

#     def integrate(self, y, dy):
#         yout = np.zeros(self.ndy)
#         nq = self.pinocchio.nq
#         nv = self.pinocchio.nv
#         nc = self.nc
#         yout[:nq] = pin.integrate(self.pinocchio, y[:nq], dy[:nv])
#         yout[-nc:] = y[-nc:] + dy[-nc:]
#         return yout

#     def Jdiff(self, y0, y1, Jfirst, Jsecond, firstsecond):
#         nq = self.pinocchio.nq
#         nv = self.pinocchio.nv
#         nc = self.pinocchio.nc
#         pin.dDifference(self.pinocchio, y0[:nq], y1[:nq], Jfirst[:nv, :nv], pin.ARG0)
#         pin.dDifference(self.pinocchio, y0[:nq], y1[:nq], Jsecond[:nv, :nv], pin.ARG0)
#         Jfirst[nv:nv+nv, nv:nv+nv] -= np.eye(nv)  # -1 on diagonal block
#         Jfirst[-nc:, -nc:] -= np.eye(nc)  # -1 on bottom-right diagonal block
#         Jsecond[nv:nv+nv, nv:nv+nv] += np.eye(nv)  # +1 on diagonal block
#         Jsecond[-nc:, -nc:] += np.eye(nc)  # +1 on bottom-right diagonal block

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

# Custom Differential Action Model (DAM) for Go2+arm (5 soft 3D contacts with the environment)
class DAMSoftContactDynamics3D_Go2(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, stateMultibody, actuationModel, costModelSum, constraintModelManager=None):
        '''
            stateMultibody         : crocoddyl.stateMultibody
            actuationModel         : crocoddyl.actuationModelFloatingBase
            costModelSum           : crocoddyl.costModelSum
            constraintModelManager : crocoddyl.constraintModelManager
        '''
        # Determine constraint dimensions if any
        self.constraints = constraintModelManager
        if(constraintModelManager is None):
            ng = 0
            nh = 0
        else:
            ng = constraintModelManager.ng
            nh = constraintModelManager.nh
        # Init class
        crocoddyl.DifferentialActionModelAbstract.__init__(self, stateMultibody, actuationModel.nu, costModelSum.nr, ng, nh)
        # Soft contact models parameters
        self.Kp        = 1e3
        self.Kv        = 1e2
        self.oPc       = np.zeros(3)
        self.pinRef    = pin.LOCAL_WORLD_ALIGNED
        ee_frame_names = ['FL_FOOT', 'FR_FOOT', 'HL_FOOT', 'HR_FOOT', 'Link6']
        self.lfFootId  = stateMultibody.pinocchio.getFrameId(ee_frame_names[0])
        self.rfFootId  = stateMultibody.pinocchio.getFrameId(ee_frame_names[1])
        self.lhFootId  = stateMultibody.pinocchio.getFrameId(ee_frame_names[2])
        self.rhFootId  = stateMultibody.pinocchio.getFrameId(ee_frame_names[3])
        self.efId      = stateMultibody.pinocchio.getFrameId(ee_frame_names[4])
        # Soft contact models 3d 
        lf_contact = ViscoElasticContact3D(stateMultibody, actuationModel, self.lfFootId, self.oPc, self.Kp, self.Kv, self.pinRef)
        rf_contact = ViscoElasticContact3D(stateMultibody, actuationModel, self.rfFootId, self.oPc, self.Kp, self.Kv, self.pinRef)
        lh_contact = ViscoElasticContact3D(stateMultibody, actuationModel, self.lhFootId, self.oPc, self.Kp, self.Kv, self.pinRef)
        rh_contact = ViscoElasticContact3D(stateMultibody, actuationModel, self.rhFootId, self.oPc, self.Kp, self.Kv, self.pinRef)
        ef_contact = ViscoElasticContact3D(stateMultibody, actuationModel, self.efId, self.oPc, self.Kp, self.Kv, self.pinRef)
        # Contact models stack
        self.contacts = ViscoElasticContact3d_Multiple(stateMultibody, actuationModel, [lf_contact, rf_contact, lh_contact, rh_contact, ef_contact])
        assert(self.contacts.nc == 5)
        assert(self.contacts.nc_tot == 15)
        self.active_contact = self.contacts.active
        assert(self.contacts.active == True)
        # To complete DAMAbstract into sth like DAMFwdDyn
        self.actuation = actuationModel
        self.costs     = costModelSum
        self.pinocchio = stateMultibody.pinocchio
        # hard coded costs 
        self.with_force_cost = True
        self.f_des           = np.zeros(self.contacts.nc_tot)
        self.f_weight        = 0.001
        
    def createData(self):
        '''
            The data is created with a custom data class that contains the filtered torque tau_plus and the activation
        '''
        data = DADSoftContactDynamics_Go2(self)
        return data

    def calc(self, data, x, f, u=None):
        '''
        Compute joint acceleration based on state, force and torques
        '''
        q = x[:self.state.nq]
        v = x[self.state.nq:]
        if(u is not None):
            pin.computeAllTerms(self.pinocchio, data.pinocchio, q, v)
            pin.forwardKinematics(self.pinocchio, data.pinocchio, q, v, np.zeros(self.state.nv))
            pin.updateFramePlacements(self.pinocchio, data.pinocchio)
            # Actuation calc
            self.actuation.calc(data.multibody.actuation, x, u)

            if(self.active_contact):
                # Compute contact wrench in spatial coordinates
                data.fext      = self.contacts.calc(data, f)
                data.fext_copy = data.fext.copy()
                # Compute joint acceleration 
                data.xout = pin.aba(self.pinocchio, data.pinocchio, q, v, data.multibody.actuation.tau, data.fext) 
                # Compute time derivative of contact force : need to forward kin with current acc
                pin.forwardKinematics(self.pinocchio, data.pinocchio, q, v, data.xout)
                data.fout = self.contacts.calc_fdot(data)
            else:
                data.xout = pin.aba(self.pinocchio, data.pinocchio, q, v, data.multibody.actuation.tau)

            pin.updateGlobalPlacements(self.pinocchio, data.pinocchio)
            # Cost calc 
            self.costs.calc(data.costs, x, u) 
            data.cost = data.costs.cost
            # Add hard-coded cost
            if(self.active_contact and self.with_force_cost):
                # Compute force residual and add force cost to total cost
                data.f_residual = f - self.f_des
                data.cost       += 0.5 * self.f_weight * data.f_residual.T @ data.f_residual
            # constraints
            if(self.constraints is not None):
                self.constraints.calc(data.constraints, x, u)
                
        else:
            pin.computeAllTerms(self.pinocchio, data.pinocchio, q, v)
            self.costs.calc(data.costs, x) 
            data.cost = data.costs.cost
            # Add hard-coded cost
            if(self.active_contact and self.with_force_cost):
                # Compute force residual and add force cost to total cost
                data.f_residual = f - self.f_des
                data.cost       += 0.5 * self.f_weight * data.f_residual.T @ data.f_residual
            # constraints
            if(self.constraints is not None):
                self.constraints.calc(data.constraints, x)



    def calcDiff(self, data, x, f, u=None):
        '''
        Compute derivatives 
        '''
        q = x[:self.state.nq]
        v = x[self.state.nq:]
        # Actuation calcDiff
        if(u is not None):
            self.actuation.calcDiff(data.multibody.actuation, x, u)

            if(self.active_contact):      
                # Derivatives of data.xout (ABA) w.r.t. x and u in LOCAL (same in WORLD)
                aba_dq, aba_dv, aba_dtau = pin.computeABADerivatives(self.pinocchio, data.pinocchio, q, v, data.multibody.actuation.tau, data.fext)
                data.Fx[:,:self.state.nv] = aba_dq 
                data.Fx[:,self.state.nv:] = aba_dv 
                data.Fx += data.pinocchio.Minv @ data.multibody.actuation.dtau_dx
                data.Fu = aba_dtau @ data.multibody.actuation.dtau_du

                # Update ABA derivatives with soft contact model contribution
                self.contacts.update_ABAderivatives(data, f)

                # Derivatives of data.fout in LOCAL : important >> UPDATE FORWARD KINEMATICS with data.xout
                pin.computeAllTerms(self.pinocchio, data.pinocchio, q, v)
                pin.forwardKinematics(self.pinocchio, data.pinocchio, q, v, data.xout)
                pin.updateFramePlacements(self.pinocchio, data.pinocchio)
                self.contacts.calcDiff(data)

            else:
                # Computing the free forward dynamics with ABA derivatives
                aba_dq, aba_dv, aba_dtau = pin.computeABADerivatives(self.pinocchio, data.pinocchio, q, v, data.multibody.actuation.tau)
                data.Fx[:,:self.state.nv] = aba_dq 
                data.Fx[:,self.state.nv:] = aba_dv 
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
                data.Lf         = self.f_weight * data.f_residual.T
                data.Lff        = self.f_weight * np.eye(self.contacts.nc_tot)
            # constraints
            if(self.constraints is not None):
                self.constraints.calcDiff(data.constraints, x, u)
        else:
            self.costs.calcDiff(data.costs, x)
            data.Lx = data.costs.Lx
            data.Lu = data.costs.Lu
            data.Lxx = data.costs.Lxx
            data.Lxu = data.costs.Lxu
            data.Luu = data.costs.Luu
            # add hard-coded cost
            if(self.active_contact and self.with_force_cost):
                data.f_residual = f - self.f_des
                data.Lf         = self.f_weight * data.f_residual.T
                data.Lff        = self.f_weight * np.eye(self.contacts.nc_tot)
            # constraints
            if(self.constraints is not None):
                self.constraints.calcDiff(data.constraints, x)
                
# Custom Differential Action Data class for go2+arm
class DADSoftContactDynamics_Go2(crocoddyl.DifferentialActionDataAbstract):
    '''
    Creates a data class with differential and augmented matrices from IAM (initialized with stateVector)
    '''
    def __init__(self, am):
        # super().__init__(am)
        crocoddyl.DifferentialActionDataAbstract.__init__(self, am)
        # Force model + derivatives
        self.fout      = np.zeros(am.contacts.nc_tot)
        self.fout_copy = np.zeros(am.contacts.nc_tot)
        self.dfdt_dx   = np.zeros((am.contacts.nc_tot,am.state.ndx))
        self.dfdt_du   = np.zeros((am.contacts.nc_tot,am.nu))
        self.dfdt_df   = np.zeros((am.contacts.nc_tot,am.contacts.nc_tot))  
        # ABA model derivatives
        self.Fx = np.zeros((am.state.nv, am.state.ndx))
        self.Fu = np.zeros((am.state.nv, am.nu))
        self.dABA_df = np.zeros((am.state.nv, am.contacts.nc_tot))
        # Cost derivatives
        self.Lx = np.zeros(am.state.ndx)
        self.Lu = np.zeros(am.actuation.nu)
        self.Lxx = np.zeros((am.state.ndx, am.state.ndx))
        self.Lxu = np.zeros((am.state.ndx, am.actuation.nu))
        self.Luu = np.zeros((am.actuation.nu, am.actuation.nu))
        self.Lf = np.zeros(am.contacts.nc_tot)
        self.Lff = np.zeros((am.contacts.nc_tot, am.contacts.nc_tot))
        self.f_residual = np.zeros(am.contacts.nc_tot)
        # External spatial force in body coordinates
        self.fext = [pin.Force.Zero() for _ in range(am.pinocchio.njoints)]
        self.fext_copy = [pin.Force.Zero() for _ in range(am.pinocchio.njoints)]
        # Data containers
        self.pinocchio  = am.pinocchio.createData()
        self.actuation_data = am.actuation.createData()
        self.multibody = crocoddyl.DataCollectorActMultibody(self.pinocchio, self.actuation_data)
        self.costs = am.costs.createData(self.multibody)
        if(am.constraints is not None):
            self.constraints = am.constraints.createData(self.multibody)   

import force_feedback_mpc
# Custom Integrated Action Model class for go2+arm
class IAMSoftContactDynamics3D_Go2(crocoddyl.ActionModelAbstract): #IntegratedActionModelAbstract
    def __init__(self, dam, dt=1e-3, withCostResidual=True, frictionConeConstaints = []):
        nu = int(dam.nu)
        nc = int(dam.contacts.nc)      # number of contact models (5)
        nf = int(dam.contacts.nc_tot)  # contact space dimension (5x3 = 15)
        nr = int(dam.costs.nr + 3)     # ef force cost)
        ng = int(dam.ng + nf + len(frictionConeConstaints))
        # crocoddyl.ActionModelAbstract.__init__(self, crocoddyl.StateAbstract(dam.state.nq + dam.state.nv + nf, dam.state.ndx + nf), nu, nr, ng, 0)
        # crocoddyl.ActionModelAbstract.__init__(self, crocoddyl.StateVector(dam.state.nq + dam.state.nv + nf), nu, nr, ng, 0)
        # crocoddyl.ActionModelAbstract.__init__(self, dam.state, nu, nr, ng, 0)
        # crocoddyl.ActionModelAbstract.__init__(self, force_feedback_mpc.StateSoftContact(dam.pinocchio, nf), nu, nr, ng, 0)
        crocoddyl.ActionModelAbstract.__init__(self, force_feedback_mpc.StateSoftContact(dam.pinocchio, nf), nu, nr, ng, 0)
        self.differential = dam
        self.stateSoft = force_feedback_mpc.StateSoftContact(dam.pinocchio, nf)
        self.nf = nf
        self.nc = nc
        self.ny = self.stateSoft.nx 
        self.ndy = self.stateSoft.ndx 
        self.dt = dt
        self.withCostResidual = withCostResidual
        # Force constraint
        self.with_force_constraint = False
        self.force_lb = np.array([-np.inf]*self.nf)
        self.force_ub = np.array([np.inf]*self.nf)
        self.g_lb = np.array([np.inf]*self.ng)
        self.g_ub = np.array([np.inf]*self.ng)
        # Friction cone constraints
        self.init_friction_cones(frictionConeConstaints)

    def init_friction_cones(self, frictionConeConstraints):
        '''
        Set friction cone constraints
        '''
        self.frictionConeConstaints = frictionConeConstraints
        self.ncones = len(frictionConeConstraints)
        if(self.ncones==0):
            self.with_friction_cone_constraint = False
        else:
            self.with_friction_cone_constraint = True
            
    def set_force_lb(self, lb):
        '''
        Set contact force lower bound and update the IAM bounds accordingly
        '''
        g_lb_new = self.g_lb.copy()
        g_lb_new[self.differential.ng:self.differential.ng+self.nf] = lb 
        self.g_lb = g_lb_new
        print("updated g_lb = ", self.g_lb)

    def set_force_ub(self, ub):
        '''
        Set contact force upper bound and update the IAM bounds accordingly
        '''
        g_ub_new = self.g_ub.copy()
        g_ub_new[self.differential.ng:self.differential.ng+self.nf] = ub 
        self.g_ub = g_ub_new
        print("updated g_ub = ", self.g_ub)

    def createData(self):
        data = IADSoftContactDynamics3D_Go2(self)
        return data
    
    def calc(self, data, y, u=None):
        nx = self.differential.state.nx
        nv = self.differential.state.nv
        nq = self.differential.state.nq
        ng_dam = self.differential.ng
        nf = self.nf
        x = y[:nx]
        f = y[-nf:]
        # q = x[:self.state.nq]
        v = x[-nv:]
        # self.control.calc(data.control, 0., u)
        if(u is not None):
            self.differential.calc(data.differential, x, f, u) 
            a = data.differential.xout
            fdot = data.differential.fout
            data.dx[:nv] = v*self.dt + a*self.dt**2
            data.dx[nv:2*nv] = a*self.dt
            data.dx[-nf:] = fdot*self.dt
            data.xnext = self.stateSoft.integrate(y, data.dx)
            data.cost = self.dt*data.differential.cost
            # Compute cost residual
            if(self.withCostResidual):
                data.r = data.differential.r
            # Compute force constraint residual
            if(self.with_force_constraint):
                data.g[self.differential.ng: self.differential.ng+nf] = f
            # Compute friction cone constraint residual
            if(self.with_friction_cone_constraint):
                for i, fc in enumerate(self.frictionConeConstaints):
                    contact_force_i = f[3*i:3*(i+1)]
                    cone_residual_i = fc.calc(contact_force_i)
                    data.g[ng_dam + nf + i] = cone_residual_i
        else:
            self.differential.calc(data.differential, x, f) 
            a = data.differential.xout
            fdot = data.differential.fout
            data.dx = np.zeros(data.dx.shape)
            data.cost = self.dt*data.differential.cost
            # Compute cost residual
            if(self.withCostResidual):
                data.r = data.differential.r
            # Compute force constraint residual
            if(self.with_force_constraint):
                data.g[self.differential.ng: self.differential.ng+nf] = f
            # Compute friction cone constraint residual
            if(self.with_friction_cone_constraint):
                for i, fc in enumerate(self.frictionConeConstaints):
                    contact_force_i = f[3*i:3*(i+1)]
                    cone_residual_i = fc.calc(contact_force_i)
                    data.g[ng_dam + nf + i] = cone_residual_i

    def calcDiff(self, data, y, u=None):
        nx = self.differential.state.nx
        ndx = self.differential.state.ndx
        nv = self.differential.state.nv
        nu = self.differential.nu
        nf = self.nf
        ng_dam = self.differential.ng
        x = y[:nx]
        f = y[-nf:]

        # Calc forward dyn derivatives
        if(u is not None):
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
            data.Fx[:nv, -nf:] = data.differential.dABA_df * self.dt**2
            data.Fx[nv:ndx, -nf:] = data.differential.dABA_df * self.dt
            # New block from augmented dynamics (bottom right corner)
            data.Fx[-nf:,-nf:] = np.eye(nf) + data.differential.dfdt_df*self.dt
            # New block from augmented dynamics (bottom left corner)
            data.Fx[-nf:, :ndx] = data.differential.dfdt_dx * self.dt
            
            data.Fu[-nf:, :] = data.differential.dfdt_du * self.dt

            self.stateSoft.JintegrateTransport(y, data.dx, data.Fx, crocoddyl.Jcomponent.second)
            data.Fx += self.stateSoft.Jintegrate(y, data.dx, crocoddyl.Jcomponent.first).tolist()[0]  # add identity to Fx = d(x+dx)/dx = d(q,v)/d(q,v)
            data.Fx[-nf:, -nf:] -= np.eye(nf) # remove identity from Ftau (due to stateSoft.Jintegrate)
            self.stateSoft.JintegrateTransport(y, data.dx, data.Fu, crocoddyl.Jcomponent.second)

            # d->Lx.noalias() = time_step_ * d->differential->Lx;
            data.Lx[:ndx] = data.differential.Lx*self.dt
            data.Lx[-nf:] = data.differential.Lf*self.dt
            data.Lxx[:ndx,:ndx] = data.differential.Lxx*self.dt
            data.Lxx[-nf:,-nf:] = data.differential.Lff*self.dt
            data.Lxu[:ndx, :nu] = data.differential.Lxu*self.dt
            data.Lu = data.differential.Lu*self.dt
            data.Luu = data.differential.Luu*self.dt

            data.Gx[0:ng_dam, 0:ndx] = data.differential.Gx
            data.Gu[0:ng_dam, 0:nu] = data.differential.Gu
            if(self.with_force_constraint):
                data.Gx[ng_dam:ng_dam+nf, ndx:ndx+nf] = np.eye(nf)
            # Compute friction cone constraint residual partials
            if(self.with_friction_cone_constraint):
                for i, fc in enumerate(self.frictionConeConstaints):
                    contact_force_i      = f[3*i:3*(i+1)]
                    if(np.linalg.norm(contact_force_i)>1e-3):
                        cone_residual_diff_i = fc.calcDiff(contact_force_i)
                        data.Gx[ng_dam+nf+i:ng_dam+nf+i+1, ndx+nf+3*i:ndx+nf+3*(i+1)] = cone_residual_diff_i
        # u = none
        else:
            # Calc forward dyn derivatives
            self.differential.calcDiff(data.differential, x, f)
            # state_->Jintegrate(x, d->dx, d->Fx, d->Fx, first, addto);
            data.Fx += self.stateSoft.Jintegrate(y, data.dx, crocoddyl.Jcomponent.first).tolist()[0]  # add identity to Fx = d(x+dx)/dx = d(q,v)/d(q,v)
            # data.Fx (nu, nu).diagonal().array() -=
            #     Scalar(1.);  // remove identity from Ftau (due to stateLPF.Jintegrate)
            data.Fx[-nf:, -nf:] -= np.eye(nf)
            # d->Lx.noalias() = time_step_ * d->differential->Lx;
            data.Lx[:ndx] = data.differential.Lx*self.dt
            data.Lx[-nf:] = data.differential.Lf*self.dt
            data.Lxx[:ndx,:ndx] = data.differential.Lxx*self.dt
            data.Lxx[-nf:,-nf:] = data.differential.Lff*self.dt
            data.Lxu[:ndx, :nu] = data.differential.Lxu*self.dt
            data.Lu = data.differential.Lu*self.dt
            data.Luu = data.differential.Luu*self.dt
            data.Gx[0:ng_dam, 0:ndx] = data.differential.Gx
            data.Gu[0:ng_dam, 0:nu] = data.differential.Gu
            if(self.with_force_constraint):
                data.Gx[ng_dam:ng_dam+nf, ndx:ndx+nf] = np.eye(nf)
            # Compute friction cone constraint residual partials
            if(self.with_friction_cone_constraint):
                for i, fc in enumerate(self.frictionConeConstaints):
                    contact_force_i      = f[3*i:3*(i+1)]
                    if(np.linalg.norm(contact_force_i)>1e-3):
                        cone_residual_diff_i = fc.calcDiff(contact_force_i)
                        data.Gx[ng_dam+nf+i:ng_dam+nf+i+1, ndx+nf+3*i:ndx+nf+3*(i+1)] = cone_residual_diff_i

class IADSoftContactDynamics3D_Go2(crocoddyl.ActionDataAbstract): 
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
        # Constraints
        self.Gx = np.zeros((am.ng, am.ndy))
        self.Gu = np.zeros((am.ng, am.nu))
        # self.dcone_df = np.zeros((am.nc, am.nf))


from mim_robots.robot_loader import load_pinocchio_wrapper
robot = load_pinocchio_wrapper('iiwa')
state = crocoddyl.StateMultibody(robot.model)
actuation = crocoddyl.ActuationModelFull(state)
costs = crocoddyl.CostModelSum(state, actuation.nu)
constraintModelManager = crocoddyl.ConstraintModelManager(state, actuation.nu)
frameId = robot.model.getFrameId('contact')
Kp = 0 ; Kv = 0 ; oPc = np.zeros(3)

# Test create data 
dam = DAMSoftContactDynamics3D_Go2(state, actuation, costs, frameId, Kp, Kv, oPc, pin.LOCAL_WORLD_ALIGNED, constraintModelManager)
dad = dam.createData()

# Test calc and calcDiff
q = pin.randomConfiguration(robot.model)
v = np.zeros(robot.model.nv)
x = np.concatenate([q, v])
u = np.random.rand(actuation.nu)
f = np.random.rand(3)
dam.calc(dad, x, f, u)
dam.calcDiff(dad, x, f, u)

# import force_feedback_mpc
# # TEST go2 with arm
# # From Go2Py/examples/09-crocoddyl.ipynb
# import pinocchio as pin
# import crocoddyl
# import pinocchio
# import numpy as np
# urdf_root_path = '/home/skleff/force_feedback_ws/Go2Py/Go2Py/assets/'
# urdf_path = '/home/skleff/force_feedback_ws/Go2Py/Go2Py/assets/urdf/go2_with_arm.urdf'
# robot = pin.RobotWrapper.BuildFromURDF(
# urdf_path, urdf_root_path, pin.JointModelFreeFlyer())

# pinRef        = pin.LOCAL_WORLD_ALIGNED
# FRICTION_CSTR = True
# MU = 0.8     # friction coefficient
# ee_frame_names = ['FL_FOOT', 'FR_FOOT', 'HL_FOOT', 'HR_FOOT', 'Link6']
# rmodel = robot.model
# rdata = robot.data
# # # set contact frame_names and_indices
# lfFootId = rmodel.getFrameId(ee_frame_names[0])
# rfFootId = rmodel.getFrameId(ee_frame_names[1])
# lhFootId = rmodel.getFrameId(ee_frame_names[2])
# rhFootId = rmodel.getFrameId(ee_frame_names[3])
# efId = rmodel.getFrameId(ee_frame_names[4])

# q0 = np.array([0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0] 
#                +4*[0.0, 0.77832842, -1.56065452] + 8*[0.0]
#                 )
# x0 =  np.concatenate([q0, np.zeros(rmodel.nv)])

# q0 = np.array([0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0] 
#                +4*[0.0, 0.77832842, -1.56065452] + 8*[0.0]
#                 )
# x0 =  np.concatenate([q0, np.zeros(rmodel.nv)])

# pinocchio.forwardKinematics(rmodel, rdata, q0)
# pinocchio.updateFramePlacements(rmodel, rdata)
# rfFootPos0 = rdata.oMf[rfFootId].translation
# rhFootPos0 = rdata.oMf[rhFootId].translation
# lfFootPos0 = rdata.oMf[lfFootId].translation
# lhFootPos0 = rdata.oMf[lhFootId].translation 
# efPos0 = rdata.oMf[efId].translation
# comRef = (rfFootPos0 + rhFootPos0 + lfFootPos0 + lhFootPos0) / 4
# comRef[2] = pinocchio.centerOfMass(rmodel, rdata, q0)[2].item() 
# print(f'The desired CoM position is: {comRef}')
# supportFeetIds = [lfFootId, rfFootId, lhFootId, rhFootId]
# supportFeePos = [lfFootPos0, rfFootPos0, lhFootPos0, rhFootPos0]

# state = crocoddyl.StateMultibody(rmodel)
# actuation = crocoddyl.ActuationModelFloatingBase(state)
# nu = actuation.nu

# f0 = np.zeros(3*5)
# u0 = np.zeros(actuation.nu)
# y0 = np.concatenate([x0, f0])
# comDes = []
# N_ocp = 100 #250
# dt = 0.02
# T = N_ocp * dt
# radius = 0.065
# for t in range(N_ocp+1):
#     comDes_t = comRef.copy()
#     w = (2 * np.pi) * 0.2 # / T
#     comDes_t[0] += radius * np.sin(w * t * dt) 
#     comDes_t[1] += radius * (np.cos(w * t * dt) - 1)
#     comDes += [comDes_t]
# running_models = []
# constraintModels = []
# for t in range(N_ocp+1):
#     costModel = crocoddyl.CostModelSum(state, nu)

#     # Add state/control reg costs
#     state_reg_weight, control_reg_weight = 1e-1, 1e-3
#     freeFlyerQWeight = [0.]*3 + [500.]*3
#     freeFlyerVWeight = [10.]*6
#     legsQWeight = [0.01]*(rmodel.nv - 6)
#     legsWWeights = [1.]*(rmodel.nv - 6)
#     stateWeights = np.array(freeFlyerQWeight + legsQWeight + freeFlyerVWeight + legsWWeights)    
#     stateResidual = crocoddyl.ResidualModelState(state, x0, nu)
#     stateActivation = crocoddyl.ActivationModelWeightedQuad(stateWeights**2)
#     stateReg = crocoddyl.CostModelResidual(state, stateActivation, stateResidual)
#     costModel.addCost("stateReg", stateReg, state_reg_weight)
#     ctrlResidual = crocoddyl.ResidualModelControl(state, nu)
#     ctrlReg = crocoddyl.CostModelResidual(state, ctrlResidual)
#     costModel.addCost("ctrlReg", ctrlReg, control_reg_weight)  

#     # Add COM task
#     com_residual = crocoddyl.ResidualModelCoMPosition(state, comDes[t], nu)
#     com_activation = crocoddyl.ActivationModelWeightedQuad(np.array([1., 1., 1.]))
#     com_track = crocoddyl.CostModelResidual(state, com_activation, com_residual) # What does it correspond to exactly?
#     costModel.addCost("comTrack", com_track, 1e5)

#     # End Effecor Position Task
#     ef_residual = crocoddyl.ResidualModelFrameTranslation(state, efId, efPos0, nu)
#     ef_activation = crocoddyl.ActivationModelWeightedQuad(np.array([1., 1., 1.]))
#     ef_track = crocoddyl.CostModelResidual(state, ef_activation, ef_residual)
#     costModel.addCost("efTrack", ef_track, 1e5)
        
#     # constraintModelManager = crocoddyl.ConstraintModelManager(state, actuation.nu)

#     # Friction cone constraint models
#     frictionConeConstaints = [FrictionConeConstraint(state, lfFootId, MU),
#                               FrictionConeConstraint(state, rfFootId, MU),
#                               FrictionConeConstraint(state, lhFootId, MU),
#                               FrictionConeConstraint(state, rhFootId, MU)]
    
#     dam = DAMSoftContactDynamics3D_Go2(state, actuation, costModel) #, constraintModelManager)
#     # dad = dam.createData()
#     # dam.calc(dad, x0, f0, u0)
#     # dam.calcDiff(dad, x0, f0, u0)
#     iam = IAMSoftContactDynamics3D_Go2(dam, dt=dt, withCostResidual=True) #, frictionConeConstaints=frictionConeConstaints)
#     # iad = iam.createData()
#     # iam.calc(iad, y0, u0)
#     # iam.calcDiff(iad, y0, u0)
#     running_models += [iam]

# import mim_solvers
# # Create shooting problem
# ocp = crocoddyl.ShootingProblem(y0, running_models[:-1], running_models[-1])
# ocp.x0 = y0

# solver = mim_solvers.SolverCSQP(ocp)
# solver.max_qp_iters = 1000
# max_iter = 500
# solver.with_callbacks = True
# solver.use_filter_line_search = True
# solver.termination_tolerance = 1e-4
# solver.eps_abs = 1e-6
# solver.eps_rel = 1e-6

# xs = [y0]*(solver.problem.T + 1)
# us = [u0]*solver.problem.T

# # us = solver.problem.quasiStatic([x0]*solver.problem.T) 
# solver.setCallbacks([mim_solvers.CallbackVerbose(), mim_solvers.CallbackLogger()])
# solver.solve(xs, us, max_iter)   

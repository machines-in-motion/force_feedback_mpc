'''
Multi-contact API for visco-elastic contact models (augmented state)
This API was implemented in order to achieve MPC on GO2 with 5 contacts
Could be slightly generalized and implemented in C++ for hardware experiments 
'''

from croco_mpc_utils.utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger

import numpy as np
np.set_printoptions(precision=6, linewidth=180, suppress=True)
np.random.seed(1)

import crocoddyl
import pinocchio as pin
import force_feedback_mpc


# 3D soft contact model 
class ViscoElasticContact3D:
    def __init__(self, state, actuation, frameId, oPc=np.zeros(3), Kp=10, Kv=0, pinRef=pin.LOCAL_WORLD_ALIGNED):
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
        self.dABA_df3d = np.zeros((self.state.nv, self.nc))    # derivative of ada (ddq) w.r.t. 3d force
        self.dfdt_dx = np.zeros((self.nc, self.state.ndx))     # derivative of fdot (3D) w.r.t. state 
        self.dfdt_du = np.zeros((self.nc, self.actuation.nu))  # derivative of fdot (3D) w.r.t. control
        self.dfdt_df = np.zeros((self.nc, self.nc))            # derivative of fdot (3D) w.r.t. forces

    def calc(self, dad, f):
        '''
        Calculate the contact force spatial wrench 
            f : 3d force
        '''
        # Placement of contact frame in WORLD
        oRf = dad.pinocchio.oMf[self.frameId].rotation
        # Compute external wrench given LOCAL 3d force
        pinForce = pin.Force(f, np.zeros(3))
        # Rotate if f is not expressed in LOCAL
        if(self.pinRef != pin.LOCAL):
            pinForce = pin.Force(oRf.T @ f, np.zeros(3))
        self.fext = self.jMf.act(pinForce)
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
        self.dABA_df3d = dad.pinocchio.Minv @ lJ[:3].T  
        # Skew term added to RNEA derivatives when force is expressed in LWA
        if(self.pinRef != pin.LOCAL):
            # logger.debug("corrective term aba LWA : \n"+str(data.pinocchio.Minv @ lJ[:3].T @ pin.skew(oRf.T @ f) @ lJ[3:]))
            dad.Fx[:,:self.state.nv] += dad.pinocchio.Minv @ lJ[:3].T @ pin.skew(oRf.T @ f) @ lJ[3:]
            # Rotate dABA/df
            self.dABA_df3d = self.dABA_df3d @ oRf.T 

    def calcDiff(self, dad):  
        '''
        Compute partial derivatives of the soft contact force time derivative w.r.t. state and input
        must be called after update_ABAderivatives
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
        da_df = a_da[:3] @ dad.dABA_df # this dad attribute was computed in update update_ABAderivatives
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
                dad.dABA_df[0:self.nv, nc_i:nc_i+ct.nc] = ct.dABA_df3d
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


# 3D Friction cone constraint
class FrictionConeConstraint:
    def __init__(self, frameId, coef):
        self.nc           = 3
        self.nr           = 1
        self.active       = True
        self.frameId      = frameId
        self.coef         = coef
        self.residual     = np.zeros(self.nr)
        self.residual_df = np.zeros((self.nr, self.nc))
        self.lb = np.array([0.])
        self.ub = np.array([np.inf])

    def calc(self, f):
        self.residual = self.coef * f[2] - np.sqrt(f[0]*f[0] + f[1]*f[1])
        return self.residual
    
    def calcDiff(self, f):
        if(np.linalg.norm(f) > 1e-3):
            self.residual_df[:, 0] = -f[0] / np.sqrt(f[0]*f[0] + f[1]*f[1])
            self.residual_df[:, 1] = -f[1] / np.sqrt(f[0]*f[0] + f[1]*f[1])
            self.residual_df[:, 2] = self.coef
        else:
            self.residual_df = np.zeros((self.nr, self.nc))
        return self.residual_df


# 3D force box constraint
class ForceBoxConstraint:
    def __init__(self, frameId, lb, ub):
        self.nc          = 3
        self.nr          = 3
        self.active      = True
        self.frameId     = frameId
        self.residual    = np.zeros(self.nr)
        self.residual_df = np.zeros((self.nr,self.nc))
        self.lb = lb
        self.ub = ub 
        
    def calc(self, f):
        self.residual = f
        return self.residual
    
    def calcDiff(self, f):
        self.residual_df = np.eye(self.nc)
        return self.residual_df


# 3D force constraint manager
class ForceConstraintManager:
    def __init__(self, constraints, contacts):
        self.constraints = constraints
        self.nr          = np.sum(np.array([cstr.nr for cstr in self.constraints])) 

        # Check that contact models are defined at the same frames
        assert(contacts is not None)
        self.contacts            = contacts.contacts
        assert(len(self.contacts)>0)
        # for each constraint 
        for cstr in self.constraints:
            found = False
            # check that a contact model is defined at the same frame
            for ct in self.contacts:
                if(ct.frameId == cstr.frameId):
                    found = True           
            if(found == False):
                print("ERROR : no contact model was found that matches the constraint on frame ", cstr.frameId)
            assert(found == True)
        
        # Construct mapping from contact models to constraint residuals
        self.contact_to_cstr_map = {}
        self.contact_to_nr_map = {}
        for ct in self.contacts:
            self.contact_to_cstr_map[ct.frameId] = []
            self.contact_to_nr_map[ct.frameId] = 0
            for cstr in self.constraints:
                if(cstr.frameId == ct.frameId):
                    self.contact_to_cstr_map[ct.frameId].append(cstr)
                    self.contact_to_nr_map[ct.frameId] += cstr.nr
        # print("MAP cstr = \ns", self.contact_to_cstr_map)
        # print("MAP nr = \ns", self.contact_to_nr_map)

        self.nc          = np.sum(np.array([cstr.nc for cstr in self.contacts])) 

        self.residual    = np.zeros(self.nr)
        self.residual_df = np.zeros((self.nr, self.nc))
        
        # check constraints types
        self.has_force_constraint = False
        if(len(constraints)>0):
            self.has_force_constraint = True
        self.lb = np.concatenate([cstr.lb for cstr in self.constraints])
        self.ub = np.concatenate([cstr.ub for cstr in self.constraints])

    def calc(self, f):
        '''
            f : stack of 3d forces for each contact model
            output > stacked constraint residuals
        '''
        nc_i = 0
        nr_i = 0
        # For each contact model
        for ct in self.contacts:
            # Current residual index
            # For each constraint active at this model
            for cstr in self.contact_to_cstr_map[ct.frameId]:
                # Compute the constraint residual
                self.residual[nr_i:nr_i+cstr.nr] = cstr.calc(f[nc_i:nc_i+ct.nc])
                nr_i += cstr.nr
            nc_i += ct.nc
        # For each constraint, stack residual 
        return self.residual

    def calcDiff(self, f):
        '''
            f : stack of 3d forces
            output > stacked constraint Jacobians
        '''
        nc_i = 0
        nr_i   = 0
        # For each contact model
        for ct in self.contacts:
            # Current residual index
            # For each constraint active at this model
            for cstr in self.contact_to_cstr_map[ct.frameId]:
                # Compute the constraint residual
                self.residual_df[nr_i:nr_i+cstr.nr, nc_i:nc_i+cstr.nc] = cstr.calcDiff(f[nc_i:nc_i+ct.nc])
                nr_i += cstr.nr
            nc_i += ct.nc
        # For each constraint, stack residual 
        return self.residual_df


# 3D Force cost 
class ForceCost:
    def __init__(self, state, frameId, f_des, f_weight, pinRef):
        self.frameId       = frameId
        self.state         = state
        self.f_des         = f_des
        self.f_weight      = np.diag([f_weight,0.,0.]) # np.diag([f_weight]*3) # 
        self.f_residual    = np.zeros(3)
        self.f_residual_x  = np.zeros((3,state.ndx))
        self.f_cost        = 0.
        self.Lf            = np.zeros(3)
        self.Lff           = np.zeros((3,3))
        self.pinRef        = pinRef
        self.nr = 3 # cost residual dimension
        self.nc = 3 # force dimension

    def calc(self, dad, f, pinRefDyn):
        # Placement of contact frame in WORLD
        oRf = dad.pinocchio.oMf[self.frameId].rotation
        # Compute force residual and add force cost to total cost
        if(self.pinRef != pinRefDyn):
            if(self.pinRef == pin.LOCAL):
                self.f_residual = oRf.T @ f - self.f_des
            else:
                self.f_residual = oRf @ f - self.f_des
        else:
            self.f_residual = f - self.f_des
        self.f_cost     = 0.5 * self.f_residual.T @ self.f_weight @ self.f_residual
        return self.f_cost
    
    def calcDiff(self, dad, f, pinRefDyn):
        # Placement of contact frame in WORLD
        oRf = dad.pinocchio.oMf[self.frameId].rotation
        lJ = pin.getFrameJacobian(self.state.pinocchio, dad.pinocchio, self.frameId, pin.LOCAL)  
        # Compute force residual and add force cost to total cost
        if(self.pinRef != pinRefDyn):
            if(self.pinRef == pin.LOCAL):
                self.f_residual = oRf.T @ f - self.f_des
                self.Lf = self.f_residual.T @ self.f_weight @ oRf.T
                self.f_residual_x[:3, :self.state.nv] = pin.skew(oRf.T @ f) @ lJ[3:]
                dad.Lx += self.f_residual.T @ self.f_weight @ self.f_residual_x
                self.Lff = self.f_weight @ oRf @ oRf.T
            else:
                self.f_residual = oRf @ f - self.f_des
                self.Lf = self.f_residual.T @ self.f_weight @ oRf
                self.f_residual_x[:3, :self.state.nv] = pin.skew(oRf @ f) @ self.oJ.bottomRows(3)
                dad.Lx += self.f_residual.T @ self.f_weight @ pin.skew(oRf @ f) @ self.f_residual_x
                self.Lff = self.f_weight @ oRf.T @ oRf
        else:
            self.f_residual = f - self.f_des
            self.Lf = self.f_residual.T @ self.f_weight
            self.Lff = self.f_weight 
        self.f_residual = f - self.f_des
        # self.f_cost     = 0.5 * self.f_weight * self.f_residual.T @ self.f_residual
        return self.Lf, self.Lff


class ForceCostManager:
    def __init__(self, forceCosts, contacts):
        self.forceCosts = forceCosts
        self.nr          = np.sum(np.array([cost.nr for cost in self.forceCosts])) 

        # Check that contact models are defined at the same frames
        assert(contacts is not None)
        self.contacts            = contacts.contacts
        assert(len(self.contacts)>0)
        # for each constraint 
        for cost in self.forceCosts:
            found = False
            # check that a contact model is defined at the same frame
            for ct in self.contacts:
                if(ct.frameId == cost.frameId):
                    found = True           
            if(found == False):
                print("ERROR : no contact model was found that matches the constraint on frame ", cost.frameId)
            assert(found == True)
        
        # Construct mapping from contact models to constraint residuals
        self.contact_to_cost_map = {}
        self.contact_to_nr_map = {}
        for ct in self.contacts:
            self.contact_to_cost_map[ct.frameId] = []
            self.contact_to_nr_map[ct.frameId] = 0
            for cost in self.forceCosts:
                if(cost.frameId == ct.frameId):
                    self.contact_to_cost_map[ct.frameId].append(cost)
                    self.contact_to_nr_map[ct.frameId] += cost.nr
        # print("MAP cost = \ns", self.contact_to_cost_map)
        # print("MAP nr = \ns", self.contact_to_nr_map)

        self.nc          = np.sum(np.array([cost.nc for cost in self.contacts])) 

        self.cost = 0.
        self.Lf            = np.zeros(self.nc)
        self.Lff           = np.zeros((self.nc,self.nc))
        # print("force cost residual nr = ", self.nr)
        # print("force cost nc = ", self.nc)


    def calc(self, dad, f):
        '''
            dad       : differential action data (soft contact 3d)
            f         : stack of 3d forces
            pinRefDyn : reference frame in which the dynamics is expressed
        output > stacked cost residuals
        '''
        nc_i = 0
        self.cost = 0
        # For each contact model
        for ct in self.contacts:
            pinRefDyn = ct.pinRef
            # Current residual index
            nr_i   = 0
            # For each constraint active at this model
            for cost in self.contact_to_cost_map[ct.frameId]:
                self.cost += cost.calc(dad, f[nc_i:nc_i+ct.nc], pinRefDyn)
                nr_i += cost.nr
            nc_i += ct.nc
        # For each constraint, stack residual 
        return self.cost

    def calcDiff(self, dad, f):
        '''
            dad       : differential action data (soft contact 3d)
            f         : stack of 3d forces
            pinRefDyn : reference frame in which the dynamics is expressed
        output > stacked cost Jacobians
        '''
        nc_i = 0
        # For each contact model
        for ct in self.contacts:
            pinRefDyn = ct.pinRef
            # Current residual index
            nr_i   = 0
            # For each constraint active at this model
            for cost in self.contact_to_cost_map[ct.frameId]:
                Lf_ct, Lff_ct = cost.calcDiff(dad, f[nc_i:nc_i+ct.nc], pinRefDyn)
                self.Lf[nc_i:nc_i+cost.nc] = Lf_ct
                self.Lff[nr_i:nr_i+cost.nr, nc_i:nc_i+cost.nc] = Lff_ct
                nr_i += cost.nr
            nc_i += ct.nc
        # For each constraint, stack residual 
        return self.Lf, self.Lff



# Custom Differential Action Model (DAM) for Go2+arm (5 soft 3D contacts with the environment)
class DAMSoftContactDynamics3D_Go2(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, stateMultibody, actuationModel, costModelSum, softContactModelsStack=None, constraintModelManager=None, forceCostManager=None):
        '''
            stateMultibody         : crocoddyl.stateMultibody
            actuationModel         : crocoddyl.actuationModelFloatingBase
            costModelSum           : crocoddyl.costModelSum
            constraintModelManager : crocoddyl.constraintModelManager
        '''
        # Determine constraint dimensions if any
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
        ee_frame_names = ['FL_FOOT', 'FR_FOOT', 'HL_FOOT', 'HR_FOOT', 'Link6']
        self.lfFootId  = stateMultibody.pinocchio.getFrameId(ee_frame_names[0])
        self.rfFootId  = stateMultibody.pinocchio.getFrameId(ee_frame_names[1])
        self.lhFootId  = stateMultibody.pinocchio.getFrameId(ee_frame_names[2])
        self.rhFootId  = stateMultibody.pinocchio.getFrameId(ee_frame_names[3])
        self.efId      = stateMultibody.pinocchio.getFrameId(ee_frame_names[4])
        # Soft contact models (3d) stack
        self.contacts = softContactModelsStack 
        if(softContactModelsStack is None):
            self.active_contact = False
            self.nc_tot = 0
            self.nc = 0
        else:
            self.active_contact = self.contacts.active
            self.nc_tot = self.contacts.nc_tot
            self.nc = self.contacts.nc
        # To complete DAMAbstract into sth like DAMFwdDyn
        self.nx          = stateMultibody.nx
        self.actuation   = actuationModel
        self.costs       = costModelSum
        self.constraints = constraintModelManager
        self.pinocchio   = stateMultibody.pinocchio
        # hard coded costs 
        self.forceCosts = forceCostManager
        if(self.forceCosts is None):
            self.nr_f            = 0
            self.with_force_cost = False
        else:
            self.nr_f            = self.forceCosts.nr
            self.with_force_cost = True
        # print("force cost red dim = ", self.nr_f)
        # Init constraint bounds
        if(self.constraints is not None):
            self.init_cstr_bounds()
    
    def init_cstr_bounds(self):
        '''
        Initialize the constraint bounds of the constraint model manager
        '''
        # dummy data and state
        dummy_data = self.createData()
        x = np.zeros(self.nx)
        f = np.zeros(self.nc_tot)
        u = np.zeros(self.actuation.nu)
        self.calc(dummy_data, x, f, u)
        # Set the constraints bounds
        self.g_lb = self.constraints.g_lb
        self.g_ub = self.constraints.g_ub
    
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
                data.cost += self.forceCosts.calc(data, f)
            # constraints
            if(self.constraints is not None):
                self.constraints.calc(data.constraints, x, u)
                
        else:
            # pass
            data.xout = np.zeros(data.xout.shape)
            data.fout = np.zeros(data.fout.shape)
            pin.computeAllTerms(self.pinocchio, data.pinocchio, q, v)
            # import pdb; pdb.set_trace()
            self.costs.calc(data.costs, x) 
            data.cost = data.costs.cost
            # Add hard-coded cost
            if(self.active_contact and self.with_force_cost):
                data.cost += self.forceCosts.calc(data, f)
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
                data.Lf, data.Lff = self.forceCosts.calcDiff(data, f)
            # constraints
            if(self.constraints is not None):
                self.constraints.calcDiff(data.constraints, x, u)
        else:
            self.costs.calcDiff(data.costs, x)
            data.Lx = data.costs.Lx
            data.Lxx = data.costs.Lxx
            # add hard-coded cost
            if(self.active_contact and self.with_force_cost):
                data.Lf, data.Lff = self.forceCosts.calcDiff(data, f)
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
        self.fout      = np.zeros(am.nc_tot)
        self.fout_copy = np.zeros(am.nc_tot)
        self.dfdt_dx   = np.zeros((am.nc_tot,am.state.ndx))
        self.dfdt_du   = np.zeros((am.nc_tot,am.nu))
        self.dfdt_df   = np.zeros((am.nc_tot,am.nc_tot))  
        # ABA model derivatives
        self.Fx = np.zeros((am.state.nv, am.state.ndx))
        self.Fu = np.zeros((am.state.nv, am.nu))
        self.dABA_df = np.zeros((am.state.nv, am.nc_tot))
        # Cost derivatives
        self.Lx = np.zeros(am.state.ndx)
        self.Lu = np.zeros(am.actuation.nu)
        self.Lxx = np.zeros((am.state.ndx, am.state.ndx))
        self.Lxu = np.zeros((am.state.ndx, am.actuation.nu))
        self.Luu = np.zeros((am.actuation.nu, am.actuation.nu))
        self.Lf = np.zeros(am.nr_f)
        self.Lff = np.zeros((am.nr_f, am.nr_f))
        self.f_residual = np.zeros(am.nr_f)
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

# Custom Integrated Action Model class for go2+arm
class IAMSoftContactDynamics3D_Go2(crocoddyl.ActionModelAbstract): #IntegratedActionModelAbstract
    def __init__(self, dam, dt=1e-3, withCostResidual=True, forceConstraintManager = None):
        
        nu = int(dam.nu)
        nr = int(dam.costs.nr + 3)     # ef force cost
        # Determine size of the force constraints
        if(forceConstraintManager is None):
            ng_f = 0 # dimension of force constraint residual
        else:
            ng_f = forceConstraintManager.nr
        # Total constraint dimension
        ng = int(dam.ng + ng_f)
        
        crocoddyl.ActionModelAbstract.__init__(self, force_feedback_mpc.StateSoftContact(dam.pinocchio, int(dam.nc_tot)), nu, nr, ng, 0)
        
        self.differential     = dam                                                            # Custom DAM
        self.stateSoft        = force_feedback_mpc.StateSoftContact(dam.pinocchio, int(dam.nc_tot)) # Custom state 
        self.nf               = dam.nc_tot                                                     # total force space dimension
        self.ny               = self.stateSoft.nx                                              # augmented state dimension (classical state + force)
        self.ndy              = self.stateSoft.ndx                                             # tangent space
        self.ng               = ng                                                             # Total contraint dimension
        self.ng_f             = ng_f                                                           # Custom force contraint dimension
        self.ng_dam           = self.differential.ng                                           # crocoddyl constraint manager dimension
        assert(self.ng == self.ng_dam + self.ng_f)
        # print("Total constrain dim = ", self.ng)
        # print("Force constrain dim = ", self.ng_f)
        # print("DAM constrain dim   = ", self.ng_dam)
        self.dt               = dt
        
        self.withCostResidual = withCostResidual
        self.forceConstraints = forceConstraintManager                                         # custom force constraint manager

        # Set up custom force constraints bounds
        self.forceConstraints = forceConstraintManager
        if(self.forceConstraints is None):
            self.with_force_constraint = False
            self.force_g_lb            = []
            self.force_g_ub            = []
            self.nc_f = 0
        else:
            self.with_force_constraint = self.forceConstraints.has_force_constraint
            self.force_g_lb            = self.forceConstraints.lb
            self.force_g_ub            = self.forceConstraints.ub
            self.nc_f                  = self.forceConstraints.nc  
        
        # Combine custom force constraint bounds with Crocoddyl constraint bounds 
        if(dam.constraints is not None):
            self.g_lb = np.concatenate([dam.g_lb, self.force_g_lb])
            self.g_ub = np.concatenate([dam.g_ub, self.force_g_ub])
        else:
            if(self.forceConstraints is not None):
                self.g_lb = self.force_g_lb
                self.g_ub = self.force_g_ub

    def createData(self):
        data = IADSoftContactDynamics3D_Go2(self)
        return data
    
    def calc(self, data, y, u=None):
        nx = self.differential.state.nx
        nv = self.differential.state.nv
        nq = self.differential.state.nq
        ng_dam = self.ng_dam
        ng_f   = self.ng_f
        nf = self.nf
        x = y[:nx]
        f = y[-nf:]
        v = x[-nv:]
        if(u is not None):
            self.differential.calc(data.differential, x, f, u) 
            a = data.differential.xout
            fdot = data.differential.fout
            data.dx[:nv] = v*self.dt + a*self.dt**2
            data.dx[nv:2*nv] = a*self.dt
            data.dx[2*nv:] = fdot*self.dt
            data.xnext = self.stateSoft.integrate(y, data.dx)
            data.cost = self.dt*data.differential.cost
            data.g[:ng_dam] = data.differential.g[:ng_dam]
            # Compute cost residual
            if(self.withCostResidual):
                data.r = data.differential.r
            # print("g before = ", data.g)
            # Compute force constraint residual
            if(self.with_force_constraint):
                # print("f residual = ", self.forceConstraints.calc(f))
                data.g[ng_dam: ng_dam+ng_f] = self.forceConstraints.calc(f)
            # print("g after = ", data.g)
        else:
            self.differential.calc(data.differential, x, f) 
            data.dx = np.zeros(data.dx.shape)
            data.xnext = y
            data.cost = data.differential.cost
            data.g[:ng_dam] = data.differential.g[:ng_dam]
            # Compute cost residual
            if(self.withCostResidual):
                data.r = data.differential.r
            if(self.with_force_constraint):
                data.g[ng_dam: ng_dam+ng_f] = self.forceConstraints.calc(f)

    def calcDiff(self, data, y, u=None):
        nx = self.differential.state.nx
        ndx = self.differential.state.ndx
        nv = self.differential.state.nv
        nu = self.differential.nu
        nf = self.nf
        ng_dam = self.ng_dam
        ng_f   = self.ng_f
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
            data.Fx[:nv, ndx:] = data.differential.dABA_df * self.dt**2
            data.Fx[nv:ndx, ndx:] = data.differential.dABA_df * self.dt
            # New block from augmented dynamics (bottom right corner)
            data.Fx[ndx:,ndx:] = np.eye(nf) + data.differential.dfdt_df*self.dt
            # New block from augmented dynamics (bottom left corner)
            data.Fx[ndx:, :ndx] = data.differential.dfdt_dx * self.dt
            data.Fu[ndx:, :] = data.differential.dfdt_du * self.dt
            self.stateSoft.JintegrateTransport(y, data.dx, data.Fx, crocoddyl.Jcomponent.second)
            data.Fx += self.stateSoft.Jintegrate(y, data.dx, crocoddyl.Jcomponent.first).tolist()[0]  # add identity to Fx = d(x+dx)/dx = d(q,v)/d(q,v)
            data.Fx[ndx:, ndx:] -= np.eye(nf) # remove identity from Ftau (due to stateSoft.Jintegrate)
            self.stateSoft.JintegrateTransport(y, data.dx, data.Fu, crocoddyl.Jcomponent.second)
            data.Lx[:ndx] = data.differential.Lx*self.dt
            if(self.differential.nr_f > 0):
                data.Lx[ndx:] = data.differential.Lf*self.dt
                data.Lxx[ndx:,ndx:] = data.differential.Lff*self.dt
            data.Lxx[:ndx,:ndx] = data.differential.Lxx*self.dt
            data.Lxu[:ndx, :nu] = data.differential.Lxu*self.dt
            data.Lu = data.differential.Lu*self.dt
            data.Luu = data.differential.Luu*self.dt
            if(ng_dam>0): # otherwise dimension issue 
                data.Gx[0:ng_dam, 0:ndx] = data.differential.Gx
                data.Gu[0:ng_dam, 0:nu] = data.differential.Gu
            # print("Gx before = ", data.Gx)
            if(self.with_force_constraint):
                # print("Gf = ", self.forceConstraints.calcDiff(f))
                if(len(data.Gx.shape) == 1):
                    data.Gx[ndx:ndx+self.nc_f] = self.forceConstraints.calcDiff(f)
                else:
                    data.Gx[ng_dam:ng_dam+ng_f, ndx:ndx+self.nc_f] = self.forceConstraints.calcDiff(f)
            # print("Gx after = ", data.Gx)
        else:
            # Calc forward dyn derivatives
            self.differential.calcDiff(data.differential, x, f)
            data.Fx += self.stateSoft.Jintegrate(y, data.dx, crocoddyl.Jcomponent.first).tolist()[0]  # add identity to Fx = d(x+dx)/dx = d(q,v)/d(q,v)
            data.Lx[:ndx] = data.differential.Lx
            data.Lxx[:ndx,:ndx] = data.differential.Lxx
            if(self.differential.nr_f > 0):
                data.Lx[ndx:] = data.differential.Lf
                data.Lxx[ndx:,ndx:] = data.differential.Lff
            if(ng_dam>0): # otherwise dimension issue
                data.Gx[0:ng_dam, 0:ndx] = data.differential.Gx
            if(self.with_force_constraint):
                if(len(data.Gx.shape) == 1):
                    data.Gx[ndx:ndx+self.nc_f] = self.forceConstraints.calcDiff(f)
                else:
                    data.Gx[ng_dam:ng_dam+ng_f, ndx:ndx+self.nc_f] = self.forceConstraints.calcDiff(f)

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
        self.Lx = np.zeros(am.ndy)
        self.Lu = np.zeros(am.differential.actuation.nu)
        self.Lxx = np.zeros((am.ndy, am.ndy))
        self.Lxu = np.zeros((am.ndy, am.differential.actuation.nu))
        self.Luu = np.zeros((am.differential.actuation.nu, am.differential.actuation.nu))
        # Constraints
        self.Gx = np.zeros((am.ng, am.ndy))
        self.Gu = np.zeros((am.ng, am.nu))
        # self.residual_df = np.zeros((am.nc, am.nf))



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
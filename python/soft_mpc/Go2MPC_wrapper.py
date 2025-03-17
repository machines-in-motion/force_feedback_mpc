
'''
Adapted from Rooholla's code in Go2Py examples
https://github.com/machines-in-motion/Go2Py/blob/mpc/examples/standard_mpc.py
'''

import numpy as np
import os
import mim_solvers
import pinocchio as pin
import crocoddyl
import pinocchio

from soft_multicontact_api import ViscoElasticContact3d_Multiple, ViscoElasticContact3D
from soft_multicontact_api import FrictionConeConstraint, ForceConstraintManager
from soft_multicontact_api import ForceCost, ForceCostManager
from soft_multicontact_api import DAMSoftContactDynamics3D_Go2, IAMSoftContactDynamics3D_Go2

import mujoco as mj


def getForceSensor(model, data, site_id):
    site_id = mj.mj_name2id(model,mj.mjtObj.mjOBJ_SITE, 'EF_force_site')
    world_R_sensor = data.xmat[site_id].reshape(3,3).T
    force_in_body = data.sensordata[-3:].reshape(3,1)
    force_in_world = world_R_sensor@force_in_body
    return force_in_world


class Go2MPC:
    def __init__(self, assets_path, HORIZON=250, friction_mu = 0.2, dt = 0.02):
        self.assets_path = assets_path
        self.HORIZON = HORIZON
        self.max_iterations = 500
        self.dt = dt
        self.urdf_path = os.path.join(assets_path, 'urdf/go2_with_arm.urdf')
        self.xml_path = os.path.join(assets_path, 'mujoco/go2_with_arm.xml')
        self.pin_robot = pin.RobotWrapper.BuildFromURDF(self.urdf_path, self.assets_path, pin.JointModelFreeFlyer())
        self.pinRef = pin.LOCAL_WORLD_ALIGNED
        self.friction_mu = friction_mu 
        self.rmodel = self.pin_robot.model
        self.rdata = self.pin_robot.data

        self.mpc_to_unitree_name_map = \
        {'FL_HAA_joint':'FL_hip_joint',
         'FL_HFE_joint':'FL_thigh_joint',
         'FL_KFE_joint':'FL_calf_joint',
         'FR_HAA_joint':'FR_hip_joint',
         'FR_HFE_joint':'FR_thigh_joint',
         'FR_KFE_joint':'FR_calf_joint',
         'HL_HAA_joint':'RL_hip_joint',
         'HL_HFE_joint':'RL_thigh_joint',
         'HL_KFE_joint':'RL_calf_joint',
         'HR_HAA_joint':'RR_hip_joint',
         'HR_HFE_joint': 'RR_thigh_joint',
         'HR_KFE_joint': 'RR_calf_joint',
         'Joint1':'Joint1',
         'Joint2':'Joint2',
         'Joint3':'Joint3',
         'Joint4':'Joint4',
         'Joint5':'Joint5',
         'Joint6':'Joint6'
         }
        self.unitree_to_mpc_name_map = {val:key for key, val in self.mpc_to_unitree_name_map.items()}
        self.unitree_joint_order = ['FR', 'FL', 'RR', 'RL']

        mpc_to_unitree_idx_map = {}
        unitree_id = 0
        for foot_name in self.unitree_joint_order:
            for actuator in ['_hip_joint', '_thigh_joint', '_calf_joint']:
                unitree_actuator_name = foot_name+actuator
                mpc_joint_name = self.unitree_to_mpc_name_map[unitree_actuator_name]
                mpc_joint_id = self.rmodel.getJointId(mpc_joint_name)
                mpc_to_unitree_idx_map[mpc_joint_id-2]=unitree_id
                unitree_id+=1

        for unitree_actuator_name in ['Joint1', 'Joint2', 'Joint3', 'Joint4', 'Joint5', 'Joint6']:
                mpc_joint_name = self.unitree_to_mpc_name_map[unitree_actuator_name]
                mpc_joint_id = self.rmodel.getJointId(mpc_joint_name)
                mpc_to_unitree_idx_map[mpc_joint_id-2]=unitree_id
                unitree_id+=1

        self.mpc_to_unitree_idx = np.zeros(18).astype(np.int32)    # mpc_state[mpc_to_unitree_idx] -> state/command in unitree order 
        self.unitree_to_mpc_idx = np.zeros(18).astype(np.int32)        # unitree_state[unitree_to_mpc] -> state/command in mpc order
        for mpc_idx, unitree_idx in mpc_to_unitree_idx_map.items():
            self.mpc_to_unitree_idx[mpc_idx] = unitree_idx
            self.unitree_to_mpc_idx[unitree_idx] = mpc_idx

        # set contact frame_names and_indices
        ee_frame_names = ['FL_FOOT', 'FR_FOOT', 'HL_FOOT', 'HR_FOOT', 'Link6']
        self.lfFootId = self.rmodel.getFrameId(ee_frame_names[0])
        self.rfFootId = self.rmodel.getFrameId(ee_frame_names[1])
        self.lhFootId = self.rmodel.getFrameId(ee_frame_names[2])
        self.rhFootId = self.rmodel.getFrameId(ee_frame_names[3])
        self.armEEId = self.rmodel.getFrameId(ee_frame_names[4])
        self.running_models = []
        self.constraintModels = []
        
        self.ccdyl_state = crocoddyl.StateMultibody(self.rmodel)
        self.ccdyl_actuation = crocoddyl.ActuationModelFloatingBase(self.ccdyl_state)
        self.nu = self.ccdyl_actuation.nu

    def initialize(self, q0=np.array([0.0, 0.0, 0.33, 0.0, 0.0, 0.0, 1.0] 
                    +4*[0.0, 0.77832842, -1.56065452] + [0.0, 0.3, -0.3, 0.0, 0.0, 0.0]
                        ), f0=np.zeros(3*5), Kp=1000, Kv=100):
        q0[11+2]=0.0
        self.q0 = q0.copy()
        self.Kp = Kp
        self.f0 = f0.copy()
        self.Kv = Kv
        self.x0 =  np.concatenate([q0, np.zeros(self.rmodel.nv)])
        self.y0 =  np.concatenate([self.x0, f0])
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        self.rfFootPos0 = self.rdata.oMf[self.rfFootId].translation
        self.rhFootPos0 = self.rdata.oMf[self.rhFootId].translation
        self.lfFootPos0 = self.rdata.oMf[self.lfFootId].translation
        self.lhFootPos0 = self.rdata.oMf[self.lhFootId].translation 
        self.armEEPos0 = self.rdata.oMf[self.armEEId].translation
        self.armEEOri0 = self.rdata.oMf[self.armEEId].rotation
        self.supportFeetIds = [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId]
        # self.supportFeePos = [self.lfFootPos0, self.rfFootPos0, self.lhFootPos0, self.rhFootPos0]
        self.xs = [self.y0]*(self.HORIZON + 1)
        self.createProblem()
        self.createSolver()
        self.us = [np.zeros(self.nu)]*self.HORIZON
        # self.us = self.solver.problem.quasiStatic([self.y0]*self.HORIZON) 

    def createProblem(self):
        #First compute the desired state of the robot
        comRef = (self.rfFootPos0 + self.rhFootPos0 + self.lfFootPos0 + self.lhFootPos0) / 4
        comRef[2] = pinocchio.centerOfMass(self.rmodel, self.rdata, self.q0)[2].item() 
        eeDes = self.armEEPos0
        comDes = []
        dt = self.dt
        # radius = 0.065
        radius = 0.0
        for t in range(self.HORIZON+1):
            comDes_t = comRef.copy()
            w = (2 * np.pi) * 0.2 # / T
            comDes_t[0] += radius * np.sin(w * t * dt) 
            comDes_t[1] += radius * (np.cos(w * t * dt) - 1)
            comDes += [comDes_t]
        body_com_ref = comDes
        arm_eff_ref = [eeDes]*(self.HORIZON+1)
        self.comDes = comDes

        # Now define the model
        for t in range(self.HORIZON+1):
            costModel = crocoddyl.CostModelSum(self.ccdyl_state, self.nu)

            # Add state/control reg costs
            state_reg_weight, control_reg_weight = 1e-1, 1e-3
            freeFlyerQWeight = [0.]*3 + [500.]*3
            freeFlyerVWeight = [10.]*6
            legsQWeight = [0.01]*(self.rmodel.nv - 6)
            legsWWeights = [1.]*(self.rmodel.nv - 6)
            stateWeights = np.array(freeFlyerQWeight + legsQWeight + freeFlyerVWeight + legsWWeights)    
            stateResidual = crocoddyl.ResidualModelState(self.ccdyl_state, self.x0, self.nu)
            stateActivation = crocoddyl.ActivationModelWeightedQuad(stateWeights**2)
            stateReg = crocoddyl.CostModelResidual(self.ccdyl_state, stateActivation, stateResidual)
            costModel.addCost("stateReg", stateReg, state_reg_weight)
            ctrlResidual = crocoddyl.ResidualModelControl(self.ccdyl_state, self.nu)
            ctrlReg = crocoddyl.CostModelResidual(self.ccdyl_state, ctrlResidual)
            costModel.addCost("ctrlReg", ctrlReg, control_reg_weight)  

            # Add COM task
            com_residual = crocoddyl.ResidualModelCoMPosition(self.ccdyl_state, comDes[t], self.nu)
            com_activation = crocoddyl.ActivationModelWeightedQuad(np.array([1., 1., 1.]))
            com_track = crocoddyl.CostModelResidual(self.ccdyl_state, com_activation, com_residual) # What does it correspond to exactly?
            costModel.addCost("comTrack", com_track, 1e5)

            # # End Effecor Position Task
            # ef_residual = crocoddyl.ResidualModelFrameTranslation(self.ccdyl_state, self.armEEId, self.armEEId, self.nu)
            # ef_activation = crocoddyl.ActivationModelWeightedQuad(np.array([1., 1., 1.]))
            # ef_track = crocoddyl.CostModelResidual(self.ccdyl_state, ef_activation, ef_residual)
            # costModel.addCost("efTrack", ef_track, 1e5)

            # Soft contact models 3d 
            lf_contact = ViscoElasticContact3D(self.ccdyl_state, self.ccdyl_actuation, self.lfFootId, self.rdata.oMf[self.lfFootId].translation, self.Kp, self.Kv, self.pinRef)
            rf_contact = ViscoElasticContact3D(self.ccdyl_state, self.ccdyl_actuation, self.rfFootId, self.rdata.oMf[self.rfFootId].translation, self.Kp, self.Kv, self.pinRef)
            lh_contact = ViscoElasticContact3D(self.ccdyl_state, self.ccdyl_actuation, self.lhFootId, self.rdata.oMf[self.lhFootId].translation, self.Kp, self.Kv, self.pinRef)
            rh_contact = ViscoElasticContact3D(self.ccdyl_state, self.ccdyl_actuation, self.rhFootId, self.rdata.oMf[self.rhFootId].translation, self.Kp, self.Kv, self.pinRef)
            ef_contact = ViscoElasticContact3D(self.ccdyl_state, self.ccdyl_actuation, self.armEEId, self.rdata.oMf[self.armEEId].translation, self.Kp, self.Kv, self.pinRef)
            # Stack models
            softContactModelsStack = ViscoElasticContact3d_Multiple(self.ccdyl_state, self.ccdyl_actuation, [lf_contact, rf_contact, lh_contact, rh_contact, ef_contact])

            # Constraints stack
            constraintModelManager = None # crocoddyl.ConstraintModelManager(self.ccdyl_state, actuation.self.self.nu)

            # Custom force cost in DAM
            forceCostManager = ForceCostManager([ForceCost(self.ccdyl_state, self.armEEId, np.array([0, 0., -10.]), 0.1, pin.LOCAL_WORLD_ALIGNED)], softContactModelsStack)

            # Create DAM with soft contact models, force costs + standard cost & constraints
            dam = DAMSoftContactDynamics3D_Go2(self.ccdyl_state, self.ccdyl_actuation, costModel, softContactModelsStack, constraintModelManager, forceCostManager)

            # Friction cone constraint models
            forceConstraintManager = ForceConstraintManager([FrictionConeConstraint(self.lfFootId, self.friction_mu),
                                                                FrictionConeConstraint(self.rfFootId, self.friction_mu),
                                                                FrictionConeConstraint(self.lhFootId, self.friction_mu),
                                                                FrictionConeConstraint(self.rhFootId, self.friction_mu)], 
                                                                softContactModelsStack)

            iam = IAMSoftContactDynamics3D_Go2(dam, dt=dt, withCostResidual=True, forceConstraintManager=forceConstraintManager)
            self.running_models += [iam]

        self.ocp = crocoddyl.ShootingProblem(self.y0, self.running_models[:-1], self.running_models[-1])
        
    def createSolver(self):
        solver = mim_solvers.SolverCSQP(self.ocp)
        solver.setCallbacks([mim_solvers.CallbackVerbose(), mim_solvers.CallbackLogger()])
        
        # solver.max_qp_iters = 75
        # solver.with_callbacks = True
        # solver.use_filter_line_search = False
        # solver.termination_tolerance = 1e-2
        # solver.eps_abs = 1e-6
        # solver.eps_rel = 0.

        solver.max_qp_iters = 1000
        solver.with_callbacks = True
        solver.use_filter_line_search = False
        solver.termination_tolerance = 1e-4
        solver.eps_abs = 1e-6
        solver.eps_rel = 1e-6

        self.solver = solver

    def getSolution(self, k=None):
        if k is None: 
            x_idx = 1
            u_idx = 0
        else:
            x_idx = k
            u_idx = k
        t = self.xs[x_idx][:3]
        quat = self.xs[x_idx][3:7]
        qx = quat[0]
        qy = quat[1]
        qz = quat[2]
        qw = quat[3]
        q = self.xs[x_idx][7:25]
        eta = self.xs[x_idx][25:25+6]
        dq = self.xs[x_idx][25+6:25+6+18]
        f = self.xs[x_idx][25+6+18:]
        constraint_norm = self.solver.constraint_norm
        return dict(
            position=t,
            orientation=np.array([qw, qx, qy, qz]), #Mujoco and uniree quaternion order
            velocity = eta[:3],
            omega = eta[3:],
            q = q[self.mpc_to_unitree_idx],
            dq = dq[self.mpc_to_unitree_idx], 
            f_rf = f[:3],
            f_rh = f[3:6],
            f_lf = f[6:9],
            f_lh = f[9:12],
            f_ee = f[12:],
            tau = self.us[u_idx][[self.mpc_to_unitree_idx]],
            constraint_norm = constraint_norm
        )
    
    def updateAndSolve(self, t, quat, q, v, omega, dq, f):
        q_ = np.zeros(self.rmodel.nq)
        dq_ = np.zeros(self.rmodel.nv)
        qw = quat[0]
        qx = quat[1]
        qy = quat[2]
        qz = quat[3]
        q_[:3] = t
        q_[3:7] = np.array([qx, qy, qz, qw])
        q_[7:] = q[self.unitree_to_mpc_idx]
        dq_[:3] = v
        dq_[3:6] = omega
        dq_[6:] = dq[self.unitree_to_mpc_idx]
        pin.framesForwardKinematics(self.rmodel, self.rdata, q_)
        y = np.hstack([q_, dq_, f])
        self.solver.problem.x0 = y
        self.xs = list(self.solver.xs[1:]) + [self.solver.xs[-1]]
        self.xs[0] = y
        self.us = list(self.us[1:]) + [self.us[-1]] 
        self.solver.solve(self.xs, self.us, self.max_iterations)
        self.xs, self.us = self.solver.xs, self.solver.us
        return self.getSolution()
    
    def solve(self):
        self.solver.solve(self.xs, self.us, self.max_iterations)
        self.xs, self.us = self.solver.xs, self.solver.us
        return self.getSolution()
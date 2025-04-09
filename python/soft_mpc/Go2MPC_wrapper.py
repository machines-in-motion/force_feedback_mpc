
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
from soft_multicontact_api import FrictionConeConstraint, ForceBoxConstraint, ForceConstraintManager
from soft_multicontact_api import ForceCost, ForceCostManager, ForceRateCostManager
from soft_multicontact_api import DAMSoftContactDynamics3D_Go2, IAMSoftContactDynamics3D_Go2

import mujoco as mj


# Add this to your validation code
def test_force_sensor_orientation(robot):
    """Apply known forces and verify sensor readings"""
    # Reset simulation
    robot.reset()
    # Apply a known force in world coordinates (e.g., +Z)
    test_force = np.array([0, 0, 10])  # 10N upward
    robot.data.xfrc_applied[robot.data.body("base").id] = [0, 0, 0, test_force[0], test_force[1], test_force[2]]
    # Step simulation
    mj.mj_step(robot.model, robot.data)    
    # Get sensor reading
    measured = getForceSensor(robot.model, robot.data, "FL_force_site")
    print("Applied force:", test_force)
    print("Measured force:", measured.squeeze())


def getForceSensor(model, data, site_name):
    """Returns force in world coordinates"""
    # Map site names to sensor names (as defined in XML)
    site_to_sensor = {
        'FL_force_site': 'FL_force',
        'FR_force_site': 'FR_force',
        'RL_force_site': 'RL_force',
        'RR_force_site': 'RR_force',
        'EF_force_site': 'EF_force'
    }
    # Get the correct sensor name
    sensor_name = site_to_sensor[site_name]
    # print("SENSOR NAME = \n", sensor_name)
    force_site_to_sensor_idx = {'FL_force_site': 10, 'FR_force_site': 13, 'RL_force_site': 16, 'RR_force_site': 19, 'EF_force_site': 22}
    sensor_id = force_site_to_sensor_idx[site_name] #mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    # print("SENSOR Id = \n", sensor_id)
    available_sensors = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_SENSOR, i) 
                        for i in range(model.nsensor)]
    # print("available sensors = \n", available_sensors)
    if sensor_id == -1:
        available_sensors = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_SENSOR, i) 
                           for i in range(model.nsensor)]
        raise ValueError(f"Sensor '{sensor_name}' not found. Available sensors: {available_sensors}")
    # Get site transform
    site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, site_name)
    site_rot = data.site_xmat[site_id].reshape(3, 3)
    # MuJoCo force sensors report force in site frame
    force_in_site = data.sensordata[sensor_id:sensor_id+3]
    # Transform to world frame
    force_in_world = site_rot @ force_in_site
    # print(" SENSOR force = \n", force_in_world)
    return force_in_world.reshape(3, 1)


def setGroundFriction(model, data, mu):
    ground_geom_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "floor")
    if ground_geom_id == -1:
        raise ValueError("Ground geometry not found. Check your XML model for the correct geom name.")
    print("Friction found for ground = ", model.geom_friction[ground_geom_id])
    model.geom_friction[ground_geom_id][0] = mu
    print("Set ground friction to : ", model.geom_friction[ground_geom_id])



def plot_ocp_solution_with_cones(mpc):

    # required to extract and plot solution
    rmodel = mpc.rmodel
    rdata = mpc.rdata
    solver = mpc.solver
    xs = mpc.solver.xs 
    supportFeetIds = mpc.supportFeetIds
    T = mpc.HORIZON
    MU = mpc.friction_mu
    # comDes = mpc.comDes

    # Extract OCP Solution 
    nq, nv, N = rmodel.nq, rmodel.nv, len(xs) 
    jointPos_sol = []
    jointVel_sol = []
    jointAcc_sol = []
    jointTorques_sol = []
    centroidal_sol = []
    force_sol = []
    xs, us = solver.xs, solver.us
    x = []
    for time_idx in range (T):
        q, v = xs[time_idx][:nq], xs[time_idx][nq:nq+nv]
        f = xs[time_idx][nq+nv:]
        pin.framesForwardKinematics(rmodel, rdata, q)
        pin.computeCentroidalMomentum(rmodel, rdata, q, v)
        centroidal_sol += [
            np.concatenate(
                [pin.centerOfMass(rmodel, rdata, q, v), np.array(rdata.hg.linear), np.array(rdata.hg.angular)]
                )
                ]
        jointPos_sol += [q]
        jointVel_sol += [v]
        force_sol    += [f]
        x += [xs[time_idx]]
        if time_idx < T-1:
            jointAcc_sol +=  [solver.problem.runningDatas[time_idx].xnext[nq::]] 
            jointTorques_sol += [us[time_idx]]

    sol = {'x':x, 'centroidal':centroidal_sol, 'jointPos':jointPos_sol, 
                        'jointVel':jointVel_sol, 'jointAcc':jointAcc_sol, 'force':force_sol,
                        'jointTorques':jointTorques_sol}       

    # Extract contact forces by hand
    sol['FL_FOOT_contact'] = [force_sol[i][0:3] for i in range(T)]     
    sol['FR_FOOT_contact'] = [force_sol[i][3:6] for i in range(T)]     
    sol['HL_FOOT_contact'] = [force_sol[i][6:9] for i in range(T)]     
    sol['HR_FOOT_contact'] = [force_sol[i][9:12] for i in range(T)]     
    sol['Link6'] = [force_sol[i][-3:] for i in range(T)]     

    # Plotting 
    import matplotlib.pyplot as plt
    constrained_sol = sol
    time_lin = np.linspace(0, T, T)
    fig, axs = plt.subplots(4, 3, constrained_layout=True)
    for i, frame_idx in enumerate(supportFeetIds):
        ct_frame_name = rmodel.frames[frame_idx].name + "_contact"
        forces = np.array(constrained_sol[ct_frame_name])
        axs[i, 0].plot(time_lin, forces[:, 0], label="Fx")
        axs[i, 1].plot(time_lin, forces[:, 1], label="Fy")
        axs[i, 2].plot(time_lin, forces[:, 2], label="Fz")
        # Add friction cone constraints 
        Fz_lb = (1./MU)*np.sqrt(forces[:, 0]**2 + forces[:, 1]**2)
        # Fz_ub = np.zeros(time_lin.shape)
        # axs[i, 2].plot(time_lin, Fz_ub, 'k-.', label='ub')
        axs[i, 2].plot(time_lin, Fz_lb, 'k-.', label='lb')
        axs[i, 0].grid()
        axs[i, 1].grid()
        axs[i, 2].grid()
        axs[i, 0].set_ylabel(ct_frame_name)
    axs[0, 0].legend()
    axs[0, 1].legend()
    axs[0, 2].legend()

    axs[3, 0].set_xlabel(r"$F_x$")
    axs[3, 1].set_xlabel(r"$F_y$")
    axs[3, 2].set_xlabel(r"$F_z$")
    fig.suptitle('Force feet', fontsize=16)

    fig, axs = plt.subplots(3, 1, constrained_layout=True)
    forces = np.array(constrained_sol['Link6'])
    axs[0].plot(time_lin, forces[:, 0], label="Fx")
    axs[1].plot(time_lin, forces[:, 1], label="Fy")
    axs[2].plot(time_lin, forces[:, 2], label="Fz")
    for i in range(3):
        axs[i].legend()
        axs[i].grid()
    fig.suptitle('Force EE', fontsize=16)

    axs[0].set_xlabel(r"$F_x$")
    axs[1].set_xlabel(r"$F_y$")
    axs[2].set_xlabel(r"$F_z$")

    # fig, axs = plt.subplots(nq, 1, constrained_layout=True)
    # for i in range(nq):
    #     jointPos_sol = np.array(jointPos_sol)
    #     axs[i].plot(time_lin, jointPos_sol[:, i], marker='.', linewidth=2, label="q_"+str(i))
    #     axs[i].grid()
    #     axs[i].legend()
    # fig.suptitle('Joint positions', fontsize=16)

    fig, axs = plt.subplots(nv//2, 2, constrained_layout=True)
    for i in range(nv//2):
        jointVel_sol = np.array(jointVel_sol)
        axs[i, 0].plot(time_lin, [0.]*time_lin.shape[0], color='r', linewidth=2)
        axs[i, 0].plot(time_lin, jointVel_sol[:, 2*i], marker='.', linewidth=2, label="v_"+str(2*i))
        axs[i, 0].grid()
        axs[i, 0].legend()
        axs[i, 1].plot(time_lin, [0.]*time_lin.shape[0], color='r', linewidth=2)
        axs[i, 1].plot(time_lin, jointVel_sol[:, 2*i+1], marker='.', linewidth=2, label="v_"+str(2*i+1))
        axs[i, 1].grid()
        axs[i, 1].legend()
    fig.suptitle('Joint velocities', fontsize=16)

    # fig, axs = plt.subplots(nv-6, 1, constrained_layout=True)
    # for i in range(nv-6):
    #     jointTorques_sol = np.array(jointTorques_sol)
    #     axs[i].plot(time_lin[:-1], jointTorques_sol[:, i], marker='.', linewidth=2, label="tau_"+str(i))
    #     axs[i].grid()
    #     axs[i].legend()
    # fig.suptitle('Joint torques', fontsize=16)


    # comDes = np.array(comDes)
    # centroidal_sol = np.array(constrained_sol['centroidal'])
    # plt.figure()
    # plt.plot(comDes[:, 0], comDes[:, 1], "--", label="reference")
    # plt.plot(centroidal_sol[:, 0], centroidal_sol[:, 1], label="solution")
    # plt.legend()
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.title("COM trajectory")
    plt.show()

class Go2MPC:
    def __init__(self, assets_path, HORIZON=250, friction_mu = 0.75, dt = 0.01):
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
        self.ee_frame_names = ['FL_FOOT', 'FR_FOOT', 'HL_FOOT', 'HR_FOOT', 'Link6']
        self.lfFootId = self.rmodel.getFrameId(self.ee_frame_names[0])
        self.rfFootId = self.rmodel.getFrameId(self.ee_frame_names[1])
        self.lhFootId = self.rmodel.getFrameId(self.ee_frame_names[2])
        self.rhFootId = self.rmodel.getFrameId(self.ee_frame_names[3])
        self.armEEId = self.rmodel.getFrameId(self.ee_frame_names[4])
        self.running_models = []
        self.constraintModels = []
        
        self.ccdyl_state = crocoddyl.StateMultibody(self.rmodel)
        self.ccdyl_actuation = crocoddyl.ActuationModelFloatingBase(self.ccdyl_state)
        self.rmodel.armature = np.zeros(self.ccdyl_state.nv)
        self.nu = self.ccdyl_actuation.nu
        self.ctrlLim = np.array([23.7, 23.7, 45.43, 23.7, 23.7, 45.43, 23.7, 23.7, 45.43, 23.7, 23.7, 45.43, \
                                    30, 30, 30, 30, 10, 10])
        # self.ctrlLim = np.array([23.7, 23.7, 45.43, 23.7, 23.7, 45.43, 23.7, 23.7, 45.43, 23.7, 23.7, 45.43] +
        #                             [np.inf]*6)
        # print(self.ccdyl_state.pinocchio.effortLimit[6:])

    def initialize(self, q0=np.array([-0.01, 0.0, 0.32, 0.0, 0.0, 0.0, 1.0] 
                    +4*[0.0, 0.77832842, -1.56065452] + [0.0, 0.3, -0.3, 0.0, 0.0, 0.0]
                        ), f0=np.array([0., 0., 0.]*4 + [0.]*3), Kp=1000, Kv=100, FREF=15):
        q0[11+2]=0.0
        self.q0 = q0.copy()
        self.v0 = np.zeros(self.rmodel.nv)
        self.Kp = Kp
        self.f0 = f0.copy()
        # np.array([-0.083766 , 0.003337, -0.051429, 
        #                     -0.063791 ,-0.002004, -0.03834,
        #                     -0.05787  ,-0.051013,  0.103739 ,
        #                     -0.053733 , 0.045646,  0.10088  ,
        #                     -41.543034,  -7.066572,  -6.30816    ]) #f0.copy()
        self.Fx_ref_ee = FREF
        self.Kv = Kv
        self.x0 =  np.concatenate([q0, self.v0])
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        self.rfFootPos0 = self.rdata.oMf[self.rfFootId].translation
        self.rhFootPos0 = self.rdata.oMf[self.rhFootId].translation
        self.lfFootPos0 = self.rdata.oMf[self.lfFootId].translation
        self.lhFootPos0 = self.rdata.oMf[self.lhFootId].translation 
        self.footPosDict = {'FL_FOOT': self.lfFootPos0,
                            'FR_FOOT': self.rfFootPos0,
                            'HL_FOOT': self.lhFootPos0,
                            'HR_FOOT': self.rhFootPos0}
        self.armEEPos0 = self.rdata.oMf[self.armEEId].translation
        self.armEEOri0 = self.rdata.oMf[self.armEEId].rotation
        self.oPc_ee = self.armEEPos0.copy()
        self.oPc_ee[0] += 0.02
        # self.supportFeetIds = [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId]
        # self.f0[-3:] = -np.diag([self.Kp]*3) @ (self.armEEPos0 - self.oPc_ee)  
        # print("\n\n f0 = ", self.f0[-3:], "\n\n")
        # print("\n\n p_ee = ", self.armEEPos0, "\n\n")
        # print("\n\n oPc = ", self.oPc_ee, "\n\n")
        self.y0 =  np.concatenate([self.x0, self.f0])
        self.xs = [self.y0]*(self.HORIZON + 1)
        self.createProblem()
        self.createSolver()
        self.u0 = np.zeros(self.nu)
        self.us = [self.u0]*self.HORIZON
        # self.us0 = np.array([-1.70292494e+00 , -2.53493996e-01,   4.00765770e+00 , 1.69757863e+00,
        #                     -2.53702321e-01 ,  4.00563075e+00,  -2.07526085e+00 ,-2.77332371e-01,
        #                     4.63287338e+00  , 2.06914143e+00,  -2.77541101e-01  ,4.63085705e+00,
        #                     -6.68587279e-04 ,  5.55902596e+00,   6.17453082e+00 ,-3.00315497e-03,
        #                     1.96635590e+00  , 3.09288758e-06]) 
        # self.us = [self.us0 for i in range(self.HORIZON)]
        # self.solver.problem.quasiStatic([self.y0]*self.HORIZON) 
        # print("wepifuwefweifo\n\n\n", self.us[0])
        

    def createProblem(self):
        #First compute the desired state of the robot
        comRef = (self.rfFootPos0 + self.rhFootPos0 + self.lfFootPos0 + self.lhFootPos0) / 4
        comRef[2] = pinocchio.centerOfMass(self.rmodel, self.rdata, self.q0)[2].item() 

        # Now define the model
        for t in range(self.HORIZON+1):
            costModel = crocoddyl.CostModelSum(self.ccdyl_state, self.nu)

            # Add state/control reg costs
            state_reg_weight, control_reg_weight = 1e-1, 1e-3
            freeFlyerQWeight = [0.]*3 + [0.]*3
            freeFlyerVWeight = [10.]*6
            legsQWeight = [0.01]*(self.rmodel.nv - 6)
            legsWWeights = [1.]*(self.rmodel.nv - 6)
            stateWeights = np.array(freeFlyerQWeight + legsQWeight + freeFlyerVWeight + legsWWeights)    
            stateResidual = crocoddyl.ResidualModelState(self.ccdyl_state, self.x0, self.nu)

            stateActivation = crocoddyl.ActivationModelWeightedQuad(stateWeights**2)
            stateReg = crocoddyl.CostModelResidual(self.ccdyl_state, stateActivation, stateResidual)

            if t == self.HORIZON:
                costModel.addCost("stateReg", stateReg, state_reg_weight*self.dt)
            else:
                costModel.addCost("stateReg", stateReg, state_reg_weight)
                
            if t != self.HORIZON:
                ctrlResidual = crocoddyl.ResidualModelControl(self.ccdyl_state, self.nu)
                ctrlReg = crocoddyl.CostModelResidual(self.ccdyl_state, ctrlResidual)
                costModel.addCost("ctrlReg", ctrlReg, control_reg_weight)  

            # Body COM Tracking Cost
            # com_residual = crocoddyl.ResidualModelCoMPosition(self.ccdyl_state, comRef, self.nu)
            # com_activation = crocoddyl.ActivationModelWeightedQuad(np.array([1., 1., 1.]))
            # com_track = crocoddyl.CostModelResidual(self.ccdyl_state, com_activation, com_residual) #
            # costModel.addCost("comTrack", com_track, 1e-4)

            # End Effecor Position Tracking Cost
            # ef_pos_ref = self.oPc_ee
            # ef_residual = crocoddyl.ResidualModelFrameTranslation(self.ccdyl_state, self.armEEId, ef_pos_ref, self.nu) # Check this cost term            
            # ef_activation = crocoddyl.ActivationModelWeightedQuad(np.array([1., 1., 1.]))
            # ef_track = crocoddyl.CostModelResidual(self.ccdyl_state, ef_activation, ef_residual)
            # # if t != self.HORIZON:
            # costModel.addCost("ef_track", ef_track, 1)
            # else:
            #     costModel.addCost("ef_track", ef_track, 1e1*self.dt)
            # feet tracking costs
            # for fname in self.ee_frame_names[:-1]:
            #     frame_idx = self.rmodel.getFrameId(fname)
            #     foot_residual = crocoddyl.ResidualModelFrameTranslation(self.ccdyl_state, frame_idx, self.footPosDict[fname], self.nu) # Check this cost term            
            #     foot_activation = crocoddyl.ActivationModelWeightedQuad(np.array([1., 1., 1.]))
            #     foot_track = crocoddyl.CostModelResidual(self.ccdyl_state, foot_activation, foot_residual)
            #     costModel.addCost(fname+"_track", foot_track, 1e-3)

            # Soft contact models 3d 
            # oPc_ee = self.rdata.oMf[self.armEEId].translation.copy() 
            lf_contact = ViscoElasticContact3D(self.ccdyl_state, self.ccdyl_actuation, self.lfFootId, self.rdata.oMf[self.lfFootId].translation.copy(), self.Kp, self.Kv, self.pinRef)
            rf_contact = ViscoElasticContact3D(self.ccdyl_state, self.ccdyl_actuation, self.rfFootId, self.rdata.oMf[self.rfFootId].translation.copy(), self.Kp, self.Kv, self.pinRef)
            lh_contact = ViscoElasticContact3D(self.ccdyl_state, self.ccdyl_actuation, self.lhFootId, self.rdata.oMf[self.lhFootId].translation.copy(), self.Kp, self.Kv, self.pinRef)
            rh_contact = ViscoElasticContact3D(self.ccdyl_state, self.ccdyl_actuation, self.rhFootId, self.rdata.oMf[self.rhFootId].translation.copy(), self.Kp, self.Kv, self.pinRef)
            ef_contact = ViscoElasticContact3D(self.ccdyl_state, self.ccdyl_actuation, self.armEEId, self.oPc_ee, self.Kp, self.Kv, self.pinRef)
            # Stack models
            softContactModelsStack = ViscoElasticContact3d_Multiple(self.ccdyl_state, self.ccdyl_actuation, [lf_contact, rf_contact, lh_contact, rh_contact, ef_contact])

            # Constraints stack
            constraintModelManager = None #crocoddyl.ConstraintModelManager(self.ccdyl_state, self.nu)

            # Custom force cost in DAM
            f_weight = np.array([1e-1, 1e-6, 1e-6])
            fdot_weights = np.array([1e-3]*12 + [1., 1., 1.])*1e-5
            if t != self.HORIZON:
                forceCostEE = ForceCost(self.ccdyl_state, self.armEEId, np.array([-self.Fx_ref_ee, 0., 0.]), f_weight, pin.LOCAL_WORLD_ALIGNED)
                forceRateCostManager = ForceRateCostManager(self.ccdyl_state, self.ccdyl_actuation, softContactModelsStack, fdot_weights)
            else:
                forceCostEE = ForceCost(self.ccdyl_state, self.armEEId, np.array([-self.Fx_ref_ee, 0., 0.]), f_weight*self.dt, pin.LOCAL_WORLD_ALIGNED)
                forceRateCostManager = ForceRateCostManager(self.ccdyl_state, self.ccdyl_actuation, softContactModelsStack, fdot_weights*self.dt)

            forceCostManager = ForceCostManager([forceCostEE], softContactModelsStack)

            # Create DAM with soft contact models, force costs + standard cost & constraints
            dam = DAMSoftContactDynamics3D_Go2(self.ccdyl_state, 
                                               self.ccdyl_actuation, 
                                               costModel, 
                                               softContactModelsStack, 
                                               constraintModelManager, 
                                               forceCostManager, 
                                               forceRateCostManager)

            # if(t != self.HORIZON):
            #     ctrlResidual2 = crocoddyl.ResidualModelControl(self.ccdyl_state, self.nu)
            #     torque_lb = -self.ctrlLim #self.pin_robot.model.effortLimit
            #     torque_ub = self.ctrlLim #self.pin_robot.model.effortLimit
            #     # print("Ctrl limit lb = \n", torque_lb)
            #     # print("Ctrl limit ub = \n", torque_ub)
            #     torqueBoxConstraint = crocoddyl.ConstraintModelResidual(self.ccdyl_state, ctrlResidual2, torque_lb, torque_ub)
            #     constraintModelManager.addConstraint("ctrlBox", torqueBoxConstraint)

            # Friction cone constraint models
            # lb_foot = np.array([-np.inf, -np.inf, 0.])
            # ub_foot = np.array([np.inf, np.inf, np.inf])
            # lb_ee = np.array([-np.inf, -np.inf, -np.inf])
            # ub_ee = np.array([0., np.inf, np.inf])
            forceConstraintManager = ForceConstraintManager([
                                                            FrictionConeConstraint(self.lfFootId, self.friction_mu),
                                                             FrictionConeConstraint(self.rfFootId, self.friction_mu),
                                                             FrictionConeConstraint(self.lhFootId, self.friction_mu),
                                                             FrictionConeConstraint(self.rhFootId, self.friction_mu),
                                                            #  ForceBoxConstraint(self.lfFootId, lb_foot, ub_foot),
                                                            #  ForceBoxConstraint(self.rfFootId, lb_foot, ub_foot),
                                                            #  ForceBoxConstraint(self.lhFootId, lb_foot, ub_foot),
                                                            #  ForceBoxConstraint(self.rhFootId, lb_foot, ub_foot),
                                                            #  ForceBoxConstraint(self.armEEId, lb_ee, ub_ee),
                                                            ], 
                                                                softContactModelsStack)

            iam = IAMSoftContactDynamics3D_Go2(dam, dt=self.dt, withCostResidual=True, forceConstraintManager=forceConstraintManager)
            self.running_models += [iam]

        self.ocp = crocoddyl.ShootingProblem(self.y0, self.running_models[:-1], self.running_models[-1])


    def test_derivatives(self):

        from numpy.random import rand
        from numpy.linalg import norm 
        TOL = 1e-2
        from test_utils import get_fdot, \
                            get_iam_cost, \
                            get_xdot, \
                            get_ynext_y, \
                            get_dam_cost, \
                            get_cstr
        from test_utils import numdiff, \
                            numdiff_q_dam_dyn, \
                            numdiff_u_iam_cost, \
                            numdiff_u_iam_dyn, \
                            numdiff_x_iam_cost, \
                            numdiff_x_iam_dyn, \
                            numdiff_x_iam_cstr, \
                            numdiff_u_iam_cstr, \
                            numdiff_q_dam_cost, \
                            numdiff_vfu_dam_cost

        y = self.y0
        u = self.u0
        x = self.x0
        q = self.q0
        v = self.v0
        f = self.f0
        rmodel = self.rmodel
        # Compute running IAM derivatives
        IAM = self.running_models[1]
        IAD = IAM.createData()
        IAM.calc(IAD, y, u)
        IAM.calcDiff(IAD, y, u)
        dcost_dy = IAD.Lx
        dcost_du = IAD.Lu
        dynext_dx = IAD.Fx 
        dynext_du = IAD.Fu 
        dcstr_dy = IAD.Gx
        dcstr_du = IAD.Gu

        # Compute terminal IAM derivatives
        IAM_t = self.running_models[-1]
        IAD_t = IAM_t.createData()
        IAM_t.calc(IAD_t, y)
        IAM_t.calcDiff(IAD_t, y)
        dcost_dy_t = IAD_t.Lx
        dynext_dy_t = IAD_t.Fx 
        dcstr_dy_t = IAD_t.Gx

        # Compute running DAM derivatives
        DAM = IAM.differential
        DAD = DAM.createData()
        DAM.calc(DAD, x, f, u)
        DAM.calcDiff(DAD, x, f, u)
        dxdot_dx = DAD.Fx
        dxdot_dq = DAD.Fx[:,:rmodel.nv]
        dxdot_dv = DAD.Fx[:,rmodel.nv:]
        dxdot_df = DAD.dABA_df
        dxdot_du = DAD.Fu
        dfdot_dq = DAD.dfdt_dx[:,:rmodel.nv]
        dfdot_dv = DAD.dfdt_dx[:,rmodel.nv:]
        dfdot_df = DAD.dfdt_df
        dfdot_du = DAD.dfdt_du
        Lq_dad = DAD.Lx[:rmodel.nv]
        Lv_dad = DAD.Lx[rmodel.nv:]
        Lf_dad = DAD.Lf
        Lu_dad = DAD.Lu
        

        # Compute terminal DAM derivatives
        DAM_t = IAM_t.differential
        DAD_t = DAM_t.createData()
        DAM_t.calc(DAD_t, x, f)
        DAM_t.calcDiff(DAD_t, x, f)
        dxdot_dx_t = DAD_t.Fx
        dxdot_dq_t = DAD_t.Fx[:,:rmodel.nv]
        dxdot_dv_t = DAD_t.Fx[:,rmodel.nv:]
        dxdot_df_t = DAD_t.dABA_df
        dfdot_dq_t = DAD_t.dfdt_dx[:,:rmodel.nv]
        dfdot_dv_t = DAD_t.dfdt_dx[:,rmodel.nv:]
        dfdot_df_t = DAD_t.dfdt_df
        Lq_dad_t = DAD_t.Lx[:rmodel.nv]
        Lv_dad_t = DAD_t.Lx[rmodel.nv:]
        Lf_dad_t = DAD_t.Lf

        # Compute running IAM derivatives with NUMDIFF
            # Dyn
        dynext_dx_ND = numdiff_x_iam_dyn(lambda y_:get_ynext_y(IAM, IAD, y_, u), y, IAM.stateSoft)
        dynext_du_ND = numdiff_u_iam_dyn(lambda u_:get_ynext_y(IAM, IAD, y, u_), u, IAM.stateSoft)
            # Cost
        dcost_dy_ND = numdiff_x_iam_cost(lambda y_:get_iam_cost(IAM, IAD, y_, u), y, IAM.stateSoft)
        dcost_du_ND = numdiff_u_iam_cost(lambda u_:get_iam_cost(IAM, IAD, y, u_), u)
            # Cstr
        dcstr_dy_ND = numdiff_x_iam_cstr(lambda y_:get_cstr(IAM, IAD, y_, u), y, IAM.stateSoft)
        dcstr_du_ND = numdiff_u_iam_cstr(lambda u_:get_cstr(IAM, IAD, y, u_), u)
        
        # Compute terminal IAM_t derivatives with NUMDIFF
        dynext_dy_t_ND = numdiff_x_iam_dyn(lambda y_:get_ynext_y(IAM_t, IAD_t, y_), y, IAM_t.stateSoft)
        dcost_dy_t_ND = numdiff_x_iam_cost(lambda y_:get_iam_cost(IAM_t, IAD_t, y_), y, IAM_t.stateSoft)
            # Cstr
        dcstr_dy_t_ND = numdiff_x_iam_cstr(lambda y_:get_cstr(IAM_t, IAD_t, y_, u), y, IAM_t.stateSoft)
        
        # Compute running DAM derivatives with NUMDIFF
        dxdot_dq_ND = numdiff_q_dam_dyn(lambda q_:get_xdot(DAM, DAD, q_, v, f, u), q, rmodel)
        dxdot_dv_ND = numdiff(lambda v_:get_xdot(DAM, DAD, q, v_, f, u), v)
        dxdot_df_ND = numdiff(lambda f_:get_xdot(DAM, DAD, q, v, f_, u), f)
        dxdot_du_ND = numdiff(lambda u_:get_xdot(DAM, DAD, q, v, f, u_), u)
        dfdot_dq_ND = numdiff_q_dam_dyn(lambda q_:get_fdot(DAM, DAD, q_, v, f, u), q, rmodel)
        dfdot_dv_ND = numdiff(lambda v_:get_fdot(DAM, DAD, q, v_, f, u), v)
        dfdot_df_ND = numdiff(lambda f_:get_fdot(DAM, DAD, q, v, f_, u), f)
        dfdot_du_ND = numdiff(lambda u_:get_fdot(DAM, DAD, q, v, f, u_), u)
        Lq_dad_ND = numdiff_q_dam_cost(lambda q_:get_dam_cost(DAM, DAD, q_, v, f, u), q, rmodel)
        Lv_dad_ND = numdiff_vfu_dam_cost(lambda v_:get_dam_cost(DAM, DAD, q, v_, f, u), v)
        Lf_dad_ND = numdiff_vfu_dam_cost(lambda f_:get_dam_cost(DAM, DAD, q, v, f_, u), f)
        Lu_dad_ND = numdiff_vfu_dam_cost(lambda u_:get_dam_cost(DAM, DAD, q, v, f, u_), u)
        
        # Compute terminal DAM_t derivatives with NUMDIFF
        dxdot_dq_t_ND = numdiff_q_dam_dyn(lambda q_:get_xdot(DAM_t, DAD_t, q_, v, f), q, rmodel)
        dxdot_dv_t_ND = numdiff(lambda v_:get_xdot(DAM_t, DAD_t, q, v_, f), v)
        dxdot_df_t_ND = numdiff(lambda f_:get_xdot(DAM_t, DAD_t, q, v, f_), f)
        dfdot_dq_t_ND = numdiff_q_dam_dyn(lambda q_:get_fdot(DAM_t, DAD_t, q_, v, f), q, rmodel)
        dfdot_dv_t_ND = numdiff(lambda v_:get_fdot(DAM_t, DAD_t, q, v_, f), v)
        dfdot_df_t_ND = numdiff(lambda f_:get_fdot(DAM_t, DAD_t, q, v, f_), f)
        Lq_dad_t_ND = numdiff_q_dam_cost(lambda q_:get_dam_cost(DAM_t, DAD_t, q_, v, f), q, rmodel)
        Lv_dad_t_ND = numdiff_vfu_dam_cost(lambda v_:get_dam_cost(DAM_t, DAD_t, q, v_, f), v)
        Lf_dad_t_ND = numdiff_vfu_dam_cost(lambda f_:get_dam_cost(DAM_t, DAD_t, q, v, f_), f)

        # Check running IAM
        assert(norm(dynext_dx - dynext_dx_ND) <= TOL)
        assert(norm(dynext_du - dynext_du_ND) <= TOL)
        assert(norm(dcost_dy - dcost_dy_ND) <= TOL)
        assert(norm(dcost_du - dcost_du_ND) <= TOL)
        assert(norm(dcstr_dy - dcstr_dy_ND) <= TOL)
        assert(norm(dcstr_du - dcstr_du_ND) <= TOL)
        
        # Check terminal IAM_t
        assert(norm(dynext_dy_t - dynext_dy_t_ND) <= TOL)
        assert(norm(dcost_dy_t - dcost_dy_t_ND) <= TOL)
        assert(norm(dcstr_dy_t - dcstr_dy_t_ND) <= TOL)

        # Check running DAM
        assert(norm(dxdot_dq - dxdot_dq_ND) <= TOL)
        assert(norm(dxdot_dv - dxdot_dv_ND) <= TOL)
        assert(norm(dxdot_df - dxdot_df_ND) <= TOL)
        assert(norm(dxdot_du - dxdot_du_ND) <= TOL)
        assert(norm(dfdot_dq - dfdot_dq_ND) <= TOL)
        assert(norm(dfdot_dv - dfdot_dv_ND) <= TOL)
        assert(norm(dfdot_df - dfdot_df_ND) <= TOL)
        assert(norm(dfdot_du - dfdot_du_ND) <= TOL)
        assert(norm(Lq_dad - Lq_dad_ND) <= TOL)
        assert(norm(Lv_dad - Lv_dad_ND) <= TOL)
        assert(norm(Lf_dad - Lf_dad_ND) <= TOL)
        assert(norm(Lu_dad - Lu_dad_ND) <= TOL)

        # Check terminal DAM_t
        assert(norm(dxdot_dq_t - dxdot_dq_t_ND) <= TOL)
        assert(norm(dxdot_dv_t - dxdot_dv_t_ND) <= TOL)
        assert(norm(dxdot_df_t - dxdot_df_t_ND) <= TOL)
        assert(norm(dfdot_dq_t - dfdot_dq_t_ND) <= TOL)
        assert(norm(dfdot_dv_t - dfdot_dv_t_ND) <= TOL)
        assert(norm(dfdot_df_t - dfdot_df_t_ND) <= TOL)
        assert(norm(Lq_dad_t - Lq_dad_t_ND) <= TOL)
        assert(norm(Lv_dad_t - Lv_dad_t_ND) <= TOL)
        assert(norm(Lf_dad_t - Lf_dad_t_ND) <= TOL)

        print("\n---> ALL TESTS PASSED.\n")



    def createSolver(self):
        solver = mim_solvers.SolverCSQP(self.ocp)
        solver.setCallbacks([mim_solvers.CallbackVerbose(), mim_solvers.CallbackLogger()])

        solver.max_qp_iters = 1000
        solver.with_callbacks = True
        solver.use_filter_line_search = False
        solver.mu_constraint = 1e1 #-1 #1e-4 #-3
        solver.mu_dynamic = 1e4 #-1
        # solver.lag_mul_inf_norm_coef = 2.
        solver.termination_tolerance = 1e-2
        solver.eps_abs = 1e-6
        solver.eps_rel = 1e-6
        # solver.extra_iteration_for_last_kkt = True
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
            f_lf = f[:3],
            f_rf = f[3:6],
            f_lh = f[6:9],
            f_rh = f[9:12],
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
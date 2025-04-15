
from Go2Py.sim.mujoco import Go2Sim
import numpy as np
import os
import sys
import mim_solvers
import friction_utils
import mujoco
import pinocchio as pin
import crocoddyl
import pinocchio
import mujoco.viewer
import time
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
    mujoco.mj_step(robot.model, robot.data)    
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
    available_sensors = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i) 
                        for i in range(model.nsensor)]
    # print("available sensors = \n", available_sensors)
    if sensor_id == -1:
        available_sensors = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i) 
                           for i in range(model.nsensor)]
        raise ValueError(f"Sensor '{sensor_name}' not found. Available sensors: {available_sensors}")
    # Get site transform
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    site_rot = data.site_xmat[site_id].reshape(3, 3)
    # MuJoCo force sensors report force in site frame
    force_in_site = data.sensordata[sensor_id:sensor_id+3]
    # Transform to world frame
    force_in_world = site_rot @ force_in_site
    # print(" SENSOR force = \n", force_in_world)
    return force_in_world.reshape(3, 1)


def setGroundFriction(model, data, mu):
    ground_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if ground_geom_id == -1:
        raise ValueError("Ground geometry not found. Check your XML model for the correct geom name.")
    print("Friction found for ground = ", model.geom_friction[ground_geom_id])
    model.geom_friction[ground_geom_id][0] = mu
    print("Set ground friction to : ", model.geom_friction[ground_geom_id])



def plot_ocp_solution(mpc):

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
    eePos_sol = []
    eeVel_sol = []
    xs, us = solver.xs, solver.us
    x = []
    f = []
    for time_idx in range (T):
        q, v = xs[time_idx][:nq], xs[time_idx][nq:nq+nv]
        for fname in mpc.ee_frame_names:
            frame_idx = rmodel.getFrameId(fname)
            ct_frame_name = rmodel.frames[frame_idx].name + "_contact"
            f.append(mpc.solver.problem.runningDatas[0].differential.multibody.contacts.contacts[ct_frame_name].f.linear)
        pin.framesForwardKinematics(rmodel, rdata, q)
        eePos_sol.append(rdata.oMf[mpc.armEEId].translation)
        eeVel_sol.append(pin.getFrameVelocity(rmodel, rdata, mpc.armEEId).linear)
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
                        'jointTorques':jointTorques_sol, 
                        'eePos': eePos_sol, 'eeVel':eeVel_sol}       

    # Extract contact forces by hand
    sol['FL_FOOT_contact'] = [force_sol[i][0:3] for i in range(T)]     
    sol['FR_FOOT_contact'] = [force_sol[i][3:6] for i in range(T)]     
    sol['HL_FOOT_contact'] = [force_sol[i][6:9] for i in range(T)]     
    sol['HR_FOOT_contact'] = [force_sol[i][9:12] for i in range(T)]     
    sol['Link6'] = [force_sol[i][-3:] for i in range(T)]     

    # sol['FL_calf2FL_dummy_fixed_contact'] = [force_sol[i][0:3] for i in range(T)]     
    # sol['FR_calf2FR_dummy_fixed_contact'] = [force_sol[i][3:6] for i in range(T)]     
    # sol['RL_calf2RL_dummy_fixed_contact'] = [force_sol[i][6:9] for i in range(T)]     
    # sol['RR_calf2RR_dummy_fixed_contact'] = [force_sol[i][9:12] for i in range(T)]     
    # sol['Link62EF_dummy_fixed'] = [force_sol[i][-3:] for i in range(T)]     

    # Plotting 
    import matplotlib.pyplot as plt
    time_lin = np.linspace(0, T, T)
    # Feet forces
    fig, axs = plt.subplots(4, 3, constrained_layout=True)
    for i, frame_idx in enumerate(supportFeetIds):
        ct_frame_name = rmodel.frames[frame_idx].name + "_contact"
        print(ct_frame_name)
        forces = np.array(sol[ct_frame_name])
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
    
    # EE Force
    fig, axs = plt.subplots(3, 1, constrained_layout=True)
    # forces = np.array(sol['Link62EF_dummy_fixed'])
    forces = np.array(sol['Link6'])
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

    ## EE pos and vel
    fig, axs = plt.subplots(3, 2, constrained_layout=True)
    eePos = np.array(sol['eePos'])
    eeVel = np.array(sol['eeVel'])
    axs[0, 0].plot(time_lin, eePos[:, 0], marker='.', linewidth=2, label="Px_ee")
    axs[1, 0].plot(time_lin, eePos[:, 1], marker='.', linewidth=2, label="Py_ee")
    axs[2, 0].plot(time_lin, eePos[:, 2], marker='.', linewidth=2, label="Pz_ee")
    axs[0, 1].plot(time_lin, eeVel[:, 0], marker='.', linewidth=2, label="Vx_ee")
    axs[1, 1].plot(time_lin, eeVel[:, 1], marker='.', linewidth=2, label="Vy_ee")
    axs[2, 1].plot(time_lin, eeVel[:, 2], marker='.', linewidth=2, label="Vz_ee")
    for i in range(3):
        axs[i,0].legend()
        axs[i,1].legend()
        axs[i,0].grid()
        axs[i,1].grid()
    fig.suptitle('EE trajectories', fontsize=16)
    axs[0, 0].set_xlabel(r"$P_x$")
    axs[1, 0].set_xlabel(r"$P_y$")
    axs[2, 0].set_xlabel(r"$P_z$")
    axs[0, 1].set_xlabel(r"$V_x$")
    axs[1, 1].set_xlabel(r"$V_y$")
    axs[2, 1].set_xlabel(r"$V_z$")

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
    plt.show()


class Go2MPCClassical:
    def __init__(self, HORIZON=250, friction_mu = 0.75, dt = 0.01, USE_MUJOCO=True):
        self.HORIZON = HORIZON
        self.max_iterations = 500
        self.dt = dt
        self.pinRef = pin.LOCAL_WORLD_ALIGNED
        self.friction_mu = friction_mu 
        self.USE_MUJOCO = USE_MUJOCO
        if(self.USE_MUJOCO):
            print("Loading XML Go2")
            self.assets_path = '/home/skleff/force_feedback_ws/Go2Py/Go2Py/assets/'
            self.urdf_path = os.path.join(self.assets_path, 'urdf/go2_with_arm.urdf')
            self.xml_path = os.path.join(self.assets_path, 'mujoco/go2_with_arm.xml')
            self.pin_robot = pin.RobotWrapper.BuildFromURDF(self.urdf_path, self.assets_path, pin.JointModelFreeFlyer())
        else:
            from mim_robots.robot_loader import load_pinocchio_wrapper
            print("Loading pinocchio wrapper Go2")
            self.pin_robot = load_pinocchio_wrapper('go2')
        self.rmodel = self.pin_robot.model
        self.rdata = self.pin_robot.data

        if(self.USE_MUJOCO):
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
        # self.ee_frame_names = ["FL_calf2FL_dummy_fixed", "FR_calf2FR_dummy_fixed", "RL_calf2RL_dummy_fixed", "RR_calf2RR_dummy_fixed", "Link62EF_dummy_fixed"]
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
                    +4*[0.0, 0.77832842, -1.56065452] + [0.0, 0.3, -0.3, 0.0, 0.0, 0.0]), FREF=15):
        q0[11+2]=0.0
        self.q0 = q0.copy()
        self.x0 =  np.concatenate([q0, np.zeros(self.rmodel.nv)])
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        self.Fx_ref_ee = FREF
        self.rfFootPos0 = self.rdata.oMf[self.rfFootId].translation
        self.rhFootPos0 = self.rdata.oMf[self.rhFootId].translation
        self.lfFootPos0 = self.rdata.oMf[self.lfFootId].translation
        self.lhFootPos0 = self.rdata.oMf[self.lhFootId].translation 
        self.armEEPos0 = self.rdata.oMf[self.armEEId].translation
        self.armEEOri0 = self.rdata.oMf[self.armEEId].rotation
        self.supportFeetIds = [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId]
        # self.supportFeePos = [self.lfFootPos0, self.rfFootPos0, self.lhFootPos0, self.rhFootPos0]
        self.xs = [self.x0]*(self.HORIZON + 1)
        self.createProblem()
        self.createSolver()
        self.us = self.solver.problem.quasiStatic([self.x0]*self.HORIZON) 
        print("wepifuwefweifo\n\n\n", self.us[0])

    def createProblem(self):

        # Now define the model
        for t in range(self.HORIZON+1):
            self.contactModel = crocoddyl.ContactModelMultiple(self.ccdyl_state, self.nu)
            costModel = crocoddyl.CostModelSum(self.ccdyl_state, self.nu)

            # Add contacts
            for i,frame_idx in enumerate(self.supportFeetIds):
                support_contact = crocoddyl.ContactModel3D(self.ccdyl_state, frame_idx, np.array([0., 0., 0.0]), self.pinRef, self.nu, np.array([0., 20.]))
                self.contactModel.addContact(self.rmodel.frames[frame_idx].name + "_contact", support_contact) 
                # print("Create ", self.rmodel.frames[frame_idx].name + "_contact")

            # Contact for the EE
            arm_contact = crocoddyl.ContactModel3D(self.ccdyl_state, self.armEEId, self.armEEPos0, pin.LOCAL_WORLD_ALIGNED, self.nu, np.array([0., 20.]))
            self.contactModel.addContact(self.rmodel.frames[self.armEEId].name + "_contact", arm_contact) 
            # print("Create ", self.rmodel.frames[self.armEEId].name + "_contact")
            
            # Add state/control regularization costs
            state_reg_weight, control_reg_weight = 1e-1, 1e-3
            freeFlyerQWeight = [0.]*3 + [500.]*3
            freeFlyerVWeight = [10.]*6
            legsQWeight = [0.01]*(self.rmodel.nv - 6)
            legsQWeight[-1] = 100
            legsQWeight[-2] = 100
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
            
            # Force tracking term
            if t != self.HORIZON:
                self.ef_des_force = pin.Force.Zero()
                self.ef_des_force.linear[0] = -self.Fx_ref_ee
                contact_force_residual = crocoddyl.ResidualModelContactForce(self.ccdyl_state, self.armEEId, self.ef_des_force, 3, self.nu)
                contact_force_activation = crocoddyl.ActivationModelWeightedQuad(np.array([1., 1., 1.]))
                contact_force_track = crocoddyl.CostModelResidual(self.ccdyl_state, contact_force_activation, contact_force_residual)
                costModel.addCost("contact_force_track", contact_force_track, 1e1)

            # Friction Cone Constraints
            constraintModelManager = crocoddyl.ConstraintModelManager(self.ccdyl_state, self.ccdyl_actuation.nu)
            if(t != self.HORIZON):
                for frame_idx in self.supportFeetIds:
                    name = self.rmodel.frames[frame_idx].name + "_contact"
                    residualFriction = friction_utils.ResidualFrictionCone(self.ccdyl_state, name, self.friction_mu, self.ccdyl_actuation.nu)
                    constraintFriction = crocoddyl.ConstraintModelResidual(self.ccdyl_state, residualFriction, np.array([0.]), np.array([np.inf]))
                    constraintModelManager.addConstraint(name + "friction", constraintFriction)

            # # End Effecor Position Tracking Cost
            # ef_pos_ref = self.armEEPos0
            # ef_residual = crocoddyl.ResidualModelFrameTranslation(self.ccdyl_state, self.armEEId, ef_pos_ref, self.nu) # Check this cost term            
            # ef_activation = crocoddyl.ActivationModelWeightedQuad(np.array([1., 1., 1.]))
            # ef_track = crocoddyl.CostModelResidual(self.ccdyl_state, ef_activation, ef_residual)
            # ef_weight = 1e5
            # if t != self.HORIZON:
            #     costModel.addCost("ef_track", ef_track, ef_weight)
            # else:
            #     costModel.addCost("ef_track", ef_track, ef_weight*self.dt)

            #     # enforce unilaterality (cannot create negative forces) Fz_(env->robot) > 0 
            #     residualForce = crocoddyl.ResidualModelContactForce(self.ccdyl_state, frame_idx, pin.Force.Zero(), 3, self.nu)
            #     forceBoxConstraint = crocoddyl.ConstraintModelResidual(self.ccdyl_state, residualForce, np.array([-np.inf, -np.inf, 0.]), np.array([np.inf, np.inf, np.inf]))
            #     constraintModelManager.addConstraint(name + "Box", forceBoxConstraint)

            # # enforce unilaterality (cannot create negative forces) Fz_(env->robot) > 0 
            # residualForce = crocoddyl.ResidualModelContactForce(self.ccdyl_state, self.armEEId, pin.Force.Zero(), 3, self.nu)
            # forceBoxConstraint = crocoddyl.ConstraintModelResidual(self.ccdyl_state, residualForce, np.array([-np.inf, -np.inf, -np.inf]), np.array([0., np.inf, np.inf]))
            # constraintModelManager.addConstraint("efBox", forceBoxConstraint)

            # ctrlResidual2 = crocoddyl.ResidualModelControl(self.ccdyl_state, self.nu)
            # torque_lb = -self.ctrlLim #self.pin_robot.model.effortLimit
            # torque_ub = self.ctrlLim #self.pin_robot.model.effortLimit
            # # print("Ctrl limit lb = \n", torque_lb)
            # # print("Ctrl limit ub = \n", torque_ub)
            # torqueBoxConstraint = crocoddyl.ConstraintModelResidual(self.ccdyl_state, ctrlResidual2, torque_lb, torque_ub)
            # constraintModelManager.addConstraint("ctrlBox", torqueBoxConstraint)
            
            dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(self.ccdyl_state, self.ccdyl_actuation, self.contactModel, costModel, constraintModelManager, 0., True)
            model = crocoddyl.IntegratedActionModelEuler(dmodel, self.dt)
            self.running_models += [model]
        self.ocp = crocoddyl.ShootingProblem(self.x0, self.running_models[:-1], self.running_models[-1])
        
    def createSolver(self):
        solver = mim_solvers.SolverCSQP(self.ocp)
        solver.setCallbacks([mim_solvers.CallbackVerbose(), mim_solvers.CallbackLogger()])
        solver.max_qp_iters = 1000
        solver.with_callbacks = True
        solver.use_filter_line_search = False
        solver.mu_constraint = -1 #1e3
        solver.lag_mul_inf_norm_coef = 10.
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
        dq = self.xs[x_idx][25+6:]
        constraint_norm = self.solver.constraint_norm
        if(self.USE_MUJOCO):
            sol_dict = dict(
                position=t,
                orientation=np.array([qw, qx, qy, qz]), #Mujoco and uniree quaternion order
                velocity = eta[:3],
                omega = eta[3:],
                q = q[self.mpc_to_unitree_idx],
                dq = dq[self.mpc_to_unitree_idx], 
                tau = self.us[u_idx][[self.mpc_to_unitree_idx]],
                constraint_norm = constraint_norm
            )
        else:
            sol_dict = dict(
                position=t,
                orientation=np.array([qw, qx, qy, qz]), #Mujoco and uniree quaternion order
                velocity = eta[:3],
                omega = eta[3:],
                q = q,
                dq = dq, 
                tau = self.us[u_idx],
                constraint_norm = constraint_norm
            )
        # Extract 3d force solution for each contact frame (4 feet + end-effector)
        for fname in self.ee_frame_names:
            frame_idx = self.rmodel.getFrameId(fname)
            ct_frame_name = self.rmodel.frames[frame_idx].name + "_contact"
            data = self.solver.problem.runningDatas[0].differential.multibody.contacts.contacts[ct_frame_name] 
            # jf = data.f                                                         # !!! expressed at parent joint 
            # lf = data.jMf.actInv(jf).linear                                     # !!! expressed in LOCAL 
            # pin.framesForwardKinematics(self.rmodel, self.rdata, self.xs[0][:25])
            # sol_dict[ct_frame_name] = self.rdata.oMf[frame_idx].rotation.T @ lf # convert into LOCAL_WORLD_ALIGNED
            sol_dict[ct_frame_name] = data.f.linear # convert into LOCAL_WORLD_ALIGNED
        return sol_dict
    
    def updateAndSolve(self, t, quat, q, v, omega, dq):
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
        x = np.hstack([q_, dq_])
        self.solver.problem.x0 = x
        self.xs = list(self.solver.xs[1:]) + [self.solver.xs[-1]]
        self.xs[0] = x
        self.us = list(self.us[1:]) + [self.us[-1]] 
        self.solver.solve(self.xs, self.us, self.max_iterations)
        self.xs, self.us = self.solver.xs, self.solver.us
        return self.getSolution()

    def updateAndSolve2(self, q, dq):
        pin.framesForwardKinematics(self.rmodel, self.rdata, q)
        pin.computeAllTerms(self.rmodel, self.rdata, q, dq)
        x = np.hstack([q, dq])
        self.solver.problem.x0 = x
        self.xs = list(self.solver.xs[1:]) + [self.solver.xs[-1]]
        self.xs[0] = x
        self.us = list(self.us[1:]) + [self.us[-1]] 
        self.solver.solve(self.xs, self.us, self.max_iterations)
        self.xs, self.us = self.solver.xs, self.solver.us
        return self.getSolution()
    
    def solve(self):
        self.solver.solve(self.xs, self.us, self.max_iterations)
        self.xs, self.us = self.solver.xs, self.solver.us
        return self.getSolution()



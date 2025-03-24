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

    def initialize(self, q0=np.array([0.0, 0.0, 0.33, 0.0, 0.0, 0.0, 1.0] 
                    +4*[0.0, 0.77832842, -1.56065452] + [0.0, 0.3, -0.3, 0.0, 0.0, 0.0]
                        )):
        q0[11+2]=0.0
        self.q0 = q0.copy()
        self.x0 =  np.concatenate([q0, np.zeros(self.rmodel.nv)])
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
        self.xs = [self.x0]*(self.HORIZON + 1)
        self.createProblem()
        self.createSolver()
        self.us = self.solver.problem.quasiStatic([self.x0]*self.HORIZON) 

    def createProblem(self):

        # Now define the model
        for t in range(self.HORIZON+1):
            self.contactModel = crocoddyl.ContactModelMultiple(self.ccdyl_state, self.nu)
            costModel = crocoddyl.CostModelSum(self.ccdyl_state, self.nu)

            # Add contacts
            for i,frame_idx in enumerate(self.supportFeetIds):
                support_contact = crocoddyl.ContactModel3D(self.ccdyl_state, frame_idx, np.array([0., 0., 0.0]), self.pinRef, self.nu, np.array([0., 0.]))
                self.contactModel.addContact(self.rmodel.frames[frame_idx].name + "_contact", support_contact) 
                # print("Create ", self.rmodel.frames[frame_idx].name + "_contact")

            # Contact for the EE
            support_contact = crocoddyl.ContactModel3D(self.ccdyl_state, self.armEEId, self.armEEPos0, pin.LOCAL_WORLD_ALIGNED, self.nu, np.array([0., 0.]))
            self.contactModel.addContact(self.rmodel.frames[self.armEEId].name + "_contact", support_contact) 
            # print("Create ", self.rmodel.frames[self.armEEId].name + "_contact")
            
            # Add state/control regularization costs
            state_reg_weight, control_reg_weight = 1e-1, 1e-3

            freeFlyerQWeight = [0.]*3 + [500.]*3
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
            
            # Force tracking term
            if t != self.HORIZON:
                self.ef_des_force = pin.Force.Zero()
                self.ef_des_force.linear[0] = -20
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
                    # constraintModelManager.addConstraint(name + "friction", constraintFriction)

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
        solver.max_qp_iters = 10000
        solver.with_callbacks = True
        solver.use_filter_line_search = False
        solver.mu_constraint = 1e3
        solver.termination_tolerance = 1e-2
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
    
    def solve(self):
        self.solver.solve(self.xs, self.us, self.max_iterations)
        self.xs, self.us = self.solver.xs, self.solver.us
        return self.getSolution()



# Add this to your validation code
def test_force_sensor_orientation():
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

# Instantiate the simulator
robot=Go2Sim(with_arm=True)
map = np.zeros((1200, 1200))
map[:,649:679] = 400
robot.updateHeightMap(map)

# Instantiate the solver
assets_path = '/home/skleff/force_feedback_ws/Go2Py/Go2Py/assets/'
MU = 0.75
mpc = Go2MPC(assets_path, HORIZON=20, friction_mu=MU)
mpc.initialize()
mpc.max_iterations=500
mpc.solve()
m = list(mpc.solver.problem.runningModels) + [mpc.solver.problem.terminalModel]

# Reset the robot
state = mpc.getSolution(0)
robot.pos0 = state['position']
robot.rot0 = state['orientation']
robot.q0 = state['q']
robot.reset()

# Solve for as many iterations as needed for the first step
mpc.max_iterations=10

Nsim = 100
# measured_forces = []
measured_forces_dict = {}
predicted_forces_dict = {}
frame_name_to_mujoco_sensor = {'FL_FOOT': 'FL_force_site', 
                               'FR_FOOT': 'FR_force_site', 
                               'HL_FOOT': 'RL_force_site', 
                               'HR_FOOT': 'RR_force_site', 
                               'Link6': 'EF_force_site'}
for fname in mpc.ee_frame_names:
    measured_forces_dict[fname]  = []
    predicted_forces_dict[fname] = []

# measured_forces_FR = []
desired_forces = []
joint_torques = []
f_des_z = np.array([20.]*Nsim) #np.linspace(50, 100, Nsim) # TODO: implement 3D force !  
# breakpoint()
WITH_INTEGRAL = False
if(WITH_INTEGRAL):
    Ki = 0.01
    err_f3d = np.zeros(3)
# Set ground friction in Mujoco
setGroundFriction(robot.model, robot.data, MU)


# test_force_sensor_orientation()

# Main simulation loop
for i in range(Nsim):
    print("Step ", i)
    # set the force setpoint
    f_des_3d = np.array([-f_des_z[i], 0, 0])
    desired_forces.append(f_des_3d)
    for action_model in m[:-1]:
        action_model.differential.costs.costs['contact_force_track'].cost.residual.reference.linear[:] = f_des_3d
    # Get state from simulation
    state = robot.getJointStates()
    q = state['q']
    dq = state['dq']
    t, quat = robot.getPose()
    v = robot.data.qvel[:3]
    omega = robot.data.qvel[3:6]
    q = np.hstack([q, np.zeros(2)])
    dq = np.hstack([dq, np.zeros(2)])
    # Solve OCP
    solution = mpc.updateAndSolve(t, quat, q, v, omega, dq)
    # Save the solution
    q = solution['q']
    dq = solution['dq']
    tau = solution['tau'].squeeze()
    joint_torques.append(tau)
    kp = np.ones(18)*0.
    kv = np.ones(18)*0.
    # Step the physics
    for fname in mpc.ee_frame_names:
        # print("Extract name ", fname, " , Mujoco sensor = ", frame_name_to_mujoco_sensor[fname])
        measured_forces_dict[fname].append(-getForceSensor(robot.model, robot.data, frame_name_to_mujoco_sensor[fname]).squeeze().copy())
        predicted_forces_dict[fname].append(solution[fname+'_contact'])
    # compute the force integral error and map it to joint torques
    if(WITH_INTEGRAL):
        err_f3d -= Ki * (f_des_3d - measured_forces_dict['Link6'][i])
        pin.computeAllTerms(mpc.rmodel, mpc.rdata, mpc.xs[0][:mpc.rmodel.nq], mpc.xs[0][mpc.rmodel.nq:])
        J = pin.getFrameJacobian(mpc.rmodel, mpc.rdata, mpc.armEEId, pin.LOCAL_WORLD_ALIGNED)
        tau_int = J[:3,6:].T @ err_f3d 
        print(tau_int)
        tau += tau_int
    for j in range(int(mpc.dt//robot.dt)):
        robot.setCommands(q, dq, kp, kv, tau)
        robot.step()

# measured_forces = np.array(measured_forces)
desired_forces = np.array(desired_forces)
joint_torques = np.array(joint_torques)
for fname in mpc.ee_frame_names:
    measured_forces_dict[fname] = np.array(measured_forces_dict[fname])
    predicted_forces_dict[fname] = np.array(predicted_forces_dict[fname])

# Visualize the measured force against the desired
import matplotlib.pyplot as plt
time_span = np.linspace(0, (Nsim-1)*1e-3, Nsim)
# EE FORCES
fig, axs = plt.subplots(3, 1, constrained_layout=True)
axs[0].plot(time_span, measured_forces_dict['Link6'][:,0],linewidth=4, color='r', marker='o',  label="Fx mea")
axs[0].plot(time_span, desired_forces[:,0], linewidth=4, color='b', marker='o', label="Fx des")
axs[1].plot(time_span, measured_forces_dict['Link6'][:,1],linewidth=4, color='r', marker='o',  label="Fy mea")
axs[1].plot(time_span, desired_forces[:,1], linewidth=4, color='b', marker='o', label="Fy des")
axs[2].plot(time_span, measured_forces_dict['Link6'][:,2],linewidth=4, color='r', marker='o',  label="Fz mea")
axs[2].plot(time_span, desired_forces[:,2], linewidth=4, color='b', marker='o', label="Fz des")
for i in range(3):
    axs[i].legend()
    axs[i].grid()
fig.suptitle('Contact force at the end-effector', fontsize=16)

# FEET FORCES (measured and predicted, with friction constraint lower bound on Fz)
fig, axs = plt.subplots(3, 4, constrained_layout=True)
for i,fname in enumerate(mpc.ee_frame_names[:-1]):
    # x,y
    axs[0, i].plot(time_span, measured_forces_dict[fname][:,0], linewidth=4, color='r', marker='o', label="Fx measured")
    axs[0, i].plot(time_span, predicted_forces_dict[fname][:,0], linewidth=4, color='b', marker='o', alpha=0.5, label="Fx predicted")
    axs[1, i].plot(time_span, measured_forces_dict[fname][:,1], linewidth=4, color='r', marker='o', label="Fy measured")
    axs[1, i].plot(time_span, predicted_forces_dict[fname][:,1], linewidth=4, color='b', marker='o', alpha=0.5, label="Fy predicted")
    axs[0, i].legend()
    # axs[0, i].title(fname)
    axs[0, i].grid()
    axs[1, i].legend()
    axs[1, i].grid()

    # z
    Fz_lb_mea = (1./MU)*np.sqrt(measured_forces_dict[fname][:, 0]**2 + measured_forces_dict[fname][:, 1]**2)
    Fz_lb_pred = (1./MU)*np.sqrt(predicted_forces_dict[fname][:, 0]**2 + predicted_forces_dict[fname][:, 1]**2)
    axs[2, i].plot(time_span, measured_forces_dict[fname][:,2], linewidth=4, color='r', marker='o', label="Fz measured")
    axs[2, i].plot(time_span, predicted_forces_dict[fname][:,2], linewidth=4, color='b', marker='o', alpha=0.5, label="Fz predicted")
    axs[2, i].plot(time_span, Fz_lb_mea, '--', linewidth=4, color='r',  alpha=0.2, label="Fz friction lb (mea)")
    axs[2, i].plot(time_span, Fz_lb_pred, '--', linewidth=4, color='b', alpha=0.2, label="Fz friction lb (pred)")
    axs[2, i].legend()
    axs[2, i].grid()

# for i in range(3):
#     axs[i].legend()
#     axs[i].grid()
fig.suptitle('Contact forces at feet FL, FR, HL, HR', fontsize=16)


# plt.plot(measured_forces[:300,0], '*')
# plt.plot(forces,'k')
plt.show()
robot.close()

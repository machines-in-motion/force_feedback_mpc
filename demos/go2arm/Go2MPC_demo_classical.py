'''
Adapted from Rooholla's code in Go2Py examples
https://github.com/machines-in-motion/Go2Py/blob/mpc/examples/standard_mpc.py

This sets up the Go2 MPC wrapper and runs an MPC simulation in Mujoco
the robot must push against a wall with its end-effector while standing
on its 4 feet and without slipping (the same task as in the original script)

Classical MPC (rigid contact force)
'''

import numpy as np
import pinocchio as pin



from Go2MPC_wrapper_classical import Go2MPCClassical, getForceSensor, setGroundFriction, plot_ocp_solution

# LOADING CONFIG parameters
import os
from force_feedback_mpc.core_mpc_utils.path_utils import load_yaml_file
CONFIG       = load_yaml_file(os.path.dirname(os.path.realpath(__file__))+'/Go2MPC_demo_classical.yml')
USE_MUJOCO   = CONFIG['USE_MUJOCO']
DT_SIMU      = CONFIG['DT_SIMU']
FREF         = CONFIG['FREF']
MU           = CONFIG['MU']
HORIZON      = CONFIG['HORIZON']
DT_OCP       = CONFIG['DT_OCP']
MAX_ITER_1   = CONFIG['MAX_ITER_1']
MAX_ITER_2   = CONFIG['MAX_ITER_2']
N_SIMU       = CONFIG['N_SIMU']
MPC_FREQ     = CONFIG['MPC_FREQ']
KP           = CONFIG['KP']
KV           = CONFIG['KV']
SIM_FREQ     = int(1./DT_SIMU)
USE_INTEGRAL = CONFIG['USE_INTEGRAL']

# Instantiate the simulator
if(USE_MUJOCO):
    from Go2Py.sim.mujoco import Go2Sim
    robot=Go2Sim(with_arm=True, dt=0.001)
    map = np.zeros((1200, 1200))
    map[:,649:679] = 400
    robot.updateHeightMap(map)
else:
    from mim_robots.robot_loader import load_bullet_wrapper
    from mim_robots.pybullet.env import BulletEnvWithGround
    import pybullet as p
    from force_feedback_mpc.core_mpc_utils import sim_utils
    env = BulletEnvWithGround(dt=DT_SIMU, server=p.GUI)
    robot = load_bullet_wrapper('go2')
    q0 = np.array([-0.01, 0.0, 0.32, 0.0, 0.0, 0.0, 1.0] + 4*[0.0, 0.77832842, -1.56065452] + [0.0, 0.3, -0.3, 0.0, 0.0, 0.0])
    v0 = np.zeros(robot.pin_robot.model.nv)
    env.add_robot(robot) 
    robot.reset_state(q0, v0)
    robot.forward_robot(q0, v0)
    sim_utils.set_lateral_friction(env.objects[0], MU)
    sim_utils.set_contact_stiffness_and_damping(env.objects[0], 10000, 500)
    contact_placement = pin.SE3(pin.rpy.rpyToMatrix(0.,np.pi/2, 0.), np.array([0.41,0.,0.]))
    contact_surface_bulletId = sim_utils.display_contact_surface(contact_placement, radius=2., bullet_endeff_ids=robot.bullet_endeff_ids)
    sim_utils.set_lateral_friction(contact_surface_bulletId, MU)
    sim_utils.set_contact_stiffness_and_damping(contact_surface_bulletId, 10000, 500)
    # floor props
    p.changeDynamics(
        env.objects[0],
        0,
        lateralFriction=MU,
        spinningFriction=0.,
        rollingFriction=0.,
    )

# Instantiate the solver
mpc = Go2MPCClassical(HORIZON=20, friction_mu=MU, dt=DT_OCP, USE_MUJOCO=USE_MUJOCO)
mpc.initialize(FREF=FREF)
mpc.max_iterations=MAX_ITER_1
mpc.solve()
m = list(mpc.solver.problem.runningModels) + [mpc.solver.problem.terminalModel]
mpc.max_iterations=MAX_ITER_2

# plot_ocp_solution(mpc)

if(USE_MUJOCO):
    # Reset the robot
    state = mpc.getSolution(0)
    robot.pos0 = state['position']
    robot.rot0 = state['orientation']
    robot.q0 = state['q']
    robot.reset()
    frame_name_to_mujoco_sensor = {'FL_FOOT': 'FL_force_site', 
                                'FR_FOOT': 'FR_force_site', 
                                'HL_FOOT': 'RL_force_site', 
                                'HR_FOOT': 'RR_force_site', 
                                'Link6': 'EF_force_site'}
    setGroundFriction(robot.model, robot.data, MU)
# else:
#     sim_utils.set_lateral_friction(contact_surface_bulletId, MU)
#     sim_utils.set_contact_stiffness_and_damping(contact_surface_bulletId, 10000, 500)


# measured_forces = []
measured_forces_dict = {}
predicted_forces_dict = {}
for fname in mpc.ee_frame_names:
    measured_forces_dict[fname]  = []
    predicted_forces_dict[fname] = []
desired_forces = []
joint_torques = []
f_des_z = np.array([15.]*N_SIMU) 
# breakpoint()
WITH_INTEGRAL = USE_INTEGRAL
if(WITH_INTEGRAL):
    Ki = 0.01
    err_f3d = np.zeros(3)
# test_force_sensor_orientation()
ANTI_WINDUP = 100

# MUJOCO SIMULATION
if(USE_MUJOCO):
    for i in range(N_SIMU):
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
        if(i%int(SIM_FREQ/MPC_FREQ)==0):
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
            # if(i%ANTI_WINDUP==0):
                # err_f3d = np.zeros(3)
            if(i%int(SIM_FREQ/MPC_FREQ)==0):
                err_f3d -= Ki * (f_des_3d - measured_forces_dict['Link6'][i])
                pin.computeAllTerms(mpc.rmodel, mpc.rdata, mpc.xs[0][:mpc.rmodel.nq], mpc.xs[0][mpc.rmodel.nq:])
                J = pin.getFrameJacobian(mpc.rmodel, mpc.rdata, mpc.armEEId, pin.LOCAL_WORLD_ALIGNED)
                tau_int = J[:3,6:].T @ err_f3d 
                # print(tau_int)
                tau += tau_int
        robot.setCommands(q, dq, kp, kv, tau)
        robot.step()
# PYBULLET SIMULATION
else:
    for i in range(N_SIMU):
        print("Step ", i)
        # set the force setpoint
        f_des_3d = np.array([-f_des_z[i], 0, 0])
        desired_forces.append(f_des_3d)
        for action_model in m[:-1]:
            action_model.differential.costs.costs['contact_force_track'].cost.residual.reference.linear[:] = f_des_3d
        # Get state from simulation
        q, dq = robot.get_state()
        # Solve OCP
        if(i%int(SIM_FREQ/MPC_FREQ)==0):
            solution = mpc.updateAndSolve2(q, dq)
            # plot_ocp_solution(mpc)
        # Save the solution
        tau = solution['tau'].squeeze()
        joint_torques.append(tau)
        # Measure forces and save predicted force from solution
        robot.forward_robot(q, dq)
        contact_status, f_mea_bullet = robot.end_effector_forces()
        # print("Contacts status = \n", contact_status)
        # print("Contacts forces = \n", f_mea_bullet)
        for k,fname in enumerate(mpc.ee_frame_names):
            # print(fname, robot.pinocchio_endeff_ids[k], robot.endeff_names[k], robot.bullet_endeff_ids[k])
            if(contact_status[robot.pinocchio_endeff_ids[k]]):
                f_mea = f_mea_bullet[k,:3]
            else:
                f_mea = np.zeros(3)
            # print("Extract name ", fname, " , Mujoco sensor = ", frame_name_to_mujoco_sensor[fname])
            measured_forces_dict[fname].append(f_mea)
            predicted_forces_dict[fname].append(solution[fname+'_contact'])
        # compute the force integral error and map it to joint torques
        if(WITH_INTEGRAL):
            # if(i%ANTI_WINDUP==0):
                # err_f3d = np.zeros(3)
            if(i%int(SIM_FREQ/MPC_FREQ)==0):
                err_f3d -= Ki * (f_des_3d - measured_forces_dict['Link6'][i])
                pin.computeAllTerms(mpc.rmodel, mpc.rdata, mpc.xs[0][:mpc.rmodel.nq], mpc.xs[0][mpc.rmodel.nq:])
                J = pin.getFrameJacobian(mpc.rmodel, mpc.rdata, mpc.armEEId, pin.LOCAL_WORLD_ALIGNED)
                tau_int = J[:3,6:].T @ err_f3d 
                # print(tau_int)
                tau += tau_int
        # Step the physics
        print(mpc.rdata.oMf[mpc.armEEId].translation)
        robot.send_joint_command(tau)
        env.step() 

# measured_forces = np.array(measured_forces)
desired_forces = np.array(desired_forces)
joint_torques = np.array(joint_torques)
for fname in mpc.ee_frame_names:
    measured_forces_dict[fname] = np.array(measured_forces_dict[fname])
    predicted_forces_dict[fname] = np.array(predicted_forces_dict[fname])

# Visualize the measured force against the desired
import matplotlib.pyplot as plt
time_span = np.linspace(0, (N_SIMU-1)*DT_SIMU, N_SIMU)
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
    axs[0, i].plot(time_span, predicted_forces_dict[fname][:,0], linewidth=4, color='b', marker='o', alpha=0.25, label="Fx predicted")
    axs[1, i].plot(time_span, measured_forces_dict[fname][:,1], linewidth=4, color='r', marker='o', label="Fy measured")
    axs[1, i].plot(time_span, predicted_forces_dict[fname][:,1], linewidth=4, color='b', marker='o', alpha=0.25, label="Fy predicted")
    axs[0, i].legend()
    # axs[0, i].title(fname)
    axs[0, i].grid()
    axs[1, i].legend()
    axs[1, i].grid()

    # z
    Fz_lb_mea = (1./MU)*np.sqrt(measured_forces_dict[fname][:, 0]**2 + measured_forces_dict[fname][:, 1]**2)
    Fz_lb_pred = (1./MU)*np.sqrt(predicted_forces_dict[fname][:, 0]**2 + predicted_forces_dict[fname][:, 1]**2)
    axs[2, i].plot(time_span, measured_forces_dict[fname][:,2], linewidth=4, color='r', marker='o', label="Fz measured")
    axs[2, i].plot(time_span, predicted_forces_dict[fname][:,2], linewidth=4, color='b', marker='o', alpha=0.25, label="Fz predicted")
    axs[2, i].plot(time_span, Fz_lb_mea, '--', linewidth=4, color='k',  alpha=0.5, label="Fz friction constraint (lower bound)")
    # axs[2, i].plot(time_span, Fz_lb_pred, '--', linewidth=4, color='b', alpha=0.2, label="Fz friction lb (pred)")
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

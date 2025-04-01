'''
Adapted from Rooholla's code in Go2Py examples
https://github.com/machines-in-motion/Go2Py/blob/mpc/examples/standard_mpc.py

This sets up the Go2 MPC wrapper and runs an MPC simulation in Mujoco
the robot must push against a wall with its end-effector while standing
on its 4 feet and without slipping (the same task as in the original script)
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

from Go2MPC_wrapper import Go2MPC, getForceSensor, setGroundFriction, plot_ocp_solution_with_cones
from Go2Py.sim.mujoco import Go2Sim
from utils import ExpMovingAvg, LPFButterOrder1, LPFButterOrder2, LPFButterOrder3

# Instantiate the simulator
robot=Go2Sim(with_arm=True, dt=0.001)
map = np.zeros((1200, 1200))
map[:,649:679] = 400
robot.updateHeightMap(map)

# Instantiate the solver
assets_path = '/home/skleff/force_feedback_ws/Go2Py/Go2Py/assets/'
MU = 0.75
mpc = Go2MPC(assets_path, HORIZON=20, friction_mu=MU, dt=0.02)
mpc.initialize()
mpc.max_iterations=100
mpc.solve()
m = list(mpc.solver.problem.runningModels) + [mpc.solver.problem.terminalModel]

# plot_ocp_solution_with_cones(mpc)
# Extract OCP Solution 
force_sol = []
xs, us = mpc.solver.xs, mpc.solver.us
for time_idx in range (mpc.HORIZON+1):
    f = xs[time_idx][-3:]
    force_sol    += [f]
force_sol = np.array(force_sol)   
time_span = np.linspace(0, mpc.HORIZON*mpc.dt, mpc.HORIZON+1)
import matplotlib.pyplot as plt
plt.plot(time_span, force_sol[:,0], label='x')
plt.plot(time_span, [-15]*(mpc.HORIZON+1), 'k-.', label='ref x')
plt.plot(time_span, force_sol[:,1], label='y')
plt.plot(time_span, force_sol[:,2], label='z')
plt.legend()
plt.show()

# Reset the robot
state = mpc.getSolution(0)
robot.pos0 = state['position']
robot.rot0 = state['orientation']
robot.q0 = state['q']
robot.reset()

# Solve for as many iterations as needed for the first step
mpc.max_iterations=10

Nsim = 200
# measured_forces = []
measured_forces_dict = {}
predicted_forces_dict = {}
frame_name_to_mujoco_sensor = {'FL_FOOT': 'FL_force_site', 
                               'FR_FOOT': 'FR_force_site', 
                               'HL_FOOT': 'RL_force_site', 
                               'HR_FOOT': 'RR_force_site', 
                               'Link6': 'EF_force_site'}
frame_name_to_sol_map = {'FL_FOOT': 'f_lf', 
                         'FR_FOOT': 'f_rf', 
                         'HL_FOOT': 'f_lh', 
                         'HR_FOOT': 'f_rh', 
                         'Link6': 'f_ee'}
sol_to_force_id_map = {'f_lf': 0, 
                       'f_rf': 3, 
                       'f_lh': 6, 
                       'f_rh': 9, 
                       'f_ee': 12}

for fname in mpc.ee_frame_names:
    measured_forces_dict[fname]  = []
    predicted_forces_dict[fname] = []

# measured_forces_FR = []
desired_forces = []
joint_torques = []
f_des_z = np.array([15.]*Nsim) 

# Set ground friction in Mujoco
setGroundFriction(robot.model, robot.data, MU)

# robot.model.geom('floor').solref = [0.031, 1.]

# CUTOFF_2 = 50 
# butter2_Fx_ee = LPFButterOrder2(fc=CUTOFF_2, fs=1./1e-3)
# butter2_Fy_ee = LPFButterOrder2(fc=CUTOFF_2, fs=1./1e-3)
# butter2_Fz_ee = LPFButterOrder2(fc=CUTOFF_2, fs=1./1e-3)
# force_est_butter2 = np.zeros(3)
# force_est_butter2_ = [] 

MPC_FREQ    = 500
SIM_FREQ    = int(1./robot.dt)


# Main simulation loop
f_mea_all = np.zeros(3*5)
for i in range(Nsim):
    print("Step ", i)
    # set the force setpoint
    f_des_3d = np.array([-f_des_z[i], 0, 0])
    desired_forces.append(f_des_3d)
    # Get state from simulation
    state = robot.getJointStates()
    q = state['q']
    dq = state['dq']
    t, quat = robot.getPose()
    v = robot.data.qvel[:3]
    omega = robot.data.qvel[3:6]
    q = np.hstack([q, np.zeros(2)])
    dq = np.hstack([dq, np.zeros(2)])
    # Measure forces
    for fname in mpc.ee_frame_names:
        f_mea = -getForceSensor(robot.model, robot.data, frame_name_to_mujoco_sensor[fname]).squeeze().copy()
        # if(fname !='Link6'):
        #     print("filter")
        #     # Filter force using butterworth LPF
        #     force_est_butter2[0] = butter2_Fx_ee.filter(f_mea[0])
        #     force_est_butter2[1] = butter2_Fy_ee.filter(f_mea[1])
        #     force_est_butter2[2] = butter2_Fz_ee.filter(f_mea[2])
            # print(fname, " : ", f_mea)
            # if(MU * f_mea[2] - np.sqrt(f_mea[0]*f_mea[0] + f_mea[1]*f_mea[1]) > 0):
            #     print(fname, " is slipping !")
        #     print(force_est_butter2)
        #     force_est_butter2_.append(force_est_butter2)
        #     # apply filter on feedback
        #     f_mea = force_est_butter2.copy()
        measured_forces_dict[fname].append(f_mea)
        id_f = sol_to_force_id_map[frame_name_to_sol_map[fname]]
        f_mea_all[id_f:id_f+3] = f_mea

        # print("Extract name ", fname, " , Mujoco sensor = ", frame_name_to_mujoco_sensor[fname], " id = ", id_f, " sol name = ", frame_name_to_sol_map[fname] )
        # print("value = ", f_mea)
    # Solve OCP
    if(i%int(SIM_FREQ/MPC_FREQ)==0):
        solution = mpc.updateAndSolve(t, quat, q, v, omega, dq, f_mea_all)
        # if(mpc.solver.KKT > 1e-4):
        #     fheor
    for fname in mpc.ee_frame_names:
        predicted_forces_dict[fname].append(solution[frame_name_to_sol_map[fname]])
    # Save the solution
    q = solution['q']
    dq = solution['dq']
    tau = solution['tau'].squeeze()
    kp = np.ones(18)*0.
    kv = np.ones(18)*0.
    # Step the physics
    # for j in range(int(mpc.dt//robot.dt)):
    robot.setCommands(q, dq, kp, kv, tau)
    robot.step()

# measured_forces = np.array(measured_forces)
desired_forces = np.array(desired_forces)
joint_torques = np.array(joint_torques)
for fname in mpc.ee_frame_names:
    measured_forces_dict[fname] = np.array(measured_forces_dict[fname])
    predicted_forces_dict[fname] = np.array(predicted_forces_dict[fname])
# force_est_butter2_ = np.array(force_est_butter2_)

# Save data 
np.savez_compressed('/tmp/soft_go2',
                    joint_torques=joint_torques,
                    measured_forces=measured_forces_dict,
                    desired_forces=desired_forces,
                    predicted_forces=predicted_forces_dict)
print("Saved MPC simulation data to /tmp/soft_go2")

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

# axs[0].plot(time_span, force_est_butter2_[:,0],linewidth=4, color='g', marker='o',  label="Fx mea (filtered)")
# axs[1].plot(time_span, force_est_butter2_[:,1],linewidth=4, color='g', marker='o',  label="Fy mea (filtered)")
# axs[2].plot(time_span, force_est_butter2_[:,2],linewidth=4, color='g', marker='o',  label="Fz mea (filtered)")
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


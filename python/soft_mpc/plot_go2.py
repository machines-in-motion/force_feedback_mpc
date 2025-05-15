
import numpy as np
import os
from force_feedback_mpc.core_mpc_utils.path_utils import load_yaml_file
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "/home/skleff/force_feedback_ws")

TYPE        = 'soft' # classical or soft

if(TYPE == 'classical'):
    from demos.go2arm.Go2MPC_wrapper_classical import Go2MPCClassical as Go2MPCWrapper
    DATA_PATH    = '/home/skleff/go2_classical_INT=False_Fmin=25_Fmax=80_maxit=1000_fweight=0.0005.npz'
    CONFIG_PATH  = '/home/skleff/force_feedback_ws/force_feedback_mpc/demos/go2arm/Go2MPC_demo_classical.yml'
else:
    from force_feedback_mpc.soft_mpc.Go2MPC_wrapper_soft import Go2MPCSoft as Go2MPCWrapper
    DATA_PATH    = '/home/skleff/go2_soft_Fmax=80_maxit=1000_fweight=0.001_CONSTANT.npz'
    CONFIG_PATH  = '/home/skleff/force_feedback_ws/force_feedback_mpc/python/soft_mpc/Go2MPC_demo_soft.yml'

print("Loading data from: ", DATA_PATH)
print("Loading config from: ", CONFIG_PATH)

# Load data and extract signals
data                 = np.load(DATA_PATH, allow_pickle=True)
jointPos             = data['jointPos']
jointVel             = data['jointVel']
gap_norm             = data['gap_norm']
constraint_norm      = data['constraint_norm']
kkt_norm             = data['kkt_norm']
joint_torques        = data['joint_torques']
measured_forces_dict = data['measured_forces'].item()
# filtered_forces      = data['filtered_forces'].item()
desired_forces       = data['desired_forces']
predicted_forces     = data['predicted_forces'].item()
ee_frame_names       = data['ee_frame_names']
# Load config file 
CONFIG  = load_yaml_file(CONFIG_PATH)
DT_SIMU = CONFIG['DT_SIMU']
N_SIMU  = CONFIG['N_SIMU']
MU      = CONFIG['MU']
MPC_FREQ= CONFIG['MPC_FREQ']


# Compute cost and constraint violation along MPC trajectory
FMIN = CONFIG['FMIN']
FMAX = CONFIG['FMAX']
HORIZON       = CONFIG['HORIZON']
DT_OCP        = CONFIG['DT_OCP']
mpc = Go2MPCWrapper(HORIZON=HORIZON, friction_mu=MU, dt=DT_OCP, USE_MUJOCO=False)
mpc.initialize(FMIN=FMIN)
m = mpc.ocp.runningModels[0]
d = m.createData()
cost = 0
violation = 0
err_f_x = 0.
err_f_y = 0.
err_f_z = 0.
f = np.zeros(15)
for i in range(N_SIMU):
    # print("Stage ", i)
    # Get state
    if(TYPE == 'classical'):
        x = np.hstack([jointPos[i], jointVel[i]])
    else:
        for k,fname in enumerate(ee_frame_names):
            f[3*k:3*(k+1)] = measured_forces_dict[fname][i]
        x = np.hstack([jointPos[i], jointVel[i], f])
    u = joint_torques[i]
    # Compute cost and constaint violation
    m = mpc.ocp.runningModels[0]
    m.calc(d, x, u)
    cost += d.cost
    fric_res_l6 = MU * np.abs(measured_forces_dict['Link6'][i][0]) - np.sqrt(measured_forces_dict['Link6'][i][1]**2 + measured_forces_dict['Link6'][i][2]**2)
    fric_res_FL = MU * np.abs(measured_forces_dict['FL_FOOT'][i][2]) - np.sqrt(measured_forces_dict['FL_FOOT'][i][0]**2 + measured_forces_dict['FL_FOOT'][i][1]**2)
    fric_res_FR = MU * np.abs(measured_forces_dict['FR_FOOT'][i][2]) - np.sqrt(measured_forces_dict['FR_FOOT'][i][0]**2 + measured_forces_dict['FR_FOOT'][i][1]**2)
    fric_res_HL = MU * np.abs(measured_forces_dict['HL_FOOT'][i][2]) - np.sqrt(measured_forces_dict['HL_FOOT'][i][0]**2 + measured_forces_dict['HL_FOOT'][i][1]**2)
    fric_res_HR = MU * np.abs(measured_forces_dict['HR_FOOT'][i][2]) - np.sqrt(measured_forces_dict['HR_FOOT'][i][0]**2 + measured_forces_dict['HR_FOOT'][i][1]**2)
    violation += min(fric_res_l6, 0)
    violation += min(fric_res_FL, 0)
    violation += min(fric_res_FR, 0)
    violation += min(fric_res_HL, 0)
    violation += min(fric_res_HR, 0)
    # cstr_lb = 0
    # cstr_ub = 0
    # if(np.linalg.norm(m.g_lb) < np.inf):
    #     cstr_lb = min(0, np.linalg.norm(d.g - m.g_lb, np.inf)) 
    #     violation += cstr_lb
    # if(np.linalg.norm(m.g_ub) < np.inf):
    #     cstr_ub = max(0, np.linalg.norm(d.g - m.g_ub, np.inf)) 
    #     violation += cstr_ub
    err_f_x += (measured_forces_dict['Link6'][i][0] - desired_forces[i][0])**2
    err_f_y += (measured_forces_dict['Link6'][i][1] - desired_forces[i][1])**2
    err_f_z += (measured_forces_dict['Link6'][i][2] - desired_forces[i][2])**2
# print("Total cost: ", cost)
print("Total constraint violation: ", violation)
print("RMSE F_ee_x = ", np.sqrt(err_f_x/N_SIMU))
print("RMSE F_ee_y = ", np.sqrt(err_f_y/N_SIMU))
print("RMSE F_ee_z = ", np.sqrt(err_f_z/N_SIMU))


# Visualize the measured force against the desired
time_span = np.linspace(0, (N_SIMU-1)*DT_SIMU, N_SIMU)
# EE FORCES
fig, axs = plt.subplots(3, 1, constrained_layout=True)
# Fx_lb_mea = (1./MU)*np.sqrt(measured_forces_dict['Link6'][:, 1]**2 + measured_forces_dict['Link6'][:, 1]**2)
# Fx_lb_pred = (1./MU)*np.sqrt(predicted_forces_dict['Link6'][:, 1]**2 + predicted_forces_dict['Link6'][:, 1]**2)
axs[0].plot(time_span, np.abs(measured_forces_dict['Link6'][:,0]),linewidth=4, color='g', marker='o', alpha=0.5, label="Fx mea")
axs[0].plot(time_span, np.abs(desired_forces[:,0]), linewidth=4, color='k', marker='o', alpha=0.25, label="Fx des")
# axs[0].plot(time_span, np.abs(predicted_forces_dict['Link6'][:,0]), linewidth=4, color='b', marker='o', alpha=0.25, label="Fx predicted")
# axs[0].plot(time_span, Fx_lb_mea, '--', linewidth=4, color='k',  alpha=0.5, label="Fx friction constraint (lower bound)")
axs[0].set_ylim(-10., 105)

axs[1].plot(time_span, measured_forces_dict['Link6'][:,1],linewidth=4, color='g', marker='o', alpha=0.5, label="Fy mea")
axs[1].plot(time_span, desired_forces[:,1], linewidth=4, color='k', marker='o', alpha=0.25, label="Fy des")
# axs[1].plot(time_span, predicted_forces_dict['Link6'][:,1], linewidth=4, color='b', marker='o', alpha=0.25, label="Fy predicted")
axs[1].set_ylim(-10., 10)

axs[2].plot(time_span, measured_forces_dict['Link6'][:,2],linewidth=4, color='g', marker='o', alpha=0.5, label="Fz mea")
axs[2].plot(time_span, desired_forces[:,2], linewidth=4, color='k', marker='o', alpha=0.25, label="Fz des")
# axs[2].plot(time_span, predicted_forces_dict['Link6'][:,2], linewidth=4, color='b', marker='o', alpha=0.25, label="Fz predicted")
axs[2].set_ylim(-25., 10)
for i in range(3):
    axs[i].legend()
    axs[i].grid()
fig.suptitle('Contact force at the end-effector', fontsize=16)

# FEET FORCES (measured and predicted, with friction constraint lower bound on Fz)
fig, axs = plt.subplots(3, 4, constrained_layout=True)
for i,fname in enumerate(ee_frame_names[:-1]):
    # x,y
    axs[0, i].plot(time_span, measured_forces_dict[fname][:,0], linewidth=4, color='g', marker='o', alpha=0.5, label="Fx measured")
    # axs[0, i].plot(time_span, predicted_forces_dict[fname][:,0], linewidth=4, color='b', marker='o', alpha=0.25, label="Fx predicted")
    axs[1, i].plot(time_span, measured_forces_dict[fname][:,1], linewidth=4, color='g', marker='o', alpha=0.5, label="Fy measured")
    # axs[1, i].plot(time_span, predicted_forces_dict[fname][:,1], linewidth=4, color='b', marker='o', alpha=0.25, label="Fy predicted")
    axs[0, i].legend()
    # axs[0, i].title(fname)
    axs[0, i].grid()
    axs[1, i].legend()
    axs[1, i].grid()

    # z
    Fz_lb_mea = (1./MU)*np.sqrt(measured_forces_dict[fname][:, 0]**2 + measured_forces_dict[fname][:, 1]**2)
    # Fz_lb_pred = (1./MU)*np.sqrt(predicted_forces_dict[fname][:, 0]**2 + predicted_forces_dict[fname][:, 1]**2)
    axs[2, i].plot(time_span, measured_forces_dict[fname][:,2], linewidth=4, color='g', marker='o', alpha=0.5, label="Fz measured")
    # axs[2, i].plot(time_span, predicted_forces_dict[fname][:,2], linewidth=4, color='b', marker='o', alpha=0.25, label="Fz predicted")
    axs[2, i].plot(time_span, Fz_lb_mea, '--', linewidth=4, color='k',  alpha=0.5, label="Fz friction constraint (lower bound)")
    # axs[2, i].plot(time_span, Fz_lb_pred, '--', linewidth=4, color='b', alpha=0.2, label="Fz friction lb (pred)")
    axs[2, i].legend()
    axs[2, i].grid()

fig.suptitle('Contact forces at feet FL, FR, HL, HR', fontsize=16)


# SOLVER METRICS
N_MPC_STEPS = int(N_SIMU*DT_SIMU*MPC_FREQ)
time_span2 = np.linspace(0, (N_SIMU-1)*DT_SIMU, N_MPC_STEPS)
# fig, axs = plt.subplots(3, 1, constrained_layout=True)
# axs[0].plot(time_span, gap_norm,linewidth=4, color='g', marker='o', alpha=0.5, label="Gap norm")
# # axs[0].plot(time_span, np.abs(desired_forces[:,0]), linewidth=4, color='k', marker='o', alpha=0.25, label="Fx des")
# # axs[0].set_ylim(0., 105)

# axs[1].plot(time_span, constraint_norm, linewidth=4, color='g', marker='o', alpha=0.5, label="Constraint norm")
# # axs[1].plot(time_span, desired_forces[:,1], linewidth=4, color='k', marker='o', alpha=0.25, label="Fy des")
# # axs[1].set_ylim(-10., 10)

# axs[2].plot(time_span, kkt_norm,linewidth=4, color='g', marker='o', alpha=0.5, label="KKT")
# axs[2].plot(time_span, np.array([1e-4]*N_MPC_STEPS), linewidth=4, color='k', marker='o', alpha=0.25, label="TOL")
# # axs[2].set_ylim(-25., 10)
# for i in range(3):
#     axs[i].legend()
#     axs[i].grid()
# fig.suptitle('Solver convergence', fontsize=16)

fig, axs = plt.subplots(2, 1, constrained_layout=True)
axs[0].plot(time_span2, gap_norm + constraint_norm, linewidth=4, color='g', marker='o', alpha=0.5, label="Constraint norm")
# axs[0].set_ylim(0., 105)
axs[1].plot(time_span, np.abs(measured_forces_dict['Link6'][:,0]), linewidth=4, color='g', marker='o', alpha=0.5, label="Measured force (Fx)")
axs[1].plot(time_span, np.abs(desired_forces[:,0]), linewidth=4, color='k', marker='o', alpha=0.25, label="Desired force (Fx)")
axs[1].set_ylim(-1., FMAX+1)
for i in range(2):
    axs[i].legend()
    axs[i].grid()
fig.suptitle('Constraint violation + Force tracking', fontsize=16)



plt.show()


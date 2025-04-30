
import numpy as np
import os
from force_feedback_mpc.core_mpc_utils.path_utils import load_yaml_file
import matplotlib.pyplot as plt

# DATA_PATH    = '/home/skleff/go2_classical_INT=True.npz'
DATA_PATH    = '/home/skleff/go2_classical_INT=False.npz'
# DATA_PATH    = '/home/skleff/go2_soft.npz'

# CONFIG_PATH  = '/home/skleff/force_feedback_ws/force_feedback_mpc/python/soft_mpc/Go2MPC_demo_soft.yml'
CONFIG_PATH  = '/home/skleff/force_feedback_ws/force_feedback_mpc/demos/go2arm/Go2MPC_demo_classical.yml'
print("Loading data from: ", DATA_PATH)
print("Loading config from: ", CONFIG_PATH)

# Load data and extract signals
data = np.load(DATA_PATH, allow_pickle=True)
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
axs[0].set_ylim(-10., 75)

axs[1].plot(time_span, measured_forces_dict['Link6'][:,1],linewidth=4, color='g', marker='o', alpha=0.5, label="Fy mea")
axs[1].plot(time_span, desired_forces[:,1], linewidth=4, color='k', marker='o', alpha=0.25, label="Fy des")
# axs[1].plot(time_span, predicted_forces_dict['Link6'][:,1], linewidth=4, color='b', marker='o', alpha=0.25, label="Fy predicted")
axs[1].set_ylim(-10., 10)

axs[2].plot(time_span, measured_forces_dict['Link6'][:,2],linewidth=4, color='g', marker='o', alpha=0.5, label="Fz mea")
axs[2].plot(time_span, desired_forces[:,2], linewidth=4, color='k', marker='o', alpha=0.25, label="Fz des")
# axs[2].plot(time_span, predicted_forces_dict['Link6'][:,2], linewidth=4, color='b', marker='o', alpha=0.25, label="Fz predicted")
axs[2].set_ylim(-10., 10)
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

plt.show()


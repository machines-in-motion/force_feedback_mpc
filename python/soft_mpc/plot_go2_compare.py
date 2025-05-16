
import numpy as np
import os
from force_feedback_mpc.core_mpc_utils.path_utils import load_yaml_file
from force_feedback_mpc.core_mpc_utils.misc_utils import moving_average
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "/home/skleff/force_feedback_ws")


# F=50 
#   MAXIT=20 
#       SQP_TOL=1e-4, FWEIGHT=0.05 both classical and soft
# DATA_PATH_1  = '/home/skleff/GO2_DATA/F=50/maxit=20/go2_classical_INT=False_Fmax=50_maxit=20_fweight=0.05.npz'
# DATA_PATH_2  = '/home/skleff/GO2_DATA/F=50/maxit=20/go2_classical_INT=True_Fmax=50_maxit=20_fweight=0.05.npz' 
# DATA_PATH_3  = '/home/skleff/GO2_DATA/F=50/maxit=20/go2_soft_Fmax=50_maxit=20_fweight=0.05.npz' 
# # # # # # # # # # # # # # # # # # 
# BEST WITH ALL 3 NOT BLOWING UP  #
# # # # # # # # # # # # # # # # # # 
#   MAXIT=1000 
#       SQP_TOL=1e-4, FWEIGHT=0.05 both classical and soft
# DATA_PATH_1  = '/home/skleff/GO2_DATA/F=50/maxit=1000/go2_classical_INT=False_Fmax=50_maxit=1000_fweight=0.05.npz'
# DATA_PATH_2  = '/home/skleff/GO2_DATA/F=50/maxit=1000/go2_classical_INT=True_Fmax=50_maxit=1000_fweight=0.05.npz' #ki=1 #'/home/skleff/go2_classical_INT=True_Fmax=50_maxit=1000_fweight=0.05.npz' #ki=0.5
# DATA_PATH_3  = '/home/skleff/GO2_DATA/F=50/maxit=1000/go2_soft_1747166584.5935109_fweight=5e-2.npz' # 5e-2'/home/skleff/go2_soft_1747164585.2440019.npz' #2e-2 #'/home/skleff/go2_soft_1747161797.710107.npz' #1e-2

# F=100 
#   MAXIT=20
#       SQP_TOL=1e-4, FWEIGHT=0.05 both classical and soft
# DATA_PATH_1  = '/home/skleff/GO2_DATA/F=100/maxit=20/go2_classical_INT=False_Fmax=100_maxit=20_fweight=0.05.npz'
# DATA_PATH_2  = '/home/skleff/GO2_DATA/F=100/maxit=20/go2_classical_INT=True_Fmax=100_maxit=20_fweight=0.05.npz' 
# DATA_PATH_3  = '/home/skleff/GO2_DATA/F=100/maxit=20/go2_soft_Fmax=100_maxit=20_fweight=0.05.npz' 
# GOOD BINARY RESULT 
#   MAXIT=1000  
#       SQP_TOL=1e-4, FWEIGHT=0.05 both classical and soft
# DATA_PATH_1  = '/home/skleff/GO2_DATA/F=100/maxit=1000/go2_classical_INT=False_Fmax=100_maxit=1000_fweight=0.05.npz'
# DATA_PATH_2  = '/home/skleff/GO2_DATA/F=100/maxit=1000/go2_classical_INT=True_Fmax=100_maxit=1000_fweight=0.05.npz' 
# DATA_PATH_3  = '/home/skleff/GO2_DATA/F=100/maxit=1000/go2_soft_Fmax=100_maxit=1000_fweight=0.05.npz' 


# # # # # # # # # # # # # # # # # # # #
# BEST BINARY RESULT (go2 flies away) # 
# # # # # # # # # # # # # # # # # # # #  
# # F=80 
# #   MAXIT=1000  
# #       SQP_TOL=1e-4, FWEIGHT=0.05 both classical and soft
# DATA_PATH_1  = '/home/skleff/GO2_DATA/F=80/go2_classical_INT=False_Fmax=80_maxit=1000_fweight=0.05.npz'
# DATA_PATH_2  = '/home/skleff/GO2_DATA/F=80/go2_classical_INT=True_Fmax=80_maxit=1000_fweight=0.05.npz' 
# DATA_PATH_3  = '/home/skleff/GO2_DATA/F=80/go2_soft_Fmax=80_maxit=1000_fweight=0.05.npz' 

# # NEW DATASET (with convergence, ramp 25 to 80, removed ee friction cone)
# DATA_PATH_1  = '/home/skleff/go2_classical_INT=False_Fmax=80_maxit=1000_fweight=0.0005.npz'
# DATA_PATH_2  = '/home/skleff/go2_classical_INT=True_Fmax=80_maxit=1000_fweight=0.0005.npz' 
# DATA_PATH_3  = '/home/skleff/go2_classical_INT=False_Fmin=25_Fmax=80_maxit=1000_fweight=0.0005.npz' 
# # DATA_PATH_3  = '/home/skleff/go2_soft_Fmax=80_maxit=1000_fweight=0.005.npz'
# # DATA_PATH_3  = '/home/skleff/go2_soft_Fmax=80_maxit=1000_fweight=0.001.npz'# violation > 0
# # DATA_PATH_3  = '/home/skleff/go2_soft_Fmax=80_maxit=1000_fweight=0.0008.npz'# TODO?
# # DATA_PATH_3  = '/home/skleff/go2_soft_Fmax=80_maxit=1000_fweight=0.0005.npz' # violation = 0 but poor tracking
# # DATA_PATH_3  = '/home/skleff/go2_soft_Fmax=80_maxit=1000_fweight=0.0001.npz' 

# # NEW DATASET (with convergence, ramp 25 to 80, removed ee friction cone, using foot force cost)
# DATA_PATH_1  = '/home/skleff/go2_classical_INT=False_Fmin=25_Fmax=80_maxit=1000_fweight=0.0005.npz'
# DATA_PATH_2  = '/home/skleff/go2_classical_INT=True_Fmin=25_Fmax=80_maxit=1000_fweight=0.0005.npz' 
# DATA_PATH_3  = '/home/skleff/go2_soft_Fmin=25_Fmax=80_maxit=1000_fweight=0.001.npz' #go2_soft_Fmin=25_Fmax=80_maxit=1000_fweight=0.001.npz' 

# NEW DATASET (with convergence , constant 80N, removed ee friction cone, tol=1e-2
DATA_PATH_1  = '/home/skleff/go2_classical_INT=False_Fmin=80_Fmax=80_maxit=1000_fweight=0.0005.npz'
DATA_PATH_2  = '/home/skleff/go2_classical_INT=True_Fmin=25_Fmax=80_maxit=1000_fweight=0.0005.npz' 
DATA_PATH_3  = '/home/skleff/go2_soft_Fmin=80_Fmax=80_maxit=1000_fweight=0.00025_tol=1e-3.npz'


# Load data and extract signals
print("Loading data from: ", DATA_PATH_1)
data1                 = np.load(DATA_PATH_1, allow_pickle=True)
print("Loading data from: ", DATA_PATH_2)
data2                 = np.load(DATA_PATH_2, allow_pickle=True)
print("Loading data from: ", DATA_PATH_3)
data3                 = np.load(DATA_PATH_3, allow_pickle=True)
measured_forces_dict1 = data1['measured_forces'].item()
measured_forces_dict2 = data2['measured_forces'].item()
measured_forces_dict3 = data3['measured_forces'].item()
desired_forces        = data1['desired_forces']

# Load config file 
CONFIG_PATH  = '/home/skleff/force_feedback_ws/force_feedback_mpc/demos/go2arm/Go2MPC_demo_classical.yml'
print("Loading config from: ", CONFIG_PATH)
CONFIG  = load_yaml_file(CONFIG_PATH)
DT_SIMU = CONFIG['DT_SIMU']
N_SIMU  = CONFIG['N_SIMU']
MU      = CONFIG['MU']
MPC_FREQ= CONFIG['MPC_FREQ']
FMIN          = CONFIG['FMIN']
FMAX          = CONFIG['FMAX']
HORIZON       = CONFIG['HORIZON']
DT_OCP        = CONFIG['DT_OCP']
N_MPC_STEPS = int(N_SIMU*DT_SIMU*MPC_FREQ)


# Compute performance statistics
err_f_norm1 = np.linalg.norm(measured_forces_dict1['Link6'] - desired_forces, axis=1)
err_f_norm2 = np.linalg.norm(measured_forces_dict2['Link6'] - desired_forces, axis=1)
err_f_norm3 = np.linalg.norm(measured_forces_dict3['Link6'] - desired_forces, axis=1)

print("Classical MPC")
print("  RMSE Fx-Fxdes = ", np.sum((measured_forces_dict1['Link6'][:,0] - desired_forces[:,0])**2, axis=0)/N_SIMU)
print("  RMSE Fy-Fydes = ", np.sum((measured_forces_dict1['Link6'][:,1] - desired_forces[:,1])**2, axis=0)/N_SIMU)
print("  RMSE Fz-Fzdes = ", np.sum((measured_forces_dict1['Link6'][:,2] - desired_forces[:,2])**2, axis=0)/N_SIMU)
print("  RMSE ||F-Fdes|| = ", np.sum(err_f_norm1**2)/N_SIMU)

print("Classical MPC + Integral")
print("  RMSE Fx-Fxdes = ", np.sum((measured_forces_dict2['Link6'][:,0] - desired_forces[:,0])**2, axis=0)/N_SIMU)
print("  RMSE Fy-Fydes = ", np.sum((measured_forces_dict2['Link6'][:,1] - desired_forces[:,1])**2, axis=0)/N_SIMU)
print("  RMSE Fz-Fzdes = ", np.sum((measured_forces_dict2['Link6'][:,2] - desired_forces[:,2])**2, axis=0)/N_SIMU)
print("  RMSE ||F-Fdes|| = ", np.sum(err_f_norm2**2)/N_SIMU)

print("Force-feedback MPC")
print("  RMSE Fx-Fxdes = ", np.sum((measured_forces_dict3['Link6'][:,0] - desired_forces[:,0])**2, axis=0)/N_SIMU)
print("  RMSE Fy-Fydes = ", np.sum((measured_forces_dict3['Link6'][:,1] - desired_forces[:,1])**2, axis=0)/N_SIMU)
print("  RMSE Fz-Fzdes = ", np.sum((measured_forces_dict3['Link6'][:,2] - desired_forces[:,2])**2, axis=0)/N_SIMU)
print("  RMSE ||F-Fdes|| = ", np.sum(err_f_norm3**2)/N_SIMU)

time_span = np.linspace(0, (N_SIMU-1)*DT_SIMU, N_SIMU)
time_span2 = np.linspace(0, (N_SIMU-1)*DT_SIMU, N_MPC_STEPS)

LABELS = ['Classical', 'Classical + Integral', 'Force-Feedback']
COLORS = ['b', 'r', 'g']

fig, axs = plt.subplots(3, 1, constrained_layout=True)
axs[0].plot(time_span2, data1['gap_norm'] + data1['constraint_norm'], linewidth=8, color=COLORS[0], marker='o', alpha=0.5, label=LABELS[0])
axs[0].plot(time_span2, data2['gap_norm'] + data2['constraint_norm'], linewidth=8, color=COLORS[1], marker='o', alpha=0.5, label=LABELS[1])
axs[0].plot(time_span2, data3['gap_norm'] + data3['constraint_norm'], linewidth=8, color=COLORS[2], marker='o', alpha=0.5, label=LABELS[2])
axs[0].tick_params(axis = 'x', labelsize=22)
axs[0].tick_params(axis = 'y', labelsize=22)
axs[0].set_xlabel('Time (s)', fontsize=22)
axs[0].set_ylabel('Constraint norm', fontsize=22)
# axs[0].yaxis.set_major_locator(plt.MaxNLocator(4))
# axs[0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))
axs[0].grid(True)
axs[0].set_yscale('log')
axs[0].set_title("Constraint norm", fontdict={'size': 30})

axs[1].plot(time_span2, data1['kkt_norm'], linewidth=8, color=COLORS[0], marker='o', alpha=0.5, label=LABELS[0])
axs[1].plot(time_span2, data2['kkt_norm'], linewidth=8, color=COLORS[1], marker='o', alpha=0.5, label=LABELS[1])
axs[1].plot(time_span2, data3['kkt_norm'], linewidth=8, color=COLORS[2], marker='o', alpha=0.5, label=LABELS[2])
axs[1].tick_params(axis = 'x', labelsize=22)
axs[1].tick_params(axis = 'y', labelsize=22)
axs[1].set_xlabel('Time (s)', fontsize=22)
axs[1].set_ylabel('KKT residual norm', fontsize=22)
# axs[1].yaxis.set_major_locator(plt.MaxNLocator(4))
# axs[1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))
axs[1].grid(True)
axs[1].set_yscale('log')
axs[1].set_title("KKT residual norm", fontdict={'size': 30})

# axs[0].set_ylim(0., 105)
# Plot Fx
axs[2].plot(time_span, np.abs(measured_forces_dict1['Link6'][:,0]), linewidth=8, color=COLORS[0], marker='o', alpha=0.5, label=LABELS[0])
axs[2].plot(time_span, np.abs(measured_forces_dict2['Link6'][:,0]), linewidth=8, color=COLORS[1], marker='o', alpha=0.5, label=LABELS[1])
axs[2].plot(time_span, np.abs(measured_forces_dict3['Link6'][:,0]), linewidth=8, color=COLORS[2], marker='o', alpha=0.5, label=LABELS[2])
axs[2].plot(time_span, np.abs(desired_forces[:,0]), linewidth=4, color='k', marker='o', alpha=0.25, label="Desired force (Fx)")
axs[2].set_ylim(-1., FMAX+1)
# # Cumulative RMSE of the force error norm
# axs[2].plot(time_span, np.sqrt(np.cumsum(err_f_norm1)), linewidth=4, color=COLORS[0], marker='o', alpha=0.7, label=f"Force error norm {LABELS[0]}")
# axs[2].plot(time_span, np.sqrt(np.cumsum(err_f_norm2)), linewidth=4, color=COLORS[1], marker='o', alpha=0.7, label=f"Force error norm {LABELS[1]}")
# axs[2].plot(time_span, np.sqrt(np.cumsum(err_f_norm3)), linewidth=4, color=COLORS[2], marker='o', alpha=0.7, label=f"Force error norm {LABELS[2]}")
# # RMSE of the force error norm
# axs[2].plot(time_span, moving_average(err_f_norm1, window_size=5), linewidth=4, color=COLORS[0], marker='o', alpha=0.7, label=f"Force error norm {LABELS[0]}")
# axs[2].plot(time_span, moving_average(err_f_norm2, window_size=5), linewidth=4, color=COLORS[1], marker='o', alpha=0.7, label=f"Force error norm {LABELS[1]}")
# axs[2].plot(time_span, moving_average(err_f_norm3, window_size=5), linewidth=4, color=COLORS[2], marker='o', alpha=0.7, label=f"Force error norm {LABELS[2]}")
axs[2].tick_params(axis = 'x', labelsize=22)
axs[2].tick_params(axis = 'y', labelsize=22)
axs[2].set_xlabel('Time (s)', fontsize=22)
axs[2].set_ylabel('Force error norm', fontsize=22)
axs[2].yaxis.set_major_locator(plt.MaxNLocator(4))
axs[2].yaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))
axs[2].grid(True)
axs[2].set_title("Force error norm", fontdict={'size': 30})

handles_f, labels_f = axs[0].get_legend_handles_labels()
fig.legend(handles_f, labels_f, loc='upper left', prop={'size': 26})
fig.align_ylabels()

plt.show()


import sys
import analysis_utils
from core_mpc_utils import data_utils
import numpy as np
import matplotlib.pyplot as plt

from core_mpc_utils.misc_utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, log_level_name=GLOBAL_LOG_LEVEL, USE_LONG_FORMAT=GLOBAL_LOG_FORMAT).logger


# tilt table of several angles around y-axis
TILT_ANGLES_DEG = [-20, -10, -5, 5, 10, 20] 
TILT_RPY = []
for angle in TILT_ANGLES_DEG:
    TILT_RPY.append([0., angle*np.pi/180, 0.])
N_EXP = len(TILT_RPY)


PREFIX = '/home/skleff/force-feedback/data/'
prefix_lpf = PREFIX+'iiwa_LPF_contact_circle_MPC_bullet__BIAS=True_NOISE=True_DELAY=True_Fp=0.25_Fc=0.5_Fs1.0'
# prefix =  PREFIX+'iiwa_contact_circle_MPC_bullet__BIAS=True_NOISE=True_DELAY=True_Fp=0.5_Fc=0.5_Fs2.0'
prefix =  PREFIX+'iiwa_contact_circle_MPC_bullet__BIAS=True_NOISE=True_DELAY=True_Fp=0.5_Fc=0.5_Fs1.0'

def main(FILTER=1, PLOT=False):
  
    lin_err_ee_xy_avg = np.zeros((2,N_EXP))
    lin_err_ee_xy_avg_NORM = np.zeros(N_EXP)
    f_ee_err_avg_z = np.zeros(N_EXP)
    f_ee_err_max_z = np.zeros(N_EXP)

    lin_err_ee_xy_avg_LPF = np.zeros((2,N_EXP))
    lin_err_ee_xy_avg_LPF_NORM = np.zeros(N_EXP)
    f_ee_err_avg_z_LPF = np.zeros(N_EXP)
    f_ee_err_max_z_LPF = np.zeros(N_EXP)

    # Compute errors 
    for n_exp in range(N_EXP):


        # Extract data
        print("Extracting data...")
        data = data_utils.extract_plot_data_from_npz(prefix+'_EXP_TILT='+str(TILT_ANGLES_DEG[n_exp])+'.npz', LPF=False)    
        # Compute absolute tracking errors |mea - ref|
        # EE tracking
        Np = data['N_plan'] ; Ns = data['N_simu']
        lin_pos_ee_ref = analysis_utils.linear_interpolation(data['lin_pos_ee_ref'], int((Ns+1)/Np))
        lin_err_ee_xy = np.zeros( (data['lin_pos_ee_mea'].shape[0], 2) )
        for i in range( lin_pos_ee_ref.shape[0] ):
            lin_err_ee_xy[i,:] = np.abs( data['lin_pos_ee_mea'][i,:2] - lin_pos_ee_ref[i,:2])
        # Average absolute error 
        lin_err_ee_xy_avg[:,n_exp] = np.sum(lin_err_ee_xy, axis=0) / Ns
        lin_err_ee_xy_avg_NORM[n_exp] = np.linalg.norm(lin_err_ee_xy_avg[:,n_exp])
        # Force tracking
        Np = data['N_plan'] ; Ns = data['N_simu']
        f_ee_ref_z = -20
        f_ee_err_z = np.zeros(data['f_ee_mea'].shape[0])
        for i in range( Ns ):
            f_ee_err_z[i] = np.abs( data['f_ee_mea'][i,2] - f_ee_ref_z)
        # Maximum (peak) absolute error along x,y,z
        f_ee_err_max_z[n_exp]   = np.max(f_ee_err_z)
        # Average absolute error 
        f_ee_err_avg_z[n_exp] = np.sum(f_ee_err_z, axis=0) / Ns
        # Smooth if necessary
        if(FILTER > 0):
            data['q_mea'] = analysis_utils.moving_average_filter(data['q_mea'].copy(), FILTER)
            data['v_mea'] = analysis_utils.moving_average_filter(data['v_mea'].copy(), FILTER)
            data['lin_pos_ee_mea'] = analysis_utils.moving_average_filter(data['lin_pos_ee_mea'].copy(), FILTER)
            data['ang_pos_ee_mea'] = analysis_utils.moving_average_filter(data['ang_pos_ee_mea'].copy(), FILTER)
            data['lin_vel_ee_mea'] = analysis_utils.moving_average_filter(data['lin_vel_ee_mea'].copy(), FILTER)
            data['ang_vel_ee_mea'] = analysis_utils.moving_average_filter(data['ang_vel_ee_mea'].copy(), FILTER)
            data['f_ee_mea']   = analysis_utils.moving_average_filter(data['f_ee_mea'].copy(), FILTER) 
        
        
        
        
        # Extract data and compute errors
        print("Extracting data LPF...")
        data = data_utils.extract_plot_data_from_npz(prefix_lpf+'_EXP_TILT='+str(TILT_ANGLES_DEG[n_exp])+'.npz', LPF=True)    
        Np = data['N_plan'] ; Ns = data['N_simu']
        lin_pos_ee_ref = analysis_utils.linear_interpolation(data['lin_pos_ee_ref'], int((Ns+1)/Np))
        lin_err_ee_xy = np.zeros( (data['lin_pos_ee_mea'].shape[0], 2) )
        for i in range( lin_pos_ee_ref.shape[0] ):
            lin_err_ee_xy[i,:] = np.abs( data['lin_pos_ee_mea'][i,:2] - lin_pos_ee_ref[i,:2])
        # Average absolute error 
        lin_err_ee_xy_avg_LPF[:,n_exp] = np.sum(lin_err_ee_xy, axis=0) / Ns
        lin_err_ee_xy_avg_LPF_NORM[n_exp] = np.linalg.norm(lin_err_ee_xy_avg_LPF[:,n_exp])
        # Force tracking
        Np = data['N_plan'] ; Ns = data['N_simu']
        f_ee_ref_z = -20
        f_ee_err_z = np.zeros(data['f_ee_mea'].shape[0])
        for i in range( Ns ):
            f_ee_err_z[i] = np.abs( data['f_ee_mea'][i,2] - f_ee_ref_z)
        # Maximum (peak) absolute error along x,y,z
        f_ee_err_max_z_LPF[n_exp]   = np.max(f_ee_err_z)
        # Average absolute error 
        f_ee_err_avg_z_LPF[n_exp] = np.sum(f_ee_err_z, axis=0) / Ns
        # Smooth if necessary
        if(FILTER > 0):
            data['q_mea'] = analysis_utils.moving_average_filter(data['q_mea'].copy(), FILTER)
            data['v_mea'] = analysis_utils.moving_average_filter(data['v_mea'].copy(), FILTER)
            data['tau_mea'] = analysis_utils.moving_average_filter(data['tau_mea'].copy(), FILTER)
            data['lin_pos_ee_mea'] = analysis_utils.moving_average_filter(data['lin_pos_ee_mea'].copy(), FILTER)
            data['ang_pos_ee_mea'] = analysis_utils.moving_average_filter(data['ang_pos_ee_mea'].copy(), FILTER)
            data['lin_vel_ee_mea'] = analysis_utils.moving_average_filter(data['lin_vel_ee_mea'].copy(), FILTER)
            data['ang_vel_ee_mea'] = analysis_utils.moving_average_filter(data['ang_vel_ee_mea'].copy(), FILTER)
            data['f_ee_mea']   = analysis_utils.moving_average_filter(data['f_ee_mea'].copy(), FILTER) 
        print("----------------------------------")    





    # Plot errors

    fig1, ax1 = plt.subplots(1, 1, figsize=(19.2,10.8)) # Avg position err x,y,z
    fig2, ax2 = plt.subplots(2, 1, figsize=(19.2,10.8)) # Avg force err z + max force z
    xyz = ['x','y','z']
    color_LPF = 'r'
    color = 'b'

    # ax1[0].plot(TILT_ANGLES_DEG, lin_err_ee_xy_avg[0, :], color=color, linestyle='-', linewidth=3, label='Classical MPC')
    # ax1[0].plot(TILT_ANGLES_DEG, lin_err_ee_xy_avg_LPF[0, :], color=color_LPF, linestyle='-', linewidth=3, label='Force feedback MPC')
    
    # ax1[1].plot(TILT_ANGLES_DEG, lin_err_ee_xy_avg[1, :], color=color, linestyle='-', linewidth=3)
    # ax1[1].plot(TILT_ANGLES_DEG, lin_err_ee_xy_avg_LPF[1, :], color=color_LPF, linestyle='-', linewidth=3)

    ax1.plot(TILT_ANGLES_DEG, lin_err_ee_xy_avg_NORM, color=color, linestyle='-', linewidth=3)
    ax1.plot(TILT_ANGLES_DEG, lin_err_ee_xy_avg_LPF_NORM, color=color_LPF, linestyle='-', linewidth=3)

    # ax1[2].plot(TILT_ANGLES_DEG, lin_err_ee_xy_avg[2, :], color=color, linestyle='-', linewidth=3)
    # ax1[2].plot(TILT_ANGLES_DEG, lin_err_ee_xy_avg_LPF[2, :], color=color_LPF, linestyle='-', linewidth=3)
    
    ax2[0].plot(TILT_ANGLES_DEG, f_ee_err_avg_z, color=color, linestyle='-', linewidth=3, label='Classical MPC')
    ax2[0].plot(TILT_ANGLES_DEG, f_ee_err_avg_z_LPF, color=color_LPF, linestyle='-', linewidth=3, label='Force feedback MPC')
    
    ax2[1].plot(TILT_ANGLES_DEG, f_ee_err_max_z, color=color, linestyle='-', linewidth=3)
    ax2[1].plot(TILT_ANGLES_DEG, f_ee_err_max_z_LPF, color=color_LPF, linestyle='-', linewidth=3)


    # For each experiment plot perf as marker
    for n_exp in range(N_EXP): 
        # # Pos x
        # ax1[0].plot(float(TILT_ANGLES_DEG[n_exp]), lin_err_ee_xy_avg[0, n_exp], 
        #                                             marker='o', markerfacecolor=color, 
        #                                             markersize=16, markeredgecolor='k')
        # ax1[0].plot(float(TILT_ANGLES_DEG[n_exp]), lin_err_ee_xy_avg_LPF[0, n_exp], 
        #                                             marker='s', markerfacecolor=color_LPF, 
        #                                             markersize=16, markeredgecolor='k')
        # ax1[0].set_ylabel('$\Delta P^{EE}_{x}$ (m)', fontsize=26)
        # ax1[0].yaxis.set_major_locator(plt.MaxNLocator(2))
        # ax1[0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.1e'))
        # ax1[0].grid(True) 
        # ax1[0].tick_params(axis = 'y', labelsize=22)
        # ax1[0].xaxis.set_ticklabels([])
        # ax1[0].tick_params(bottom=False)

        # # Pos y
        # ax1[1].plot(float(TILT_ANGLES_DEG[n_exp]), lin_err_ee_xy_avg[1, n_exp], 
        #                                             marker='o', markerfacecolor=color, 
        #                                             markersize=16, markeredgecolor='k')
        # ax1[1].plot(float(TILT_ANGLES_DEG[n_exp]), lin_err_ee_xy_avg_LPF[1, n_exp], 
        #                                             marker='s', markerfacecolor=color_LPF, 
        #                                             markersize=16, markeredgecolor='k')
        
        # ax1[1].set_ylabel('$\Delta P^{EE}_{y}$  (m)', fontsize=26)
        # ax1[1].yaxis.set_major_locator(plt.MaxNLocator(2))
        # ax1[1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.1e'))
        # ax1[1].grid(True) 
        # ax1[1].tick_params(axis = 'y', labelsize=22)
        # ax1[1].set_xlabel('Angle (deg)', fontsize=26)
        # ax1[1].tick_params(axis = 'x', labelsize = 22)


        # NORM
        ax1.plot(float(TILT_ANGLES_DEG[n_exp]), lin_err_ee_xy_avg_NORM[n_exp], 
                                                    marker='o', markerfacecolor=color, 
                                                    markersize=16, markeredgecolor='k')
        ax1.plot(float(TILT_ANGLES_DEG[n_exp]), lin_err_ee_xy_avg_LPF_NORM[n_exp], 
                                                    marker='s', markerfacecolor=color_LPF, 
                                                    markersize=16, markeredgecolor='k')
        
        ax1.set_ylabel('$|| \Delta P^{EE}_{xy} ||$  (m)', fontsize=26)
        ax1.yaxis.set_major_locator(plt.MaxNLocator(2))
        ax1.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1e'))
        ax1.grid(True) 
        ax1.tick_params(axis = 'y', labelsize=22)
        ax1.set_xlabel('Angle (deg)', fontsize=26)
        ax1.tick_params(axis = 'x', labelsize = 22)


        # Avg force err z
        ax2[0].plot(float(TILT_ANGLES_DEG[n_exp]), f_ee_err_avg_z[n_exp], 
                                                     marker='o', markerfacecolor=color,
                                                     markersize=16, markeredgecolor='k')
        ax2[0].plot(float(TILT_ANGLES_DEG[n_exp]), f_ee_err_avg_z_LPF[n_exp],
                                                     marker='s', markerfacecolor=color_LPF,
                                                     markersize=16, markeredgecolor='k')
        ax2[0].set_ylabel('$\Delta \lambda_{z}$  (m)', fontsize=26)
        ax2[0].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax2[0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.1e'))
        ax2[0].grid(True) 
        ax2[0].tick_params(axis = 'y', labelsize=22)
        ax2[0].xaxis.set_ticklabels([])
        ax2[0].tick_params(bottom=False)



        # Max force z
        ax2[1].plot(float(TILT_ANGLES_DEG[n_exp]), f_ee_err_max_z[n_exp], 
                                                     marker='o', markerfacecolor=color,
                                                     markersize=16, markeredgecolor='k')
        ax2[1].plot(float(TILT_ANGLES_DEG[n_exp]), f_ee_err_max_z_LPF[n_exp], 
                                                     marker='s', markerfacecolor=color_LPF,
                                                     markersize=16, markeredgecolor='k')
        ax2[1].set_ylabel('$\lambda^{max}_{z}$  (m)', fontsize=26)
        ax2[1].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax2[1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.1e'))
        ax2[1].grid(True) 
        ax2[1].tick_params(axis = 'y', labelsize=22)
        ax2[1].set_xlabel('Angle (deg)', fontsize=26)
        ax2[1].tick_params(axis = 'x', labelsize = 22)

    # # Legend
    # handles, labels = ax[i].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='best', bbox_to_anchor=(0.4, 0.38, 0.5, 0.5), prop={'size': 26}) 
    # # Label on y axis for all subplots
    # fig.text(0.05, 0.5, 'End-effector position (m)', ha='center', va='center', rotation='vertical', fontsize=26)


    # Legend error
    handles1, labels1 = ax1.get_legend_handles_labels()
    fig1.legend(handles1, labels1, loc='upper right', prop={'size': 26})
    # Legend error norm 
    handles2, labels2 = ax2[0].get_legend_handles_labels()
    fig2.legend(handles2, labels2, loc='upper right', prop={'size': 26})
    # titles
    # fig1.suptitle('End-effector position')
    # fig2.suptitle('Normal force')
    # Save, show , clean
    fig1.savefig('/home/skleff/force-feedback/data//pos_err.png')
    fig2.savefig('/home/skleff/force-feedback/data//force_err.png')
    plt.show()
    plt.close('all')



if __name__=='__main__':
    if len(sys.argv) <= 1:
        print("Usage: python plot_perf_tilt.py [arg1: FILTER (int)]")
        sys.exit(0)
    sys.exit(main(int(sys.argv[1])))



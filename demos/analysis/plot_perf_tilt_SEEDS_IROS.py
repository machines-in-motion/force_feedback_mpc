import sys
from turtle import rt
import analysis_utils
from core_mpc_utils import data_utils, pin_utils
import numpy as np
import matplotlib.pyplot as plt

from core_mpc_utils.misc_utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, log_level_name=GLOBAL_LOG_LEVEL, USE_LONG_FORMAT=GLOBAL_LOG_FORMAT).logger


# tilt table of several angles around y-axis
# TILT_ANGLES_DEG = [-20, -10, -5, 5, 10, 20] 
TILT_ANGLES_DEG = [-20, -15, -10, -5, 0, 5, 10, 15, 20] 

TILT_RPY = []
for angle in TILT_ANGLES_DEG:
    TILT_RPY.append([0., angle*np.pi/180, 0.])
N_EXP = len(TILT_RPY)
# random seeds 
# SEEDS   = [1, 2, 3]
SEEDS = [1, 2, 3, 4, 5]

N_SEEDS = len(SEEDS)
# data sets 
PREFIX     = '/home/skleff/force-feedback/data/'
prefix_lpf = PREFIX + 'iiwa_LPF_contact_circle_MPC_bullet__BIAS=True_NOISE=True_DELAY=True_Fp=0.25_Fc=0.5_Fs1.0'
prefix     = PREFIX + 'iiwa_contact_circle_MPC_bullet__BIAS=True_NOISE=True_DELAY=True_Fp=0.5_Fc=0.5_Fs1.0'

def main(FILTER=1):
    
    p_avg_abs_err_xy_NORM = np.zeros((N_SEEDS, N_EXP))
    f_avg_abs_err_z = np.zeros((N_SEEDS, N_EXP))
    f_abs_max_z = np.zeros((N_SEEDS, N_EXP))
    cycles_not_in_contact = np.zeros((N_SEEDS, N_EXP))

    p_avg_abs_err_xy_NORM_LPF = np.zeros((N_SEEDS, N_EXP))
    f_avg_abs_err_z_LPF = np.zeros((N_SEEDS, N_EXP))
    f_abs_max_z_LPF = np.zeros((N_SEEDS, N_EXP))
    cycles_not_in_contact_LPF = np.zeros((N_SEEDS, N_EXP))

    for n_seed in range(N_SEEDS):
        print("Seed "+str(n_seed) + "/" + str(N_SEEDS))
        # Compute errors 
        for n_exp in range(N_EXP):
            print("  Angle "+ str(n_exp) + "/" + str(N_EXP))

            # Extract data
            print("Extracting data...")
            data = data_utils.extract_plot_data_from_npz(prefix+'_EXP_TILT='+str(TILT_ANGLES_DEG[n_exp])+"_SEED="+str(SEEDS[n_seed])+'.npz', LPF=False)    
            # Compute absolute tracking errors |mea - ref|
            # EE tracking
            Np = data['N_plan'] ; Ns = data['N_simu']
            lin_pos_ee_ref = analysis_utils.linear_interpolation(data['lin_pos_ee_ref'], int((Ns+1)/Np))
            lin_err_ee_xy = np.zeros( (data['lin_pos_ee_mea'].shape[0], 2) )
            for i in range( lin_pos_ee_ref.shape[0] ):
                lin_err_ee_xy[i,:] = np.abs( data['lin_pos_ee_mea'][i,:2] - lin_pos_ee_ref[i,:2])
            # Average absolute error 
            p_avg_abs_err_xy_NORM[n_seed, n_exp] = np.linalg.norm( np.sum(lin_err_ee_xy, axis=0) / Ns )
            # Force tracking
            Np = data['N_plan'] ; Ns = data['N_simu'] 
            f_ee_mea = data['f_ee_mea'] #pin_utils.get_f_(data['q_mea'], data['v_mea'], data['u_des_SIMU'], data['pin_model'], data['pin_model'].getFrameId('contact') ) #
            f_ee_ref_z = -20 ; f_ee_err_z = np.zeros(f_ee_mea.shape[0])
            for i in range( Ns ):
                f_ee_err_z[i] = np.abs( f_ee_mea[i,2] - f_ee_ref_z)
            # Maximum (peak) absolute error along x,y,z
            f_abs_max_z[n_seed, n_exp]   = np.max(f_ee_err_z)
            # Average absolute error 
            f_avg_abs_err_z[n_seed, n_exp] = np.sum(f_ee_err_z, axis=0) / Ns

            bool_contact = np.isclose(f_ee_mea[:,2], np.zeros(f_ee_mea[:,2].shape), rtol=1e-3)
            cycles_not_in_contact[n_seed, n_exp] = (100.*np.count_nonzero(bool_contact))/Ns
            # print(cycles_not_in_contact[n_seed, n_exp])

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
            data = data_utils.extract_plot_data_from_npz(prefix_lpf+'_EXP_TILT='+str(TILT_ANGLES_DEG[n_exp])+"_SEED="+str(SEEDS[n_seed])+'.npz', LPF=True)    
            Np = data['N_plan'] ; Ns = data['N_simu']
            lin_pos_ee_ref = analysis_utils.linear_interpolation(data['lin_pos_ee_ref'], int((Ns+1)/Np))
            lin_err_ee_xy = np.zeros( (data['lin_pos_ee_mea'].shape[0], 2) )
            for i in range( lin_pos_ee_ref.shape[0] ):
                lin_err_ee_xy[i,:] = np.abs( data['lin_pos_ee_mea'][i,:2] - lin_pos_ee_ref[i,:2])
            # Average absolute error 
            p_avg_abs_err_xy_NORM_LPF[n_seed, n_exp] = np.linalg.norm( np.sum(lin_err_ee_xy, axis=0) / Ns )
            # Force tracking
            Np = data['N_plan'] ; Ns = data['N_simu']
            f_ee_ref_z = -20
            f_ee_mea = data['f_ee_mea'] #pin_utils.get_f_(data['q_mea'], data['v_mea'], data['tau_mea'], data['pin_model'], data['pin_model'].getFrameId('contact') )
            f_ee_err_z = np.zeros(f_ee_mea.shape[0])
            for i in range( Ns ):
                f_ee_err_z[i] = np.abs( f_ee_mea[i,2] - f_ee_ref_z)
            # Maximum (peak) absolute error along x,y,z
            f_abs_max_z_LPF[n_seed, n_exp]   = np.max(f_ee_err_z)
            # Average absolute error 
            f_avg_abs_err_z_LPF[n_seed, n_exp] = np.sum(f_ee_err_z, axis=0) / Ns

            bool_contact = np.isclose(f_ee_mea[:,2], np.zeros(f_ee_mea[:,2].shape), rtol=1e-3)
            cycles_not_in_contact_LPF[n_seed, n_exp] = (100.*np.count_nonzero(bool_contact))/Ns
            # print(cycles_not_in_contact_LPF[n_seed, n_exp])

            # import time
            # time.sleep(1000)

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

        # # Collect perfs for current seed
        # p_avg_abs_err_xy_NORM[n_seed, :] = lin_err_ee_xy_avg_NORM
        # f_avg_abs_err_z[n_seed, :]       = f_ee_err_avg_z
        # f_abs_max_z[n_seed, :]           = f_ee_err_max_z

        # p_avg_abs_err_xy_NORM_LPF[n_seed, :] = lin_err_ee_xy_avg_LPF_NORM
        # f_avg_abs_err_z_LPF[n_seed, :]       = f_ee_err_avg_z_LPF
        # f_abs_max_z_LPF[n_seed, :]           = f_ee_err_max_z_LPF


    # Average perfs over seeds
    p_avg_abs_err_xy_NORM_AVG     = np.sum(p_avg_abs_err_xy_NORM, axis=0) / N_SEEDS
    f_avg_abs_err_z_AVG           = np.sum(f_avg_abs_err_z, axis=0) / N_SEEDS
    f_abs_max_z_AVG               = np.sum(f_abs_max_z, axis=0) / N_SEEDS
    p_avg_abs_err_xy_NORM_AVG_LPF = np.sum(p_avg_abs_err_xy_NORM_LPF, axis=0) / N_SEEDS
    f_avg_abs_err_z_AVG_LPF       = np.sum(f_avg_abs_err_z_LPF, axis=0) / N_SEEDS
    f_abs_max_z_AVG_LPF           = np.sum(f_abs_max_z_LPF, axis=0) / N_SEEDS
    cycles_not_in_contact_AVG     = np.sum(cycles_not_in_contact, axis=0) / N_SEEDS
    cycles_not_in_contact_AVG_LPF = np.sum(cycles_not_in_contact_LPF, axis=0) / N_SEEDS

    # Plot 
    fig1, ax1 = plt.subplots(1, 1, figsize=(19.2,10.8)) # Err position 
    fig2, ax2 = plt.subplots(2, 1, figsize=(19.2,10.8)) # Err force + max force 
    fig3, ax3 = plt.subplots(1, 1, figsize=(19.2,10.8)) # Timings

    color_LPF = 'r'
    color = 'b'
    
    
    # POSITION Averages 
    ax1.plot(TILT_ANGLES_DEG, p_avg_abs_err_xy_NORM_AVG, color=color, linestyle='-', linewidth=4)
    ax1.plot(TILT_ANGLES_DEG, p_avg_abs_err_xy_NORM_AVG_LPF, color=color_LPF, linestyle='-', linewidth=4)
    # # For each seed
    # for n_seed in range(N_SEEDS):
    #     ax1.plot(TILT_ANGLES_DEG, p_avg_abs_err_xy_NORM[n_seed,:], color=color, linestyle='-', linewidth=3, alpha=0.3)
    #     ax1.plot(TILT_ANGLES_DEG, p_avg_abs_err_xy_NORM_LPF[n_seed,:], color=color_LPF, linestyle='-', linewidth=3, alpha=0.3)

    # FORCE ERR
    # Average
    ax2[0].plot(TILT_ANGLES_DEG, f_avg_abs_err_z_AVG, color=color, linestyle='-', linewidth=4, label='Classical MPC')
    ax2[0].plot(TILT_ANGLES_DEG, f_avg_abs_err_z_AVG_LPF, color=color_LPF, linestyle='-', linewidth=4, label='Force feedback MPC')

    # FORCE MAX
    # Average 
    ax2[1].plot(TILT_ANGLES_DEG, f_abs_max_z_AVG, color=color, linestyle='-', linewidth=4)
    ax2[1].plot(TILT_ANGLES_DEG, f_abs_max_z_AVG_LPF, color=color_LPF, linestyle='-', linewidth=4)


    # TIMINGS
    ax3.plot(TILT_ANGLES_DEG, cycles_not_in_contact_AVG, color=color, linestyle='-', linewidth=4)
    ax3.plot(TILT_ANGLES_DEG, cycles_not_in_contact_AVG_LPF, color=color_LPF, linestyle='-', linewidth=4)

    # For each experiment plot perf as marker
    for n_exp in range(N_EXP): 

        # POSITON
        ax1.plot(float(TILT_ANGLES_DEG[n_exp]), p_avg_abs_err_xy_NORM_AVG[n_exp], 
                                                    marker='o', markerfacecolor=color, 
                                                    markersize=18, markeredgecolor='k')
        ax1.plot(float(TILT_ANGLES_DEG[n_exp]), p_avg_abs_err_xy_NORM_AVG_LPF[n_exp], 
                                                    marker='s', markerfacecolor=color_LPF, 
                                                    markersize=18, markeredgecolor='k')
        for n_seed in range(N_SEEDS):
            ax1.plot(TILT_ANGLES_DEG[n_exp], p_avg_abs_err_xy_NORM[n_seed,n_exp], marker='o', markerfacecolor=color, markersize=16, alpha=0.3)
            ax1.plot(TILT_ANGLES_DEG[n_exp], p_avg_abs_err_xy_NORM_LPF[n_seed,n_exp], marker='s', markerfacecolor=color_LPF, markersize=16, alpha=0.3)
        ax1.set_ylabel('$|| \Delta P^{EE}_{xy} ||$  (m)', fontsize=30)
        ax1.yaxis.set_major_locator(plt.MaxNLocator(4))
        # ax1.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1e'))
        ax1.grid(True) 
        ax1.tick_params(axis = 'y', labelsize=24)
        ax1.set_xlabel('Angle (deg)', fontsize=30)
        ax1.tick_params(axis = 'x', labelsize = 24)


        # FORCE ERR
        ax2[0].plot(float(TILT_ANGLES_DEG[n_exp]), f_avg_abs_err_z_AVG[n_exp], 
                                                     marker='o', markerfacecolor=color,
                                                     markersize=18, markeredgecolor='k')
        ax2[0].plot(float(TILT_ANGLES_DEG[n_exp]), f_avg_abs_err_z_AVG_LPF[n_exp],
                                                     marker='s', markerfacecolor=color_LPF,
                                                     markersize=18, markeredgecolor='k')
        for n_seed in range(N_SEEDS):
            ax2[0].plot(TILT_ANGLES_DEG[n_exp], f_avg_abs_err_z[n_seed,n_exp], marker='o', markerfacecolor=color, markersize=16, alpha=0.3)
            ax2[0].plot(TILT_ANGLES_DEG[n_exp], f_avg_abs_err_z_LPF[n_seed,n_exp], marker='s', markerfacecolor=color_LPF, markersize=16, alpha=0.3)
        ax2[0].set_ylabel('$\Delta \lambda_{z}$  (N)', fontsize=30)
        ax2[0].yaxis.set_major_locator(plt.MaxNLocator(3))
        # ax2[0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.1e'))
        ax2[0].grid(True) 
        ax2[0].tick_params(axis = 'y', labelsize=24)
        ax2[0].xaxis.set_ticklabels([])
        ax2[0].tick_params(bottom=False)



        # FORCE MAX
        ax2[1].plot(float(TILT_ANGLES_DEG[n_exp]), f_abs_max_z_AVG[n_exp], 
                                                     marker='o', markerfacecolor=color,
                                                     markersize=18, markeredgecolor='k')
        ax2[1].plot(float(TILT_ANGLES_DEG[n_exp]), f_abs_max_z_AVG_LPF[n_exp], 
                                                     marker='s', markerfacecolor=color_LPF,
                                                     markersize=18, markeredgecolor='k')
        for n_seed in range(N_SEEDS):
            ax2[1].plot(TILT_ANGLES_DEG[n_exp], f_abs_max_z[n_seed,n_exp], marker='o', markerfacecolor=color, markersize=16, alpha=0.3)
            ax2[1].plot(TILT_ANGLES_DEG[n_exp], f_abs_max_z_LPF[n_seed,n_exp], marker='s', markerfacecolor=color_LPF, markersize=16, alpha=0.3)
        ax2[1].set_ylabel('$\lambda^{max}_{z}$  (N)', fontsize=30)
        ax2[1].yaxis.set_major_locator(plt.MaxNLocator(3))
        # ax2[1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
        ax2[1].grid(True) 
        ax2[1].tick_params(axis = 'y', labelsize=24)
        ax2[1].set_xlabel('Angle (deg)', fontsize=30)
        ax2[1].tick_params(axis = 'x', labelsize = 24)


        # TIME
        ax3.plot(float(TILT_ANGLES_DEG[n_exp]), cycles_not_in_contact_AVG[n_exp], 
                                                    marker='o', markerfacecolor=color, 
                                                    markersize=18, markeredgecolor='k')
        ax3.plot(float(TILT_ANGLES_DEG[n_exp]), cycles_not_in_contact_AVG_LPF[n_exp], 
                                                    marker='s', markerfacecolor=color_LPF, 
                                                    markersize=18, markeredgecolor='k')
        for n_seed in range(N_SEEDS):
            ax3.plot(TILT_ANGLES_DEG[n_exp], cycles_not_in_contact[n_seed,n_exp], marker='o', markerfacecolor=color, markersize=16, alpha=0.3)
            ax3.plot(TILT_ANGLES_DEG[n_exp], cycles_not_in_contact_LPF[n_seed,n_exp], marker='s', markerfacecolor=color_LPF, markersize=16, alpha=0.3)
        ax3.set_ylabel('Simulation cycles (%)', fontsize=30)
        ax3.yaxis.set_major_locator(plt.MaxNLocator(10))
        # ax3.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1e'))
        ax3.set_yscale('symlog')
        ax3.grid(True) 
        ax3.tick_params(axis = 'y', labelsize=24)
        ax3.set_xlabel('Angle (deg)', fontsize=30)
        ax3.tick_params(axis = 'x', labelsize = 24)



    # Legend error
    handles1, labels1 = ax1.get_legend_handles_labels()
    fig1.legend(handles1, labels1, loc='upper right', prop={'size': 30})
    # Legend error norm 
    handles2, labels2 = ax2[0].get_legend_handles_labels()
    fig2.legend(handles2, labels2, loc='upper right', prop={'size': 30})
    # titles
    # Save, show , clean
    fig1.savefig('/home/skleff/force-feedback/data/pos_err.png')
    fig2.savefig('/home/skleff/force-feedback/data/force_err.png')
    fig3.savefig('/home/skleff/force-feedback/data/time_free.png')
    plt.show()
    plt.close('all')



if __name__=='__main__':
    if len(sys.argv) <= 1:
        print("Usage: python plot_perf_tilt.py [arg1: FILTER (int)]")
        sys.exit(0)
    sys.exit(main(int(sys.argv[1])))



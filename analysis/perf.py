
# which_plots         = ['f'] 
# PLOT_PREDICTIONS    = True 
# pred_plot_sampling  = 100 
# SAVE                = False
# SAVE_DIR            = None 
# SAVE_NAME           = None
# SHOW                = True
# AUTOSCALE           = False
# args = [which_plots, PLOT_PREDICTIONS, pred_plot_sampling, SAVE, SAVE_DIR, SAVE_NAME, SHOW, AUTOSCALE]

# sd1.plot_mpc_results(d1, *args)


from croco_mpc_utils.utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger

from croco_mpc_utils.ocp_core_data import load_data
import analysis_utils


import sys
import numpy as np
import matplotlib.pyplot as plt
from analysis_utils import linear_interpolation

TORQUE_CONTROL = True
if(TORQUE_CONTROL):
    PREFIX = '/home/sebastien/force_feedback_simulation_data/with_tracking/'
else:
    PREFIX = '/home/sebastien/force_feedback_simulation_data/no_tracking/'
    
prefix_lpf       = PREFIX+'iiwa_LPF_sanding_MPC_bullet__BIAS=True_NOISE=True_DELAY=True_Fp=0.5_Fc=1.0_Fs2.0'
prefix_soft      = PREFIX+'iiwa_aug_soft_sanding_MPC_bullet__BIAS=True_NOISE=True_DELAY=True_Fp=0.5_Fc=1.0_Fs2.0'
prefix_classical = PREFIX+'iiwa_sanding_MPC_bullet__BIAS=True_NOISE=True_DELAY=True_Fp=0.5_Fc=1.0_Fs2.0'

CUTOFF = 3. # in seconds

# tilt table of several angles around y-axis
TILT_ANGLES_DEG = [6, 4, 2, 0, -2, -4, -6] 
TILT_RPY = []
for angle in TILT_ANGLES_DEG:
    TILT_RPY.append([angle*np.pi/180, 0., 0.])
N_EXP = len(TILT_RPY)
SEEDS = [19, 71, 89, 83, 41, 73, 17, 47, 29, 7]
N_SEEDS = len(SEEDS)
  
torque_error_AVG_NORM_classical   = np.zeros((N_SEEDS, N_EXP))
position_error_AVG_NORM_classical = np.zeros((N_SEEDS, N_EXP))
force_error_AVG_classical         = np.zeros((N_SEEDS, N_EXP))
force_error_LAT_classical         = np.zeros((N_SEEDS, N_EXP))
force_error_MAX_classical         = np.zeros((N_SEEDS, N_EXP))
cycles_not_in_contact_classical   = np.zeros((N_SEEDS, N_EXP))

torque_error_AVG_NORM_lpf   = np.zeros((N_SEEDS, N_EXP))
position_error_AVG_NORM_lpf = np.zeros((N_SEEDS, N_EXP))
force_error_AVG_lpf         = np.zeros((N_SEEDS, N_EXP))
force_error_LAT_lpf         = np.zeros((N_SEEDS, N_EXP))
force_error_MAX_lpf         = np.zeros((N_SEEDS, N_EXP))
cycles_not_in_contact_lpf   = np.zeros((N_SEEDS, N_EXP))

torque_error_AVG_NORM_soft   = np.zeros((N_SEEDS, N_EXP))
position_error_AVG_NORM_soft = np.zeros((N_SEEDS, N_EXP))
force_error_AVG_soft         = np.zeros((N_SEEDS, N_EXP))
force_error_MAX_soft         = np.zeros((N_SEEDS, N_EXP))
cycles_not_in_contact_soft   = np.zeros((N_SEEDS, N_EXP))


# Compute errors 
FILTER  = 20

for n_seed in range(N_SEEDS):

    logger.debug("Seed "+str(n_seed+1) + "/" + str(N_SEEDS))

    for n_exp in range(N_EXP):
        print('EXP_TILT='+str(TILT_ANGLES_DEG[n_exp])+'_SEED='+str(SEEDS[n_seed]))
        logger.debug("Experiment n°"+str(n_exp+1)+"/"+str(N_EXP))
        # Extract data classical
        sd   = load_data(prefix_classical+'_EXP_TILT='+str(TILT_ANGLES_DEG[n_exp])+'_SEED='+str(SEEDS[n_seed])+'.npz')
        data = sd
        # Interpolate 
        tau_des = linear_interpolation(data['u_des_PLAN'], int( data['N_simu'] / data['N_plan'] ))
        # Smooth if necessary
        if(FILTER > 0):
            data['lin_pos_ee_mea'] = analysis_utils.moving_average_filter(data['lin_pos_ee_mea'].copy(), FILTER)
            data['f_ee_mea']       = analysis_utils.moving_average_filter(data['f_ee_mea'].copy(), FILTER) 
            tau_mea                = analysis_utils.moving_average_filter(data['tau_mea_SIMU'].copy(), FILTER)  
            tau_des                = analysis_utils.moving_average_filter(tau_des.copy(), FILTER)  
        # Compute MAE |mea - ref|
        N_START_SIMU = int(CUTOFF*data['simu_freq'])
        N_START_PLAN = int(CUTOFF*data['plan_freq'])
        Np = data['N_plan'] - N_START_PLAN
        Ns = data['N_simu'] - N_START_SIMU
        
        # Torque error 
        tau_err = []
        for i in range( Ns - 4 ):
            tau_err.append(np.abs(tau_mea[i+N_START_SIMU, :] - tau_des[i+N_START_SIMU, :]))
        torque_error_AVG_NORM_classical[n_seed, n_exp] = np.mean(tau_err) 
        # Position error
        position_error = [] 
        for i in range( Ns ):
            position_error.append(np.abs(data['lin_pos_ee_mea'][i+N_START_SIMU,:2] - data['lin_pos_ee_ref'][int(i*Np/Ns)+N_START_PLAN,:2]))
        position_error_AVG_NORM_classical[n_seed, n_exp] = np.mean(position_error) 
        # Force error
        force_reference = data['frameForceRef'][2] 
        force_error = [] 
        for i in range( Ns ):
            force_error.append(np.abs(data['f_ee_mea'][i+N_START_SIMU,2] - force_reference))
        force_error_MAX_classical[n_seed, n_exp] = np.max(np.abs(data['f_ee_mea'][N_START_SIMU:,2]))
        force_error_LAT_classical[n_seed, n_exp] = np.mean(np.abs(data['f_ee_mea'][N_START_SIMU:,:2]))
        force_error_AVG_classical[n_seed, n_exp] = np.mean(force_error) 
        # Is in contact
        bool_contact = np.isclose(data['f_ee_mea'][N_START_SIMU:,2], np.zeros(data['f_ee_mea'][N_START_SIMU:,2].shape), rtol=1e-3)
        cycles_not_in_contact_classical[n_seed, n_exp] = (100.*np.count_nonzero(bool_contact))/Ns
        
        logger.warning("Classical MPC avg torque error    = "+str(torque_error_AVG_NORM_classical[n_seed, n_exp] ))
        logger.warning("Classical MPC avg position error  = "+str(position_error_AVG_NORM_classical[n_seed, n_exp] ))
        logger.warning("Classical MPC avg force error     = "+str(force_error_AVG_classical[n_seed, n_exp] ))
        logger.warning("Classical MPC max force           = "+str(force_error_MAX_classical[n_seed, n_exp] ))
        logger.warning("Classical MPC not-in-contact rate = "+str(cycles_not_in_contact_classical[n_seed, n_exp] ))
        logger.warning("Classical MPC LATERAL FORCES NORM = "+str(force_error_LAT_classical[n_seed, n_exp] ))




        # Extract LPF
        sd   = load_data(prefix_lpf+'_EXP_TILT='+str(TILT_ANGLES_DEG[n_exp])+'_SEED='+str(SEEDS[n_seed])+'.npz')
        data = sd
        # Interpolate 
        tau_des = linear_interpolation(data['tau_des_PLAN'], int( data['N_simu'] / data['N_plan'] ))
        # Smooth if necessary
        if(FILTER > 0):
            data['lin_pos_ee_mea'] = analysis_utils.moving_average_filter(data['lin_pos_ee_mea'].copy(), FILTER)
            data['f_ee_mea']       = analysis_utils.moving_average_filter(data['f_ee_mea'].copy(), FILTER) 
            data['tau_mea']        = analysis_utils.moving_average_filter(data['tau_mea'].copy(), FILTER)  
            tau_des                = analysis_utils.moving_average_filter(tau_des.copy(), FILTER)  
        # Compute MAE |mea - ref|
        N_START_SIMU = int(CUTOFF*data['simu_freq'])
        N_START_PLAN = int(CUTOFF*data['plan_freq'])
        Np = data['N_plan'] - N_START_PLAN
        Ns = data['N_simu'] - N_START_SIMU

        # Torque error 
        tau_err = []
        for i in range( Ns ):
            tau_err.append(np.abs(data['tau_mea'][i+N_START_SIMU, :] - tau_des[i+N_START_SIMU, :]))
        torque_error_AVG_NORM_lpf[n_seed, n_exp] = np.mean(tau_err) 
        # Position error
        position_error = [] 
        for i in range( Ns ):
            position_error.append(np.abs(data['lin_pos_ee_mea'][i+N_START_SIMU,:2] - data['lin_pos_ee_ref'][int(i*Np/Ns)+N_START_PLAN,:2]))
        position_error_AVG_NORM_lpf[n_seed, n_exp] = np.mean(position_error) 
        # Force error
        force_reference = data['frameForceRef'][2] 
        force_error = [] 
        for i in range( Ns ):
            force_error.append(np.abs(data['f_ee_mea'][i+N_START_SIMU,2] - force_reference))
        force_error_MAX_lpf[n_seed, n_exp]   = np.max(np.abs(data['f_ee_mea'][N_START_SIMU:,2]))
        force_error_LAT_lpf[n_seed, n_exp] = np.mean(np.abs(data['f_ee_mea'][N_START_SIMU:,:2]))
        force_error_AVG_lpf[n_seed, n_exp] = np.mean(force_error) 
        # Is in contact
        bool_contact = np.isclose(data['f_ee_mea'][N_START_SIMU:,2], np.zeros(data['f_ee_mea'][N_START_SIMU:,2].shape), rtol=1e-3)
        cycles_not_in_contact_lpf[n_seed, n_exp] = (100.*np.count_nonzero(bool_contact))/Ns
         
        logger.warning("LPF MPC avg torque error    = "+str(torque_error_AVG_NORM_lpf[n_seed, n_exp] ))
        logger.warning("LPF MPC avg position error  = "+str(position_error_AVG_NORM_lpf[n_seed, n_exp] ))
        logger.warning("LPF MPC avg force error     = "+str(force_error_AVG_lpf[n_seed, n_exp] ))
        logger.warning("LPF MPC max force           = "+str(force_error_MAX_lpf[n_seed, n_exp] ))
        logger.warning("LPF MPC not-in-contact rate = "+str(cycles_not_in_contact_lpf[n_seed, n_exp] ))     
        logger.warning("LPF MPC LATERAL FORCES NORM = "+str(force_error_LAT_lpf[n_seed, n_exp] ))




        # Extract soft 
        sd   = load_data(prefix_soft+'_EXP_TILT='+str(TILT_ANGLES_DEG[n_exp])+'_SEED='+str(SEEDS[n_seed])+'.npz')
        data = sd
        # Interpolate 
        tau_des = linear_interpolation(data['u_des_PLAN'], int( data['N_simu'] / data['N_plan'] ))
        # Smooth if necessary
        if(FILTER > 0):
            data['lin_pos_ee_mea'] = analysis_utils.moving_average_filter(data['lin_pos_ee_mea'].copy(), FILTER)
            data['f_ee_mea']       = analysis_utils.moving_average_filter(data['f_ee_mea'].copy(), FILTER) 
            tau_mea                = analysis_utils.moving_average_filter(data['tau_mea_SIMU'].copy(), FILTER)  
            tau_des                = analysis_utils.moving_average_filter(tau_des.copy(), FILTER)  
        # Compute MAE |mea - ref|
        N_START_SIMU = int(CUTOFF*data['simu_freq'])
        N_START_PLAN = int(CUTOFF*data['plan_freq'])
        Np = data['N_plan'] - N_START_PLAN
        Ns = data['N_simu'] - N_START_SIMU
        
        # Torque error 
        tau_err = []
        for i in range( Ns - 4 ):
            tau_err.append(np.abs(tau_mea[i+N_START_SIMU, :] - tau_des[i+N_START_SIMU, :]))
        torque_error_AVG_NORM_soft[n_seed, n_exp] = np.mean(tau_err) 
        # Position error
        position_error = [] 
        for i in range( Ns ):
            position_error.append(np.abs(data['lin_pos_ee_mea'][i+N_START_SIMU,:2] - data['lin_pos_ee_ref'][int(i*Np/Ns)+N_START_PLAN,:2]))
        position_error_AVG_NORM_soft[n_seed, n_exp] = np.mean(position_error) 
        # Force error
        force_reference = data['frameForceRef'][2] 
        force_error = [] 
        for i in range( Ns ):
            force_error.append(np.abs(data['f_ee_mea'][i+N_START_SIMU] - force_reference))
        force_error_MAX_soft[n_seed, n_exp]   = np.max(np.abs(data['f_ee_mea'][N_START_SIMU:]))
        force_error_AVG_soft[n_seed, n_exp] = np.sum(force_error, axis=0) / Ns
        # Is in contact
        bool_contact = np.isclose(data['f_ee_mea'][N_START_SIMU:], np.zeros(data['f_ee_mea'][N_START_SIMU:].shape), rtol=1e-3)
        cycles_not_in_contact_soft[n_seed, n_exp] = (100.*np.count_nonzero(bool_contact))/Ns
        
        logger.warning("Soft MPC avg torque error    = "+str(torque_error_AVG_NORM_soft[n_seed, n_exp] ))
        logger.warning("Soft MPC avg position error  = "+str(position_error_AVG_NORM_soft[n_seed, n_exp] ))
        logger.warning("Soft MPC avg force error     = "+str(force_error_AVG_soft[n_seed, n_exp] ))
        logger.warning("Soft MPC max force           = "+str(force_error_MAX_soft[n_seed, n_exp] ))
        logger.warning("Soft MPC not-in-contact rate = "+str(cycles_not_in_contact_soft[n_seed, n_exp] ))    
        
        # #  Plot position reference and errors
        # fig, ax = plt.subplots(2, 1, figsize=(19.2,10.8)) 
        # tspan = np.linspace(0, data['T_tot'], position_reference.shape[0])
        # # ax[0].plot(tspan, position_reference[:,0], label='ref_x')
        # ax[0].plot(tspan, position_error[:,1], label='error_x')
        # # ax[1].plot(tspan, position_reference[:,1], label='ref_y')
        # ax[1].plot(tspan, position_error[:,1], label='error_y')
        # plt.show()


# Plot 
fig0, ax0 = plt.subplots(1, 1, figsize=(19.2,10.8)) # Torque errors
fig1, ax1 = plt.subplots(1, 1, figsize=(19.2,10.8)) # Err position 
fig2, ax2 = plt.subplots(2, 1, figsize=(19.2,10.8)) # Err force + max force 
fig3, ax3 = plt.subplots(1, 1, figsize=(19.2,10.8)) # Timings


# Average perfs over seeds
torque_error_AVG_NORM_classical_AVG = np.sum(torque_error_AVG_NORM_classical, axis=0) / N_SEEDS
torque_error_AVG_NORM_lpf_AVG       = np.sum(torque_error_AVG_NORM_lpf, axis=0) / N_SEEDS
torque_error_AVG_NORM_soft_AVG      = np.sum(torque_error_AVG_NORM_soft, axis=0) / N_SEEDS

position_error_AVG_NORM_classical_AVG = np.sum(position_error_AVG_NORM_classical, axis=0) / N_SEEDS
position_error_AVG_NORM_lpf_AVG       = np.sum(position_error_AVG_NORM_lpf, axis=0) / N_SEEDS
position_error_AVG_NORM_soft_AVG      = np.sum(position_error_AVG_NORM_soft, axis=0) / N_SEEDS

force_error_AVG_classical_AVG = np.sum(force_error_AVG_classical, axis=0) / N_SEEDS
force_error_AVG_lpf_AVG       = np.sum(force_error_AVG_lpf, axis=0) / N_SEEDS
force_error_AVG_soft_AVG      = np.sum(force_error_AVG_soft, axis=0) / N_SEEDS

force_error_MAX_classical_AVG = np.sum(force_error_MAX_classical, axis=0) / N_SEEDS
force_error_MAX_lpf_AVG       = np.sum(force_error_MAX_lpf, axis=0) / N_SEEDS
force_error_MAX_soft_AVG      = np.sum(force_error_MAX_soft, axis=0) / N_SEEDS

cycles_not_in_contact_classical_AVG = np.sum(cycles_not_in_contact_classical, axis=0) / N_SEEDS
cycles_not_in_contact_lpf_AVG       = np.sum(cycles_not_in_contact_lpf, axis=0) / N_SEEDS
cycles_not_in_contact_soft_AVG      = np.sum(cycles_not_in_contact_soft, axis=0) / N_SEEDS

colors                    = ['b', 'r', 'g']
labels                    = ['Classical MPC', 'Torque-feedback MPC', 'Force-feedback MPC']
markers                   = ['o', 's', 'D']
torque_errors_AVG           = [torque_error_AVG_NORM_classical_AVG, torque_error_AVG_NORM_lpf_AVG, torque_error_AVG_NORM_soft_AVG]
position_errors_AVG         = [position_error_AVG_NORM_classical_AVG, position_error_AVG_NORM_lpf_AVG, position_error_AVG_NORM_soft_AVG]
force_errors_AVG_AVG        = [force_error_AVG_classical_AVG, force_error_AVG_lpf_AVG, force_error_AVG_soft_AVG]
force_errors_MAX_AVG        = [force_error_MAX_classical_AVG, force_error_MAX_lpf_AVG, force_error_MAX_soft_AVG]
cycles_not_in_contact_AVG   = [cycles_not_in_contact_classical_AVG, cycles_not_in_contact_lpf_AVG, cycles_not_in_contact_soft_AVG]

torque_errors          = [torque_error_AVG_NORM_classical, torque_error_AVG_NORM_lpf, torque_error_AVG_NORM_soft]
position_errors        = [position_error_AVG_NORM_classical, position_error_AVG_NORM_lpf, position_error_AVG_NORM_soft]
force_errors_AVG       = [force_error_AVG_classical, force_error_AVG_lpf, force_error_AVG_soft]
force_errors_MAX       = [force_error_MAX_classical, force_error_MAX_lpf, force_error_MAX_soft]
cycles_not_in_contact  = [cycles_not_in_contact_classical, cycles_not_in_contact_lpf, cycles_not_in_contact_soft]

# Save data 
logger.info('Saving data...')
error_dict = {}
error_dict['SEEDS']                 = SEEDS
error_dict['TILT_ANGLES_DEG']       = TILT_ANGLES_DEG
# Each MAE is a table [seed, angle]
error_dict['torque_errors']         = torque_errors           # MAEs in that order : classical, lpf, soft
error_dict['position_errors']       = position_errors         # MAEs in that order : classical, lpf, soft
error_dict['force_errors_AVG']      = force_errors_AVG        # MAEs in that order : classical, lpf, soft
error_dict['force_errors_MAX']      = force_errors_MAX        # MAEs in that order : classical, lpf, soft
error_dict['cycles_not_in_contact'] = cycles_not_in_contact   # MAEs in that order : classical, lpf, soft
save_path = PREFIX+'errors_TORQUE_CONTROL='+str(TORQUE_CONTROL)+'_full.npz'
np.savez_compressed(save_path, data=error_dict)
logger.info("Saved data to "+str(save_path)+" !")

pwroejfref
for i in range(3):
    # Plot average lines
    if(n_seed == N_SEEDS-1 and n_exp == N_EXP-1):
        lab = labels[i]
    else:
        lab = None
    ax0.plot(TILT_ANGLES_DEG, torque_errors_AVG[i], color=colors[i], linestyle='-', linewidth=3, label=lab)
    ax1.plot(TILT_ANGLES_DEG, position_errors_AVG[i], color=colors[i], linestyle='-', linewidth=3, label=lab)
    ax2[0].plot(TILT_ANGLES_DEG, force_errors_AVG_AVG[i], color=colors[i], linestyle='-', linewidth=3, label=lab)
    ax2[1].plot(TILT_ANGLES_DEG, force_errors_MAX_AVG[i], color=colors[i], linestyle='-', linewidth=3, label=lab)
    ax3.plot(TILT_ANGLES_DEG, cycles_not_in_contact_AVG[i], color=colors[i], linestyle='-', linewidth=4, label=lab)

    # For each experiment plot perf as marker 
    for n_exp in range(N_EXP): 
        ax0.plot(float(TILT_ANGLES_DEG[n_exp]), torque_errors_AVG[i][n_exp], 
                                                marker=markers[i], markerfacecolor=colors[i], 
                                                markersize=18, markeredgecolor='k')
        ax1.plot(float(TILT_ANGLES_DEG[n_exp]), position_errors_AVG[i][n_exp], 
                                                marker=markers[i], markerfacecolor=colors[i], 
                                                markersize=18, markeredgecolor='k')
        ax2[0].plot(float(TILT_ANGLES_DEG[n_exp]), force_errors_AVG_AVG[i][n_exp], 
                                                        marker=markers[i], markerfacecolor=colors[i],
                                                        markersize=18, markeredgecolor='k')
        ax2[1].plot(float(TILT_ANGLES_DEG[n_exp]), force_errors_MAX_AVG[i][n_exp], 
                                                        marker=markers[i], markerfacecolor=colors[i],
                                                        markersize=18, markeredgecolor='k')
        ax3.plot(float(TILT_ANGLES_DEG[n_exp]), cycles_not_in_contact_AVG[i][n_exp], 
                                                        marker=markers[i], markerfacecolor=colors[i],
                                                        markersize=18, markeredgecolor='k')
        # Plot each seed exp with  transparent marker
        for n_seed in range(N_SEEDS):
            ax0.plot(TILT_ANGLES_DEG[n_exp], torque_errors[i][n_seed,n_exp], marker=markers[i], markerfacecolor=colors[i], markersize=16, alpha=0.3)
            ax1.plot(TILT_ANGLES_DEG[n_exp], position_errors[i][n_seed,n_exp], marker=markers[i], markerfacecolor=colors[i], markersize=16, alpha=0.3)
            ax2[0].plot(TILT_ANGLES_DEG[n_exp], force_errors_AVG[i][n_seed,n_exp], marker=markers[i], markerfacecolor=colors[i], markersize=16, alpha=0.3)
            ax2[1].plot(TILT_ANGLES_DEG[n_exp], force_errors_MAX[i][n_seed,n_exp], marker=markers[i], markerfacecolor=colors[i], markersize=16, alpha=0.3)
            ax3.plot(TILT_ANGLES_DEG[n_exp], cycles_not_in_contact[i][n_seed,n_exp], marker=markers[i], markerfacecolor=colors[i], markersize=16, alpha=0.3)

# Set axis and stuff
ax0.set_ylabel('$|| \tau^{mea} - \tau^{des} ||$  (N)', fontsize=26)
ax0.yaxis.set_major_locator(plt.MaxNLocator(2))
ax0.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1e'))
ax0.grid(True) 
ax0.tick_params(axis = 'y', labelsize=22)
ax0.set_xlabel('Angle (deg)', fontsize=26)
ax0.tick_params(axis = 'x', labelsize = 22)

ax1.set_ylabel('$|| \Delta P^{EE}_{xy} ||$  (m)', fontsize=26)
ax1.yaxis.set_major_locator(plt.MaxNLocator(2))
ax1.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1e'))
ax1.grid(True) 
ax1.tick_params(axis = 'y', labelsize=22)
ax1.set_xlabel('Angle (deg)', fontsize=26)
ax1.tick_params(axis = 'x', labelsize = 22)


ax2[0].set_ylabel('$\Delta \lambda_{z}$  (m)', fontsize=26)
ax2[0].yaxis.set_major_locator(plt.MaxNLocator(2))
ax2[0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.1e'))
ax2[0].grid(True) 
ax2[0].tick_params(axis = 'y', labelsize=22)
ax2[0].xaxis.set_ticklabels([])
ax2[0].tick_params(bottom=False)

ax2[1].set_ylabel('$\lambda^{max}_{z}$  (m)', fontsize=26)
ax2[1].yaxis.set_major_locator(plt.MaxNLocator(2))
ax2[1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.1e'))
ax2[1].grid(True) 
ax2[1].tick_params(axis = 'y', labelsize=22)
ax2[1].set_xlabel('Angle (deg)', fontsize=26)
ax2[1].tick_params(axis = 'x', labelsize = 22)

ax3.set_ylabel('Simulation cycles (%)', fontsize=30)
ax3.yaxis.set_major_locator(plt.MaxNLocator(10))
ax3.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1e'))
ax3.set_yscale('symlog')
ax3.grid(True) 
ax3.tick_params(axis = 'y', labelsize=24)
ax3.set_xlabel('Angle (deg)', fontsize=30)
ax3.tick_params(axis = 'x', labelsize = 24)

# Legend torque
handles0, labels0 = ax0.get_legend_handles_labels()
fig0.legend(handles0, labels0, loc='upper right', prop={'size': 26})
# Legend error
handles1, labels1 = ax1.get_legend_handles_labels()
fig1.legend(handles1, labels1, loc='upper right', prop={'size': 26})
# Legend error norm 
handles2, labels2 = ax2[0].get_legend_handles_labels()
fig2.legend(handles2, labels2, loc='upper right', prop={'size': 26})
# Legend contacts
handles3, labels3 = ax3.get_legend_handles_labels()
fig3.legend(handles3, labels3, loc='upper right', prop={'size': 26})
# Save, show , clean
fig0.savefig(PREFIX+'tau_err_'+str(CUTOFF)+'_fix_err.png')
fig1.savefig(PREFIX+'pos_err_'+str(CUTOFF)+'_fix_err.png')
fig2.savefig(PREFIX+'force_err_'+str(CUTOFF)+'_fix_err.png')
fig3.savefig(PREFIX+'contact_timings_'+str(CUTOFF)+'_fix_err.png')
plt.show()
plt.close('all')


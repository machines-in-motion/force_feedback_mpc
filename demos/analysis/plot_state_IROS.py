from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import matplotlib
import analysis_utils
import numpy as np
import sys

from core_mpc_utils.misc_utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, log_level_name=GLOBAL_LOG_LEVEL, USE_LONG_FORMAT=GLOBAL_LOG_FORMAT).logger

from core_mpc_utils import data_utils

PREFIX = '/home/skleff/force-feedback/data/'
data_file_name = 'iiwa_contact_circle_MPC_bullet__BIAS=True_NOISE=True_DELAY=True_Fp=0.5_Fc=0.5_Fs1.0' 
data_file_name_lpf = 'iiwa_LPF_contact_circle_MPC_bullet__BIAS=True_NOISE=True_DELAY=True_Fp=0.25_Fc=0.5_Fs1.0'
# iiwa_LPF_contact_circle_MPC_bullet__BIAS=True_NOISE=True_DELAY=True_Fp=0.25_Fc=0.5_Fs1.0
# Plot options
PLOT_PREDICTIONS = True ; pred_plot_sampling=100
SAVE=False ; SAVE_DIR=None ; SAVE_NAME=None,
SHOW=True 
AUTOSCALE=False


def main(LPF=False, FILTER=0):


    if(LPF):
        print('Opening '+data_file_name_lpf+'...')
        plot_data = data_utils.extract_plot_data_from_npz(PREFIX + data_file_name_lpf + '.npz', LPF=True)
    else:
        print('Opening '+data_file_name+'...')
        plot_data = data_utils.extract_plot_data_from_npz(PREFIX + data_file_name + '.npz', LPF=False)

    # Smooth if necessary
    print('Filtering signals...')
    if(FILTER > 0):
        plot_data['lin_pos_ee_mea'] = analysis_utils.moving_average_filter(plot_data['lin_pos_ee_mea'].copy(), FILTER)
        plot_data['f_ee_mea']       = analysis_utils.moving_average_filter(plot_data['f_ee_mea'].copy(), FILTER) 
    # Plot
    print('Plotting end-eff data (linear)...')
    T_tot = plot_data['T_tot']
    N_simu = plot_data['N_simu']
    N_ctrl = plot_data['N_ctrl']
    N_plan = plot_data['N_plan']
    dt_plan = plot_data['dt_plan']
    T_h = plot_data['T_h']
    N_h = plot_data['N_h']
    # Create time spans for X and U + Create figs and subplots
    t_span_simu = np.linspace(0, T_tot, N_simu+1)
    # t_span_ctrl = np.linspace(0, T_tot, N_ctrl+1)
    t_span_plan = np.linspace(0, T_tot, N_plan+1)
    nq = 7
    fig, ax = plt.subplots(nq, 3, figsize=(19.2,10.8), sharex='col') 
    # fig_f, ax_f = plt.subplots(3, 1, figsize=(19.2,10.8), sharex='col')
    for i in range(nq):

        if(PLOT_PREDICTIONS):

            # Extract state predictions of i^th joint
            q_pred_i = plot_data['q_pred'][:,:,i]
            v_pred_i = plot_data['v_pred'][:,:,i]
            tau_pred_i = plot_data['tau_pred'][:,:,i]
            # For each planning step in the trajectory
            for j in range(0, N_plan, pred_plot_sampling):
                # Receding horizon = [j,j+N_h]
                t0_horizon = j*dt_plan
                tspan_x_pred = np.linspace(t0_horizon, t0_horizon + T_h, N_h+1)
                tspan_u_pred = np.linspace(t0_horizon, t0_horizon + T_h - dt_plan, N_h)
                # Set up lists of (x,y) points for predicted positions and velocities
                points_q = np.array([tspan_x_pred, q_pred_i[j,:]]).transpose().reshape(-1,1,2)
                points_v = np.array([tspan_x_pred, v_pred_i[j,:]]).transpose().reshape(-1,1,2)
                points_tau = np.array([tspan_x_pred, tau_pred_i[j,:]]).transpose().reshape(-1,1,2)
                # Set up lists of segments
                segs_q = np.concatenate([points_q[:-1], points_q[1:]], axis=1)
                segs_v = np.concatenate([points_v[:-1], points_v[1:]], axis=1)
                segs_tau= np.concatenate([points_tau[:-1], points_tau[1:]], axis=1)
                # Make collections segments
                cm = plt.get_cmap('Greys_r') 
                lc_q = LineCollection(segs_q, cmap=cm, zorder=-1)
                lc_v = LineCollection(segs_v, cmap=cm, zorder=-1)
                lc_tau = LineCollection(segs_tau, cmap=cm, zorder=-1)
                lc_q.set_array(tspan_x_pred)
                lc_v.set_array(tspan_x_pred) 
                lc_tau.set_array(tspan_x_pred)
                # Customize
                lc_q.set_linestyle('-')
                lc_v.set_linestyle('-')
                lc_tau.set_linestyle('-')
                lc_q.set_linewidth(1)
                lc_v.set_linewidth(1)
                lc_tau.set_linewidth(1)
                # Plot collections
                ax[i,0].add_collection(lc_q)
                ax[i,1].add_collection(lc_v)
                ax[i,2].add_collection(lc_tau)
                # Scatter to highlight points
                colors = np.r_[np.linspace(0.1, 1, N_h), 1] 
                my_colors = cm(colors)
                ax[i,0].scatter(tspan_x_pred, q_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys) #c='black', 
                ax[i,1].scatter(tspan_x_pred, v_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys) #c='black',
                ax[i,2].scatter(tspan_x_pred, tau_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys) #c='black', 

        # Joint position
        # ax[i,0].plot(t_span_plan, plot_data['q_des_PLAN'][:,i], color='b', linestyle='-', marker='.', label='Desired (PLAN rate)', alpha=0.1)
        # ax[i,0].plot(t_span_ctrl, plot_data['q_des_CTRL'][:,i], color='g', marker=None, linestyle='-', label='Desired (CTRL rate)', alpha=0.3)
        # ax[i,0].plot(t_span_simu, plot_data['q_des_SIMU'][:,i], color='y', linestyle='-', marker='.', label='Desired (SIMU rate)', alpha=0.5)
        # ax[i,0].plot(t_span_simu, plot_data['q_mea'][:,i], 'r-', label='Measured (WITH noise)', linewidth=1, alpha=0.3)
        ax[i,0].plot(t_span_simu, plot_data['q_mea_no_noise'][:,i], color='r', marker=None, linestyle='-', label='Measured', alpha=0.6)
        if('stateReg' in plot_data['WHICH_COSTS']):
            ax[i,0].plot(t_span_plan[:-1], plot_data['state_ref'][:,i], color=[0.,1.,0.,0.], linestyle='-.', marker=None, label='Reference', alpha=0.9)
        ax[i,0].set_ylabel('$q_{}$'.format(i), fontsize=12)
        ax[i,0].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax[i,0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
        ax[i,0].grid(True)
        
        # Joint velocity 
        # ax[i,1].plot(t_span_plan, plot_data['v_des_PLAN'][:,i], color='b', linestyle='-', marker='.', label='Desired (PLAN rate)', alpha=0.1)
        # ax[i,1].plot(t_span_ctrl, plot_data['v_des_CTRL'][:,i], color='g', marker=None, linestyle='-', label='Desired (CTRL)', alpha=0.3)
        # ax[i,1].plot(t_span_simu, plot_data['v_des_SIMU'][:,i], color='y', linestyle='-', marker='.', label='Desired (SIMU)', alpha=0.5)
        # ax[i,1].plot(t_span_simu, plot_data['v_mea'][:,i], 'r-', label='Measured (WITH noise)', linewidth=1, alpha=0.3)
        ax[i,1].plot(t_span_simu, plot_data['v_mea_no_noise'][:,i], color='r', marker=None, linestyle='-', label='Measured', alpha=0.6)
        if('stateReg' in plot_data['WHICH_COSTS']):
            ax[i,1].plot(t_span_plan[:-1], plot_data['state_ref'][:,i+nq], color=[0.,1.,0.,0.], linestyle='-.', marker=None, label='Reference', alpha=0.9)
        ax[i,1].set_ylabel('$v_{}$'.format(i), fontsize=12)
        ax[i,1].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax[i,1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
        ax[i,1].grid(True)

        # Joint torques
        # ax[i,2].plot(t_span_plan, plot_data['tau_des_PLAN'][:,i], color='b', linestyle='-', marker='.', label='Desired (PLAN rate)', alpha=0.1)
        # ax[i,2].plot(t_span_ctrl, plot_data['tau_des_CTRL'][:,i], color='g', marker=None, linestyle='-', label='Desired (CTRL rate)', alpha=0.3)
        # ax[i,2].plot(t_span_simu, plot_data['tau_des_SIMU'][:,i], color='y', linestyle='-', marker='.', label='Desired (SIMU rate)', alpha=0.5)
        # ax[i,2].plot(t_span_simu, plot_data['tau_mea'][:,i], 'r-', label='Measured (WITH noise)', linewidth=1, alpha=0.3)
        ax[i,2].plot(t_span_simu, plot_data['tau_mea_no_noise'][:,i], color='r', marker=None, linestyle='-', label='Measured', alpha=0.6)
        if('ctrlReg' or 'ctrlRegGrav' in plot_data['WHICH_COSTS']):
            ax[i,2].plot(t_span_plan[:-1], plot_data['ctrl_ref'][:,i], color=[0.,1.,0.,0.], linestyle='-.', marker=None, label='Reference', alpha=0.9)
        # ax[i,2].plot(t_span_simu, plot_data['grav'][:,i], color='k', marker=None, linestyle='-.', label='Reg (grav)', alpha=0.6)
        ax[i,2].set_ylabel('$\\tau{}$'.format(i), fontsize=12)
        ax[i,2].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax[i,2].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
        ax[i,2].grid(True)

        # Add xlabel on bottom plot of each column
        if(i == nq-1):
            ax[i,0].set_xlabel('t(s)', fontsize=16)
            ax[i,1].set_xlabel('t(s)', fontsize=16)
            ax[i,2].set_xlabel('t(s)', fontsize=16)
        # Legend
        handles_x, labels_x = ax[i,0].get_legend_handles_labels()
        fig.legend(handles_x, labels_x, loc='upper right', prop={'size': 16})
    # TOL = 1e-5; 
    # for i in range(nq):
    #     ax_q_ylim = 1.1*max(np.max(np.abs(plot_data['q_mea_no_noise'][:,i])), TOL)
    #     ax_v_ylim = 1.1*max(np.max(np.abs(plot_data['v_mea_no_noise'][:,i])), TOL)
    #     ax_tau_ylim = 1.1*max(np.max(np.abs(plot_data['tau_mea_no_noise'][:,i])), TOL)
    #     ax[i,0].set_ylim(-ax_q_ylim, ax_q_ylim) 
    #     ax[i,1].set_ylim(-ax_v_ylim, ax_v_ylim) 
    #     ax[i,2].set_ylim(-ax_tau_ylim, ax_tau_ylim) 

    # y axis labels
    fig.text(0.06, 0.5, 'Joint position (rad)', va='center', rotation='vertical', fontsize=12)
    fig.text(0.345, 0.5, 'Joint velocity (rad/s)', va='center', rotation='vertical', fontsize=12)
    fig.text(0.625, 0.5, 'Joint torque (Nm)', va='center', rotation='vertical', fontsize=12)
    fig.subplots_adjust(wspace=0.37)


    # # Plot endeff
    # for i in range(3):

    #     if(PLOT_PREDICTIONS):
    #         f_ee_pred_i = plot_data['f_ee_pred'][:, :, i]
    #         # For each planning step in the trajectory
    #         for j in range(0, N_plan, pred_plot_sampling):
    #             # Receding horizon = [j,j+N_h]
    #             t0_horizon = j*dt_plan
    #             tspan_x_pred = np.linspace(t0_horizon, t0_horizon + T_h - dt_plan, N_h)
    #             # Set up lists of (x,y) points for predicted positions
    #             points_f = np.array([tspan_x_pred, f_ee_pred_i[j,:]]).transpose().reshape(-1,1,2)
    #             # Set up lists of segments
    #             segs_f = np.concatenate([points_f[:-1], points_f[1:]], axis=1)
    #             # Make collections segments
    #             cm = plt.get_cmap('Greys_r') 
    #             lc_f = LineCollection(segs_f, cmap=cm, zorder=-1)
    #             lc_f.set_array(tspan_x_pred)
    #             # Customize
    #             lc_f.set_linestyle('-')
    #             lc_f.set_linewidth(1)
    #             # Plot collections
    #             ax_f[i].add_collection(lc_f)
    #             # Scatter to highlight points
    #             colors = np.r_[np.linspace(0.1, 1, N_h-1), 1] 
    #             my_colors = cm(colors)
    #             ax_f[i].scatter(tspan_x_pred, f_ee_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys)
       
    #     # EE linear force
    #     # ax_f[i].plot(t_span_plan[:-1], plot_data['f_ee_des_PLAN'][:,i], color='b', linestyle='-', marker='.', label='Desired', alpha=0.1)
    #     # ax_p[i,0].plot(t_span_ctrl, plot_data['f_ee_des_CTRL'][:,i], 'g-', label='Desired (CTRL rate)', alpha=0.5)
    #     # ax_p[i,0].plot(t_span_simu, plot_data['f_ee_des_SIMU'][:,i], color='y', linestyle='-', marker='.', label='Desired (SIMU rate)', alpha=0.5)
    #     ax_f[i].plot(t_span_simu[:-1], plot_data['f_ee_mea'][:,i], 'r-', label='Measured', linewidth=2, alpha=0.6)
    #     # ax_p[i].plot(t_span_simu, plot_data['f_ee_mea_no_noise'][:,i], 'r-', label='Ground truth', linewidth=2)
    #     # Plot reference
    #     if('force' in plot_data['WHICH_COSTS']):
    #         ax_f[i].plot(t_span_plan[:-1], plot_data['f_ee_ref'][:,i], color=[0.,1.,0.,0.], linestyle='-.', linewidth=2., label='Reference', alpha=0.9)
    #     ax_f[i].set_ylabel('$\\lambda^{EE}_%s$  (N)'%xyz[i], fontsize=26)
    #     ax_f[i].yaxis.set_major_locator(plt.MaxNLocator(2))
    #     ax_f[i].yaxis.set_major_formatter(plt.FormatStrFormatter('%.3e'))
    #     ax_f[i].tick_params(labelsize=22, bottom=False, labelbottom=False)
    #     ax_f[i].grid(True) 
    # # Align
    # fig_f.align_ylabels(ax_f[:])
    # ax_f[-1].set_xlabel('t (s)', fontsize=26)
    # ax_f[-1].tick_params(labelsize=22, bottom=True, labelbottom=True)



    # # Set ylim if any
    # TOL = 1e-3
    # if(AUTOSCALE):
    #     ax_p_ylim = 1.1*max(np.max(np.abs(plot_data['lin_pos_ee_mea'])), TOL)
    #     ax_f_ylim = 1.1*max(np.max(np.abs(plot_data['f_ee_mea'])), TOL)
    #     for i in range(3):
    #         ax_f[i].set_ylim(-ax_f_ylim, ax_f_ylim) 
    #     for i in range(2):
    #         ax_p[i].set_ylim(-ax_p_ylim, ax_p_ylim) 
    # handles_p, labels_p = ax_p[0].get_legend_handles_labels()
    # fig_p.legend(handles_p, labels_p, loc='upper right', prop={'size': 26})
    # handles_f, labels_f = ax_f[0].get_legend_handles_labels()
    # fig_f.legend(handles_f, labels_f, loc='upper right', prop={'size': 26})
    # # Titles
    # # fig_p.suptitle('End-effector trajectories', size=18)
    # # Save figs
    # if(SAVE):
    #     fig_p.savefig(PREFIX+'pos_ee_IROS.png')
    #     fig_f.savefig(PREFIX+'f_ee_IROS.png')

    if(SHOW):
        plt.show() 

    # return fig_p, ax_p, fig_f, ax_f



if __name__=='__main__':
    if len(sys.argv) <= 1:
        print("Usage: python plot_ee_f_IROS.py [arg1: LPF (bool)] [arg2: FILTER (int)]")
        sys.exit(0)
    sys.exit(main(int(sys.argv[1]), int(sys.argv[2])))



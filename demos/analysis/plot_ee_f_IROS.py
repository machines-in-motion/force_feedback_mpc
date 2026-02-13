from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import matplotlib
import analysis_utils
import numpy as np
import sys


import sys
sys.path.append('.')

from core_mpc_utils.misc_utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger

from force_feedback_mpc.core_mpc_utils import data_utils

PREFIX = '/home/skleff/force-feedback/data/'
data_file_name = 'iiwa_contact_circle_MPC_bullet__BIAS=True_NOISE=True_DELAY=True_Fp=0.5_Fc=0.5_Fs1.0' 
data_file_name_lpf = 'iiwa_LPF_contact_circle_MPC_bullet__BIAS=True_NOISE=True_DELAY=True_Fp=0.25_Fc=0.5_Fs1.0'
# iiwa_LPF_contact_circle_MPC_bullet__BIAS=True_NOISE=True_DELAY=True_Fp=0.25_Fc=0.5_Fs1.0
# Plot options
PLOT_PREDICTIONS = True ; pred_plot_sampling=25
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
    fig_p, ax_p = plt.subplots(2, 1, figsize=(19.2,10.8), sharex='col') 
    fig_f, ax_f = plt.subplots(3, 1, figsize=(19.2,10.8), sharex='col')

    xyz = ['x', 'y', 'z']

    # Plot endeff
    
    for i in range(2):

        if(PLOT_PREDICTIONS):
            lin_pos_ee_pred_i = plot_data['lin_pos_ee_pred'][:, :, i]
            # For each planning step in the trajectory
            for j in range(0, N_plan, pred_plot_sampling):
                # Receding horizon = [j,j+N_h]
                t0_horizon = j*dt_plan
                tspan_x_pred = np.linspace(t0_horizon, t0_horizon + T_h, N_h+1)
                # Set up lists of (x,y) points for predicted positions
                points_p = np.array([tspan_x_pred, lin_pos_ee_pred_i[j,:]]).transpose().reshape(-1,1,2)
                # Set up lists of segments
                segs_p = np.concatenate([points_p[:-1], points_p[1:]], axis=1)
                # Make collections segments
                cm = plt.get_cmap('Greys_r') 
                lc_p = LineCollection(segs_p, cmap=cm, zorder=-1)
                lc_p.set_array(tspan_x_pred)
                # Customize
                lc_p.set_linestyle('-')
                lc_p.set_linewidth(1)
                # Plot collections
                ax_p[i].add_collection(lc_p)
                # Scatter to highlight points
                colors = np.r_[np.linspace(0.1, 1, N_h), 1] 
                my_colors = cm(colors)
                ax_p[i].scatter(tspan_x_pred, lin_pos_ee_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys)

        # EE position
        # ax_p[i].plot(t_span_plan, plot_data['lin_pos_ee_des_PLAN'][:,i], color='b', linestyle='-', marker='.', label='Desired', alpha=0.1)
        # ax_p[i,0].plot(t_span_ctrl, plot_data['lin_pos_ee_des_CTRL'][:,i] , 'g-', label='Desired (CTRL rate)', alpha=0.5)
        # ax_p[i,0].plot(t_span_simu, plot_data['lin_pos_ee_des_SIMU'][:,i], color='y', linestyle='-', marker='.', label='Desired (SIMU rate)', alpha=0.5)
        ax_p[i].plot(t_span_simu, plot_data['lin_pos_ee_mea'][:,i], 'r-', label='Measured', linewidth=1, alpha=0.3)
        # ax_p[i].plot(t_span_simu, plot_data['lin_pos_ee_mea_no_noise'][:,i], 'r-', label='Ground truth', linewidth=2)
        # Plot reference
        if('translation' in plot_data['WHICH_COSTS']):
            ax_p[i].plot(t_span_plan[:-1], plot_data['lin_pos_ee_ref'][:,i], color=[0.,1.,0.,0.], linestyle='-.', linewidth=2., label='Reference', alpha=0.9)
        ax_p[i].set_ylabel('$\Delta P^{EE}_{%s}$ (m)'%xyz[i], fontsize=26)
        ax_p[i].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax_p[i].yaxis.set_major_formatter(plt.FormatStrFormatter('%.1e'))
        ax_p[i].tick_params(labelsize=22, bottom=False, labelbottom=False)
        ax_p[i].grid(True) 
    # Align
    fig_p.align_ylabels(ax_p[:])
    ax_p[-1].set_xlabel('t (s)', fontsize=26)
    ax_p[-1].tick_params(labelsize=22, bottom=True, labelbottom=True)


    # Plot endeff
    for i in range(3):

        if(PLOT_PREDICTIONS):
            f_ee_pred_i = plot_data['f_ee_pred'][:, :, i]
            # For each planning step in the trajectory
            for j in range(0, N_plan, pred_plot_sampling):
                # Receding horizon = [j,j+N_h]
                t0_horizon = j*dt_plan
                tspan_x_pred = np.linspace(t0_horizon, t0_horizon + T_h - dt_plan, N_h)
                # Set up lists of (x,y) points for predicted positions
                points_f = np.array([tspan_x_pred, f_ee_pred_i[j,:]]).transpose().reshape(-1,1,2)
                # Set up lists of segments
                segs_f = np.concatenate([points_f[:-1], points_f[1:]], axis=1)
                # Make collections segments
                cm = plt.get_cmap('Greys_r') 
                lc_f = LineCollection(segs_f, cmap=cm, zorder=-1)
                lc_f.set_array(tspan_x_pred)
                # Customize
                lc_f.set_linestyle('-')
                lc_f.set_linewidth(1)
                # Plot collections
                ax_f[i].add_collection(lc_f)
                # Scatter to highlight points
                colors = np.r_[np.linspace(0.1, 1, N_h-1), 1] 
                my_colors = cm(colors)
                ax_f[i].scatter(tspan_x_pred, f_ee_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys)
       
        # EE linear force
        # ax_f[i].plot(t_span_plan[:-1], plot_data['f_ee_des_PLAN'][:,i], color='b', linestyle='-', marker='.', label='Desired', alpha=0.1)
        # ax_p[i,0].plot(t_span_ctrl, plot_data['f_ee_des_CTRL'][:,i], 'g-', label='Desired (CTRL rate)', alpha=0.5)
        # ax_p[i,0].plot(t_span_simu, plot_data['f_ee_des_SIMU'][:,i], color='y', linestyle='-', marker='.', label='Desired (SIMU rate)', alpha=0.5)
        ax_f[i].plot(t_span_simu[:-1], plot_data['f_ee_mea'][:,i], 'r-', label='Measured', linewidth=2, alpha=0.6)
        # ax_p[i].plot(t_span_simu, plot_data['f_ee_mea_no_noise'][:,i], 'r-', label='Ground truth', linewidth=2)
        # Plot reference
        if('force' in plot_data['WHICH_COSTS']):
            ax_f[i].plot(t_span_plan[:-1], plot_data['f_ee_ref'][:,i], color=[0.,1.,0.,0.], linestyle='-.', linewidth=2., label='Reference', alpha=0.9)
        ax_f[i].set_ylabel('$\\lambda^{EE}_%s$  (N)'%xyz[i], fontsize=26)
        ax_f[i].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax_f[i].yaxis.set_major_formatter(plt.FormatStrFormatter('%.3e'))
        ax_f[i].tick_params(labelsize=22, bottom=False, labelbottom=False)
        ax_f[i].grid(True) 
    # Align
    fig_f.align_ylabels(ax_f[:])
    ax_f[-1].set_xlabel('t (s)', fontsize=26)
    ax_f[-1].tick_params(labelsize=22, bottom=True, labelbottom=True)



    # Set ylim if any
    TOL = 1e-3
    if(AUTOSCALE):
        ax_p_ylim = 1.1*max(np.max(np.abs(plot_data['lin_pos_ee_mea'])), TOL)
        ax_f_ylim = 1.1*max(np.max(np.abs(plot_data['f_ee_mea'])), TOL)
        for i in range(3):
            ax_f[i].set_ylim(-ax_f_ylim, ax_f_ylim) 
        for i in range(2):
            ax_p[i].set_ylim(-ax_p_ylim, ax_p_ylim) 
    handles_p, labels_p = ax_p[0].get_legend_handles_labels()
    fig_p.legend(handles_p, labels_p, loc='upper right', prop={'size': 26})
    handles_f, labels_f = ax_f[0].get_legend_handles_labels()
    fig_f.legend(handles_f, labels_f, loc='upper right', prop={'size': 26})
    # Titles
    # fig_p.suptitle('End-effector trajectories', size=18)
    # Save figs
    if(SAVE):
        fig_p.savefig(PREFIX+'pos_ee_IROS.png')
        fig_f.savefig(PREFIX+'f_ee_IROS.png')

    if(SHOW):
        plt.show() 

    # return fig_p, ax_p, fig_f, ax_f



if __name__=='__main__':
    if len(sys.argv) <= 1:
        print("Usage: python plot_ee_f_IROS.py [arg1: LPF (bool)] [arg2: FILTER (int)]")
        sys.exit(0)
    sys.exit(main(int(sys.argv[1]), int(sys.argv[2])))



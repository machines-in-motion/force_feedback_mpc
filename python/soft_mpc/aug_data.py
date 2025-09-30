"""
@package force_feedback
@file classical_mpc/init_data.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft.
@date 2020-05-18
@brief Initialize / extract data for MPC simulation (soft contact)
"""

import numpy as np
from croco_mpc_utils import pinocchio_utils as pin_utils
from croco_mpc_utils.ocp_data import OCPDataHandlerClassical, MPCDataHandlerClassical

from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import matplotlib

from croco_mpc_utils.utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger



# Classical OCP data handler : extract data + generate fancy plots
class OCPDataHandlerSoftContactAugmented(OCPDataHandlerClassical):

  def __init__(self, ocp, softContactModel):
    super().__init__(ocp)
    self.softContactModel = softContactModel

  # Temporary patch for augmented soft ocp  --> need to clean it up
  def extract_data(self, xs, us, model):
    '''
    Extract data from OCP solver 
    Patch for augmented soft contact formulation 
    extracting the contact force from the state 
    and desired force from augmented DAM.
    Set 0 angular force by default.
    '''
    ocp_data = super().extract_data(xs, us)
    ocp_data['nq'] = model.nq
    ocp_data['nv'] = model.nv
    ocp_data['nx'] = model.nq+model.nv
    # Compute the visco-elastic contact force & extract the reference force from DAM
    xs = np.array(ocp_data['xs'])
    nq = ocp_data['nq']
    nv = ocp_data['nv']
    if(self.softContactModel.nc == 3):
        fs_lin = np.array([xs[i,-3:] for i in range(ocp_data['T'])])
        fdes_lin = np.array([self.ocp.runningModels[i].differential.f_des for i in range(ocp_data['T'])])
    else:
        fs_lin = np.zeros((ocp_data['T'],3))
        fs_lin[:,self.softContactModel.mask] = [xs[i,-1] for i in range(ocp_data['T'])]
        # fs_lin[:,self.softContactModel.mask] = np.array([self.softContactModel.computeForce_(ocp_data['pin_model'], xs[i,:nq], xs[i,nq:nq+nv]) for i in range(ocp_data['T'])])
        fdes_lin = np.zeros((ocp_data['T'],3))
        # fdes_lin[:,self.softContactModel.mask] = np.array([self.ocp.runningModels[i].differential.f_des for i in range(ocp_data['T'])])
    fs_ang = np.zeros((ocp_data['T'], 3))
    fdes_ang = np.zeros((ocp_data['T'], 3))
    ocp_data['fs'] = np.hstack([fs_lin, fs_ang])
    ocp_data['force_ref'] = np.hstack([fdes_lin, fdes_ang])
    return ocp_data




# Classical MPC data handler : initialize, extract data + generate fancy plots
class MPCDataHandlerSoftContactAugmented(MPCDataHandlerClassical):

  def __init__(self, config, robot, nc):
    super().__init__(config, robot)
    self.ny = self.nx + nc
    self.nc = nc

  # Allocate data 
  def init_predictions(self):
    '''
    Allocate data for state, control & force predictions
    '''
    self.state_pred     = np.zeros((self.N_plan, self.N_h+1, self.ny)) # Predicted states  ( xs : {x* = (q*, v*)} )
    self.ctrl_pred      = np.zeros((self.N_plan, self.N_h, self.nu))   # Predicted torques ( us : {u*} )
    self.force_pred     = np.zeros((self.N_plan, self.N_h, 6))         # Predicted EE contact forces
    self.state_des_PLAN = np.zeros((self.N_plan+1, self.ny))           # Predicted states at planner frequency  ( x* interpolated at PLAN freq )
    self.ctrl_des_PLAN  = np.zeros((self.N_plan, self.nu))             # Predicted torques at planner frequency ( u* interpolated at PLAN freq )
    self.force_des_PLAN = np.zeros((self.N_plan, 6))                   # Predicted EE contact forces planner frequency  
    self.state_des_CTRL = np.zeros((self.N_ctrl+1, self.ny))           # Reference state at motor drivers freq ( x* interpolated at CTRL freq )
    self.ctrl_des_CTRL  = np.zeros((self.N_ctrl, self.nu))             # Reference input at motor drivers freq ( u* interpolated at CTRL freq )
    self.force_des_CTRL = np.zeros((self.N_ctrl, 6))                   # Reference EE contact force at motor drivers freq
    self.state_des_SIMU = np.zeros((self.N_simu+1, self.ny))           # Reference state at actuation freq ( x* interpolated at SIMU freq )
    self.ctrl_des_SIMU  = np.zeros((self.N_simu, self.nu))             # Reference input at actuation freq ( u* interpolated at SIMU freq )
    self.force_des_SIMU = np.zeros((self.N_simu, 6))                   # Reference EE contact force at actuation freq

  def init_measurements(self, y0):
    '''
    Allocate data for simulation state & force measurements 
    '''
    self.state_mea_SIMU                = np.zeros((self.N_simu+1, self.ny))   # Measured states ( x^mea = (q, v) from actuator & PyB at SIMU freq )
    self.state_mea_no_noise_SIMU       = np.zeros((self.N_simu+1, self.ny))   # Measured states ( x^mea = (q, v) from actuator & PyB at SIMU freq ) without noise
    self.tau_mea_SIMU                  = np.zeros((self.N_simu, self.nu))     # Measured torque 
    self.tau_mea_derivative_SIMU       = np.zeros((self.N_simu, self.nu))     # Measured torque derivative
    self.state_mea_SIMU[0, :]          = y0
    self.state_mea_no_noise_SIMU[0, :] = y0

  def init_sim_data(self, y0):
    '''
    Allocate and initialize MPC simulation data
    '''
    # sim_data = {}
    # MPC & simulation parameters
    self.N_plan = int(self.T_tot*self.plan_freq)         # Total number of planning steps in the simulation
    self.N_ctrl = int(self.T_tot*self.ctrl_freq)         # Total number of control steps in the simulation 
    self.N_simu = int(self.T_tot*self.simu_freq)         # Total number of simulation steps 
    self.T_h = self.N_h*self.dt                          # Duration of the MPC horizon (s)
    self.dt_ctrl = float(1./self.ctrl_freq)              # Duration of 1 control cycle (s)
    self.dt_plan = float(1./self.plan_freq)              # Duration of 1 planning cycle (s)
    self.dt_simu = float(1./self.simu_freq)              # Duration of 1 simulation cycle (s)
    self.OCP_TO_PLAN_RATIO = self.dt_plan / self.dt
    # Init actuation model
    self.init_actuation_model()
    # Cost references 
    self.init_cost_references()
    # Predictions
    self.init_predictions()
    # Measurements
    self.init_measurements(y0)

    # OCP solver-specific data
    if(self.RECORD_SOLVER_DATA):
      self.init_solver_data()
   
    logger.info("Initialized MPC simulation data.")

    if(self.INIT_LOG):
      self.print_sim_params(self.init_log_display_time)

  def record_predictions(self, nb_plan, ocpSolver):
    '''
    - Records the MPC prediction of at the current step 
    '''
    self.state_pred[nb_plan, :, :] = np.array(ocpSolver.xs)
    self.ctrl_pred[nb_plan, :, :] = np.array(ocpSolver.us)
    # Extract relevant predictions for interpolations to MPC frequency
    self.y_curr = self.state_pred[nb_plan, 0, :]    # y0* = measured state    (q^,  v^, f^ )
    self.y_pred = self.state_pred[nb_plan, 1, :]    # y1* = predicted state   (q1*, v1*, f1*) 
    self.u_curr = self.ctrl_pred[nb_plan, 0, :]     # u0* = optimal control

  def record_cost_references(self, nb_plan, ocpSolver):
    '''
    Handy function for MPC + clean plots
    Extract and record cost references of DAM into sim_data at i^th simulation step
     # careful, ref is hard-coded only for the first node
    '''
    # Get nodes
    super().record_cost_references(nb_plan, ocpSolver)
    m = ocpSolver.problem.runningModels[0]
    self.f_ee_ref[nb_plan, :self.nc] = -m.differential.f_des


  def record_simu_cycle_measured(self, nb_simu, y_mea_SIMU, y_mea_no_noise_SIMU, tau_mea_SIMU):
    '''
    Records the measurements of state, torque and contact forces at the current simulation cycle
     Input:
      nb_simu             : simulation cycle number
      y_mea_SIMU          : measured position-velocity state from rigid-body physics simulator + measured torque
      y_mea_no_noise_SIMU :  " " without sensing noise
    NOTE: this fucntion also computes the derivatives of the joint torques 
    '''
    self.state_mea_no_noise_SIMU[nb_simu+1, :] = y_mea_SIMU
    self.state_mea_SIMU[nb_simu+1, :]          = y_mea_no_noise_SIMU
    self.tau_mea_SIMU[nb_simu, :]              = tau_mea_SIMU
    if(nb_simu > 0):
        self.tau_mea_derivative_SIMU[nb_simu, :] = (tau_mea_SIMU - self.tau_mea_SIMU[nb_simu-1, :])/self.dt_simu



  def record_plan_cycle_desired(self, nb_plan):
    '''
    - Records the planning cycle data (state, control)
    If an interpolation to planning frequency is needed, here is the place where to implement it
    '''
    if(nb_plan==0):
        self.state_des_PLAN[nb_plan, :] = self.y_curr  
    self.ctrl_des_PLAN[nb_plan, :]      = self.u_curr   
    self.state_des_PLAN[nb_plan+1, :]   = self.y_curr + self.OCP_TO_PLAN_RATIO * (self.y_pred - self.y_curr)    


  # Extract MPC simu-specific plotting data from sim data
  def extract_data(self, frame_of_interest):
    '''
    Extract plot data from simu data
    '''
    logger.info('Extracting plot data from simulation data...')
    
    plot_data = self.__dict__.copy()
    # Get costs
    plot_data['WHICH_COSTS'] = self.WHICH_COSTS
    # Robot model & params
    plot_data['pin_model'] = self.rmodel
    self.id_endeff = self.rmodel.getFrameId(frame_of_interest)
    nq = self.nq ; nv = self.nv ; nu = self.nv ; nc = self.nc
    plot_data['nc'] = nc
    # Control predictions
    plot_data['u_pred'] = self.ctrl_pred
      # Extract 1st prediction
    plot_data['u_des_PLAN'] = self.ctrl_des_PLAN
    # State predictions (at PLAN freq)
    plot_data['q_pred']     = self.state_pred[:,:,:nq]
    plot_data['v_pred']     = self.state_pred[:,:,nq:nq+nv]
    plot_data['q_des_PLAN'] = self.state_des_PLAN[:,:nq]
    plot_data['v_des_PLAN'] = self.state_des_PLAN[:,nq:nq+nv] 
    # State measurements (at SIMU freq)
    plot_data['q_mea']          = self.state_mea_SIMU[:,:nq]
    plot_data['v_mea']          = self.state_mea_SIMU[:,nq:nq+nv]
    # plot_data['f_mea'] = self.state_mea_SIMU[:,-nc:]
    plot_data['q_mea_no_noise'] = self.state_mea_no_noise_SIMU[:,:nq]
    plot_data['v_mea_no_noise'] = self.state_mea_no_noise_SIMU[:,nq:nq+nv]
    plot_data['f_mea_no_noise'] = self.state_mea_no_noise_SIMU[:,-nc:]
    # Extract EE force
    plot_data['f_ee_pred']     = self.state_pred[:,:,-nc:]
    plot_data['f_ee_mea']      = self.state_mea_SIMU[:,-nc:]
    plot_data['f_ee_des_PLAN'] = self.state_des_PLAN[:,-nc:]
    # Extract gravity torques
    plot_data['grav'] = np.zeros((self.N_simu+1, nq))
    # Torque measurements
    plot_data['u_mea'] = self.tau_mea_SIMU
    # print(plot_data['pin_model'])
    for i in range(plot_data['N_simu']+1):
      plot_data['grav'][i,:] = pin_utils.get_u_grav(plot_data['q_mea'][i,:], plot_data['pin_model'])
    # EE predictions (at PLAN freq)
      # Linear position velocity of EE
    plot_data['lin_pos_ee_pred'] = np.zeros((self.N_plan, self.N_h+1, 3))
    plot_data['lin_vel_ee_pred'] = np.zeros((self.N_plan, self.N_h+1, 3))
      # Angular position velocity of EE
    plot_data['ang_pos_ee_pred'] = np.zeros((self.N_plan, self.N_h+1, 3)) 
    plot_data['ang_vel_ee_pred'] = np.zeros((self.N_plan, self.N_h+1, 3)) 
    for node_id in range(self.N_h+1):
        plot_data['lin_pos_ee_pred'][:, node_id, :] = pin_utils.get_p_(plot_data['q_pred'][:, node_id, :], plot_data['pin_model'], self.id_endeff)
        plot_data['lin_vel_ee_pred'][:, node_id, :] = pin_utils.get_v_(plot_data['q_pred'][:, node_id, :], plot_data['v_pred'][:, node_id, :], plot_data['pin_model'], self.id_endeff)
        plot_data['ang_pos_ee_pred'][:, node_id, :] = pin_utils.get_rpy_(plot_data['q_pred'][:, node_id, :], plot_data['pin_model'], self.id_endeff)
        plot_data['ang_vel_ee_pred'][:, node_id, :] = pin_utils.get_w_(plot_data['q_pred'][:, node_id, :], plot_data['v_pred'][:, node_id, :], plot_data['pin_model'], self.id_endeff)
    # EE measurements (at SIMU freq)
      # Linear
    plot_data['lin_pos_ee_mea']          = pin_utils.get_p_(plot_data['q_mea'], self.rmodel, self.id_endeff)
    plot_data['lin_vel_ee_mea']          = pin_utils.get_v_(plot_data['q_mea'], plot_data['v_mea'], self.rmodel, self.id_endeff)
    plot_data['lin_pos_ee_mea_no_noise'] = pin_utils.get_p_(plot_data['q_mea_no_noise'], plot_data['pin_model'], self.id_endeff)
    plot_data['lin_vel_ee_mea_no_noise'] = pin_utils.get_v_(plot_data['q_mea_no_noise'], plot_data['v_mea_no_noise'], plot_data['pin_model'], self.id_endeff)
      # Angular
    plot_data['ang_pos_ee_mea']          = pin_utils.get_rpy_(plot_data['q_mea'], self.rmodel, self.id_endeff)
    plot_data['ang_vel_ee_mea']          = pin_utils.get_w_(plot_data['q_mea'], plot_data['v_mea'], self.rmodel, self.id_endeff)
    plot_data['ang_pos_ee_mea_no_noise'] = pin_utils.get_rpy_(plot_data['q_mea_no_noise'], plot_data['pin_model'], self.id_endeff)
    plot_data['ang_vel_ee_mea_no_noise'] = pin_utils.get_w_(plot_data['q_mea_no_noise'], plot_data['v_mea_no_noise'], plot_data['pin_model'], self.id_endeff)
    # EE des
      # Linear
    plot_data['lin_pos_ee_des_PLAN'] = pin_utils.get_p_(plot_data['q_des_PLAN'], self.rmodel, self.id_endeff)
    plot_data['lin_vel_ee_des_PLAN'] = pin_utils.get_v_(plot_data['q_des_PLAN'], plot_data['v_des_PLAN'], self.rmodel, self.id_endeff)
     # Angular
    plot_data['ang_pos_ee_des_PLAN'] = pin_utils.get_rpy_(plot_data['q_des_PLAN'], self.rmodel, self.id_endeff)
    plot_data['ang_vel_ee_des_PLAN'] = pin_utils.get_w_(plot_data['q_des_PLAN'], plot_data['v_des_PLAN'], self.rmodel, self.id_endeff)
    return plot_data

  def plot_mpc_force(self, plot_data, PLOT_PREDICTIONS=False, 
                            pred_plot_sampling=100, 
                            SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
                            SHOW=True, AUTOSCALE=False):
      '''
      Plot EE force data
      Input:
        plot_data                 : plotting data
        PLOT_PREDICTIONS          : True or False
        pred_plot_sampling        : plot every pred_plot_sampling prediction 
                                    to avoid huge amount of plotted data 
                                    ("1" = plot all)
        SAVE, SAVE_DIR, SAVE_NAME : save plots as .png
        SHOW                      : show plots
        AUTOSCALE                 : rescale y-axis of endeff plot 
                                    based on maximum value taken
      '''
      logger.info('Plotting force data...')
      T_tot = plot_data['T_tot']
      N_simu = plot_data['N_simu']
      N_ctrl = plot_data['N_ctrl']
      N_plan = plot_data['N_plan']
      dt_plan = plot_data['dt_plan']
      dt_simu = plot_data['dt_simu']
      dt_ctrl = plot_data['dt_ctrl']
      T_h = plot_data['T_h']
      N_h = plot_data['N_h']
      nc = plot_data['nc']
      # Create time spans for X and U + Create figs and subplots
      t_span_simu = np.linspace(0, T_tot, N_simu+1)
      t_span_plan = np.linspace(0, T_tot, N_plan+1)
      fig, ax = plt.subplots(3, 1, figsize=(19.2,10.8), sharex='col') 
      # Plot endeff
      xyz = ['x', 'y', 'z']
      for i in range(3):
          if(PLOT_PREDICTIONS):
              if(nc == 1):
                 if(i == 1 or i == 0):
                    f_ee_pred_i = np.zeros(plot_data['f_ee_pred'][:, :, 0].shape)
                 else:
                    f_ee_pred_i = plot_data['f_ee_pred'][:, :, 0]
              else:
                 f_ee_pred_i = plot_data['f_ee_pred'][:, :, i]
              # For each planning step in the trajectory
              for j in range(0, N_plan, pred_plot_sampling):
                  # Receding horizon = [j,j+N_h]
                  t0_horizon = j*dt_plan
                  tspan_x_pred = np.array([t0_horizon + sum(plot_data['dts'][:i]) for i in range(len(plot_data['dts']))]) #np.linspace(t0_horizon, t0_horizon + T_h, N_h+1)
                  # Set up lists of (x,y) points for predicted positions and velocities
                  points_q = np.array([tspan_x_pred, f_ee_pred_i[j,:]]).transpose().reshape(-1,1,2)
                  # Set up lists of segments
                  segs_q = np.concatenate([points_q[:-1], points_q[1:]], axis=1)
                  # Make collections segments
                  cm = plt.get_cmap('Greys_r') 
                  lc_q = LineCollection(segs_q, cmap=cm, zorder=-1)
                  lc_q.set_array(tspan_x_pred)
                  # Customize
                  lc_q.set_linestyle('-')
                  lc_q.set_linewidth(1)
                  # Plot collections
                  ax[i].add_collection(lc_q)
                  # Scatter to highlight points
                  colors = np.r_[np.linspace(0.1, 1, N_h), 1] 
                  my_colors = cm(colors)
                  ax[i].scatter(tspan_x_pred, f_ee_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys) #c='black', 

        
          # EE linear force
          if(nc == 1):
              if(i == 1 or i == 0):
                f_ee_des_PLAN = np.zeros(plot_data['f_ee_des_PLAN'][:, 0].shape)
                f_ee_mea = np.zeros(plot_data['f_ee_mea'][:, 0].shape)
                f_ee_ref = np.zeros(plot_data['f_ee_ref'][:, 0].shape)
              else:
                f_ee_des_PLAN = plot_data['f_ee_des_PLAN'][:, 0]
                f_ee_mea = plot_data['f_ee_mea'][:, 0]
                f_ee_ref = plot_data['f_ee_ref'][:, 0]
          else:
              f_ee_des_PLAN = plot_data['f_ee_des_PLAN'][:, i]
              f_ee_mea = plot_data['f_ee_mea'][:, i]
              f_ee_ref = plot_data['f_ee_ref'][:, i]
          ax[i].plot(t_span_plan, f_ee_des_PLAN, color='b', linestyle='-', marker='.', label='Desired (PLAN rate)', alpha=0.1)
          ax[i].plot(t_span_simu, f_ee_mea, 'r-', label='Measured', linewidth=2, alpha=0.6)
          # Plot reference
          if('force' in plot_data['WHICH_COSTS']):
              ax[i].plot(t_span_plan[:-1], f_ee_ref, color=[0.,1.,0.,0.], linestyle='-.', linewidth=2., label='Reference', alpha=0.9)
          ax[i].set_ylabel('$\\lambda^{EE}_%s$  (N)'%xyz[i], fontsize=16)
          ax[i].yaxis.set_major_locator(plt.MaxNLocator(2))
          ax[i].yaxis.set_major_formatter(plt.FormatStrFormatter('%.3e'))
          ax[i].grid(True)
      
    #   # Align
      fig.align_ylabels(ax[0])
    #   fig.align_ylabels(ax[:,1])
      ax[i].set_xlabel('t (s)', fontsize=16)
    #   ax[i,1].set_xlabel('t (s)', fontsize=16)
      # Set ylim if any
      TOL = 1e-3
      if(AUTOSCALE):
          ax_ylim1 = 1.1*max(np.max(np.abs(plot_data['f_ee_pred'])), TOL) # 1.1*max( np.nanmax(np.abs(plot_data['f_ee_mea'])), TOL )
          ax_ylim2 = 1.1*max(np.max(np.abs(plot_data['f_ee_mea'])), TOL) 
          ax_ylim = max(ax_ylim1, ax_ylim2)
          for i in range(3):
              ax[i].set_ylim(-ax_ylim, ax_ylim) 
              # ax[i].set_ylim(-10, 10) 

      handles_p, labels_p = ax[0].get_legend_handles_labels()
      fig.legend(handles_p, labels_p, loc='upper right', prop={'size': 16})
      # Titles
      fig.suptitle('End-effector forces', size=18)
      # Save figs
      if(SAVE):
          figs = {'f': fig}
          if(SAVE_DIR is None):
              logger.error("Please specify SAVE_DIR")
          if(SAVE_NAME is None):
              SAVE_NAME = 'testfig'
          for name, fig in figs.items():
              fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
      
      if(SHOW):
          plt.show() 
      
      return fig, ax

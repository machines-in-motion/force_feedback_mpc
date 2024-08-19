"""
@package force_feedback
@file LPF_reaching_MPC.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and LAAS-CNRS
@date 2020-05-18
@brief Closed-loop 'LPF torque feedback' MPC for static target task 
"""

'''
The robot is tasked with reaching a static EE target 
Trajectory optimization using Crocoddyl in closed-loop MPC 
(feedback from stateLPF y=(q,v,tau), control w = unfiltered torque)
Using PyBullet simulator & GUI for rigid-body dynamics + visualization

The goal of this script is to simulate MPC with torque feedback where
the actuation dynamics is modeled as a low pass filter (LPF) in the optimization.
  - The letter y denotes the augmented state of joint positions, velocities
    and filtered torques while the letter 'w' denotes the unfiltered torque 
    input to the actuation model. 
  - We optimize (y*,w*) using Crocoddyl but we send tau* to the simulator (NOT w*)
  - Simulator = custom actuation model (not LPF) + PyBullet RBD
'''




import sys
sys.path.append('.')

from core_mpc_utils.misc_utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger

import numpy as np  
np.random.seed(1)
np.set_printoptions(precision=4, linewidth=180)


from core_mpc_utils import ocp, path_utils, pin_utils, mpc_utils, misc_utils


from lpf_mpc.data import DDPDataHandlerLPF, MPCDataHandlerLPF
from lpf_mpc.ocp import OptimalControlProblemLPF, getJointAndStateIds

def main(robot_name='iiwa', simulator='bullet', PLOT_INIT=False):

  # # # # # # # # # # # # # # # # # # #
  ### LOAD ROBOT MODEL and SIMU ENV ### 
  # # # # # # # # # # # # # # # # # # # 
  # Read config file
  config, config_name = path_utils.load_config_file(__file__, robot_name)
  # Create a simulation environment & simu-pin wrapper 
  dt_simu = 1./float(config['simu_freq'])  
  q0 = np.asarray(config['q0'])
  v0 = np.asarray(config['dq0'])
  x0 = np.concatenate([q0, v0])   
  if(simulator == 'bullet'):
    from core_mpc_utils import sim_utils as simulator_utils
    env, robot_simulator, base_placement = simulator_utils.init_bullet_simulation(robot_name, dt=dt_simu, x0=x0)
    robot = robot_simulator.pin_robot
  elif(simulator == 'raisim'):
    from core_mpc_utils import raisim_utils as simulator_utils
    env, robot_simulator, _ = simulator_utils.init_raisim_simulation(robot_name, dt=dt_simu, x0=x0)  
    robot = robot_simulator
  else:
    logger.error('Please choose a simulator from ["bullet", "raisim"] !')
  # Get dimensions 
  nq, nv = robot.model.nq, robot.model.nv; 
  nu = nq 



  # # # # # # # # # 
  ### OCP SETUP ###
  # # # # # # # # # 
  # Setup Crocoddyl OCP and create solver
  ug = pin_utils.get_u_grav(q0, robot.model, config['armature']) 
  lpf_joint_names = ['A2', 'A3'] #, 'A4'] #['A6'] #robot.model.names[1:] # #
  _, lpfStateIds = getJointAndStateIds(robot.model, lpf_joint_names)
  n_lpf = len(lpf_joint_names) 
  _, nonLpfStateIds = getJointAndStateIds(robot.model, list(set(robot.model.names[1:]) - set(lpf_joint_names)) )
  logger.debug("LPF state ids ")
  logger.debug(lpfStateIds)
  logger.debug("Non LPF state ids ")
  logger.debug(nonLpfStateIds)
  y0 = np.concatenate([x0, ug[lpfStateIds]])
  ddp = OptimalControlProblemLPF(robot, config, lpf_joint_names).initialize(y0, callbacks=True)
  # Warm start and solve 
  xs_init = [y0 for i in range(config['N_h']+1)]
  us_init = [ug for i in range(config['N_h'])]
  ddp.solve(xs_init, us_init, maxiter=100, isFeasible=False)
  # Plot initial solution
  frame_name = config['frameTranslationFrameName']
  if(PLOT_INIT):
    ddp_handler = DDPDataHandlerLPF(ddp)
    ddp_data = ddp_handler.extract_data(ee_frame_name=frame_name)
    _, _ = ddp_handler.plot_ddp_results(ddp_data, markers=['.'], SHOW=True)




  # # # # # # # # # # #
  ### INIT MPC SIMU ###
  # # # # # # # # # # #
  sim_data = MPCDataHandlerLPF(config, robot, n_lpf)
  sim_data.init_sim_data(y0)
    # Replan & control counters
  nb_plan = 0
  nb_ctrl = 0
  # Additional simulation blocks 
  communicationModel = mpc_utils.CommunicationModel(config)
  actuationModel     = mpc_utils.ActuationModel(config, nu)
  sensingModel       = mpc_utils.SensorModel(config, naug=n_lpf)

  # Display target
  if(hasattr(simulator_utils, 'display_ball')):
    p_ball = np.asarray(config['frameTranslationRef'])
    simulator_utils.display_ball(p_ball, robot_base_pose=base_placement, RADIUS=.05, COLOR=[1.,0.,0.,.6])


  # # # # # # # # # # # #
  ### SIMULATION LOOP ###
  # # # # # # # # # # # #

  # SIMULATE
  for i in range(config['N_simu']): 

      if(i%config['log_rate']==0 and config['LOG']): 
        print('')
        logger.info("SIMU step "+str(i)+"/"+str(config['N_simu']))
        print('')

    # Solve OCP if we are in a planning cycle (MPC/planning frequency)
      if(i%int(sim_data.simu_freq/sim_data.plan_freq) == 0):
          # Reset x0 to measured state + warm-start solution
          ddp.problem.x0 = sim_data.state_mea_SIMU[i, :]
          xs_init = list(ddp.xs[1:]) + [ddp.xs[-1]]
          xs_init[0] = sim_data.state_mea_SIMU[i, :]
          us_init = list(ddp.us[1:]) + [ddp.us[-1]] 
          # Solve OCP & record MPC predictions
          ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=False)
          # Record MPC predictions, cost refs and solver data
          sim_data.record_predictions(nb_plan, ddp)
          # Record cost references
          sim_data.record_cost_references(nb_plan, ddp)
          # Record solver data (optional)
          sim_data.record_solver_data(nb_plan, ddp) 
          # Model communication between computer --> robot
          communicationModel.step(sim_data.y_pred, sim_data.w_curr)
          # Select reference control and state for the current PLAN cycle
          sim_data.record_plan_cycle_desired(nb_plan)
          # Increment planning counter
          nb_plan += 1

    # If we are in a control cycle select reference torque to send to the actuator (motor driver input frequency)
      if(i%int(sim_data.simu_freq/sim_data.ctrl_freq) == 0):        
          sim_data.record_ctrl_cycle_desired(nb_ctrl)
          nb_ctrl += 1
          
    # Simulate actuation/sensing and step simulator (physics simulation frequency)
      # Select reference control and state for the current SIMU cycle
      sim_data.record_simu_cycle_desired(i)
      # Torque applied by motor on actuator : interpolate current torque and predicted torque 
      tau_ref_SIMU =  sim_data.y_ref_SIMU[-n_lpf:] 
      # Actuation model ( tau_ref_SIMU ==> tau_mea_SIMU ) 
        # Simulate imperfect actuation (for dimensions that are assumed perfectly actuated by the MPC)
        #  sim_data.w_ref_SIMU, sim_data.ctrl_des_SIMU 
      tau_mea_SIMU = sim_data.w_ref_SIMU 
      tau_mea_SIMU[nonLpfStateIds] = actuationModel.step(i, sim_data.w_ref_SIMU[nonLpfStateIds], sim_data.ctrl_des_SIMU[nonLpfStateIds]) 
        # Simulate imperfect actuation (for dimensions that are asssumed to be LPF-actuated by the MPC)
      tau_mea_SIMU[lpfStateIds] = actuationModel.step(i, tau_ref_SIMU, sim_data.state_mea_SIMU[:,-n_lpf:])    
      # RICCATI GAINS TO INTERPOLATE
      if(config['RICCATI']):
        K = ddp.K[0]
        alpha = np.exp(-2*np.pi*config['f_c']*config['dt'])
        Ktilde  = (1-alpha)*sim_data.OCP_TO_PLAN_RATIO*K
        # Ktilde[:,nq+nv:nq+nv+n_lpf] += ( 1 - (1-alpha)*sim_data.OCP_TO_PLAN_RATIO )*np.eye(nv)[:,lpfStateIds] # only for torques
        tau_mea_SIMU += Ktilde[:,:nq+nv].dot(ddp.problem.x0[:nq+nv] - sim_data.state_mea_SIMU[i,:nq+nv]) #position vel
        # tau_mea_SIMU += Ktilde[:,:-nq].dot(ddp.problem.x0[:-nq] - sim_data.state_mea_SIMU[i,:-nq])       # torques
      # Send output of actuation torque to the RBD simulator 
      robot_simulator.send_joint_command(tau_mea_SIMU)
      env.step()
      # Measure new state from simulation :
      q_mea_SIMU, v_mea_SIMU = robot_simulator.get_state()
      # Update pinocchio model
      robot_simulator.forward_robot(q_mea_SIMU, v_mea_SIMU)
      # Record data (unnoised)
      y_mea_SIMU = np.concatenate([q_mea_SIMU, v_mea_SIMU, tau_mea_SIMU[lpfStateIds]]).T 
      sim_data.state_mea_no_noise_SIMU[i+1, :] = y_mea_SIMU
      # Sensor model (optional noise + filtering)
      sim_data.state_mea_SIMU[i+1, :] = sensingModel.step(i, y_mea_SIMU, sim_data.state_mea_SIMU)




  # # # # # # # # # # #
  # PLOT SIM RESULTS  #
  # # # # # # # # # # #
  save_dir = '/tmp'
  save_name = 'test'
  save_name = config_name+'_'+simulator+'_'+\
                          '_BIAS='+str(config['SCALE_TORQUES'])+\
                          '_NOISE='+str(config['NOISE_STATE'] or config['NOISE_TORQUES'])+\
                          '_DELAY='+str(config['DELAY_OCP'] or config['DELAY_SIM'])+\
                          '_Fp='+str(sim_data.plan_freq/1000)+'_Fc='+str(sim_data.ctrl_freq/1000)+'_Fs'+str(sim_data.simu_freq/1000)
  #  Extract plot data from sim data
  plot_data = sim_data.extract_data(frame_of_interest=frame_name)
  #  Plot results
  sim_data.plot_mpc_results(plot_data, which_plots=config['WHICH_PLOTS'],
                                  PLOT_PREDICTIONS=True, 
                                  pred_plot_sampling=int(sim_data.plan_freq/10),
                                  SAVE=False,
                                  SAVE_DIR=save_dir,
                                  SAVE_NAME=save_name,
                                  AUTOSCALE=True)
  # Save optionally
  if(config['SAVE_DATA']):
    sim_data.save_data(sim_data, save_name=save_name, save_dir=save_dir)





if __name__=='__main__':
    args = misc_utils.parse_MPC_script(sys.argv[1:])
    main(args.robot_name, args.simulator, args.PLOT_INIT)
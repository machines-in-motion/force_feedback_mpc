"""
@package force_feedback
@file LPF_contact_MPC.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2021, New York University & LAAS-CNRS
@date 2021-10-28
@brief Closed-loop 'LPF torque feedback' MPC for force task
"""

'''
The robot is tasked with exerting a constant normal force with its EE 
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
np.set_printoptions(precision=4, linewidth=180)
RANDOM_SEED = 1

from core_mpc_utils import path_utils, mpc_utils, misc_utils

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
    env, robot_simulator, _ = simulator_utils.init_bullet_simulation(robot_name, dt=dt_simu, x0=x0)
    robot = robot_simulator.pin_robot
  elif(simulator == 'raisim'):
    from core_mpc_utils import raisim_utils as simulator_utils
    env, robot_simulator = simulator_utils.init_raisim_simulation(robot_name, dt=dt_simu, x0=x0)  
    robot = robot_simulator
  else:
    logger.error('Please choose a simulator from ["bullet", "raisim"] !')
  # Get dimensions 
  nq, nv = robot.model.nq, robot.model.nv; nu = nq
  # Placement of LOCAL end-effector frame w.r.t. WORLD frame
  frame_of_interest = config['frame_of_interest']
  id_endeff = robot.model.getFrameId(frame_of_interest)
  
  # EE translation target : contact point + vertical offset (radius of the ee ball)
  contactPosition = np.asarray(config['contacts'][0]['contactPosition'])
  offset = config['contact_plane_offset']
  contactTranslationTarget = contactPosition + offset
  simulator_utils.display_ball(contactTranslationTarget, RADIUS=0.02, COLOR=[1.,0.,0.,0.2])
  # Display contact surface
  import pinocchio as pin
  contact_placement = pin.SE3(np.eye(3), np.asarray(contactPosition))
  contactId = simulator_utils.display_contact_surface(contact_placement, bullet_endeff_ids=robot_simulator.bullet_endeff_ids)
  # Make the contact soft (e.g. tennis ball or sponge on the robot)
  simulator_utils.set_lateral_friction(contactId, 0.9)
  simulator_utils.set_contact_stiffness_and_damping(contactId, 10000, 500)

  # # # # # # # # 
  ### OCP SETUP ###
  # # # # # # # # # 
  # Create DDP solver + compute warm start torque
  # f_ext = pin_utils.get_external_joint_torques(contact_placement.copy(), config['frameForceRef'], robot)
  u0 = pin_utils.get_u_grav(q0, robot.model, config['armature']) # pin_utils.get_tau(q0, v0, np.zeros((nq,1)), f_ext, robot.model, config['armature'])
  lpf_joint_names = robot.model.names[1:] #['A2', 'A3'] 
  _, lpfStateIds = getJointAndStateIds(robot.model, lpf_joint_names)
  n_lpf = len(lpf_joint_names) 
  _, nonLpfStateIds = getJointAndStateIds(robot.model, list(set(robot.model.names[1:]) - set(lpf_joint_names)) )
  logger.debug("LPF state ids ")
  logger.debug(lpfStateIds)
  logger.debug("Non LPF state ids ")
  logger.debug(nonLpfStateIds)
  y0 = np.concatenate([x0, u0[lpfStateIds]])
  ddp = OptimalControlProblemLPF(robot, config, lpf_joint_names).initialize(y0, callbacks=False)
  # !!! Deactivate all costs & contact models initially !!!
  models = list(ddp.problem.runningModels) + [ddp.problem.terminalModel]
  # for k,m in enumerate(models):
  #   m.differential.contacts.contacts["contact"].contact.active = False
  #   # m.differential.costs.costs["force"].active = False
  MAX_FORCE_WEIGHT = config['frameForceWeight'] #1

  # Warmstart and solve
  xs_init = [y0 for i in range(config['N_h']+1)]
  us_init = [u0 for i in range(config['N_h'])]
  ddp.solve(xs_init, us_init, maxiter=100, isFeasible=False)

  # logger.debug(ddp.problem.terminalModel.differential.costs.costs['stateReg'])
  # logger.debug(ddp.problem.terminalModel.differential.costs.costs['force'])

  # Plot initial solution
  if(PLOT_INIT):
    ddp_handler = DDPDataHandlerLPF(ddp, n_lpf)
    ddp_data = ddp_handler.extract_data(ee_frame_name=frame_of_interest, ct_frame_name=frame_of_interest)
    _, _ = ddp_handler.plot_ddp_results(ddp_data, which_plots=config['WHICH_PLOTS'], 
                                                        colors=['r'], 
                                                        markers=['.'], 
                                                        SHOW=True)  


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
  actuationModel     = mpc_utils.ActuationModel(config, nu, SEED=RANDOM_SEED)
  sensingModel       = mpc_utils.SensorModel(config, naug=n_lpf, SEED=RANDOM_SEED)



  # # # # # # # # # # # #
  ### SIMULATION LOOP ###
  # # # # # # # # # # # #
  from analysis.analysis_utils import MPCBenchmark
  bench = MPCBenchmark()

  # Horizon in simu cycles
  NH_SIMU   = int(config['N_h']*sim_data.dt/sim_data.dt_simu)
  T_REACH   = int(config['T_REACH']/sim_data.dt_simu)
  T_CONTACT = int(config['T_CONTACT']/sim_data.dt_simu)
  OCP_TO_MPC_CYCLES = 1./(sim_data.dt_plan / config['dt'])
  OCP_TO_SIMU_CYCLES = 1./(sim_data.dt_simu / config['dt'])
  logger.debug("Size of MPC horizon in simu cycles = "+str(NH_SIMU))
  logger.debug("Start of reaching phase in simu cycles = "+str(T_REACH))
  logger.debug("Start of contact phase in simu cycles = "+str(T_CONTACT))
  logger.debug("Start of contact phase in simu cycles = "+str(OCP_TO_MPC_CYCLES))

  # logger.debug(ddp.problem.runningModels[0].differential.costs.costs['force'].cost.residual.reference.vector)

  # SIMULATE
  for i in range(config['N_simu']): 

      if(i%config['log_rate']==0 and config['LOG']): 
        print('')
        logger.info("SIMU step "+str(i)+"/"+str(config['N_simu']))
        print('')


      # If tracking phase enters the MPC horizon, start updating models from the end with tracking models      
      if(i >= T_REACH and i < T_REACH + NH_SIMU):
        # If current time matches an OCP node 
        if(int(T_REACH - i)%OCP_TO_SIMU_CYCLES == 0):
          # Select IAM
          if(int(T_REACH - i) == 0):
            logger.debug("Update terminal model to TRACKING")
            ddp.problem.terminalModel.differential.costs.costs["translation"].active = True
            ddp.problem.terminalModel.differential.costs.costs["translation"].cost.residual.reference = contactTranslationTarget
          else:
            node_idx = config['N_h'] + int((T_REACH - i)/OCP_TO_SIMU_CYCLES)
            ddp.problem.runningModels[node_idx].differential.costs.costs["translation"].active = True
            ddp.problem.runningModels[node_idx].differential.costs.costs["translation"].cost.residual.reference = contactTranslationTarget

      if(i == T_CONTACT): qref = sim_data.state_mea_SIMU[i, :nq]

      # If contact phase enters horizon start updating models from the the end with contact models
      if(i >= T_CONTACT and i <= T_CONTACT + NH_SIMU):
        # If current time matches an OCP node 
        if(int(T_CONTACT - i)%OCP_TO_SIMU_CYCLES == 0):
          # Select IAM
          if (int(T_CONTACT - i) == 0):
            logger.debug("Update terminal model to CONTACT")
            ddp.problem.terminalModel.differential.contacts.contacts["contact"].contact.active = True
            # ddp.problem.terminalModel.differential.costs.costs["force"].active = False #.weight = 0.
            # ddp.problem.terminalModel.differential.costs.costs["force"].cost.weight = MAX_FORCE_WEIGHT
          else:
            node_idx = config['N_h'] + int((T_CONTACT - i)/OCP_TO_SIMU_CYCLES)
            # weight = (MAX_FORCE_WEIGHT/config['N_h']) * node_idx
            ddp.problem.runningModels[node_idx].differential.contacts.contacts["contact"].contact.active = True
            ddp.problem.runningModels[node_idx].differential.costs.costs["force"].active = True 
            ddp.problem.runningModels[node_idx].differential.costs.costs["force"].cost.weight = MAX_FORCE_WEIGHT # weight #
            ddp.problem.runningModels[node_idx].differential.costs.costs["stateReg"].cost.residual.reference = np.concatenate([qref, np.zeros(nv)])    
            logger.debug(ddp.problem.runningModels[node_idx].differential.contacts.contacts["contact"].contact.active)
            # logger.debug(ddp.problem.runningModels[node_idx].differential.contacts.active)
      if(i == T_CONTACT + NH_SIMU): logger.debug("Fully inside the CONTACT phase")


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
      # Record interpolated desired state, control and force at SIM frequency
      sim_data.record_simu_cycle_desired(i)
      # Torque applied by motor on actuator : interpolate current torque and predicted torque 
      tau_ref_SIMU =  sim_data.y_ref_SIMU[-nu:] 
      # Actuation model ( tau_ref_SIMU ==> tau_mea_SIMU ) 
      tau_mea_SIMU = actuationModel.step(i, tau_ref_SIMU, sim_data.state_mea_SIMU[:,-nu:])   
      # RICCATI GAINS TO INTERPOLATE
      if(config['RICCATI']):
        K = ddp.K[0]
        alpha = np.exp(-2*np.pi*config['f_c']*config['dt'])
        Ktilde  = (1-alpha)*sim_data.OCP_TO_PLAN_RATIO*K
        Ktilde[:,2*nq:3*nq] += ( 1 - (1-alpha)*sim_data.OCP_TO_PLAN_RATIO )*np.eye(nq) # only for torques
        tau_mea_SIMU += Ktilde[:,:nq+nv].dot(ddp.problem.x0[:nq+nv] - sim_data.state_mea_SIMU[i,:nq+nv]) #position vel
        tau_mea_SIMU += Ktilde[:,:-nq].dot(ddp.problem.x0[:-nq] - sim_data.state_mea_SIMU[i,:-nq])       # torques
      #  Send output of actuation torque to the RBD simulator 
      robot_simulator.send_joint_command(tau_mea_SIMU)
      env.step()
      # Measure new state from simulation 
      q_mea_SIMU, v_mea_SIMU = robot_simulator.get_state()
      # Update pinocchio model
      robot_simulator.forward_robot(q_mea_SIMU, v_mea_SIMU)
      f_mea_SIMU = simulator_utils.get_contact_wrench(robot_simulator, id_endeff, sim_data.PIN_REF_FRAME)
      if(i%50==0): 
        logger.info("f_mea = "+str(f_mea_SIMU))
      # Record data (unnoised)
      y_mea_SIMU = np.concatenate([q_mea_SIMU, v_mea_SIMU, tau_mea_SIMU]).T 
      sim_data.state_mea_no_noise_SIMU[i+1, :] = y_mea_SIMU
      # Sensor model ( simulation state ==> noised / filtered state )
      sim_data.state_mea_SIMU[i+1, :] = sensingModel.step(i, y_mea_SIMU, sim_data.state_mea_SIMU)
      sim_data.force_mea_SIMU[i, :] = f_mea_SIMU


  # # # # # # # # # # #
  # PLOT SIM RESULTS  #
  # # # # # # # # # # #
  save_dir = '/tmp'
  save_name = config_name+'_bullet_'+\
                          '_BIAS='+str(config['SCALE_TORQUES'])+\
                          '_NOISE='+str(config['NOISE_STATE'] or config['NOISE_TORQUES'])+\
                          '_DELAY='+str(config['DELAY_OCP'] or config['DELAY_SIM'])+\
                          '_Fp='+str(sim_data.plan_freq/1000)+'_Fc='+str(sim_data.ctrl_freq/1000)+'_Fs'+str(sim_data.simu_freq/1000)
  #  Extract plot data from sim data
  plot_data = sim_data.extract_data(frame_of_interest=frame_of_interest)
  #  Plot results
  sim_data.plot_mpc_results(plot_data, which_plots=sim_data.WHICH_PLOTS,
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
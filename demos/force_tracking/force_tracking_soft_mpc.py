"""
@package force_feedback
@file demos/contact/aug_soft_contact_MPC.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2021, New York University & LAAS-CNRS
@date 2022-08-12
@brief Closed-loop MPC for force task 
"""

'''
The robot is tasked with exerting a constant normal force  
Trajectory optimization using Crocoddyl in closed-loop MPC 
(feedback from state x=(q,v,f), control u = tau) 
Using PyBullet simulator & GUI for rigid-body dynamics + visualization

The goal of this script is to simulate MPC with state feedback, optionally
imperfect actuation (bias, noise, delays) at higher frequency
'''


import sys
sys.path.append('.')

from core_mpc_utils.misc_utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger


import numpy as np  
np.set_printoptions(precision=4, linewidth=180)

from core_mpc_utils import path_utils, pin_utils, mpc_utils, misc_utils
from core_mpc_utils import ocp as ocp_utils

from soft_mpc.aug_ocp import OptimalControlProblemSoftContactAugmented
from soft_mpc.aug_data import DDPDataHandlerSoftContactAugmented, MPCDataHandlerSoftContactAugmented
from soft_mpc.utils import SoftContactModel3D, SoftContactModel1D



def main(robot_name='iiwa', simulator='bullet', PLOT_INIT=False):


  # # # # # # # # # # # # # # # # # # #
  ### LOAD ROBOT MODEL and SIMU ENV ### 
  # # # # # # # # # # # # # # # # # # # 
  # Read config file
  config, config_name = path_utils.load_config_file(__file__, robot_name)
  # config_name = 'soft_mpc_contact'
  # file = '/home/skleff/ws/workspace/src/force_feedback_dgh/config/soft_mpc_contact.yml' # 
  # config = path_utils.load_yaml_file(file)
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
    env, robot_simulator = simulator_utils.init_raisim_simulation(robot_name, dt=dt_simu, x0=x0)  
    robot = robot_simulator
  else:
    logger.error('Please choose a simulator from ["bullet", "raisim"] !')
  # Get dimensions 
  nq, nv = robot.model.nq, robot.model.nv; nu = nq
  # Placement of LOCAL end-effector frame w.r.t. WORLD frame
  frame_of_interest = config['frame_of_interest']
  id_endeff = robot.model.getFrameId(frame_of_interest)
  
  # simulator_utils.print_dynamics_info(1, 9)
  # EE translation target : contact point + vertical offset (radius of the ee ball)
  contactTranslationTarget = np.asarray(config['contactPosition']) + np.asarray(config['oPc_offset'])
  simulator_utils.display_ball(contactTranslationTarget, RADIUS=0.02, COLOR=[1.,0.,0.,0.2])
  # Display contact surface
  import pinocchio as pin
  contact_placement = pin.SE3(np.eye(3), np.asarray(config['contactPosition']))
  contactId = simulator_utils.display_contact_surface(contact_placement, bullet_endeff_ids=robot_simulator.bullet_endeff_ids)
  # Make the contact soft (e.g. tennis ball or sponge on the robot)
  simulator_utils.set_lateral_friction(contactId, 0.9)
  simulator_utils.set_contact_stiffness_and_damping(contactId, 10000, 500)

  # Contact model
    # Contact model anchor point
  oPc = contact_placement.translation + np.asarray(config['oPc_offset'])
  simulator_utils.display_ball(oPc, base_placement, RADIUS=0.01, COLOR=[1.,0.,0.,1.])
  if('1D' in config['contactType']):
      softContactModel = SoftContactModel1D(np.asarray(config['Kp']), np.asarray(config['Kv']), oPc, id_endeff, config['contactType'], config['pinRefFrame'])
  else:
      softContactModel = SoftContactModel3D(np.asarray(config['Kp']), np.asarray(config['Kv']), oPc, id_endeff, config['pinRefFrame'])

  softContactModel.print()
  
  # Measure initial force in pybullet
  f0 = simulator_utils.get_contact_wrench(robot_simulator, id_endeff, softContactModel.pinRefFrame)
  y0 = np.concatenate([x0, f0[-softContactModel.nc:]])  
  # Get corresponding external torques
  f_ext0 = pin_utils.get_external_joint_torques(contact_placement, f0, robot)   
  # # Visco-elastic prediction (model)
  # f_ext0 = softContactModel.computeExternalWrench(robot.model, robot.data)   

  # # # # # # # # # 
  ### OCP SETUP ###
  # # # # # # # # #
  # Compute initial gravity compensation torque torque   
  u0 = pin_utils.get_tau(q0, v0, np.zeros(nq), f_ext0, robot.model, np.zeros(nq))
  logger.debug(str(u0))
  # Setup Croco OCP and create solver
  ddp = OptimalControlProblemSoftContactAugmented(robot, config).initialize(y0, softContactModel, callbacks=False) #True)
  
  # !!! Deactivate all costs & contact models initially !!!
  models = list(ddp.problem.runningModels) + [ddp.problem.terminalModel]
  for k,m in enumerate(models):
    m.differential.active_contact = False
  MAX_FORCE_WEIGHT = config['frameForceWeight'] #1
  STEP_FORCE_WEIGHT = 0.01

  # Warmstart and solve
  xs_init = [y0 for i in range(config['N_h']+1)]
  us_init = [u0 for i in range(config['N_h'])]
  ddp.solve(xs_init, us_init, maxiter=100, isFeasible=False)

  
  if(PLOT_INIT):
    #  Plot
    ddp_handler = DDPDataHandlerSoftContactAugmented(ddp, softContactModel)
    ddp_data = ddp_handler.extract_data(ee_frame_name=frame_of_interest, ct_frame_name=frame_of_interest)
    _, _ = ddp_handler.plot_ddp_results(ddp_data, which_plots=config['WHICH_PLOTS'], 
                                                        colors=['r'], 
                                                        markers=['.'], 
                                                        SHOW=True)  
  
  # # # # # # # # # # #
  ### INIT MPC SIMU ###
  # # # # # # # # # # #
  sim_data = MPCDataHandlerSoftContactAugmented(config, robot, softContactModel.nc)
  sim_data.init_sim_data(y0)
    # Replan & control counters
  nb_plan = 0
  nb_ctrl = 0
  # Additional simulation blocks 
  communicationModel = mpc_utils.CommunicationModel(config)
  actuationModel     = mpc_utils.ActuationModel(config, nu=nu)
  sensingModel       = mpc_utils.SensorModel(config, naug=softContactModel.nc)
  velocityEstimator  = mpc_utils.VelocityEstimator(dt_simu)

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
  logger.debug("OCP to PLAN time ratio = "+str(OCP_TO_MPC_CYCLES))

  # SIMULATE
  for i in range(sim_data.N_simu): 

      if(i%config['log_rate']==0 and config['LOG']): 
        print('')
        logger.info("SIMU step "+str(i)+"/"+str(sim_data.N_simu))
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
            ddp.problem.terminalModel.differential.active_contact = True
            # ddp.problem.terminalModel.differential.costs.costs["translation"].active = False #.weight = 0.
            ddp.problem.terminalModel.differential.f_weight = MAX_FORCE_WEIGHT
          else:
            node_idx = config['N_h'] + int((T_CONTACT - i)/OCP_TO_SIMU_CYCLES)
            # weight = (MAX_FORCE_WEIGHT/config['N_h']) * node_idx
            ddp.problem.runningModels[node_idx].differential.active_contact = True
            ddp.problem.runningModels[node_idx].differential.f_weight = MAX_FORCE_WEIGHT # weight #
            ddp.problem.runningModels[node_idx].differential.costs.costs["stateReg"].cost.residual.reference = np.concatenate([qref, np.zeros(nv)])    
            
      if(i == T_CONTACT + NH_SIMU): logger.debug("Fully inside the CONTACT phase")

      # # If we are fully inside the contact phase, update nodes 1 by 1 until they all have max weights
      # if(i >= T_CONTACT + NH_SIMU and i <= T_CONTACT + 2*NH_SIMU):
      #   if(int(T_CONTACT + NH_SIMU - i)%OCP_TO_SIMU_CYCLES == 0):
      #     for node_idx in range(config['N_h']+1):
      #       if(node_idx == config['N_h']):
      #         weight = min(ddp.problem.terminalModel.differential.f_weight + STEP_FORCE_WEIGHT, MAX_FORCE_WEIGHT)
      #         ddp.problem.terminalModel.differential.f_weight = weight
      #       else:
      #         weight = min(ddp.problem.runningModels[node_idx].differential.f_weight + STEP_FORCE_WEIGHT, MAX_FORCE_WEIGHT)
      #         ddp.problem.runningModels[node_idx].differential.f_weight = weight

    # Solve OCP if we are in a planning cycle (MPC/planning frequency)
      if(i%int(sim_data.simu_freq/sim_data.plan_freq) == 0):
          # Reset x0 to measured state + warm-start solution
          ddp.problem.x0 = sim_data.state_mea_SIMU[i, :]
          xs_init = list(ddp.xs[1:]) + [ddp.xs[-1]]
          xs_init[0] = sim_data.state_mea_SIMU[i, :]
          us_init = list(ddp.us[1:]) + [ddp.us[-1]] 
          # Solve OCP 
          bench.start_timer()
          bench.start_croco_profiler()
          ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=False)
          bench.record_profiles()
          bench.stop_timer(nb_iter=ddp.iter)
          bench.stop_croco_profiler()
          # Record MPC predictions, cost references and solver data 
          sim_data.record_predictions(nb_plan, ddp)
          sim_data.record_cost_references(nb_plan, ddp)
          sim_data.record_solver_data(nb_plan, ddp) 
          # Model communication delay between computer & robot (buffered OCP solution)
          communicationModel.step(sim_data.y_pred, sim_data.u_curr)
          # Record interpolated desired state, control and force at MPC frequency
          sim_data.record_plan_cycle_desired(nb_plan)
          # Increment planning counter
          nb_plan += 1


    # If we are in a control cycle select reference torque to send to the actuator (motor driver input frequency)
      if(i%int(sim_data.simu_freq/sim_data.ctrl_freq) == 0):   
          # Record interpolated desired state, control and force at CTRL frequency
          sim_data.record_ctrl_cycle_desired(nb_ctrl)     
          # Increment control counter
          nb_ctrl += 1
          
          
    # Simulate actuation/sensing and step simulator (physics simulation frequency)
      # Record interpolated desired state, control and force at SIM frequency
      sim_data.record_simu_cycle_desired(i)
      # RICCATI GAINS TO INTERPOLATE
      tau_mea_SIMU = sim_data.u_ref_SIMU
      if(config['RICCATI']):
        tau_mea_SIMU += ddp.K[0][:,:nq+nv].dot(sim_data.y_ref_SIMU[:nq+nv] - sim_data.state_mea_SIMU[i,:][:nq+nv])
        # tau_mea_SIMU += ddp.K[0].dot(sim_data.y_ref_SIMU - sim_data.state_mea_SIMU[i,:])
      # Actuation model ( tau_ref_SIMU ==> tau_mea_SIMU ) 
      tau_mea_SIMU = actuationModel.step(i, tau_mea_SIMU, sim_data.ctrl_des_SIMU)
      # tau_mea_SIMU = actuationModel.step(i, sim_data.u_ref_SIMU, sim_data.ctrl_des_SIMU) if actuation BEFORE riccati?
      #  Send output of actuation torque to the RBD simulator 0
      robot_simulator.send_joint_command(tau_mea_SIMU)
      env.step()
      # Measure new state from simulation 
      if(config['FD_ESTIMATE_VELOCITY']==False):
        q_mea_SIMU, v_mea_SIMU = robot_simulator.get_state()
      else:
        q_mea_SIMU, _ = robot_simulator.get_state()
        v_mea_SIMU = velocityEstimator.FD1_estimate(q_mea_SIMU)
      # Update pinocchio model
      robot_simulator.forward_robot(q_mea_SIMU, v_mea_SIMU)
      f_mea_SIMU = simulator_utils.get_contact_wrench(robot_simulator, id_endeff, softContactModel.pinRefFrame)
      if(i%50==0): 
        logger.info("f_mea = "+str(f_mea_SIMU))
      # Record data (unnoised)
      y_mea_SIMU = np.concatenate([q_mea_SIMU, v_mea_SIMU, f_mea_SIMU[:softContactModel.nc]]).T 
      # y_mea_SIMU = np.concatenate([q_mea_SIMU, v_mea_SIMU, sim_data.y_pred[-softContactModel.nc:]]).T 
      sim_data.state_mea_no_noise_SIMU[i+1, :] = y_mea_SIMU
      # Sensor model ( simulation state ==> noised / filtered state )
      sim_data.state_mea_SIMU[i+1, :] = sensingModel.step(i, y_mea_SIMU, sim_data.state_mea_SIMU)

  bench.plot_timer()
  # bench.plot_profiles()
  # bench.plot_avg_profiles()
  # # # # # # # # # # #
  # PLOT SIM RESULTS  #
  # # # # # # # # # # #
  save_dir = '/tmp'
  save_name = config_name+'_'+simulator+'_'+\
                          '_BIAS='+str(config['SCALE_TORQUES'])+\
                          '_NOISE='+str(config['NOISE_STATE'] or config['NOISE_TORQUES'])+\
                          '_DELAY='+str(config['DELAY_OCP'] or config['DELAY_SIM'])+\
                          '_Fp='+str(sim_data.plan_freq/1000)+'_Fc='+str(sim_data.ctrl_freq/1000)+'_Fs'+str(sim_data.simu_freq/1000)
  # Extract plot data from sim data
  plot_data = sim_data.extract_data(frame_of_interest=frame_of_interest)
  # Plot results
  sim_data.plot_mpc_results(plot_data, which_plots=sim_data.WHICH_PLOTS,
                                      PLOT_PREDICTIONS=True, 
                                      pred_plot_sampling=int(sim_data.plan_freq/10),
                                      SAVE=False,
                                      SAVE_DIR=save_dir,
                                      SAVE_NAME=save_name,
                                      AUTOSCALE=False)
  # Save optionally
  if(config['SAVE_DATA']):
    sim_data.save_data(sim_data, save_name=save_name, save_dir=save_dir)





if __name__=='__main__':
    args = misc_utils.parse_MPC_script(sys.argv[1:])
    main(args.robot_name, args.simulator, args.PLOT_INIT)
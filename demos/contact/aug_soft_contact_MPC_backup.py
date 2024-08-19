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
from turtle import color
sys.path.append('.')

from core_mpc_utils.misc_utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger


import numpy as np  
np.set_printoptions(precision=4, linewidth=180)

from core_mpc_utils import path_utils, pin_utils, mpc_utils, misc_utils

from soft_mpc.aug_ocp import OptimalControlProblemSoftContactAugmented
from soft_mpc.aug_data import DDPDataHandlerSoftContactAugmented, MPCDataHandlerSoftContactAugmented
from soft_mpc.utils import SoftContactModel3D, SoftContactModel1D



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
    # logger.info("End-effector ids = "+str(robot_simulator.bullet_endeff_ids))
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
  ee_frame_placement = robot.data.oMf[robot.model.getFrameId(frame_of_interest)]
  oMf = robot.data.oMf[id_endeff]
  # Placement of contact surface w.r.t. LOCAL frame : offset in the normal direction
  # import pinocchio as pin
  contact_placement = ee_frame_placement.copy() # pin.SE3.Identity() #
  # contact_placement.translation = np.array([0.5, 0., 0.01])
  M_ct = robot.data.oMf[id_endeff].copy()
  contact_placement.translation = contact_placement.act( np.asarray(config['contact_plane_offset']) ) 
  contactId = simulator_utils.display_contact_surface(contact_placement, bullet_endeff_ids=robot_simulator.bullet_endeff_ids)
  # Set contact surface model parameters in simulator
  logger.debug("dynamics of contact surface BEFORE change  :")
  # simulator_utils.print_dynamics_info(contactId)
  # for bid in robot_simulator.bullet_endeff_ids:
  #   simulator_utils.print_dynamics_info(robot_simulator.robotId, bid)
  simulator_utils.set_lateral_friction(contactId, 0.9)
  # simulator_utils.set_contact_stiffness_and_damping(contactId, 1e6, 1e3)
  simulator_utils.set_contact_stiffness_and_damping(contactId, 1e4, 10)
  # simulator_utils.set_contact_stiffness_and_damping(contactId, 1e5, 316)
  # logger.debug("dynamics of contact surface AFTER change  :")
  # simulator_utils.print_dynamics_info(contactId)

  # # Warm start state = IK of circle trajectory
  # logger.info("Computing warm-start using Inverse Kinematics...")
  # q_ws, v_ws, eps = pin_utils.IK_placement(robot, q_ws, id_endeff, pin.SE3., DT=1e-2, IT_MAX=100)

  # Contact model
    # Contact model
  oPc = contact_placement.translation + np.asarray(config['oPc_offset'])
  simulator_utils.display_ball(oPc, base_placement, RADIUS=0.01, COLOR=[1.,0.,0.,1.])
  if('1D' in config['contactType']):
      softContactModel = SoftContactModel1D(config['Kp'], config['Kv'], oPc, id_endeff, config['contactType'], config['pinRefFrame'])
  else:
      softContactModel = SoftContactModel3D(config['Kp'], config['Kv'], oPc, id_endeff, config['pinRefFrame'])

  f0 = simulator_utils.get_contact_wrench(robot_simulator, id_endeff, softContactModel.pinRefFrame)
  y0 = np.concatenate([x0, f0[-softContactModel.nc:]])  
  # y0 = np.concatenate([x0, softContactModel.computeForce_(robot.model, q0, v0)])  

  # logger.debug("initial augmented state = ")
  # logger.debug(str(y0))

  # import time
  # time.sleep(1)

  # # # # # # # # # 
  ### OCP SETUP ###
  # # # # # # # # #
  # Compute initial external wrench for each joint
  # f_ext0 = softContactModel.computeExternalWrench(robot.model, robot.data)    # Visco-elastic prediction (model)
  f_ext0 = pin_utils.get_external_joint_torques(contact_placement, f0, robot)   # Measured in PyBullet initially
  u0 = pin_utils.get_tau(q0, v0, np.zeros(nq), f_ext0, robot.model, np.zeros(nq))
  config['ctrlRegRef'] = np.zeros(nu) # u0
  # Setup Croco OCP and create solver
  ddp = OptimalControlProblemSoftContactAugmented(robot, config).initialize(y0, softContactModel, callbacks=False) #True)
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

  # SIMULATE
  for i in range(sim_data.N_simu): 

      if(i%config['log_rate']==0 and config['LOG']): 
        print('')
        logger.info("SIMU step "+str(i)+"/"+str(sim_data.N_simu))
        print('')

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
      # RICCATI GAINS TO INTERPOLATE (only state for now)
      tau_mea_SIMU = sim_data.u_ref_SIMU
      if(config['RICCATI']):
        # tau_mea_SIMU = ddp.K[0][:,:nq+nv].dot(ddp.problem.x0[:nq+nv] - sim_data.state_mea_SIMU[i,:][:nq+nv])
        tau_mea_SIMU += ddp.K[0].dot(ddp.problem.x0 - sim_data.state_mea_SIMU[i,:])
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
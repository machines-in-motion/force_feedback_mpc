"""
@package force_feedback
@file aug_soft_sanding_MPC.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2021, New York University & LAAS-CNRS
@date 2023-04-04
@brief Closed-loop 'augmented soft contact' MPC for sanding task 
"""

'''
The robot is tasked with exerting a constant normal force  while drawing a circle
Trajectory optimization using Crocoddyl in closed-loop MPC 
(feedback from state x=(q,v,f), control u = tau) 
Using PyBullet simulator & GUI for rigid-body dynamics + visualization

The goal of this script is to simulate MPC with state feedback, optionally
imperfect actuation (bias, noise, delays) at higher frequency
'''


import sys
sys.path.append('.')

from croco_mpc_utils.utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger


import numpy as np  
np.set_printoptions(precision=4, linewidth=180)
RANDOM_SEED = 1 #19


from force_feedback_mpc.core_mpc_utils import path_utils, misc_utils, mpc_utils
from force_feedback_mpc.core_mpc_utils import sim_utils as simulator_utils

from croco_mpc_utils import pinocchio_utils as pin_utils
from croco_mpc_utils.math_utils import circle_point_WORLD

from force_feedback_mpc.soft_mpc.aug_ocp import OptimalControlProblemSoftContactAugmented
from force_feedback_mpc.soft_mpc.aug_data import OCPDataHandlerSoftContactAugmented, MPCDataHandlerSoftContactAugmented
from force_feedback_mpc.soft_mpc.utils import SoftContactModel1D
from croco_mpc_utils.utils import load_yaml_file

import mim_solvers
from mim_robots.robot_loader import load_bullet_wrapper
from mim_robots.pybullet.env import BulletEnvWithGround
import pybullet as p

RESET_ANCHOR_POINT = True

import time
import pinocchio as pin
import os

# @profile
def solveOCP(q, v, f, solver, nb_iter, target_reach, anchor_point, TASK_PHASE, target_force):
        t = time.time()
        # Nullify the (x,y) forces in WORLD to yield a 1D contact model
        # f_ = f.copy()
        # f_[0] = 0.  # 1Dz contact
        # f_[1] = 0.  # 1Dz contact
        # Update initial state + warm-start
        x = np.concatenate([q, v, f]) # 1Dz contact
        solver.problem.x0 = x
        xs_init = list(solver.xs)[1:] + [list(solver.xs)[-1]]
        xs_init[0] = x
        us_init = list(solver.us)[1:] + [list(solver.us)[-1]] 
        # Get OCP nodes
        m = list(solver.problem.runningModels) + [solver.problem.terminalModel]
        # Update OCP for reaching phase
        if(TASK_PHASE == 1):
            for k in range( solver.problem.T+1 ):
                m[k].differential.costs.costs["translation"].active = True
                m[k].differential.costs.costs["translation"].cost.residual.reference = target_reach[k]
        # Update OCP for "increase weights" phase
        if(TASK_PHASE == 2):
            for k in range( solver.problem.T+1 ):
                w = min(2.*k , 10.)
                m[k].differential.costs.costs["translation"].cost.residual.reference = target_reach[k]
                m[k].differential.costs.costs["translation"].weight = w
        # Update OCP for contact phase
        if(TASK_PHASE == 3):
            for k in range( solver.problem.T+1 ):
                m[k].differential.active_contact = True
                m[k].differential.f_des = np.array([-target_force[k]])
                m[k].differential.oPc = anchor_point
                m[k].differential.costs.costs["translation"].cost.residual.reference = target_reach[k]
                m[k].differential.costs.costs["translation"].cost.activation.weights = np.array([1., 1., 0.])
                m[k].differential.costs.costs["translation"].weight = 50.
                m[k].differential.costs.costs['rotation'].active = True
                m[k].differential.costs.costs['rotation'].cost.residual.reference = pin.utils.rpyToMatrix(np.pi, 0., np.pi)                      
        # Update OCP for circle phase
        if(TASK_PHASE == 4):
            for k in range( solver.problem.T+1 ):
                m[k].differential.costs.costs["translation"].weight = 150. 
                m[k].differential.f_des = np.array([-target_force[k]]) 
                m[k].differential.costs.costs["translation"].cost.residual.reference = target_reach[k]
        # Solve OCP 
        solver.solve(xs_init, us_init, maxiter=nb_iter, isFeasible=False)
        # Send solution to parent process + riccati gains
        solve_time = time.time()
        return solver.us[0], solver.xs[1], solver.K[0], solve_time - t, solver.iter, solver.KKT




def main(SAVE_DIR, TORQUE_TRACKING):

  # # # # # # # # # # # # # # # # # # #
  ### LOAD ROBOT MODEL and SIMU ENV ### 
  # # # # # # # # # # # # # # # # # # # 
  # Read config file
  config_name = 'polishing_soft'
  config = path_utils.load_yaml_file(os.path.dirname(os.path.realpath(__file__))+'/'+config_name+'.yml')
  # Create a simulation environment & simu-pin wrapper 
  dt_simu = 1./float(config['simu_freq'])  
  q0 = np.asarray(config['q0'])
  v0 = np.asarray(config['dq0'])
  x0 = np.concatenate([q0, v0])  
  env             = BulletEnvWithGround(dt=dt_simu, server=p.GUI)
  robot_simulator = load_bullet_wrapper('iiwa_ft_sensor_shell', locked_joints=['A7'])
  env.add_robot(robot_simulator) 
  robot_simulator.reset_state(q0, v0)
  robot_simulator.forward_robot(q0, v0)
  robot = robot_simulator.pin_robot
  
  # Get dimensions 
  nq, nv = robot.model.nq, robot.model.nv; nu = nq
  # Placement of LOCAL end-effector frame w.r.t. WORLD frame
  frame_of_interest = config['frame_of_interest']
  id_endeff = robot.model.getFrameId(frame_of_interest)
  
  # EE translation target : contact point + vertical offset (radius of the ee ball)
  oPc = np.asarray(config['contactPosition']) + np.asarray(config['oPc_offset'])
  simulator_utils.display_ball(oPc, RADIUS=0.02, COLOR=[1.,0.,0.,0.2])
  # Display contact surface + optional tilt
  import pinocchio as pin
  contact_placement   = pin.SE3(np.eye(3), np.asarray(config['contactPosition']))
  contact_placement_0 = contact_placement.copy()
  TILT_RPY = np.zeros(3)
  if(config['TILT_SURFACE']):
    TILT_RPY = [config['TILT_PITCH_LOCAL_DEG']*np.pi/180, 0., 0.]
    contact_placement = pin_utils.rotate(contact_placement, rpy=TILT_RPY)
  contact_surface_bulletId = simulator_utils.display_contact_surface(pin.SE3(np.eye(3), np.asarray(config['contactPosition'])), bullet_endeff_ids=robot_simulator.bullet_endeff_ids)
  # Make the contact soft (e.g. tennis ball or sponge on the robot)
  simulator_utils.set_lateral_friction(contact_surface_bulletId, 0.5)
  simulator_utils.set_contact_stiffness_and_damping(contact_surface_bulletId, 10000, 500)
  
  # Contact model
  softContactModel = SoftContactModel1D(Kp=np.asarray(config['Kp']), 
                                        Kv=np.asarray(config['Kv']), 
                                        oPc=oPc,
                                        frameId=id_endeff, 
                                        contactType=config['contactType'], 
                                        pinRef=config['pinRefFrame'])
  
  # Measure initial force in pybullet
  f0 = np.zeros(6)
  assert(softContactModel.nc == 1)
  assert(softContactModel.pinRefFrame == pin.LOCAL_WORLD_ALIGNED)
  softContactModel.print()
  
  MASK = softContactModel.mask
  y0 = np.concatenate([ x0, np.array([f0[MASK]]) ])  
  RESET_ANCHOR_POINT = bool(config['RESET_ANCHOR_POINT'])
  anchor_point = oPc.copy()
  
  # # # # # # # # # 
  ### OCP SETUP ###
  # # # # # # # # #
  # Compute initial gravity compensation torque torque   
  f_ext0 = [pin.Force.Zero() for _ in robot.model.joints] # pin_utils.get_external_joint_torques(contact_placement, f0, robot)   
  y0 = np.concatenate([x0, f0[-softContactModel.nc:]])  
  u0 = pin_utils.get_tau(q0, v0, np.zeros(nq), f_ext0, robot.model, np.zeros(nq))
  # Warmstart and solve
  xs_init = [y0 for i in range(config['N_h']+1)]
  us_init = [u0 for i in range(config['N_h'])] 
  # Setup Croco OCP and create solver
  ocp = OptimalControlProblemSoftContactAugmented(robot, config).initialize(y0, softContactModel)
  solver = mim_solvers.SolverSQP(ocp)
  solver.regMax                 = 1e6
  solver.reg_max                = 1e6
  solver.termination_tolerance  = 0.0001 
  solver.use_filter_line_search = True
  solver.filter_size            = config['maxiter']
  # !!! Deactivate all costs & contact models initially !!!
  models = list(solver.problem.runningModels) + [solver.problem.terminalModel]
  for k,m in enumerate(models):
      m.differential.costs.costs["translation"].active = False
      m.differential.active_contact = False
      m.differential.f_des = np.zeros(1)
      m.differential.cost_ref = pin.LOCAL_WORLD_ALIGNED
      m.differential.costs.costs['rotation'].active = False
      m.differential.costs.costs['rotation'].cost.residual.reference = pin.utils.rpyToMatrix(np.pi, 0., np.pi)
  
  solver.solve(xs_init, us_init, maxiter=100, isFeasible=False)
  
  # Setup tracking problem with circle ref EE trajectory + Warm start state = IK of circle trajectory
  OCP_TO_CTRL_RATIO = int(config['dt']/dt_simu)
        
  RADIUS = config['frameCircleTrajectoryRadius'] 
  OMEGA  = config['frameCircleTrajectoryVelocity']
  
  # Force trajectory
  F_MIN = 5.
  F_MAX = config['frameForceRef'][2]
  N_total = int((config['T_tot'] - config['T_CONTACT']) / dt_simu + config['N_h']*OCP_TO_CTRL_RATIO)
  N_ramp  = int((config['T_RAMP'] - config['T_CONTACT']) / dt_simu)
  
  target_force_traj             = np.zeros((N_total, 3))
  target_force_traj[:N_ramp, 2] = [F_MIN + (F_MAX - F_MIN)*i/N_ramp for i in range(N_ramp)]
  target_force_traj[N_ramp:, 2] = F_MAX
  target_force                  = np.zeros(config['N_h']+1)
  
  # Circle trajectory 
  N_total_pos = int((config['T_tot'] - config['T_REACH'])/dt_simu + config['N_h']*OCP_TO_CTRL_RATIO)
  N_circle    = int((config['T_tot'] - config['T_CIRCLE'])/dt_simu + config['N_h']*OCP_TO_CTRL_RATIO )
  target_position_traj = np.zeros( (N_total_pos, 3) )
  # absolute desired position
  target_position_traj[0:N_circle, :] = [np.array([oPc[0] + RADIUS * (1-np.cos(i*dt_simu*OMEGA)), 
                                                    oPc[1] - RADIUS * np.sin(i*dt_simu*OMEGA),
                                                    oPc[2]]) for i in range(N_circle)]
  target_position_traj[N_circle:, :] = target_position_traj[N_circle-1,:]
  target_position = np.zeros((config['N_h']+1, 3)) 
  target_position[:,:] = oPc.copy() 
  # import matplotlib.pyplot as plt
  # plt.plot(target_position_traj[:,0])
  # plt.plot(target_position_traj[:,1])
  # plt.show()
  
  #   # Plot initial solution
  #   # if(PLOT_INIT):
  #   ocp_data_handler = OCPDataHandlerSoftContactAugmented(solver.problem, softContactModel)
  #   ocp_data = ocp_data_handler.extract_data(solver.xs, solver.us, robot.model)
  #   _, _ = ocp_data_handler.plot_ocp_results(ocp_data, which_plots=config['WHICH_PLOTS'], 
  #                                                         colors=['r'], 
  #                                                         markers=['.'], 
  #                                                         SHOW=True)  
  
  
  
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
  actuationModel     = mpc_utils.ActuationModel(config, nu=nu, SEED=RANDOM_SEED)
  sensingModel       = mpc_utils.SensorModel(config, naug=softContactModel.nc, SEED=RANDOM_SEED)
  if(int(TORQUE_TRACKING) == 0):
      use = False
  else:
      use = True
  torqueController   = mpc_utils.LowLevelTorqueController(config, nu=nu, use=use)
  antiAliasingFilter = mpc_utils.AntiAliasingFilter()
  
  # Display target circle  trajectory (reference)
  if(config['DISPLAY_EE']):
    nb_points = 20 
    for i in range(nb_points):
        t = (i/nb_points)*2*np.pi/OMEGA
        # pl = pin_utils.rotate(contact_placement_0, rpy=TILT_RPY)
        pos = circle_point_WORLD(t, contact_placement_0, radius=RADIUS, omega=OMEGA, LOCAL_PLANE=config['CIRCLE_LOCAL_PLANE'])
        simulator_utils.display_ball(pos, RADIUS=0.01, COLOR=[1., 0., 0., 1.])
  
    draw_rate = 1000
  
  
  # # # # # # # # # # # #
  ### SIMULATION LOOP ###
  # # # # # # # # # # # #
  # from core_mpc.analysis_utils import MPCBenchmark
  # bench = MPCBenchmark()
  
  # Horizon in simu cycles
  TASK_PHASE       = 0
  NH_SIMU          = int(config['N_h']*config['dt']/sim_data.dt_simu)
  T_REACH          = int(config['T_REACH']/sim_data.dt_simu)
  T_TRACK          = int(config['T_TRACK']/sim_data.dt_simu)
  T_CONTACT        = int(config['T_CONTACT']/sim_data.dt_simu)
  T_CIRCLE         = int(config['T_CIRCLE']/sim_data.dt_simu)
  OCP_TO_MPC_RATIO = config['dt'] / sim_data.dt_plan
  logger.debug("Size of MPC horizon in simu cycles     = "+str(NH_SIMU))
  logger.debug("Start of REACH phase in simu cycles    = "+str(T_REACH))
  logger.debug("Start of TRACK phase in simu cycles    = "+str(T_TRACK))
  logger.debug("Start of CONTACT phase in simu cycles  = "+str(T_CONTACT))
  logger.debug("Start of RAMP phase in simu cycles     = "+str(T_CONTACT))
  logger.debug("Start of CIRCLE phase in simu cycles   = "+str(T_CIRCLE))
  logger.debug("OCP to PLAN ratio (# of re-replannings between two OCP nodes) = "+str(OCP_TO_MPC_RATIO))
  logger.debug("OCP to SIMU ratio (# of simulate steps between two OCP nodes) = "+str(OCP_TO_CTRL_RATIO))
  
  # SIMULATE
  sim_data.tau_mea_SIMU[0,:] = solver.us[0]
  count = 0
  f_err = []
  p_err = []
  
  for i in range(sim_data.N_simu): 
      
      if(i%config['log_rate']==0 and config['LOG']): 
        print('')
        logger.info("SIMU step "+str(i)+"/"+str(sim_data.N_simu))
        print('')
  
      # # # # # # # # # 
      # # Update OCP  #
      # # # # # # # # # 
      time_to_reach   = int(i - T_REACH)
      time_to_track   = int(i - T_TRACK)
      time_to_contact = int(i - T_CONTACT)
      time_to_circle  = int(i - T_CIRCLE)
  
  
  
      if(time_to_reach == 0): 
          logger.warning("Entering reaching phase")
      # If tracking phase enters the MPC horizon, start updating models from the end with tracking models      
      if(0 <= time_to_reach and time_to_reach <= NH_SIMU):
          TASK_PHASE = 1
  
      if(time_to_track == 0): 
          logger.warning("Entering tracking phase")
      # If "increase weights" phase enters the MPC horizon, start updating models from the end with tracking models      
      if(0 <= time_to_track and time_to_track <= NH_SIMU):
          TASK_PHASE = 2
          
      if(time_to_contact == 0): 
          # Record end-effector position at the time of the contact switch
          position_at_contact_switch = robot.data.oMf[id_endeff].translation.copy()
          target_position[:,:] = position_at_contact_switch.copy()
          logger.warning("Entering contact phase")
          if(RESET_ANCHOR_POINT): 
              anchor_point = position_at_contact_switch + config['oPc_offset']
          logger.warning("   Anchor point = "+str(anchor_point)+" )")
      # If contact phase enters horizon start updating models from the the end with contact models
      if(0 <= time_to_contact and time_to_contact <= NH_SIMU):
          TASK_PHASE = 3
  
  
      if(0 <= time_to_contact and time_to_contact%OCP_TO_CTRL_RATIO == 0):
          tf  = time_to_contact + (config['N_h']+1)*OCP_TO_CTRL_RATIO
          target_force = target_force_traj[time_to_contact:tf:OCP_TO_CTRL_RATIO, 2]
  
      if(time_to_circle == 0): 
          logger.warning("Entering circle phase")
      # If circle tracking phase enters the MPC horizon, start updating models from the end with tracking models      
      if(0 <= time_to_circle and time_to_circle <= NH_SIMU):
          TASK_PHASE = 4
  
      if(0 <= time_to_circle and time_to_circle%OCP_TO_CTRL_RATIO == 0):
          # set position refs over current horizon
          tf  = time_to_circle + (config['N_h']+1)*OCP_TO_CTRL_RATIO
          # Target in (x,y)  = circle trajectory + offset to start from current position instead of absolute target
          target_position[:,:2] = target_position_traj[time_to_circle:tf:OCP_TO_CTRL_RATIO,:2] + position_at_contact_switch[:2] - oPc[:2]
  
  
  
      # # # # # # # # #
      # # Solve OCP # #
      # # # # # # # # #
      # Solve OCP if we are in a planning cycle (MPC/planning frequency)
      if(i%int(sim_data.simu_freq/sim_data.plan_freq) == 0):
          # Anti-aliasing filter for measured state
          x_filtered = antiAliasingFilter.step(nb_plan, i, sim_data.plan_freq, sim_data.simu_freq, sim_data.state_mea_SIMU)
          # Reset x0 to measured state + warm-start solution
          q = x_filtered[:nq]
          v = x_filtered[nq:nq+nv]
          f = x_filtered[-softContactModel.nc:]
          # Solve OCP 
          solveOCP(q, v, f, solver, config['maxiter'], target_position, anchor_point, TASK_PHASE, target_force)
          # Record MPC predictions, cost references and solver data 
          sim_data.record_predictions(nb_plan, solver)
          sim_data.record_cost_references(nb_plan, solver)
          sim_data.record_solver_data(nb_plan, solver) 
          # Model communication delay between computer & robot (buffered OCP solution)
          communicationModel.step(sim_data.y_pred, sim_data.u_curr)
          # Record interpolated desired state, control and force at MPC frequency
          sim_data.record_plan_cycle_desired(nb_plan)
          # Increment planning counter
          nb_plan += 1
          # torqueController.reset_integral_error()
  
  
  
      # # # # # # # # # #
      # # Send policy # #
      # # # # # # # # # #
      # If we are in a control cycle send reference torque to motor driver and compute the motor torque
      if(i%int(sim_data.simu_freq/sim_data.ctrl_freq) == 0):   
          # Anti-aliasing filter on measured torques (sim-->ctrl)
          tau_mea_CTRL            = antiAliasingFilter.step(nb_ctrl, i, sim_data.ctrl_freq, sim_data.simu_freq, sim_data.tau_mea_SIMU)
          tau_mea_derivative_CTRL = antiAliasingFilter.step(nb_ctrl, i, sim_data.ctrl_freq, sim_data.simu_freq, sim_data.tau_mea_derivative_SIMU)
          # Select the desired torque 
          tau_des_CTRL = sim_data.u_curr.copy()
          # Optionally interpolate to the control frequency using Riccati gains
          if(config['RICCATI']):
            y_filtered = antiAliasingFilter.step(nb_ctrl, i, sim_data.ctrl_freq, sim_data.simu_freq, sim_data.state_mea_SIMU)
            # tau_des_CTRL += solver.K[0][:,:nq+nv].dot(solver.problem.x0[:nq+nv] - y_filtered[:nq+nv]) #position vel
            tau_des_CTRL += solver.K[0].dot(solver.problem.x0 - y_filtered) #position vel force
          # Compute the motor torque 
          tau_mot_CTRL = torqueController.step(tau_des_CTRL, tau_mea_CTRL, tau_mea_derivative_CTRL)
          # Increment control counter
          nb_ctrl += 1
  
  
          
      # Simulate actuation
      tau_mea_SIMU = actuationModel.step(tau_mot_CTRL, joint_vel=sim_data.state_mea_SIMU[i,nq:nq+nv])
      # Step PyBullet simulator
      robot_simulator.send_joint_command(tau_mea_SIMU)
      env.step()
      # Measure new state + forces from simulation 
      q_mea_SIMU, v_mea_SIMU = robot_simulator.get_state()
      robot_simulator.forward_robot(q_mea_SIMU, v_mea_SIMU)
      f_mea_SIMU = robot_simulator.end_effector_forces()[1][0]
      fz_mea_SIMU = np.array([f_mea_SIMU[2]])
      if(i%1000==0): 
        logger.info("f_mea  = "+str(f_mea_SIMU))
        
      # Compute force and position errors
      if(i >= T_CIRCLE):
        count+=1
        f_err.append(np.abs(f_mea_SIMU[2] - target_force[0]))
        p_err.append(np.abs(robot_simulator.pin_robot.data.oMf[id_endeff].translation[:2] - target_position[0][:2]))
      
      # Record data (unnoised)
      y_mea_SIMU = np.concatenate([q_mea_SIMU, v_mea_SIMU, fz_mea_SIMU]).T 
      # Simulate sensing 
      y_mea_no_noise_SIMU = sensingModel.step(y_mea_SIMU)
      # Record measurements of state, torque and forces 
      sim_data.record_simu_cycle_measured(i, y_mea_SIMU, y_mea_no_noise_SIMU, tau_mea_SIMU)
  
      # Display real 
      if(config['DISPLAY_EE'] and i%draw_rate==0):
        pos = robot_simulator.pin_robot.data.oMf[id_endeff].translation.copy()
        simulator_utils.display_ball(pos, RADIUS=0.03, COLOR=[0.,0.,1.,0.3])
  
  # bench.plot_timer()
  # # # # # # # # # # #
  # PLOT SIM RESULTS  #
  # # # # # # # # # # #
  #   FILTER = 1000
  #   if(FILTER > 0):
  #       from core_mpc import analysis_utils
  #       print("Filtering")
  #       f_err = analysis_utils.moving_average_filter(np.array([f_err.copy()]), FILTER)
  #       p_err = analysis_utils.moving_average_filter(np.array([p_err.copy()]), FILTER)
  #   import matplotlib.pyplot as plt
  #   tlin = np.linspace(0, (sim_data.N_simu-1)*dt_simu, sim_data.N_simu)
  #   plt.plot(tlin, kkt)
  #   plt.plot(tlin, nbiter)
  #   plt.show()
  logger.warning("count = "+str(count))
  logger.warning("------------------------------------")
  logger.warning("------------------------------------")
  logger.warning(" Fz MAE  = "+str(np.mean(f_err)))
  logger.warning(" Pxy MAE = "+str(np.mean(p_err)))
  logger.warning("------------------------------------")
  logger.warning("------------------------------------")
  
  save_dir = '/tmp'
  save_name = config_name+\
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
    # args = misc_utils.parse_MPC_script(sys.argv[1:])
    # main(args.SAVE_DIR)
    main(sys.argv[1], sys.argv[2])
"""
@package force_feedback
@file sanding_MPC.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2021, New York University & LAAS-CNRS
@date 2021-10-28
@brief Closed-loop classical MPC for sanding task
"""

'''
The robot is tasked with exerting a constant normal force with its EE 
while drawing a circle on the contact surface
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

from croco_mpc_utils.utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger


import numpy as np  
np.set_printoptions(precision=4, linewidth=180)
RANDOM_SEED = 1 #19
logger.debug("1")


from core_mpc_utils import mpc_utils, path_utils
from core_mpc_utils import sim_utils as simulator_utils

from croco_mpc_utils import pinocchio_utils as pin_utils
from croco_mpc_utils.ocp_data import MPCDataHandlerClassical, OCPDataHandlerClassical
from croco_mpc_utils.ocp import OptimalControlProblemClassical
from croco_mpc_utils.math_utils import circle_point_WORLD
from croco_mpc_utils.utils import load_yaml_file


import mim_solvers
from mim_robots.robot_loader import load_bullet_wrapper
from mim_robots.pybullet.env import BulletEnvWithGround
import pybullet as p

import time
import pinocchio as pin

import os
logger.debug("2")

def solveOCP(q, v, solver, nb_iter, target_reach, TASK_PHASE, target_force):
        # print(target_force)
        t = time.time()
        # Update initial state + warm-start
        x = np.concatenate([q, v])
        solver.problem.x0 = x
        xs_init = list(solver.xs[1:]) + [solver.xs[-1]]
        xs_init[0] = x
        us_init = list(solver.us[1:]) + [solver.us[-1]] 
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
                w = min(2.*k , 5)
                m[k].differential.costs.costs["translation"].active = True
                m[k].differential.costs.costs["translation"].cost.residual.reference = target_reach[k]
                m[k].differential.costs.costs["translation"].weight = w
        # Update OCP for contact phase
        if(TASK_PHASE == 3):
            for k in range( solver.problem.T+1 ):
                m[k].differential.costs.costs["translation"].active = True
                m[k].differential.costs.costs["translation"].cost.residual.reference = target_reach[k]
                m[k].differential.costs.costs["translation"].weight = 2.
                m[k].differential.costs.costs['rotation'].active = True
                m[k].differential.costs.costs['rotation'].cost.residual.reference = pin.utils.rpyToMatrix(np.pi, 0, np.pi)
                # activate contact and force cost
                m[k].differential.contacts.changeContactStatus("contact", True)
                if(k!=solver.problem.T):
                    fref = pin.Force(np.array([0., 0., target_force[k], 0., 0., 0.]))
                    m[k].differential.costs.costs["force"].active = True
                    m[k].differential.costs.costs["force"].cost.residual.reference = fref
                # if(k!=solver.problem.T and k != 0):
                    m[k].differential.constraints.changeConstraintStatus('forceBox', True)
                
        # Update OCP for circle phase
        if(TASK_PHASE == 4):
            for k in range( solver.problem.T+1 ):
                m[k].differential.costs.costs["translation"].active = True
                m[k].differential.costs.costs["translation"].cost.residual.reference = target_reach[k]
                m[k].differential.costs.costs["translation"].cost.activation.weights = np.array([1., 1., 0.])
                m[k].differential.costs.costs["translation"].weight = 60. 
                m[k].differential.costs.costs['rotation'].active = True
                m[k].differential.costs.costs['rotation'].cost.residual.reference = pin.utils.rpyToMatrix(np.pi, 0, np.pi)
                # activate contact and force cost
                m[k].differential.contacts.changeContactStatus("contact", True)
                if(k!=solver.problem.T):
                    fref = pin.Force(np.array([0., 0., target_force[k], 0., 0., 0.]))
                    m[k].differential.costs.costs["force"].active = True
                    m[k].differential.costs.costs["force"].cost.residual.reference = fref
        # Solve OCP 
        solver.solve(xs_init, us_init, maxiter=nb_iter, isFeasible=False)
        # Send solution to parent process + riccati gains
        solve_time = time.time()
        return solver.us[0], solver.xs[1], solver.K[0], solve_time - t, solver.iter, solver.KKT

logger.debug("3")

TORQUE_TRACKING=0
# def main(SAVE_DIR, TORQUE_TRACKING):

# # # # # # # # # # # # # # # # # # #
### LOAD ROBOT MODEL and SIMU ENV ### 
# # # # # # # # # # # # # # # # # # # 
# Read config file
config_name = 'polishing_classical_obstacle'
config = path_utils.load_yaml_file(os.path.dirname(os.path.realpath(__file__))+'/'+config_name+'.yml')
# Create a simulation environment & simu-pin wrapper 
dt_simu = 1./float(config['simu_freq'])  
q0 = np.asarray(config['q0'])
v0 = np.asarray(config['dq0'])
x0 = np.concatenate([q0, v0])  
env             = BulletEnvWithGround(dt=dt_simu, server=p.GUI)
robot_simulator = load_bullet_wrapper('iiwa_convex_ft_sensor_shell', locked_joints=['A7'])
env.add_robot(robot_simulator) 
robot_simulator.reset_state(q0, v0)
robot_simulator.forward_robot(q0, v0)
robot = robot_simulator.pin_robot
logger.debug("4")
# Get dimensions 
nq, nv = robot.model.nq, robot.model.nv; nu = nq
# Placement of LOCAL end-effector frame w.r.t. WORLD frame
frame_of_interest = config['frame_of_interest']
id_endeff = robot.model.getFrameId(frame_of_interest)
logger.debug("5")
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
logger.debug("6")
# Create obstacle
capsule_id = simulator_utils.setup_obstacle_collision(robot_simulator, robot, config)
# # # # # # # # # 
### OCP SETUP ###
# # # # # # # # # 
# Init shooting problem and solver
ocp = OptimalControlProblemClassical(robot, config).initialize(x0)
# Warmstart and solve
xs_init = [x0 for i in range(config['N_h']+1)]
us_init = ocp.quasiStatic(xs_init[:-1]) 
# us_init = [u0 for i in range(config['N_h'])] 
solver = mim_solvers.SolverCSQP(ocp)
solver.with_callbacks         = config['with_callbacks']
solver.use_filter_line_search = config['use_filter_line_search']
solver.filter_size            = config['filter_size']
solver.warm_start             = config['warm_start']
solver.termination_tolerance  = config['solver_termination_tolerance']
solver.max_qp_iters           = config['max_qp_iter']
solver.eps_abs                = config['qp_termination_tol_abs']
solver.eps_rel                = config['qp_termination_tol_rel']
solver.warm_start_y           = config['warm_start_y']
solver.reset_rho              = config['reset_rho']  
solver.mu_dynamic             = config["mu_dynamic"]
solver.mu_constraint          = config["mu_constraint"]
solver.regMax                 = 1e6
solver.reg_max                = 1e6
# !!! Deactivate all costs & contact models initially !!!
models = list(solver.problem.runningModels) + [solver.problem.terminalModel]
datas = list(solver.problem.runningDatas) + [solver.problem.terminalData]
for k,m in enumerate(models):
    # logger.debug(str(m.differential.costs.active.tolist()))
    m.differential.costs.costs["translation"].active = False
    m.differential.contacts.changeContactStatus("contact", False)
    m.differential.costs.costs['rotation'].active = False
    m.differential.costs.costs['rotation'].cost.residual.reference = pin.utils.rpyToMatrix(np.pi, 0., np.pi)
    # set each collision constraint bounds to [0, inf]
  
    if(k!= config['N_h']):
        m.differential.constraints.constraints['forceBox'].constraint.updateBounds(
                    np.array([0.]),
                    np.array([25])) 
        m.differential.constraints.changeConstraintStatus('forceBox', True)

    for col_idx in range(len(robot.collision_model.collisionPairs)):
        # only populates the bounds of the constraint item (not the manager)
        m.differential.constraints.constraints['collisionBox_' + str(col_idx)].constraint.updateBounds(
                    np.array([0.]),
                    np.array([np.inf])) 
        # needed to pass the bounds to the manager
        m.differential.constraints.changeConstraintStatus('collisionBox_' + str(col_idx), True)
# ERPGOEJR
# solver.setCallbacks([mim_solvers.CallbackVerbose(), mim_solvers.CallbackLogger()])
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
# # Plot initial solution
# # if(PLOT_INIT):
# ocp_data_handler = OCPDataHandlerClassical(solver.problem)
# ocp_data = ocp_data_handler.extract_data(solver.xs, solver.us)
# _, _ = ocp_data_handler.plot_ocp_results(ocp_data, which_plots=config['WHICH_PLOTS'], markers=['.'], colors=['b'], SHOW=True)
# # # # # # # # # # #
### INIT MPC SIMU ###
# # # # # # # # # # #
sim_data = MPCDataHandlerClassical(config, robot)
sim_data.init_sim_data(x0)
sim_data.record_simu_cycle_measured(0, x0, x0, us_init[0], np.zeros(6))
  # Replan & control counters
nb_plan = 0
nb_ctrl = 0
# Additional simulation blocks 
communicationModel = mpc_utils.CommunicationModel(config)
actuationModel     = mpc_utils.ActuationModel(config, nu=nu, SEED=RANDOM_SEED)
sensingModel       = mpc_utils.SensorModel(config, SEED=RANDOM_SEED)
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
# x_filtered = x0
count=0
f_err = []
p_err = []
err_f_i = 0.
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
        # x_filtered = antiAliasingFilter.iir(x_filtered, sim_data.state_mea_SIMU[i], fc=10, fs=sim_data.plan_freq)
        # Reset x0 to measured state + warm-start solution
        q = x_filtered[:nq]
        v = x_filtered[nq:nq+nv]
        # Solve OCP 
        solveOCP(q, v, solver, config['maxiter'], target_position, TASK_PHASE, target_force)
        # Record MPC predictions, cost references and solver data 
        sim_data.record_predictions(nb_plan, solver)
        sim_data.record_cost_references(nb_plan, solver)
        sim_data.record_solver_data(nb_plan, solver) 
        # Model communication delay between computer & robot (buffered OCP solution)
        communicationModel.step(sim_data.x_pred, sim_data.u_curr)
        # Record interpolated desired state, control and force at MPC frequency
        sim_data.record_plan_cycle_desired(nb_plan)
        # Increment planning counter
        nb_plan += 1
      #   torqueController.reset_integral_error()
        

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
          x_filtered = antiAliasingFilter.step(nb_ctrl, i, sim_data.ctrl_freq, sim_data.simu_freq, sim_data.state_mea_SIMU)
          # x_filtered = antiAliasingFilter.iir(x_filtered, sim_data.state_mea_SIMU[i], 0.5)
          tau_des_CTRL += solver.K[0].dot(solver.problem.x0 - x_filtered)
        
        if(config['USE_LATERAL_FRICTION'] and 0 <= time_to_contact):
            Jac = pin.computeFrameJacobian(robot.model, robot.data, q, id_endeff, pin.LOCAL_WORLD_ALIGNED)[:3, :nq]
            force_est =  robot_simulator.end_effector_forces()[1][0]
            tau_des_CTRL -= Jac.T @ np.array([force_est[0], force_est[1], 0])
        if(config['USE_FORCE_PI'] and 0 <= time_to_contact):
            Jac = pin.computeFrameJacobian(robot.model, robot.data, q, id_endeff, pin.LOCAL_WORLD_ALIGNED)[:3, :nq]
            force_est =  robot_simulator.end_effector_forces()[1][0]
            err_f_p = target_force[0] - force_est[2]
            err_f_i += err_f_p
            F_PI = config['F_Kp'] * err_f_p + config['F_Ki'] * err_f_i
            tau_des_CTRL -= Jac.T @ np.array([0, 0, F_PI])

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
    if(i%1000==0): 
      logger.info("f_mea  = "+str(f_mea_SIMU))
    # Compute force and position errors
    if(i >= T_CIRCLE):
      count+=1
      f_err.append(np.abs(f_mea_SIMU[2] - target_force[0]))
      p_err.append(np.abs(robot_simulator.pin_robot.data.oMf[id_endeff].translation[:2] - target_position[0][:2]))
    # Record data (unnoised)
    x_mea_SIMU = np.concatenate([q_mea_SIMU, v_mea_SIMU]).T 
    # Simulate sensing 
    x_mea_no_noise_SIMU = sensingModel.step(x_mea_SIMU)
    # Record measurements of state, torque and forces 
    sim_data.record_simu_cycle_measured(i, x_mea_SIMU, x_mea_no_noise_SIMU, tau_mea_SIMU, f_mea_SIMU)
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



# if __name__=='__main__':
#     # args = misc_utils.parse_MPC_script(sys.argv[1:])
#     # main(args.SAVE_DIR)
#     main(sys.argv[1], sys.argv[2])
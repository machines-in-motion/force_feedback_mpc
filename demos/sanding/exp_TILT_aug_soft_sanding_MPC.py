"""
@package force_feedback
@file exp_TILT_aug_soft_sanding_MPC.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2021, New York University & LAAS-CNRS
@date 2023-04-03
@brief Closed-loop 'Augmented Soft Contact Force feedback' MPC for sanding task
"""

'''
The robot is tasked with exerting a constant normal force with its EE 
while drawing a circle on the contact surface
Trajectory optimization using Crocoddyl in closed-loop MPC 
(feedback from stateSoft y=(q,v,f), control tau = joint torque
Using PyBullet simulator & GUI for rigid-body dynamics + visualization

The goal of this script is to simulate MPC with force feedback where
the contact force is modeled as a spring damper
  - Simulator = custom actuation model + PyBullet RBD
'''

import sys
sys.path.append('.')

from croco_mpc_utils.utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger


import numpy as np  
np.set_printoptions(precision=4, linewidth=180)

from core_mpc_utils import path_utils, misc_utils, mpc_utils
from core_mpc_utils import sim_utils as simulator_utils

from croco_mpc_utils import pinocchio_utils as pin_utils
from croco_mpc_utils.ocp_core_data import save_data
from croco_mpc_utils.math_utils import circle_point_WORLD
from soft_mpc.aug_ocp import OptimalControlProblemSoftContactAugmented
from soft_mpc.aug_data import MPCDataHandlerSoftContactAugmented
from soft_mpc.utils import SoftContactModel3D, SoftContactModel1D

import mim_solvers
from mim_robots.robot_loader import load_bullet_wrapper
from mim_robots.pybullet.env import BulletEnvWithGround
import pybullet as p


RESET_ANCHOR_POINT = True

# tilt table of several angles around y-axis
TILT_ANGLES_DEG = [6, 4, 2, 0, -2, -4, -6] 
TILT_RPY = []
for angle in TILT_ANGLES_DEG:
    TILT_RPY.append([angle*np.pi/180, 0., 0.])
N_EXP = len(TILT_RPY)
SEEDS = [19, 71, 89, 83, 41, 73, 17, 47, 29, 7]
N_SEEDS = len(SEEDS)


from aug_soft_sanding_MPC import solveOCP
from croco_mpc_utils.utils import load_yaml_file
import pinocchio as pin

# SAVE_DIR = '/home/skleff/Desktop/soft_contact_sim_exp/with_torque_control'

def main(SAVE_DIR, TORQUE_TRACKING):

    # # # # # # # # # # # # # # # # # # #
    ### LOAD ROBOT MODEL and SIMU ENV ### 
    # # # # # # # # # # # # # # # # # # # 
    # Read config file
    # config, config_name = path_utils.load_config_file('aug_soft_sanding_MPC', robot_name)
    config_name = 'iiwa_aug_soft_sanding_MPC'
    config = load_yaml_file('/home/sebastien/workspace_seb/src/force-feedback/demos/sanding/config/iiwa_aug_soft_sanding_MPC.yml')

    logger.warning("save dir = "+SAVE_DIR)
    logger.warning("tracking = "+str(TORQUE_TRACKING))
    
    # Create a simulation environment & simu-pin wrapper 
    dt_simu = 1./float(config['simu_freq'])  
    q0 = np.asarray(config['q0'])
    v0 = np.asarray(config['dq0'])
    x0 = np.concatenate([q0, v0])  
    env             = BulletEnvWithGround(dt=dt_simu, server=p.DIRECT)
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


    for n_seed in range(N_SEEDS):
        
        print("\n ============================================================================= ")
        print(" ============================================================================= ")
        print(" Set Random Seed to "+str(SEEDS[n_seed]) + " ("+str(n_seed+1)+"/"+str(N_SEEDS)+")")
        print(" ============================================================================= ")
        print(" ============================================================================= \n")
        np.random.seed(SEEDS[n_seed])

        for n_exp in range(N_EXP):

            print("\n       ============================================================================= ")
            print("       Set TILT ANGLE to "+str(TILT_ANGLES_DEG[n_exp]) + " ("+str(n_exp+1)+"/"+str(N_EXP)+")")
            print("       ============================================================================= \n")
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


            # Reset robot to initial state and set table
            robot_simulator.reset_state(q0, v0)
            robot_simulator.forward_robot(q0, v0)
            # Display contact surface + optional tilt
            contact_placement        = pin.SE3(np.eye(3), np.asarray(config['contactPosition']))
            contact_placement_0      = contact_placement.copy()
            contact_placement        = pin_utils.rotate(contact_placement, rpy=TILT_RPY[n_exp])
            contact_surface_bulletId = simulator_utils.display_contact_surface(contact_placement, bullet_endeff_ids=robot_simulator.bullet_endeff_ids)
            # Make the contact soft (e.g. tennis ball or sponge on the robot)
            simulator_utils.set_lateral_friction(contact_surface_bulletId, 0.5)
            simulator_utils.set_contact_stiffness_and_damping(contact_surface_bulletId, 10000, 500)
            
            # Display target circle  trajectory (reference)
            if(config['DISPLAY_EE']):
                nb_points = 20 
                ballsIdTarget = np.zeros(nb_points, dtype=int)
                ballsIdReal = []
                for i in range(nb_points):
                    t = (i/nb_points)*2*np.pi/OMEGA
                    pos = circle_point_WORLD(t, contact_placement_0, radius=RADIUS, omega=OMEGA, LOCAL_PLANE=config['CIRCLE_LOCAL_PLANE'])
                    ballsIdTarget[i] = simulator_utils.display_ball(pos, RADIUS=0.01, COLOR=[1., 0., 0., 1.])
                draw_rate = 1000


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
            actuationModel     = mpc_utils.ActuationModel(config, nu=nu, SEED=SEEDS[n_seed])
            sensingModel       = mpc_utils.SensorModel(config, naug=softContactModel.nc, SEED=SEEDS[n_seed])
            if(int(TORQUE_TRACKING) == 0):
                use = False
            else:
                use = True
            torqueController   = mpc_utils.LowLevelTorqueController(config, nu=nu, use=use)
            antiAliasingFilter = mpc_utils.AntiAliasingFilter()

            
            

            # # # # # # # # # # # #
            ### SIMULATION LOOP ###
            # # # # # # # # # # # #
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
                    ballId = simulator_utils.display_ball(pos, RADIUS=0.03, COLOR=[0.,0.,1.,0.3])
                    ballsIdReal.append(ballId)
            
            # # Remove table
            simulator_utils.remove_body_from_sim(contact_surface_bulletId)
            if(config['DISPLAY_EE']):
                for ballId in ballsIdTarget:
                    simulator_utils.remove_body_from_sim(ballId)
                for ballId in ballsIdReal:
                    simulator_utils.remove_body_from_sim(ballId)
                
            # # # # # # # # # # #
            # PLOT SIM RESULTS  #
            # # # # # # # # # # #
            logger.warning("count = "+str(count))
            logger.warning("------------------------------------")
            logger.warning("------------------------------------")
            logger.warning(" Fz MAE  = "+str(np.mean(f_err)))
            logger.warning(" Pxy MAE = "+str(np.mean(p_err)))
            logger.warning("------------------------------------")
            logger.warning("------------------------------------")
            save_dir = SAVE_DIR 
            save_name = config_name+'_bullet_'+\
                                    '_BIAS='+str(config['SCALE_TORQUES'])+\
                                    '_NOISE='+str(config['NOISE_STATE'] or config['NOISE_TORQUES'])+\
                                    '_DELAY='+str(config['DELAY_OCP'] or config['DELAY_SIM'])+\
                                    '_Fp='+str(sim_data.plan_freq/1000)+'_Fc='+str(sim_data.ctrl_freq/1000)+'_Fs'+str(sim_data.simu_freq/1000)+\
                                    '_EXP_TILT='+str(TILT_ANGLES_DEG[n_exp])+\
                                    '_SEED='+str(SEEDS[n_seed])
            # Save optionally   
            if(config['SAVE_DATA']):
                plot_data = sim_data.extract_data(frame_of_interest=frame_of_interest)
                save_data(plot_data, save_dir=save_dir, save_name=save_name)
                # sim_data.save_data(sim_data, save_name=save_name, save_dir=save_dir)




if __name__=='__main__':
    # args = misc_utils.parse_MPC_script(sys.argv[1:])
    # main(args.SAVE_DIR)
    main(sys.argv[1], sys.argv[2])
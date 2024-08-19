"""
@package force_feedback
@file LPF_circle_MPC.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and LAAS-CNRS
@date 2020-05-18
@brief Closed-loop MPC for tracking a circle trajectory (LPF)
"""

'''
The robot is tasked with tracking a circle trajectory 
Trajectory optimization using Crocoddyl in closed-loop MPC 
(feedback from stateLPF y=(q,v,tau), control u = w 
Using PyBullet simulator & GUI for rigid-body dynamics + visualization

The goal of this script is to simulate closed-loop MPC on a simple reaching task 
'''

import sys
sys.path.append('.')

from core_mpc_utils.misc_utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger


import numpy as np  
np.set_printoptions(precision=4, linewidth=180)

from core_mpc_utils import ocp, path_utils, pin_utils, mpc_utils, misc_utils


from lpf_mpc.data import DDPDataHandlerLPF, MPCDataHandlerLPF
from lpf_mpc.ocp import OptimalControlProblemLPF

WARM_START_IK = True


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
    # Initial placement
    id_endeff = robot.model.getFrameId(config['frameTranslationFrameName'])
    nq, nv = robot.model.nq, robot.model.nv; ny = nq+nv+nq; nu = nq
    M_ee = robot.data.oMf[id_endeff].copy()



    # # # # # # # # # 
    ### OCP SETUP ###
    # # # # # # # # # 
    N_h = config['N_h']
    dt = config['dt']
    # Setup Croco OCP and create solver
    ug = pin_utils.get_u_grav(q0, robot.model, config['armature']) 
    y0 = np.concatenate([x0, ug])
    ddp = OptimalControlProblemLPF(robot, config).initialize(y0, callbacks=False) 
    # Setup tracking problem with circle ref EE trajectory
    models = list(ddp.problem.runningModels) + [ddp.problem.terminalModel]
    RADIUS = config['frameCircleTrajectoryRadius'] 
    OMEGA  = config['frameCircleTrajectoryVelocity']
    for k,m in enumerate(models):
        # Ref
        t = min(k*config['dt'], 2*np.pi/OMEGA)
        p_ee_ref = ocp.circle_point_WORLD(t, M_ee.copy(), 
                                                    radius=RADIUS,
                                                    omega=OMEGA,
                                                    LOCAL_PLANE=config['CIRCLE_LOCAL_PLANE'])
        # Cost translation
        m.differential.costs.costs['translation'].cost.residual.reference = p_ee_ref
    
    # Warm start state = IK of circle trajectory
    if(WARM_START_IK):
        logger.info("Computing warm-start using Inverse Kinematics...")
        xs_init = [] 
        us_init = []
        q_ws = q0
        for k,m in enumerate(list(ddp.problem.runningModels) + [ddp.problem.terminalModel]):
            # Get ref placement
            p_ee_ref = m.differential.costs.costs['translation'].cost.residual.reference
            Mref = M_ee.copy()
            Mref.translation = p_ee_ref        
            q_ws, v_ws, eps = pin_utils.IK_placement(robot, q_ws, id_endeff, Mref, DT=1e-2, IT_MAX=100)
            tau_ws = pin_utils.get_u_grav(q_ws, robot.model, config['armature'])
            xs_init.append(np.concatenate([q_ws, v_ws, tau_ws]))
            if(k<N_h):
                us_init.append(tau_ws)
    
    # Classical warm start using initial config
    else:
        xs_init = [y0 for i in range(config['N_h']+1)]
        us_init = [ug for i in range(config['N_h'])]
    
    # Solve 
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
    sim_data = MPCDataHandlerLPF(config, robot)
    sim_data.init_sim_data(y0)
        # Replan & control counters
    nb_plan = 0
    nb_ctrl = 0
    # Additional simulation blocks 
    communicationModel = mpc_utils.CommunicationModel(config)
    actuationModel     = mpc_utils.ActuationModel(config, nu)
    sensingModel       = mpc_utils.SensorModel(config, ntau=nu)


    # Display target circle
    nb_points = 20 
    for i in range(nb_points):
        t = (i/nb_points)*2*np.pi/OMEGA
        # if(i%20==0):
        pos = ocp.circle_point_WORLD(t, M_ee, radius=RADIUS, omega=OMEGA, LOCAL_PLANE=config['CIRCLE_LOCAL_PLANE'])
        simulator_utils.display_ball(pos, RADIUS=0.02)

    draw_rate = 200

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
            # Current simulation time
            t_simu = i*dt_simu 
            # Setup tracking problem with circle ref EE trajectory
            models = list(ddp.problem.runningModels) + [ddp.problem.terminalModel]
            for k,m in enumerate(models):
                # Ref
                t = min(t_simu + k*sim_data.dt, 2*np.pi/OMEGA)
                p_ee_ref = ocp.circle_point_WORLD(t, M_ee, 
                                                    radius=RADIUS,
                                                    omega=OMEGA,
                                                    LOCAL_PLANE=config['CIRCLE_LOCAL_PLANE'])
                # Cost translation
                m.differential.costs.costs['translation'].cost.residual.reference = p_ee_ref
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
        # Send output of actuation torque to the RBD simulator 
        robot_simulator.send_joint_command(tau_mea_SIMU)
        env.step()
        # Measure new state from simulation :
        q_mea_SIMU, v_mea_SIMU = robot_simulator.get_state()
        # Update pinocchio model
        robot_simulator.forward_robot(q_mea_SIMU, v_mea_SIMU)
        # Record data (unnoised)
        y_mea_SIMU = np.concatenate([q_mea_SIMU, v_mea_SIMU, tau_mea_SIMU]).T 
        sim_data.state_mea_no_noise_SIMU[i+1, :] = y_mea_SIMU
        # Sensor model (optional noise + filtering)
        sim_data.state_mea_SIMU[i+1, :] = sensingModel.step(i, y_mea_SIMU, sim_data.state_mea_SIMU)


        # Display real 
        if(i%draw_rate==0):
            pos = robot_simulator.pin_robot.data.oMf[id_endeff].translation.copy()
            simulator_utils.display_ball(pos, RADIUS=0.03, COLOR=[0.,0.,1.,0.3])


    print('--------------------------------')
    print('Simulation exited successfully !')
    print('--------------------------------')




    # # # # # # # # # # #
    # PLOT SIM RESULTS  #
    # # # # # # # # # # #
    save_dir = '/tmp'
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
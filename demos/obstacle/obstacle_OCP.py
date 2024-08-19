"""
@package force_feedback
@file obstacle_OCP.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft.
@date 2020-05-18
@brief OCP for static EE position task 
"""

'''
The robot is tasked with reaching a static EE target while avoiding an obstacle
Trajectory optimization using Crocoddyl (state x=(q,v))
The goal of this script is to setup the OCP (a.k.a. play with weights)
'''

from re import L
import sys
sys.path.append('.')

from core_mpc_utils.misc_utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger


import numpy as np  
np.set_printoptions(precision=4, linewidth=180)

from core_mpc_utils import path_utils, pin_utils, misc_utils

from classical_mpc.ocp import OptimalControlProblemClassical
from classical_mpc.data import DDPDataHandlerClassical

def main(robot_name, PLOT, DISPLAY):

    # # # # # # # # # # # #
    ### LOAD ROBOT MODEL ## 
    # # # # # # # # # # # # 
    # Read config file and initial state
    config, _ = path_utils.load_config_file(__file__, robot_name)
    q0 = np.asarray(config['q0'])
    v0 = np.asarray(config['dq0'])
    x0 = np.concatenate([q0, v0])   
    # Make pin wrapper
    robot = pin_utils.load_robot_wrapper(robot_name)
    # Get initial frame placement + dimensions of joint space
    frame_name = config['frameTranslationFrameName']
    id_endeff = robot.model.getFrameId(frame_name)
    nq = robot.model.nq
    # Update robot model with initial state
    robot.framesForwardKinematics(q0)
    robot.computeJointJacobians(q0)
    M_ee = robot.data.oMf[id_endeff]



    # # # # # # # # # 
    ### OCP SETUP ###
    # # # # # # # # # 
    # Setup Croco OCP and create solver
    ddp = OptimalControlProblemClassical(robot, config).initialize(x0, callbacks=True)
    # Warmstart and solve
    ug = pin_utils.get_u_grav(q0, robot.model, config['armature'])
    xs_init = [x0 for i in range(config['N_h']+1)]
    us_init = [ug  for i in range(config['N_h'])]
    ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=False)


    #  Plot
    if(PLOT):
        ddp_handler = DDPDataHandlerClassical(ddp)
        ddp_data = ddp_handler.extract_data(ee_frame_name=frame_name, ct_frame_name=frame_name)
        _, _ = ddp_handler.plot_ddp_results(ddp_data, which_plots=config['WHICH_PLOTS'], markers=['.'], colors=['b'], SHOW=True)


    # from pinocchio.visualize import MeshcatVisualizer
    # viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
    # viz.initViewer(loadModel=True)
    # viz.loadViewerModel()
    # viz.display()
    # Visualize motion in Gepetto-viewer
    if(DISPLAY):
        import time
        import pinocchio as pin
        N_h = config['N_h']
        # Init viewer
        from core_mpc_utils import gepetto_utils
        viz = gepetto_utils.launch_viewer(robot, q0)
        # viz = pin.visualize.GepettoVisualizer(robot.model, robot.collision_model, robot.visual_model)
        # viz.initViewer()
        # viz.loadViewerModel()
        # viz.display(q0)
        # viewer = viz.viewer; 
        gui = viz.viewer.gui


        draw_rate = int(N_h/50)
        log_rate  = int(N_h/10)    
        tar_color  = [0., 1., 0., 1.]
        real_color = [0., 0., 1., 0.3]
        obs_color = [1., 0., 1., 1.]
        caps_color = [1., 1., .5, 0.2]
        tar_size    = 0.03
        real_size   = 0.02
        pause = 0.05
        # cleanup
        gepetto_utils.clear_viewer(gui)
        # Display target , obstacle and capsule around robot 
        gepetto_utils.display_sphere(gui, 'world/EE_ref', config['frameTranslationRef'], tar_size, tar_color)
        m_obs = pin.SE3(np.eye(3), config['collisionObstaclePosition'])
        gepetto_utils.display_capsule(gui, 'world/obs', m_obs, config['obstacleRadius'], config['obstacleLength'], obs_color)
        # gepetto_utils.display_sphere(gui, 'world/caps', config['frameTranslationRef'], tar_size, tar_color)

        for i in range(N_h):
            gepetto_utils.display_sphere(gui, 'world/EE_sol_'+str(i), config['frameTranslationRef'], real_size, real_color)
            # gui.addLandmark('world/EE_sol_', 0.25)
            # clean DDP sol
            # if(gui.nodeExists('world/EE_sol_'+str(i))):
            #     gui.deleteNode('world/EE_sol_'+str(i), True)
            #     gui.deleteLandmark('world/EE_sol_'+str(i))

        # gepetto_utils.display_sphere(gui, 'world/obs', np.array([0.4, 0.4, 0.8]), obs_size, obs_color)
            # clean ref
        # if(gui.nodeExists('world/EE_ref')):
        #     gui.deleteNode('world/EE_ref', True)
            # clean obs
        if(gui.nodeExists('world/obs')):
            gui.deleteNode('world/obs', True)
            # clean caps
        if(gui.nodeExists('world/caps')):
            gui.deleteNode('world/caps', True)
        for i in range(N_h):
            # clean DDP sol
            if(gui.nodeExists('world/EE_sol_'+str(i))):
                gui.deleteNode('world/EE_sol_'+str(i), True)
                gui.deleteLandmark('world/EE_sol_'+str(i))
        # Get initial EE placement + tf
        ee_frame_placement = M_ee.copy()
        tf_ee = list(pin.SE3ToXYZQUAT(ee_frame_placement))
        # Get ref EE placement + tf
        # m_ee_ref = ee_frame_placement.copy()
        # m_ee_ref.translation = np.asarray(config['frameTranslationRef'])
        # tf_ee_ref = list(pin.SE3ToXYZQUAT(m_ee_ref))
        m_obs = pin.SE3.Identity()
        m_obs.translation = np.asarray(config['collisionObstaclePosition'])
        tf_obs = list(pin.se3ToXYZQUAT(m_obs))
        m_caps = robot.data.oMf[robot.model.getFrameId(config['collisionFrameName'])]
        tf_caps = list(pin.se3ToXYZQUAT(m_caps))
        # Display ref
        # gui.addSphere('world/EE_ref', tar_size, tar_color)
        # gui.applyConfiguration('world/EE_ref', tf_ee_ref)
        # Display obs
        gui.addCapsule('world/obs', config['collisionObstacleSize'], 0.5, obs_color)
        gui.applyConfiguration('world/obs', tf_obs)
        # Display sol init + landmark
        gui.addSphere('world/EE_sol_', real_size, real_color)
        gui.addLandmark('world/EE_sol_', 0.25)
        gui.applyConfiguration('world/EE_sol_', tf_ee)
        # Display capsule around link of interest
        gui.addCapsule('world/caps', config['collisionCapsuleRadius'], config['collisionCapsuleLength'], caps_color)
        gui.applyConfiguration('world/caps', tf_caps)
        # Refresh and wait
        gui.refresh()
        logger.info("Visualizing...")
        time.sleep(1.)
        # Animate
        for i in range(N_h):
            # Display robot in config q
            q = ddp.xs[i][:nq]
            viz.display(q)
            # Display EE traj and ref circle traj
            if(i%draw_rate==0):
                # EE trajectory
                robot.framesForwardKinematics(q)
                # Target 
                m_ee = robot.data.oMf[id_endeff].copy()
                tf_ee = list(pin.SE3ToXYZQUAT(m_ee))
                gui.addSphere('world/EE_sol_'+str(i), real_size, real_color)
                gui.applyConfiguration('world/EE_sol_'+str(i), tf_ee)
                # Capsule
                m_caps = robot.data.oMf[robot.model.getFrameId(config['collisionFrameName'])].copy()
                tf_caps = list(pin.se3ToXYZQUAT(m_caps))
                if(gui.nodeExists('world/caps')):
                    gui.deleteNode('world/caps', True)
                gui.addCapsule('world/caps', config['collisionCapsuleRadius'], config['collisionCapsuleLength'], caps_color)
                gui.applyConfiguration('world/caps', tf_caps)
                # gui.applyConfiguration('world/EE_sol_', tf_ee)
            gui.refresh()
            if(i%log_rate==0):
                logger.info("Display config n°"+str(i))
            time.sleep(pause)



if __name__=='__main__':
    args = misc_utils.parse_OCP_script(sys.argv[1:])
    main(args.robot_name, args.PLOT, args.DISPLAY)
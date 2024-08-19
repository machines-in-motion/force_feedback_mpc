"""
@package force_feedback
@file rotation_OCP.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft.
@date 2020-05-18
@brief OCP for tracking EE rotation
"""

'''
The robot is tasked with rotation its EE about normal axis
Trajectory optimization using Crocoddyl
The goal of this script is to setup OCP (a.k.a. play with weights)
'''

import sys
sys.path.append('.')

from core_mpc_utils.misc_utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger


import numpy as np  
np.set_printoptions(precision=4, linewidth=180)

from core_mpc_utils import path_utils, pin_utils, misc_utils, ocp

from classical_mpc.ocp import OptimalControlProblemClassical
from classical_mpc.data import DDPDataHandlerClassical


def main(robot_name, PLOT, DISPLAY):


    # # # # # # # # # # # #
    ### LOAD ROBOT MODEL ## 
    # # # # # # # # # # # # 
    # Read config file
    config, _ = path_utils.load_config_file(__file__, robot_name)
    q0 = np.asarray(config['q0'])
    v0 = np.asarray(config['dq0'])
    x0 = np.concatenate([q0, v0])   
    # Get pin wrapper
    robot = pin_utils.load_robot_wrapper(robot_name)
    # Get initial frame placement + dimensions of joint space
    frame_name = config['frame_of_interest']
    id_endeff = robot.model.getFrameId(frame_name)
    nq, nv = robot.model.nq, robot.model.nv
    nx = nq+nv; nu = nq
    # Update robot model with initial state
    robot.framesForwardKinematics(q0)
    robot.computeJointJacobians(q0)
    M_ee = robot.data.oMf[id_endeff]


    # # # # # # # # # 
    ### OCP SETUP ###
    # # # # # # # # # 
    N_h = config['N_h']
    dt = config['dt']
    # Setup Croco OCP and create solver
    ddp = OptimalControlProblemClassical(robot, config).initialize(x0, callbacks=True)
    # Setup tracking problem with oritantation ref for EE trajectory
    models = list(ddp.problem.runningModels) + [ddp.problem.terminalModel]
    OMEGA  = config['frameRotationTrajectoryVelocity']
    for k,m in enumerate(models):
        # Ref
        t = min(k*config['dt'], 2*np.pi/OMEGA)
        # Desired RPY in WORLD frame
        R_ee_ref_WORLD = ocp.rotation_orientation_WORLD(t, M_ee.copy(), 
                                                                 omega=OMEGA, 
                                                                 LOCAL_AXIS=config['ROTATION_LOCAL_AXIS'])
        # Cost translation
        m.differential.costs.costs['rotation'].cost.residual.reference = R_ee_ref_WORLD


    # Warm start state = IK of circle trajectory
    WARM_START_IK = False
    if(WARM_START_IK):
        logger.info("Computing warm-start using Inverse Kinematics...")
        xs_init = [] 
        us_init = []
        q_ws = q0
        for k,m in enumerate(models):
            Mref = M_ee.copy()
            Mref.rotation = m.differential.costs.costs['rotation'].cost.residual.reference
            q_ws, v_ws, eps = pin_utils.IK_placement(robot, q_ws, id_endeff, Mref, DT=1e-2, IT_MAX=100)
            xs_init.append(np.concatenate([q_ws, v_ws]))
        us_init = [pin_utils.get_u_grav(xs_init[i][:nq], robot.model, config['armature']) for i in range(N_h)]

    # Classical warm start using initial config
    else:
        ug  = pin_utils.get_u_grav(q0, robot.model, config['armature'])
        xs_init = [x0 for i in range(config['N_h']+1)]
        us_init = [ug for i in range(config['N_h'])]

    # Solve 
    ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=False)

    #  Plot
    if(PLOT):
        ddp_handler = DDPDataHandlerClassical(ddp)
        ddp_data = ddp_handler.extract_data(ee_frame_name=frame_name, ct_frame_name=frame_name)
        _, _ = ddp_handler.plot_ddp_results(ddp_data, which_plots=config['WHICH_PLOTS'], markers=['.'], colors=['b'], SHOW=True)




    pause = 0.02 # in s
    if(DISPLAY):
        import time
        import pinocchio as pin
        models = list(ddp.problem.runningModels) + [ddp.problem.terminalModel]
        # Init viewer
        robot.initViewer(loadModel=True)
        robot.display(q0)
        viewer = robot.viz.viewer; gui = viewer.gui

        draw_rate = int(N_h/50)
        log_rate  = int(N_h/10)
        
        ref_color  = [1., 0., 0., 1.]
        real_color = [0., 0., 1., 0.3]
        
        ref_size    = 0.01
        real_size   = 0.02
        wrench_coef = 0.02

        # Display reference trajectory as red spheres
        if('translation' in config['WHICH_COSTS'] or 'placement' in config['WHICH_COSTS'] or 'rotation' in config['WHICH_COSTS']):

            # Remove circle ref traj and EE traj if already displayed
            for i in range(N_h):
                if(viewer.gui.nodeExists('world/EE_ref'+str(i))):
                    viewer.gui.deleteNode('world/EE_ref'+str(i), True)
            
            viewer.gui.refresh()

        # Display EE trajectory as blue spheres
        for i in range(N_h):      
            if(viewer.gui.nodeExists('world/EE_'+str(i))):
                viewer.gui.deleteNode('world/EE_'+str(i), True)

        viewer.gui.refresh()
        
        logger.info("Visualizing...")

        time.sleep(1.)

        for i in range(N_h):
            # Display robot in config q
            q = ddp.xs[i][:nq]
            robot.display(q)

            # Display EE traj and ref circle traj
            if(i%draw_rate==0):
                if('translation' or 'placement' in config['WHICH_COSTS']):
                    # EE ref circle trajectory
                    m_ee_ref = M_ee.copy()
                    if('translation' in config['WHICH_COSTS']):
                        m_ee_ref.translation = models[i].differential.costs.costs['translation'].cost.residual.reference
                    elif('placement' in config['WHICH_COSTS']):
                        m_ee_ref = models[i].differential.costs.costs['placement'].cost.residual.reference.copy()
                    elif('rotation' in config['WHICH_COSTS']):
                        m_ee_ref.rotation = models[i].differential.costs.costs['rotation'].cost.residual.reference
                    tf_ee_ref = list(pin.SE3ToXYZQUAT(m_ee_ref))
                    viewer.gui.addSphere('world/EE_ref'+str(i), ref_size, ref_color)
                    viewer.gui.applyConfiguration('world/EE_ref'+str(i), tf_ee_ref)
                # EE trajectory
                robot.framesForwardKinematics(q)
                m_ee = robot.data.oMf[id_endeff].copy()
                tf_ee = list(pin.SE3ToXYZQUAT(m_ee))
                viewer.gui.addSphere('world/EE_'+str(i), real_size, real_color)
                viewer.gui.applyConfiguration('world/EE_'+str(i), tf_ee)
            

            viewer.gui.refresh()

            if(i%log_rate==0):
                logger.info("Display config n°"+str(i))

            time.sleep(pause)



if __name__=='__main__':
    args = misc_utils.parse_OCP_script(sys.argv[1:])
    main(args.robot_name, args.PLOT, args.DISPLAY)
"""
@package force_feedback
@file LPF_circle_OCP.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft.
@date 2020-05-18
@brief OCP for circle trajectory tracking task (with Low-Pass-Filter)
"""

'''
The robot is tasked with tracking a circle trajectory 
Trajectory optimization using Crocoddyl (stateLPF x=(q,v,tau))
The goal of this script is to setup OCP (a.k.a. play with weights)
'''

import sys
sys.path.append('.')

from core_mpc_utils.misc_utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger


import numpy as np  
np.set_printoptions(precision=4, linewidth=180)

from core_mpc_utils import path_utils, pin_utils, misc_utils, ocp

from lpf_mpc.ocp import OptimalControlProblemLPF
from lpf_mpc.data import DDPDataHandlerLPF

WARM_START_IK = True


def main(robot_name='iiwa', PLOT=True, DISPLAY=True):


    # # # # # # # # # # # #
    ### LOAD ROBOT MODEL ## 
    # # # # # # # # # # # # 
    # Read config file
    config, _ = path_utils.load_config_file(__file__, robot_name)
    q0 = np.asarray(config['q0'])
    v0 = np.asarray(config['dq0'])
    x0 = np.concatenate([q0, v0])   
    # Get pin wrapper + set model to init state
    robot = pin_utils.load_robot_wrapper(robot_name)
    robot.framesForwardKinematics(q0)
    robot.computeJointJacobians(q0)
    # Get initial frame placement + dimensions of joint space
    frame_name = config['frameTranslationFrameName']
    id_endeff = robot.model.getFrameId(frame_name)
    M_ee = robot.data.oMf[id_endeff]
    nq = robot.model.nq; nv = robot.model.nv; nx = nq+nv; nu = nq


    #################
    ### OCP SETUP ###
    #################
    N_h = config['N_h']
    dt = config['dt']
    # Setup Croco OCP and create solver
    ug = pin_utils.get_u_grav(q0, robot.model, config['armature']) 
    y0 = np.concatenate([x0, ug])
    ddp = OptimalControlProblemLPF(robot, config).initialize(y0, callbacks=True)
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
    ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=False)

    if(PLOT):
        #  Plot
        ddp_handler = DDPDataHandlerLPF(ddp)
        ddp_data = ddp_handler.extract_data(ee_frame_name=frame_name, ct_frame_name=frame_name)
        _, _ = ddp_handler.plot_ddp_results(ddp_data, which_plots=config['WHICH_PLOTS'], 
                                                            colors=['r'], 
                                                            markers=['.'], 
                                                            SHOW=True)
                                                            
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

        # Display reference trajectory as red spheres
        if('translation' or 'placement' in config['WHICH_COSTS']):

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
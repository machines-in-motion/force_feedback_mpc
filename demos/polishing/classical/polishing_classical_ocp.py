"""
@package force_feedback
@file sanding_OCP.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft.
@date 2020-05-18
@brief OCP for static EE pose task with the KUKA iiwa 
"""

'''
The robot is tasked with exerting a constant normal force at its EE
while drawing a circle on the contact surface
Trajectory optimization using Crocoddyl
The goal of this script is to setup OCP (a.k.a. play with weights)
'''


import sys
sys.path.append('.')

from croco_mpc_utils.utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger


import numpy as np  
np.set_printoptions(precision=4, linewidth=180)

from force_feedback_mpc.core_mpc_utils import path_utils, misc_utils

from croco_mpc_utils import pinocchio_utils as pin_utils
from croco_mpc_utils.ocp import OptimalControlProblemClassical
from croco_mpc_utils.math_utils import circle_point_WORLD
from croco_mpc_utils.ocp_data import OCPDataHandlerClassical

import mim_solvers
from mim_robots.robot_loader import load_pinocchio_wrapper

WARM_START_IK = True

import os

def main(PLOT=False, DISPLAY=True):

    # # # # # # # # # # # #
    ### LOAD ROBOT MODEL ## 
    # # # # # # # # # # # # 
    # Read config file
    config = path_utils.load_yaml_file(os.path.dirname(os.path.realpath(__file__))+'/polishing_classical.yml')
    q0 = np.asarray(config['q0'])
    v0 = np.asarray(config['dq0'])
    x0 = np.concatenate([q0, v0])   
    # Get pin wrapper
    robot = load_pinocchio_wrapper('iiwa', locked_joints=['A7'])
    # Get initial frame placement + dimensions of joint space
    frame_name = config['contacts'][0]['contactModelFrameName']
    id_endeff = robot.model.getFrameId(frame_name)
    nq, nv = robot.model.nq, robot.model.nv
    # Update robot model with initial state
    robot.framesForwardKinematics(q0)
    robot.computeJointJacobians(q0)
    M_ee = robot.data.oMf[id_endeff]



    # # # # # # # # # 
    ### OCP SETUP ###
    # # # # # # # # # 
    N_h = config['N_h']
    # Setup Croco OCP and create solver
    ocp = OptimalControlProblemClassical(robot, config).initialize(x0)
    # Setup tracking problem with circle ref EE trajectory
    models = list(ocp.runningModels) + [ocp.terminalModel]
    RADIUS = config['frameCircleTrajectoryRadius'] 
    OMEGA  = config['frameCircleTrajectoryVelocity']
    for k,m in enumerate(models):
        # Ref
        t = min(k*config['dt'], 2*np.pi/OMEGA)
        p_ee_ref = circle_point_WORLD(t, M_ee, 
                                                radius=RADIUS,
                                                omega=OMEGA,
                                                LOCAL_PLANE=config['CIRCLE_LOCAL_PLANE'])
        # Cost translation
        m.differential.costs.costs['translation'].cost.residual.reference = p_ee_ref

    # Warm start state = IK of circle trajectory
    if(WARM_START_IK):
        logger.info("Computing warm-start using Inverse Kinematics...")
        xs_init = [] 
        us_init = []
        q_ws = q0
        for k,m in enumerate(models):
            # Get ref placement
            p_ee_ref = m.differential.costs.costs['translation'].cost.residual.reference
            Mref = M_ee.copy()
            Mref.translation = p_ee_ref
            # Get corresponding forces at each joint
            f_ext = pin_utils.get_external_joint_torques(Mref, config['frameForceRef'], robot)
            # Get joint state from IK
            q_ws, v_ws, _ = pin_utils.IK_placement(robot, q_ws, id_endeff, Mref, DT=1e-2, IT_MAX=100)
            xs_init.append(np.concatenate([q_ws, v_ws]))
            if(k<N_h):
                us_init.append(pin_utils.get_tau(q_ws, v_ws, np.zeros((nq,1)), f_ext, robot.model))
    # Classical warm start using initial config
    else:
        f_ext = pin_utils.get_external_joint_torques(M_ee, config['frameForceRef'], robot)
        u0 = pin_utils.get_tau(q0, v0, np.zeros((nq,1)), f_ext, robot.model, config['armature'])
        xs_init = [x0 for i in range(config['N_h']+1)]
        us_init = [u0 for i in range(config['N_h'])]

    # Solve initial
    solver = mim_solvers.SolverSQP(ocp)
    solver.setCallbacks([mim_solvers.CallbackVerbose(), mim_solvers.CallbackLogger()])
    solver.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=False) 


    #  Plot
    if(PLOT):
        ocp_data_handler = OCPDataHandlerClassical(solver.problem)
        ocp_data = ocp_data_handler.extract_data(solver.xs, solver.us)
        _, _ = ocp_data_handler.plot_ocp_results(ocp_data, which_plots=config['WHICH_PLOTS'], markers=['.'], colors=['b'], SHOW=True)


    
    force_axis = 'z'
    xyz = {'x': 0, 'y': 1, 'z': 2}

    pause = 0.02 # in s
    if(DISPLAY):
        import time
        import pinocchio as pin
        N_h = config['N_h']
        # Init viewer
        robot.initViewer(loadModel=True)
        robot.display(q0)
        viewer = robot.viz.viewer; gui = viewer.gui

        draw_rate = int(N_h/50)
        log_rate  = int(N_h/10)
        
        ref_color  = [1., 0., 0., 1.]
        real_color = [0., 0., 1., 0.3]
        ct_color   = [0., 1., 0., 1.]
        
        ref_size    = 0.01
        real_size   = 0.02
        ct_size     = 0.02
        wrench_coef = 0.02

        # Display contact point as sphere + landmark
        if(len(config['contacts']) != 0):

            # Placement of contact in WORLD frame = EE placement + tennis ball radius offset
            ct_frame_placement = M_ee.copy()
            offset = 0.036
            offset_vec_LOCAL = np.zeros(3)
            offset_vec_LOCAL[xyz[force_axis]] += offset
            ct_frame_placement.translation = ct_frame_placement.act(offset_vec_LOCAL) 
            tf_contact = list(pin.SE3ToXYZQUAT(ct_frame_placement))

            # Delete contact point node if already displayed
            if(gui.nodeExists('world/contact_point')):
                gui.deleteNode('world/contact_point', True)
                gui.deleteLandmark('world/contact_point')
            # Display contact point node as green sphere
            gui.addSphere('world/contact_point', ct_size, ct_color)
            gui.addLandmark('world/contact_point', .25)
            gui.applyConfiguration('world/contact_point', tf_contact)
            
            viewer.gui.refresh()

            # Display reference contact wrench as red arrow
            if('force' in config['WHICH_COSTS']):
                # Display contact force as arrow
                f_des_LOCAL = np.asarray(config['frameForceRef'])
                ct_frame_placement_aligned = ct_frame_placement.copy()
                    # Because applying tf on arrow makes arrow coincide with x-axis of tf placement
                    # if force is along z axis in local frame , need to transform x-->z , i.e. -90° around y
                if(force_axis == 'z'):
                    rotation_matrix = pin.rpy.rpyToMatrix(0., -np.pi/2, 0.)
                    # if force is along x axis in local frame , no transform
                elif(force_axis == 'x'):
                    rotation_matrix = pin.rpy.rpyToMatrix(0., 0., 0.)
                    # if force is along y axis in local frame , need to transform x-->y , i.e. +90° around z
                elif(force_axis == 'y'):
                    rotation_matrix = pin.rpy.rpyToMatrix(0., 0., np.pi/2)
                ct_frame_placement_aligned.rotation = ct_frame_placement_aligned.rotation.dot(rotation_matrix)
                # ct_frame_placement_aligned.rotation = ct_frame_placement_aligned.rotation.dot(pin.rpy.rpyToMatrix(0., -np.pi/2, 0.))
                tf_contact_aligned = list(pin.SE3ToXYZQUAT(ct_frame_placement_aligned))
                arrow_length = wrench_coef*np.linalg.norm(f_des_LOCAL)
                # Remove force arrow if already displayed
                if(gui.nodeExists('world/ref_wrench')):
                    gui.deleteNode('world/ref_wrench', True)
                # Display force arrow
                gui.addArrow('world/ref_wrench', ref_size, arrow_length, ref_color)
                gui.applyConfiguration('world/ref_wrench', tf_contact_aligned )
                
                viewer.gui.refresh()

            # Display friction cones 
            if('friction' in config['WHICH_COSTS']):
                mu = config['mu']
                frictionConeColor = [1., 1., 0., 0.3]
                m_generatrices = np.matrix(np.empty([3, 4]))
                m_generatrices[:, 0] = np.matrix([mu, mu, 1.]).T
                m_generatrices[:, 0] = m_generatrices[:, 0] / np.linalg.norm(m_generatrices[:, 0])
                m_generatrices[:, 1] = m_generatrices[:, 0]
                m_generatrices[0, 1] *= -1.
                m_generatrices[:, 2] = m_generatrices[:, 0]
                m_generatrices[:2, 2] *= -1.
                m_generatrices[:, 3] = m_generatrices[:, 0]
                m_generatrices[1, 3] *= -1.
                generatrices = m_generatrices
                v = [[0., 0., 0.]]
                for k in range(m_generatrices.shape[1]):
                    v.append(m_generatrices[:3, k].T.tolist()[0])
                v.append(m_generatrices[:3, 0].T.tolist()[0])
                result = robot.viewer.gui.addCurve('world/cone', v, frictionConeColor)
                robot.viewer.gui.setCurveMode('world/cone', 'TRIANGLE_FAN')
                for k in range(m_generatrices.shape[1]):
                    l = robot.viewer.gui.addLine('world/cone_ray/' + str(k), [0., 0., 0.],
                                                        m_generatrices[:3, k].T.tolist()[0], frictionConeColor)
                robot.viewer.gui.setScale('world/cone', [.5, .5, .5])
                robot.viewer.gui.setVisibility('world/cone', "ALWAYS_ON_TOP")

                viewer.gui.refresh()

        # Display reference trajectory as red spheres
        if('translation' in config['WHICH_COSTS']):

            # Remove circle ref traj and EE traj if already displayed
            for i in range(N_h):
                if(viewer.gui.nodeExists('world/EE_ref'+str(i))):
                    viewer.gui.deleteNode('world/EE_ref'+str(i), True)
            
            viewer.gui.refresh()

        # Display EE trajectory as blue spheres
        for i in range(N_h):      
            if(viewer.gui.nodeExists('world/EE_'+str(i))):
                viewer.gui.deleteNode('world/EE_'+str(i), True)

        viewer.gui.refresh()
        
        logger.info("Visualizing...")

        # Clean force arrow if already displayed
        if(gui.nodeExists('world/force')):
            gui.deleteNode('world/force', True)
        # Display force arrow
        if('force' in config['WHICH_COSTS']):
            gui.addArrow('world/force', real_size, arrow_length, real_color)

        time.sleep(1.)

        for i in range(N_h):
            # Display robot in config q
            q = solver.xs[i][:nq]
            robot.display(q)

            # Display EE traj and ref circle traj
            if(i%draw_rate==0):
                if('translation' in config['WHICH_COSTS']):
                    # EE ref circle trajectory
                    m_ee_ref = M_ee.copy()
                    m_ee_ref.translation = models[i].differential.costs.costs['translation'].cost.residual.reference
                    tf_ee_ref = list(pin.SE3ToXYZQUAT(m_ee_ref))
                    viewer.gui.addSphere('world/EE_ref'+str(i), ref_size, ref_color)
                    viewer.gui.applyConfiguration('world/EE_ref'+str(i), tf_ee_ref)
                # EE trajectory
                robot.framesForwardKinematics(q)
                m_ee = robot.data.oMf[id_endeff].copy()
                tf_ee = list(pin.SE3ToXYZQUAT(m_ee))
                viewer.gui.addSphere('world/EE_'+str(i), real_size, real_color)
                viewer.gui.applyConfiguration('world/EE_'+str(i), tf_ee)
            
            # Move contact point 
            m_ct = m_ee.copy()
            m_ct.translation = m_ct.act(offset_vec_LOCAL) 
            tf_ct = list(pin.SE3ToXYZQUAT(m_ct))
            gui.applyConfiguration('world/contact_point', tf_ct)

            # Display force (magnitude and placement)
            if('force' in config['WHICH_COSTS']):
                # Display wrench
                wrench = solver.problem.runningDatas[i].differential.multibody.contacts.contacts[config['contacts'][0]['contactModelFrameName']].f.vector
                gui.resizeArrow('world/force', real_size, wrench_coef*np.linalg.norm(wrench[xyz[force_axis]]))
                m_ct_aligned = m_ct.copy()
                    # Because applying tf on arrow makes arrow coincide with x-axis of tf placement
                    # but force is along z axis in local frame so need to transform x-->z , i.e. -90° around y
                # m_ct_aligned.rotation = m_ct_aligned.rotation.dot(pin.rpy.rpyToMatrix(0., -np.pi/2, 0.))
                m_ct_aligned.rotation = m_ct_aligned.rotation.dot(rotation_matrix)
                gui.applyConfiguration('world/force', list(pin.SE3ToXYZQUAT(m_ct_aligned)) )
                # Move reference wrench 
                m_ct_ref_aligned = m_ee_ref.copy()
                m_ct_ref_aligned.translation = m_ct_ref_aligned.act(offset_vec_LOCAL) 
                # m_ct_ref_aligned.rotation = m_ct_ref_aligned.rotation.dot(pin.rpy.rpyToMatrix(0., -np.pi/2, 0.))
                m_ct_ref_aligned.rotation = m_ct_ref_aligned.rotation.dot(rotation_matrix)
                tf_ct_ref_aligned = list(pin.SE3ToXYZQUAT(m_ct_ref_aligned))
                gui.applyConfiguration('world/ref_wrench', tf_ct_ref_aligned)

            # Display the friction cones
            if('friction' in config['WHICH_COSTS']):
                position = m_ee
                position.rotation = m_ee.rotation
                robot.viewer.gui.applyConfiguration('world/cone', list(np.array(pin.SE3ToXYZQUAT(position)).squeeze()))
                robot.viewer.gui.setVisibility('world/cone', "ON")

            viewer.gui.refresh()

            if(i%log_rate==0):
                logger.info("Display config n°"+str(i))

            time.sleep(pause)


if __name__=='__main__':
    args = misc_utils.parse_OCP_script(sys.argv[1:])
    main(args.PLOT, args.DISPLAY)
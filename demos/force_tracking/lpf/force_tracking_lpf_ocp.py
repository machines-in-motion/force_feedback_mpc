"""
@package force_feedback
@file LPF_contact_OCP.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft.
@date 2020-05-18
@brief OCP for static EE pose task  
"""

'''
The robot is tasked with applying a constant normal force in contact with a wall
Trajectory optimization using Crocoddyl (feedback from stateLPF x=(q,v,tau))
The goal of this script is to setup OCP (play with weights)
'''

import sys
sys.path.append('.')

from croco_mpc_utils.misc_utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger


import numpy as np  
np.set_printoptions(precision=4, linewidth=180)

from force_feedback_mpc.core_mpc_utils import path_utils, pin_utils, misc_utils

from lpf_mpc.ocp import OptimalControlProblemLPF
from lpf_mpc.data import DDPDataHandlerLPF

import pinocchio as pin

def main(robot_name, PLOT, DISPLAY):


    # # # # # # # # # # # #
    ### LOAD ROBOT MODEL ## 
    # # # # # # # # # # # # 
    # Read config file
    # config, _ = path_utils.load_config_file(__file__, robot_name)
    config = path_utils.load_yaml_file('/home/skleff/ws/workspace/src/force_feedback_dgh/config/reduced_lpf_mpc_contact.yml')
    q0 = np.asarray(config['q0'])
    v0 = np.asarray(config['dq0'])
    x0 = np.concatenate([q0, v0])   
    # Get pin wrapper
    # Get pin wrapper
    from robot_properties_kuka.config import IiwaReducedConfig
    CONTROLLED_JOINTS = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6']
    QREF              = np.zeros(7)
    robot             = IiwaReducedConfig.buildRobotWrapper(controlled_joints=CONTROLLED_JOINTS, qref=QREF)
    # Get initial frame placement + dimensions of joint space
    frame_name = config['frame_of_interest']
    id_endeff = robot.model.getFrameId(frame_name)
    nq, nv = robot.model.nq, robot.model.nv
    nx = nq+nv; nu = nq
    # Update robot model with initial state
    robot.framesForwardKinematics(q0)
    robot.computeJointJacobians(q0)
    M_ct = robot.data.oMf[id_endeff]



    # # # # # # # # # 
    ### OCP SETUP ###
    # # # # # # # # # 
    # Warm start and reg

    # Define initial state
    f_ext = pin_utils.get_external_joint_torques(M_ct, config['frameForceRef'], robot)
    u0 = pin_utils.get_tau(q0, v0, np.zeros((nq,1)), f_ext, robot.model, config['armature'])
    y0 = np.concatenate([x0, u0])
    # Setup Croco OCP and create solver
    ddp = OptimalControlProblemLPF(robot, config, lpf_joint_names=robot.model.names[1:]).initialize(y0, callbacks=True)
    # Warmstart and solve
    # ug = pin_utils.get_u_grav(q0, robot.model)
    xs_init = [y0 for i in range(config['N_h']+1)]
    us_init = [u0 for i in range(config['N_h'])]

    models = list(ddp.problem.runningModels) + [ddp.problem.terminalModel]
    for k,m in enumerate(models):
        m.differential.costs.costs["translation"].cost.residual.reference = np.array([0.65, 0., -0.01])
    for k,m in enumerate(models[:-1]):
        m.differential.costs.costs["force"].active = True
        m.differential.costs.costs["force"].cost.residual.reference = pin.Force(np.array([0., 0., 50, 0., 0., 0.]))
        
    ddp.with_callbacks = True
    ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=False)


    if(PLOT):
        #  Plot
        ddp_handler = DDPDataHandlerLPF(ddp, n_lpf=len(robot.model.names[1:]))
        ddp_data = ddp_handler.extract_data(ee_frame_name=frame_name, ct_frame_name=frame_name)
        _, _ = ddp_handler.plot_ddp_results(ddp_data, which_plots=['all'], 
                                                            colors=['r'], 
                                                            markers=['.'], 
                                                            SHOW=True)


    # Display solution in Gepetto Viewer
    if(DISPLAY):
        import crocoddyl
        display = crocoddyl.GepettoDisplay(robot, frameNames=[frame_name])
        display.displayFromSolver(ddp, factor=0.1)


if __name__=='__main__':
    args = misc_utils.parse_OCP_script(sys.argv[1:])
    main(args.robot_name, args.PLOT, args.DISPLAY)
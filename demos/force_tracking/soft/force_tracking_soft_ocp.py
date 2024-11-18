"""
@package force_feedback
@file demos/contact/aug_soft_contact_OCP.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft.
@date 2022-08-12
@brief OCP for static EE pose task  
"""

'''
The robot is tasked with applying a constant normal force in contact with a wall
Trajectory optimization using Crocoddyl using the DAMSoftcontactAugmented where contact force
is linear visco-elastic (spring damper model) and part of the state 
The goal of this script is to setup OCP (play with weights)
'''

import sys
sys.path.append('.')

from croco_mpc_utils.misc_utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger


import numpy as np  
np.set_printoptions(precision=4, linewidth=180)

from force_feedback_mpc.core_mpc_utils import path_utils, pin_utils, misc_utils

from soft_mpc.aug_ocp import OptimalControlProblemSoftContactAugmented
from soft_mpc.aug_data import DDPDataHandlerSoftContactAugmented
from soft_mpc.utils import SoftContactModel3D, SoftContactModel1D

def main(robot_name, PLOT, DISPLAY):


    # # # # # # # # # # # #
    ### LOAD ROBOT MODEL ## 
    # # # # # # # # # # # # 
    # Read config file
    # config, _ = path_utils.load_config_file(__file__, robot_name)
    config = path_utils.load_yaml_file('/home/skleff/ws/workspace/src/force_feedback_dgh/config/reduced_soft_mpc_contact1d.yml')
    q0 = np.asarray(config['q0'])
    v0 = np.asarray(config['dq0'])
    x0 = np.concatenate([q0, v0]) 
    # Get pin wrapper
    from robot_properties_kuka.config import IiwaReducedConfig
    CONTROLLED_JOINTS = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6']
    QREF              = np.zeros(7)
    robot                     = IiwaReducedConfig.buildRobotWrapper(controlled_joints=CONTROLLED_JOINTS, qref=QREF)
    # Get initial frame placement + dimensions of joint space
    frame_name = config['frame_of_interest']
    id_endeff = robot.model.getFrameId(frame_name)
    nq, nv = robot.model.nq, robot.model.nv
    nx = nq+nv; nu = nq
    # Update robot model with initial state
    robot.framesForwardKinematics(q0)
    robot.computeJointJacobians(q0)
    oMf = robot.data.oMf[id_endeff]
    # Contact model
    oPc = oMf.translation + np.asarray(config['oPc_offset'])
    oMc = oMf.copy()
    oMc.translation = oPc.copy()
    if('1D' in config['contactType']):
        softContactModel = SoftContactModel1D(np.asarray(config['Kp']), np.asarray(config['Kv']), oPc, id_endeff, config['contactType'], config['pinRefFrame'])
    else:
        softContactModel = SoftContactModel3D(np.asarray(config['Kp']), np.asarray(config['Kv']), oPc, id_endeff, config['pinRefFrame'])
    
    # v0 = np.random.rand(nv)
    f0 = 45 #200 #softContactModel.computeForce_(robot.model, q0, v0)
    y0 = np.hstack([x0, f0])  
    
    logger.debug(str(y0))
    # # # # # # # # # 
    ### OCP SETUP ###
    # # # # # # # # # 
    softContactModel.print()
    ddp = OptimalControlProblemSoftContactAugmented(robot, config).initialize(y0, softContactModel, callbacks=True)
    # Warmstart and solve
    import pinocchio as pin
    Nh = config['N_h']
    f_ext = pin_utils.get_external_joint_torques(oMc, config['frameForceRef'], robot)
    u0 = pin_utils.get_tau(q0, v0, np.zeros((nq,1)), f_ext, robot.model, config['armature'])
    
    # ddp.xs = [y0 for i in range(Nh+1)]
    # ddp.us = [u0 for i in range(Nh)]
    xs_init = [y0 for i in range(config['N_h']+1)]
    us_init = [u0 for i in range(config['N_h'])]

    models = list(ddp.problem.runningModels) + [ddp.problem.terminalModel]
    import pinocchio as pin
    for k,m in enumerate(models):
        m.differential.cost_ref = pin.LOCAL_WORLD_ALIGNED

    ddp.with_callbacks = True
    ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=False)
    
    print(ddp.xs[0])
    # werpighewoib
    if(PLOT):
        #  Plot
        ddp_handler = DDPDataHandlerSoftContactAugmented(ddp, softContactModel)
        ddp_data = ddp_handler.extract_data(ee_frame_name=frame_name, ct_frame_name=frame_name, model=robot.model)
        _, _ = ddp_handler.plot_ddp_results(ddp_data, which_plots=['f'], 
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

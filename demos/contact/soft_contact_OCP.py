"""
@package force_feedback
@file demos/contact/soft_contact_OCP.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft.
@date 2020-05-18
@brief OCP for static EE pose task  
"""

'''
The robot is tasked with applying a constant normal force in contact with a wall
Trajectory optimization using Crocoddyl using the DAMSoftcontact where contact force
is linear visco-elastic (spring damper model)
The goal of this script is to setup OCP (play with weights)
'''

import sys
sys.path.append('.')

from core_mpc_utils.misc_utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger


import numpy as np  
np.set_printoptions(precision=4, linewidth=180)

from core_mpc_utils import path_utils, pin_utils, misc_utils

from soft_mpc.ocp import OptimalControlProblemSoftContact
from soft_mpc.data import DDPDataHandlerSoftContact
from soft_mpc.utils import SoftContactModel3D, SoftContactModel1D

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
    oMf = robot.data.oMf[id_endeff]
    # Contact model
    oPc = oMf.translation + np.asarray(config['oPc_offset'])
    if('1D' in config['contactType']):
        softContactModel = SoftContactModel1D(config['Kp'], config['Kv'], oPc, id_endeff, config['contactType'], config['pinRefFrame'])
    else:
        softContactModel = SoftContactModel3D(config['Kp'], config['Kv'], oPc, id_endeff, config['pinRefFrame'])

    # # # # # # # # # 
    ### OCP SETUP ###
    # # # # # # # # # 
    # Warm start and reg
    # Compute initial visco-elastic force
    fext0 = softContactModel.computeExternalWrench(robot.model, robot.data)
    # Setup Croco OCP and create solver
    ddp = OptimalControlProblemSoftContact(robot, config).initialize(x0, softContactModel, callbacks=True)
    # Warmstart and solve
    xs_init = [x0 for i in range(config['N_h']+1)]
    us_init = [pin_utils.get_tau(q0, v0, np.zeros(nq), fext0, robot.model, np.zeros(nq)) for i in range(config['N_h'])] 
    ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=False)

    if(PLOT):
        #  Plot
        ddp_handler = DDPDataHandlerSoftContact(ddp, softContactModel)
        ddp_data = ddp_handler.extract_data(ee_frame_name=frame_name, ct_frame_name=frame_name)
        _, _ = ddp_handler.plot_ddp_results(ddp_data, which_plots=config['WHICH_PLOTS'], 
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
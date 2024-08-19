"""
@package force_feedback
@file contact_constrained_OCP.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft.
@date 2020-05-18
@brief OCP for normal force task
"""

'''
The robot is tasked with exerting a constant normal force at its EE
Trajectory optimization using Crocoddyl
The goal of this script is to setup OCP (a.k.a. play with weights)
The solver used here is either ProxQP or FADMM to handle inequality constraints on the force
'''


import sys
sys.path.append('.')

from core_mpc_utils.misc_utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger


import numpy as np  
np.set_printoptions(precision=4, linewidth=180)

from core_mpc_utils import path_utils, pin_utils, misc_utils

from classical_mpc.ocp import OptimalControlProblemClassicalWithConstraints
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
    frame_name = config['frameForceFrameName']
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
    # Setup Croco OCP and create solver
    ddp = OptimalControlProblemClassicalWithConstraints(robot, config).initialize(x0, callbacks=True)
    # Warmstart and solve
    # f_ext = pin_utils.get_external_joint_torques(M_ct, config['frameForceRef'], robot)
    # u0 = pin_utils.get_tau(q0, v0, np.zeros((nq,1)), f_ext, robot.model, config['armature'])
    xs_init = [x0 for i in range(config['N_h']+1)]
    us_init = [np.zeros(nv) for i in range(config['N_h'])]
    # import pdb # 
    # pdb.set_trace()
    ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=False)
    #  Plot
    if(PLOT):
        ddp_handler = DDPDataHandlerClassical(ddp)
        ddp_data = ddp_handler.extract_data(ee_frame_name=frame_name, ct_frame_name=frame_name)
        _, _ = ddp_handler.plot_ddp_results(ddp_data, which_plots=config['WHICH_PLOTS'], markers=['.'], colors=['b'], SHOW=True)


    # Display solution in Gepetto Viewer
    if(DISPLAY):
        import crocoddyl
        display = crocoddyl.GepettoDisplay(robot, frameNames=[frame_name])
        display.displayFromSolver(ddp, factor=0.1)


if __name__=='__main__':
    args = misc_utils.parse_OCP_script(sys.argv[1:])
    main(args.robot_name, args.PLOT, args.DISPLAY)
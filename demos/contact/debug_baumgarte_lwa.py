"""
@package force_feedback
@file debug_baumgarte_lwa.py
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
'''


import sys
sys.path.append('.')

from core_mpc_utils.misc_utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger


import numpy as np  
np.set_printoptions(precision=4, linewidth=180)

from core_mpc_utils import ocp, path_utils, pin_utils, plot_utils, data_utils, misc_utils



def main():


    # # # # # # # # # # # #
    ### LOAD ROBOT MODEL ## 
    # # # # # # # # # # # # 
    # Read config file
    config, _ = path_utils.load_config_file('contact_OCP', 'iiwa')
    q0 = np.asarray(config['q0'])
    v0 = np.asarray(config['dq0'])
    x0 = np.concatenate([q0, v0])   
    # Get pin wrapper
    robot = pin_utils.load_robot_wrapper('iiwa')
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
    ddp = ocp.init_DDP(robot, config, x0, callbacks=True) 
    # Warmstart and solve
    f_ext = pin_utils.get_external_joint_torques(M_ct, config['frameForceRef'], robot)
    u0 = pin_utils.get_tau(q0, v0, np.zeros((nq,1)), f_ext, robot.model, config['armature'])
    xs_init = [x0 for i in range(config['N_h']+1)]
    us_init = [u0 for i in range(config['N_h'])]
    ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=False)
    
    
    # SIMULATE USING ACTION MODEL + PD control + disturb initial state
    Kp = 1*np.ones(nq)  
    Kd = 2*np.sqrt(Kp) 
    xs = np.zeros((ddp.problem.T+1, nx))
    np.random.seed(1)
    x0 += 0.01*np.random.rand(nx)
    q0 = x0[:nq]
    xs[0,:] = x0
    f0 = ddp.problem.runningDatas[0].differential.multibody.contacts.contacts['contact'].f.vector
    f_ext = pin_utils.get_external_joint_torques(M_ct, f0, robot)
    us = np.zeros((ddp.problem.T, nq))
    for i in range(ddp.problem.T):
        # print("step "+str(i))
        q,v = xs[i,:nq], xs[i,nq:]
        a = - Kp * (q - q0) - Kd * v
        us[i,:] = a
        tau = pin_utils.get_tau(q, v, a, f_ext, robot.model, config['armature'])
        us[i,:] = tau.copy()
        m = ddp.problem.runningModels[i]
        d = m.createData()
        m.calc(d, xs[i,:], us[i,:])
        xs[i+1,:] = d.xnext
    # print(us[10:])

    ddp_data = data_utils.extract_ddp_data(ddp, ee_frame_name=frame_name, ct_frame_name=frame_name)
    ddp_data['xs'] = list(xs)
    ddp_data['us'] = list(us)
    _, _ = plot_utils.plot_ddp_results(ddp_data, which_plots=config['WHICH_PLOTS'], markers=['.'], colors=['b'], SHOW=True)


if __name__=='__main__':
    main()
"""
@package force_feedback
@file contact_OCP.py
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

from core_mpc_utils import path_utils, misc_utils

from croco_mpc_utils import pinocchio_utils as pin_utils
from croco_mpc_utils.ocp_data import OCPDataHandlerClassical
from croco_mpc_utils.ocp import OptimalControlProblemClassical

import mim_solvers
from mim_robots.robot_loader import load_pinocchio_wrapper
import os


PLOT = True

# # # # # # # # # # # #
### LOAD ROBOT MODEL ## 
# # # # # # # # # # # # 
# Read config file
config_name = 'force_tracking_classical'
config = path_utils.load_yaml_file(os.path.dirname(os.path.realpath(__file__))+'/'+config_name+'.yml')
q0 = np.asarray(config['q0'])
v0 = np.asarray(config['dq0'])
x0 = np.concatenate([q0, v0])   
# Get pin wrapper
robot = load_pinocchio_wrapper('iiwa_ft_sensor_shell', locked_joints=['A7'])
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
ocp = OptimalControlProblemClassical(robot, config).initialize(x0)
# solver = ocp_utils.init_DDP(robot, config, x0, callbacks=True) 
# Warmstart and solve
f_ext = pin_utils.get_external_joint_torques(M_ct, config['frameForceRef'], robot)
u0 = pin_utils.get_tau(q0, v0, np.zeros((nq,1)), f_ext, robot.model, config['armature'])
xs_init = [x0 for i in range(config['N_h']+1)]
us_init = [u0 for i in range(config['N_h'])]
solver = mim_solvers.SolverCSQP(ocp)

# !!! Activate contact models !!!
models = list(solver.problem.runningModels) + [solver.problem.terminalModel]
datas = list(solver.problem.runningDatas) + [solver.problem.terminalData]
for k,m in enumerate(models):
    m.differential.contacts.changeContactStatus("contact", True)

solver.setCallbacks([mim_solvers.CallbackVerbose(), mim_solvers.CallbackLogger()])
solver.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=False)
#  Plot
if(PLOT):
    ocp_handler = OCPDataHandlerClassical(ocp)
    ocp_data = ocp_handler.extract_data(solver.xs, solver.us)
    _, _ = ocp_handler.plot_ocp_results(ocp_data, which_plots=config['WHICH_PLOTS'], markers=['.'], colors=['b'], SHOW=True)
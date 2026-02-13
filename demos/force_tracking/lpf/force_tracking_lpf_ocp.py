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
import os
# Add python directory to path to support imports when running from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../python')))
sys.path.append('.')

from croco_mpc_utils.utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger


import numpy as np  
np.set_printoptions(precision=4, linewidth=180)

from force_feedback_mpc.core_mpc_utils import path_utils, misc_utils

from croco_mpc_utils import pinocchio_utils as pin_utils
from force_feedback_mpc.lpf_mpc.ocp import OptimalControlProblemLPF
from force_feedback_mpc.lpf_mpc.data import OCPDataHandlerLPF

import pinocchio as pin

import mim_solvers

PLOT = True
DISPLAY = True

# # # # # # # # # # # #
### LOAD ROBOT MODEL ## 
# # # # # # # # # # # # 
# Read config file
config_name = 'force_tracking_lpf'
config = path_utils.load_yaml_file(os.path.dirname(os.path.realpath(__file__)) + '/' + config_name + '.yml')
q0 = np.asarray(config['q0'])
v0 = np.asarray(config['dq0'])
x0 = np.concatenate([q0, v0])   
# Get pin wrapper
from mim_robots.robot_loader import load_pinocchio_wrapper
robot = load_pinocchio_wrapper('iiwa_ft_sensor_shell', locked_joints=['A7'])
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
ocp = OptimalControlProblemLPF(robot, config, lpf_joint_names=robot.model.names[1:])
problem = ocp.initialize(y0)
# Warmstart and solve
import mim_solvers
solver = mim_solvers.SolverSQP(problem)
solver.regMax = 1e6
# ug = pin_utils.get_u_grav(q0, robot.model)
xs_init = [y0 for i in range(config['N_h']+1)]
us_init = [u0 for i in range(config['N_h'])]

models = list(solver.problem.runningModels) + [solver.problem.terminalModel]
for k,m in enumerate(models):
    m.differential.costs.costs["translation"].cost.residual.reference = np.array([0.65, 0., -0.01])
for k,m in enumerate(models[:-1]):
    m.differential.costs.costs["force"].active = True
    m.differential.costs.costs["force"].cost.residual.reference = pin.Force(np.array([0., 0., 50, 0., 0., 0.]))
    
solver.with_callbacks = True
solver.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=False)


if(PLOT):
    #  Plot
    ddp_handler = OCPDataHandlerLPF(solver.problem, n_lpf=len(robot.model.names[1:]))
    ddp_data = ddp_handler.extract_data(ee_frame_name=frame_name, ct_frame_name=frame_name)
    _, _ = ddp_handler.plot_ddp_results(ddp_data, which_plots=['all'], 
                                                        colors=['r'], 
                                                        markers=['.'], 
                                                        SHOW=True)
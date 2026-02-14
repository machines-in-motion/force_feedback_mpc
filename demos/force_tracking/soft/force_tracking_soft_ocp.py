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
from force_feedback_mpc.soft_mpc.aug_ocp import OptimalControlProblemSoftContactAugmented
from force_feedback_mpc.soft_mpc.aug_data import OCPDataHandlerSoftContactAugmented
from force_feedback_mpc.soft_mpc.utils import SoftContactModel3D, SoftContactModel1D

import mim_solvers

PLOT = True
DISPLAY = True

# # # # # # # # # # # #
### LOAD ROBOT MODEL ## 
# # # # # # # # # # # # 
# Read config file
config_name = 'force_tracking_soft'
config = path_utils.load_yaml_file(os.path.dirname(os.path.realpath(__file__)) + '/' + config_name + '.yml')
q0 = np.asarray(config['q0'])
v0 = np.asarray(config['dq0'])
x0 = np.concatenate([q0, v0]) 
# Get pin wrapper
from mim_robots.robot_loader import load_pinocchio_wrapper
robot = load_pinocchio_wrapper('iiwa_ft_sensor_shell', locked_joints=['A7', 'A6', 'A5'])
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

# helper to ensure 1D array
def get_1d_array(val):
    arr = np.asarray(val)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr

Kp = get_1d_array(config['Kp'])
Kv = get_1d_array(config['Kv'])

if('1D' in config['contactType']):
    softContactModel = SoftContactModel1D(Kp, Kv, oPc, id_endeff, config['contactType'], config['pinRefFrame'])
else:
    softContactModel = SoftContactModel3D(Kp, Kv, oPc, id_endeff, config['pinRefFrame'])
y0 = np.hstack([x0, softContactModel.computeForce_(robot.model, q0, v0)])  
logger.debug(str(y0))
# v0 = np.random.rand(nv)
f0 = 45 #200 #softContactModel.computeForce_(robot.model, q0, v0)
y0 = np.hstack([x0, f0])  

logger.debug(str(y0))
# # # # # # # # # 
### OCP SETUP ###
# # # # # # # # # 
softContactModel.print()
# Initialize OCP once
ocp_wrapper = OptimalControlProblemSoftContactAugmented(robot, config)
problem = ocp_wrapper.initialize(y0, softContactModel)
# Warmstart and solve
import pinocchio as pin
import mim_solvers
solver = mim_solvers.SolverSQP(problem)
solver.regMax = 1e6
Nh = config['N_h']
f_ext = pin_utils.get_external_joint_torques(oMc, config['frameForceRef'], robot)
u0 = pin_utils.get_tau(q0, v0, np.zeros((nq,1)), f_ext, robot.model, config['armature'])

# ddp.xs = [y0 for i in range(Nh+1)]
# ddp.us = [u0 for i in range(Nh)]
xs_init = [y0 for i in range(config['N_h']+1)]
us_init = [u0 for i in range(config['N_h'])]

models = list(solver.problem.runningModels) + [solver.problem.terminalModel]
import pinocchio as pin
for k,m in enumerate(models):
    m.differential.cost_ref = pin.LOCAL_WORLD_ALIGNED

solver.setCallbacks([mim_solvers.CallbackVerbose(), mim_solvers.CallbackLogger()])
solver.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=False)

print(solver.xs[0])

if(PLOT):
    #  Plot
    ocp_data_handler = OCPDataHandlerSoftContactAugmented(solver.problem, softContactModel)
    ocp_data = ocp_data_handler.extract_data(solver.xs, solver.us, model=robot.model)
    _, _ = ocp_data_handler.plot_ocp_results(ocp_data, which_plots=['f'], 
                                                        colors=['r'], 
                                                        markers=['.'], 
                                                        SHOW=True)
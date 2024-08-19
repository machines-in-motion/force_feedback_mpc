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

from croco_mpc_utils.utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger


import numpy as np  
np.set_printoptions(precision=4, linewidth=180)

from croco_mpc_utils import pinocchio_utils
from croco_mpc_utils.utils import *

from core_mpc_utils import misc_utils

from soft_mpc.aug_ocp_constraints import OptimalControlProblemSoftContactAugmentedWithConstraints
from soft_mpc.aug_data import OCPDataHandlerSoftContactAugmented
from soft_mpc.utils import SoftContactModel3D, SoftContactModel1D

from mim_robots.robot_loader import load_pinocchio_wrapper
import mim_solvers

# def main(robot_name, PLOT, DISPLAY):


# # # # # # # # # # # #
### LOAD ROBOT MODEL ## 
# # # # # # # # # # # # 
# Read config file
# config, _ = path_utils.load_config_file(__file__, robot_name)
config = load_yaml_file('/home/skleff/force-feedback/demos/contact/config/force_tracking_soft_constraints.yml')
q0 = np.asarray(config['q0'])
v0 = np.asarray(config['dq0'])
x0 = np.concatenate([q0, v0]) 
# Get pin wrapper
robot                     = load_pinocchio_wrapper('iiwa', locked_joints=['A7'])
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
f0 = 0 #200 #softContactModel.computeForce_(robot.model, q0, v0)
y0 = np.hstack([x0, f0])  

logger.debug(str(y0))
# # # # # # # # # 
### OCP SETUP ###
# # # # # # # # # 
softContactModel.print()
# config['N_h'] = 10
# config['frameForceWeight'] = [0.1]
problem = OptimalControlProblemSoftContactAugmentedWithConstraints(robot, config).initialize(y0, softContactModel)
# Warmstart and solve
f_ext = pinocchio_utils.get_external_joint_torques(oMc, config['frameForceRef'], robot)
u0 = pinocchio_utils.get_tau(q0, v0, np.zeros((nq,1)), f_ext, robot.model) #, config['armature'])

xs_init = [y0 for i in range(config['N_h']+1)]
us_init = [u0 for i in range(config['N_h'])]

# models = list(problem.runningModels) + [problem.terminalModel]
# print(problem.runningModels)
# for k,m in enumerate(models):
#     print(m.differential.constraint)

# solver = mim_solvers.SolverSQP(problem)
solver = mim_solvers.SolverCSQP(problem)
solver.termination_tolerance = 1e-4
solver.with_callbacks = True 
max_iter = 100
# solver.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=False)

# PLOT = True
# if(PLOT):
#     #  Plot
#     ddp_handler = OCPDataHandlerSoftContactAugmented(solver.problem, softContactModel)
#     ddp_data = ddp_handler.extract_data(solver.xs, solver.us, robot.model)
#     _, _ = ddp_handler.plot_ocp_results(ddp_data, which_plots=['f'], 
#                                                         colors=['r'], 
#                                                         markers=['.'], 
#                                                         SHOW=True)

# # if __name__=='__main__':
# #     args = misc_utils.parse_OCP_script(sys.argv[1:])
# #     main(args.robot_name, args.PLOT, args.DISPLAY)

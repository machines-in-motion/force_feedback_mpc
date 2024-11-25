"""
@package force_feedback
@file aug_soft_sanding_MPC.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2021, New York University & LAAS-CNRS
@date 2023-04-04
@brief Testing OCP with low-pass filter actuation (LPF)
"""

import sys
sys.path.append('.')

from croco_mpc_utils.utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger


import numpy as np  
np.set_printoptions(precision=4, linewidth=180)
RANDOM_SEED = 1 #19


from force_feedback_mpc.core_mpc_utils import path_utils, misc_utils, mpc_utils

from croco_mpc_utils import pinocchio_utils as pin_utils

from force_feedback_mpc.lpf_mpc.data import OCPDataHandlerLPF
from force_feedback_mpc.lpf_mpc.ocp import OptimalControlProblemLPF, getJointAndStateIds
from croco_mpc_utils.utils import load_yaml_file

import mim_solvers
from mim_robots.robot_loader import load_pinocchio_wrapper



import time
import pinocchio as pin
import os


# # # # # # # # # # # # # # # # # # #
### LOAD ROBOT MODEL and SIMU ENV ### 
# # # # # # # # # # # # # # # # # # # 
# Read config file
config_name = 'lpf_ocp_1d'
config = path_utils.load_yaml_file(os.path.dirname(os.path.realpath(__file__))+'/'+config_name+'.yml')
# Create a simulation environment & simu-pin wrapper 
q0 = np.asarray(config['q0'])
v0 = np.asarray(config['dq0'])
x0 = np.concatenate([q0, v0])  
robot = load_pinocchio_wrapper('iiwa_convex_ft_sensor_shell', locked_joints=['A7'])

# Get dimensions 
nq, nv = robot.model.nq, robot.model.nv; nu = nq
# Placement of LOCAL end-effector frame w.r.t. WORLD frame
frame_of_interest = config['frame_of_interest']
id_endeff = robot.model.getFrameId(frame_of_interest)

from force_feedback_mpc.core_mpc_utils import sim_utils
sim_utils.setup_obstacle_collision_no_sim(robot, config)

# # # # # # # # # 
### OCP SETUP ###
# # # # # # # # # 
# Apply masks on joints to extract LPF joints
u0 = pin_utils.get_u_grav(q0, robot.model)
lpf_joint_names = robot.model.names[1:] #['A1', 'A2', 'A3', 'A4'] #  #
_, lpfStateIds = getJointAndStateIds(robot.model, lpf_joint_names)
n_lpf = len(lpf_joint_names)
_, nonLpfStateIds = getJointAndStateIds(robot.model, list(set(robot.model.names[1:]) - set(lpf_joint_names)) )
logger.debug("LPF state ids ")
logger.debug(lpfStateIds)
logger.debug("Non LPF state ids ")
logger.debug(nonLpfStateIds)
y0 = np.concatenate([x0, u0[lpfStateIds]])
      
# Init shooting problem and solver
ocp = OptimalControlProblemLPF(robot, config, lpf_joint_names).initialize(y0)
# Warmstart and solve
xs_init = [y0 for _ in range(config['N_h']+1)] 
us_init = [u0 for _ in range(config['N_h'])]
solver = mim_solvers.SolverCSQP(ocp)
solver.with_callbacks         = config['with_callbacks']
solver.use_filter_line_search = config['use_filter_line_search']
solver.filter_size            = config['filter_size']
solver.warm_start             = config['warm_start']
solver.termination_tolerance  = config['solver_termination_tolerance']
solver.max_qp_iters           = config['max_qp_iter']
solver.eps_abs                = config['qp_termination_tol_abs']
solver.eps_rel                = config['qp_termination_tol_rel']
solver.warm_start_y           = config['warm_start_y']
solver.reset_rho              = config['reset_rho']  
solver.mu_dynamic             = config["mu_dynamic"]
solver.mu_constraint          = config["mu_constraint"]
solver.regMax                 = 1e6
solver.reg_max                = 1e6
models = list(ocp.runningModels) + [ocp.terminalModel]
datas = list(solver.problem.runningDatas) + [solver.problem.terminalData]
for k,m in enumerate(models):
    m.differential.costs.costs["translation"].active = False
    m.differential.contacts.changeContactStatus("contact", False)
    m.differential.costs.costs['rotation'].active = False
    m.differential.costs.costs['rotation'].cost.residual.reference = pin.utils.rpyToMatrix(np.pi, 0., np.pi)
    # set each collision constraint bounds to [0, inf]
    if(k!=0 and k!= config['N_h']):
        m.differential.constraints.constraints['forceBox'].constraint.updateBounds(
                    np.array([0.]),
                    np.array([25])) 
        m.differential.constraints.changeConstraintStatus('forceBox', True)
    for col_idx in range(len(robot.collision_model.collisionPairs)):
        # only populates the bounds of the constraint item (not the manager)
        m.differential.constraints.constraints['collisionBox_' + str(col_idx)].constraint.updateBounds(
                    np.array([0.]),
                    np.array([np.inf])) 
        # needed to pass the bounds to the manager
        m.differential.constraints.changeConstraintStatus('collisionBox_' + str(col_idx), True)

m = models[0]
d = datas[0]
for with_armature in [True, False]:
    # Check calc of IAM LPF
    for with_lpf_torque_constraint in [True, False]:   
        print("torque_cstr = "+str(with_lpf_torque_constraint))   
        m.with_lpf_torque_constraint = with_lpf_torque_constraint
        print("        > Check IAM.calc")
        m.calc(d, y0, u0)
        print("        > Check IAM.calcDiff")
        m.calcDiff(d, y0, u0)


# for k,m in enumerate(models):
#     m.differential.active_contact = False
#     m.differential.f_des = np.zeros(1)
#     m.differential.cost_ref = pin.LOCAL_WORLD_ALIGNED
#     m.differential.ref     = pin.LOCAL_WORLD_ALIGNED
#     # FORCE CONSTRAINT STUFF
#     # # if(k!=0):
#     # # m.force_lb =  np.array([-10000.])
#     # m.with_force_constraint = True
#     #     # m.force_lb = np.array([-10000.])
#     #     # m.force_ub = np.array([10000.])
#     # # m.force_ub =  np.array([10000.])
#     # COLLISION CONSTRAINT STUFF
#     # set each collision constraint bounds to [0, inf]
#     for col_idx in range(len(robot.collision_model.collisionPairs)):
#         # only populates the bounds of the constraint item (not the manager)
#         m.differential.constraints.constraints['collisionBox_' + str(col_idx)].constraint.updateBounds(
#                     np.array([0.]),
#                     np.array([np.inf])) 
#         # needed to pass the bounds to the manager
#         m.differential.constraints.changeConstraintStatus('collisionBox_' + str(col_idx), True)
#         # need to set explicitly the IAM bounds
#         # m.g_lb = -0.0001*np.ones([m.ng]) # needs to be slightly negative (bug to investigate)
#         m.g_lb = np.zeros([m.ng]) # needs to be slightly negative (bug to investigate)
#         m.g_ub = np.array([np.inf]*m.ng)
# # wfhwef
# solver.setCallbacks([mim_solvers.CallbackVerbose(), mim_solvers.CallbackLogger()])
# solver.solve(xs_init, us_init, maxiter=100, isFeasible=False)
"""
@package force_feedback
@file aug_soft_sanding_MPC.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2021, New York University & LAAS-CNRS
@date 2023-04-04
@brief Testing OCP with visco-elastic contact (1Dz) 
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

from force_feedback_mpc.soft_mpc.aug_ocp import OptimalControlProblemSoftContactAugmented
from force_feedback_mpc.soft_mpc.aug_data import OCPDataHandlerSoftContactAugmented
from force_feedback_mpc.soft_mpc.utils import SoftContactModel1D
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
config_name = 'polishing_soft_obstacle'
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

# Contact model
oPc=np.asarray(config['contactPosition']) + np.asarray(config['oPc_offset'])
softContactModel = SoftContactModel1D(Kp=np.asarray(config['Kp']), 
                                      Kv=np.asarray(config['Kv']), 
                                      oPc=oPc,
                                      frameId=id_endeff, 
                                      contactType=config['contactType'], 
                                      pinRef=config['pinRefFrame'])

# Measure initial force in pybullet
f0 = np.zeros(6)
assert(softContactModel.nc == 1)
assert(softContactModel.pinRefFrame == pin.LOCAL_WORLD_ALIGNED)
softContactModel.print()

MASK = softContactModel.mask
y0 = np.concatenate([ x0, np.array([f0[MASK]]) ])  
RESET_ANCHOR_POINT = bool(config['RESET_ANCHOR_POINT'])
anchor_point = oPc.copy()

# # # # # # # # # 
### OCP SETUP ###
# # # # # # # # #
# Compute initial gravity compensation torque torque   
f_ext0 = [pin.Force.Zero() for _ in robot.model.joints] # pin_utils.get_external_joint_torques(contact_placement, f0, robot)   
y0 = np.concatenate([x0, f0[-softContactModel.nc:]])  
u0 = pin_utils.get_tau(q0, v0, np.zeros(nq), f_ext0, robot.model, np.zeros(nq))
# Warmstart and solve
xs_init = [y0 for i in range(config['N_h']+1)]
us_init = [u0 for i in range(config['N_h'])] 
# Setup Croco OCP and create solver
ocp = OptimalControlProblemSoftContactAugmented(robot, config).initialize(y0, softContactModel)

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
# !!! Deactivate all costs & contact models initially !!!
models = list(solver.problem.runningModels) + [solver.problem.terminalModel]
datas = list(solver.problem.runningDatas) + [solver.problem.terminalData]
for k,m in enumerate(models):
    m.differential.costs.costs["translation"].active = False
    m.differential.active_contact = False
    m.differential.f_des = np.zeros(1)
    m.differential.cost_ref = pin.LOCAL_WORLD_ALIGNED
    m.differential.ref = pin.LOCAL_WORLD_ALIGNED
    m.differential.costs.costs['rotation'].active = False
    m.differential.costs.costs['rotation'].cost.residual.reference = pin.utils.rpyToMatrix(np.pi, 0., np.pi)
    # FORCE CONSTRAINT STUFF
    # # if(k!=0):
    # # m.force_lb =  np.array([-10000.])
    # m.with_force_constraint = True
    #     # m.force_lb = np.array([-10000.])
    #     # m.force_ub = np.array([10000.])
    # # m.force_ub =  np.array([10000.])
    # COLLISION CONSTRAINT STUFF
    # set each collision constraint bounds to [0, inf]
    for col_idx in range(len(robot.collision_model.collisionPairs)):
        # only populates the bounds of the constraint item (not the manager)
        m.differential.constraints.constraints['collisionBox_' + str(col_idx)].constraint.updateBounds(
                    np.array([0.]),
                    np.array([np.inf])) 
        # needed to pass the bounds to the manager
        m.differential.constraints.changeConstraintStatus('collisionBox_' + str(col_idx), True)
        # need to set explicitly the IAM bounds
        # m.g_lb = -0.0001*np.ones([m.ng]) # needs to be slightly negative (bug to investigate)
        m.g_lb = np.zeros([m.ng]) # needs to be slightly negative (bug to investigate)
        m.g_ub = np.array([np.inf]*m.ng)
# wfhwef
# solver.setCallbacks([mim_solvers.CallbackVerbose(), mim_solvers.CallbackLogger()])
solver.solve(xs_init, us_init, maxiter=100, isFeasible=False)
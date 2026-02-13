"""
@package force_feedback
@file demos/contact/aug_soft_contact_OCP.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft.
@date 2022-08-12
@brief OCP for sanding task  
"""

'''
The robot is tasked with exerting a constant normal force at its EE
while drawing a circle on the contact surface
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

from force_feedback_mpc.core_mpc_utils import path_utils, misc_utils

from croco_mpc_utils import pinocchio_utils as pin_utils
from force_feedback_mpc.soft_mpc.aug_ocp import OptimalControlProblemSoftContactAugmented
from force_feedback_mpc.soft_mpc.aug_data import OCPDataHandlerSoftContactAugmented
from force_feedback_mpc.soft_mpc.utils import SoftContactModel1D
from croco_mpc_utils.math_utils import circle_point_WORLD

import mim_solvers
from mim_robots.robot_loader import load_pinocchio_wrapper
import os

# def main(robot_name, PLOT, DISPLAY):


# # # # # # # # # # # #
### LOAD ROBOT MODEL ## 
# # # # # # # # # # # # 
# Read config file
config_name = 'polishing_soft'
config = path_utils.load_yaml_file(os.path.dirname(os.path.realpath(__file__))+'/'+config_name+'.yml')
q0 = np.asarray(config['q0'])
v0 = np.asarray(config['dq0'])
x0 = np.concatenate([q0, v0]) 
# Get pin wrapper
robot = load_pinocchio_wrapper('iiwa', locked_joints=['A7'])
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
    softContactModel = SoftContactModel1D(np.asarray(config['Kp']), np.asarray(config['Kv']), oPc, id_endeff, config['contactType'], config['pinRefFrame'])
else:
    softContactModel = SoftContactModel3D(np.asarray(config['Kp']), np.asarray(config['Kv']), oPc, id_endeff, config['pinRefFrame'])
y0 = np.hstack([x0, softContactModel.computeForce_(robot.model, q0, v0)])  
logger.debug(str(y0))



# # # # # # # # # 
### OCP SETUP ###
# # # # # # # # # 
# Warm start and reg

# Setup Croco OCP and create solver
softContactModel.print()
ocp = OptimalControlProblemSoftContactAugmented(robot, config).initialize(y0, softContactModel)
# Warmstart and solve
xs_init = [y0 for i in range(config['N_h']+1)]
fext0 = softContactModel.computeExternalWrench_(robot.model, y0[:nq], y0[:nv])
us_init = [pin_utils.get_tau(y0[:nq], y0[:nv], np.zeros(nv), fext0, robot.model, np.zeros(nv)) for i in range(config['N_h'])] 

# Set the force cost reference frame to LWA 
models = list(ocp.runningModels) + [ocp.terminalModel]
import pinocchio as pin
for k,m in enumerate(models):
    m.differential.cost_ref = pin.LOCAL_WORLD_ALIGNED

# Setup tracking problem with circle ref EE trajectory
RADIUS = config['frameCircleTrajectoryRadius'] 
OMEGA  = config['frameCircleTrajectoryVelocity']
for k,m in enumerate(models):
    # Ref
    t = min(k*config['dt'], 2*np.pi/OMEGA)
    p_ee_ref = circle_point_WORLD(t, oMf, 
                                            radius=RADIUS,
                                            omega=OMEGA,
                                            LOCAL_PLANE=config['CIRCLE_LOCAL_PLANE'])
    # Cost translation
    m.differential.costs.costs['translation'].cost.residual.reference = p_ee_ref
    # Contact model 1D update z ref (WORLD frame)
    m.differential.oPc[:2] = p_ee_ref[:2]


# Warm start state = IK of circle trajectory
logger.info("Computing warm-start using Inverse Kinematics...")
xs_init = [] 
us_init = []
q_ws = q0
for k,m in enumerate(list(ocp.runningModels) + [ocp.terminalModel]):
    # Get ref placement
    p_ee_ref = m.differential.costs.costs['translation'].cost.residual.reference
    Mref = oMf.copy()
    Mref.translation = p_ee_ref
    # Get joint state from IK
    q_ws, v_ws, eps = pin_utils.IK_placement(robot, q_ws, id_endeff, Mref, DT=1e-2, IT_MAX=100)
    xs_init.append(np.concatenate([q_ws, v_ws, np.array([softContactModel.computeForce_(robot.model, q_ws, v_ws)])]))

solver = mim_solvers.SolverSQP(ocp)
solver.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=False)

if(PLOT):
    #  Plot
    ddp_handler = OCPDataHandlerSoftContactAugmented(solver.problem, softContactModel)
    ddp_data = ddp_handler.extract_data(solver.xs, solver.us, robot.model)
    _, _ = ddp_handler.plot_ocp_results(ddp_data, which_plots=config['WHICH_PLOTS'], 
                                                        colors=['r'], 
                                                        markers=['.'], 
                                                        SHOW=True)


# Display solution in Gepetto Viewer
if(DISPLAY):
    import crocoddyl
    display = crocoddyl.GepettoDisplay(robot, frameNames=[frame_name])
    display.displayFromSolver(solver, factor=0.1)


# if __name__=='__main__':
#     args = misc_utils.parse_OCP_script(sys.argv[1:])
#     main(args.robot_name, args.PLOT, args.DISPLAY)

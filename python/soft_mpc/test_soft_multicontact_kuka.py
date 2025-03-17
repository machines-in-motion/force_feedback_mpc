'''
Testing the multicontact API for soft contacts on the Kuka arm
Solving an OCP with a single visco-elastic contact 3D on the end-effector
'''
import numpy as np

import crocoddyl
import pinocchio as pin

from soft_multicontact_api import ViscoElasticContact3d_Multiple, ViscoElasticContact3D
from soft_multicontact_api import FrictionConeConstraint, ForceBoxConstraint, ForceConstraintManager
from soft_multicontact_api import ForceCost, ForceCostManager
from soft_multicontact_api import DAMSoftContactDynamics3D_Go2, IAMSoftContactDynamics3D_Go2


## KUKA EXAMPLE FOR DEBUGGING
from mim_robots.robot_loader import load_pinocchio_wrapper
robot                  = load_pinocchio_wrapper('iiwa')
state                  = crocoddyl.StateMultibody(robot.model)
actuation              = crocoddyl.ActuationModelFull(state)
# costs                  = crocoddyl.CostModelSum(state, actuation.nu)
# constraintModelManager = crocoddyl.ConstraintModelManager(state, actuation.nu)
frameId                = robot.model.getFrameId('contact')
Kp                     = 10
Kv                     = 10 
oPc                    = np.array([0.65, 0., 0.01])

# Initial conditions
q = np.array([0., 1.05, 0., -1.13, 0.2,  0.79, 0.]) # pin.randomConfiguration(robot.model)
v = np.zeros(robot.model.nv)
x = np.concatenate([q, v])
u = np.random.rand(actuation.nu)
MU = 0.7 # friction coeff
CONTACT    = True
CONSTRAINT = False
FRICTION_C = True
FORCE_C    = False   
BOTH_C     = False
FORCE_COST = True   
if(CONTACT):
    f = np.random.rand(3)
else:
    f = np.random.rand(0)

y = np.concatenate([x, f])

N             = 10
runningModels = []
for i in range(N):
    # Costs
    costs = crocoddyl.CostModelSum(state, actuation.nu)
    xRegCost = crocoddyl.CostModelResidual(state, 
                                          crocoddyl.ActivationModelWeightedQuad(np.ones(state.nx)**2), 
                                          crocoddyl.ResidualModelState(state, x, actuation.nu))
    uRegCost = crocoddyl.CostModelResidual(state, 
                                          crocoddyl.ActivationModelWeightedQuad(np.ones(actuation.nu)**2), 
                                          crocoddyl.ResidualModelControlGrav(state))
    costs.addCost("stateReg", xRegCost, 1e-2)
    costs.addCost("ctrlReg", uRegCost, 1e-5)

    # Soft contact models 
    if(CONTACT):
        softContactModelsStack = ViscoElasticContact3d_Multiple(state, actuation, [ViscoElasticContact3D(state, actuation, frameId, oPc, Kp, Kv, pin.LOCAL_WORLD_ALIGNED)]) 
    else:
        softContactModelsStack = None

    # Standard constraints 
    if(CONSTRAINT):
        constraintModelManager = crocoddyl.ConstraintModelManager(state, actuation.nu)
        uBoxCstr = crocoddyl.ConstraintModelResidual(state, crocoddyl.ResidualModelControl(state, actuation.nu), -robot.model.effortLimit, robot.model.effortLimit)  
        xlb = np.concatenate([robot.model.lowerPositionLimit, [-np.inf]*robot.model.nv])
        xub = np.concatenate([robot.model.upperPositionLimit, [np.inf]*robot.model.nv])
        xBoxCstr = crocoddyl.ConstraintModelResidual(state, crocoddyl.ResidualModelState(state, actuation.nu), xlb, xub)  
        constraintModelManager.addConstraint("ctrlBox", uBoxCstr)
        constraintModelManager.addConstraint("stateBox", xBoxCstr)
    else:
        constraintModelManager = None
    
    # Custom force cost in DAM
    if(FORCE_COST):
        forceCostManager = ForceCostManager([ForceCost(state, frameId, np.array([0]*3), 0., pin.LOCAL_WORLD_ALIGNED)], softContactModelsStack)
    else:
        forceCostManager = None

    # Create DAM with soft contact models, force costs + standard cost & constraints
    dam = DAMSoftContactDynamics3D_Go2(state, actuation, costs, softContactModelsStack, constraintModelManager, forceCostManager)
    
    # Custom force constraints for IAM
    if(FORCE_C):
        forceConstraintManager = ForceConstraintManager([ForceBoxConstraint(frameId, np.array([-np.inf]*3), np.array([np.inf]*3))], softContactModelsStack)
    elif(FRICTION_C):
        forceConstraintManager = ForceConstraintManager([FrictionConeConstraint(frameId, MU)], softContactModelsStack)
    elif(BOTH_C):
        forceConstraintManager = ForceConstraintManager([FrictionConeConstraint(frameId, MU),
                                                         ForceBoxConstraint(frameId, np.array([-np.inf]*3), np.array([np.inf]*3))], softContactModelsStack)
    else:
        forceConstraintManager = None

    # Create custom IAM with force constraints
    iam = IAMSoftContactDynamics3D_Go2(dam, dt=0.01, withCostResidual=True, forceConstraintManager=forceConstraintManager)

    runningModels.append(iam)


import mim_solvers
# Create shooting problem
ocp = crocoddyl.ShootingProblem(y, runningModels, runningModels[-1])
# ocp.x0 = y

solver = mim_solvers.SolverCSQP(ocp)
solver.max_qp_iters = 1000
max_iter = 500
solver.with_callbacks = True
solver.use_filter_line_search = False
solver.termination_tolerance = 1e-4
solver.eps_abs = 1e-6
solver.eps_rel = 1e-6

xs = [y]*(solver.problem.T + 1)
# us = [u]*solver.problem.T
us = solver.problem.quasiStatic([y]*solver.problem.T) 
solver.setCallbacks([mim_solvers.CallbackVerbose(), mim_solvers.CallbackLogger()])
solver.solve(xs, us, max_iter)   
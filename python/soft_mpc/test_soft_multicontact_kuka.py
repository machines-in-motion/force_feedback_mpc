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

SOLVING_OCP = False
TESTING_API = True

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

if(SOLVING_OCP):
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


if(TESTING_API):
    from numpy.random import rand
    from numpy.linalg import norm 
    np.random.seed(10)
    TOL = 1e-4
    # Numerical difference function
    def numdiff(f,x0,h=1e-6):
        f0 = f(x0).copy()
        x = x0.copy()
        Fx = []
        for ix in range(len(x)):
            x[ix] += h
            Fx.append((f(x)-f0)/h)
            x[ix] = x0[ix]
        return np.array(Fx).T


    # Compute differential action model derivatives
    dam = iam.differential
    dad = dam.createData()
    # dad = iad.differential
    dam.calc(dad, x, f, u)
    dam.calcDiff(dad, x, f, u)
    dxdot_dx = dad.Fx
    dxdot_df = dad.dABA_df
    dxdot_du = dad.Fu
    dfdot_dx = dad.dfdt_dx
    dfdot_df = dad.dfdt_df
    dfdot_du = dad.dfdt_du

    # Compute integral action model derivatives
    iam = runningModels[0]
    iad = iam.createData()
    iam.calc(iad, y, u)
    iam.calcDiff(iad, y, u)
    dcost_dy = iad.Lx
    dcost_du = iad.Lu
    dynext_dy = iad.Fx 
    dynext_du = iad.Fu 

    # Finite differences
    def get_xdot(dam, dad, x, f, u):
        dam.calc(dad, x, f, u)
        return dad.xout 
    
    def get_fdot(dam, dad, x, f, u):
        dam.calc(dad, x, f, u)
        return dad.fout

    def get_ynext(iam, iad, y, u):
        iam.calc(iad, y, u)
        return iad.xnext 
    
    def get_iam_cost(iam, iad, y, u):
        iam.calc(iad, y, u)
        return iad.cost

    dxdot_dx_ND = numdiff(lambda x_:get_xdot(dam, dad, x_, f, u), x)
    dxdot_df_ND = numdiff(lambda f_:get_xdot(dam, dad, x, f_, u), f)
    dxdot_du_ND = numdiff(lambda u_:get_xdot(dam, dad, x, f, u_), u)
    dfdot_dx_ND = numdiff(lambda x_:get_fdot(dam, dad, x_, f, u), x)
    dfdot_df_ND = numdiff(lambda f_:get_fdot(dam, dad, x, f_, u), f)
    dfdot_du_ND = numdiff(lambda u_:get_fdot(dam, dad, x, f, u_), u)

    dynext_dy_ND = numdiff(lambda y_:get_ynext(iam, iad, y_, u), y)
    dynext_du_ND = numdiff(lambda u_:get_ynext(iam, iad, y, u_), u)
    # dcost_dy_ND = numdiff(lambda y_:get_iam_cost(iam, iad, y_, u), y)
    # dcost_du_ND = numdiff(lambda u_:get_iam_cost(iam, iad, y, u_), u)

    assert(norm(dxdot_dx - dxdot_dx_ND) <= TOL)
    assert(norm(dxdot_df - dxdot_df_ND) <= TOL)
    assert(norm(dxdot_du - dxdot_du_ND) <= TOL)
    assert(norm(dfdot_dx - dfdot_dx_ND) <= TOL)
    assert(norm(dfdot_df - dfdot_df_ND) <= TOL)
    assert(norm(dfdot_du - dfdot_du_ND) <= TOL)
    assert(norm(dynext_dy - dynext_dy_ND) <= TOL)
    assert(norm(dynext_du - dynext_du_ND) <= TOL)
    # assert(norm(dcost_dy - dcost_dy_ND) <= TOL)
    # assert(norm(dcost_du - dcost_dy_ND) <= TOL)
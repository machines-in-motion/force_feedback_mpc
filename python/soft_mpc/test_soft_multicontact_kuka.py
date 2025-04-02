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
rmodel = robot.model
rdata = robot.data
# Initial conditions
q = np.array([0., 1.05, 0., -1.13, 0.2,  0.79, 0.]) # pin.randomConfiguration(robot.model)
v = np.zeros(robot.model.nv)
x = np.concatenate([q, v])
u = np.random.rand(actuation.nu)
MU = 0.7 # friction coeff
CONTACT    = True
CONSTRAINT = True
FRICTION_C = False
FORCE_C    = False   
BOTH_C     = True
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
    DAM = DAMSoftContactDynamics3D_Go2(state, actuation, costs, softContactModelsStack, constraintModelManager, forceCostManager)
    
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
    IAM = IAMSoftContactDynamics3D_Go2(DAM, dt=0.01, withCostResidual=True, forceConstraintManager=forceConstraintManager)

    runningModels.append(IAM)

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
    from test_utils import get_fdot, \
                           get_iam_cost, \
                           get_xdot, \
                           get_ynext_y, \
                           get_dam_cost, \
                           get_cstr
    from test_utils import numdiff, \
                           numdiff_q_dam_dyn, \
                           numdiff_u_iam_cost, \
                           numdiff_u_iam_dyn, \
                           numdiff_x_iam_cost, \
                           numdiff_x_iam_dyn, \
                           numdiff_x_iam_cstr, \
                           numdiff_u_iam_cstr, \
                           numdiff_q_dam_cost, \
                           numdiff_vfu_dam_cost


    # Compute running IAM derivatives
    IAM = runningModels[1]
    IAD = IAM.createData()
    IAM.calc(IAD, y, u)
    IAM.calcDiff(IAD, y, u)
    dcost_dy = IAD.Lx
    dcost_du = IAD.Lu
    dynext_dx = IAD.Fx 
    dynext_du = IAD.Fu 
    dcstr_dy = IAD.Gx
    dcstr_du = IAD.Gu

    # Compute terminal IAM derivatives
    IAM_t = runningModels[-1]
    IAD_t = IAM_t.createData()
    IAM_t.calc(IAD_t, y)
    IAM_t.calcDiff(IAD_t, y)
    dcost_dy_t = IAD_t.Lx
    dynext_dy_t = IAD_t.Fx 
    dcstr_dy_t = IAD_t.Gx

    # Compute running DAM derivatives
    DAM = IAM.differential
    DAD = DAM.createData()
    DAM.calc(DAD, x, f, u)
    DAM.calcDiff(DAD, x, f, u)
    dxdot_dx = DAD.Fx
    dxdot_dq = DAD.Fx[:,:rmodel.nv]
    dxdot_dv = DAD.Fx[:,rmodel.nv:]
    dxdot_df = DAD.dABA_df
    dxdot_du = DAD.Fu
    dfdot_dq = DAD.dfdt_dx[:,:rmodel.nv]
    dfdot_dv = DAD.dfdt_dx[:,rmodel.nv:]
    dfdot_df = DAD.dfdt_df
    dfdot_du = DAD.dfdt_du
    Lq_dad = DAD.Lx[:rmodel.nv]
    Lv_dad = DAD.Lx[rmodel.nv:]
    Lf_dad = DAD.Lf
    Lu_dad = DAD.Lu
    

    # Compute terminal DAM derivatives
    DAM_t = IAM_t.differential
    DAD_t = DAM_t.createData()
    DAM_t.calc(DAD_t, x, f)
    DAM_t.calcDiff(DAD_t, x, f)
    dxdot_dx_t = DAD_t.Fx
    dxdot_dq_t = DAD_t.Fx[:,:rmodel.nv]
    dxdot_dv_t = DAD_t.Fx[:,rmodel.nv:]
    dxdot_df_t = DAD_t.dABA_df
    dfdot_dq_t = DAD_t.dfdt_dx[:,:rmodel.nv]
    dfdot_dv_t = DAD_t.dfdt_dx[:,rmodel.nv:]
    dfdot_df_t = DAD_t.dfdt_df
    Lq_dad_t = DAD_t.Lx[:rmodel.nv]
    Lv_dad_t = DAD_t.Lx[rmodel.nv:]
    Lf_dad_t = DAD_t.Lf

    # Compute running IAM derivatives with NUMDIFF
        # Dyn
    dynext_dx_ND = numdiff_x_iam_dyn(lambda y_:get_ynext_y(IAM, IAD, y_, u), y, IAM.stateSoft)
    dynext_du_ND = numdiff_u_iam_dyn(lambda u_:get_ynext_y(IAM, IAD, y, u_), u, IAM.stateSoft)
        # Cost
    dcost_dy_ND = numdiff_x_iam_cost(lambda y_:get_iam_cost(IAM, IAD, y_, u), y, IAM.stateSoft)
    dcost_du_ND = numdiff_u_iam_cost(lambda u_:get_iam_cost(IAM, IAD, y, u_), u)
        # Cstr
    dcstr_dy_ND = numdiff_x_iam_cstr(lambda y_:get_cstr(IAM, IAD, y_, u), y, IAM.stateSoft)
    dcstr_du_ND = numdiff_u_iam_cstr(lambda u_:get_cstr(IAM, IAD, y, u_), u)
    
    # Compute terminal IAM_t derivatives with NUMDIFF
    dynext_dy_t_ND = numdiff_x_iam_dyn(lambda y_:get_ynext_y(IAM_t, IAD_t, y_), y, IAM_t.stateSoft)
    dcost_dy_t_ND = numdiff_x_iam_cost(lambda y_:get_iam_cost(IAM_t, IAD_t, y_), y, IAM_t.stateSoft)
        # Cstr
    dcstr_dy_t_ND = numdiff_x_iam_cstr(lambda y_:get_cstr(IAM_t, IAD_t, y_, u), y, IAM_t.stateSoft)
    
    # Compute running DAM derivatives with NUMDIFF
    dxdot_dq_ND = numdiff_q_dam_dyn(lambda q_:get_xdot(DAM, DAD, q_, v, f, u), q, rmodel)
    dxdot_dv_ND = numdiff(lambda v_:get_xdot(DAM, DAD, q, v_, f, u), v)
    dxdot_df_ND = numdiff(lambda f_:get_xdot(DAM, DAD, q, v, f_, u), f)
    dxdot_du_ND = numdiff(lambda u_:get_xdot(DAM, DAD, q, v, f, u_), u)
    dfdot_dq_ND = numdiff_q_dam_dyn(lambda q_:get_fdot(DAM, DAD, q_, v, f, u), q, rmodel)
    dfdot_dv_ND = numdiff(lambda v_:get_fdot(DAM, DAD, q, v_, f, u), v)
    dfdot_df_ND = numdiff(lambda f_:get_fdot(DAM, DAD, q, v, f_, u), f)
    dfdot_du_ND = numdiff(lambda u_:get_fdot(DAM, DAD, q, v, f, u_), u)
    Lq_dad_ND = numdiff_q_dam_cost(lambda q_:get_dam_cost(DAM, DAD, q_, v, f, u), q, rmodel)
    Lv_dad_ND = numdiff_vfu_dam_cost(lambda v_:get_dam_cost(DAM, DAD, q, v_, f, u), v)
    Lf_dad_ND = numdiff_vfu_dam_cost(lambda f_:get_dam_cost(DAM, DAD, q, v, f_, u), f)
    Lu_dad_ND = numdiff_vfu_dam_cost(lambda u_:get_dam_cost(DAM, DAD, q, v, f, u_), u)
    
    # Compute terminal DAM_t derivatives with NUMDIFF
    dxdot_dq_t_ND = numdiff_q_dam_dyn(lambda q_:get_xdot(DAM_t, DAD_t, q_, v, f), q, rmodel)
    dxdot_dv_t_ND = numdiff(lambda v_:get_xdot(DAM_t, DAD_t, q, v_, f), v)
    dxdot_df_t_ND = numdiff(lambda f_:get_xdot(DAM_t, DAD_t, q, v, f_), f)
    dfdot_dq_t_ND = numdiff_q_dam_dyn(lambda q_:get_fdot(DAM_t, DAD_t, q_, v, f), q, rmodel)
    dfdot_dv_t_ND = numdiff(lambda v_:get_fdot(DAM_t, DAD_t, q, v_, f), v)
    dfdot_df_t_ND = numdiff(lambda f_:get_fdot(DAM_t, DAD_t, q, v, f_), f)
    Lq_dad_t_ND = numdiff_q_dam_cost(lambda q_:get_dam_cost(DAM_t, DAD_t, q_, v, f), q, rmodel)
    Lv_dad_t_ND = numdiff_vfu_dam_cost(lambda v_:get_dam_cost(DAM_t, DAD_t, q, v_, f), v)
    Lf_dad_t_ND = numdiff_vfu_dam_cost(lambda f_:get_dam_cost(DAM_t, DAD_t, q, v, f_), f)

    # Check running IAM
    assert(norm(dynext_dx - dynext_dx_ND) <= TOL)
    assert(norm(dynext_du - dynext_du_ND) <= TOL)
    assert(norm(dcost_dy - dcost_dy_ND) <= TOL)
    assert(norm(dcost_du - dcost_du_ND) <= TOL)
    assert(norm(dcstr_dy - dcstr_dy_ND) <= TOL)
    assert(norm(dcstr_du - dcstr_du_ND) <= TOL)
    
    # Check terminal IAM_t
    assert(norm(dynext_dy_t - dynext_dy_t_ND) <= TOL)
    assert(norm(dcost_dy_t - dcost_dy_t_ND) <= TOL)
    assert(norm(dcstr_dy_t - dcstr_dy_t_ND) <= TOL)

    # Check running DAM
    assert(norm(dxdot_dq - dxdot_dq_ND) <= TOL)
    assert(norm(dxdot_dv - dxdot_dv_ND) <= TOL)
    assert(norm(dxdot_df - dxdot_df_ND) <= TOL)
    assert(norm(dxdot_du - dxdot_du_ND) <= TOL)
    assert(norm(dfdot_dq - dfdot_dq_ND) <= TOL)
    assert(norm(dfdot_dv - dfdot_dv_ND) <= TOL)
    assert(norm(dfdot_df - dfdot_df_ND) <= TOL)
    assert(norm(dfdot_du - dfdot_du_ND) <= TOL)
    assert(norm(Lq_dad - Lq_dad_ND) <= TOL)
    assert(norm(Lv_dad - Lv_dad_ND) <= TOL)
    assert(norm(Lf_dad - Lf_dad_ND) <= TOL)
    assert(norm(Lu_dad - Lu_dad_ND) <= TOL)

    # Check terminal DAM_t
    assert(norm(dxdot_dq_t - dxdot_dq_t_ND) <= TOL)
    assert(norm(dxdot_dv_t - dxdot_dv_t_ND) <= TOL)
    assert(norm(dxdot_df_t - dxdot_df_t_ND) <= TOL)
    assert(norm(dfdot_dq_t - dfdot_dq_t_ND) <= TOL)
    assert(norm(dfdot_dv_t - dfdot_dv_t_ND) <= TOL)
    assert(norm(dfdot_df_t - dfdot_df_t_ND) <= TOL)
    assert(norm(Lq_dad_t - Lq_dad_t_ND) <= TOL)
    assert(norm(Lv_dad_t - Lv_dad_t_ND) <= TOL)
    assert(norm(Lf_dad_t - Lf_dad_t_ND) <= TOL)

    print("\n---> ALL TESTS PASSED.\n")
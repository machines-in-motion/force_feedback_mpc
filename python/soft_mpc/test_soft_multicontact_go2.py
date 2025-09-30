'''
Testing the multicontact API for soft contacts on the Go2 + arm
Solving an OCP with 5 visco-elastic contacts 3D on the end-effector and feet
'''
import numpy as np

import crocoddyl
import pinocchio as pin

from soft_multicontact_api import ViscoElasticContact3d_Multiple, ViscoElasticContact3D
from soft_multicontact_api import FrictionConeConstraint, ForceBoxConstraint, ForceConstraintManager
from soft_multicontact_api import ForceCost, ForceCostManager, ForceRateCostManager
from soft_multicontact_api import DAMSoftContactDynamics3D_Go2, IAMSoftContactDynamics3D_Go2

SOLVING_OCP = False
TESTING_API = True

### GO2 + ARM EXAMPLE (adapted from Rooholla standard MPC)
import pinocchio as pin
import crocoddyl
import pinocchio

np.random.seed(10)

urdf_root_path = '/home/skleff/force_feedback_ws/Go2Py/Go2Py/assets/'
urdf_path = '/home/skleff/force_feedback_ws/Go2Py/Go2Py/assets/urdf/go2_with_arm.urdf'
robot = pin.RobotWrapper.BuildFromURDF(
urdf_path, urdf_root_path, pin.JointModelFreeFlyer())

pinRef        = pin.LOCAL_WORLD_ALIGNED
FRICTION_CSTR = True
MU = 0.8     # friction coefficient
ee_frame_names = ['FL_FOOT', 'FR_FOOT', 'HL_FOOT', 'HR_FOOT', 'Link6']
rmodel = robot.model
rdata = robot.data
Kp                     = 1000
Kv                     = 100 
# # set contact frame_names and_indices
lfFootId = rmodel.getFrameId(ee_frame_names[0])
rfFootId = rmodel.getFrameId(ee_frame_names[1])
lhFootId = rmodel.getFrameId(ee_frame_names[2])
rhFootId = rmodel.getFrameId(ee_frame_names[3])
efId = rmodel.getFrameId(ee_frame_names[4])

q0 = np.array([0.0, 0.0, 0.33, 0.0, 0.0, 0.0, 1.0] 
                    +4*[0.0, 0.77832842, -1.56065452] + [0.0, 0.3, -0.3, 0.0, 0.0, 0.0]
                        )
q0[11+2]=0.0
x0 =  np.concatenate([q0, np.zeros(rmodel.nv)])

pinocchio.forwardKinematics(rmodel, rdata, q0)
pinocchio.updateFramePlacements(rmodel, rdata)
lfFootPos0 = rdata.oMf[lfFootId].translation
rfFootPos0 = rdata.oMf[rfFootId].translation
lhFootPos0 = rdata.oMf[lhFootId].translation 
rhFootPos0 = rdata.oMf[rhFootId].translation
efPos0 = rdata.oMf[efId].translation
comRef = (rfFootPos0 + rhFootPos0 + lfFootPos0 + lhFootPos0) / 4
comRef[2] = pinocchio.centerOfMass(rmodel, rdata, q0)[2].item() 
print(f'The desired CoM position is: {comRef}')
supportFeetIds = [lfFootId, rfFootId, lhFootId, rhFootId]
supportFeePos = [lfFootPos0, rfFootPos0, lhFootPos0, rhFootPos0]

state = crocoddyl.StateMultibody(rmodel)
actuation = crocoddyl.ActuationModelFloatingBase(state)
nu = actuation.nu

f0 = np.random.rand(3*5)
u0 = np.zeros(actuation.nu)
y0 = np.concatenate([x0, f0])
q = q0.copy()
v = np.zeros(rmodel.nv)
x = x0.copy()
y = y0.copy()
f = f0.copy()
u = u0.copy()
comDes = []
N_ocp = 20
dt = 0.02
T = N_ocp * dt
radius = 0.0
for t in range(N_ocp+1):
    comDes_t = comRef.copy()
    w = (2 * np.pi) * 0.2 # / T
    comDes_t[0] += radius * np.sin(w * t * dt) 
    comDes_t[1] += radius * (np.cos(w * t * dt) - 1)
    comDes += [comDes_t]
runningModels = []
constraintModels = []
for t in range(N_ocp+1):
    costModel = crocoddyl.CostModelSum(state, nu)

    # Add state/control reg costs
    state_reg_weight, control_reg_weight = 1e-1, 1e-3
    freeFlyerQWeight = [0.]*3 + [50.]*3
    freeFlyerVWeight = [10.]*6
    legsQWeight = [0.01]*(rmodel.nv - 6)
    legsWWeights = [1.]*(rmodel.nv - 6)
    stateWeights = np.array(freeFlyerQWeight + legsQWeight + freeFlyerVWeight + legsWWeights)    
    stateResidual = crocoddyl.ResidualModelState(state, x0, nu)
    stateActivation = crocoddyl.ActivationModelWeightedQuad(stateWeights**2)
    stateReg = crocoddyl.CostModelResidual(state, stateActivation, stateResidual)
    if(t == N_ocp):
        costModel.addCost("stateReg", stateReg, state_reg_weight*dt)
    else:    
        costModel.addCost("stateReg", stateReg, state_reg_weight)
    ctrlResidual = crocoddyl.ResidualModelControl(state, nu)
    ctrlReg = crocoddyl.CostModelResidual(state, ctrlResidual)
    if(t != N_ocp):
        costModel.addCost("ctrlReg", ctrlReg, control_reg_weight)  

    # Add COM task
    com_residual = crocoddyl.ResidualModelCoMPosition(state, comDes[t], nu)
    com_activation = crocoddyl.ActivationModelWeightedQuad(np.array([1., 1., 1.]))
    com_track = crocoddyl.CostModelResidual(state, com_activation, com_residual) # What does it correspond to exactly?
    costModel.addCost("comTrack", com_track, 1e-2)

    # End Effecor Position Task
    ef_residual = crocoddyl.ResidualModelFrameTranslation(state, efId, efPos0, nu)
    ef_activation = crocoddyl.ActivationModelWeightedQuad(np.array([1., 1., 1.]))
    ef_track = crocoddyl.CostModelResidual(state, ef_activation, ef_residual)
    if t == N_ocp:
        costModel.addCost("efTransTrack", ef_track, 1e-1*dt)
    else:
        costModel.addCost("efTransTrack", ef_track, 1e-1)

    # End Effecor Orientation Task
    ef_rotation_residual = crocoddyl.ResidualModelFrameRotation(state, efId, rdata.oMf[efId].rotation, nu)
    ef_rot_activation = crocoddyl.ActivationModelWeightedQuad(np.array([1., 1., 1.]))
    ef_rot_track = crocoddyl.CostModelResidual(state, ef_rot_activation, ef_rotation_residual)
    if t == N_ocp:
        costModel.addCost("efRotTrack", ef_rot_track, 1e-2*dt)
    else:
        costModel.addCost("efRotTrack", ef_rot_track, 1e-2)

    # Soft contact models 3d 
    lf_contact = ViscoElasticContact3D(state, actuation, lfFootId, rdata.oMf[lfFootId].translation, Kp, Kv, pinRef)
    rf_contact = ViscoElasticContact3D(state, actuation, rfFootId, rdata.oMf[rfFootId].translation, Kp, Kv, pinRef)
    lh_contact = ViscoElasticContact3D(state, actuation, lhFootId, rdata.oMf[lhFootId].translation, Kp, Kv, pinRef)
    rh_contact = ViscoElasticContact3D(state, actuation, rhFootId, rdata.oMf[rhFootId].translation, Kp, Kv, pinRef)
    ef_contact = ViscoElasticContact3D(state, actuation, efId, rdata.oMf[efId].translation, Kp, Kv, pinRef)
    # Stack models
    softContactModelsStack = ViscoElasticContact3d_Multiple(state, actuation, [lf_contact, rf_contact, lh_contact, rh_contact, ef_contact])

    # Constraints stack
    # constraintModelManager = None 
    constraintModelManager = crocoddyl.ConstraintModelManager(state, actuation.nu)
    uBoxCstr = crocoddyl.ConstraintModelResidual(state, crocoddyl.ResidualModelControl(state, actuation.nu), -robot.model.effortLimit, robot.model.effortLimit)  
    xlb = np.concatenate([robot.model.lowerPositionLimit, [-np.inf]*robot.model.nv])
    xub = np.concatenate([robot.model.upperPositionLimit, [np.inf]*robot.model.nv])
    xBoxCstr = crocoddyl.ConstraintModelResidual(state, crocoddyl.ResidualModelState(state, actuation.nu), xlb, xub)  
    constraintModelManager.addConstraint("ctrlBox", uBoxCstr)
    constraintModelManager.addConstraint("stateBox", xBoxCstr)
    

    # Custom force cost in DAM
    f_weight = np.array([1., 1., 1.])*1e-3
    if t == N_ocp:
        forceCostManager = ForceCostManager([ ForceCost(state, efId, np.array([-25, 0., 0.]), f_weight*dt, pinRef) ], softContactModelsStack)
    else:
        forceCostManager = ForceCostManager([ ForceCost(state, efId, np.array([-25, 0., 0.]), f_weight, pinRef) ], softContactModelsStack)
    
    # Custom cost on the contact force rate 
    fdot_weights = np.ones(f.shape)*1e-4
    forceRateCostManager = ForceRateCostManager(state, actuation, softContactModelsStack, fdot_weights)

    # Create DAM with soft contact models, force costs + standard cost & constraints
    dam = DAMSoftContactDynamics3D_Go2(state, 
                                       actuation, 
                                       costModel, 
                                       softContactModelsStack, 
                                       constraintModelManager, 
                                       forceCostManager,
                                       forceRateCostManager)

    # Friction cone constraint models
    lb = np.array([0, 0, 0])
    ub = np.array([np.inf, np.inf, np.inf])
    forceConstraintManager = \
    ForceConstraintManager([
                            FrictionConeConstraint(lfFootId, MU),
                            FrictionConeConstraint(rfFootId, MU),
                            FrictionConeConstraint(lhFootId, MU),
                            FrictionConeConstraint(rhFootId, MU),
                            ForceBoxConstraint(efId, ub, lb)], 
                                softContactModelsStack)

    iam = IAMSoftContactDynamics3D_Go2(dam, dt=dt, withCostResidual=True, forceConstraintManager=forceConstraintManager)
    runningModels += [iam]


if(SOLVING_OCP):
    import mim_solvers

    # Create shooting problem
    ocp = crocoddyl.ShootingProblem(y0, runningModels[:-1], runningModels[-1])
    ocp.x0 = y0

    solver = mim_solvers.SolverCSQP(ocp)
    solver.max_qp_iters = 500
    max_iter = 100
    solver.with_callbacks = True
    solver.use_filter_line_search = False
    # solver.mu_constraint = 1e-1
    solver.termination_tolerance = 1e-4
    solver.eps_abs = 1e-6
    solver.eps_rel = 1e-6

    xs = [y0]*(solver.problem.T + 1)
    us = [u0]*solver.problem.T

    # us = solver.problem.quasiStatic([x0]*solver.problem.T) 
    solver.setCallbacks([mim_solvers.CallbackVerbose(), mim_solvers.CallbackLogger()])
    solver.solve(xs, us, max_iter)   

    print(solver.constraint_norm)
    # Extract OCP Solution 
    nq, nv, N = rmodel.nq, rmodel.nv, len(xs) 
    jointPos_sol = []
    jointVel_sol = []
    jointAcc_sol = []
    jointTorques_sol = []
    centroidal_sol = []
    force_sol = []
    xs, us = solver.xs, solver.us
    x = []
    for time_idx in range (N):
        q, v = xs[time_idx][:nq], xs[time_idx][nq:nq+nv]
        f = xs[time_idx][nq+nv:]
        pin.framesForwardKinematics(rmodel, rdata, q)
        pin.computeCentroidalMomentum(rmodel, rdata, q, v)
        centroidal_sol += [
            np.concatenate(
                [pin.centerOfMass(rmodel, rdata, q, v), np.array(rdata.hg.linear), np.array(rdata.hg.angular)]
                )
                ]
        jointPos_sol += [q]
        jointVel_sol += [v]
        force_sol    += [f]
        x += [xs[time_idx]]
        if time_idx < N-1:
            jointAcc_sol +=  [solver.problem.runningDatas[time_idx].xnext[nq::]] 
            jointTorques_sol += [us[time_idx]]

    sol = {'x':x, 'centroidal':centroidal_sol, 'jointPos':jointPos_sol, 
                        'jointVel':jointVel_sol, 'jointAcc':jointAcc_sol, 'force':force_sol,
                        'jointTorques':jointTorques_sol}       

    # Extract contact forces by hand
    sol['FL_FOOT_contact'] = [force_sol[i][0:3] for i in range(N)]     
    sol['FR_FOOT_contact'] = [force_sol[i][3:6] for i in range(N)]     
    sol['HL_FOOT_contact'] = [force_sol[i][6:9] for i in range(N)]     
    sol['HR_FOOT_contact'] = [force_sol[i][9:12] for i in range(N)]     
    sol['Link6'] = [force_sol[i][12:15] for i in range(N)]     

    # Plotting 
    import matplotlib.pyplot as plt
    constrained_sol = sol
    time_lin = np.linspace(0, T, solver.problem.T+1)
    fig, axs = plt.subplots(4, 3, constrained_layout=True)
    for i, frame_idx in enumerate(supportFeetIds):
        ct_frame_name = rmodel.frames[frame_idx].name + "_contact"
        forces = np.array(constrained_sol[ct_frame_name])
        axs[i, 0].plot(time_lin, forces[:, 0], label="Fx", marker='.')
        axs[i, 1].plot(time_lin, forces[:, 1], label="Fy", marker='.')
        axs[i, 2].plot(time_lin, forces[:, 2], label="Fz", marker='.')
        # Add friction cone constraints 
        Fz_lb = (1./MU)*np.sqrt(forces[:, 0]**2 + forces[:, 1]**2)
        # Fz_ub = np.zeros(time_lin.shape)
        # axs[i, 2].plot(time_lin, Fz_ub, 'k-.', label='ub')
        axs[i, 2].plot(time_lin, Fz_lb, 'k-.', label='lb')
        axs[i, 0].grid()
        axs[i, 1].grid()
        axs[i, 2].grid()
        axs[i, 0].set_ylabel(ct_frame_name)
    axs[0, 0].legend()
    axs[0, 1].legend()
    axs[0, 2].legend()

    axs[3, 0].set_xlabel(r"$F_x$")
    axs[3, 1].set_xlabel(r"$F_y$")
    axs[3, 2].set_xlabel(r"$F_z$")
    fig.suptitle('Force', fontsize=16)

    ee_force = np.array(constrained_sol['Link6'])
    fig, axs = plt.subplots(3, 1, constrained_layout=True)
    axs[0].plot(time_lin, ee_force[:, 0], label="End-effector force", marker='.')
    axs[0].set_ylabel("F_x")
    axs[1].plot(time_lin, ee_force[:, 1], marker='.')
    axs[1].set_ylabel("F_y")
    axs[2].plot(time_lin, ee_force[:, 2], marker='.')
    axs[2].set_ylabel("F_z")
    axs[2].set_xlabel("time")
    fig.legend()


    comDes = np.array(comDes)
    centroidal_sol = np.array(constrained_sol['centroidal'])
    plt.figure()
    plt.plot(comDes[:, 0], comDes[:, 1], "--", label="reference")
    plt.plot(centroidal_sol[:, 0], centroidal_sol[:, 1], label="solution")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("COM trajectory")
    plt.show()


if(TESTING_API):

    from numpy.random import rand
    from numpy.linalg import norm 
    TOL = 1e-2
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
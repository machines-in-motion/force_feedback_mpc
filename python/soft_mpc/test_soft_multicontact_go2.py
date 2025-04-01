'''
Testing the multicontact API for soft contacts on the Go2 + arm
Solving an OCP with 5 visco-elastic contacts 3D on the end-effector and feet
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

### GO2 + ARM EXAMPLE (adapted from Rooholla standard MPC)
import pinocchio as pin
import crocoddyl
import pinocchio
import numpy as np
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

f0 = np.zeros(3*5)
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
    freeFlyerQWeight = [0.]*3 + [500.]*3
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
    constraintModelManager = None 
    # crocoddyl.ConstraintModelManager(state, actuation.nu)

    # Custom force cost in DAM
    if t == N_ocp:
        forceCostManager = ForceCostManager([ ForceCost(state, efId, np.array([-25, 0., 0.]), 1e-3*dt, pinRef) ], softContactModelsStack)
    else:
        # forceCostManager = ForceCostManager([ ForceCost(state, efId, np.array([-25, 0., 0.]), 1e-3, pinRef, fdotReg=0.01) ], softContactModelsStack)
        forceCostManager = ForceCostManager([ ForceCost(state, efId, np.array([-25, 0., 0.]), 1e-3, pinRef) ], softContactModelsStack)

    # Create DAM with soft contact models, force costs + standard cost & constraints
    dam = DAMSoftContactDynamics3D_Go2(state, actuation, costModel, softContactModelsStack, constraintModelManager, forceCostManager)

    # Friction cone constraint models
    lb = np.array([0, 0, 0])
    ub = np.array([np.inf, np.inf, np.inf])
    forceConstraintManager = \
    ForceConstraintManager([
                            FrictionConeConstraint(lfFootId, MU),
                            FrictionConeConstraint(rfFootId, MU),
                            FrictionConeConstraint(lhFootId, MU),
                            FrictionConeConstraint(rhFootId, MU)],
                            # ForceBoxConstraint(efId, ub, lb)], 
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
    np.random.seed(10)
    TOL = 1e-3
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

    def numdiff_q_manifold(f, q0, model, h=1e-6):
        '''
        Numdiff in tangent space for Differential Action models
        '''
        f0 = f(q0).copy()
        q = q0.copy()
        # Create perturbation vector
        v = np.zeros(model.nv)
        Fq = []
        for iv in range(model.nv):
            # Apply perturbation to the iv-th component
            v[iv] = h
            q = pin.integrate(model, q0, v)
            # Compute finite difference
            Fq.append((f(q) - f0) / h)
            # Reset perturbation
            v[iv] = 0.0
        return np.array(Fq).T


    def numdiff_x_iam_dyn(f, x0, state, h=1e-6):
        '''
        Numdiff in tangent space for Integrated Action models
        partial of dynamics (xnext) w.r.t. state x
        '''
        f0 = f(x0).copy()
        x = x0.copy()
        # Create perturbation vector
        dx = np.zeros(state.ndx)
        Fx = []
        for idx in range(state.ndx):
            # Apply perturbation to the iv-th component
            dx[idx] = h
            x = state.integrate(x0, dx)
            # Compute finite difference
            dx_h = state.diff(f0, f(x))
            Fx.append(dx_h / h)
            # Reset perturbation
            dx[idx] = 0.0
        return np.array(Fx).T

    def numdiff_u_iam_dyn(f, u0, state, h=1e-6):
        '''
        Numdiff in tangent space for Integrated Action models
        partial of dynamics (xnext) w.r.t. control u
        '''
        f0 = f(u0).copy()
        u = u0.copy()
        # Create perturbation vector
        du = np.zeros_like(u0) 
        Fu = []
        for idu in range(len(du)):
            # Apply perturbation to the iv-th component
            u[idu] += h
            # Compute finite difference
            Fu.append(state.diff(f0, f(u)) / h)
            # Reset perturbation
            u[idu] -= h
        return np.array(Fu).T


    def numdiff_x_iam_cost(f, x0, state, h=1e-6):
        '''
        Numdiff in tangent space for Integrated Action models
        partial of cost (scalar) w.r.t. control u
        '''
        f0 = f(x0)
        x = x0.copy()
        # Create perturbation vector
        dx = np.zeros(state.ndx)
        Fx = []
        for idx in range(state.ndx):
            # Apply perturbation to the iv-th component
            dx[idx] = h
            x = state.integrate(x0, dx)
            # Compute finite difference
            Fx.append( ( f(x) - f0 )/ h)
            # Reset perturbation
            dx[idx] = 0.0
        return np.array(Fx).T

    def numdiff_u_iam_cost(f, x0, state, h=1e-6):
        '''
        Numdiff in tangent space for Integrated Action models
        partial of cost (scalar) w.r.t. control u
        '''
        f0 = f(u0)
        u = u0.copy()
        # Create perturbation vector
        du = np.zeros_like(u0) 
        Fu = []
        for idu in range(len(du)):
            # Apply perturbation to the iv-th component
            u[idu] += h
            # Compute finite difference
            Fu.append( ( f(u) - f0 )/ h)
            # Reset perturbation
            u[idu] -= h
        return np.array(Fu).T

    # Compute differential action model derivatives
    dam = iam.differential
    dad = dam.createData()
    # dad = iad.differential
    dam.calc(dad, x, f, u)
    dam.calcDiff(dad, x, f, u)
    dxdot_dx = dad.Fx
    dxdot_dq = dad.Fx[:,:rmodel.nv]
    dxdot_dv = dad.Fx[:,rmodel.nv:]
    dxdot_df = dad.dABA_df
    dxdot_du = dad.Fu
    dfdot_dq = dad.dfdt_dx[:,:rmodel.nv]
    dfdot_dv = dad.dfdt_dx[:,rmodel.nv:]
    dfdot_df = dad.dfdt_df
    dfdot_du = dad.dfdt_du

    # Compute integral action model derivatives
    iam = runningModels[10]
    iad = iam.createData()
    iam.calc(iad, y, u)
    iam.calcDiff(iad, y, u)
    dcost_dy = iad.Lx
    dcost_du = iad.Lu
    dynext_dx = iad.Fx 
    dynext_du = iad.Fu 
    # Test terminal model
    iad_t = iam.createData()
    iam.calc(iad_t, y)
    iam.calcDiff(iad_t, y)
    dcost_dy_t = iad_t.Lx
    dynext_dx_t = iad_t.Fx 

    # Finite differences
    def get_xdot(dam, dad, q, v, f, u):
        x = np.concatenate([q,v])
        dam.calc(dad, x, f, u)
        return dad.xout 

    def get_fdot(dam, dad, q, v, f, u):
        x = np.concatenate([q,v])
        dam.calc(dad, x, f, u)
        return dad.fout

    def get_ynext(iam, iad, q, v, f, u):
        y = np.concatenate([q,v,f])
        iam.calc(iad, y, u)
        return iad.xnext 

    def get_ynext_y(iam, iad, y, u=None):
        if(u is not None):
            iam.calc(iad, y, u)
        else:
            iam.calc(iad, y)
        return iad.xnext 
    
    def get_iam_cost(iam, iad, y, u=None):
        if(u is not None):
            iam.calc(iad, y, u)
        else:
            iam.calc(iad, y)
        return iad.cost

    dxdot_dq_ND = numdiff_q_manifold(lambda q_:get_xdot(dam, dad, q_, v, f, u), q, rmodel)
    dxdot_dv_ND = numdiff(lambda v_:get_xdot(dam, dad, q, v_, f, u), v)
    dxdot_df_ND = numdiff(lambda f_:get_xdot(dam, dad, q, v, f_, u), f)
    dxdot_du_ND = numdiff(lambda u_:get_xdot(dam, dad, q, v, f, u_), u)
    dfdot_dq_ND = numdiff_q_manifold(lambda q_:get_fdot(dam, dad, q_, v, f, u), q, rmodel)
    dfdot_dv_ND = numdiff(lambda v_:get_fdot(dam, dad, q, v_, f, u), v)
    dfdot_df_ND = numdiff(lambda f_:get_fdot(dam, dad, q, v, f_, u), f)
    dfdot_du_ND = numdiff(lambda u_:get_fdot(dam, dad, q, v, f, u_), u)

    dynext_dx_ND = numdiff_x_iam_dyn(lambda y_:get_ynext_y(iam, iad, y_, u), y, iam.stateSoft)
    dynext_du_ND = numdiff_u_iam_dyn(lambda u_:get_ynext_y(iam, iad, y, u_), u, iam.stateSoft)
    dcost_dy_ND = numdiff_x_iam_cost(lambda y_:get_iam_cost(iam, iad, y_, u), y, iam.stateSoft)
    dcost_du_ND = numdiff_u_iam_cost(lambda u_:get_iam_cost(iam, iad, y, u_), u, iam.stateSoft)
    
    dynext_dx_terminal_ND = numdiff_x_iam_dyn(lambda y_:get_ynext_y(iam, iad, y_), y, iam.stateSoft)
    dcost_dy_terminal_ND = numdiff_x_iam_cost(lambda y_:get_iam_cost(iam, iad, y_), y, iam.stateSoft)

    assert(norm(dxdot_dq - dxdot_dq_ND) <= TOL)
    assert(norm(dxdot_dv - dxdot_dv_ND) <= TOL)
    assert(norm(dxdot_df - dxdot_df_ND) <= TOL)
    assert(norm(dxdot_du - dxdot_du_ND) <= TOL)
    assert(norm(dfdot_dq - dfdot_dq_ND) <= TOL)
    assert(norm(dfdot_dv - dfdot_dv_ND) <= TOL)
    assert(norm(dfdot_df - dfdot_df_ND) <= TOL)
    assert(norm(dfdot_du - dfdot_du_ND) <= TOL)
    assert(norm(dynext_dx - dynext_dx_ND) <= TOL)
    assert(norm(dynext_du - dynext_du_ND) <= TOL)
    assert(norm(dcost_dy - dcost_dy_ND) <= TOL)
    assert(norm(dynext_dx_t - dynext_dx_terminal_ND) <= TOL)
    assert(norm(dcost_dy_t - dcost_dy_terminal_ND) <= TOL)
    print("\n---> ALL TESTS PASSED.\n")
import numpy as np
np.set_printoptions(precision=6, linewidth=180, suppress=True)
np.random.seed(1)

import example_robot_data 
import pinocchio as pin

# Load robot and setup params
robot = example_robot_data.load('talos_arm')
nq = robot.model.nq; nv = robot.model.nv; nu = nq; nx = nq+nv
# q0 = np.random.rand(nq) 
q0 = np.array([.5,-1,1.5,0,0,-0.5,0])
v0 = np.random.rand(nv)
x0 = np.concatenate([q0, v0])
robot.framesForwardKinematics(q0)
tau = np.random.rand(nq)
print("x0  = "+str(x0))
print("tau = "+str(tau))


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

# Forward dynamics in LOCAL or WORLD, inverting KKT : ground truth in LOCAL and LWA
def fdyn_local(model, data, frameId, x, tau, Kp, Kv, oP, mask):
    '''
    compute joint acc using LOCAL Force
    '''
    q = x[:nq]
    v = x[nq:]
    pin.computeAllTerms(model, data, q, v)
    pin.forwardKinematics(model, data, q, v, np.zeros(nq))
    pin.updateFramePlacements(model, data)
    # Compute visco-elastic contact force 
    oRf = data.oMf[frameId].rotation
    lJ = pin.getFrameJacobian(model, data, frameId, pin.LOCAL)
    pdot = lJ[:3] @ v
    lv = pin.getFrameVelocity(model, data, frameId, pin.LOCAL).linear
    assert(np.linalg.norm(lv - pdot) < 1e-4)
    force = (-Kp * oRf.T @ ( data.oMf[frameId].translation - oP ) - Kv*lv)[mask]
    
    force2 = force_local(model, data, frameId, x, Kp, Kv, oP, mask)
    assert(np.linalg.norm(force2 - force) < 1e-4)
    
    fext = [pin.Force.Zero() for _ in range(model.njoints)]
    fLOCAL = np.zeros(3) ; fLOCAL[mask] = force
    lwrench = pin.Force(fLOCAL, np.zeros(3))
    fext[model.frames[frameId].parent] = model.frames[frameId].placement.act(lwrench)
    aq = pin.aba(model, data, q, v, tau, fext)
    # print("acc = \n")
    # print(aq)
    return aq

# Forward dynamics in LOCAL or WORLD, inverting KKT : ground truth in LOCAL and LWA
def fdyn_world(model, data, frameId, x, tau, Kp, Kv, oP, mask):
    '''
    compute joint acc using WORLD force
    '''
    q = x[:nq]
    v = x[nq:]
    pin.computeAllTerms(model, data, q, v)
    pin.forwardKinematics(model, data, q, v, np.zeros(nq))
    pin.updateFramePlacements(model, data)
    # Compute visco-elastic contact force 
    oRf = data.oMf[frameId].rotation
    oJ = pin.getFrameJacobian(model, data, frameId, pin.LOCAL_WORLD_ALIGNED)
    pdot = oJ[:3] @ v
    ov = pin.getFrameVelocity(model, data, frameId, pin.LOCAL_WORLD_ALIGNED).linear
    assert(np.linalg.norm(ov - pdot) < 1e-4)
    force = (-Kp * ( data.oMf[frameId].translation - oP ) - Kv*ov)[mask]

    force2 = force_world(model, data, frameId, x, Kp, Kv, oP, mask)
    assert(np.linalg.norm(force2 - force) < 1e-4)

    fext = [pin.Force.Zero() for _ in range(model.njoints)]
    fWORLD = np.zeros(3) ; fWORLD[mask] = force
    owrench = pin.Force(oRf.T @ fWORLD, np.zeros(3))
    fext[model.frames[frameId].parent] = model.frames[frameId].placement.act(owrench)
    aq = pin.aba(model, data, q, v, tau, fext)
    # print("acc = \n")
    # print(aq)
    return aq

# Forward dynamics in LOCAL or WORLD, inverting KKT : ground truth in LOCAL and LWA
def force_local(model, data, frameId, x, Kp, Kv, oP, mask):
    '''
    compute the contact force in LOCAL
    '''
    q = x[:nq]
    v = x[nq:]
    pin.computeAllTerms(model, data, q, v)
    pin.forwardKinematics(model, data, q, v, np.zeros(nq))
    pin.updateFramePlacements(model, data)
    # Compute visco-elastic contact force 
    oRf = data.oMf[frameId].rotation
    lJ = pin.getFrameJacobian(model, data, frameId, pin.LOCAL)
    pdot = lJ[:3] @ v
    lv = pin.getFrameVelocity(model, data, frameId, pin.LOCAL).linear
    assert(np.linalg.norm(lv - pdot) < 1e-4)
    ov = pin.getFrameVelocity(model, data, frameId, pin.LOCAL_WORLD_ALIGNED).linear
    assert(np.linalg.norm(ov - oRf @ lv) < 1e-4)
    force = (-Kp * oRf.T @ (data.oMf[frameId].translation - oP) - Kv*lv)[mask]
    return force

# Forward dynamics in LOCAL or WORLD, inverting KKT : ground truth in LOCAL and LWA
def force_world(model, data, frameId, x, Kp, Kv, oP, mask):
    '''
    compute the contact force in WORLD
    '''
    q = x[:nq]
    v = x[nq:]
    pin.computeAllTerms(model, data, q, v)
    pin.forwardKinematics(model, data, q, v, np.zeros(nq))
    pin.updateFramePlacements(model, data)
    # Compute visco-elastic contact force 
    oJ = pin.getFrameJacobian(model, data, frameId, pin.LOCAL_WORLD_ALIGNED)
    pdot = oJ[:3] @ v
    ov = pin.getFrameVelocity(model, data, frameId, pin.LOCAL_WORLD_ALIGNED).linear
    assert(np.linalg.norm(ov - pdot) < 1e-4)
    force = (-Kp*(data.oMf[frameId].translation - oP) - Kv*ov)[mask]
    return force




# TESTS 
contactFrameName = "gripper_left_fingertip_1_link"
model = robot.model
data = robot.model.createData()
frameId = model.getFrameId(contactFrameName)
parentId = model.frames[frameId].parent
Kp = 100 #need to increase tolerances of assert if high gains
Kv = 2*np.sqrt(Kp)
pin.computeAllTerms(model, data, q0, v0)
pin.forwardKinematics(model, data, q0, v0, np.zeros(nq))
pin.updateFramePlacements(model, data)
oRf = data.oMf[frameId].rotation
# anchor point world
oPc = np.random.rand(3) 
# lPc = oRf.T @ oPc
# mask = 0
mask = [2]
# mask = 2

# Compute visco-elastic contact force 
lJ = pin.getFrameJacobian(model, data, frameId, pin.LOCAL)
oJ = pin.getFrameJacobian(model, data, frameId, pin.LOCAL_WORLD_ALIGNED)
lv = pin.getFrameVelocity(model, data, frameId, pin.LOCAL).linear
ov = pin.getFrameVelocity(model, data, frameId, pin.LOCAL_WORLD_ALIGNED).linear
assert(np.linalg.norm(oJ[:3] - oRf @ lJ[:3]) < 1e-4)
assert(np.linalg.norm(lv - lJ[:3] @ v0) < 1e-4)
assert(np.linalg.norm(ov - oJ[:3] @ v0) < 1e-4)
assert(np.linalg.norm(ov - oRf @ lv) < 1e-4)

# LOCAL force and joint acc
lf3d = -Kp* oRf.T @ ( data.oMf[frameId].translation - oPc) - Kv*lv
lf = lf3d[mask]
fLOCAL = np.zeros(3) ; fLOCAL[mask] = lf
lfext = [pin.Force.Zero() for _ in range(model.njoints)]
lwrench = pin.Force(fLOCAL, np.zeros(3))
lfext[parentId] = model.frames[frameId].placement.act(lwrench)
laq = pin.aba(model, data, q0, v0, tau, lfext)
assert(np.linalg.norm(laq - fdyn_local(model, data, frameId, x0, tau, Kp, Kv, oPc, mask)) <1e-4)
# WORLD force and joint acc
of3d = -Kp* ( data.oMf[frameId].translation - oPc) - Kv*ov
of = of3d[mask]
assert(np.linalg.norm(of3d - oRf @ lf3d) < 1e-4)
fWORLD = np.zeros(3) ; fWORLD[mask] = of
ofext = [pin.Force.Zero() for _ in range(model.njoints)]
owrench = pin.Force(fWORLD, np.zeros(3))
lwaXf = pin.SE3.Identity() ; lwaXf.rotation = oRf ; lwaXf.translation = np.zeros(3)
ofext[parentId] = model.frames[frameId].placement.act(lwaXf.actInv(owrench))
oaq = pin.aba(model, data, q0, v0, tau, ofext)
assert(np.linalg.norm(oaq - fdyn_world(model, data, frameId, x0, tau, Kp, Kv, oPc, mask)) <1e-4)
# assert(np.linalg.norm(oaq - laq) < 1e-4) # no longer true in 1D !!!
print("joint acc (LOCAL) : \n", laq)
print("joint acc (WORLD) : \n", oaq)
# check force at joint level are the same
jfext_lin = lfext[parentId].linear
jfext_ang = lfext[parentId].angular
ofext_lin = ofext[parentId].linear
ofext_ang = ofext[parentId].angular
# assert(np.linalg.norm(jfext_lin - ofext_lin) < 1e-4)  # no longer true in 1D !!!
# assert(np.linalg.norm(jfext_ang - ofext_ang) < 1e-4)  # no longer true in 1D !!!
# Check derivatives of the force against numdiff
    # ND
dlf_dx_ND = numdiff(lambda x_:force_local(model, data, frameId, x_, Kp, Kv, oPc, mask), x0)
dof_dx_ND = numdiff(lambda x_:force_world(model, data, frameId, x_, Kp, Kv, oPc, mask), x0)
    # Analytic
        # local 
dlf3d_dx = np.zeros((3,nx))
dlf_dx = np.zeros((1,nx))
lv_partial_dq, lv_partial_dv = pin.getFrameVelocityDerivatives(model, data, frameId, pin.LOCAL) 
dlf3d_dx[:,:nq] = -Kp * (lJ[:3] + pin.skew(oRf.T @ (data.oMf[frameId].translation - oPc)) @ lJ[3:]) - Kv*lv_partial_dq[:3]
dlf3d_dx[:,nq:] = -Kv*lv_partial_dv[:3]
dlf_dx = dlf3d_dx[mask,:]
assert(np.linalg.norm(dlf_dx - dlf_dx_ND) < 1e-3)
#         # world ### FAILS because of frameVelDerivatives in LWA : need to add some skew term to make it work (because dq depends on dt on lie algebra)
# no canonical definition of the derivative of LWA . Choose Justin's or finite diff's w.r.t. dq
# dof_dx = np.zeros((3,nx))
# ov_partial_dq, ov_partial_dv = pin.getFrameVelocityDerivatives(model, data, frameId, pin.LOCAL_WORLD_ALIGNED) 
# dof_dx[:,:nq] = -Kp * oJ[:3] - Kv*ov_partial_dq[:3]
# dof_dx[:,nq:] = -Kv*ov_partial_dv[:3]
# assert(np.linalg.norm(dof_dx - dof_dx_ND) < 1e-4)       
        # world 2 (rotate local)
dof3d_dx = np.zeros((3,nx))
dof_dx = np.zeros((1,nx))
dof3d_dx[:,:nq] = oRf @ dlf3d_dx[:,:nq] - pin.skew(oRf @ lf3d) @ oJ[3:]
dof3d_dx[:,nq:] = oRf @ dlf3d_dx[:,nq:]
dof_dx = dof3d_dx[mask,:]
assert(np.linalg.norm(dof_dx - dof_dx_ND) < 1e-3)
# Compute the derivative of joint acceleration
    # LOCAL 
laba_dq, laba_dv, laba_dtau = pin.computeABADerivatives(model, data, q0, v0, tau, lfext)
# print('LOCAL aba derivatives q : \n', laba_dq)
# laba_dx = np.hstack([laba_dq.copy(), laba_dv.copy()])
laba_dq_copy = laba_dq.copy()
laba_dv_copy = laba_dv.copy()
# print('LOCAL aba derivatives x : \n', laba_dx)
ldaq_dx = np.zeros((nq, nx))
ldaq_du = np.zeros((nq, nq))
dlf_dx_full = np.zeros((3,2*nv))
dlf_dx_full[mask,:] = dlf_dx
# ldaq_dx[:,:nq] = laba_dq_copy + data.Minv @ lJ[mask].T @ dlf_dx[:,:nq]
# ldaq_dx[:,nq:] = laba_dv_copy + data.Minv @ lJ[mask].T @ dlf_dx[:,nq:]
ldaq_dx[:,:nq] = laba_dq_copy + data.Minv @ lJ[:3].T[:,mask] @ dlf_dx[:,:nq]
ldaq_dx[:,nq:] = laba_dv_copy + data.Minv @ lJ[:3].T[:,mask] @ dlf_dx[:,nq:]
ldaq_du = laba_dtau
    # compare numdiff
ldaq_dx_ND = numdiff(lambda x_:fdyn_local(model, data, frameId, x_, tau, Kp, Kv, oPc, mask), x0)
assert(np.linalg.norm(ldaq_dx - ldaq_dx_ND) < 2e-3)
    # WORLD (not same as LOCAL in 1D !!!!)
oaba_dq, oaba_dv, oaba_dtau = pin.computeABADerivatives(model, data, q0, v0, tau, ofext)
# print('WORLD aba derivatives q : \n', oaba_dq)
oaba_dx = np.hstack([oaba_dq, oaba_dv])
oaba_dq_copy = oaba_dq.copy()
oaba_dv_copy = oaba_dv.copy()
# print('WORLD aba derivatives x : \n', oaba_dx)
# check that numdiff aba = aba derivatives in WORLD
def aba_world(model, data, state, torque, world_fext):
    pos = state[:nq]
    vel = state[nq:]
    return pin.aba(model, data, pos, vel, torque, world_fext)
oaba_dx_ND = numdiff(lambda x_:aba_world(model, data, x_, tau, ofext), x0)
assert(np.linalg.norm(oaba_dx_ND - oaba_dx) < 1e-3) # OK 
# Formula for d (aq) / dx from ABA derivatives and force derivatives
odaq_dx = np.zeros((nq, nx))
odaq_du = np.zeros((nq, nu))
# d(aq)/dx = d(ABA)/dq + Minv*Jt*df/dq 
dof_dx_full = np.zeros((3,2*nv))
dof_dx_full[mask,:] = dof_dx
# odaq_dx[:,:nq] = oaba_dq_copy + data.Minv @ lJ[:3].T @ (oRf.T @ dof_dx_full[:,:nq] + pin.skew(oRf.T @ fWORLD) @ lJ[3:]) 
# odaq_dx[:,nq:] = oaba_dv_copy + data.Minv @ lJ[:3].T @ (oRf.T @ dof_dx_full[:,nq:])
odaq_dx[:,:nq] = oaba_dq_copy + \
    data.Minv @ lJ[:3].T @ (oRf.T[:,mask] @ dof_dx[:,:nq] + pin.skew(oRf.T @ fWORLD) @ lJ[3:]) 
odaq_dx[:,nq:] = oaba_dv_copy + \
    data.Minv @ lJ[:3].T @ (oRf.T[:,mask] @ dof_dx[:,nq:])
odaq_du = oaba_dtau
    # compare numdiff
odaq_dx_ND = numdiff(lambda x_:fdyn_world(model, data, frameId, x_, tau, Kp, Kv, oPc, mask), x0)
assert(np.linalg.norm(odaq_dx - odaq_dx_ND) < 1e-2) 


# Check implemented class
from soft_mpc.soft_models_1D import DAMSoftContactDynamics1D
# from soft_mpc.soft_models_1D_augmented import DAMSoftContactDynamics1D as DAM1DAugmented
import crocoddyl
# State, actuation, cost models
state = crocoddyl.StateMultibody(model)
actuation = crocoddyl.ActuationModelFull(state)
runningCostModel = crocoddyl.CostModelSum(state)

# Custom DAM to check 
ref = pin.LOCAL # pin.LOCAL_WORLD_ALIGNED
dam = DAMSoftContactDynamics1D(state, actuation, runningCostModel, frameId, '1Dz', Kp, Kv, oPc, pinRefFrame=ref)
# dam2 = DAM1DAugmented(state, actuation, runningCostModel, frameId, Kp, Kv, oPc, pinRefFrame=ref)
dad = dam.createData()
# Numdiff version 
RTOL            = 1e-2 
ATOL            = 1e-1 
dam_nd = crocoddyl.DifferentialActionModelNumDiff(dam, True)
dam_nd.disturbance = 1e-6
dad_nd = dam_nd.createData()
# TEST CALC
dam.calc(dad, x0, tau)
dam_nd.calc(dad_nd, x0, tau)
# dad2 = dam2.createData()
# dam2.calc(dad2, x0, lf, tau)
print("custom aq = \n", dad.xout)
print("ND xout = \n", dad_nd.xout)
if(ref == pin.LOCAL):
    assert(np.linalg.norm(dad.xout - laq) < 1e-3)
else: 
    assert(np.linalg.norm(dad.xout - oaq) < 1e-3)
# TEST CALCDIFF 
dam.calcDiff(dad, x0, tau)
dam_nd.calcDiff(dad_nd, x0, tau)
assert(np.linalg.norm(dad.Fx - dad_nd.Fx )< 1e-2)
if(ref == pin.LOCAL):
    assert(np.linalg.norm(dad.Fx - ldaq_dx )< 1e-3)
else:
    assert(np.linalg.norm(dad.Fx - odaq_dx )< 1e-2)


# Further checks using DAM free + setup shooting problem + solver
# the soft contact DAM should coincide with DAM free when Kp, Kv = 0

# Control regularization cost
uref = np.random.rand(nq) 
uResidual = crocoddyl.ResidualModelControl(state, uref)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)
  # State regularization cost
xResidual = crocoddyl.ResidualModelState(state, x0)
xRegCost = crocoddyl.CostModelResidual(state, xResidual)
  # Frame translation cost
endeff_frame_id = model.getFrameId("gripper_left_fingertip_1_link") ; endeff_translation = oPc 
frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(state, endeff_frame_id, endeff_translation)
frameTranslationCost = crocoddyl.CostModelResidual(state, frameTranslationResidual)
  # Cost model 
runningCostModel.addCost("stateReg", xRegCost, 1e-2)
runningCostModel.addCost("ctrlReg", uRegCost, 1e-4)
# runningCostModel.addCost("translation", frameTranslationCost, 1)
# terminalCostModel.addCost("stateReg", xRegCost, 1.)

# free model
damf = crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, runningCostModel)
dadf = damf.createData()
# soft contact (abstract)
# damc1 = DAMSoftContactDynamics1(state, actuation, runningCostModel, frameId, 0, 0., oPc=np.zeros(3), pinRefFrame=pin.LOCAL)
# dadc1 = damc1.createData()
# soft contact (free)
damc2 = DAMSoftContactDynamics1D(state, actuation, runningCostModel, frameId, '1Dz', Kp=0, Kv=0., oPc=np.zeros(3), pinRefFrame=pin.LOCAL)
dadc2 = damc2.createData()
# Check DAM
damf.calc(dadf, x0, tau)
# damc1.calc(dadc1, x0, tau)
damc2.calc(dadc2, x0, tau)
# assert(np.linalg.norm(dadf.xout - dadc1.xout)<1e-4)
assert(np.linalg.norm(dadf.xout - dadc2.xout)<1e-4)
# assert(np.linalg.norm(dadf.cost - dadc1.cost)<1e-4)
assert(np.linalg.norm(dadf.cost - dadc2.cost)<1e-4)
# prin
damf.calcDiff(dadf, x0, tau)
# damc1.calcDiff(dadc1, x0, tau)
damc2.calcDiff(dadc2, x0, tau)
# assert(np.linalg.norm(dadf.Fx - dadc1.Fx)<1e-4)
assert(np.linalg.norm(dadf.Fx - dadc2.Fx)<1e-4)
# Check IAM
dt = 0.001
iamf = crocoddyl.IntegratedActionModelEuler(damf, dt)
# iamc1 = crocoddyl.IntegratedActionModelEuler(damc1, dt)
iamc2 = crocoddyl.IntegratedActionModelEuler(damc2, dt)
iadf = iamf.createData()
# iadc1 = iamc1.createData()
iadc2 = iamc2.createData()
iamf.calc(iadf, x0, tau)
# iamc1.calc(iadc1, x0, tau)
iamc2.calc(iadc2, x0, tau)
# assert(np.linalg.norm(iadf.xnext - iadc1.xnext)<1e-4)
assert(np.linalg.norm(iadf.xnext - iadc2.xnext)<1e-4)
iamf.calcDiff(iadf, x0, tau)
# iamc1.calcDiff(iadc1, x0, tau)
iamc2.calcDiff(iadc2, x0, tau)
# assert(np.linalg.norm(iadf.Fx - iadc1.Fx)<1e-4)
# assert(np.linalg.norm(iadf.Fu - iadc1.Fu)<1e-4)
# assert(np.linalg.norm(iadf.Lx - iadc1.Lx)<1e-4)
# assert(np.linalg.norm(iadf.Lu - iadc1.Lu)<1e-4)
assert(np.linalg.norm(iadf.Fx - iadc2.Fx)<1e-4)
assert(np.linalg.norm(iadf.Fu - iadc2.Fu)<1e-4)
assert(np.linalg.norm(iadf.Lx - iadc2.Lx)<1e-4)
assert(np.linalg.norm(iadf.Lu - iadc2.Lu)<1e-4)

tamf = crocoddyl.IntegratedActionModelEuler(damf, 0.)
# tamc1 = crocoddyl.IntegratedActionModelEuler(damc1, 0.)
tamc2 = crocoddyl.IntegratedActionModelEuler(damc2, 0.)


N = 100
pbf = crocoddyl.ShootingProblem(x0, [iamf]*N, tamf)
# pbc1 = crocoddyl.ShootingProblem(x0, [iamc1]*N, tamc1)
pbc2 = crocoddyl.ShootingProblem(x0, [iamc2]*N, tamc2)

# Compare calc and calcDiff
# Warm start : initial state + gravity compensation
lfext0 = [pin.Force.Zero() for _ in range(model.njoints)]
from core_mpc_utils import pin_utils
xs_init = [x0 for i in range(N+1)]
us_init = [pin_utils.get_tau(q0, v0, np.zeros(nq), lfext0, model, np.zeros(nq)) for i in range(N)] #ddp.problem.quasiStatic(xs_init[:-1])
# pbf.calc(xs_init, us_init)


ddpf = crocoddyl.SolverFDDP(pbf)
# ddpc1 = crocoddyl.SolverFDDP(pbc1)
ddpc2 = crocoddyl.SolverFDDP(pbc2)

ddpf.setCallbacks([crocoddyl.CallbackLogger(),
                crocoddyl.CallbackVerbose()])
# ddpc1.setCallbacks([crocoddyl.CallbackLogger(),
#                 crocoddyl.CallbackVerbose()])
ddpc2.setCallbacks([crocoddyl.CallbackLogger(),
                crocoddyl.CallbackVerbose()])
# ddpc.reg_incFactor = 1.01

# # Warm start : initial state + gravity compensation
# lfext0 = [pin.Force.Zero() for _ in range(model.njoints)]
# from core_mpc import pin_utils
# xs_init = [x0 for i in range(N+1)]
# us_init = [pin_utils.get_tau(q0, v0, np.zeros(nq), lfext0, model, np.zeros(nq)) for i in range(N)] #ddp.problem.quasiStatic(xs_init[:-1])
# ddpf.solve(xs_init, us_init, maxiter=100, isFeasible=False)
# ddpc2.solve(xs_init, us_init, maxiter=100, isFeasible=False)


ddpf.solve([], [], maxiter=100, isFeasible=False)
# ddpc1.solve([], [], maxiter=100, isFeasible=False)
ddpc2.solve([], [], maxiter=100, isFeasible=False)


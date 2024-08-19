'''
Debugging calc and calcDiff of DAMContactFwdDyn
# Solution 2 : Perform all calculations directly in the LWA frame 
# I.e. express everything in WORLD frame at the contact level 
# It is "cleaner" in theory , but right now doubt about pin.frameAccDerivatives in LWA 
(its a_da not equal LWA_Jc) so contact computes LOCAL then rotates 
FOR contact 1D
'''


from lib2to3.pgen2.token import AT
import numpy as np
np.set_printoptions(precision=6, linewidth=180, suppress=True)

import example_robot_data 
import pinocchio as pin
import crocoddyl 


class bcolors:
    DEBUG = '\033[1m'+'\033[96m'
    ERROR = '\033[1m'+'\033[91m'
    ENDC = '\033[0m'
testcolormap = {False: bcolors.ERROR , True: bcolors.DEBUG}


ND_DISTURBANCE  = 1e-6
GAUSS_APPROX    = True
RTOL            = 1e-3 
ATOL            = 1e-3 
RANDOM_SEED     = 1
np.random.seed(RANDOM_SEED)

# Test parameters 
PIN_REFERENCE_FRAME         = pin.WORLD
MASK                        = 2 # along z     
# d->a0[0] = d->a.linear()[mask_] + d->vw[(mask_+1)%3] * d->vv[(mask_+2)%3] - d->vw[(mask_+2)%3] * d->vv[(mask_+1)%3];
ALIGN_LOCAL_WITH_WORLD      = False
TORQUE_SUCH_THAT_ZERO_FORCE = False
ZERO_JOINT_VELOCITY         = False

print(bcolors.DEBUG + "Reference frame = " + str(PIN_REFERENCE_FRAME) + bcolors.ENDC)


# Load robot and setup params
robot = example_robot_data.load('talos_arm')
nq = robot.model.nq; nv = robot.model.nv; nu = nq; nx = nq+nv
# q0 = np.random.rand(nq) 
q0 = np.array([.5,-1,1.5,0,0,-0.5,0])
if(ZERO_JOINT_VELOCITY): 
    print(bcolors.DEBUG + "Set zero joint velocity" + bcolors.ENDC)
    v0 = np.zeros(nq)  
else: 
    v0 = np.random.rand(nv)
x0 = np.concatenate([q0, v0])
robot.framesForwardKinematics(q0)
# Optionally align LOCAL frame with WORLD frame
if(ALIGN_LOCAL_WITH_WORLD):
    print(bcolors.DEBUG + "Aligned LOCAL frame with WORLD" + bcolors.ENDC)
    # Add a custom frame aligned with WORLD to have oRf = identity
    parentFrameId = robot.model.getFrameId("gripper_left_fingertip_1_link")
    parentFrame = robot.model.frames[parentFrameId]
    W_M_j = robot.data.oMi[parentFrame.parent]
    W_M_c = pin.SE3(np.eye(3), W_M_j.act(parentFrame.placement.translation))
    # Add a frame
    customFrame = pin.Frame('contact_frame', parentFrame.parent, parentFrameId, W_M_j.actInv(W_M_c), pin.OP_FRAME)
    robot.model.addFrame(customFrame)
    contactFrameName = customFrame.name 
    # Update data
    robot.data = robot.model.createData() 
    robot.framesForwardKinematics(q0)
    robot.computeJointJacobians(q0)
else:
    contactFrameName = "gripper_left_fingertip_1_link"
contactFrameId = robot.model.getFrameId(contactFrameName)
# Optionally pick torque s.t. force is zero
if(TORQUE_SUCH_THAT_ZERO_FORCE):
    print(bcolors.DEBUG + "Select tau s.t. contact force = 0" + bcolors.ENDC)
    # Compute rnea( q=q0, vq=v0, aq=J^+ * gamma, fext=0 )
    fext = [pin.Force.Zero() for i in range(robot.model.njoints)]
    pin.computeAllTerms(robot.model, robot.data, q0, v0)
    J = pin.getFrameJacobian(robot.model, robot.data, contactFrameId, pin.LOCAL)[:3,:]
    gamma = -pin.getFrameClassicalAcceleration(robot.model, robot.data, contactFrameId, pin.LOCAL)
    aq    = np.linalg.pinv(J) @ gamma.vector[:3]
    tau   = pin.rnea(robot.model, robot.data, q0, v0, aq, fext)
else:
    tau = np.random.rand(nq)
print("x0  = "+str(x0))
print("tau = "+str(tau))
# print("Contact frame placement oRf : \n"+str(robot.data.oMf[contactFrameId]))

nc = 1

CT_REF = np.zeros(3)
GAINS  = [0., 0.]


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


# contact calc : acceleration drift + Jacobian
def contactCalc2bis(model, data, frameId, x, aq, ref):
    '''
    approach 1 : everything in REF
    '''
    pin.computeAllTerms(model, data, x[:nq], x[nq:])
    pin.forwardKinematics(model, data, x[:nq], x[nq:], aq)
    pin.updateFramePlacement(model, data, frameId)
    R = data.oMf[frameId].rotation
    v = pin.getFrameVelocity(model, data, frameId, pin.LOCAL)
    a = pin.getFrameAcceleration(model, data, frameId, pin.LOCAL)
    # print(a.vector)
    a0 = pin.getFrameClassicalAcceleration(model, data, frameId, pin.LOCAL).linear
    la0_no_bg = a0.copy()
    fJf = pin.getFrameJacobian(model, data, frameId, pin.LOCAL)
    assert(np.linalg.norm(a.linear + np.cross(v.angular, v.linear) - a0) <= 1e-6 )
    # Baumgarte term
    if(GAINS[0] != 0.):
        a0 += GAINS[0] * R.T @ (data.oMf[frameId].translation - CT_REF)
    if(GAINS[1] != 0.):
        a0 += GAINS[1] * v.linear
    if(ref == pin.WORLD or ref == pin.LOCAL_WORLD_ALIGNED):
        a0_tmp = a0.copy()
        v_world = pin.getFrameVelocity(model, data, frameId, pin.LOCAL_WORLD_ALIGNED)
        a_world = pin.getFrameAcceleration(model, data, frameId, pin.LOCAL_WORLD_ALIGNED)
        a0_world = pin.getFrameClassicalAcceleration(model, data, frameId, pin.LOCAL_WORLD_ALIGNED).linear
        assert(np.linalg.norm(a_world.linear + np.cross(v_world.angular, v_world.linear) - a0_world) <= 1e-6 )
        assert(np.linalg.norm(R @ la0_no_bg - a0_world) <= 1e-6 )
        wJf = pin.getFrameJacobian(model, data, frameId, pin.LOCAL_WORLD_ALIGNED)
        assert(np.linalg.norm(R @ fJf[:3,:] - wJf[:3,:]) <= 1e-6 ) 
        a0 = R @ a0_tmp
    return a0[MASK]

# Contact calcDiff : acceleration derivatives
def contactCalcDiff2Bis(model, data, frameId, x, ref):
    fJf = pin.getFrameJacobian(model, data, frameId, pin.LOCAL)
    v_partial_dq, a_partial_dq, a_partial_dv, a_partial_da = pin.getFrameAccelerationDerivatives(model, data, frameId, pin.LOCAL) 
    v = pin.getFrameVelocity(model, data, frameId, pin.LOCAL)
    vv = v.linear ; vw = v.angular
    da0_dx_3d = np.zeros((3,nx))
    assert(np.linalg.norm(a_partial_da - fJf) <= 1e-6 )
    assert(np.linalg.norm(da0_dx_3d) <= 1e-6 )
    da0_dx_3d[:,:nv] = a_partial_dq[:3,:]
    da0_dx_3d[:,:nv] += pin.skew(vw) @ v_partial_dq[:3,:]
    da0_dx_3d[:,:nv] -= pin.skew(vv) @ v_partial_dq[3:,:]
    da0_dx_3d[:,nv:] = a_partial_dv[:3,:]
    da0_dx_3d[:,nv:] += pin.skew(vw) @ fJf[:3,:] 
    da0_dx_3d[:,nv:] -= pin.skew(vv) @ fJf[3:,:]
    da0_dx = da0_dx_3d.copy()
    R = data.oMf[frameId].rotation
    # Add Baumgarte term in LOCAL
    if (GAINS[0] != 0.):
        tmp_skew = pin.skew( R.T @ (data.oMf[frameId].translation - CT_REF) )
        da0_dx[:,:nv] += GAINS[0] * ( tmp_skew @ fJf[3:,:] + fJf[:3,:] )
    if (GAINS[1] != 0.):
        da0_dx[:,:nv] += GAINS[1] * v_partial_dq[:3,:]
        da0_dx[:,nv:] += GAINS[1] * fJf[:3,:]
    # Then express in WORLD
    if(ref == pin.WORLD or ref == pin.LOCAL_WORLD_ALIGNED):
        # not optimal, need to keep a0 in memory 
        a0 = pin.getFrameClassicalAcceleration(model, data, frameId, pin.LOCAL).linear
        a0 += GAINS[0] * R.T @ (data.oMf[frameId].translation - CT_REF)
        a0 += GAINS[1] * v.linear
        # print("contactCalcDiff2Bis.a0 = ", a0)
        da0_dx_temp = da0_dx.copy()     
        da0_dx = R @ da0_dx_temp
        Jw = pin.getFrameJacobian(model, data, frameId, pin.LOCAL_WORLD_ALIGNED)[3:,:]
        da0_dx[:,:nq] -= pin.skew(R@a0)@Jw
    return da0_dx[MASK]


print("\n")

# # Checking calc and calcDiff with zero acceleration (no forces)
# frameId = contactFrameId
# model = robot.model
# data = robot.model.createData()
# pin.computeAllTerms(model, data, q0, v0)
# pin.forwardKinematics(model, data, q0, v0, np.zeros(nq))
# pin.computeForwardKinematicsDerivatives(model, data, q0, v0, np.zeros(nq))
# pin.computeJointJacobians(model, data, q0)
# pin.updateFramePlacements(model, data)
# pin.computeForwardKinematicsDerivatives(model, data, q0, v0, np.zeros(nq)) 
# pin.computeRNEADerivatives(model, data, q0, v0, np.zeros(nq)) 
# da0_dx_ND_2bis = numdiff(lambda x_:contactCalc2bis(model, data, contactFrameId, x_, np.zeros(nq), PIN_REFERENCE_FRAME), x0)
# da0_dx_2bis = contactCalcDiff2Bis(model, data, contactFrameId, x0, PIN_REFERENCE_FRAME)
# test_da0_dx_2bis = np.allclose(da0_dx_2bis, da0_dx_ND_2bis, RTOL, ATOL)
# # print(da0_dx_2bis)
# # print(da0_dx_ND_2bis)
# print(testcolormap[test_da0_dx_2bis] + "   -- Test da0_dx ND with drift (aq=0): " + str(test_da0_dx_2bis) + bcolors.ENDC)


# # If we want to get the same results as in dam2.py, need to compute joint acc and force with forwardDynamics(Jc,a0)
# # and then update RNEA derivatives using joint acc, f_ext . It should be ckecked that in this case the classical acc is 0 
# # since the constraint is resolved 
# frameId = contactFrameId
# model = robot.model
# data = robot.model.createData()
# pin.computeAllTerms(model, data, q0, v0)
# R = data.oMf[frameId].rotation
# # Compute forward dynamics with drift obtained from contact model 
#     # Drift 
# a0 = contactCalc2bis(model, data, frameId, x0, np.zeros(nq), PIN_REFERENCE_FRAME)
# if(PIN_REFERENCE_FRAME == pin.LOCAL):
#     fJf = pin.getFrameJacobian(model, data, frameId, pin.LOCAL)
# else:
#     fJf = pin.getFrameJacobian(model, data, frameId, pin.LOCAL_WORLD_ALIGNED)
# pin.forwardDynamics(model, data, tau, fJf[2:3,:], np.array([a0]))
# if(PIN_REFERENCE_FRAME == pin.LOCAL):
#     lf3d = np.array([0.,0.,data.lambda_c[0]])
# else:
#     lf3d = (R.T)[:,2] * data.lambda_c[0]
# # get force at joint level
# fext = [pin.Force.Zero() for _ in range(model.njoints)]
# jMf = model.frames[frameId].placement
# fext[model.frames[frameId].parent] = jMf.act(pin.Force( lf3d, np.zeros(3) ))
#     # Get derivatives
# pin.computeForwardKinematicsDerivatives(model, data, q0, v0, data.ddq) 
# pin.computeRNEADerivatives(model,data, q0, v0, data.ddq, fext)
#     # Check constraint acc = 0
# if(PIN_REFERENCE_FRAME == pin.LOCAL):
#     assert(np.linalg.norm(pin.getFrameClassicalAcceleration(model, data, frameId, pin.LOCAL).linear[MASK]) <= 1e-6 )
# else:
#     assert(np.linalg.norm(pin.getFrameClassicalAcceleration(model, data, frameId, pin.LOCAL_WORLD_ALIGNED).linear[MASK]) <= 1e-6 )
# da0_dx_ND = numdiff(lambda x_:contactCalc2bis(model, data, contactFrameId, x_, data.ddq, PIN_REFERENCE_FRAME), x0)
# da0_dx = contactCalcDiff2Bis(model, data, contactFrameId, x0, PIN_REFERENCE_FRAME)
# test_da0_dx    = np.allclose(da0_dx, da0_dx_ND, RTOL, ATOL)
# print(testcolormap[test_da0_dx] + "   -- Test da0_dx ND with constraint : " + str(test_da0_dx) + bcolors.ENDC)
# print("da0_dx OLD : \n", da0_dx)


# Forward dynamics in LOCAL or LWA by inverting KKT "manually" : ground truth in LOCAL and LWA
def fdyn(model, data, frameId, x, tau, ref):
    '''
    fwdyn(x,u) = forward contact dynamics(q,v,tau) 
    returns the concatenation of configuration acceleration and contact forces 
    '''
    q=x[:nq]
    v=x[nq:]
    pin.computeAllTerms(model,data,q,v)
    pin.forwardKinematics(model,data,q,v,v*0)
    pin.updateFramePlacements(model,data)
    M = data.M
    if(ref == pin.WORLD or ref == pin.LOCAL_WORLD_ALIGNED):
        J = pin.getFrameJacobian(model,data,frameId,pin.LOCAL_WORLD_ALIGNED)[2:3,:]
        a0 = pin.getFrameClassicalAcceleration(model,data,frameId,pin.LOCAL_WORLD_ALIGNED).linear[2]
    else:
        J = pin.getFrameJacobian(model, data, frameId, pin.LOCAL)[2:3,:]
        a0 = pin.getFrameClassicalAcceleration(model,data,frameId,pin.LOCAL).linear[2]
    b = data.nle
    K = np.block([ [M,J.T],[J,np.zeros([nc,nc])] ])
    k = np.concatenate([ tau-b, -np.array([a0]) ])
    # print(a0)
    af = np.linalg.inv(K)@k
    return af 


# METHOD 2: CONTACT LWA (rotated LOCAL) + adjust DAM (skew term in rnea_dx)
def fdyn2bis(model, data, frameId, x, tau, ref): 
    q=x[:nq]
    v=x[nq:]
    pin.computeAllTerms(model, data, q, v)
    pin.forwardKinematics(model, data, q, v, np.zeros(nq))
    pin.updateFramePlacements(model,data)
    # Compute drift + jacobian in ref
    a0 = contactCalc2bis(model, data, frameId, x, np.zeros(nq), ref)
    if(ref == pin.WORLD or ref == pin.LOCAL_WORLD_ALIGNED):
        R = data.oMf[frameId].rotation 
        J = pin.getFrameJacobian(model, data, frameId, pin.LOCAL_WORLD_ALIGNED)[2:3,:]
        assert(np.linalg.norm(a0 - pin.getFrameClassicalAcceleration(model,data,frameId,pin.LOCAL_WORLD_ALIGNED).linear[2])<1e-6)
        assert(np.linalg.norm(J - (R @ pin.getFrameJacobian(model, data, frameId, pin.LOCAL)[:3,:])[2:3])<1e-6)
    else:
        J = pin.getFrameJacobian(model, data, frameId, pin.LOCAL)[2:3,:]
        assert(np.linalg.norm(a0 - pin.getFrameClassicalAcceleration(model,data,frameId,pin.LOCAL).linear[2])<1e-6)
    # Call forward dynamics to get joint acc and force ( in local or lwa frame ) ok     identical to ground truth
    pin.forwardDynamics(model, data, tau, J, np.array([a0]))
    # print(data.ddq)
    return np.hstack([data.ddq, -data.lambda_c]) 


# METHOD 2: CONTACT LWA (rotated LOCAL) + adjust DAM (skew term in rnea_dx)
def fdyn_diff2bis(model, data, frameId, x, tau, ref):
    '''
    computes partial derivatives of joint acc (and force)
        using hard-coded contact model 3D calcDiff()
    '''
    q = x[:nv]
    v = x[nv:]
    Fx = np.zeros((nq,nx))
    df_dx = np.zeros((nc,nx))
    R = data.oMf[frameId].rotation
    # Compute RNEA derivatives and KKT inverse using force and joint acc computed in fwdDyn
        # here we need LOCAL
    if(ref == pin.LOCAL):
        lf3d = np.array([0.,0.,data.lambda_c[0]])
        J = pin.getFrameJacobian(model, data, frameId, pin.LOCAL)[2:3,:]  
    else:
        lf3d = (R.T)[:,2] * data.lambda_c[0] 
        J = pin.getFrameJacobian(model, data, frameId, pin.LOCAL_WORLD_ALIGNED)[2:3,:] 
    jMf = model.frames[frameId].placement
    fext = [pin.Force.Zero() for _ in range(model.njoints)]
    fext[model.frames[frameId].parent] = jMf.act(pin.Force( lf3d, np.zeros(3) ))    
    pin.computeRNEADerivatives(model, data, q, v, data.ddq, fext) 
    drnea_dx = np.hstack([data.dtau_dq, data.dtau_dv])
    Kinv = pin.getKKTContactDynamicMatrixInverse(model, data, J)  
    # Contact derivatives 
    pin.computeForwardKinematicsDerivatives(model, data, q0, v0, data.ddq) 
    da0_dx = contactCalcDiff2Bis(model, data, frameId, x, ref)
    # Check KKT inverse 
    K = np.block([ [data.M,J.T],[J,np.zeros([1,1])] ])
    assert(np.linalg.norm(np.linalg.inv(K) - Kinv) < 1e-6)
    # correct rnea derivatives
    if(ref == pin.WORLD or ref == pin.LOCAL_WORLD_ALIGNED):
        lJ = pin.getFrameJacobian(model, data, frameId, pin.LOCAL) 
        drnea_dx[:,:nv] -= lJ[:3,:].T @ pin.skew(lf3d) @ lJ[3:,:]
        # print(lJ[:3,:].T @ pin.skew(lf3d) @ lJ[3:,:])
    # Fillout partials of DAM 
    a_partial_dtau = Kinv[:nv, :nv]
    a_partial_da   = Kinv[:nv, -nc:]     
    f_partial_dtau = Kinv[-nc:, :nv]
    f_partial_da   = Kinv[-nc:, -nc:]

    Fx = -a_partial_dtau @ drnea_dx 
    # Fx[:,:nv] = -a_partial_dtau @ data.dtau_dq
    # Fx[:,nv:] = -a_partial_dtau @ data.dtau_dv
    Fx -= a_partial_da @ np.array([da0_dx])  
    Fx += a_partial_dtau @ np.zeros((nq,nx))
    # Fu = a_partial_dtau @ np.eye(nq)
    df_dx[:nc,:] = f_partial_dtau @ drnea_dx
    # df_dx[:nc, :nv]  = f_partial_dtau @ data.dtau_dq
    # df_dx[:nc, -nv:] = f_partial_dtau @ data.dtau_dv
    df_dx[:nc, :]   += f_partial_da @ np.array([da0_dx])  
    df_dx[:nc, :]   -= f_partial_dtau @ np.zeros((nq,nx))
    # df_du[:nc, :]  = -f_partial_dtau @ np.eye(nq)
    # print(da0_dx)
    # print(data.ddq)
    return np.vstack([Fx, -df_dx])



# If we want to get the same results as in dam2.py, need to compute joint acc and force with forwardDynamics(Jc,a0)
# and then update RNEA derivatives using joint acc, f_ext . It should be ckecked that in this case the classical acc is 0 
# since the constraint is resolved 
frameId = contactFrameId
model = robot.model
data = robot.model.createData()
# Compare fdyn ground truth and fdyn2bis
af = fdyn(model, data, frameId, x0, tau, PIN_REFERENCE_FRAME)
af2bis = fdyn2bis(model, data, frameId, x0, tau, PIN_REFERENCE_FRAME)
assert(np.linalg.norm(af - af2bis)<1e-6)
# af = np.hstack([data.ddq, -data.lambda_c]) 
daf_dx_ND_0 = numdiff(lambda x_:fdyn(model, data, frameId, x_, tau, PIN_REFERENCE_FRAME), x0)
daf_dx_ND_2bis = numdiff(lambda x_:fdyn2bis(model, data, frameId, x_, tau, PIN_REFERENCE_FRAME), x0)
daf_dx_2bis    = fdyn_diff2bis(model, data, frameId, x0, tau, PIN_REFERENCE_FRAME)
test_daf_dx_0    = np.allclose(daf_dx_ND_0, daf_dx_ND_2bis, RTOL, ATOL)
print(testcolormap[test_daf_dx_0] + "   -- Test fdyn_ND vs fdyn2bis_ND : " + str(test_daf_dx_0) + bcolors.ENDC)
print(np.isclose(daf_dx_2bis, daf_dx_ND_2bis, RTOL, ATOL))
test_daf_dx_2bis    = np.allclose(daf_dx_2bis, daf_dx_ND_2bis, RTOL, ATOL)
# print(daf_dx_ND_0[-1:])
# print(daf_dx_2bis[-1:])
# print(daf_dx_ND_2bis[-1:])
print(testcolormap[test_daf_dx_2bis] + "   -- Test fdyn_diff2bis vs fdyn2bis_ND : " + str(test_daf_dx_2bis) + bcolors.ENDC)






# Calc a0
frameId = contactFrameId
model = robot.model
ldata = robot.model.createData()
# forward dynamics in LOCAL 
pin.computeAllTerms(model, ldata, q0, v0)
pin.forwardKinematics(model, ldata, q0, v0, np.zeros(nq))
pin.updateFramePlacements(model, ldata)
la0_3d = pin.getFrameClassicalAcceleration(model, ldata, frameId, pin.LOCAL).linear
la0 = pin.getFrameClassicalAcceleration(model, ldata, frameId, pin.LOCAL).linear[2:3]
lJ = pin.getFrameJacobian(model, ldata, frameId, pin.LOCAL)
lJc = lJ[2:3,:]
pin.forwardDynamics(model, ldata, tau, lJc, la0)
lK = np.block([ [ldata.M, lJc.T],[lJc, np.zeros([1,1])] ])
laf = np.linalg.inv(lK) @ np.concatenate([tau - ldata.nle, -la0])
assert(np.linalg.norm(laf[:nv] - ldata.ddq) <1e-6)
assert(np.linalg.norm(laf[-nc:] + ldata.lambda_c) <1e-6)
# Derivatives in LOCAL
fext = [pin.Force.Zero() for _ in range(model.njoints)]
f = pin.Force.Zero()
jMf = model.frames[frameId].placement
f.linear = jMf.rotation[:,2] * ldata.lambda_c[0] ; f.angular = np.cross(jMf.translation, f.linear)
fext[model.frames[frameId].parent] = f
pin.computeRNEADerivatives(model, ldata, q0, v0, ldata.ddq, fext) 
ldrnea_dx = np.hstack([ldata.dtau_dq, ldata.dtau_dv])
lKinv = pin.getKKTContactDynamicMatrixInverse(model, ldata, lJc)  
assert(np.linalg.norm(lKinv - np.linalg.inv(lK)) <1e-6)
assert(np.linalg.norm(fdyn(model, ldata, frameId, x0, tau, pin.LOCAL) - laf) < 1e-6)
ldaf_dx_ND = numdiff(lambda x_:fdyn(model, ldata, frameId, x_, tau, pin.LOCAL), x0)
ldk_dx = -lK @ ldaf_dx_ND
    # Frame acceleration derivative
v_partial_dq, a_partial_dq, a_partial_dv, a_partial_da = pin.getFrameAccelerationDerivatives(model, ldata, frameId, pin.LOCAL) 
v = pin.getFrameVelocity(model, ldata, frameId, pin.LOCAL)
vv = v.linear ; vw = v.angular
lda0_dx_3d = np.zeros((3,nx))
lda0_dx_3d[:,:nv] = a_partial_dq[:3,:]
lda0_dx_3d[:,:nv] += pin.skew(vw) @ v_partial_dq[:3,:]
lda0_dx_3d[:,:nv] -= pin.skew(vv) @ v_partial_dq[3:,:]
lda0_dx_3d[:,nv:] = a_partial_dv[:3,:]
lda0_dx_3d[:,nv:] += pin.skew(vw) @ lJ[:3,:]
lda0_dx_3d[:,nv:] -= pin.skew(vv) @ lJ[3:,:]
lda0_dx = lda0_dx_3d[2]
assert(np.linalg.norm(lda0_dx - contactCalcDiff2Bis(model, ldata, frameId, x0, pin.LOCAL))<1e-4)
assert(np.linalg.norm(ldk_dx[-nc:] - lda0_dx) <1e-4)
assert(np.linalg.norm(ldk_dx[:nv] - ldrnea_dx) <1e-4)
assert(np.linalg.norm(ldaf_dx_ND + lKinv @ np.vstack([ldrnea_dx, lda0_dx])) <1e-3)
# print(ldata.ddq)
# print(laf)
# print(ldaf_dx_ND[-1:])




# forward dynamics in WORLD
wdata = model.createData()
pin.computeAllTerms(model, wdata, q0, v0)
pin.forwardKinematics(model, wdata, q0, v0, np.zeros(nq))
pin.updateFramePlacements(model, wdata)
R = wdata.oMf[frameId].rotation
wa0_3d = pin.getFrameClassicalAcceleration(model, wdata, frameId, pin.LOCAL_WORLD_ALIGNED).linear
wa0 = pin.getFrameClassicalAcceleration(model, wdata, frameId, pin.LOCAL_WORLD_ALIGNED).linear[2:3]
assert(np.linalg.norm(wa0_3d - R@la0_3d) <1e-6)
assert(np.linalg.norm(wa0 - (R@la0_3d)[2]) <1e-6)
wJf_full = pin.getFrameJacobian(model, wdata, frameId, pin.LOCAL_WORLD_ALIGNED)
wJf = pin.getFrameJacobian(model, wdata, frameId, pin.LOCAL_WORLD_ALIGNED)[2:3,:]
pin.forwardDynamics(model, wdata, tau, wJf, wa0)
wK = np.block([ [wdata.M, wJf.T],[wJf, np.zeros([1,1])] ])
waf = np.linalg.inv(wK) @ np.concatenate([tau - wdata.nle, -wa0])
print("WORLD aq = \n", waf[:nv])
print("WORLD ddx = \n", wdata.ddq)
print("WORLD force 1D = \n", waf[-nc:])
print("WORLD Jc 1D = \n", wJf)
Kinv = np.linalg.inv(wK)
a_partial_dtau = Kinv[:nv, :nv]
a_partial_da   = Kinv[:nv, -nc:]     
f_partial_dtau = Kinv[-nc:, :nv]
f_partial_da   = Kinv[-nc:, -nc:]
# print("WORLD Kinv = \n", np.linalg.inv(wK))
# print("WORLD a_partial_dtau = \n", a_partial_dtau)
# print("WORLD a_partial_da = \n", a_partial_da)
# print("WORLD f_partial_dtau = \n", f_partial_dtau)
# print("WORLD f_partial_da = \n", f_partial_da)
assert(np.linalg.norm(waf[:nv] - wdata.ddq) <1e-6)
assert(np.linalg.norm(fdyn(model, wdata, frameId, x0, tau, pin.LOCAL_WORLD_ALIGNED) - waf) < 1e-6)
# assert(np.linalg.norm(waf[:nv] - laf[:nv] ) < 1e-6) # no longer true in 1D !!!!!
assert(np.linalg.norm(waf[-nc:] + wdata.lambda_c) <1e-6)
# Derivatives in WORLD
fext = [pin.Force.Zero() for _ in range(model.njoints)]
# wf3d = np.zeros(3) ; wf3d[2] = wdata.lambda_c[0] ; lf3d = R.T @ wf3d 
# above line is the same as : 
lf3d = (R.T)[:,2] * wdata.lambda_c[0] 
print("WORLD data.lambda_c = \n", wdata.lambda_c)
jMf = model.frames[frameId].placement
fext[model.frames[frameId].parent] = jMf.act(pin.Force( lf3d, np.zeros(3) ))
print("fext = \n", fext)
pin.computeRNEADerivatives(model, wdata, q0, v0, wdata.ddq, fext) 
wdrnea_dx = np.hstack([wdata.dtau_dq, wdata.dtau_dv])
# assert(np.linalg.norm(wdrnea_dx -ldrnea_dx) <1e-4) # no longer true in 1D !!!!!
    # additional term  
lJ = pin.getFrameJacobian(model, wdata, frameId, pin.LOCAL)
wdrnea_dx[:,:nv] -= lJ[:3].T @ pin.skew(lf3d) @ lJ[3:] 
print("skew term = \n", -lJ[:3].T @ pin.skew(lf3d) @ lJ[3:] )
wKinv = pin.getKKTContactDynamicMatrixInverse(model, wdata, wJf)  
assert(np.linalg.norm(wKinv - np.linalg.inv(wK))<1e-6)
assert(np.linalg.norm(fdyn(model, wdata, frameId, x0, tau, pin.LOCAL_WORLD_ALIGNED) - waf) < 1e-6)
wdaf_dx_ND = numdiff(lambda x_:fdyn(model, wdata, frameId, x_, tau, pin.LOCAL_WORLD_ALIGNED), x0)
wdk_dx = -wK @ wdaf_dx_ND
# check rnea derivatives
assert(np.linalg.norm(wdk_dx[:nv] - wdrnea_dx)<1e-4)
    # Frame acc derivative 
# need to recompute local with proper ddq 
v_partial_dq, a_partial_dq, a_partial_dv, a_partial_da = pin.getFrameAccelerationDerivatives(model, wdata, frameId, pin.LOCAL) 
v = pin.getFrameVelocity(model, wdata, frameId, pin.LOCAL) ; vv = v.linear ; vw = v.angular
lda0_dx_3d = np.zeros((3,nx))
lda0_dx_3d[:,:nv] = a_partial_dq[:3,:]
lda0_dx_3d[:,:nv] += pin.skew(vw) @ v_partial_dq[:3,:]
lda0_dx_3d[:,:nv] -= pin.skew(vv) @ v_partial_dq[3:,:]
lda0_dx_3d[:,nv:] = a_partial_dv[:3,:]
lda0_dx_3d[:,nv:] += pin.skew(vw) @ lJ[:3,:]
lda0_dx_3d[:,nv:] -= pin.skew(vv) @ lJ[3:,:]
wda0_dx_3d = R @ lda0_dx_3d
pin.computeForwardKinematicsDerivatives(model, wdata, q0, v0, wdata.ddq ) # very important CAREFUL ddq not same in LOCAL and WORLD
# Check that acc has been constrained to 0 in LWA 
wa = pin.getFrameClassicalAcceleration(model, wdata, frameId, pin.LOCAL_WORLD_ALIGNED).linear
assert(np.linalg.norm(wa[2]) <1e-6)
# But careful the skew term is NOT zero !
Jw = pin.getFrameJacobian(model, wdata, frameId, pin.LOCAL_WORLD_ALIGNED)[3:,:]
wda0_dx_3d[:,:nv] -= pin.skew(wa)@Jw
assert(np.linalg.norm(pin.skew(wa)[:,2]@Jw + (pin.skew(wa)@Jw)[2] ) < 1e-4)
w_da0_dx_ND = numdiff(lambda x_:contactCalc2bis(model, wdata, frameId, x_, wdata.ddq, pin.LOCAL_WORLD_ALIGNED), x0)
assert(np.linalg.norm(w_da0_dx_ND - wdk_dx[-nc:]) < 1e-4)
assert(np.linalg.norm(wda0_dx_3d[2] - wdk_dx[-nc:]) < 1e-4)
# print("numdiff  : \n", w_da0_dx_ND)    # numdiff
# print("analytic : \n", wdk_dx[-nc:])      # analytical 
print("formula  : \n", wda0_dx_3d[2])     # formula 3D projected into 1D
# rnea derivatives
wdaf_dx = -wKinv @ np.vstack([wdrnea_dx, wda0_dx_3d[2]])
assert(np.linalg.norm(wdaf_dx_ND - wdaf_dx) <1e-3)

print("OK")
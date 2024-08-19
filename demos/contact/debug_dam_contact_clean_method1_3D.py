'''
Debugging calc and calcDiff of DAMContactFwdDyn
# Solution 2 : Perform all calculations directly in the LWA frame 
# I.e. express everything in WORLD frame at the contact level 
# It is "cleaner" in theory , but right now doubt about pin.frameAccDerivatives in LWA 
(its a_da not equal LWA_Jc) so contact computes LOCAL then rotates 
'''


import numpy as np
np.set_printoptions(precision=3, linewidth=180, suppress=True)

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

nc = 3

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

# METHOD 1 : CONTACT LOCAL 
def contactCalc1(model, data, frameId, x, a):
    '''
    approach 1 : everything in LOCAL
    '''
    pin.computeAllTerms(model, data, x[:nq], x[nq:])
    pin.forwardKinematics(model, data, x[:nq], x[nq:], a)
    pin.updateFramePlacement(model, data, frameId)
    v = pin.getFrameVelocity(model, data, frameId, pin.LOCAL)
    a = pin.getFrameAcceleration(model, data, frameId, pin.LOCAL)
    a0 = pin.getFrameClassicalAcceleration(model, data, frameId, pin.LOCAL).linear
    assert(np.linalg.norm(a.linear + np.cross(v.angular, v.linear) - a0) <= 1e-6 )
    # Baumgarte term
    if(GAINS[0] != 0.):
        R = data.oMf[frameId].rotation
        a0 += GAINS[0] * R.T @ (data.oMf[frameId].translation - CT_REF)
    if(GAINS[1] != 0.):
        a0 += GAINS[1] * v.linear
    return a0

# METHOD 1 : CONTACT LOCAL 
def contactCalcDiff1(model, data, frameId, x):
    fJf = pin.getFrameJacobian(model, data, frameId, pin.LOCAL)
    v_partial_dq, a_partial_dq, a_partial_dv, a_partial_da = pin.getFrameAccelerationDerivatives(model, data, frameId, pin.LOCAL) 
    assert(np.linalg.norm(a_partial_da - fJf) <= 1e-6 )
    v = pin.getFrameVelocity(model, data, frameId, pin.LOCAL)
    vv = v.linear ; vw = v.angular
    da0_dx = np.zeros((nc,nx))
    assert(np.linalg.norm(da0_dx) <= 1e-6 )
    da0_dx[:,:nv] = a_partial_dq[:3,:]
    da0_dx[:,:nv] += pin.skew(vw) @ v_partial_dq[:3,:]
    da0_dx[:,:nv] -= pin.skew(vv) @ v_partial_dq[3:,:]
    da0_dx[:,nv:] = a_partial_dv[:3,:]
    da0_dx[:,nv:] += pin.skew(vw) @ fJf[:3,:] 
    da0_dx[:,nv:] -= pin.skew(vv) @ fJf[3:,:]
    if (GAINS[0] != 0.):
        R = data.oMf[frameId].rotation
        tmp_skew = pin.skew( R.T @ (data.oMf[frameId].translation - CT_REF) )
        da0_dx[:,:nv] += GAINS[0] * ( tmp_skew @ fJf[3:,:] + fJf[:3,:] )
    if (GAINS[1] != 0.):
        da0_dx[:,:nv] += GAINS[1] * v_partial_dq[:3,:]
        da0_dx[:,nv:] += GAINS[1] * fJf[:3,:]
    return da0_dx

# Checking calc and calcDiff with zero acceleration (no forces)
frameId = contactFrameId
model = robot.model
data = robot.model.createData()
pin.computeAllTerms(model, data, q0, v0)
pin.forwardKinematics(model, data, q0, v0, np.zeros(nq))
pin.computeForwardKinematicsDerivatives(model, data, q0, v0, np.zeros(nq))
pin.computeJointJacobians(model, data, q0)
pin.updateFramePlacements(model, data)
pin.computeForwardKinematicsDerivatives(model, data, q0, v0, np.zeros(nq)) 
pin.computeRNEADerivatives(model, data, q0, v0, np.zeros(nq)) 
da0_dx_ND_1 = numdiff(lambda x_:contactCalc1(model, data, contactFrameId, x_, np.zeros(nq)), x0)
da0_dx_1 = contactCalcDiff1(model, data, contactFrameId, x0)
test_da0_dx_1 = np.allclose(da0_dx_1, da0_dx_ND_1, RTOL, ATOL)
print(testcolormap[test_da0_dx_1] + "   -- Test da0_dx numdiff (1) drift : " + str(test_da0_dx_1) + bcolors.ENDC)
# if(not test_da0_dx_1):
#     print("analytic = \n", da0_dx_1)
#     print("numdiff \n", da0_dx_ND_1)

print("\n")

# If we want to get the same results as in dam2.py, need to compute joint acc and force with forwardDynamics(Jc,a0)
# and then update RNEA derivatives using joint acc, f_ext . It should be ckecked that in this case the classical acc is 0 
# since the constraint is resolved 
frameId = contactFrameId
model = robot.model
data = robot.model.createData()
pin.computeAllTerms(model, data, q0, v0)
# Compute forward dynamics with drift obtained from contact model 
    # Drift 
a0 = contactCalc1(model, data, frameId, x0, np.zeros(nq))
fJf = pin.getFrameJacobian(model, data, frameId, pin.LOCAL)
pin.forwardDynamics(model, data, tau, fJf[:3,:], a0)
    # get force at joint level
fext = [pin.Force.Zero() for _ in range(model.njoints)]
fext[model.frames[frameId].parent] = model.frames[frameId].placement.act(pin.Force(data.lambda_c, np.zeros(3)))
    # Get derivatives
pin.computeForwardKinematicsDerivatives(model, data, q0, v0, data.ddq) 
pin.computeRNEADerivatives(model,data, q0, v0, data.ddq, fext)
    # Check constraint acc = 0
# assert(np.linalg.norm(pin.getFrameClassicalAcceleration(model, data, frameId, pin.LOCAL).linear) <= 1e-6 )
# Check numdiff
da0_dx_ND_1 = numdiff(lambda x_:contactCalc1(model, data, contactFrameId, x_, data.ddq), x0)
da0_dx_1 = contactCalcDiff1(model, data, contactFrameId, x0)
test_da0_dx_1 = np.allclose(da0_dx_1, da0_dx_ND_1, RTOL, ATOL)
print(testcolormap[test_da0_dx_1] + "   -- Test da0_dx numdiff (1) with constraint: " + str(test_da0_dx_1) + bcolors.ENDC)
# if(not test_da0_dx_1):
    # print("analytic = \n", da0_dx_1)
    # print("numdiff \n", da0_dx_ND_1)



# Forward dynamics


# Forward dynamics in LOCAL or WORLD, inverting KKT : ground truth in LOCAL and LWA
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
        J = pin.getFrameJacobian(model,data,frameId,pin.LOCAL_WORLD_ALIGNED)[:3,:]
        a0 = pin.getFrameClassicalAcceleration(model,data,frameId,pin.LOCAL_WORLD_ALIGNED).linear
    else:
        J = pin.getFrameJacobian(model, data, frameId, pin.LOCAL)[:3,:]
        a0 = pin.getFrameClassicalAcceleration(model,data,frameId,pin.LOCAL).linear
    b = data.nle
    K = np.block([ [M,J.T],[J,np.zeros([3,3])] ])
    k = np.concatenate([ tau-b, -a0 ])
    af = np.linalg.inv(K)@k
    return af 


# METHOD 1: CONTACT LOCAL + CHANGE DAM
def fdyn1(model, data, frameId, x, tau, ref): 
    q=x[:nq]
    v=x[nq:]
    pin.computeAllTerms(model, data, q, v)
    pin.forwardKinematics(model, data, q, v, np.zeros(nq))
    # Compute drift + jacobian in LOCAL 
    a0 = contactCalc1(model, data, frameId, x, np.zeros(nq))
    J = pin.getFrameJacobian(model, data, frameId, pin.LOCAL)[:3,:]
    # Call forward dynamics to get joint acc and force
    pin.forwardDynamics(model, data, tau, J, a0)
    # Compute force in the right frame !!!
    if(ref == pin.LOCAL):
        f = data.lambda_c
    else:
        R = data.oMf[frameId].rotation
        f = R @ data.lambda_c 
    return np.hstack([data.ddq, f]) 


# METHOD 1: CONTACT LOCAL + CHANGE DAM
def fdyn_diff1(model, data, frameId, x, tau, ref):
    '''
    computes partial derivatives of joint acc (and force)
        using hard-coded contact model 3D calcDiff()
    '''
    q = x[:nv]
    v = x[nv:]
    Fx = np.zeros((nq,nx))
    df_dx = np.zeros((3,nx))
    # Compute RNEA derivatives and KKT inverse using force and joint acc computed in fwdDyn
        # We use local quantities !
    fext = [pin.Force.Zero() for _ in range(model.njoints)]
    fext[model.frames[frameId].parent] = model.frames[frameId].placement.act(pin.Force(data.lambda_c, np.zeros(3)))
    pin.computeRNEADerivatives(model, data, q, v, data.ddq, fext) 
    # print("drnea_dx = \n", np.hstack([data.dtau_dq, data.dtau_dv]))
    J = pin.getFrameJacobian(model, data, frameId, pin.LOCAL)[:3,:]   
    Kinv = pin.getKKTContactDynamicMatrixInverse(model, data, J)  
    # Contact derivatives 
    da0_dx = contactCalcDiff1(model, data, frameId, x)
    # Fillout partials of DAM 
    a_partial_dtau = Kinv[:nv, :nv]
    a_partial_da   = Kinv[:nv, -nc:]     
    f_partial_dtau = Kinv[-nc:, :nv]
    f_partial_da   = Kinv[-nc:, -nc:]
    Fx[:,:nv] = -a_partial_dtau @ data.dtau_dq
    Fx[:,nv:] = -a_partial_dtau @ data.dtau_dv
    Fx -= a_partial_da @ da0_dx[:nc] 
    Fx += a_partial_dtau @ np.zeros((nq,nx))
    # Fu = a_partial_dtau @ np.eye(nq)
    df_dx[:nc, :nv]  = f_partial_dtau @ data.dtau_dq
    df_dx[:nc, -nv:] = f_partial_dtau @ data.dtau_dv
    df_dx[:nc, :]   += f_partial_da @ da0_dx[:nc] 
    df_dx[:nc, :]   -= f_partial_dtau @ np.zeros((nq,nx))
    # df_du[:nc, :]  = -f_partial_dtau @ np.eye(nq)
    # # if world, transform force and derivatives here
    if(ref == pin.WORLD or ref == pin.LOCAL_WORLD_ALIGNED):
        R = data.oMf[frameId].rotation
        Jw = pin.getFrameJacobian(model, data, frameId, pin.LOCAL_WORLD_ALIGNED)[3:,:]
        df_dx[:nc,:] = R @ df_dx[:nc,:]
        df_dx[:nc,:nv] -= pin.skew(R @ data.lambda_c)@Jw
    return np.vstack([Fx, df_dx])



# If we want to get the same results as in dam2.py, need to compute joint acc and force with forwardDynamics(Jc,a0)
# and then update RNEA derivatives using joint acc, f_ext . It should be ckecked that in this case the classical acc is 0 
# since the constraint is resolved 
frameId = contactFrameId
model = robot.model
data = robot.model.createData()
daf_dx_ND_0 = numdiff(lambda x_:fdyn(model, data, frameId, x_, tau, PIN_REFERENCE_FRAME), x0)
daf_dx_ND_1 = numdiff(lambda x_:fdyn1(model, data, frameId, x_, tau, PIN_REFERENCE_FRAME), x0)
daf_dx_1    = fdyn_diff1(model, data, frameId, x0, tau, PIN_REFERENCE_FRAME)
# daf_dx_1    = fdyn_diff1(model, data, frameId, x0, tau, PIN_REFERENCE_FRAME)
# test_daf_dx_0    = np.allclose(daf_dx_ND_0, -daf_dx_ND_1, RTOL, ATOL)
# print(daf_dx_ND_0)
# print(daf_dx_ND_1)
test_daf_dx_1    = np.allclose(daf_dx_1, daf_dx_ND_1, RTOL, ATOL)
# print(testcolormap[test_daf_dx_0] + "   -- Test daf_dx (using dyn) : " + str(test_daf_dx_0) + bcolors.ENDC)
print(testcolormap[test_daf_dx_1] + "   -- Test daf_dx (using calc1) : " + str(test_daf_dx_1) + bcolors.ENDC)

print(np.isclose(daf_dx_ND_1, daf_dx_1, RTOL, ATOL))
print(daf_dx_ND_1)
print(daf_dx_1)

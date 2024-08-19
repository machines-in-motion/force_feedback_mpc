'''
Debugging calc and calcDiff of DAMContactFwdDyn
# Solution 2 : Perform all calculations directly in the LWA frame 
# I.e. express everything in WORLD frame at the contact level 
# It is "cleaner" in theory , but right now doubt about pin.frameAccDerivatives in LWA 
(its a_da not equal LWA_Jc) so contact computes LOCAL then rotates 
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
PIN_REFERENCE_FRAME         = pin.LOCAL_WORLD_ALIGNED     
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
    # print("contact.calc.a0 = \n", a0)
    return a0

# Contact calcDiff : acceleration derivatives
def contactCalcDiff2Bis(model, data, frameId, x, ref):
    fJf = pin.getFrameJacobian(model, data, frameId, pin.LOCAL)
    v_partial_dq, a_partial_dq, a_partial_dv, a_partial_da = pin.getFrameAccelerationDerivatives(model, data, frameId, pin.LOCAL) 
    # check relation between joint and frame acc derivatives (through fXj)
    jv_partial_dq, ja_partial_dq, ja_partial_dv, ja_partial_da = pin.getJointAccelerationDerivatives(model, data, model.frames[frameId].parent, pin.LOCAL)
    fXj = model.frames[frameId].placement.actionInverse
    assert(np.linalg.norm(fXj @ jv_partial_dq - v_partial_dq) <= 1e-4 )
    assert(np.linalg.norm(fXj @ ja_partial_dq - a_partial_dq) <= 1e-4 )
    assert(np.linalg.norm(fXj @ ja_partial_dv - a_partial_dv) <= 1e-4 )
    # assert(np.linalg.norm(fXj*ja_partial_da - a_partial_da) <= 1e-6 )
    v = pin.getFrameVelocity(model, data, frameId, pin.LOCAL)
    vv = v.linear ; vw = v.angular
    da0_dx = np.zeros((nc,nx))
    assert(np.linalg.norm(a_partial_da - fJf) <= 1e-6 )
    assert(np.linalg.norm(da0_dx) <= 1e-6 )
    da0_dx[:,:nv] = a_partial_dq[:3,:]
    da0_dx[:,:nv] += pin.skew(vw) @ v_partial_dq[:3,:]
    da0_dx[:,:nv] -= pin.skew(vv) @ v_partial_dq[3:,:]
    da0_dx[:,nv:] = a_partial_dv[:3,:]
    da0_dx[:,nv:] += pin.skew(vw) @ fJf[:3,:] 
    da0_dx[:,nv:] -= pin.skew(vv) @ fJf[3:,:]
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
        da0_dx[:,:nq] -= pin.skew(R @ a0)@Jw
    return da0_dx


print("\n")

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
# da0_dx_ND_2 = numdiff(lambda x_:contactCalc2(model, data, contactFrameId, x_, np.zeros(nq), PIN_REFERENCE_FRAME), x0)
# da0_dx_2    = contactCalcDiff2(model, data, contactFrameId, x0, PIN_REFERENCE_FRAME)
da0_dx_ND_2bis = numdiff(lambda x_:contactCalc2bis(model, data, contactFrameId, x_, np.zeros(nq), PIN_REFERENCE_FRAME), x0)
da0_dx_2bis = contactCalcDiff2Bis(model, data, contactFrameId, x0, PIN_REFERENCE_FRAME)
# test_da0_dx_2    = np.allclose(da0_dx_2, da0_dx_ND_2, RTOL, ATOL)
test_da0_dx_2bis = np.allclose(da0_dx_2bis, da0_dx_ND_2bis, RTOL, ATOL)
# test_da0_dx_2crossed = np.allclose(da0_dx_2, da0_dx_2bis, RTOL, ATOL)
# print(testcolormap[test_da0_dx_2] + "   -- Test da0_dx numdiff (2) drift : " + str(test_da0_dx_2) + bcolors.ENDC)
# if(not test_da0_dx_2):
#     print("analytic 2 = \n", da0_dx_2)
#     print("numdiff \n", da0_dx_ND_2)
print(testcolormap[test_da0_dx_2bis] + "   -- Test da0_dx ND with drift (aq=0) : " + str(test_da0_dx_2bis) + bcolors.ENDC)
# if(not test_da0_dx_2bis):
    # print("analytic 2bis = \n", da0_dx_2bis)
    # print("numdiff \n", da0_dx_ND_2bis)
# print(testcolormap[test_da0_dx_2crossed] + "   -- Test da0_dx crossed (2 and 2bis) drift: " + str(test_da0_dx_2crossed) + bcolors.ENDC)


print("\n")

# Contact calcDiff : acceleration derivatives
'''
Checking formula : 
    ref_{ d(alpha) / dx } =  FD[ ref_alpha / dx ]

with LHS       := getFrameAccelerationDerivatives(ref)
     ref_alpha := getFrameAcceleration(ref)

OK for ref=LOCAL and ref=WORLD but NOT for ref=LWA
'''
def a_partial_dx(model, data, frameId, x, ref):
    _, a_partial_dq, a_partial_dv, _ = pin.getFrameAccelerationDerivatives(model, data, frameId, ref)
    a_partial_dx = np.hstack([ a_partial_dq, a_partial_dv ])
    return a_partial_dx

def linear_acc(model, data, frameId, x, a, ref):
    pin.forwardKinematics(model,data,x[:nq],x[nq:nq+nv],a)
    pin.updateFramePlacements(model,data)
    a = pin.getFrameAcceleration(model,data,frameId,ref).vector
    return a

def classical_acc(model, data, frameId, x, a, ref):
    pin.forwardKinematics(model,data,x[:nq],x[nq:nq+nv],a)
    pin.updateFramePlacements(model,data)
    a = pin.getFrameClassicalAcceleration(model,data,frameId,ref).linear
    return a

# Check that a_partial_dx matches the derivative of the acceleration 
# not the case in LWA, but OK in LOCAL --> see inside pinocchio : could explain why contactCalcDiff2 fails in LWA !
# fix is to use contactCalcDiff2bis, which uses LOCAL computations to derive LWA quantities (ugly but it works) 
aq0 = np.random.rand(nq) # data.ddq
pin.forwardKinematics(model,data, q0, v0, aq0)
pin.updateFramePlacements(model,data)
pin.computeForwardKinematicsDerivatives(model,data, q0, v0, aq0)
linear_acc_ND = numdiff(lambda x_:linear_acc(model, data, frameId, x_, aq0, pin.WORLD), x0)
a_partial_dx  = a_partial_dx(model, data, frameId, x0, pin.WORLD) #[:3,:]
test_partial_dx    = np.allclose(linear_acc_ND, a_partial_dx, RTOL, ATOL)
print(testcolormap[test_partial_dx] + "   -- Test frame_acc_partial_dx = ND(frame_acc) : " + str(test_partial_dx) + bcolors.ENDC)
# print("analytic d(spatial a) / dx: \n", a_partial_dx)
# print("numdiff(spatial a)= \n", linear_acc_ND)


print("\n")




'''
Since acceleration derivatives in LWA from pinocchio do not match FD of the acceleration in LWA
we derive expressions of acceleration derivatives  in LWA in terms of LOCAL WORLD derivatives
We then choose the most convenient / cheapest one to implement (e.g. lowest number of operations)
    1. Express LWA derivatives in terms of LOCAL
    2. "    "   "   "   "   "   "   "   "  WORLD
    3. "    "   "   "   "   "   "   "   "  LWA 
'''


# Check relation between jointVelDerivatives LWA and WORLD 
'''
Checking formula : 
    LWA_{ d(nu) / dx } =  X * W_{ d(nu) / dx }
'''
aq0 = np.zeros(nq)
pin.forwardKinematics(model,data, q0, v0, aq0)
pin.updateFramePlacements(model,data)
pin.computeForwardKinematicsDerivatives(model,data, q0, v0, aq0)
parentJointId = model.frames[contactFrameId].parent
assert(parentJointId == 6)
# lwav_partial_dq, lwav_partial_dv = pin.getJointVelocityDerivatives(model, data, parentJointId, pin.LOCAL_WORLD_ALIGNED) 
# wv_partial_dq, wv_partial_dv = pin.getJointVelocityDerivatives(model, data, parentJointId, pin.WORLD) 
lwav_partial_dq, lwav_partial_dv = pin.getFrameVelocityDerivatives(model, data, contactFrameId, pin.LOCAL_WORLD_ALIGNED) 
wv_partial_dq, wv_partial_dv = pin.getFrameVelocityDerivatives(model, data, contactFrameId, pin.WORLD) 
# print(lwav_partial_dq)
# print(data.oMf[frameId].action @ wv_partial_dq)
p = data.oMf[frameId].translation
# p = data.oMi[parentJointId].translation
X = np.block([ [np.eye(3),-pin.skew(p)],[np.zeros((3,3)), np.eye(3)] ]) 
lwav_partial_dx = np.hstack([lwav_partial_dq, lwav_partial_dv])
wv_partial_dx = np.hstack([wv_partial_dq, wv_partial_dv])
# print("LWA{ d(spatial v) / dx } = \n", lwav_partial_dx)
# print("X * W{ d(spatial v) / dx } = \n", X @ wv_partial_dx)
assert(np.linalg.norm(lwav_partial_dx - X.dot(wv_partial_dx)) < 1e-3)


# Check relation between acc derivatives LWA and WORLD
'''
Checking formula : 
    LWA_{ d(alpha) / dx } =  X * W_{ d(alpha) / dx }
'''
aq0 = np.zeros(nq)
pin.forwardKinematics(model,data, q0, v0, np.zeros(nq))
pin.updateFramePlacements(model,data)
pin.computeForwardKinematicsDerivatives(model,data, q0, v0, np.zeros(nq))
parentJointId = model.frames[contactFrameId].parent
assert(parentJointId == 6)
_, lwaa_partial_dq, lwaa_partial_dv, _ = pin.getFrameAccelerationDerivatives(model, data, contactFrameId, pin.LOCAL_WORLD_ALIGNED) 
_, wa_partial_dq, wa_partial_dv, _ = pin.getFrameAccelerationDerivatives(model, data, contactFrameId, pin.WORLD)
p = data.oMf[frameId].translation
X = np.block([ [np.eye(3),-pin.skew(p)],[np.zeros((3,3)), np.eye(3)] ]) 
lwaa_partial_dx = np.hstack([lwaa_partial_dq, lwaa_partial_dv])
wa_partial_dx = np.hstack([wa_partial_dq, wa_partial_dv])
# print("LWA{ d(spatial a) / dx } = \n", lwaa_partial_dx)
# print("X * W{ d(spatial a) / dx } = \n", X @ wa_partial_dx)
assert(np.linalg.norm(lwaa_partial_dx - X.dot(wa_partial_dx)) < 1e-3)



# # Check relation in LWA acc deriv NOT OK
# '''
# Checking formulas :
#     LWA_{ d(a) / dx }   =  FD[ LWA_a ]                                      

# where a is the linear CLASSICAL acceleration , i.e. a = alpha[:3] + nu[3:] x nu[:3]
# Fails
# '''
# aq0 = np.zeros(nq)
# ref = pin.LOCAL_WORLD_ALIGNED
# pin.forwardKinematics(model,data, q0, v0, np.zeros(nq))
# pin.updateFramePlacements(model,data)
# pin.computeForwardKinematicsDerivatives(model,data, q0, v0, np.zeros(nq))
# lwav_partial_dq, lwaa_partial_dq, lwaa_partial_dv, lwaa_partial_da = pin.getFrameAccelerationDerivatives(model, data, frameId, ref) 
# lwav = pin.getFrameVelocity(model, data, frameId, ref)
# fJf = pin.getFrameJacobian(model, data, frameId, ref)
# vv = lwav.linear ; vw = lwav.angular
# R = data.oMf[frameId].rotation
# da0_dx = np.zeros((nc,nx))
# assert(np.linalg.norm(lwaa_partial_da - fJf) <= 1e-6 )
# assert(np.linalg.norm(da0_dx) <= 1e-6 )
# da0_dx[:,:nv] = lwaa_partial_dq[:3,:]
# da0_dx[:,:nv] += pin.skew(vw) @ lwav_partial_dq[:3,:]
# da0_dx[:,:nv] -= pin.skew(vv) @ lwav_partial_dq[3:,:]
# da0_dx[:,nv:] = lwaa_partial_dv[:3,:]
# da0_dx[:,nv:] += pin.skew(vw) @ fJf[:3,:] 
# da0_dx[:,nv:] -= pin.skew(vv) @ fJf[3:,:]
# # check 
# lwa_classical_acc_ND = numdiff(lambda x_:classical_acc(model, data, frameId, x_, aq0, ref), x0)
# assert(np.linalg.norm(da0_dx - lwa_classical_acc_ND) < 1e-3)




# Check relation between acceleration in LWA and LOCAL 
'''
Checking formulas :
    L_{ d(a) / dx }   =  FD[ L_a ]                                               (sanity check) 
    LWA_{ d(a) / dx } =  R * L_{ d(a) / dx } - pin.skew( LWA_a0 )x O_J[3:]       (main check)

where a is the linear CLASSICAL acceleration , i.e. a = alpha[:3] + nu[3:] x nu[:3]
'''
aq0 = np.zeros(nq)
pin.forwardKinematics(model,data, q0, v0, np.zeros(nq))
pin.updateFramePlacements(model,data)
pin.computeForwardKinematicsDerivatives(model,data, q0, v0, np.zeros(nq))
lv_partial_dq, la_partial_dq, la_partial_dv, la_partial_da = pin.getFrameAccelerationDerivatives(model, data, frameId, pin.LOCAL) 
lv = pin.getFrameVelocity(model, data, frameId, pin.LOCAL)
fJf = pin.getFrameJacobian(model, data, frameId, pin.LOCAL)
vv = lv.linear ; vw = lv.angular
lwa_a = classical_acc(model, data, frameId, x0, aq0, pin.LOCAL_WORLD_ALIGNED)
R = data.oMf[frameId].rotation
da0_dx = np.zeros((nc,nx))
assert(np.linalg.norm(la_partial_da - fJf) <= 1e-6 )
assert(np.linalg.norm(da0_dx) <= 1e-6 )
da0_dx[:,:nv] = la_partial_dq[:3,:]
da0_dx[:,:nv] += pin.skew(vw) @ lv_partial_dq[:3,:]
da0_dx[:,:nv] -= pin.skew(vv) @ lv_partial_dq[3:,:]
da0_dx[:,nv:] = la_partial_dv[:3,:]
da0_dx[:,nv:] += pin.skew(vw) @ fJf[:3,:] 
da0_dx[:,nv:] -= pin.skew(vv) @ fJf[3:,:]
    # sanity check of LOCAL derivatives of classical acceleration (linear)
l_classical_acc_ND = numdiff(lambda x_:classical_acc(model, data, frameId, x_, aq0, pin.LOCAL), x0)
assert(np.linalg.norm(l_classical_acc_ND - da0_dx) < 1e-3)
    # Then express in LWA classical acc (linear) and check against FD 
da0_dx_temp = da0_dx.copy()     
da0_dx = R @ da0_dx_temp
Jw = pin.getFrameJacobian(model, data, frameId, pin.LOCAL_WORLD_ALIGNED)[3:,:]
da0_dx[:,:nq] -= pin.skew(lwa_a)@Jw
lwa_classical_acc_ND = numdiff(lambda x_:classical_acc(model, data, frameId, x_, aq0, pin.LOCAL_WORLD_ALIGNED), x0)
assert(np.linalg.norm(da0_dx - lwa_classical_acc_ND) < 1e-3)




# Check relation between acceleration in LWA and LWA pinocchio output + WORLD stuff 
'''
Checking formula : 
    MISTAKE IN THE FORMULA ?
'''
def lwa_classical_acc_from_world(model, data, frameId, x, a):
    pin.forwardKinematics(model,data,x[:nq],x[nq:nq+nv],a)
    pin.updateFramePlacements(model,data)
    ov = pin.getFrameVelocity(model,data,frameId,pin.WORLD)
    oa = pin.getFrameAcceleration(model,data,frameId,pin.WORLD)
    p = data.oMf[frameId].translation
    lwa_classical_acc = oa.linear - pin.skew(p) @ oa.angular + pin.skew(ov.angular) @ (ov.linear - pin.skew(p) @ ov.angular)
    return lwa_classical_acc



aq0 = np.random.rand(nq)
pin.computeAllTerms(model, data, q0, v0)
pin.forwardKinematics(model,data, q0, v0, aq0)
pin.updateFramePlacements(model,data)
pin.computeForwardKinematicsDerivatives(model,data, q0, v0, aq0)
p = data.oMf[frameId].translation
# R = data.oMf[frameId].rotation
# LWA stuff
lwa_alpha = pin.getFrameAcceleration(model, data, frameId, pin.LOCAL_WORLD_ALIGNED) 
lwa_nu = pin.getFrameVelocity(model, data, frameId, pin.LOCAL_WORLD_ALIGNED)
lwa_a = pin.getFrameClassicalAcceleration(model, data, frameId, pin.LOCAL_WORLD_ALIGNED)
assert(np.linalg.norm(lwa_alpha.linear + np.cross(lwa_nu.angular, lwa_nu.linear) - lwa_a.linear) <1e-6)
assert(np.linalg.norm(lwa_classical_acc_from_world(model, data, frameId, x0, aq0) - lwa_a.linear) < 1e-6)
assert(np.linalg.norm(lwa_classical_acc_from_world(model, data, frameId, x0, aq0) - classical_acc(model, data, frameId, x0, aq0, pin.LOCAL_WORLD_ALIGNED)) < 1e-6)
_, lwa_alpha_partial_dq, lwa_alpha_partial_dv, lwa_alpha_da = pin.getFrameAccelerationDerivatives(model, data, frameId, pin.LOCAL_WORLD_ALIGNED) 
lwa_alpha_partial_dx = np.hstack([lwa_alpha_partial_dq, lwa_alpha_partial_dv])
lwa_nu_partial_dq, lwa_nu_partial_dv = pin.getFrameVelocityDerivatives(model, data, frameId, pin.LOCAL_WORLD_ALIGNED)
lwa_nu_partial_dx = np.hstack([lwa_nu_partial_dq, lwa_nu_partial_dv])
lwaJf = pin.getFrameJacobian(model, data, frameId, pin.LOCAL_WORLD_ALIGNED)
assert(np.linalg.norm(lwaJf - lwa_alpha_da) <1e-6) 
# WORLD stuff
o_alpha = pin.getFrameAcceleration(model, data, frameId, pin.WORLD) 
o_nu = pin.getFrameVelocity(model, data, frameId, pin.WORLD)
o_nu_partial_dq, o_nu_partial_dv = pin.getFrameVelocityDerivatives(model, data, frameId, pin.WORLD)
o_nu_partial_dx = np.hstack([o_nu_partial_dq, o_nu_partial_dv])
_, o_alpha_partial_dq, o_alpha_partial_dv, o_alpha_da = pin.getFrameAccelerationDerivatives(model, data, frameId, pin.WORLD) 
o_alpha_partial_dx = np.hstack([o_alpha_partial_dq, o_alpha_partial_dv])
oJf = pin.getFrameJacobian(model, data, frameId, pin.WORLD)
assert(np.linalg.norm(oJf - o_alpha_da) <1e-6) 
    # sanity check : relation between vel, acc + Derivatives in LWA and WORLD 
assert(np.linalg.norm(lwa_nu.linear - (o_nu.linear - pin.skew(p) @ o_nu.angular)) < 1e-6)
assert(np.linalg.norm(lwa_alpha.linear - (o_alpha.linear - pin.skew(p) @ o_alpha.angular)) < 1e-6)
assert(np.linalg.norm(lwa_nu_partial_dx[3:,:] - o_nu_partial_dx[3:,:]) < 1e-6)
assert(np.linalg.norm(lwa_nu_partial_dx[:3,:] - (o_nu_partial_dx[:3,:] - pin.skew(p) @ o_nu_partial_dx[3:,:])) < 1e-6)
assert(np.linalg.norm(lwa_alpha_partial_dx[:3,:] - (o_alpha_partial_dx[:3,:] - pin.skew(p) @ o_alpha_partial_dx[3:,:])) < 1e-6)
# Compute LWA derivatives as function of world and LWA quantities
da0_dx = np.zeros((nc,nq))
da0_dx = lwa_alpha_partial_dx[:3,:]
da0_dx += pin.skew(o_nu.angular) @ lwa_nu_partial_dx[:3,:] - pin.skew(lwa_nu.linear) @ o_nu_partial_dx[3:,:] 
da0_dx[:,:nq] += pin.skew(o_alpha.angular) @ lwaJf[:3,:]
da0_dx[:,:nq] += pin.skew(o_nu.angular) @ (pin.skew(o_nu.angular) @ lwaJf[:3,:])
lwa_classical_acc_ND = numdiff(lambda x_:lwa_classical_acc_from_world(model, data, frameId, x_, aq0), x0)
# print(da0_dx)
# print(lwa_classical_acc_ND)
# print(np.isclose(da0_dx, lwa_classical_acc_ND, RTOL, ATOL))
assert(np.linalg.norm(da0_dx - lwa_classical_acc_ND) <1e-3)


# print("world jac : \n", oJf)
# print("lwa jac : \n", lwaJf)







# If we want to get the same results as in dam2.py, need to compute joint acc and force with forwardDynamics(Jc,a0)
# and then update RNEA derivatives using joint acc, f_ext . It should be ckecked that in this case the classical acc is 0 
# since the constraint is resolved 
frameId = contactFrameId
model = robot.model
data = robot.model.createData()
pin.computeAllTerms(model, data, q0, v0)
R = data.oMf[frameId].rotation
# Compute forward dynamics with drift obtained from contact model 
    # Drift 
a0 = contactCalc2bis(model, data, frameId, x0, np.zeros(nq), PIN_REFERENCE_FRAME)
if(PIN_REFERENCE_FRAME == pin.LOCAL):
    fJf = pin.getFrameJacobian(model, data, frameId, pin.LOCAL)
else:
    fJf = pin.getFrameJacobian(model, data, frameId, pin.LOCAL_WORLD_ALIGNED)
pin.forwardDynamics(model, data, tau, fJf[:3,:], a0)
if(PIN_REFERENCE_FRAME == pin.LOCAL):
    f = data.lambda_c
else:
    f = R.T @ data.lambda_c
    # get force at joint level
fext = [pin.Force.Zero() for _ in range(model.njoints)]
fext[model.frames[frameId].parent] = model.frames[frameId].placement.act(pin.Force(f, np.zeros(3)))
    # Get derivatives
pin.computeForwardKinematicsDerivatives(model, data, q0, v0, data.ddq) 
pin.computeRNEADerivatives(model,data, q0, v0, data.ddq, fext)
    # Check constraint acc = 0
assert(np.linalg.norm(pin.getFrameClassicalAcceleration(model, data, frameId, pin.LOCAL).linear) <= 1e-6 )
da0_dx_ND = numdiff(lambda x_:contactCalc2bis(model, data, contactFrameId, x_, data.ddq, PIN_REFERENCE_FRAME), x0)
da0_dx = contactCalcDiff2Bis(model, data, contactFrameId, x0, PIN_REFERENCE_FRAME)
test_da0_dx    = np.allclose(da0_dx, da0_dx_ND, RTOL, ATOL)
print(testcolormap[test_da0_dx] + "   -- Test da0_dx ND with constraint : " + str(test_da0_dx) + bcolors.ENDC)



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
        J = pin.getFrameJacobian(model, data, frameId, pin.LOCAL_WORLD_ALIGNED)[:3,:]
        assert(np.linalg.norm(a0 - pin.getFrameClassicalAcceleration(model,data,frameId,pin.LOCAL_WORLD_ALIGNED).linear)<1e-6)
        assert(np.linalg.norm(J - data.oMf[frameId].rotation @ pin.getFrameJacobian(model, data, frameId, pin.LOCAL)[:3,:])<1e-6)
    else:
        J = pin.getFrameJacobian(model, data, frameId, pin.LOCAL)[:3,:]
        assert(np.linalg.norm(a0 - pin.getFrameClassicalAcceleration(model,data,frameId,pin.LOCAL).linear)<1e-6)
    # print("fdyn2bis.a0 = ", a0)
    # Call forward dynamics to get joint acc and force ( in local or lwa frame ) ok     identical to ground truth
    pin.forwardDynamics(model, data, tau, J, a0)
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
    df_dx = np.zeros((3,nx))
    R = data.oMf[frameId].rotation
    # Compute RNEA derivatives and KKT inverse using force and joint acc computed in fwdDyn
        # here we need LOCAL
    if(ref == pin.LOCAL):
        f = data.lambda_c
        J = pin.getFrameJacobian(model, data, frameId, pin.LOCAL)[:3,:]  
    else:
        f = R.T @ data.lambda_c 
        J = pin.getFrameJacobian(model, data, frameId, pin.LOCAL_WORLD_ALIGNED)[:3,:]  
    fext = [pin.Force.Zero() for _ in range(model.njoints)]
    fext[model.frames[frameId].parent] = model.frames[frameId].placement.act(pin.Force(f, np.zeros(3)))
    pin.computeRNEADerivatives(model, data, q, v, data.ddq, fext) 
    drnea_dx = np.hstack([data.dtau_dq, data.dtau_dv])
    Kinv = pin.getKKTContactDynamicMatrixInverse(model, data, J)  
    # Contact derivatives 
    pin.computeForwardKinematicsDerivatives(model, data, q0, v0, data.ddq) 
    da0_dx = contactCalcDiff2Bis(model, data, frameId, x, ref)
    # print(da0_dx)
    # Check KKT inverse 
    K = np.block([ [data.M,J.T],[J,np.zeros([3,3])] ])
    assert(np.linalg.norm(np.linalg.inv(K) - Kinv) < 1e-6)
    # print("dtau_dq before= \n", drnea_dx[:,:nv] )
    # correct rnea derivatives
    if(ref == pin.WORLD or ref == pin.LOCAL_WORLD_ALIGNED):
        lJ = pin.getFrameJacobian(model, data, frameId, pin.LOCAL) 
        # print("skew term = \n", lJ[:3,:].T @ pin.skew(R.T @ data.lambda_c) @ oJ[3:,:])
        drnea_dx[:,:nv] -= lJ[:3,:].T @ pin.skew(R.T @ data.lambda_c) @ lJ[3:,:]
    # print("ddq = \n", data.ddq)
    # print("fext = \n", fext)
    # print("dtau_dq = \n", drnea_dx[:,:nv] )
    # # print("oRf @ lambda_c = \n", R @ data.lambda_c)
    # print("skew term = \n", -lJ[:3,:].T @ pin.skew(R.T @ data.lambda_c) @ lJ[3:,:])
    
    
    # # Check rnea' and a'
    # dy_dx = numdiff(lambda x_:fdyn(model, data, frameId, x_, tau, PIN_REFERENCE_FRAME), x)
    # a0 = pin.getFrameClassicalAcceleration(model,data,frameId, PIN_REFERENCE_FRAME).linear
    # print(a0)
    # k = np.concatenate([ tau - data.nle, -a0 ])
    # dk_dx = np.vstack([drnea_dx, da0_dx])
    # y = -np.linalg.inv(K) @ k
    #     # check a'
    # print(np.isclose(-K@dy_dx, np.vstack([drnea_dx, da0_dx]), RTOL, ATOL))
    # da0_dx_truth = (-K @ dy_dx)[-nc:,:]
    # print("truth = \n", da0_dx_truth)
    # print(da0_dx)
    # print(np.isclose(da0_dx_truth, da0_dx, RTOL, ATOL))
    # assert(np.linalg.norm(da0_dx_truth -  da0_dx) < 1e-4) 
    #     #  Check rnea'
    # drnea_dx_truth = (-K @ dy_dx)[:nv,:]
    # print('after = \n', drnea_dx)
    # print('truth = \n', drnea_dx_truth)
    # print(np.isclose(drnea_dx_truth, drnea_dx, RTOL, ATOL))
    # print(np.isclose(da0_dx_truth, da0_dx, RTOL, ATOL))
    # # assert(np.linalg.norm(drnea_dx -  drnea_dx_before) < 1e-3) 
    # # assert(np.linalg.norm(drnea_dx_truth -  drnea_dx) < 1e-3)


    # Fillout partials of DAM 
    a_partial_dtau = Kinv[:nv, :nv]
    a_partial_da   = Kinv[:nv, -nc:]     
    f_partial_dtau = Kinv[-nc:, :nv]
    f_partial_da   = Kinv[-nc:, -nc:]
    
    # print("a_partial_dtau = \n", a_partial_dtau)
    # print("a_partial_da = \n", a_partial_da)
    # print("f_partial_dtau = \n", f_partial_dtau)
    # print("f_partial_da = \n", f_partial_da)

    Fx = -a_partial_dtau @ drnea_dx 
    # Fx[:,:nv] = -a_partial_dtau @ data.dtau_dq
    # Fx[:,nv:] = -a_partial_dtau @ data.dtau_dv
    Fx -= a_partial_da @ da0_dx[:nc] 
    Fx += a_partial_dtau @ np.zeros((nq,nx))
    # Fu = a_partial_dtau @ np.eye(nq)
    df_dx[:nc,:] = f_partial_dtau @ drnea_dx
    # df_dx[:nc, :nv]  = f_partial_dtau @ data.dtau_dq
    # df_dx[:nc, -nv:] = f_partial_dtau @ data.dtau_dv
    df_dx[:nc, :]   += f_partial_da @ da0_dx[:nc] 
    df_dx[:nc, :]   -= f_partial_dtau @ np.zeros((nq,nx))
    # df_du[:nc, :]  = -f_partial_dtau @ np.eye(nq)

    # print("Fx = \n", np.vstack([Fx, -df_dx]))
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

np.hstack([data.ddq, -data.lambda_c]) 
daf_dx_ND_0 = numdiff(lambda x_:fdyn(model, data, frameId, x_, tau, PIN_REFERENCE_FRAME), x0)
daf_dx_ND_2bis = numdiff(lambda x_:fdyn2bis(model, data, frameId, x_, tau, PIN_REFERENCE_FRAME), x0)
daf_dx_2bis    = fdyn_diff2bis(model, data, frameId, x0, tau, PIN_REFERENCE_FRAME)

test_daf_dx_0    = np.allclose(daf_dx_ND_0, daf_dx_ND_2bis, RTOL, ATOL)
print(testcolormap[test_daf_dx_0] + "   -- Test fdyn_ND vs fdyn2bis_ND : " + str(test_daf_dx_0) + bcolors.ENDC)

test_daf_dx_2bis    = np.allclose(daf_dx_2bis, daf_dx_ND_2bis, RTOL, ATOL)
print(testcolormap[test_daf_dx_2bis] + "   -- Test fdyn_diff2bis vs fdyn2bis_ND : " + str(test_daf_dx_2bis) + bcolors.ENDC)

# print(np.isclose(daf_dx_ND_2bis, daf_dx_2bis, RTOL, ATOL))
# print(daf_dx_ND_2bis)
# print(daf_dx_2bis)





# Calc a0
frameId = contactFrameId
model = robot.model
data = robot.model.createData()


# forward dynamics in LOCAL 
pin.computeAllTerms(model, data, q0, v0)
pin.forwardKinematics(model, data, q0, v0, np.zeros(nq))
pin.updateFramePlacements(model, data)
la0 = pin.getFrameClassicalAcceleration(model, data, frameId, pin.LOCAL).linear
lJ = pin.getFrameJacobian(model, data, frameId, pin.LOCAL)
print("LOCAL a0 = \n", la0)
print("LOCAL Jc = \n", lJ[:3])
pin.forwardDynamics(model, data, tau, lJ[:3,:], la0)
lK = np.block([ [data.M, lJ[:3,:].T],[lJ[:3,:], np.zeros([3,3])] ])
laf = np.linalg.inv(lK) @ np.concatenate([tau - data.nle, -la0])
assert(np.linalg.norm(laf[:nv] - data.ddq) <1e-6)
assert(np.linalg.norm(laf[-nc:] + data.lambda_c) <1e-6)
# Derivatives in LOCAL
fext = [pin.Force.Zero() for _ in range(model.njoints)]
fext[model.frames[frameId].parent] = model.frames[frameId].placement.act(pin.Force(data.lambda_c, np.zeros(3)))
pin.computeRNEADerivatives(model, data, q0, v0, data.ddq, fext) 
ldrnea_dx = np.hstack([data.dtau_dq, data.dtau_dv])
print("LOCAL data.dtau_dq = \n", data.dtau_dq)
# print("local rnea 1 :  \n", ldrnea_dx)
lKinv = pin.getKKTContactDynamicMatrixInverse(model, data, lJ[:3,:])  
assert(np.linalg.norm(lKinv - np.linalg.inv(lK)) <1e-6)
assert(np.linalg.norm(fdyn(model, data, frameId, x0, tau, pin.LOCAL) - laf) < 1e-6)
ldaf_dx_ND = numdiff(lambda x_:fdyn(model, data, frameId, x_, tau, pin.LOCAL), x0)
ldk_dx = -lK @ ldaf_dx_ND
    # Frame acceleration derivative
v_partial_dq, a_partial_dq, a_partial_dv, a_partial_da = pin.getFrameAccelerationDerivatives(model, data, frameId, pin.LOCAL) 
v = pin.getFrameVelocity(model, data, frameId, pin.LOCAL)
vv = v.linear ; vw = v.angular
lda0_dx = np.zeros((nc,nx))
lda0_dx[:,:nv] = a_partial_dq[:3,:]
lda0_dx[:,:nv] += pin.skew(vw) @ v_partial_dq[:3,:]
lda0_dx[:,:nv] -= pin.skew(vv) @ v_partial_dq[3:,:]
lda0_dx[:,nv:] = a_partial_dv[:3,:]
lda0_dx[:,nv:] += pin.skew(vw) @ lJ[:3,:]
lda0_dx[:,nv:] -= pin.skew(vv) @ lJ[3:,:]
assert(np.linalg.norm(lda0_dx - contactCalcDiff2Bis(model, data, frameId, x0, pin.LOCAL))<1e-4)
assert(np.linalg.norm(ldk_dx[-nc:] - lda0_dx) <1e-4)
assert(np.linalg.norm(ldk_dx[:nv] - ldrnea_dx) <1e-4)
assert(np.linalg.norm(ldaf_dx_ND + lKinv @ np.concatenate([ldrnea_dx, lda0_dx])) <1e-3)



# forward dynamics in WORLD
pin.computeAllTerms(model, data, q0, v0)
pin.forwardKinematics(model, data, q0, v0, np.zeros(nq))
pin.updateFramePlacements(model, data)
R = data.oMf[frameId].rotation
wa0 = pin.getFrameClassicalAcceleration(model, data, frameId, pin.LOCAL_WORLD_ALIGNED).linear
wJf = pin.getFrameJacobian(model, data, frameId, pin.LOCAL_WORLD_ALIGNED)[:3,:]
# print("WORLD a0 = \n", wa0)
# print("WORLD Jc = \n", wJf)
pin.forwardDynamics(model, data, tau, wJf, wa0)
wK = np.block([ [data.M, wJf.T],[wJf, np.zeros([3,3])] ])
waf = np.linalg.inv(wK) @ np.concatenate([tau - data.nle, -wa0])
print("WORLD aq = \n", waf[:nv])
print("WORLD force = \n", waf[-nc:])
Kinv = np.linalg.inv(wK)
a_partial_dtau = Kinv[:nv, :nv]
a_partial_da   = Kinv[:nv, -nc:]     
f_partial_dtau = Kinv[-nc:, :nv]
f_partial_da   = Kinv[-nc:, -nc:]
# print("WORLD Kinv = \n", np.linalg.inv(wK))
print("WORLD a_partial_dtau = \n", a_partial_dtau)
print("WORLD a_partial_da = \n", a_partial_da)
print("WORLD f_partial_dtau = \n", f_partial_dtau)
print("WORLD f_partial_da = \n", f_partial_da)
assert(np.linalg.norm(waf[:nv] - data.ddq) <1e-6)
assert(np.linalg.norm(waf[:nv] - laf[:nv] ) < 1e-6)
assert(np.linalg.norm(waf[-nc:] + data.lambda_c) <1e-6)
# Derivatives in WORLD
fext = [pin.Force.Zero() for _ in range(model.njoints)]
fext[model.frames[frameId].parent] = model.frames[frameId].placement.act(pin.Force(R.T @ data.lambda_c, np.zeros(3)))
pin.computeRNEADerivatives(model, data, q0, v0, data.ddq, fext) 
print(q0, v0, data.ddq, fext)
wdrnea_dx = np.hstack([data.dtau_dq, data.dtau_dv])
# print("WORLD data.dtau_dq before = \n", data.dtau_dq)
assert(np.linalg.norm(wdrnea_dx -ldrnea_dx) <1e-4)
    # additional term  
lJ = pin.getFrameJacobian(model, data, frameId, pin.LOCAL) 
wdrnea_dx[:,:nv] -= lJ[:3].T @ pin.skew(R.T @ data.lambda_c) @ lJ[3:]
# print("WORLD data.dtau_dq after = \n", wdrnea_dx[:,:nv])
wKinv = pin.getKKTContactDynamicMatrixInverse(model, data, wJf)  
assert(np.linalg.norm(wKinv - np.linalg.inv(wK))<1e-6)
assert(np.linalg.norm(fdyn(model, data, frameId, x0, tau, pin.LOCAL_WORLD_ALIGNED) - waf) < 1e-6)
wdaf_dx_ND = numdiff(lambda x_:fdyn(model, data, frameId, x_, tau, pin.LOCAL_WORLD_ALIGNED), x0)
wdk_dx = -wK @ wdaf_dx_ND
    # Frame acc derivative 
wda0_dx = R @ lda0_dx
Jw = pin.getFrameJacobian(model, data, frameId, pin.LOCAL_WORLD_ALIGNED)[3:,:]
pin.computeForwardKinematicsDerivatives(model, data, q0, v0, data.ddq) # very important 
    # print("linear acc after fwdDyn : ", pin.getFrameClassicalAcceleration(model, data, frameId, pin.LOCAL_WORLD_ALIGNED).linear)
wa = pin.getFrameClassicalAcceleration(model, data, frameId, pin.LOCAL_WORLD_ALIGNED).linear
assert(np.linalg.norm(wa) <1e-6)
# wda0_dx[:,:nq] -= pin.skew(wa)@Jw # no skew term after because acc = 0
# print("wkew term = ",  pin.skew(wa)@Jw)
print("WORLD da0_dx = \n", wda0_dx)
assert(np.linalg.norm(wda0_dx - contactCalcDiff2Bis(model, data, frameId, x0, pin.LOCAL_WORLD_ALIGNED) ) <1e-6 )
assert(np.linalg.norm(wdk_dx[-nc:] - wda0_dx) <1e-4)
    # rnea derivatives
assert(np.linalg.norm(wdk_dx[:nv] - wdrnea_dx)<1e-4)
    # all
wdaf_dx = -wKinv @ np.vstack([wdrnea_dx, wda0_dx])
assert(np.linalg.norm(wdaf_dx_ND - wdaf_dx) <1e-3)

# print(wdrnea_dx)
assert(np.linalg.norm(wdaf_dx[:nq] - ldaf_dx_ND[:nq]) <1e-3)

# print(wdaf_dx[:nq])
# print(ldaf_dx_ND[:nq])
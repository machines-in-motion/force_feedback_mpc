import sys

sys.path.append('.')

import numpy as np
np.set_printoptions(precision=6, linewidth=180, suppress=True)
np.random.seed(1)
import crocoddyl
import pinocchio as pin

from core_mpc.pin_utils import load_robot_wrapper
# from classical_mpc.data import DDPDataHandlerClassical
# from core_mpc import pin_utils


REF_FRAME = pin.LOCAL

robot = load_robot_wrapper('iiwa')
model = robot.model ; data = model.createData()
nq = model.nq; nv = model.nv; nu = nq; nx = nq+nv
q0 = np.array([0.1, 0.7, 0., 0.7, -0.5, 1.5, 0.])
v0 = np.zeros(nv) #np.random.rand(nv)
x0 = np.concatenate([q0, v0])
pin.computeAllTerms(robot.model, robot.data, q0, v0)
pin.forwardKinematics(model, data, q0, v0, np.zeros(nq))
pin.updateFramePlacements(model, data)
robot.computeJointJacobians(q0)
frameId = model.getFrameId('contact')

# initial ee position and contact anchor point
oPf = data.oMf[frameId].translation
oRf = data.oMf[frameId].rotation
oPc = oPf + np.array([0.05,.0, 0]) # + cm in x world np.random.rand(3)
print("initial EE position (WORLD) = \n", oPf)
print("anchor point (WORLD)        = \n", oPc)
ov = pin.getFrameVelocity(model, data, frameId, pin.WORLD).linear
print("initial EE velocity (WORLD) = \n", ov)
# contact gains
Kp = 1000
Kv = 2*np.sqrt(Kp)
print("stiffness = ", Kp)
print("damping   = ", Kv)
# initial force in WORLD + at joint level
of0 = -Kp*(oPf - oPc) - Kv*ov
lf0 = oRf.T @ of0
fext0 = [pin.Force.Zero() for _ in range(model.njoints)]
fext0[model.frames[frameId].parent] = model.frames[frameId].placement.act(pin.Force(lf0, np.zeros(3)))
print("initial force (WORLD) = \n", of0)
print("initial force (LOCAL) = \n", lf0)

# State and actuation model
state = crocoddyl.StateMultibody(model)
actuation = crocoddyl.ActuationModelFull(state)

# Running and terminal cost models
runningCostModel = crocoddyl.CostModelSum(state)
terminalCostModel = crocoddyl.CostModelSum(state)


# Create cost terms 
  # Control regularization cost
uref = np.zeros(nq) #pin_utils.get_tau(q0, np.zeros(nv), np.zeros(nq), fext0, model, np.zeros(nq)) #np.random.rand(nq) 
uResidual = crocoddyl.ResidualModelControl(state, uref)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)
  # State regularization cost
xResidual = crocoddyl.ResidualModelState(state, np.concatenate([q0, np.zeros(nv)]))
xRegCost = crocoddyl.CostModelResidual(state, xResidual)
endeff_frame_id = model.getFrameId("contact")
  # endeff_translation = robot.data.oMf[endeff_frame_id].translation.copy()
endeff_translation = oPc #np.array([-0.4, 0.3, 0.7]) # move endeff +10 cm along x in WORLD frame
frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(state, endeff_frame_id, endeff_translation)
frameTranslationCost = crocoddyl.CostModelResidual(state, frameTranslationResidual)
  # frame velocity 
frameVelocityResidual = crocoddyl.ResidualModelFrameVelocity(state, endeff_frame_id, pin.Motion.Zero(), pin.WORLD)
frameVelocityCost = crocoddyl.CostModelResidual(state, frameVelocityResidual)
  # Populate cost model 
runningCostModel.addCost("stateReg", xRegCost, 1e-2)
runningCostModel.addCost("ctrlReg", uRegCost, 1e-4)
# runningCostModel.addCost("stateReg", xRegCost, 1e-2)
# runningCostModel.addCost("ctrlReg", uRegCost, 1e-4)
runningCostModel.addCost("translation", frameTranslationCost, 1e-1)
terminalCostModel.addCost("stateReg", xRegCost, 1e-2)
terminalCostModel.addCost("translation", frameTranslationCost, 1e-1)
# terminalCostModel.addCost("velocity", frameVelocityCost, 1)




def check(A, B, tol=1e-6):
    assert(np.linalg.norm(A - B) < tol)




# Python model
from soft_models_3D import DAMSoftContactDynamics3D
dam_py = DAMSoftContactDynamics3D(state, actuation, runningCostModel, frameId, Kp, Kv, oPc, pinRefFrame=REF_FRAME)
dam_py.set_force_cost(np.array([0.,0.,0.]), 1e-2)
dad_py = dam_py.createData()
# C++ model (bindings)
import sobec
dam_cpp = sobec.sobec_pywrap.DifferentialActionModelSoftContact3DFwdDynamics(state, actuation, runningCostModel, frameId, Kp, Kv, oPc, REF_FRAME)
dam_cpp.set_force_cost(np.array([0.,0.,0.]), 1e-2)
dad_cpp = dam_cpp.createData()
# Check basic parameters are the same
check(dam_py.f_des, dam_cpp.f_des)
check(dam_py.oPc, dam_cpp.oPc)
check(dam_py.Kp, dam_cpp.Kp)
check(dam_py.Kv, dam_cpp.Kv)
check(dam_py.f_weight, dam_cpp.f_weight)
check(dam_py.frameId, dam_cpp.id)
check(dam_py.pinRef, dam_cpp.ref)


# Create free model 
dam_free = crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, runningCostModel)
dad_free = dam_free.createData()

# Create numdiff model 
dam_cpp_nd = crocoddyl.DifferentialActionModelNumDiff(dam_cpp, True)
dad_cpp_nd = dam_cpp_nd.createData()
np.set_printoptions(precision=2)

dam_py.calc(dad_py, x0, uref)
dam_cpp.calc(dad_cpp, x0, uref)
dam_cpp_nd.calc(dad_cpp_nd, x0, uref)
print("py calc = \n", dad_py.xout, "\n", dad_py.cost)
print("cpp calc = \n", dad_cpp.xout, "\n", dad_cpp.cost)
print("cpp ND calc = \n", dad_cpp_nd.xout, "\n", dad_cpp_nd.cost)

dam_py.calcDiff(dad_py, x0, uref)
dam_cpp.calcDiff(dad_cpp, x0, uref)
dam_cpp_nd.calcDiff(dad_cpp_nd, x0, uref)

print("py calcDiff Fx = \n", dad_py.Fx, "\n", dad_py.Fx)
print("cpp calcDiff Fx = \n", dad_cpp.Fx, "\n", dad_cpp.Fx)
print("cpp ND calcDiff Fx = \n", dad_cpp_nd.Fx, "\n", dad_cpp_nd.Fx)


print("py calcDiff Lx = \n",         dad_py.Lx, "\n",     dad_py.Lx)
print("cpp calcDiff Lx = \n",       dad_cpp.Lx, "\n",    dad_cpp.Lx)
print("cpp ND calcDiff Lx = \n", dad_cpp_nd.Lx, "\n", dad_cpp_nd.Lx)



# check(dad_py.xout, dad_cpp.xout)
# check(dad_py.cost, dad_cpp.cost)
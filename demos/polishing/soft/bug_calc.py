
import crocoddyl
import pinocchio as pin
import numpy as np
from mim_robots.robot_loader import load_pinocchio_wrapper
import force_feedback_mpc

robot = load_pinocchio_wrapper('iiwa', locked_joints=['A7'])

# minimal croco problem
nx = robot.model.nq + robot.model.nv
state = crocoddyl.StateMultibody(robot.model)
actuation = crocoddyl.ActuationModelFull(state)
costModel = crocoddyl.CostModelSum(state, actuation.nu)
frameId = 20
Kp = np.ones(1)
Kv = np.ones(1)
oPc = np.ones(3)
ref = pin.LOCAL
mask = force_feedback_mpc.Vector3MaskType.z

costUreg = crocoddyl.ResidualModelControl(state, actuation.nu)
costXreg = crocoddyl.ResidualModelState(state, np.zeros(nx), actuation.nu)
costModel.addCost("ureg", crocoddyl.CostModelResidual(state, costUreg), 0.1)
costModel.addCost("xreg", crocoddyl.CostModelResidual(state, costXreg), 0.1)
dam = force_feedback_mpc.DAMSoftContact1DAugmentedFwdDynamics(state, actuation, costModel, frameId, Kp, Kv, oPc, ref, mask)

dt=0.1
iam = force_feedback_mpc.IAMSoftContactAugmented(dam , dt)

x0 = np.zeros(nx)
f0 = np.zeros(1)
u0 = np.zeros(actuation.nu)

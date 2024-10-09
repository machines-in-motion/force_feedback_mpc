
import crocoddyl
import pinocchio as pin
import numpy as np
from mim_robots.robot_loader import load_pinocchio_wrapper
import force_feedback_mpc


robot_name = 'iiwa'

robot = load_pinocchio_wrapper(robot_name) #, locked_joints=['A7'])
print("Loaded ", robot_name)

DAM_TYPE = '1D' # '3D'

print("Construct OCP with DAM_"+str(DAM_TYPE))

# minimal croco problem
nx = robot.model.nq + robot.model.nv
state = crocoddyl.StateMultibody(robot.model)
actuation = crocoddyl.ActuationModelFull(state)
costModel = crocoddyl.CostModelSum(state, actuation.nu)
costUreg = crocoddyl.ResidualModelControl(state, actuation.nu)
costXreg = crocoddyl.ResidualModelState(state, np.zeros(nx), actuation.nu)
costModel.addCost("ureg", crocoddyl.CostModelResidual(state, costUreg), 0.1)
costModel.addCost("xreg", crocoddyl.CostModelResidual(state, costXreg), 0.1)
frameId = robot.model.getFrameId('contact')
# instantiate custom DAM
oPc = np.ones(3)
if(DAM_TYPE == '1D'):
    nc = 1
    Kp = np.ones(1)
    Kv = np.ones(1)
    mask = force_feedback_mpc.Vector3MaskType.z
    DAM = force_feedback_mpc.DAMSoftContact1DAugmentedFwdDynamics(state, actuation, costModel, frameId, Kp, Kv, oPc, mask)
if(DAM_TYPE == '3D'):
    nc = 3
    Kp = np.ones(3)
    Kv = np.ones(3)
    DAM = force_feedback_mpc.DAMSoftContact3DAugmentedFwdDynamics(state, actuation, costModel, frameId, Kp, Kv, oPc)

print("Created DAM "+str(nc)+"D")

# dummy state
x0 = np.zeros(nx)
f0 = np.zeros(2)
y0 = np.concatenate([x0, f0])
u0 = np.zeros(actuation.nu)

print('x0 = \n', x0)
print('f0 = \n', f0)
print('y0 = \n', y0)
print('u0 = \n', u0)

# calc DAM
DAD = DAM.createData()
# DAD = crocoddyl.DifferentialActionDataAbstract(DAM)
DAM.calc(DAD, x0, f0, u0)

# # custom IAM
# dt=0.1
# iam = force_feedback_mpc.IAMSoftContactAugmented(DAM , dt)
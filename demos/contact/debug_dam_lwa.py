"""
@package force_feedback
@file debug_baumgarte_lwa.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft.
@date 2020-05-18
@brief OCP for normal force task
"""

'''
The robot is tasked with exerting a constant normal force at its EE
Trajectory optimization using Crocoddyl
The goal of this script is to setup OCP (a.k.a. play with weights)
'''


import sys
sys.path.append('.')

from core_mpc_utils.misc_utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger


# import logging
# FORMAT_SHORT  = '[%(levelname)s] %(name)s : %(message)s'
# logging.basicConfig(format=FORMAT_SHORT)
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

import numpy as np  
np.set_printoptions(precision=6, linewidth=180, suppress=True)

import crocoddyl
import pinocchio as pin
import sobec

WITH_COSTS      = False
ND_DISTURBANCE  = 1e-6
GAUSS_APPROX    = False
RTOL            = 1e-3 #1e-3
ATOL            = 1e-4 #1e-5
RANDOM_SEED     = 1
np.random.seed(RANDOM_SEED)
CONTACT_FRAME   = pin.LOCAL


# Load robot 
# from example_robot_data import loadTalosArm 
# robot = loadTalosArm()
# robot = pin_utils.load_robot_wrapper('talos_arm')
from core_mpc_utils import pin_utils
robot = pin_utils.load_robot_wrapper('talos_arm')
nq = robot.model.nq; nv = robot.model.nv; nu = nq; nx = nq+nv
q0 = np.random.rand(nq) 
v0 = np.random.rand(nv) #np.zeros(nq)  #
x0 = np.concatenate([q0, v0])
robot.framesForwardKinematics(q0)
robot.computeJointJacobians(q0)
tau = np.random.rand(nq)
logger.info("tau random = "+str(tau))


# # Add a custom frame aligned with WORLD to have oRf = identity
# parent_frame_id = robot.model.getFrameId("gripper_left_fingertip_1_link")
# parent_frame = robot.model.frames[parent_frame_id]
# j_p_f = parent_frame.placement.translation
# W_M_j = robot.data.oMi[parent_frame.parent]
# W_p_c = W_M_j.act(j_p_f) # center coincides with previous frame
# W_R_c = np.eye(3)        # axes aligned with WORLD axes
# W_M_c = pin.SE3(W_R_c, W_p_c)
# # Add a frame
# customFrame = pin.Frame('contact_frame', parent_frame.parent, parent_frame_id, W_M_j.actInv(W_M_c), pin.OP_FRAME)
# robot.model.addFrame(customFrame)
# contact_frame_name = customFrame.name 
# contact_frame_id = robot.model.getFrameId(contact_frame_name)
# # Update data
# robot.data = robot.model.createData() #new data (with +1 frame)
# robot.framesForwardKinematics(q0)
# robot.computeJointJacobians(q0)


# State and actuation model
state = crocoddyl.StateMultibody(robot.model)
actuation = crocoddyl.ActuationModelFull(state)
# Running and terminal cost models
runningCostModel = crocoddyl.CostModelSum(state)
# terminalCostModel = crocoddyl.CostModelSum(state)
# Contact model 
contactModel = sobec.ContactModelMultiple(state, actuation.nu)
    # Create 3D contact on the en-effector frame
contact_frame_name = "gripper_left_fingertip_1_link" #'gripper_right_fingertip_1_link' #"arm_right_7_link" 
contact_frame_id = robot.model.getFrameId(contact_frame_name)
parent_frame_id = robot.model.frames[contact_frame_id].parent
contact_position = robot.data.oMf[contact_frame_id].translation.copy()
baumgarte_gains  = np.array([0., 0])
contact3d = sobec.ContactModel3D(state, contact_frame_id, contact_position, baumgarte_gains, CONTACT_FRAME) 
nc = 3
    # Populate contact model with contacts
contactModel.addContact("contact_"+contact_frame_name, contact3d, active=True)


# f_ext = [pin.Force.Zero() for i in range(robot.model.njoints)]
# pin.computeAllTerms(robot.model, robot.data, q0, v0)
# J = pin.getFrameJacobian(robot.model, robot.data, contact_frame_id, pin.LOCAL)[:3,:]
# gamma = -pin.getFrameClassicalAcceleration(robot.model, robot.data, contact_frame_id, pin.LOCAL)
# aq    = np.linalg.pinv(J) @ gamma.vector[:3]
# tau   = pin.rnea(robot.model, robot.data, q0, v0, aq, f_ext)
# print("tau such that f=0 : RNEA(q,vq,aq=J^+ a0, f_ext=0)", tau)

# # Check that force is zero using the above torque
# nle     = pin.nonLinearEffects(robot.model, robot.data, q0, v0)
# Minv = pin.computeMinverse(robot.model, robot.data, q0) 
# # f = (JMiJ')^+ ( JMi (b-tau) + gamma )
# import eigenpy
# LDLT = eigenpy.LDLT(J @ Minv @ J.T)
# f =  LDLT.solve(J @ Minv @ (nle - tau) + gamma.vector[:3])
# # print("FORCE = ", pin_utils.get_f_(q0, v0, tau, robot.model, contact_frame_id, np.zeros(nq), REG=0))
# print("ee force : fwd_dyn( tau=NLE(q,v), J*aq )= \n", f)
# print("")
# # Joint forces corresponding to the wrench 
# wrench = np.hstack([f,np.zeros(3)])
# f_ext2 = pin_utils.get_external_joint_torques(robot.data.oMf[contact_frame_id], wrench, robot)
# print("f_ext2 = \n", f_ext2)
# print("rnea(q,vq,aq,f_ext2) = \n", pin.rnea(robot.model, robot.data, q0, v0, aq, f_ext2))



# Create Differential Action Model (DAM), i.e. continuous dynamics and cost functions
enable_force = True
DAM    = sobec.DifferentialActionModelContactFwdDynamics(state, actuation, contactModel, runningCostModel, inv_damping=0, enable_force=enable_force)
DAM_ND = crocoddyl.DifferentialActionModelNumDiff(DAM, GAUSS_APPROX)
DAD    = DAM.createData()
DAD_ND = DAM_ND.createData()
DAM_ND.disturbance = ND_DISTURBANCE


# Allocate new data to test model against python 
model         = robot.model.copy()
data          = model.createData()
contactData   = contactModel.createData(data)
actuationData = actuation.createData()
costData      = runningCostModel.createData(crocoddyl.DataCollectorAbstract())


# # Check contact model 
# pin.computeAllTerms(model, data, q0, v0)
# logger.debug("--- TEST CONTACT JAC AND DRIFT ---")
# logger.debug("   -- jacobian LOCAL-->WORLD_ALIGNED transformation --")
# pin.updateFramePlacement(model, data, contact_frame_id)
# fJf = pin.getFrameJacobian(model, data, contact_frame_id, pin.LOCAL)
# oJf = pin.getFrameJacobian(model, data, contact_frame_id, pin.LOCAL_WORLD_ALIGNED)
# oRf = data.oMf[contact_frame_id].rotation
# logger.debug(np.allclose(oJf[:3], oRf @ fJf[:3], RTOL, ATOL))
# logger.debug(np.allclose(oJf[3:], oRf @ fJf[3:], RTOL, ATOL))

# logger.debug("   -- drift LOCAL-->LOCAL_WORLD_ALIGNED transformation --")
# a0_L = pin.getFrameClassicalAcceleration(model, data, contact_frame_id, pin.LOCAL).linear
# a0_W = pin.getFrameClassicalAcceleration(model, data, contact_frame_id, pin.LOCAL_WORLD_ALIGNED).linear
# logger.debug(np.allclose(a0_W, oRf @ a0_L, RTOL, ATOL))

# logger.debug("   -- drift classical vs spatial (LOCAL) --")
# v_L = pin.getFrameVelocity(model, data, contact_frame_id, pin.LOCAL)
# a_L = pin.getFrameAcceleration(model, data, contact_frame_id, pin.LOCAL)
# logger.debug(np.allclose(a0_L, a_L.linear + np.cross(v_L.angular, v_L.linear), RTOL, ATOL))

# logger.debug("   -- drift classical vs spatial (LWA) --")
# v_W = pin.getFrameVelocity(model, data, contact_frame_id, pin.LOCAL_WORLD_ALIGNED)
# a_W = pin.getFrameAcceleration(model, data, contact_frame_id, pin.LOCAL_WORLD_ALIGNED)
# logger.debug(np.allclose(a0_W, a_W.linear + np.cross(v_W.angular, v_W.linear), RTOL, ATOL))


oRf = data.oMf[contact_frame_id].rotation

# calc versus ND
DAM.calc(DAD, x0, tau)
DAM_ND.calc(DAD_ND, x0, tau)
logger.debug("--- TEST CALC FUNCTION ---")
logger.debug("   -- xout (model vs numdiff) --")
# logger.info("MODEL.xout   : "+str(DAD.xout))
# logger.info("NUMDIFF.xout : "+str(DAD_ND.xout))
logger.debug(np.allclose(DAD.xout, DAD_ND.xout, RTOL, ATOL))

# Compute calc in python
pin.computeAllTerms(model, data, q0, v0)
pin.computeCentroidalMomentum(model, data)
actuation.calc(actuationData, x0, tau)
contactModel.calc(contactData, x0)


# check contact forces OK equal in both cases
jMf = model.frames[contact_frame_id].placement
# WORLD ALIGNED
if(CONTACT_FRAME == pin.WORLD or CONTACT_FRAME == pin.LOCAL_WORLD_ALIGNED):
    # print('Force in WORLD_ALIGNED frame')
    pin.forwardDynamics(model, data, actuationData.tau, contactData.Jc, contactData.a0)
    # print(data.ddq)
    contactModel.updateForce(contactData, data.lambda_c)
    print("   force in WORLD_ALIGNED from pin.ForwardDynamics(tau, Jc, a0) = \n", data.lambda_c)
    f_W_joint = jMf.act(pin.Force(oRf.T @ data.lambda_c, np.zeros(3)))
    # print("   force at JOINT level  : \n", f_W_joint)
    fext = [f for f in contactData.fext]
    # print(fext)
    # print("   force at JOINT level (from fext) : \n", fext[parent_frame_id])
    logger.debug(np.allclose(f_W_joint.linear, fext[parent_frame_id].linear, RTOL, ATOL))
    logger.debug(np.allclose(f_W_joint.angular, fext[parent_frame_id].angular, RTOL, ATOL))
    logger.debug(np.allclose(f_W_joint.linear, contactData.contacts['contact_'+contact_frame_name].f.linear, RTOL, ATOL))
    logger.debug(np.allclose(f_W_joint.angular, contactData.contacts['contact_'+contact_frame_name].f.angular, RTOL, ATOL))
# LOCAL
elif(CONTACT_FRAME == pin.LOCAL):
    # print('Force in LOCAL frame')
    pin.forwardDynamics(model, data, actuationData.tau, contactData.Jc, contactData.a0)
    # print(data.ddq)
    contactModel.updateForce(contactData, data.lambda_c)
    print("force in LOCAL from pin.ForwardDynamics(tau, Jc, a0) = \n", data.lambda_c)
    f_L_joint = jMf.act(pin.Force(data.lambda_c, np.zeros(3)))
    # print("force at JOINT level  : \n", f_L_joint)
    fext = [f for f in contactData.fext]
    # print(fext)
    # print("force at JOINT level (from fext) : \n", fext[parent_frame_id])
    logger.debug(np.allclose(f_L_joint.linear, fext[parent_frame_id].linear, RTOL, ATOL))
    logger.debug(np.allclose(f_L_joint.angular, fext[parent_frame_id].angular, RTOL, ATOL))
    logger.debug(np.allclose(f_L_joint.linear, contactData.contacts['contact_'+contact_frame_name].f.linear, RTOL, ATOL))
    logger.debug(np.allclose(f_L_joint.angular, contactData.contacts['contact_'+contact_frame_name].f.angular, RTOL, ATOL))

# print(fext)
# Go on with DAM.Calc
# Jc = np.zeros((nc, nv))
# Jc[:nc, :nv] = contactData.Jc
pin.forwardDynamics(model, data, actuationData.tau, contactData.Jc, contactData.a0)
xout = data.ddq
contactModel.updateAcceleration(contactData, xout)
contactModel.updateForce(contactData, data.lambda_c)
runningCostModel.calc(costData, x0, tau)
# print(xout)
# # Compare against model 
# logger.debug("   -- a0 (model vs python) -- ")
# # logger.info("a0 :"+str(contactData.a0))
# logger.debug(np.allclose(contactData.a0, DAD.multibody.contacts.a0, RTOL, ATOL))
# if(CONTACT_FRAME == pin.WORLD or CONTACT_FRAME == pin.LOCAL_WORLD_ALIGNED):
#     logger.debug(np.allclose(contactData.a0, a0_W, RTOL, ATOL))
# elif(CONTACT_FRAME == pin.LOCAL):
#     logger.debug(np.allclose(contactData.a0, a0_L, RTOL, ATOL))

# logger.debug("   -- Jc (model vs python) --")
# logger.info("Jc = \n"+str(contactData.Jc))
# logger.debug(np.allclose(contactData.Jc, DAD.multibody.contacts.Jc, RTOL, ATOL))
# if(CONTACT_FRAME == pin.WORLD or CONTACT_FRAME == pin.LOCAL_WORLD_ALIGNED):
#     logger.debug(np.allclose(contactData.Jc, oJf[:3], RTOL, ATOL))
# elif(CONTACT_FRAME == pin.LOCAL):
#     logger.debug(np.allclose(contactData.Jc, fJf[:3], RTOL, ATOL))

# logger.debug("   -- tau (model vs python) --")
# # logger.info("tau = "+str(actuationData.tau))
# logger.debug(np.allclose(actuationData.tau, tau, RTOL, ATOL))

# logger.debug("   -- xout (model vs python) --")
# # logger.info("xout = "+str(xout))
# logger.debug(np.allclose(xout, DAD.xout, RTOL, ATOL))

# logger.debug("   -- lambda_c (model vs python) --")
# # logger.info("lambda_c = "+str(data.lambda_c))
# logger.debug(np.allclose(data.lambda_c, DAD.multibody.pinocchio.lambda_c, RTOL, ATOL))


print("\n")

# calcDiff
logger.debug("--- TEST CALCDIFF FUNCTION ---")
DAM.calcDiff(DAD, x0, tau)
DAM_ND.calcDiff(DAD_ND, x0, tau)

logger.debug("   -- Test Fu (model vs numdiff) --")
# logger.info("MODEL.Fu   :\n "+ str(DAD.Fu))
# logger.info("NUMDIFF.Fu :\n "+ str(DAD_ND.Fu))
logger.debug(np.allclose(DAD.Fu, DAD_ND.Fu, RTOL, ATOL))

logger.debug("   -- Test Fx (model vs numdiff) --")
# logger.debug("           Fx")
# logger.info("MODEL.Fx   :\n "+ str(DAD.Fx))
# logger.info("NUMDIFF.Fx :\n "+ str(DAD_ND.Fx))
logger.debug("           Fq")
logger.debug(np.allclose(DAD.Fx[:,:nq], DAD_ND.Fx[:,:nq], RTOL, ATOL))
# logger.debug("\n"+str(np.isclose(DAD.Fx[:,:nq], DAD_ND.Fx[:,:nq], RTOL, ATOL)))
logger.debug("           Fv")
logger.debug(np.allclose(DAD.Fx[:,nq:], DAD_ND.Fx[:,nq:], RTOL, ATOL))
# logger.debug("\n"+str(np.isclose(DAD.Fx[:,nq:], DAD_ND.Fx[:,nq:], RTOL, ATOL)))

# Calc vs pinocchio analytical 
# print("TAU = ", pin_utils.get_tau(q0, v0, xout, contactData.fext, model, np.zeros(nq)))
pin.computeRNEADerivatives(model, data, q0, v0, xout, contactData.fext)
Kinv = pin.getKKTContactDynamicMatrixInverse(model, data, contactData.Jc) #Jc[:nc])

# print(Kinv)
actuation.calcDiff(actuationData, x0, tau)
contactModel.calcDiff(contactData, x0) 

print("da0_dx = ", contactData.contacts['contact_'+contact_frame_name].da0_dx)


# logger.debug("   -- Test KKT (model vs python) --")
# # logger.info("PIN.KKTinv   :\n "+ str(Kinv))
# # logger.info("MODEL.KKTinv :\n "+ str(DAD.Kinv))
# logger.debug(np.allclose(Kinv, DAD.Kinv, RTOL, ATOL))
# KKT = np.zeros((nq+nc, nq+nc))
# KKT[:nq,:nq] = data.M         ; KKT[:nq,nq:] = contactData.Jc.T
# KKT[nq:,:nq] = contactData.Jc ; KKT[nq:,nq:] = np.zeros((nc,nc))
# logger.debug(np.allclose(Kinv, np.linalg.inv(KKT), RTOL, ATOL))

# logger.debug("   -- Test dtau_dq (model vs python) --")
# # logger.info("dtau_dq :\n"+str(data.dtau_dq))
# # logger.info("dtau_dq :\n"+str(DAD.multibody.pinocchio.dtau_dq))
# logger.debug(np.allclose(data.dtau_dq, DAD.multibody.pinocchio.dtau_dq, RTOL, ATOL))

# logger.debug("   -- Test dtau_dv (model vs python) --")
# # logger.info("dtau_dv :\n"+str(data.dtau_dv))
# # logger.info("dtau_dv :\n"+str(DAD.multibody.pinocchio.dtau_dv))
# logger.debug(np.allclose(data.dtau_dv, DAD.multibody.pinocchio.dtau_dv, RTOL, ATOL))

# logger.debug("   -- Test actuation.dtau_dx (model vs python) --")
# # logger.info("dtau_dx :\n"+str(actuationData.dtau_dx))
# # logger.info("dtau_dx :\n"+str(DAD.multibody.actuation.dtau_dx))
# logger.debug(np.allclose(actuationData.dtau_dx, DAD.multibody.actuation.dtau_dx, RTOL, ATOL))

# logger.debug("   -- Test actuation.dtau_du (model vs python) --")
# # logger.info("dtau_du :\n"+str(actuationData.dtau_du))
# # logger.info("dtau_du :\n"+str(DAD.multibody.actuation.dtau_du))
# logger.debug(np.allclose(actuationData.dtau_du, DAD.multibody.actuation.dtau_du, RTOL, ATOL))

# logger.debug("   -- Test contact.da0_dx (model vs python) --")
# # logger.info("da0_dx :\n"+str(contactData.da0_dx))
# # logger.info("da0_dx :\n"+str(DAD.multibody.contacts.da0_dx))
# logger.debug(np.allclose(contactData.da0_dx, DAD.multibody.contacts.da0_dx, RTOL, ATOL))

da0_dx = np.zeros((nc, nx))
# print(contactData.da0_dx )
da0_dx[:nc, :nx] = contactData.da0_dx 

# da0_dx[:nc, :nq] += pin.skew(oRf @ contactData.a0) @ oJf[3:]
# da0_dx[:nc, :nq] = oRf.T @ da0_dx[:nc, :nq]
# da0_dx[:nc, nq:] = oRf.T @ da0_dx[:nc, nq:]
# print(pin.skew(oRf @ contactData.a0) @ oJf[3:])
# Fill out stuff 
a_partial_dtau = Kinv[:nv, :nv]
a_partial_da   = Kinv[:nv, -nc:]     
f_partial_dtau = Kinv[-nc:, :nv]
f_partial_da   = Kinv[-nc:, -nc:]

Fx = np.zeros((nv, nx))
Fx[:,:nq] = -a_partial_dtau @ data.dtau_dq
Fx[:,nq:] = -a_partial_dtau @ data.dtau_dv
Fx -= a_partial_da @ da0_dx[:nc]
Fx += a_partial_dtau @ actuationData.dtau_dx
Fu = a_partial_dtau @ actuationData.dtau_du

if(enable_force):
    df_dx = np.zeros((nc, nx))
    df_du = np.zeros((nc, nu))

    df_dx[:nc, :nv]  = f_partial_dtau @ data.dtau_dq
    df_dx[:nc, -nv:] = f_partial_dtau @ data.dtau_dv
    df_dx[:nc, :]   += f_partial_da @ da0_dx[:nc]
    df_dx[:nc, :]   -= f_partial_dtau @ actuationData.dtau_dx

    df_du[:nc, :] = -f_partial_dtau @ actuationData.dtau_du

    # Update acc and force derivatives
    contactModel.updateAccelerationDiff(contactData, Fx[-nv:,:])
    contactModel.updateForceDiff(contactData, df_dx[:nc, :], df_du[:nc, :])


    # logger.debug("   -- Test ddv_dx (model vs python) --")
    # # logger.info("PIN.ddv_dx   :\n "+ str(contactData.ddv_dx))
    # # logger.info("MODEL.ddv_dx :\n "+ str(DAD.multibody.contacts.ddv_dx))
    # logger.debug(np.allclose(contactData.ddv_dx, DAD.multibody.contacts.ddv_dx, RTOL, ATOL))


    # logger.debug("   -- Test df_dx (model vs python) --")
    # # logger.info("PIN.df_dx   :\n "+ str(contactData.contacts["contact_"+contact_frame_name].df_dx))
    # # logger.info("MODEL.df_dx :\n "+ str(DAD.multibody.contacts.contacts["contact_"+contact_frame_name].df_dx))
    # logger.debug(np.allclose(contactData.contacts["contact_"+contact_frame_name].df_dx, DAD.multibody.contacts.contacts["contact_"+contact_frame_name].df_dx, RTOL, ATOL))

    # logger.debug("   -- Test df_du (model vs python) --")
    # # logger.info("PIN.df_du   :\n "+ str(contactData.contacts["contact_"+contact_frame_name].df_du))
    # # logger.info("MODEL.df_du :\n "+ str(DAD.multibody.contacts.contacts["contact_"+contact_frame_name].df_du))
    # logger.debug(np.allclose(contactData.contacts["contact_"+contact_frame_name].df_du, DAD.multibody.contacts.contacts["contact_"+contact_frame_name].df_du, RTOL, ATOL))


logger.debug("   -- Test Fu (python vs numdiff) --")
# logger.info("PYTHON.Fu   :\n "+ str(Fu))
# logger.info("NUMDIFF.Fu :\n "+ str(DAD_ND.Fu))
logger.debug(np.allclose(Fu, DAD_ND.Fu, RTOL, ATOL))

logger.debug("   -- Test Fx (python vs numdiff) --")
logger.info("PYTHON.Fx   :\n "+ str(Fx))
logger.info("NUMDIFF.Fx :\n "+ str(DAD_ND.Fx))
logger.debug("           Fq")
logger.debug(np.allclose(Fx[:,:nq], DAD_ND.Fx[:,:nq], RTOL, ATOL))
# logger.debug("\n"+str(np.isclose(DAD.Fx[:,:nq], DAD_ND.Fx[:,:nq], RTOL, ATOL)))
logger.debug("           Fv")
logger.debug(np.allclose(Fx[:,nq:], DAD_ND.Fx[:,nq:], RTOL, ATOL))
# logger.debug("\n"+str(np.isclose(DAD.Fx[:,nq:], DAD_ND.Fx[:,nq:], RTOL, ATOL)))


runningCostModel.calcDiff(costData, x0, tau)

logger.debug("   -- Test Fu (model vs python) --")
logger.debug(np.allclose(Fu, DAD.Fu, RTOL, ATOL))
# logger.debug("   -- Test Fu (numdiff vs python) --")
# logger.debug(np.allclose(Fu, DAD_ND.Fu, RTOL, ATOL))
logger.debug("   -- Test Fx (model vs python) --")
# logger.info("PIN.Fx   :\n "+ str(Fx))
# logger.info("MODEL.Fx :\n "+ str(DAD.Fx))
logger.debug(np.allclose(Fx, DAD.Fx, RTOL, ATOL))
# logger.debug("\n"+str(np.isclose(Fx, DAD.Fx, RTOL, ATOL)))



# if(WITH_COSTS):
#     # Control regularization cost
#     uResidual = crocoddyl.ResidualModelContactControlGrav(state)
#     uRegCost = crocoddyl.CostModelResidual(state, uResidual)
#     # State regularization cost
#     xResidual = crocoddyl.ResidualModelState(state, x0)
#     xRegCost = crocoddyl.CostModelResidual(state, xResidual)
#     # End-effector frame force cost
#     desired_wrench = np.array([0., 0., -20., 0., 0., 0.])
#     frameForceResidual = crocoddyl.ResidualModelContactForce(state, contact_frame_id, pin.Force(desired_wrench), nc, actuation.nu)
#     contactForceCost = crocoddyl.CostModelResidual(state, frameForceResidual)
#     # Populate cost models with cost terms
#     runningCostModel.addCost("stateReg", xRegCost, 1e-2)
#     runningCostModel.addCost("ctrlRegGrav", uRegCost, 1e-4)
#     runningCostModel.addCost("force", contactForceCost, 10.)
#     terminalCostModel.addCost("stateReg", xRegCost, 1e-2)

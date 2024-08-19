import sys
sys.path.append('.')

import numpy as np
np.set_printoptions(precision=6, linewidth=180, suppress=True)
np.random.seed(10)
import crocoddyl
import pinocchio as pin

from soft_mpc.soft_models_3D_augmented import DAMSoftContactDynamics3D, IAMSoftContactDynamics3D, StateSoftContact3D
from core_mpc.pin_utils import load_robot_wrapper
from soft_mpc.data import DDPDataHandlerSoftContact
from core_mpc_utils import pin_utils



robot = load_robot_wrapper('iiwa')
model = robot.model ; data = model.createData()
nq = model.nq; nv = model.nv; nu = nq; nx = nq+nv
q0 = np.array([0.1, 0.7, 0., 0.7, -0.5, 1.5, 0.]) #np.random.rand(nq) #
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
oPc = oPf + np.array([0.05,.0, 0]) # + cm in x world np.random.rand(3) #np.random.rand(3) #
print("initial EE position (WORLD) = \n", oPf)
print("anchor point (WORLD)        = \n", oPc)
ov = pin.getFrameVelocity(model, data, frameId, pin.WORLD).linear
print("initial EE velocity (WORLD) = \n", ov)
# contact gains
Kp = 100.
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




# check derivatives against numdiff !!!!
def numdiff(f,x0,h=1e-6):
    if(type(f(x0))!=np.ndarray):
      f0 = f(x0)
    else:
      f0 = f(x0).copy()
    x = x0.copy()
    Fx = []
    for ix in range(len(x)):
        x[ix] += h
        Fx.append((f(x)-f0)/h)
        x[ix] = x0[ix]
    return np.array(Fx).T

# Create Differential Action Model (DAM), i.e. continuous dynamics and cost functions
dam = DAMSoftContactDynamics3D(state, actuation, runningCostModel, frameId, Kp, Kv, oPc, pinRefFrame=pin.LOCAL)
fref = np.random.rand(3)
dam.set_force_cost(fref, 1e-2)
# dam_t = DAMSoftContactDynamics3D(state, actuation, terminalCostModel, frameId, Kp, Kv, oPc, pinRefFrame=pin.LOCAL)
tau = np.random.rand(nq)
y0 = np.concatenate([x0, lf0])
dad = dam.createData()
dam.calc(dad, x0, lf0, tau)
dam.calcDiff(dad, x0, lf0, tau)
dam_ND = DAMSoftContactDynamics3D(state, actuation, runningCostModel, frameId, Kp, Kv, oPc, pinRefFrame=pin.LOCAL)
dam_ND.set_force_cost(fref, 1e-2)
dad_ND = dam_ND.createData()
# Test xout
def get_xout(model, data, x, f, u):
  model.calc(data, x, f, u)
  return data.xout
dxout_dx_ND = numdiff(lambda x_:get_xout(dam_ND, dad_ND, x_, lf0, tau), x0)
dxout_dx = dad.Fx
assert(np.linalg.norm(dxout_dx_ND - dxout_dx )< 1e-2) 
dxout_du_ND = numdiff(lambda u_:get_xout(dam_ND, dad_ND, x0, lf0, u_), tau)
dxout_du = dad.Fu
assert(np.linalg.norm(dxout_du_ND - dxout_du )< 1e-2) 
dxout_df_ND = numdiff(lambda f_:get_xout(dam_ND, dad_ND, x0, f_, tau), lf0)
dxout_df = dad.dABA_df
assert(np.linalg.norm(dxout_df_ND - dxout_df )< 1e-2)
# Test fout
def get_fout(model, data, x, f, u):
  model.calc(data, x, f, u)
  return data.fout
dfout_dx_ND = numdiff(lambda x_:get_fout(dam_ND, dad_ND, x_, lf0, tau), x0)
dfout_dx = dad.dfdt_dx
assert(np.linalg.norm(dfout_dx_ND - dfout_dx )< 1e-2)
dfout_du_ND = numdiff(lambda u_:get_fout(dam_ND, dad_ND, x0, lf0, u_), tau)
dfout_du = dad.dfdt_du
assert(np.linalg.norm(dfout_du_ND - dfout_du )< 1e-2) 
dfout_df_ND = numdiff(lambda f_:get_fout(dam_ND, dad_ND, x0, f_, tau), lf0)
dfout_df = dad.dfdt_df
assert(np.linalg.norm(dfout_df_ND - dfout_df )< 1e-2)  
#  Test cost
def get_cost(model, data, x, f, u):
  model.calc(data, x, f, u)
  return data.cost
dcost_dx_ND = numdiff(lambda x_:get_cost(dam_ND, dad_ND, x_, lf0, tau), x0)
dcost_dx = dad.Lx
assert(np.linalg.norm(dcost_dx_ND - dcost_dx )< 1e-2)
dcost_du_ND = numdiff(lambda u_:get_cost(dam_ND, dad_ND, x0, lf0, u_), tau)
dcost_du = dad.Lu
assert(np.linalg.norm(dcost_du_ND - dcost_du )< 1e-2) 
dcost_df_ND = numdiff(lambda f_:get_cost(dam_ND, dad_ND, x0, f_, tau), lf0)
dcost_df = dad.Lf
assert(np.linalg.norm(dcost_df_ND - dcost_df )< 1e-2)  


# Create Differential Action Model (DAM), i.e. continuous dynamics and cost functions
dam = DAMSoftContactDynamics3D(state, actuation, runningCostModel, frameId, Kp, Kv, oPc, pinRefFrame=pin.LOCAL_WORLD_ALIGNED)
fref = np.random.rand(3)
dam.set_force_cost(fref, 1e-2)
# dam_t = DAMSoftContactDynamics3D(state, actuation, terminalCostModel, frameId, Kp, Kv, oPc, pinRefFrame=pin.LOCAL_WORLD_ALIGNED)
tau = np.random.rand(nq)
y0 = np.concatenate([x0, of0])
dad = dam.createData()
dam.calc(dad, x0, of0, tau)
dam.calcDiff(dad, x0, of0, tau)
dam_ND = DAMSoftContactDynamics3D(state, actuation, runningCostModel, frameId, Kp, Kv, oPc, pinRefFrame=pin.LOCAL_WORLD_ALIGNED)
dam_ND.set_force_cost(fref, 1e-2)
dad_ND = dam_ND.createData()
# Test xout
def get_xout(model, data, x, f, u):
  model.calc(data, x, f, u)
  return data.xout
dxout_dx_ND = numdiff(lambda x_:get_xout(dam_ND, dad_ND, x_, of0, tau), x0)
dxout_dx = dad.Fx
assert(np.linalg.norm(dxout_dx_ND - dxout_dx )< 1e-2) 
dxout_du_ND = numdiff(lambda u_:get_xout(dam_ND, dad_ND, x0, of0, u_), tau)
dxout_du = dad.Fu
assert(np.linalg.norm(dxout_du_ND - dxout_du )< 1e-2) 
dxout_df_ND = numdiff(lambda f_:get_xout(dam_ND, dad_ND, x0, f_, tau), of0)
dxout_df = dad.dABA_df
assert(np.linalg.norm(dxout_df_ND - dxout_df )< 1e-2)
# Test fout
def get_fout(model, data, x, f, u):
  model.calc(data, x, f, u)
  return data.fout
dfout_dx_ND = numdiff(lambda x_:get_fout(dam_ND, dad_ND, x_, of0, tau), x0)
dfout_dx = dad.dfdt_dx
assert(np.linalg.norm(dfout_dx_ND - dfout_dx )< 1e-2)
dfout_du_ND = numdiff(lambda u_:get_fout(dam_ND, dad_ND, x0, of0, u_), tau)
dfout_du = dad.dfdt_du
assert(np.linalg.norm(dfout_du_ND - dfout_du )< 1e-2) 
dfout_df_ND = numdiff(lambda f_:get_fout(dam_ND, dad_ND, x0, f_, tau), of0)
dfout_df = dad.dfdt_df
assert(np.linalg.norm(dfout_df_ND - dfout_df )< 1e-2)  
#  Test cost
def get_cost(model, data, x, f, u):
  model.calc(data, x, f, u)
  return data.cost
dcost_dx_ND = numdiff(lambda x_:get_cost(dam_ND, dad_ND, x_, of0, tau), x0)
dcost_dx = dad.Lx
assert(np.linalg.norm(dcost_dx_ND - dcost_dx )< 1e-2)
dcost_du_ND = numdiff(lambda u_:get_cost(dam_ND, dad_ND, x0, of0, u_), tau)
dcost_du = dad.Lu
assert(np.linalg.norm(dcost_du_ND - dcost_du )< 1e-2) 
dcost_df_ND = numdiff(lambda f_:get_cost(dam_ND, dad_ND, x0, f_, tau), of0)
dcost_df = dad.Lf
assert(np.linalg.norm(dcost_df_ND - dcost_df )< 1e-2)  





# # Create Integrated Action Model (IAM), i.e. Euler integration of continuous dynamics and cost
# dt=1e-3
# dam = DAMSoftContactDynamics3D(state, actuation, runningCostModel, frameId, Kp, Kv, oPc, pinRefFrame=pin.LOCAL_WORLD_ALIGNED)
# fref = np.random.rand(3)
# dam.set_force_cost(fref, 1e-2)
# iam = IAMSoftContactDynamics3D(dam, dt)
# tau = np.random.rand(nq)
# y0 = np.concatenate([x0, lf0])
# iad = iam.createData()
# iam.calc(iad, y0, tau)
# iam.calcDiff(iad, y0, tau)
# iam_ND = IAMSoftContactDynamics3D(dam, dt)
# iad_ND = iam_ND.createData()
# # Test xout
# def get_xnext(model, data, y, u):
#   model.calc(data, y, u)
#   return data.xnext
# dxnext_dx_ND = numdiff(lambda y_:get_xnext(iam_ND, iad_ND, y_, tau), y0)
# dxnext_dx = iad.Fx
# assert(np.linalg.norm(dxnext_dx_ND - dxnext_dx )< 1e-2) 
# dxnext_du_ND = numdiff(lambda u_:get_xnext(iam_ND, iad_ND, y0, u_), tau)
# dxnext_du = iad.Fu
# assert(np.linalg.norm(dxnext_du_ND - dxnext_du )< 1e-2) 
# # Test cost
# def get_cost(model, data, x, u):
#   model.calc(data, x, u)
#   return data.cost
# dcost_dx_ND = numdiff(lambda y_:get_cost(iam_ND, iad_ND, y_, tau), y0)
# dcost_dx = iad.Lx
# assert(np.linalg.norm(dcost_dx_ND - dcost_dx )< 1e-2)
# dcost_du_ND = numdiff(lambda u_:get_cost(iam_ND, iad_ND, y0, u_), tau)
# dcost_du = iad.Lu
# assert(np.linalg.norm(dcost_du_ND - dcost_du )< 1e-2) 




# REF_FRAME = pin.LOCAL
# # # Python proto
# # dam = DAMSoftContactDynamics3D(state, actuation, runningCostModel, frameId, Kp, Kv, oPc, pinRefFrame=REF_FRAME)
# # dam_t = DAMSoftContactDynamics3D(state, actuation, runningCostModel, frameId, Kp, Kv, oPc, pinRefFrame=REF_FRAME)
# # dt=1e-3
# # dam.set_force_cost(np.array([0.,0.,10.]), 1e-2)
# # runningModel = IAMSoftContactDynamics3D(dam, dt)
# # terminalModel = IAMSoftContactDynamics3D(dam_t, 0.)

# # Sobec C++ bindings
# import sobec
# dam = sobec.DAMSoftContact3DAugmentedFwdDynamics(state, actuation, runningCostModel, frameId, Kp, Kv, oPc, REF_FRAME)
# dam_t = sobec.DAMSoftContact3DAugmentedFwdDynamics(state, actuation, runningCostModel, frameId, Kp, Kv, oPc, REF_FRAME)
# dt=1e-3
# dam.set_force_cost(np.array([0.,0.,10.]), 1e-2)
# runningModel = sobec.IAMSoftContact3DAugmented(dam, dt)
# terminalModel = sobec.IAMSoftContact3DAugmented(dam_t, 0.)


# # # Optionally add armature to take into account actuator's inertia
# # runningModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.])
# # terminalModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.])

# # Create the shooting problem
# T = 200
# y0 = np.concatenate([x0, of0])
# problem = crocoddyl.ShootingProblem(y0, [runningModel] * T, terminalModel)

# # Create solver + callbacks
# ddp = crocoddyl.SolverFDDP(problem)
# ddp.setCallbacks([crocoddyl.CallbackLogger(),
#                 crocoddyl.CallbackVerbose()])
# # Warm start : initial state + gravity compensation
# ys_init = [y0 for i in range(T+1)]
# us_init = [pin_utils.get_tau(q0, v0, np.zeros(nq), fext0, model, np.zeros(nq)) for i in range(T)] #ddp.problem.quasiStatic(xs_init[:-1])

# # Solve
# ddp.solve(ys_init, us_init, maxiter=100, isFeasible=False)

# # Extract data
# from soft_mpc.utils import SoftContactModel3D
# softContactModel = SoftContactModel3D(Kp, Kv, oPc, frameId, REF_FRAME)
# data_handler = DDPDataHandlerSoftContact(ddp, softContactModel)
# ddp_data = data_handler.extract_data_augmented(ee_frame_name='contact', ct_frame_name='contact')
# fs_lin = np.zeros((T,3))
# xs = np.array(ddp_data['xs'])
# fs_lin = np.array([softContactModel.computeForce_(model, xs[i,:nq], xs[i,nq:nq+nv]) for i in range(T)])
# fs_ang = np.zeros((ddp_data['T'], 3))
# f_prediction= np.hstack([fs_lin, fs_ang])

# # Plot data + predictions
# fix, ax = data_handler.plot_ddp_results(ddp_data, markers=['.'], SHOW=False)
# for i in range(3): ax['f'][i,0].plot(np.linspace(0, T*dt, T), f_prediction[:,i], 'ro', label='pred')
# import matplotlib.pyplot as plt
# plt.show()




# DISPLAY = False
# # Visualize motion in Gepetto-viewer
# if(DISPLAY):
#     import time
#     import pinocchio as pin
#     N_h = T
#     # Init viewer
#     viz = pin.visualize.GepettoVisualizer(robot.model, robot.collision_model, robot.visual_model)
#     viz.initViewer()
#     viz.loadViewerModel()
#     viz.display(q0)
#     # time.sleep(100)
#     gui = viz.viewer.gui
#     draw_rate = int(N_h/50)
#     log_rate  = int(N_h/10)    
#     ref_color  = [1., 0., 0., 1.]
#     real_color = [0., 0., 1., 0.3]
#     ref_size    = 0.02
#     real_size   = 0.02
#     pause = 0.05
#     # cleanup
#         # clean ref
#     if(gui.nodeExists('world/EE_ref')):
#         gui.deleteNode('world/EE_ref', True)
#     for i in range(N_h):
#         # clean DDP sol
#         if(gui.nodeExists('world/EE_sol_'+str(i))):
#             gui.deleteNode('world/EE_sol_'+str(i), True)
#             gui.deleteLandmark('world/EE_sol_'+str(i))
#     # Get initial EE placement + tf
#     ee_frame_placement = data.oMf[frameId]
#     tf_ee = list(pin.SE3ToXYZQUAT(ee_frame_placement))
#     # Display sol init + landmark
#     gui.addSphere('world/EE_sol_', real_size, real_color)
#     gui.addLandmark('world/EE_sol_', 0.25)
#     gui.applyConfiguration('world/EE_sol_', tf_ee)
#     # Get anchor point = ref EE placement + tf
#     ref_frame_placement = data.oMf[frameId].copy()
#     ref_frame_placement.translation = oPc
#     tf_ref = list(pin.SE3ToXYZQUAT(ref_frame_placement))
#     # Display ref
#     gui.addSphere('world/EE_ref', ref_size, ref_color)
#     gui.applyConfiguration('world/EE_ref', tf_ref)
#     # Refresh and wait
#     gui.refresh()
#     time.sleep(1.)
#     # Animate
#     for i in range(N_h):
#         # Display robot in config q
#         q = ddp.xs[i][:nq]
#         viz.display(q)
#         # Display EE traj and ref circle traj
#         if(i%draw_rate==0):
#             # EE trajectory
#             robot.framesForwardKinematics(q)
#             m_ee = robot.data.oMf[frameId].copy()
#             tf_ee = list(pin.SE3ToXYZQUAT(m_ee))
#             gui.addSphere('world/EE_sol_'+str(i), real_size, real_color)
#             gui.applyConfiguration('world/EE_sol_'+str(i), tf_ee)
#             gui.applyConfiguration('world/EE_sol_', tf_ee)
#         gui.refresh()
#         if(i%log_rate==0):
#             print("Display config n°"+str(i))
#         time.sleep(pause)
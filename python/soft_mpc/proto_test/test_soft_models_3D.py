import sys

sys.path.append('.')

import numpy as np
np.set_printoptions(precision=6, linewidth=180, suppress=True)
np.random.seed(1)
import crocoddyl
import pinocchio as pin

from soft_mpc.soft_models_3D import DAMSoftContactDynamics3D
from core_mpc.pin_utils import load_robot_wrapper
from classical_mpc.data import DDPDataHandlerClassical
from core_mpc_utils import pin_utils


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



# Create Differential Action Model (DAM), i.e. continuous dynamics and cost functions
# dam = DAMSoftContactDynamics3D(state, actuation, runningCostModel, frameId, Kp, Kv, oPc, pinRefFrame=REF_FRAME)
import sobec
dam = sobec.DifferentialActionModelSoftContact3DFwdDynamics(state, actuation, runningCostModel, frameId, Kp, Kv, oPc, REF_FRAME)
dam.set_force_cost(np.array([0.,0.,0.]), 1e-2)
# dam_t = DAMSoftContactDynamics3D(state, actuation, terminalCostModel, frameId, Kp, Kv, oPc, pinRefFrame=REF_FRAME)
dam_t = sobec.DifferentialActionModelSoftContact3DFwdDynamics(state, actuation, terminalCostModel, frameId, Kp, Kv, oPc, REF_FRAME)



# Create Integrated Action Model (IAM), i.e. Euler integration of continuous dynamics and cost
dt=5e-3
# runningModel = crocoddyl.IntegratedActionModelRK4(dam, dt)
# terminalModel = crocoddyl.IntegratedActionModelRK4(dam_t, 0.)
runningModel = crocoddyl.IntegratedActionModelEuler(dam, dt)
terminalModel = crocoddyl.IntegratedActionModelEuler(dam_t, 0.)


# # Optionally add armature to take into account actuator's inertia
# runningModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.])
# terminalModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.])

# Create the shooting problem
T = 200
problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

# Create solver + callbacks
ddp = crocoddyl.SolverFDDP(problem)
ddp.setCallbacks([crocoddyl.CallbackLogger(),
                crocoddyl.CallbackVerbose()])
# Warm start : initial state + gravity compensation
xs_init = [x0 for i in range(T+1)]
us_init = [pin_utils.get_tau(q0, v0, np.zeros(nq), fext0, model, np.zeros(nq)) for i in range(T)] #ddp.problem.quasiStatic(xs_init[:-1])

# Solve
ddp.solve(xs_init, us_init, maxiter=100, isFeasible=False)

# Extract data
data_handler = DDPDataHandlerClassical(ddp)
ddp_data = data_handler.extract_data(ee_frame_name='contact', ct_frame_name='contact')
  # Extract soft force
xs = np.array(ddp_data['xs'])
ps = pin_utils.get_p_(xs[:,:nq], model, frameId)
vs = pin_utils.get_v_(xs[:,:nq], xs[:,nq:], model, frameId, ref=pin.WORLD)
  # Force in WORLD aligned frame
fs_lin = np.array([data.oMf[frameId].rotation @ (-Kp*(ps[i,:] - oPc) - Kv*vs[i,:]) for i in range(T)])
fs_ang = np.zeros((T, 3))
ddp_data['fs'] = np.hstack([fs_lin, fs_ang])
ddp_data['force_ref'] = [np.concatenate([dam.f_des, np.zeros(3)]) for i in range(T) ]

# Plot data
data_handler.plot_ddp_results(ddp_data, markers=['.'], SHOW=True)

DISPLAY = False
# Visualize motion in Gepetto-viewer
if(DISPLAY):
    import time
    import pinocchio as pin
    N_h = T
    # Init viewer
    viz = pin.visualize.GepettoVisualizer(robot.model, robot.collision_model, robot.visual_model)
    viz.initViewer()
    viz.loadViewerModel()
    viz.display(q0)
    # time.sleep(100)
    gui = viz.viewer.gui
    draw_rate = int(N_h/50)
    log_rate  = int(N_h/10)    
    ref_color  = [1., 0., 0., 1.]
    real_color = [0., 0., 1., 0.3]
    ref_size    = 0.02
    real_size   = 0.02
    pause = 0.05
    # cleanup
        # clean ref
    if(gui.nodeExists('world/EE_ref')):
        gui.deleteNode('world/EE_ref', True)
    for i in range(N_h):
        # clean DDP sol
        if(gui.nodeExists('world/EE_sol_'+str(i))):
            gui.deleteNode('world/EE_sol_'+str(i), True)
            gui.deleteLandmark('world/EE_sol_'+str(i))
    # Get initial EE placement + tf
    ee_frame_placement = data.oMf[frameId]
    tf_ee = list(pin.SE3ToXYZQUAT(ee_frame_placement))
    # Display sol init + landmark
    gui.addSphere('world/EE_sol_', real_size, real_color)
    gui.addLandmark('world/EE_sol_', 0.25)
    gui.applyConfiguration('world/EE_sol_', tf_ee)
    # Get anchor point = ref EE placement + tf
    ref_frame_placement = data.oMf[frameId].copy()
    ref_frame_placement.translation = oPc
    tf_ref = list(pin.SE3ToXYZQUAT(ref_frame_placement))
    # Display ref
    gui.addSphere('world/EE_ref', ref_size, ref_color)
    gui.applyConfiguration('world/EE_ref', tf_ref)
    # Refresh and wait
    gui.refresh()
    time.sleep(1.)
    # Animate
    for i in range(N_h):
        # Display robot in config q
        q = ddp.xs[i][:nq]
        viz.display(q)
        # Display EE traj and ref circle traj
        if(i%draw_rate==0):
            # EE trajectory
            robot.framesForwardKinematics(q)
            m_ee = robot.data.oMf[frameId].copy()
            tf_ee = list(pin.SE3ToXYZQUAT(m_ee))
            gui.addSphere('world/EE_sol_'+str(i), real_size, real_color)
            gui.applyConfiguration('world/EE_sol_'+str(i), tf_ee)
            gui.applyConfiguration('world/EE_sol_', tf_ee)
        gui.refresh()
        if(i%log_rate==0):
            print("Display config n°"+str(i))
        time.sleep(pause)





# # Simulate system 
# Nsim = 1000
# # iam = crocoddyl.IntegratedActionModelEuler(dam, dt)
# # iad = iam.createData()
# dad = dam.createData()
# q
# tau = pin_utils.get_tau(q0, np.zeros(nq), np.zeros(nq), fext0, model, np.zeros(nq)) 
# x = x0
# q = q0 ; vq = v0
# xs = np.zeros((Nsim+1, nx))
# us = np.zeros((Nsim, nq))
# xs[0,:] = x
# DT = 1e-3
# for i in range(Nsim):
#   # desired acceleration 
#   aq_des = -Kp*(q-q0) - Kv*vq
#   print("des = ", aq_des)
#   # Compute joint acceleration (forward dynamics)
#   dam.calc(dad, x, tau)
#   aq = dad.xout
#   print("real = ", aq)
#   # Integrate and update state
#   vq = vq + aq*DT
#   q = q + vq*DT
#   x = np.concatenate([q, vq])
#   xs[i+1,:] = x
#   us[i,:] = tau
#   # update model and data 
#   pin.forwardKinematics(model, data, q, vq, aq_des)
#   pin.updateFramePlacements(model, data)
#   # Compute contact force
#   of = -Kp*(data.oMf[frameId].translation - oPc) - Kv*pin.getFrameVelocity(model, data, frameId, pin.WORLD).linear
#   lf = oRf.T @ of
#   fext = [pin.Force.Zero() for _ in range(model.njoints)]
#   fext[model.frames[frameId].parent] = model.frames[frameId].placement.act(pin.Force(lf, np.zeros(3)))
#   # Compute torque
#   tau = pin_utils.get_tau(x[:nq], x[nq:], np.zeros(nq), fext, model, np.zeros(nq)) 

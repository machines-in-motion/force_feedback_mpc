'''
Adapted from Rooholla's code in Go2Py examples
https://github.com/machines-in-motion/Go2Py/blob/mpc/examples/standard_mpc.py

This sets up the Go2 MPC wrapper and runs an MPC simulation in Mujoco
the robot must push against a wall with its end-effector while standing
on its 4 feet and without slipping (the same task as in the original script)
'''

import numpy as np
import os
import mim_solvers
import pinocchio as pin
import crocoddyl
import pinocchio

from soft_multicontact_api import ViscoElasticContact3d_Multiple, ViscoElasticContact3D
from soft_multicontact_api import FrictionConeConstraint, ForceConstraintManager
from soft_multicontact_api import ForceCost, ForceCostManager
from soft_multicontact_api import DAMSoftContactDynamics3D_Go2, IAMSoftContactDynamics3D_Go2

from Go2MPC_wrapper import Go2MPC, getForceSensor
from Go2Py.sim.mujoco import Go2Sim

# Instantiate the simulator
robot=Go2Sim(with_arm=True)
map = np.zeros((1200, 1200))
map[:,649:679] = 400
robot.updateHeightMap(map)

# Instantiate the solver
assets_path = '/home/skleff/force_feedback_ws/Go2Py/Go2Py/assets/'
mpc = Go2MPC(assets_path, HORIZON=20, friction_mu=0.1)
mpc.initialize()
mpc.solve()
m = list(mpc.solver.problem.runningModels) + [mpc.solver.problem.terminalModel]

# # Reset the robot
# state = mpc.getSolution(0)
# robot.pos0 = state['position']
# robot.rot0 = state['orientation']
# robot.q0 = state['q']
# robot.reset()


# # Solve for as many iterations as needed for the first step
# mpc.max_iterations=10 #500

# Nsim = 50
# measured_forces = []
# forces = np.linspace(0, 50, Nsim) 
# # breakpoint()
# WITH_INTEGRAL = True
# if(WITH_INTEGRAL):
#     err_f = 0
#     Ki = 0.1
#     err_f6d = np.zeros(6)
# for i in range(Nsim):
#     print("Step ", i)
#     # set the force setpoint
#     for action_model in m:
#         action_model.differential.costs.costs['contact_force_track'].cost.residual.reference.linear[:] = np.array([-forces[i], 0, 0])

#     state = robot.getJointStates()
#     q = state['q']
#     dq = state['dq']
#     t, quat = robot.getPose()
#     v = robot.data.qvel[:3]
#     omega = robot.data.qvel[3:6]
#     q = np.hstack([q, np.zeros(2)])
#     dq = np.hstack([dq, np.zeros(2)])
#     solution = mpc.updateAndSolve(t, quat, q, v, omega, dq)
#     # Reduce the max iteration count to ensure real-time execution
#     mpc.max_iterations=10
#     q = solution['q']
#     dq = solution['dq']
#     tau = solution['tau'].squeeze()
#     kp = np.ones(18)*0.
#     kv = np.ones(18)*0.
#     # Step the physics
#     force_site_to_sensor_idx = {'FL_force_site': 0, 'FR_force_site': 1, 'RL_force_site': 2, 'RR_force_site': 3, 'EF_force_site': 4}
#     force_sensor_site = 'EF_force_site'
#     f_mea = getForceSensor(robot.model, robot.data, force_sensor_site).squeeze().copy()
#     measured_forces.append(f_mea)
#     # compute the force integral error and map it to joint torques
#     if(WITH_INTEGRAL):
#         err_f6d[0] += Ki * (forces[i] - f_mea[0])
#         pin.computeAllTerms(mpc.rmodel, mpc.rdata, mpc.xs[0][:mpc.rmodel.nq], mpc.xs[0][mpc.rmodel.nq:])
#         J = pin.getFrameJacobian(mpc.rmodel, mpc.rdata, mpc.armEEId, pin.LOCAL_WORLD_ALIGNED)
#         tau_int = J[:,6:].T @ err_f6d 
#         print(tau_int)
#         tau += tau_int
#     for j in range(int(mpc.dt//robot.dt)):
#         robot.setCommands(q, dq, kp, kv, tau)
#         robot.step()

# measured_forces = np.array(measured_forces)

# # Visualize the measured force against the desired
# import matplotlib.pyplot as plt
# plt.plot(measured_forces[:300,0], '*')
# plt.plot(forces,'k')
# plt.show()
# robot.close()




# required to extract and plot solution
rmodel = mpc.rmodel
rdata = mpc.rdata
solver = mpc.solver
xs = mpc.solver.xs 
supportFeetIds = mpc.supportFeetIds
T = mpc.HORIZON
MU = mpc.friction_mu
comDes = mpc.comDes

# Extract OCP Solution 
nq, nv, N = rmodel.nq, rmodel.nv, len(xs) 
jointPos_sol = []
jointVel_sol = []
jointAcc_sol = []
jointTorques_sol = []
centroidal_sol = []
force_sol = []
xs, us = solver.xs, solver.us
x = []
for time_idx in range (N):
    q, v = xs[time_idx][:nq], xs[time_idx][nq:nq+nv]
    f = xs[time_idx][nq+nv:]
    pin.framesForwardKinematics(rmodel, rdata, q)
    pin.computeCentroidalMomentum(rmodel, rdata, q, v)
    centroidal_sol += [
        np.concatenate(
            [pin.centerOfMass(rmodel, rdata, q, v), np.array(rdata.hg.linear), np.array(rdata.hg.angular)]
            )
            ]
    jointPos_sol += [q]
    jointVel_sol += [v]
    force_sol    += [f]
    x += [xs[time_idx]]
    if time_idx < N-1:
        jointAcc_sol +=  [solver.problem.runningDatas[time_idx].xnext[nq::]] 
        jointTorques_sol += [us[time_idx]]

sol = {'x':x, 'centroidal':centroidal_sol, 'jointPos':jointPos_sol, 
                    'jointVel':jointVel_sol, 'jointAcc':jointAcc_sol, 'force':force_sol,
                    'jointTorques':jointTorques_sol}       

# Extract contact forces by hand
sol['FL_FOOT_contact'] = [force_sol[i][0:3] for i in range(N)]     
sol['FR_FOOT_contact'] = [force_sol[i][3:6] for i in range(N)]     
sol['HL_FOOT_contact'] = [force_sol[i][6:9] for i in range(N)]     
sol['HR_FOOT_contact'] = [force_sol[i][9:12] for i in range(N)]     
sol['Link6'] = [force_sol[i][12:15] for i in range(N)]     

# Plotting 
import matplotlib.pyplot as plt
constrained_sol = sol
time_lin = np.linspace(0, T, solver.problem.T+1)
fig, axs = plt.subplots(4, 3, constrained_layout=True)
for i, frame_idx in enumerate(supportFeetIds):
    ct_frame_name = rmodel.frames[frame_idx].name + "_contact"
    forces = np.array(constrained_sol[ct_frame_name])
    axs[i, 0].plot(time_lin, forces[:, 0], label="Fx")
    axs[i, 1].plot(time_lin, forces[:, 1], label="Fy")
    axs[i, 2].plot(time_lin, forces[:, 2], label="Fz")
    # Add friction cone constraints 
    Fz_lb = (1./MU)*np.sqrt(forces[:, 0]**2 + forces[:, 1]**2)
    # Fz_ub = np.zeros(time_lin.shape)
    # axs[i, 2].plot(time_lin, Fz_ub, 'k-.', label='ub')
    axs[i, 2].plot(time_lin, Fz_lb, 'k-.', label='lb')
    axs[i, 0].grid()
    axs[i, 1].grid()
    axs[i, 2].grid()
    axs[i, 0].set_ylabel(ct_frame_name)
axs[0, 0].legend()
axs[0, 1].legend()
axs[0, 2].legend()

axs[3, 0].set_xlabel(r"$F_x$")
axs[3, 1].set_xlabel(r"$F_y$")
axs[3, 2].set_xlabel(r"$F_z$")
fig.suptitle('Force', fontsize=16)


comDes = np.array(comDes)
centroidal_sol = np.array(constrained_sol['centroidal'])
plt.figure()
plt.plot(comDes[:, 0], comDes[:, 1], "--", label="reference")
plt.plot(centroidal_sol[:, 0], centroidal_sol[:, 1], label="solution")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("COM trajectory")
plt.show()
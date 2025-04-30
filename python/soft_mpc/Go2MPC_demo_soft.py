'''
Adapted from Rooholla's code in Go2Py examples
https://github.com/machines-in-motion/Go2Py/blob/mpc/examples/standard_mpc.py

This sets up the Go2 MPC wrapper and runs an MPC simulation in Mujoco
the robot must push against a wall with its end-effector while standing
on its 4 feet and without slipping (the same task as in the original script)

Force-feedback MPC (soft contact force in the state)
'''

import numpy as np
import pinocchio as pin



from Go2MPC_wrapper_soft import Go2MPCSoft, getForceSensor, setGroundFriction, plot_ocp_solution
from utils import LPFButterOrder2

# LOADING CONFIG parameters
import os
from force_feedback_mpc.core_mpc_utils.path_utils import load_yaml_file
CONFIG        = load_yaml_file(os.path.dirname(os.path.realpath(__file__))+'/Go2MPC_demo_soft.yml')
USE_MUJOCO    = CONFIG['USE_MUJOCO']
DT_SIMU       = CONFIG['DT_SIMU']
FREF          = CONFIG['FREF']
MU            = CONFIG['MU']
HORIZON       = CONFIG['HORIZON']
DT_OCP        = CONFIG['DT_OCP']
MAX_ITER_1    = CONFIG['MAX_ITER_1']
MAX_ITER_2    = CONFIG['MAX_ITER_2']
N_SIMU        = CONFIG['N_SIMU']
MPC_FREQ      = CONFIG['MPC_FREQ']
KP            = CONFIG['KP']
KV            = CONFIG['KV']
SIM_FREQ      = int(1./DT_SIMU)
RECORD_VIDEO  = CONFIG['RECORD_VIDEO']
DATA_SAVE_DIR = CONFIG['DATA_SAVE_DIR']

# Instantiate the simulator
if(USE_MUJOCO):
    from Go2Py.sim.mujoco import Go2Sim
    robot = Go2Sim(with_arm=True, dt=DT_SIMU)
    map = np.zeros((1200, 1200))
    map[:,649:679] = 400
    robot.updateHeightMap(map)
else:
    # Load PyBullet simulation environment
    from mim_robots.robot_loader import load_bullet_wrapper
    from mim_robots.pybullet.env import BulletEnvWithGround
    import pybullet as p
    from force_feedback_mpc.core_mpc_utils import sim_utils
    env = BulletEnvWithGround(dt=DT_SIMU, server=p.DIRECT)
    # env = BulletEnvWithGround(dt=DT_SIMU, server=p.GUI)
    robot = load_bullet_wrapper('go2')
    q0 = np.array([-0.01, 0.0, 0.32, 0.0, 0.0, 0.0, 1.0] + 4*[0.0, 0.77832842, -1.56065452] + [0.0, 0.3, -0.3, 0.0, 0.0, 0.0])
    v0 = np.zeros(robot.pin_robot.model.nv)
    env.add_robot(robot) 
    robot.reset_state(q0, v0)
    robot.forward_robot(q0, v0)
    sim_utils.set_lateral_friction(env.objects[0], MU)
    sim_utils.set_contact_stiffness_and_damping(env.objects[0], 10000, 500)
    contact_placement = pin.SE3(pin.rpy.rpyToMatrix(0.,np.pi/2, 0.), np.array([0.42,0.,0.]))
    contact_surface_bulletId = sim_utils.display_contact_surface(contact_placement, radius=2., bullet_endeff_ids=robot.bullet_endeff_ids)
    sim_utils.set_lateral_friction(contact_surface_bulletId, MU)
    sim_utils.set_contact_stiffness_and_damping(contact_surface_bulletId, 10000, 500)
    # Meshcat visualization
    import meshcat.geometry as g
    import meshcat.transformations as tf
    from pinocchio.visualize import MeshcatVisualizer
    viz = MeshcatVisualizer(robot.pin_robot.model, robot.pin_robot.collision_model, robot.pin_robot.visual_model)
    viz.initViewer()
    # Override transparency of some links 
    ee_frame_names = ['FL_FOOT', 'FR_FOOT', 'HL_FOOT', 'HR_FOOT', 'Link6', 'ef_tip', \
                      'FL_KFE', 'FR_KFE', 'HL_KFE', 'HR_KFE', 'HR_HAA', 'Link5']
    for fname in ee_frame_names:
        fid = robot.pin_robot.model.getFrameId(fname)
        fname = robot.pin_robot.model.frames[fid].name + '_0'
        print("changing mesh of ", fname)
        g_id = viz.visual_model.getGeometryId(fname)
        viz.visual_model.geometryObjects[g_id].meshColor = np.array([1., 1., 1., 0.5])
    viz.loadViewerModel()
    viz.display(q0)
    surface_tf = tf.translation_matrix(contact_placement.translation) @ tf.rotation_matrix(np.pi/2, [0, 1, 0])  
    viz.viewer["contact_surface"].set_object(
        g.Box([4.0, 4.0, 0.001]),
        g.MeshLambertMaterial(color=0x00aaff, opacity=0.3)
    )
    viz.viewer["contact_surface"].set_transform(surface_tf)



# Instantiate the solver
mpc = Go2MPCSoft(HORIZON=HORIZON, friction_mu=MU, dt=DT_OCP, USE_MUJOCO=USE_MUJOCO)
mpc.initialize(FREF=FREF)
mpc.max_iterations = MAX_ITER_1
# mpc.test_derivatives()
mpc.solve()
m = list(mpc.solver.problem.runningModels) + [mpc.solver.problem.terminalModel]
# Reset number of iter for MPC loop
mpc.max_iterations=MAX_ITER_2


# Meshcat setup
arrows, arrow_ee, cones = mpc.setup_meshcat_force_vizualisation(viz, MU)

# # plot_ocp_solution(mpc)
# # Extract OCP Solution 
# force_sol = []
# xs, us = mpc.solver.xs, mpc.solver.us
# for time_idx in range (mpc.HORIZON+1):
#     f = xs[time_idx][-3:]
#     force_sol    += [f]
# force_sol = np.array(force_sol)   
# time_span = np.linspace(0, mpc.HORIZON*mpc.dt, mpc.HORIZON+1)
# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(time_span, force_sol[:,0], label='x')
# plt.plot(time_span, [-FREF]*(mpc.HORIZON+1), 'k-.', label='ref x')
# plt.plot(time_span, force_sol[:,1], label='y')
# plt.plot(time_span, force_sol[:,2], label='z')
# plt.title("End-effector forces")
# plt.legend()
# plt.show()

# Reset the robot and prepare force sensors and physics parameters
if(USE_MUJOCO):
    state = mpc.getSolution(0)
    robot.pos0 = state['position']
    robot.rot0 = state['orientation']
    robot.q0 = state['q']
    robot.reset()
    # mapping force sensors in mujoco to mpc model names
    frame_name_to_mujoco_sensor = {'FL_FOOT': 'FL_force_site', 
                                'FR_FOOT': 'FR_force_site', 
                                'HL_FOOT': 'RL_force_site', 
                                'HR_FOOT': 'RR_force_site', 
                                'Link6': 'EF_force_site'}
    # physics
    setGroundFriction(robot.model, robot.data, MU)


frame_name_to_sol_map = {'FL_FOOT': 'f_lf', 
                        'FR_FOOT': 'f_rf', 
                        'HL_FOOT': 'f_lh', 
                        'HR_FOOT': 'f_rh', 
                        'Link6': 'f_ee'}
sol_to_force_id_map = {'f_lf': 0, 
                    'f_rf': 3, 
                    'f_lh': 6, 
                    'f_rh': 9, 
                    'f_ee': 12}
    
# Prepare logs
measured_forces_dict = {}
predicted_forces_dict = {}
for fname in mpc.ee_frame_names:
    measured_forces_dict[fname]  = []
    predicted_forces_dict[fname] = []
desired_forces = []
joint_torques = []
f_des_z = np.array([FREF]*N_SIMU) 

# Force filtering
butter_dict = {}
filtered_forces_dict = {}
filtered_forces_dict = {}
for fname in mpc.ee_frame_names:
    filtered_forces_dict[fname]  = []
    if(fname == 'Link6'):
        CUTOFF_2 = 100 
    else:
        CUTOFF_2 = 100
    butter_dict[fname] = {'x': LPFButterOrder2(fc=CUTOFF_2, fs=1./1e-3),
                          'y': LPFButterOrder2(fc=CUTOFF_2, fs=1./1e-3),
                          'z': LPFButterOrder2(fc=CUTOFF_2, fs=1./1e-3)}
alpha_LPF = 0.5


# Main simulation loop
f_mea_all = np.zeros(3*5)
f_mea_all_filtered = np.zeros(3*5)


# MUJOCO SIMULATION
if(USE_MUJOCO):
    for i in range(N_SIMU):
        print("Step ", i)
        # set the force setpoint
        f_des_3d = np.array([-f_des_z[i], 0, 0])
        desired_forces.append(f_des_3d)
        # Get state from simulation
        state = robot.getJointStates()
        q = state['q']
        dq = state['dq']
        t, quat = robot.getPose()
        v = robot.data.qvel[:3]
        omega = robot.data.qvel[3:6]
        q = np.hstack([q, np.zeros(2)])
        dq = np.hstack([dq, np.zeros(2)])
        # Measure forces
        for fname in mpc.ee_frame_names:
            # Raw measured force
            f_mea = -getForceSensor(robot.model, robot.data, frame_name_to_mujoco_sensor[fname]).squeeze().copy()
            measured_forces_dict[fname].append(f_mea)
            id_f = sol_to_force_id_map[frame_name_to_sol_map[fname]]
            f_mea_all[id_f:id_f+3] = f_mea
            # #  Filtered force
            # if(i==0):
            #     f_mea_filtered = f_mea  
            # else:
            #     # LPF filtering
            #     # f_mea_filtered = alpha_LPF * filtered_forces_dict[fname][i-1] + (1 - alpha_LPF) * f_mea
            #     # Butterworth filtering 
            f_mea_filtered = np.zeros(3)
            f_mea_filtered[0] = butter_dict[fname]['x'].filter(f_mea[0])
            f_mea_filtered[1] = butter_dict[fname]['y'].filter(f_mea[1])
            f_mea_filtered[2] = butter_dict[fname]['z'].filter(f_mea[2])
            filtered_forces_dict[fname].append(f_mea_filtered)
            f_mea_all_filtered[id_f:id_f+3] = f_mea_filtered   
        # Solve OCP
        if(i%int(SIM_FREQ/MPC_FREQ)==0):
            solution = mpc.updateAndSolve(t, quat, q, v, omega, dq, f_mea_all)
        for fname in mpc.ee_frame_names:
            predicted_forces_dict[fname].append(solution[frame_name_to_sol_map[fname]])
        # Save the solution
        q = solution['q']
        dq = solution['dq']
        tau = solution['tau'].squeeze()
        kp = np.ones(18)*KP #10.
        kv = np.ones(18)*KV #1
        # Step the physics
        robot.setCommands(q, dq, kp, kv, tau)
        robot.step()
# PYBULLET SIMULATION
else:   
    image_array_list = []
    jointPos = []
    jointVel = []
    
    for i in range(N_SIMU):
        print("Step ", i)
        # set the force setpoint
        f_des_3d = np.array([-f_des_z[i], 0, 0])
        desired_forces.append(f_des_3d)
        # Get state from simulation
        q, dq = robot.get_state()
        jointPos.append(q)
        jointVel.append(dq)
        # Measure forces in PyBullet simulation
        robot.forward_robot(q, dq)
        contact_status, f_mea_bullet = robot.end_effector_forces()
        for k,fname in enumerate(mpc.ee_frame_names):
            if(contact_status[robot.pinocchio_endeff_ids[k]]):
                f_mea = f_mea_bullet[k,:3]
            else:
                f_mea = np.zeros(3)
            measured_forces_dict[fname].append(f_mea)
            filtered_forces_dict[fname].append(f_mea)
            f_mea_all[3*k:3*(k+1)] = f_mea
            f_mea_all_filtered[3*k:3*(k+1)] = f_mea
        # Solve OCP
        if(i%int(SIM_FREQ/MPC_FREQ)==0):
            solution = mpc.updateAndSolve2(q, dq, f_mea_all)
            # plot_ocp_solution(mpc)
        # Record predicted forces and visualize forces+cones in meshcat
        for fname in mpc.ee_frame_names:
            # Save predicted forces from OCP solution
            predicted_forces_dict[fname].append(solution[frame_name_to_sol_map[fname]])

        # MESHCAT VISUALIZATION
        viz.display(q)
        # update contact force and cone 
        for k, fid in enumerate(mpc.supportFeetIds):
            fname = robot.pin_robot.model.frames[fid].name
            contactLoc = robot.pin_robot.data.oMf[fid].translation
            arrows[k].anchor_as_vector(contactLoc, measured_forces_dict[fname][i])
        arrow_ee.anchor_as_vector(robot.pin_robot.data.oMf[mpc.armEEId].translation, measured_forces_dict["Link6"][i])
        if(RECORD_VIDEO):
            image_array_list.append(viz.captureImage())

        # Save the optimal control solution
        tau = solution['tau'].squeeze()
        joint_torques.append(tau)
        # Step the physics
        robot.send_joint_command(tau)
        env.step() 

if(RECORD_VIDEO):
    import imageio
    def create_video_from_rgba(images, output_path, fps=50):
        """
        Create an MP4 video from an RGBA image array.

        Args:
            images (list): List of RGBA image arrays.
            output_path (str): Path to save the resulting MP4 video.
            fps (int): Frames per second for the video (default: 200).
        """
        writer = imageio.get_writer(output_path, format='ffmpeg', fps=fps)
        print("saving to ")
        print(output_path)
        print(writer)
        for img in images:
            writer.append_data(img)
        writer.close()
        print("Closed writer")
    output_path = '/home/skleff/go2_mpc_soft.mp4'
    create_video_from_rgba(image_array_list, output_path)

# import pickle
# data = {'jointPos': jointPos, 
#         'measured_forces': measured_forces_dict, 
#         'supportFeedIds': mpc.supportFeetIds,
#         'supportFeetPos': mpc.supportFeetPos0,
#         'armEEId': mpc.armEEId,
#         'armEEName': 'Link6',
#         'mu': MU,
#         'rmodel': robot.pin_robot.model}
# # Pickling (serializing) and saving to a file
# filename = '/home/skleff/meshcat_data.pkl'
# with open(filename, 'wb') as file:
#     pickle.dump(data, file)
# # # Unpickling (deserializing) from the file
# # with open(filename, 'rb') as file:
# #     loaded_data = pickle.load(file)

desired_forces = np.array(desired_forces)
joint_torques = np.array(joint_torques)
jointPos = np.array(jointPos)
jointVel = np.array(jointVel)
for fname in mpc.ee_frame_names:
    measured_forces_dict[fname] = np.array(measured_forces_dict[fname])
    filtered_forces_dict[fname] = np.array(filtered_forces_dict[fname])
    predicted_forces_dict[fname] = np.array(predicted_forces_dict[fname])

# Save data 
import time
np.savez_compressed(DATA_SAVE_DIR+'_'+str(time.time())+'.npz',
                    jointPos=jointPos,
                    jointVel=jointVel,
                    joint_torques=joint_torques,
                    measured_forces=measured_forces_dict,
                    filtered_forces=filtered_forces_dict,
                    desired_forces=desired_forces,
                    predicted_forces=predicted_forces_dict,
                    ee_frame_names=mpc.ee_frame_names)
print("Saved MPC simulation data to "+DATA_SAVE_DIR)
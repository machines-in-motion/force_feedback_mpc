import numpy as np
import pinocchio as pin

from core_mpc_utils.misc_utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger

# Check installed pkg
import importlib
FOUND_PYBULLET_PKG            = importlib.util.find_spec("pybullet")               is not None
FOUND_ROB_PROP_KUKA_PKG       = importlib.util.find_spec("robot_properties_kuka")  is not None
FOUND_ROB_PROP_TALOS_PKG      = importlib.util.find_spec("robot_properties_talos") is not None
FOUND_BULLET_UTILS_PKG        = importlib.util.find_spec("bullet_utils")           is not None
# FOUND_EXAMPLE_ROBOT_DATA_PKG  = importlib.util.find_spec("example_robot_data")     is not None

if(FOUND_PYBULLET_PKG): 
    import pybullet as p
else:
    logger.error('You need to install PyBullet ( https://pypi.org/project/pybullet/ )')

if(FOUND_BULLET_UTILS_PKG):
    from bullet_utils.env import BulletEnvWithGround
else:
    logger.error('You need to install bullet_utils ( https://github.com/machines-in-motion/bullet_utils )')


# Global & default settings (change CAREFULLY)
SUPPORTED_ROBOTS         = ['iiwa', 'iiwa_reduced', 'talos_arm', 'talos_reduced']

TALOS_DEFAULT_MESH_PATH  = '/opt/openrobots/share'

IIWA_DEFAULT_BASE_POS = [0, 0, 0]
IIWA_DEFAULT_BASE_RPY = [0, 0, 0]

TALOS_ARM_DEFAULT_BASE_POS = [0, 0, 0]
TALOS_ARM_DEFAULT_BASE_RPY = [0, 0, 0]

TALOS_REDUCED_DEFAULT_BASE_POS = [0, 0, 1.02]
TALOS_REDUCED_DEFAULT_BASE_RPY = [0, 0, 0]



# Load robot in PyBullet environment 
def init_bullet_simulation(robot_name, dt=1e3, x0=None):
    '''
    Initialize a PyBullet simulation environment with robot SUPPORTED_ROBOTS
      INPUT:
        robot_name : robot name in SUPPORTED_ROBOTS (string)
        dt         : simulator time step (double)
        x0         : initial robot state (q0,v0) (vector nq+nv)
    '''
    if(robot_name not in SUPPORTED_ROBOTS):
        logger.error("Specified robot not supported ! Select a robot in "+str(SUPPORTED_ROBOTS))
    else:
        if(robot_name == 'iiwa'):
            return init_iiwa_bullet(dt=dt, x0=x0)
        elif(robot_name == 'talos_arm'):
            return init_talos_arm_bullet(dt=dt, x0=x0)
        elif(robot_name == 'talos_reduced'):
            return init_talos_reduced_bullet(dt=dt, x0=x0)
        elif(robot_name == 'iiwa_reduced'):
            return init_iiwa_reduced_bullet(dt=dt, x0=x0)


# Load KUKA arm in PyBullet environment
def init_iiwa_bullet(dt=1e3, x0=None, pos=IIWA_DEFAULT_BASE_POS, orn=IIWA_DEFAULT_BASE_RPY):
    '''
    Loads KUKA LBR iiwa model in PyBullet simulator
    using the PinBullet wrapper to simplify interactions
      INPUT:
        dt      : simulator time step (double)
        x0      : initial robot state (q0,v0) (vector nq+nv)
        pos     : position of the kuka base in simulator WORLD frame (vector3)
        orn     : orientation of the kuka base in simulator WORLD frame ()
    '''
    if(FOUND_ROB_PROP_KUKA_PKG):
        try: 
            from robot_properties_kuka.iiwaWrapper import IiwaRobot as IiwaRobot
            from robot_properties_kuka.config import IiwaConfig
        except:
            logger.error("The IiwaRobot was not found.")
    else:
        logger.error('You need to install robot_properties_kuka ( https://github.com/machines-in-motion/robot_properties_kuka )')
    logger.info("Initializing KUKA iiwa in PyBullet simulator...\n")
    # Create PyBullet sim environment + initialize sumulator
    env = BulletEnvWithGround(p.DIRECT, dt=dt)
    orn_quat = p.getQuaternionFromEuler(orn)
    base_placement = pin.XYZQUATToSE3(pos + list(orn_quat))
    config = IiwaConfig()
    robot_simulator = env.add_robot(IiwaRobot(config, pos, orn_quat))
    # Initialize
    if(x0 is None):
        q0 = np.array([0.1, 0.7, 0., 0.7, -0.5, 1.5, 0.]) 
        dq0 = np.zeros(robot_simulator.pin_robot.model.nv)
    else:
        q0 = x0[:robot_simulator.pin_robot.model.nq]
        dq0 = x0[robot_simulator.pin_robot.model.nv:]
    robot_simulator.reset_state(q0, dq0)
    robot_simulator.forward_robot(q0, dq0)
    return env, robot_simulator, base_placement


# Load TALOS arm in PyBullet environment
def init_talos_arm_bullet(dt=1e3, x0=None, pos=TALOS_ARM_DEFAULT_BASE_POS, orn=TALOS_ARM_DEFAULT_BASE_RPY):
    '''
    Loads TALOS left arm model in PyBullet simulator
    using the PinBullet wrapper to simplify interactions
      INPUT:
        dt        : simulator time step
        x0        : initial robot state (pos and vel)
    '''
    if(FOUND_ROB_PROP_TALOS_PKG):
        try:
            from robot_properties_talos.talosArmWrapper import TalosArmRobot
        except:
            logger.error("The wrapper TalosArmRobot was not found.")
    else:
        logger.error('You need to install robot_properties_talos ( https://github.com/machines-in-motion/robot_properties_talos )')
    # Info log
    logger.info("Initializing TALOS left arm in PyBullet simulator...\n")
    # Create PyBullet sim environment + initialize sumulator
    env = BulletEnvWithGround(p.GUI, dt=dt)
    orn_quat = p.getQuaternionFromEuler(orn)
    base_placement = pin.XYZQUATToSE3(pos + list(orn_quat)) 
    robot_simulator = env.add_robot(TalosArmRobot(pos, orn_quat))
    # Initialize
    if(x0 is None):
        q0 = np.array([2., 0., 0., 0., 0., 0., 0.])
        dq0 = np.zeros(robot_simulator.pin_robot.model.nv)
    else:
        q0 = x0[:robot_simulator.pin_robot.model.nq]
        dq0 = x0[robot_simulator.pin_robot.model.nv:]
    robot_simulator.reset_state(q0, dq0)
    robot_simulator.forward_robot(q0, dq0)
    return env, robot_simulator, base_placement


# Load TALOS arm in PyBullet environment
def init_talos_reduced_bullet(dt=1e3, x0=None, pos=TALOS_REDUCED_DEFAULT_BASE_POS, orn=TALOS_REDUCED_DEFAULT_BASE_RPY):
    '''
    Loads TALOS left arm model in PyBullet simulator
    using the PinBullet wrapper to simplify interactions
      INPUT:
        dt        : simulator time step
        x0        : initial robot state (pos and vel)
    '''
    if(FOUND_ROB_PROP_TALOS_PKG):
        try:
            from robot_properties_talos.talosReducedWrapper import TalosReducedRobot
        except:
            logger.error("The wrapper TalosReducedRobot was not found.")
    else:
        logger.error('You need to install robot_properties_talos ( https://github.com/machines-in-motion/robot_properties_talos )')
    logger.info("Initializing TALOS reduced model in PyBullet simulator...\n")
    # Create PyBullet sim environment + initialize sumulator
    env = BulletEnvWithGround(p.DIRECT, dt=dt)
    orn_quat = p.getQuaternionFromEuler(orn)
    base_placement = pin.XYZQUATToSE3(pos + list(orn_quat)) 
    robot_simulator = env.add_robot(TalosReducedRobot(pos, orn_quat))
    # Initialize
    if(x0 is None):
        q0 = np.array([2., 0., 0., 0., 0., 0., 0.])
        dq0 = np.zeros(robot_simulator.pin_robot.model.nv)
    else:
        q0 = x0[:robot_simulator.pin_robot.model.nq]
        dq0 = x0[robot_simulator.pin_robot.model.nv:]
    robot_simulator.reset_state(q0, dq0)
    robot_simulator.forward_robot(q0, dq0)
    # To allow collisions with all parts of the robot if there is a contact surface (for contact & sanding tasks)
    # for i in range(p.getNumJoints(robot_simulator.robotId)):
    #     robot_simulator.bullet_endeff_ids.append(i)
    # robot_simulator.endeff_names = [] 
    return env, robot_simulator, base_placement


# Load TALOS arm in PyBullet environment
def init_iiwa_reduced_bullet(dt=1e3, x0=None, pos=IIWA_DEFAULT_BASE_POS, orn=IIWA_DEFAULT_BASE_RPY):
    '''
    Loads IIWA reduced arm model in PyBullet simulator
    using the PinBullet wrapper to simplify interactions
      INPUT:
        dt        : simulator time step
        x0        : initial robot state (pos and vel)
    '''
    if(FOUND_ROB_PROP_KUKA_PKG):
        try: 
            from robot_properties_kuka.iiwaReducedWrapper import IiwaReducedRobot
            from robot_properties_kuka.config import IiwaReducedConfig
        except:
            logger.error("The IiwaReducedRobot was not found.")
    else:
        logger.error('You need to install robot_properties_kuka ( https://github.com/machines-in-motion/robot_properties_kuka )')
    logger.info("Initializing KUKA iiwa in PyBullet simulator...\n")
    controlled_joints =  ['A1', 'A2', 'A3', 'A4', 'A5', 'A6']
    logger.info("Reduced model with controlled joints = "+str(controlled_joints))
    qref = np.zeros(7)
    config = IiwaReducedConfig()
    # Create PyBullet sim environment + initialize sumulator
    env = BulletEnvWithGround(p.GUI, dt=dt)
    orn_quat = p.getQuaternionFromEuler(orn)
    base_placement = pin.XYZQUATToSE3(pos + list(orn_quat)) 
    robot_simulator = env.add_robot(IiwaReducedRobot(config, controlled_joints, qref, pos, orn_quat))
    # Initialize
    if(x0 is None):
        q0 = qref[:len(controlled_joints)]
        dq0 = np.zeros(robot_simulator.pin_robot.model.nv)
    else:
        q0 = x0[:robot_simulator.pin_robot.model.nq]
        dq0 = x0[robot_simulator.pin_robot.model.nv:]
    robot_simulator.reset_state(q0, dq0)
    robot_simulator.forward_robot(q0, dq0)
    # To allow collisions with all parts of the robot if there is a contact surface (for contact & sanding tasks)
    # for i in range(p.getNumJoints(robot_simulator.robotId)):
    #     robot_simulator.bullet_endeff_ids.append(i)
    # robot_simulator.endeff_names = [] 
    return env, robot_simulator, base_placement




# PROTOTYPE  : angular part does not work for now
def get_contact_wrench(pybullet_simulator, id_endeff, ref=pin.LOCAL):
    '''
    Get contact wrench in ref contact frame
     pybullet_simulator : pinbullet wrapper object
     id_endeff          : frame of interest 
     ref                : pin ref frame in which wrench is expressed
    This function works like PinBulletWrapper.get_force() but also accounts for torques
    The linear force returned by this function should match the one returned by get_force()
    (get_force() must be transformed into LOCAL by lwaMf.actInv if ref=pin.LOCAL
     no transform otherwise) 
    '''
    contact_points = p.getContactPoints()
    total_wrench = pin.Force.Zero() #np.zeros(6)
    oMf = pybullet_simulator.pin_robot.data.oMf[id_endeff]
    p_endeff = oMf.translation
    active_contacts_frame_ids = []
    for ci in reversed(contact_points):
        # remove contact points that are not from 
        p_ct = np.array(ci[6])
        contact_normal = np.array(ci[7])
        normal_force = ci[9]
        lateral_friction_direction_1 = np.array(ci[11])
        lateral_friction_force_1 = ci[10]
        lateral_friction_direction_2 = np.array(ci[13])
        lateral_friction_force_2 = ci[12]
        # keep contact point only if it concerns one of the reduced model's endeffectors
        if ci[3] in pybullet_simulator.bullet_endeff_ids:
            i = np.where(np.array(pybullet_simulator.bullet_endeff_ids) == ci[3])[0][0]
        elif ci[4] in pybullet_simulator.bullet_endeff_ids:
            i = np.where(np.array(pybullet_simulator.bullet_endeff_ids) == ci[4])[0][0]
        else:
            continue
        if pybullet_simulator.pinocchio_endeff_ids[i] in active_contacts_frame_ids:
            continue
        active_contacts_frame_ids.append(pybullet_simulator.pinocchio_endeff_ids[i])
        # Wrench at the detected contact point in simulator WORLD
        o_linear = normal_force * contact_normal + \
                   lateral_friction_force_1 * lateral_friction_direction_1 + \
                   lateral_friction_force_2 * lateral_friction_direction_2
        l_linear  = oMf.rotation.T @ o_linear
            # compute torque w.r.t. frame of interest
        l_angular = np.cross(oMf.rotation.T @ (p_ct - p_endeff), l_linear)
        l_wrench = np.concatenate([l_linear, l_angular]) 
        total_wrench += pin.Force(l_wrench)
    # if local nothing to do
    if(ref==pin.LOCAL):
        return -total_wrench.vector
    # otherwise transform into LWA
    else:
        lwaMf = oMf.copy()
        lwaMf.translation = np.zeros(3)
        return -lwaMf.act(total_wrench).vector

# Get joint torques from robot simulator
def get_contact_joint_torques(pybullet_simulator, id_endeff):
    '''
    Get joint torques due to external wrench
    '''
    wrench = get_contact_wrench(pybullet_simulator, id_endeff)
    jac = pybullet_simulator.pin_robot.data.J
    joint_torques = jac.T @ wrench
    return joint_torques




# Display
def display_ball(p_des, robot_base_pose=pin.SE3.Identity(), RADIUS=.05, COLOR=[1.,1.,1.,1.]):
    '''
    Create a sphere visual object in PyBullet (no collision)
    Transformed because reference p_des is in pinocchio WORLD frame, which is different
    than PyBullet WORLD frame if the base placement in the simulator is not (eye(3), zeros(3))
    INPUT: 
        p_des           : desired position of the ball in pinocchio.WORLD
        robot_base_pose : initial pose of the robot BASE in bullet.WORLD
        RADIUS          : radius of the ball
        COLOR           : color of the ball
    '''
    # logger.debug&("Creating PyBullet sphere visual...")
    # pose of the sphere in bullet WORLD
    M = pin.SE3(np.eye(3), p_des)  # ok for talos reduced since pin.W = bullet.W but careful with talos_arm if base is moved
    quat = pin.SE3ToXYZQUAT(M)     
    visualBallId = p.createVisualShape(shapeType=p.GEOM_SPHERE,
                                       radius=RADIUS,
                                       rgbaColor=COLOR,
                                       visualFramePosition=quat[:3],
                                       visualFrameOrientation=quat[3:])
    ballId = p.createMultiBody(baseMass=0.,
                               baseInertialFramePosition=[0.,0.,0.],
                               baseVisualShapeIndex=visualBallId,
                               basePosition=[0.,0.,0.],
                               useMaximalCoordinates=False)

    return ballId


# Load contact surface in PyBullet for contact experiments
def display_contact_surface(M, robotId=1, radius=.5, length=0.0, bullet_endeff_ids=[], TILT=[0., 0., 0.]):
    '''
    Creates contact surface object in PyBullet as a flat cylinder 
      M              : contact placement expressed in simulator WORLD frame
      robotId        : id of the robot in simulator
      radius         : radius of cylinder
      length         : length of cylinder
      TILT           : RPY tilt of the surface
    '''
    logger.info("Creating PyBullet contact surface...")
    # Tilt contact surface (default 0)
    TILT_rotation = pin.utils.rpyToMatrix(TILT[0], TILT[1], TILT[2])
    M.rotation = TILT_rotation.dot(M.rotation)
    # Get quaternion
    quat = pin.SE3ToXYZQUAT(M)
    visualShapeId = p.createVisualShape(shapeType=p.GEOM_CYLINDER,
                                        radius=radius,
                                        length=length,
                                        rgbaColor=[.1, .8, .1, .5],
                                        visualFramePosition=quat[:3],
                                        visualFrameOrientation=quat[3:])
    # With collision
    if(len(bullet_endeff_ids)!=0):
      collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_CYLINDER,
                                                radius=radius,
                                                height=length,
                                                collisionFramePosition=quat[:3],
                                                collisionFrameOrientation=quat[3:])
      contactId = p.createMultiBody(baseMass=0.,
                                    baseInertialFramePosition=[0.,0.,0.],
                                    baseCollisionShapeIndex=collisionShapeId,
                                    baseVisualShapeIndex=visualShapeId,
                                    basePosition=[0.,0.,0.],
                                    useMaximalCoordinates=False)
                    
      # Desactivate collisions for all links
      for i in range(p.getNumJoints(robotId)):
            p.setCollisionFilterPair(contactId, robotId, -1, i, 1) # 0
            # logger.info("Set collision pair ("+str(contactId)+","+str(robotId)+"."+str(i)+") to True")
    #   # activate collisions only for EE ids
    #   for ee_id in bullet_endeff_ids:
    #         p.setCollisionFilterPair(contactId, robotId, -1, ee_id, 1)
    #         logger.info("Set collision pair ("+str(contactId)+","+str(robotId)+"."+str(ee_id)+") to True")
      return contactId
    # Without collisions
    else:
      contactId = p.createMultiBody(baseMass=0.,
                        baseInertialFramePosition=[0.,0.,0.],
                        baseVisualShapeIndex=visualShapeId,
                        basePosition=[0.,0.,0.],
                        useMaximalCoordinates=False)
      return contactId


# Load contact surface in PyBullet for contact experiments
def remove_body_from_sim(bodyId):
    '''
    Removes bodyfrom sim env
    '''
    logger.info("Removing body "+str(bodyId)+" from simulation !")
    p.removeBody(bodyId)


def print_dynamics_info(bodyId, linkId=-1):
    '''
    Returns pybullet dynamics info
    '''
    logger.info("Body n°"+str(bodyId))
    d = p.getDynamicsInfo(bodyId, linkId)
    print(d)
    logger.info("  mass                   : "+str(d[0]))
    logger.info("  lateral_friction       : "+str(d[1]))
    logger.info("  local_inertia_diagonal : "+str(d[2]))
    logger.info("  local_inertia_pos      : "+str(d[3]))
    logger.info("  local_inertia_orn      : "+str(d[4]))
    logger.info("  restitution            : "+str(d[5]))
    logger.info("  rolling friction       : "+str(d[6]))
    logger.info("  spinning friction      : "+str(d[7]))
    logger.info("  contact damping        : "+str(d[8]))
    logger.info("  contact stiffness      : "+str(d[9]))
    logger.info("  body type              : "+str(d[10]))
    logger.info("  collision margin       : "+str(d[11]))


# Set lateral friction coefficient to PyBullet body
def set_lateral_friction(bodyId, coef, linkId=-1):
  '''
  Set lateral friction coefficient to PyBullet body
  Input :
    bodyId : PyBullet body unique id
    linkId : linkId . Default : -1 (base link)
    coef   : friction coefficient in (0,1)
  '''
  p.changeDynamics(bodyId, linkId, lateralFriction=coef, rollingFriction=0., spinningFriction=0.) 
  logger.info("Set friction of body n°"+str(bodyId)+" (link n°"+str(linkId)+") to "+str(coef)) 

# Set contact stiffness coefficient to PyBullet body
def set_contact_stiffness_and_damping(bodyId, Ks, Kd, linkId=-1):
  '''
  Set contact stiffness coefficient to PyBullet body
  Input :
    bodyId : PyBullet body unique id
    linkId : linkId . Default : -1 (base link)
    Ks, Kd : stiffness and damping coefficients
  '''
#   p.changeDynamics(bodyId, linkId, restitution=0.2) 
  p.changeDynamics(bodyId, linkId, contactStiffness=Ks, contactDamping=Kd) 
  logger.info("Set contact stiffness of body n°"+str(bodyId)+" (link n°"+str(linkId)+") to "+str(Ks)) 
  logger.info("Set contact damping of body n°"+str(bodyId)+" (link n°"+str(linkId)+") to "+str(Kd)) 


# Set contact stiffness coefficient to PyBullet body
def set_contact_restitution(bodyId, Ks, Kd, linkId=-1):
  '''
  Set contact restitution coefficient to PyBullet body
  Input :
    bodyId : PyBullet body unique id
    linkId : linkId . Default : -1 (base link)
    coef   : restitution coefficient
  '''
  p.changeDynamics(bodyId, linkId, restitution=0.2) 
  logger.info("Set restitution of body n°"+str(bodyId)+" (link n°"+str(linkId)+") to "+str(Ks)) 





# def rotationMatrixFromTwoVectors(a, b):
#     a_copy = a / np.linalg.norm(a)
#     b_copy = b / np.linalg.norm(b)
#     a_cross_b = np.cross(a_copy, b_copy, axis=0)
#     s = np.linalg.norm(a_cross_b)
#     if s == 0:
#         return np.eye(3)
#     c = a_copy.dot(b_copy) 
#     ab_skew = pin.skew(a_cross_b)
#     return np.eye(3) + ab_skew + ( (1 - c) / (s**2) ) * ab_skew.dot(ab_skew) 

# def weighted_moving_average(series, lookback = None):
#     if not lookback:
#         lookback = len(series)
#     if len(series) == 0:
#         return 0
#     assert 0 < lookback <= len(series)

#     wma = 0
#     lookback_offset = len(series) - lookback
#     for index in range(lookback + lookback_offset - 1, lookback_offset - 1, -1):
#         weight = index - lookback_offset + 1
#         wma += series[index] * weight
#     return wma / ((lookback ** 2 + lookback) / 2)

# def hull_moving_average(series, lookback):
#     assert lookback > 0
#     hma_series = []
#     for k in range(int(lookback ** 0.5), -1, -1):
#         s = series[:-k or None]
#         wma_half = weighted_moving_average(s, min(lookback // 2, len(s)))
#         wma_full = weighted_moving_average(s, min(lookback, len(s)))
#         hma_series.append(wma_half * 2 - wma_full)
#     return weighted_moving_average(hma_series)

# N = 500
# X = np.linspace(-10,10,N)
# Y = np.vstack([np.sin(X), np.cos(X)]).T
# W = Y + np.vstack([np.random.normal(0., .2, N), np.random.normal(0, .2, N)]).T
# Z = Y.copy()
# lookback=50
# for i in range(N):
#     if(i==0):
#         pass
#     else:
#         Z[i,:] = hull_moving_average(W[:i,:], min(lookback,i))
# fig, ax = plt.subplots(1,2)
# ax[0].plot(X, Y[:,0], 'b-', label='ground truth')
# ax[0].plot(X, W[:,0], 'g-', label='noised data')
# ax[0].plot(X, Z[:,0], 'r-', label='HMA') 
# ax[1].plot(X, Y[:,1], 'b-', label='ground truth')
# ax[1].plot(X, W[:,1], 'g-', label='noised data')
# ax[1].plot(X, Z[:,1], 'r-', label='HMA') 
# ax[0].legend()
# plt.show()



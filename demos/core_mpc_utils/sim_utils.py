import numpy as np
import pinocchio as pin
import hppfcl

from croco_mpc_utils.utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger

# Check installed pkg
import importlib.util
FOUND_PYBULLET_PKG = importlib.util.find_spec("pybullet") is not None

if(FOUND_PYBULLET_PKG): 
    import pybullet as p
else:
    logger.error('You need to install PyBullet ( https://pypi.org/project/pybullet/ )')


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

def display_box(M, EXTENTS, COLOR=[0.662, 0.662, 0.662, 1.0]) -> int:
    '''
    Create a cube visual object in PyBullet.
    INPUT:
        M       : cube pose (SE3)
        EXTENTS : cube extents (tuple/list of 3 scalars for x, y, z dimensions)
        COLOR   : cube color (RGBA)
    '''
    # Convert the pose to the PyBullet format
    QUAT = pin.SE3ToXYZQUAT(M)

    HALF_EXTENT = (EXTENTS[0]/2, EXTENTS[1]/2, EXTENTS[2]/2)
    visualShapeIdCUBE = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=HALF_EXTENT,  # Specify the half extents for the cube
        rgbaColor=COLOR
    )

    cubeId = p.createMultiBody(
        baseMass=0,
        baseInertialFramePosition=[0, 0, 0],
        baseVisualShapeIndex=visualShapeIdCUBE,
        basePosition=QUAT[:3],
        baseOrientation=QUAT[3:],
        useMaximalCoordinates=True,
    )

    return cubeId

def display_capsule(M, RADIUS, LENGTH, COLOR=[1., 0., 0., 1.]) -> int:
    '''
    Create a capsule visual object in PyBullet
    INPUT:
        M               : capsule pose (SE3)
        RADIUS          : capsule radius (scalar)
        LEGNTH          : capsule length (scalar)
        COLOR           : capsule color (RGBA)
    '''
    quat = pin.SE3ToXYZQUAT(M)
    visualShapeIdCAPS = p.createVisualShape(
        p.GEOM_CAPSULE,
        radius=RADIUS,
        length=LENGTH,
        rgbaColor=COLOR,
        visualFramePosition=np.zeros(3),
        visualFrameOrientation=np.zeros(3),
    )
    capsId = p.createMultiBody(
        baseMass=0,
        baseInertialFramePosition=[0, 0, 0],
        baseVisualShapeIndex=visualShapeIdCAPS,
        basePosition=quat[:3],
        baseOrientation=quat[3:],
        useMaximalCoordinates=True,
    )
    return capsId

def create_box_obstacle(OBSTACLE_POSE, EXTENTS, name="obstacle"):
    '''
    Create a spherical obstacle geometric model
    '''
    # Creating the hppfcl shape
    # EXTENT2 = (EXTENTS[0], EXTENTS[1], EXTENTS[2])
    OBSTACLE = hppfcl.Box(*EXTENTS)
    # Adding the shape to the collision model
    OBSTACLE_GEOM_OBJECT = pin.GeometryObject(
        name,
        0,
        0,
        OBSTACLE,
        OBSTACLE_POSE,
    )
    return OBSTACLE_GEOM_OBJECT

def create_caps_obstacle(OBSTACLE_POSE, RADIUS, LENGTH, name="obstacle"):
    '''
    Create a capsule obstacle geometric model
    '''
    capsule_geom = hppfcl.Capsule(RADIUS, LENGTH)
    OBSTACLE_GEOM_OBJECT = pin.GeometryObject(name, 0, 0, capsule_geom, OBSTACLE_POSE)
    return OBSTACLE_GEOM_OBJECT

def transform_model_into_capsules(cmodel):
    """Modifying the collision model to transform the spheres/cylinders into capsules which makes it easier to have a fully constrained robot."""
    collision_model_reduced_copy = cmodel.copy()
    list_names_capsules = []

    # Going through all the goemetry objects in the collision model
    for geom_object in collision_model_reduced_copy.geometryObjects:
        if isinstance(geom_object.geometry, hppfcl.Cylinder):
            # Sometimes for one joint there are two cylinders, which need to be defined by two capsules for the same link.
            # Hence the name convention here.
            if (geom_object.name[:-1] + "capsule_0") in list_names_capsules:
                name = geom_object.name[:-1] + "capsule_" + "1"
            else:
                name = geom_object.name[:-1] + "capsule_" + "0"
            list_names_capsules.append(name)
            placement = geom_object.placement
            parentJoint = geom_object.parentJoint
            parentFrame = geom_object.parentFrame
            geometry = geom_object.geometry
            geom = pin.GeometryObject(
                name,
                parentFrame,
                parentJoint,
                hppfcl.Capsule(geometry.radius, geometry.halfLength),
                placement,
            )
            geom.meshColor = np.array([249, 136, 126, 125]) / 255
            cmodel.addGeometryObject(geom)
            cmodel.removeGeometryObject(geom_object.name)
        elif (
            isinstance(geom_object.geometry, hppfcl.Sphere)
            and "link" in geom_object.name
        ):
            cmodel.removeGeometryObject(geom_object.name)
    logger.debug("8")
    return cmodel

def setup_obstacle_collision(robot_simulator, pin_robot, config):
 
  # Creating the obstacle
  OBSTACLE1_POSE        = pin.SE3(np.eye(3), np.array(config["OBSTACLE1_POSE"]))
  OBSTACLE2_POSE        = pin.SE3(np.eye(3), np.array(config["OBSTACLE2_POSE"]))
  OBSTACLE_RADIUS1      = config["OBSTACLE_RADIUS1"]
  OBSTACLE_RADIUS2      = tuple(config["OBSTACLE_RADIUS2"])
  OBSTACLE1_GEOM_OBJECT = create_box_obstacle(OBSTACLE1_POSE, OBSTACLE_RADIUS1, name="obstacle1")
  OBSTACLE2_GEOM_OBJECT = create_box_obstacle(OBSTACLE2_POSE, OBSTACLE_RADIUS2, name="obstacle2")
  logger.debug("7")

  # Adding obstacle to collision model (pinocchio)
  pin_robot.collision_model = transform_model_into_capsules(pin_robot.collision_model)
  logger.debug("9")
  pin_robot.collision_model.addGeometryObject(OBSTACLE1_GEOM_OBJECT)
  pin_robot.collision_model.addGeometryObject(OBSTACLE2_GEOM_OBJECT)
  
  # display in pybullet + add to collision model 
  capsule_id = display_box(OBSTACLE1_POSE, OBSTACLE_RADIUS1)
  display_box(OBSTACLE2_POSE, OBSTACLE_RADIUS2)
  robot_simulator.pin_robot.collision_model = transform_model_into_capsules(robot_simulator.pin_robot.collision_model)
  robot_simulator.pin_robot.collision_model.addGeometryObject(OBSTACLE1_GEOM_OBJECT)
  robot_simulator.pin_robot.collision_model.addGeometryObject(OBSTACLE2_GEOM_OBJECT)
  return capsule_id

    # head = SimHead(robot_simulator, vicon_name='cube10', with_sliders=False)
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



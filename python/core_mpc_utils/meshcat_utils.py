import pinocchio as pin
import numpy as np
try:
    import meshcat
    import meshcat.geometry as g
    import meshcat.transformations as tf
except:
    print("You need to install meshcat.")


# def updateForceVisualization(viz, pin_robot, supportFeePos, ee_frame_names):
#     for i, contactLoc in enumerate(mpc.supportFeePos):
#         ct_frame_name = mpc.rmodel.frames[mpc.supportFeetIds[i]].name + "_contact"
#         forces.append(np.array(constrained_sol[ct_frame_name])[:, :3])
#         arrows[i].set_location(contactLoc)

def meshcat_material(r, g, b, a):
        material = meshcat.geometry.MeshPhongMaterial()
        material.color = int(r * 255) * 256 ** 2 + int(g * 255) * 256 + int(b * 255)
        material.opacity = a
        material.linewidth = 5.0
        return material

def addViewerBox(viz, name, sizex, sizey, sizez, rgba):
    if isinstance(viz, pin.visualize.MeshcatVisualizer):
        viz.viewer[name].set_object(meshcat.geometry.Box([sizex, sizey, sizez]),
                                meshcat_material(*rgba))


def addLineSegment(viz, name, vertices, rgba):
    if isinstance(viz, pin.visualize.MeshcatVisualizer):
        viz.viewer[name].set_object(meshcat.geometry.LineSegments(
                    meshcat.geometry.PointsGeometry(np.array(vertices)),     
                    meshcat_material(*rgba)
                    ))

def addPoint(viz, name, vertices, rgba):
    if isinstance(viz, pin.visualize.MeshcatVisualizer):
        viz.viewer[name].set_object(meshcat.geometry.Points(
                    meshcat.geometry.PointsGeometry(np.array(vertices)),     
                    meshcat_material(*rgba)
                    ))

def meshcat_transform(x, y, z, q, u, a, t):
    return np.array(pin.XYZQUATToSE3([x, y, z, q, u, a, t]))

def applyViewerConfiguration(viz, name, xyzquat):
    if isinstance(viz, pin.visualize.MeshcatVisualizer):
        viz.viewer[name].set_transform(meshcat_transform(*xyzquat))


class Arrow(object):
    def __init__(self, meshcat_vis, name, 
                 location=[0,0,0], 
                 vector=[0,0,1],
                 length_scale=1,
                 color=0xff0000):

        self.vis = meshcat_vis[name]
        self.cone = self.vis["cone"]
        self.line = self.vis["line"]
        self.material = g.MeshBasicMaterial(color=color, reflectivity=0.5)
        
        self.location, self.length_scale = location, length_scale
        self.anchor_as_vector(location, vector)
    
    def _update(self):
        # pass
        translation = tf.translation_matrix(self.location)
        rotation = self.orientation
        offset = tf.translation_matrix([0, self.length/2, 0])
        self.pose = translation @ rotation @ offset
        self.vis.set_transform(self.pose)
        
    def set_length(self, length, update=True):
        self.length = length * self.length_scale
        cone_scale = self.length/0.08
        self.line.set_object(g.Cylinder(height=self.length, radius=0.005), self.material)
        self.cone.set_object(g.Cylinder(height=0.015, 
                                        radius=0.01, 
                                        radiusTop=0., 
                                        radiusBottom=0.01),
                             self.material)
        self.cone.set_transform(tf.translation_matrix([0.,cone_scale*0.04,0]))
        if update:
            self._update()
        
    def set_direction(self, direction, update=True):
        orientation = np.eye(4)
        orientation[:3, 0] = np.cross([1,0,0], direction)
        orientation[:3, 1] = direction
        orientation[:3, 2] = np.cross(orientation[:3, 0], orientation[:3, 1])
        self.orientation = orientation
        if update:
            self._update()
    
    def set_location(self, location, update=True):
        self.location = location
        if update:
            self._update()
        
    def anchor_as_vector(self, location, vector, update=True):
        self.set_direction(np.array(vector)/np.linalg.norm(vector), False)
        self.set_location(location, False)
        self.set_length(np.linalg.norm(vector), False)
        if update:
            self._update()

    def delete(self):
        self.vis.delete()




import numpy as np
import meshcat.geometry as g
import meshcat.transformations as tf


class Cone(object):
    def __init__(self, meshcat_vis, name,
                 location=[0, 0, 0], mu=1,
                 vector=[0, 0, 1],
                 length_scale=0.1):
        
        self.vis = meshcat_vis[name]
        self.cone = self.vis["cone"]
        self.base_circle = self.vis["base_circle"]

        self.material = g.MeshPhongMaterial(
            color=0x00ff00,
            opacity=0.5,
            transparent=True,
            emissive=0x22aa22,
            side=2
        )

        self.circle_material = g.LineBasicMaterial(color=0xffffff, linewidth=4, opacity=1.)

        self.mu = mu * length_scale
        self.location, self.length_scale = location, length_scale
        self.anchor_as_vector(location, vector)

        self._create_base_circle()  # moved after anchor so self.length is available

    def _create_base_circle(self):
        num_segments = 64
        theta = np.linspace(0, 2 * np.pi, num_segments)
        x = 0.2 * np.cos(theta)
        z = 0.2 * np.sin(theta)
        y = np.zeros_like(theta)
        points = np.column_stack((x, y, z)).astype(np.float32)
        geometry = g.PointsGeometry(points)
        self.base_circle.set_object(
            g.Line(geometry, self.circle_material)
        )

    def _update(self):
        translation = tf.translation_matrix(self.location)
        rotation = self.orientation
        offset = tf.translation_matrix([0, self.length / 2, 0])
        self.pose = translation @ rotation @ offset
        self.vis.set_transform(self.pose)

        # Position base circle at the base of the cone
        base_offset = tf.translation_matrix([0, -self.length / 2, 0])
        self.base_circle.set_transform(self.pose @ base_offset)

    def set_length(self, length, update=True):
        self.length = length * self.length_scale
        cone_scale = self.length
        self.cone.set_object(g.Cylinder(
            height=cone_scale,
            radius=self.mu,
            radiusTop=self.mu,
            radiusBottom=0),
            self.material)
        if update:
            self._update()

    def set_direction(self, direction, update=True):
        direction = np.array(direction)
        direction = direction / np.linalg.norm(direction)

        up = np.array([0, 1, 0])
        axis = np.cross(up, direction)
        angle = np.arccos(np.clip(np.dot(up, direction), -1.0, 1.0))

        if np.linalg.norm(axis) < 1e-6:
            rotation = tf.identity_matrix()
        else:
            axis = axis / np.linalg.norm(axis)
            rotation = tf.rotation_matrix(angle, axis)

        self.orientation = rotation
        if update:
            self._update()

    def set_location(self, location, update=True):
        self.location = location
        if update:
            self._update()

    def anchor_as_vector(self, location, vector, update=True):
        self.set_direction(vector, False)
        self.set_location(location, False)
        self.set_length(np.linalg.norm(vector), False)
        if update:
            self._update()

    def delete(self):
        self.vis.delete()

import pinocchio as pin
from pinocchio.visualize import GepettoVisualizer
import numpy as np


def launch_viewer(robot, q0) :
    """
    Initialize gepetto viewer.
    [in] robot : RobotWrapper
    [in] q0    : configuration
    [out] Gepetto visualizer
    """
    viz = GepettoVisualizer(robot.model, robot.collision_model, robot.visual_model)
    viz.initViewer()
    viz.loadViewerModel()
    viz.display(q0)
    return viz

def display_sphere(gui, name, coord, radius=.1, color=[1.,0.,0.,1.]) :
    """
    Display a sphere in gepetto viewer.
    [in] viz    : Gepetto visualizer
    [in] name   : Name of the object
    [in] coord  : Coordinates of the center of the sphere
    [in] radius : Sphere radius
    [in] color  : Shpere color
    """
    if(type(coord)==list):
        coord = np.asarray(coord)
    M = pin.SE3(np.identity(3), coord)
    tf = list(pin.SE3ToXYZQUAT(M))
    if (gui.nodeExists(name)):
        gui.deleteNode(name, True)
    gui.addSphere(name, radius, color)
    gui.applyConfiguration(name, tf)
    gui.refresh()

def display_capsule(gui, name, pose, radius=.05, length=0.5, color=[1.,0.,0.,1.]) :
    """
    Display a capsule in gepetto viewer.
    [in] viz    : Gepetto visualizer
    [in] name   : Name of the object
    [in] pose   : SE3 placement of the capsule (pin.SE3)
    [in] radius : capsule radius
    [in] length : capsule length
    [in] color  : capsule color
    """
    tf = list(pin.SE3ToXYZQUAT(pose))
    if (gui.nodeExists(name)):
        gui.deleteNode(name, True)
    gui.addCapsule(name, radius, length, color)
    gui.applyConfiguration(name, tf)
    gui.refresh()

def display_landmark(gui, nodeName, size):
    gui.addLandmark(nodeName, size)

def clear_viewer(gui) :
    """
    Clear all nodes from viewer but robot
    """
    exclude = ['gepetto-gui', 'hpp-gui', 'hpp-gui/floor', 'python-pinocchio', 'world', 'world/pinocchio']
    for node in gui.getNodeList():
        if(node in exclude or 'world/pinocchio' in node):
            pass
        else:
            gui.deleteNode(node, True)
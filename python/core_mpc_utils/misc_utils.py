

import argparse

def parse_OCP_script(argv=None):
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--robot_name", type=str, default='iiwa', help="Name of the robot")
    PARSER.add_argument('--PLOT', action='store_true', default=False, help="Plot OCP solution")
    PARSER.add_argument('--DISPLAY', action='store_true', default=False, help="Animate solution in Gepetto Viewer")
    return PARSER.parse_args(argv)


def parse_MPC_script(argv=None):
    PARSER = argparse.ArgumentParser()
    # PARSER.add_argument("--robot_name", type=str, default='iiwa', help="Name of the robot")
    # PARSER.add_argument('--simulator', type=str, default='bullet', help="Name of the simulator")
    # PARSER.add_argument('--PLOT_INIT', action='store_true', default=False, help="Plot warm-start solution")
    PARSER.add_argument('--SAVE_DIR', type=str, default='/tmp/', help="Where to save the sim data")
    return PARSER.parse_args(argv)

import numpy as np
import pinocchio as pin
import eigenpy
from numpy.linalg import pinv
import time

import importlib
found_robot_properties_kuka_pkg = importlib.util.find_spec("robot_properties_kuka") is not None
found_robot_properties_talos_pkg = importlib.util.find_spec("robot_properties_talos") is not None
# found_example_robot_data_pkg = importlib.util.find_spec("example_robot_data") is not None


from croco_mpc_utils.utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger


if(not found_robot_properties_kuka_pkg):
    logger.error("Either install robot_properties_kuka, or directly build the pinocchio robot wrapper from URDF file.")
if(not found_robot_properties_kuka_pkg):
    logger.error("Either install robot_properties_talos, or directly build the pinocchio robot wrapper from URDF file.")


SUPPORTED_ROBOTS = ['iiwa', 'iiwa_reduced', 'talos_arm', 'talos_reduced', 'talos_full']


# Returns pinocchio robot wrapper
def load_robot_wrapper(robot_name):
    logger.info('Loading robot wrapper : "'+str(robot_name)+'"...')
    # Load iiwa robot wrapper 
    if(robot_name not in SUPPORTED_ROBOTS):
        logger.error('Unknown robot name ! Choose a robot in supported robots '+str(SUPPORTED_ROBOTS))
    else:
        # Load full iiwa wrapper
        if(robot_name == 'iiwa'):
            from robot_properties_kuka.config import IiwaConfig
            config = IiwaConfig()
            robot = config.buildRobotWrapper()
        # Load reduced iiwa wrapper
        if(robot_name == 'iiwa_reduced'):
            from robot_properties_kuka.config import IiwaReducedConfig
            controlled_joints = ['A1', 'A2', 'A3', 'A4']
            qref = np.array([0.1, 0.7, 0., 0.7, -0.5, 1.5, 0.]) 
            robot = IiwaReducedConfig.buildRobotWrapper(controlled_joints, qref)
        # Load talos left arm robot wrapper
        elif(robot_name == 'talos_arm'):
            from robot_properties_talos.config import TalosArmConfig
            robot = TalosArmConfig.buildRobotWrapper()
        # Load talos reduced robot wrapper
        elif(robot_name =='talos_reduced'):
            from robot_properties_talos.config import TalosReducedConfig
            robot = TalosReducedConfig.buildRobotWrapper()
        # Load talos full robot wrapper
        elif(robot_name == 'talos_full'):
            from robot_properties_talos.config import TalosFullConfig
            robot = TalosFullConfig.buildRobotWrapper()
            # import example_robot_data
            # robot = example_robot_data.load('talos') 
        return robot


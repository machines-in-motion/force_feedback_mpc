import argparse

def parse_OCP_script(argv=None):
    PARSER = argparse.ArgumentParser()
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

from croco_mpc_utils.utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger


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

import numpy as np

def moving_average(data, window_size):
    """
    Compute the moving average of a 1D array.
    
    Parameters:
    - data: 1D array-like, the input signal
    - window_size: int, size of the moving average window

    Returns:
    - smoothed_data: 1D numpy array of the same length as input
    """
    if window_size < 1:
        raise ValueError("Window size must be at least 1.")
    
    data = np.asarray(data)
    cumsum = np.cumsum(np.insert(data, 0, 0))
    smoothed = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
    
    # Pad the start to maintain original length
    pad_width = window_size // 2
    return np.pad(smoothed, (pad_width, data.size - smoothed.size - pad_width), mode='edge')

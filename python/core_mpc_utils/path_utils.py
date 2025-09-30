import importlib_resources
import yaml
import os
import re 

from croco_mpc_utils.utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger


# Load a yaml file (e.g. simu config file)
def load_yaml_file(yaml_file):
    '''
    Load config file (yaml)
    '''
    with open(yaml_file) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data 



# Returns urdf path of a kuka robot 
def get_urdf_path(robot_name, robot_family='kuka'):
    # Get config file
    with importlib_resources.path("robot_properties_"+robot_family, "config.py") as p:
        pkg_dir = p.parent.absolute()
    urdf_path = pkg_dir/"robot_properties_kuka"/(robot_name + ".urdf")
    return str(urdf_path)



# Returns urdf path of a kuka robot 
def get_mesh_dir(robot_family='kuka'):
    # Get config file
    with importlib_resources.path("robot_properties_"+robot_family, "config.py") as p:
        pkg_dir = p.parent.absolute()
    urdf_dir = pkg_dir
    return str(urdf_dir)



# Load config file
def load_config_file(script_name, robot_name='iiwa'):
    '''
    Loads YAML config file in demos/task_dir/config as a python dictionary
    '''
    task_name         = re.split("\.", os.path.basename(script_name))[0] # drop ".py"
    if('lpf' in task_name.lower()):
        task_dir = re.split("_", task_name)[1]                      # drop "_OCP" or "_MPC"
    elif('soft' in task_name.lower() and 'aug' not in task_name.lower()):
        task_dir = re.split("_", task_name)[1]                      # drop "_OCP" or "_MPC"
    elif('aug' in task_name.lower() and 'aug' in task_name.lower()):
        task_dir = re.split("_", task_name)[2]
    else:
        task_dir = re.split("_", task_name)[0]
    # config_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'../demos', 'config/'))
    # print(task_dir, task_name)
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'../demos', task_dir, 'config/'))
    config_name = robot_name+'_'+task_name
    config_file = config_path+"/"+config_name+".yml"
    print("")
    logger.info("Loading config file '"+str(config_file)+"'...")
    print("")
    config = load_yaml_file(config_file)
    return config, config_name


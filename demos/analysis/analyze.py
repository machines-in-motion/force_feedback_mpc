import sys
from . import analysis_utils
from force_feedback_mpc.core_mpc_utils import data_utils, plot_utils
import numpy as np

from core_mpc_utils.misc_utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, log_level_name=GLOBAL_LOG_LEVEL, USE_LONG_FORMAT=GLOBAL_LOG_FORMAT).logger


def main(npz_path=None, FILTER=1, PLOT=False):
  
  # load plot data
  if npz_path is None:
    logger.error("Please specify a DATASET to analyze !")
  else:
    # Extract data
    LPF = 'LPF' in npz_path
    print(" Extracting data (LPF = "+str(LPF)+")")
    data = data_utils.extract_plot_data_from_npz(npz_path, LPF=LPF)    

    # Compute absolute tracking errors |mea - ref|
     # EE tracking
    Np = data['N_plan'] ; Ns = data['N_simu']
    lin_pos_ee_ref = analysis_utils.linear_interpolation(data['lin_pos_ee_ref'], int((Ns+1)/Np))
    lin_err_ee_xyz = np.zeros(data['lin_pos_ee_mea'].shape)
    for i in range( lin_pos_ee_ref.shape[0] ):
        lin_err_ee_xyz[i,:] = np.abs( data['lin_pos_ee_mea'][i,:] - lin_pos_ee_ref[i,:])
    # Maximum (peak) absolute error along x,y,z
    lin_err_ee_max_x   = np.max(lin_err_ee_xyz[:,0])
    lin_err_ee_max_y   = np.max(lin_err_ee_xyz[:,1])
    lin_err_ee_max_z   = np.max(lin_err_ee_xyz[:,2])
    # Cumulative absolute error
    lin_err_ee_xyz_sum = np.sum(lin_err_ee_xyz, axis=0)
    # Average absolute error 
    lin_err_ee_xyz_avg = lin_err_ee_xyz_sum / Ns
    # Logs
    print("\n")
    print("EE tracking errors : \n")
    # print(" Peak abs. EE error along x   : "+str(lin_err_ee_max_x))
    # print(" Peak abs. EE error along y   : "+str(lin_err_ee_max_y))
    # print(" Peak abs. EE error along z   : "+str(lin_err_ee_max_z))
    # print(" Cumulative abs. EE xyz error : "+str(lin_err_ee_xyz_sum))
    print(" Average abs. EE xyz error      : "+str(lin_err_ee_xyz_avg))
    print(" Average abs. EE xyz error norm : "+str(np.linalg.norm(lin_err_ee_xyz_avg)))
    print("\n")
    print("----------------------------------")
     # Force tracking
    Np = data['N_plan'] ; Ns = data['N_simu']
    f_ee_ref_z = -20
    f_ee_err_z = np.zeros(data['f_ee_mea'].shape[0])
    for i in range( Ns ):
        f_ee_err_z[i] = np.abs( data['f_ee_mea'][i,2] - f_ee_ref_z)
    # Maximum (peak) absolute error along x,y,z
    f_ee_err_max_z   = np.max(f_ee_err_z)
    # Cumulative absolute error
    f_ee_err_sum_z = np.sum(f_ee_err_z, axis=0)
    # Average absolute error 
    f_ee_err_avg_z = f_ee_err_sum_z / Ns
    # Logs
    print("\n")
    print("FORCE tracking errors : \n")
    print(" Peak abs. FORCE error along z : "+str(f_ee_err_max_z))
    # print(" Cumulative abs. FORCE z error : "+str(f_ee_err_sum_z))
    print(" Average abs. FORCE z error    : "+str(f_ee_err_avg_z))
    print("\n")

    bool_contact = np.isclose(data['f_ee_mea'][:,2], np.zeros(data['f_ee_mea'][:,2].shape), rtol=1e-6)
    cycles_not_in_contact = (100.*np.count_nonzero(bool_contact))/Ns
    print(" # Cycles not in contact       : "+str(cycles_not_in_contact))
    print("\n")


    # Smooth if necessary
    if(FILTER > 0):
        data['q_mea'] = analysis_utils.moving_average_filter(data['q_mea'].copy(), FILTER)
        data['v_mea'] = analysis_utils.moving_average_filter(data['v_mea'].copy(), FILTER)
        if(LPF):
            data['tau_mea'] = analysis_utils.moving_average_filter(data['tau_mea'].copy(), FILTER)
        data['lin_pos_ee_mea'] = analysis_utils.moving_average_filter(data['lin_pos_ee_mea'].copy(), FILTER)
        data['ang_pos_ee_mea'] = analysis_utils.moving_average_filter(data['ang_pos_ee_mea'].copy(), FILTER)
        data['lin_vel_ee_mea'] = analysis_utils.moving_average_filter(data['lin_vel_ee_mea'].copy(), FILTER)
        data['ang_vel_ee_mea'] = analysis_utils.moving_average_filter(data['ang_vel_ee_mea'].copy(), FILTER)
        data['f_ee_mea']   = analysis_utils.moving_average_filter(data['f_ee_mea'].copy(), FILTER) 


    # # Plot
    # WHICH_PLOTS = ['f', 'ee']
    # if(LPF):
    #     plot_utils.plot_mpc_results_LPF(data, which_plots=WHICH_PLOTS, PLOT_PREDICTIONS=False)
    # else:
    #     plot_utils.plot_mpc_results(data, which_plots=WHICH_PLOTS, PLOT_PREDICTIONS=False)



if __name__=='__main__':
    if len(sys.argv) <= 2:
        print("Usage: python analyze.py [arg1: npz_path (str)] [arg2: FILTER (int)]")
        sys.exit(0)
    sys.exit(main(sys.argv[1], int(sys.argv[2])))



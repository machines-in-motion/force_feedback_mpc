
from core_mpc_utils.misc_utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger

from core_mpc.data import load_data

prefix = '/home/skleff/force-feedback/data/soft_contact_article'
sd1 = load_data(prefix+'/iiwa_sanding_MPC_bullet__BIAS=True_NOISE=True_DELAY=True_Fp=1.0_Fc=1.0_Fs5.0.npz')
sd2 = load_data(prefix+'/iiwa_LPF_sanding_MPC_bullet__BIAS=True_NOISE=True_DELAY=True_Fp=1.0_Fc=1.0_Fs5.0.npz')
sd3 = load_data(prefix+'/iiwa_aug_soft_sanding_MPC_bullet__BIAS=True_NOISE=True_DELAY=True_Fp=1.0_Fc=1.0_Fs5.0.npz')

d1 = sd1.extract_data(frame_of_interest='contact')
d2 = sd2.extract_data(frame_of_interest='contact')
d3 = sd3.extract_data(frame_of_interest='contact')

which_plots         = ['f'] 
PLOT_PREDICTIONS    = True 
pred_plot_sampling  = 100 
SAVE                = False
SAVE_DIR            = None 
SAVE_NAME           = None
SHOW                = True
AUTOSCALE           = False
args = [which_plots, PLOT_PREDICTIONS, pred_plot_sampling, SAVE, SAVE_DIR, SAVE_NAME, SHOW, AUTOSCALE]

sd1.plot_mpc_results(d1, *args)
sd2.plot_mpc_results(d2, *args)
sd3.plot_mpc_results(d3, *args)
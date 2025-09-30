
import numpy as np

from croco_mpc_utils.utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger


class AntiAliasingFilter:
  def __init__(self, filter_type='iir'):
      supported_filter_types = ["iir", "moving_average"]
      try: assert(filter_type in supported_filter_types)
      except: logger.error("AntiAliasingFilter type should be in "+str(supported_filter_types))
      self.filter_type = filter_type
      logger.info("Created AntiAliasingFilter of type "+str(self.filter_type))

  def iir(self, last_output, input_data, fc=100, fs=1000):
      '''
       Infinite Impulse Response (IIR) filter 
        output = g*last_output + (1-g)*input_data
        cutoff frequency say 1kHz
        exp(- 2 pi f 1/5000)
      '''
      gamma = np.exp(-2*np.pi*fc/fs)
      return gamma*last_output + (1-gamma)*input_data
      
     
  def step(self, nb1, nb2, freq1, freq2, data_at_freq2):
      '''
      nb1   : number of cycles elapsed at freq1
      nb2   : number of cycles elpased at freq2
      freq1 : low frequency
      freq2 : high frequency 
      '''
      # Using moving average

      try: 
          assert(freq1 <= freq2)
      except:
          logger.error("freq1 must be <= freq2 !!!")
      filterSize = min(nb1, int(freq2/freq1))
      if(filterSize == 0):
          return data_at_freq2[0]
      else:
        data_at_freq1 = self.moving_average(data_at_freq2[nb2-filterSize:nb2], filterSize)
        return data_at_freq1[-1]     
      
  def moving_average(self, input_data, filter_size=1):
      '''
      moving average on 1st dimension of some array ((N,n))
      '''
      output_data = input_data.copy() 
      # Filter them with moving average
      for i in range( output_data.shape[0] ):
          n_sum = min(i, filter_size)
          # Sum up over window
          for k in range(n_sum):
              output_data[i,:] += input_data[i-k-1, :]
          # Divide by number of samples
          output_data[i,:] = output_data[i,:] / (n_sum + 1)
      return output_data 




class LowLevelTorqueController:
    def __init__(self, config, nu, use):
      '''
      Takes in a reference torque (e.g. from MPC) and computes the motor torque 
      to be sent to the robot's motors, optionally using a PID+ controller
      '''
      self.config = config
      self.nu = nu
      # Simulate low-level torque control 
      self.TORQUE_TRACKING   = use           
      logger.info("Created LowLevelTorqueController(TORQUE_TRACKING="+str(self.TORQUE_TRACKING)+").")
      # PID gains for inner control loop 
      self.gain_P = self.config['Kp_low']*np.eye(nu)      
      self.gain_I = self.config['Ki_low']*np.eye(nu)
      self.gain_D = self.config['Kd_low']*np.eye(nu)
      self.err_I  = np.zeros(nu)

    def reset_integral_error(self):
      '''
      Reset integral error to 0
      '''
      self.err_I = np.zeros(self.nu)

    def step(self, reference_torque, measured_torque, measured_torque_derivative):
      '''
      Computes the motor torque
       Input: 
        reference_torque           : desired torque computed by e.g. the MPC 
        measured_torque            : torque measured after actuation effects
        measured_torque_derivative : derivative of the measured torque
       Output:
        motor_torque : torque to be sent to the robot's motor
      '''
      # Feedforward 
      motor_torque = reference_torque.copy()
      # Optional PID feedback term 
      if(self.TORQUE_TRACKING and len(measured_torque) !=0):
          # print(self.TORQUE_TRACKING)
          self.err_P = measured_torque - reference_torque             
          self.err_I += self.err_P
          self.err_D = measured_torque_derivative                 
          motor_torque -= 1.*self.gain_P.dot(self.err_P) 
          motor_torque -= 0.2*self.gain_I.dot(self.err_I)
          motor_torque -= 0.0001*self.gain_D.dot(self.err_D)
      return motor_torque



class ActuationModel:

    def __init__(self, config, nu, SEED=1):
        '''
        Actuation model with parameters defined in config YAML file
        
        The actuation model takes in a motor torque as control input and 
        it computes the combined effects of transmission, noise, friction,
        uncertainty etc. and it returns as an output the measured joint torque.

        It can simulates, depending on the sim parameters 
         - delay
         - noise
         - affine bias
         - dry friction
        '''
        np.random.seed(SEED)
        self.config = config
        self.nu = nu
        # Scaling of desired torque
        self.alpha = np.random.uniform(low=self.config['alpha_min'], high=self.config['alpha_max'], size=(nu,))
        self.beta  = np.random.uniform(low=self.config['beta_min'], high=self.config['beta_max'], size=(nu,))
        # Delays 
        self.delay_sim_cycle = int(self.config['delay_sim_cycle'])       # in simu cycles
        self.buffer_sim      = []                                        # buffer for measured torque delayed by e.g. actuation and/or sensing 
        # Noise
        self.var_u = np.asarray(self.config['var_u'])
        # Actuation model options
        self.DELAY_SIM         = config['DELAY_SIM']                            # Add delay in reference torques (low-level)
        self.SCALE_TORQUES     = config['SCALE_TORQUES']                        # Affinescaling of reference torque
        self.NOISE_TORQUES     = config['NOISE_TORQUES']                        # Moving average smoothing of reference torques
        self.STATIC_FRICTION   = config['STATIC_FRICTION']                      # Simulate static friction
        self.VISCOUS_FRICTION  = config['VISCOUS_FRICTION']                     # Simulate viscous friction
        self.tau_sf_max        = config['static_friction_max_torque']           # Max. static friction torque
        self.tau_vf_slope      = config['viscous_friction_slope']               # Slope of the viscous friction torque w.r.t. velocity
        logger.info("Created ActuationModel(DELAY_SIM="+str(self.DELAY_SIM)+
                    ", SCALE_TORQUES="+str(self.SCALE_TORQUES)+
                    ", NOISE_TORQUES="+str(self.NOISE_TORQUES)+
                    ", STATIC_FRICTION="+str(self.STATIC_FRICTION)+
                    ", VISCOUS_FRICTION="+str(self.VISCOUS_FRICTION)+").")
        if(self.SCALE_TORQUES):
          logger.info("Torques scaling : alpha = "+str(self.alpha)+" | beta = "+str(self.beta))

    def step(self, motor_torque, joint_vel=None):
        '''
        Transforms motor torque into a measured torque
        
        See the paper https://la.disneyresearch.com/wp-content/uploads/Toward-Controlling-a-KUKA-LBR-IIWA-for-Interactive-Tracking-Paper.pdf
          1. FRI receives desired torque tau_d (at 1kHz)
          2. Motor board computes motor torque tau_m = tau_d + PID(tau_d, tau_j)
          tau_j reflects the combined effects of motor inertia, friction, transmission etc. --> approximated by delay, scaling, noise and filtering

        Simulates (optionally) 
         - delay tau_mot(i) = tau_mot(i-delay) due to e.g. transmission
         - affine bias on torques a*tau_mot(i) + b due to model uncertainty
         - Gaussian noise on tau_mot(i) due to torque sensor noise
         - static friction on tau_mot(i) += tanh(a*v)
        Input: 
          i            : current simulation cycle number
          motor_torque : desired torque by motor
          joint_vel    : joint velocity (used for static / viscous friction)
        '''
        # Perfect actuation if all options = False
        measured_torque = motor_torque.copy()
        # Delay of ref  
        if(self.DELAY_SIM and len(measured_torque) !=0):
          self.buffer_sim.append(measured_torque)            
          if(len(self.buffer_sim) < self.delay_sim_cycle):    
            pass
          else:                          
            measured_torque = self.buffer_sim.pop(-self.delay_sim_cycle)
        # Affine scaling of ref
        if(self.SCALE_TORQUES and len(measured_torque) !=0):
          measured_torque = self.alpha * measured_torque + self.beta
        # Gaussian noise on ref  
        if(self.NOISE_TORQUES and len(measured_torque) !=0):
            noise_u = np.random.normal(0., self.var_u, self.nu)
            measured_torque += noise_u
        #  Static friction
        if(self.STATIC_FRICTION and len(measured_torque) !=0):
           sf_torque = self.tau_sf_max*np.tanh(10*joint_vel) 
          #  print("static friction = ", sf_torque)
           measured_torque -= sf_torque
        if(self.VISCOUS_FRICTION and len(measured_torque) !=0):
           vf_torque = self.tau_vf_slope * joint_vel
          #  print("viscous friction = ", vf_torque)
           measured_torque -= vf_torque
        return measured_torque



class CommunicationModel:

    def __init__(self, config):
        '''
        Communication model with parameters defined in config YAML file
        Simulates (optionally)
         - delay in OCP solution (x*,u*)
        '''
        self.config = config
        # Delay OCP computation
        self.x_buffer_OCP = []                                           # buffer for desired states delayed by OCP computation time
        self.u_buffer_OCP = []                                           # buffer for desired controls delayed by OCP computation time
        # Sensing model options
        self.DELAY_OCP         = config['DELAY_OCP']                     # Add delay in OCP solution (i.e. ~1ms resolution time)
        logger.info("Created CommunicationModel(DELAY_OCP="+str(self.DELAY_OCP)+").")

    def step(self, predicted_state, current_control):
        '''
        Delays input predicted state and current control by 
        using a buffer. Returns the delayed input variables
        Simulates (optionally)
         - delay in OCP solution (x*,u*)
        '''
        # Delay OCP solution due to computation time
        if(self.DELAY_OCP):
          delay = int(self.config['delay_OCP_ms'] * 1e-3 * self.config['plan_freq']) # in planning cycles
          self.x_buffer_OCP.append(predicted_state)
          self.u_buffer_OCP.append(current_control)
          if(len(self.x_buffer_OCP) < delay): 
            pass
          else:                            
            predicted_state = self.x_buffer_OCP.pop(-delay)
          if(len(self.u_buffer_OCP) < delay): 
            pass
          else:
            current_control = self.u_buffer_OCP.pop(-delay)
        return predicted_state, current_control



class SensorModel:

    def __init__(self, config, naug=0, SEED=1):
        '''
        Sensing model with parameters defined in config YAML file
        Simulates (optionally)
         - gaussian noise on measured state
         naug : for augmented state (lpf, soft contact, etc..). 0 by default
        '''
        np.random.seed(SEED)
        self.config = config
        self.nq = len(config['q0']) 
        self.nv = len(config['dq0'])
        self.naug = naug
        # White noise on desired torque and measured state
        self.var_q = np.asarray(self.config['var_q'])
        self.var_v = np.asarray(self.config['var_v'])
        if(self.naug > 0):
          self.var_aug = np.asarray(self.config['var_aug'])[:naug]
        # Sensing model options
        self.NOISE_STATE       = config['NOISE_STATE']                   # Add Gaussian noise on the measured state 
        logger.info("Created SensorModel(NOISE_STATE="+str(self.NOISE_STATE)+").")

    def step(self, measured_state):
        '''
        Transforms simulator state into a measured state
        Simulates (optionally)
         - gaussian noise on measured state
        '''
        # Optional Gaussian noise on measured state 
        if(self.NOISE_STATE):
          noise_q = np.random.normal(0., self.var_q, self.nq)
          noise_v = np.random.normal(0., self.var_v, self.nv)
          if(self.naug > 0):
            noise_tau = np.random.normal(0., self.var_aug, self.naug)
            measured_state += np.concatenate([noise_q, noise_v, noise_tau]).T
          else:
            measured_state += np.concatenate([noise_q, noise_v]).T
        return measured_state


class VelocityEstimator:
  def __init__(self, dt): #, order=1):
    self.dt = dt
    self.q_prev = None
    # self.q_prev2 
    # self.order = order
    # self.nv = len(self.q_prev)
  
  def FD1_estimate(self, q):
    if(self.q_prev is None):
      v = np.zeros(len(q))
    else:
      v = (q - self.q_prev)/self.dt
    self.q_prev = q
    return v

    # else:
  #   return (q - self.q_prev)/self.dt

  # def FD2_estimate(self, q, dt=self.dt):
  #   return q - q_prev


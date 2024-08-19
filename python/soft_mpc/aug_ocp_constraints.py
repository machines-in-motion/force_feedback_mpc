"""
@package force_feedback
@file soft_mpc/ocp.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft.
@date 2020-05-18
@brief Initializes the OCP + solver (visco-elastic contact)
"""

import crocoddyl
import numpy as np

import force_feedback_mpc
from soft_mpc.aug_ocp import OptimalControlProblemSoftContactAugmented

from croco_mpc_utils import pinocchio_utils as pin_utils

from croco_mpc_utils.utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger


class OptimalControlProblemSoftContactAugmentedWithConstraints(OptimalControlProblemSoftContactAugmented):
  '''
  Helper class for soft contact (augmented) OCP setup with Crocoddyl
  '''
  def __init__(self, robot, config):
    '''
    Override base class constructor if necessary
    '''
    super().__init__(robot, config)
  
  def check_config(self):
    '''
    Override base class checks if necessary
    '''
    super().check_config()
    self.check_attribute('Kp')
    self.check_attribute('Kv')
    self.check_attribute('oPc_offset')
    self.check_attribute('pinRefFrame')
    self.check_attribute('contactType')

  def create_constraint_model_manager(self, state, actuation, node_id):
    '''
    Initialize a constraint model manager and adds constraints to it 
    '''
    constraintModelManager = crocoddyl.ConstraintModelManager(state, actuation.nu)
    # State limits
    if('stateBox' in self.WHICH_CONSTRAINTS and node_id != 0):
      stateBoxConstraint = self.create_state_constraint(state, actuation)   
      constraintModelManager.addConstraint('stateBox', stateBoxConstraint)
    # Control limits
    if('ctrlBox' in self.WHICH_CONSTRAINTS):
      ctrlBoxConstraint = self.create_ctrl_constraint(state, actuation)
      constraintModelManager.addConstraint('ctrlBox', ctrlBoxConstraint)
    # End-effector position limits
    if('translationBox' in self.WHICH_CONSTRAINTS and node_id != 0):
      translationBoxConstraint = self.create_translation_constraint(state, actuation)
      constraintModelManager.addConstraint('translationBox', translationBoxConstraint)
    # if(node_id != 0):
    #   constraintModelManager.constraints['stateBox'].constraint.updateBounds(np.asarray(self.stateLowerLimit), np.asarray(self.stateUpperLimit))
    return constraintModelManager

  def create_differential_action_model(self, state, actuation, softContactModel, constraintModelManager):
    '''
    Initialize a differential action model with soft contact
    '''
    # 3D contact
    if(softContactModel.nc == 3):
      dam = force_feedback_mpc.DAMSoftContact3DAugmentedFwdDynamics(state, 
                              actuation, 
                              crocoddyl.CostModelSum(state, nu=actuation.nu),
                              softContactModel.frameId, 
                              softContactModel.Kp,
                              softContactModel.Kv,
                              softContactModel.oPc,
                              softContactModel.pinRefFrame,
                              constraintModelManager )
    # 1D contact
    elif(softContactModel.nc == 1):
      dam = force_feedback_mpc.DAMSoftContact1DAugmentedFwdDynamics(state, 
                              actuation, 
                              crocoddyl.CostModelSum(state, nu=actuation.nu),
                              softContactModel.frameId, 
                              softContactModel.Kp,
                              softContactModel.Kv,
                              softContactModel.oPc,
                              softContactModel.pinRefFrame,
                              softContactModel.maskType,
                              constraintModelManager )
    else:
      logger.error("softContactModel.nc = 3 or 1")
    return dam

  def init_running_model_derived(self, state, actuation, runningModel, softContactModel):
    '''
  Populate running model with costs and contacts
    '''
    self.init_running_model(state, actuation, runningModel, softContactModel)
    if('forceBox' in self.WHICH_CONSTRAINTS):
      runningModel.with_force_constraint = True
      self.check_attribute('forceLowerLimit')
      self.check_attribute('forceUpperLimit')
      runningModel.force_lb = np.asarray(self.forceLowerLimit)
      runningModel.force_ub = np.asarray(self.forceUpperLimit)

  def init_terminal_model_derived(self, state, actuation, runningModel, softContactModel):
    '''
  Populate running model with costs and contacts
    '''
    self.init_terminal_model(state, actuation, runningModel, softContactModel)
    if('forceBox' in self.WHICH_CONSTRAINTS):
      runningModel.with_force_constraint = True
      self.check_attribute('forceLowerLimit')
      self.check_attribute('forceUpperLimit')
      runningModel.force_lb = np.asarray(self.forceLowerLimit)
      runningModel.force_ub = np.asarray(self.forceUpperLimit)


  def initialize(self, y0, softContactModel, callbacks=False):
    '''
    Initializes OCP and  solver from config parameters and initial state
    Soft contact (visco-elastic) augmented formulation, i.e. visco-elastic
    contact force is part of the state . Supported 3D formulation only for now
      INPUT: 
          y0                : initial state of shooting problem
          softContactModel  : SoftContactModel3D (see in utils)
          callbacks         : display Crocoddyl's DDP solver callbacks
      OUTPUT:
         solver

     A cost term on a variable z(x,u) has the generic form w * a( r( z(x,u) - z0 ) )
     where w <--> cost weight, e.g. 'stateRegWeight' in config file
           r <--> residual model depending on some reference z0, e.g. 'stateRegRef'
                  When ref is set to 'DEFAULT' in YAML file, default references hard-coded here are used
           a <--> weighted activation, with weights e.g. 'stateRegWeights' in config file 
           z <--> can be state x, control u, frame position or velocity, contact force, etc.
    ''' 
    
  # State and actuation models
    state     = crocoddyl.StateMultibody(self.rmodel)
    actuation = crocoddyl.ActuationModelFull(state)
    
    
  # Create IAMs
    runningModels = []
    for i in range(self.N_h):  
      # Create constraint manager and constraints
        constraintModelManager = self.create_constraint_model_manager(state, actuation, i)
      # Create DAM (Contact or FreeFwd), IAM LPF and initialize costs+contacts+constraints
        dam = self.create_differential_action_model(state, actuation, softContactModel, constraintModelManager) 
        print(constraintModelManager.g_lb)
        runningModels.append(force_feedback_mpc.IAMSoftContactAugmented( dam, self.dt ))
        self.init_running_model_derived(state, actuation, runningModels[i], softContactModel)
        
    # Terminal model
    constraintModelManager = self.create_constraint_model_manager(state, actuation, self.N_h)
    dam_t = self.create_differential_action_model(state, actuation, softContactModel, constraintModelManager)  
    terminalModel = force_feedback_mpc.IAMSoftContactAugmented( dam_t, 0. )
    self.init_terminal_model_derived(state, actuation, terminalModel, softContactModel)
    
    logger.info("Created IAMs.")  


  # Create the shooting problem
    problem = crocoddyl.ShootingProblem(y0, runningModels, terminalModel)


  # Finish
    self.success_log(softContactModel)
    
    return problem
  #   self.check_attribute('with_callbacks')
  #   self.check_attribute('use_filter_ls')
  #   self.check_attribute('filter_size')
  #   self.check_attribute('warm_start')
  #   self.check_attribute('termination_tol')
  #   self.check_attribute('max_qp_iters')
  #   self.check_attribute('qp_termination_tol_abs')
  #   self.check_attribute('qp_termination_tol_rel')
  #   self.check_attribute('warm_start_y')
  #   self.check_attribute('reset_rho')
  #   ddp.with_callbacks = self.with_callbacks
  #   ddp.use_filter_ls = self.use_filter_ls
  #   ddp.filter_size = self.filter_size
  #   ddp.warm_start = self.warm_start
  #   ddp.termination_tol = self.termination_tol
  #   ddp.max_qp_iters = self.max_qp_iters
  #   ddp.eps_abs = self.qp_termination_tol_abs
  #   ddp.eps_rel = self.qp_termination_tol_rel
  #   ddp.warm_start_y = self.warm_start_y
  #   ddp.reset_rho = self.reset_rho
  
  # # Callbacks
  #   if(callbacks):
  #     ddp.setCallbacks([crocoddyl.CallbackLogger(),
  #                       crocoddyl.CallbackVerbose()])
  
  # # Warm start : initial state + gravity compensation
  #   ddp.xs = [y0 for i in range(self.N_h+1)]
  #   fext0 = softContactModel.computeExternalWrench_(self.rmodel, y0[:self.nq], y0[:self.nv])
  #   ddp.us = [pin_utils.get_tau(y0[:self.nq], y0[:self.nv], np.zeros(self.nv), fext0, self.rmodel, np.zeros(self.nq)) for i in range(self.N_h)] #ddp.problem.quasiStatic(xs_init[:-1])

  # # Finish
  #   logger.info("OCP is ready !")
  #   # logger.info(  "USE_SOBEC_BINDINGS = "+str(USE_SOBEC_BINDINGS))
  #   logger.info("    COSTS        = "+str(self.WHICH_COSTS))
  #   logger.info("    CONSTRAINTS  = "+str(self.WHICH_CONSTRAINTS))
  #   logger.info("    SOFT CONTACT MODEL [ oPc="+str(softContactModel.oPc)+\
  #     " , Kp="+str(softContactModel.Kp)+\
  #       ', Kv='+str(softContactModel.Kv)+\
  #       ', pinRefFrame='+str(softContactModel.pinRefFrame)+']')
    
  #   return ddp
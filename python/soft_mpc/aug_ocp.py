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

from croco_mpc_utils.ocp_core import OptimalControlProblemAbstract
from croco_mpc_utils import pinocchio_utils as pin_utils

from croco_mpc_utils.utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger



class OptimalControlProblemSoftContactAugmented(OptimalControlProblemAbstract):
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

  def parse_constraints(self):
    '''
    Parses the YAML dict of constraints and count them
    '''
    if(not hasattr(self, 'WHICH_CONSTRAINTS')):
      self.nb_constraints = 0
    else:
      if('None' in self.WHICH_CONSTRAINTS):
        self.nb_constraints = 0
      else:
        self.nb_constraints = len(self.WHICH_CONSTRAINTS)

  def create_constraint_model_manager(self, state, actuation, node_id):
    '''
    Initialize a constraint model manager and adds constraints to it 
    '''
    constraintModelManager = crocoddyl.ConstraintModelManager(state, actuation.nu)
    # State limits
    if('stateBox' in self.WHICH_CONSTRAINTS and node_id != 0):
      logger.warning("! ! ! ! ! ! ! ! ! ! ! ! ! ! !")
      logger.warning("UNDEFINED BEHAVIOR FOR STATE CONSTRAINTS in SOFT CONTACT")
      logger.warning("! ! ! ! ! ! ! ! ! ! ! ! ! ! !")
      stateBoxConstraint = self.create_state_constraint(state, actuation)   
      constraintModelManager.addConstraint('stateBox', stateBoxConstraint)
    # Control limits
    if('ctrlBox' in self.WHICH_CONSTRAINTS and node_id != self.N_h):
      ctrlBoxConstraint = self.create_ctrl_constraint(state, actuation)
      constraintModelManager.addConstraint('ctrlBox', ctrlBoxConstraint)
    # End-effector position limits
    if('translationBox' in self.WHICH_CONSTRAINTS and node_id != 0):
      translationBoxConstraint = self.create_translation_constraint(state, actuation)
      constraintModelManager.addConstraint('translationBox', translationBoxConstraint)
    # # Contact force 
    # if('forceBox' in self.WHICH_CONSTRAINTS and node_id != 0 and node_id != self.N_h):
    #   logger.warning("! ! ! ! ! ! ! ! ! ! ! ! ! ! !")
    #   logger.warning("Force constraint not implemented yet for the SOFT CONTACT AUGMENTED MODEL")
    #   logger.warning("! ! ! ! ! ! ! ! ! ! ! ! ! ! !")
    #   forceBoxConstraint = self.create_force_constraint(state, actuation)
    #   constraintModelManager.addConstraint('forceBox', forceBoxConstraint)
    if('collisionBox' in self.WHICH_CONSTRAINTS):
        collisionBoxConstraints = self.create_collision_constraints(state, actuation)
        for i, collisionBoxConstraint in enumerate(collisionBoxConstraints):
          constraintModelManager.addConstraint('collisionBox_' + str(i), collisionBoxConstraint)

    return constraintModelManager
  
  def create_running_cost_model(self, state, actuation):
    '''
    Create running cost model sum
    '''
    costModelSum = crocoddyl.CostModelSum(state, nu=actuation.nu)
  # Create and add cost function terms to current IAM
    # State regularization 
    if('stateReg' in self.WHICH_COSTS):
      xRegCost = self.create_state_reg_cost(state, actuation)
      costModelSum.addCost("stateReg", xRegCost, self.stateRegWeight)
    # Control regularization
    if('ctrlReg' in self.WHICH_COSTS ):
      uRegCost = self.create_ctrl_reg_cost(state)
      costModelSum.addCost("ctrlReg", uRegCost, self.ctrlRegWeight)
    # State limits penalizationself.
    if('stateLim' in self.WHICH_COSTS):
      xLimitCost = self.create_state_limit_cost(state, actuation)
      costModelSum.addCost("stateLim", xLimitCost, self.stateLimWeight)
    # Control limits penalization
    if('ctrlLim' in self.WHICH_COSTS):
      uLimitCost = self.create_ctrl_limit_cost(state)
      costModelSum.addCost("ctrlLim", uLimitCost, self.ctrlLimWeight)
    # End-effector placement 
    if('placement' in self.WHICH_COSTS):
      framePlacementCost = self.create_frame_placement_cost(state, actuation)
      costModelSum.addCost("placement", framePlacementCost, self.framePlacementWeight)
    # End-effector velocity
    if('velocity' in self.WHICH_COSTS): 
      frameVelocityCost = self.create_frame_velocity_cost(state, actuation)
      costModelSum.addCost("velocity", frameVelocityCost, self.frameVelocityWeight)
    # Frame translation cost
    if('translation' in self.WHICH_COSTS):
      frameTranslationCost = self.create_frame_translation_cost(state, actuation)
      costModelSum.addCost("translation", frameTranslationCost, self.frameTranslationWeight)
    # End-effector orientation 
    if('rotation' in self.WHICH_COSTS):
      frameRotationCost = self.create_frame_rotation_cost(state, actuation)
      costModelSum.addCost("rotation", frameRotationCost, self.frameRotationWeight)
    
    return costModelSum

  def create_terminal_cost_model(self, state, actuation):
    '''
    Create terminal cost model sum
      All cost weights are scaled by the OCP integration step dt
      because the terminal model is not integrated in calc
    '''
    costModelSum_t = crocoddyl.CostModelSum(state, nu=actuation.nu)
  # Create and add terminal cost models to terminal IAM
  #   State regularization
    if('stateReg' in self.WHICH_COSTS):
      xRegCost = self.create_state_reg_cost(state, actuation)
      costModelSum_t.addCost("stateReg", xRegCost, self.stateRegWeightTerminal*self.dt)
    # State limits
    if('stateLim' in self.WHICH_COSTS):
      xLimitCost = self.create_state_limit_cost(state, actuation)
      costModelSum_t.addCost("stateLim", xLimitCost, self.stateLimWeightTerminal*self.dt)
    # EE placement
    if('placement' in self.WHICH_COSTS):
      framePlacementCost = self.create_frame_placement_cost(state, actuation)
      costModelSum_t.addCost("placement", framePlacementCost, self.framePlacementWeightTerminal*self.dt)
    # EE velocity
    if('velocity' in self.WHICH_COSTS):
      frameVelocityCost = self.create_frame_velocity_cost(state, actuation)
      costModelSum_t.addCost("velocity", frameVelocityCost, self.frameVelocityWeightTerminal*self.dt)
    # EE translation
    if('translation' in self.WHICH_COSTS):
      frameTranslationCost = self.create_frame_translation_cost(state, actuation)
      costModelSum_t.addCost("translation", frameTranslationCost, self.frameTranslationWeightTerminal*self.dt)
    # End-effector orientation 
    if('rotation' in self.WHICH_COSTS):
      frameRotationCost = self.create_frame_rotation_cost(state, actuation)
      costModelSum_t.addCost("rotation", frameRotationCost, self.frameRotationWeightTerminal*self.dt)
    
    return costModelSum_t
  
  def create_differential_action_model(self, state, actuation, costModelSum, softContactModel, constraintModelManager=None):
    '''
    Initialize a differential action model with soft contact
    '''
    # 3D contact
    if(softContactModel.nc == 3):
      # # If there is friction
      # if(self.check_attribute('mu')):
      #   self.check_attribute('eps')
      #   logger.warning("Simulate dynamic friction for lateral forces : mu="+str(self.mu)+", eps="+str(self.eps))
      #   dam = DAMSoft3DAugmentedFriction(state, 
      #                           actuation, 
      #                           crocoddyl.CostModelSum(state, nu=actuation.nu),
      #                           softContactModel.frameId, 
      #                           softContactModel.Kp,
      #                           softContactModel.Kv,
      #                           softContactModel.oPc,
      #                           softContactModel.pinRefFrame )
      #   dam.mu = self.mu
      #   dam.eps = self.eps
      # else:
      if(constraintModelManager is None):
        dam = force_feedback_mpc.DAMSoftContact3DAugmentedFwdDynamics(state, 
                                actuation, 
                                crocoddyl.CostModelSum(state, nu=actuation.nu),
                                softContactModel.frameId, 
                                softContactModel.Kp,
                                softContactModel.Kv,
                                softContactModel.oPc )
      else:
        dam = force_feedback_mpc.DAMSoftContact3DAugmentedFwdDynamics(state, 
                                actuation, 
                                crocoddyl.CostModelSum(state, nu=actuation.nu),
                                softContactModel.frameId, 
                                softContactModel.Kp,
                                softContactModel.Kv,
                                softContactModel.oPc,
                                constraintModelManager )
    elif(softContactModel.nc == 1):
      if(constraintModelManager is None):
        dam = force_feedback_mpc.DAMSoftContact1DAugmentedFwdDynamics(state, 
                                actuation, 
                                # crocoddyl.CostModelSum(state, nu=actuation.nu),
                                costModelSum,
                                softContactModel.frameId, 
                                softContactModel.Kp,
                                softContactModel.Kv,
                                softContactModel.oPc,
                                softContactModel.maskType )
      else:
        dam = force_feedback_mpc.DAMSoftContact1DAugmentedFwdDynamics(state, 
                                actuation, 
                                # crocoddyl.CostModelSum(state, nu=actuation.nu),
                                costModelSum,
                                softContactModel.frameId, 
                                softContactModel.Kp,
                                softContactModel.Kv,
                                softContactModel.oPc,
                                softContactModel.maskType,
                                constraintModelManager )
    else:
      logger.error("softContactModel.nc = 3 or 1")

    # set dynamics and cost reference frames
    dam.ref      = softContactModel.pinRefFrame
    dam.cost_ref = softContactModel.pinRefFrame
    logger.warning("Set dynamics and cost reference frames to "+str(softContactModel.pinRefFrame))
    return dam

  def finalize_running_model(self, state, actuation, runningModel, softContactModel, node_id):
    '''
    Populate running model with hard-coded costs 
    '''
  # Create and add cost function terms to current IAM
    # Control regularization (gravity)
    if('ctrlRegGrav' in self.WHICH_COSTS):
      self.check_attribute('ctrlRegGravWeight')
      runningModel.differential.with_gravity_torque_reg = True
      runningModel.differential.tau_grav_weight = self.ctrlRegGravWeight
    # Frame force cost
    if('force' in self.WHICH_COSTS):
      self.check_attribute('frameForceRef')
      self.check_attribute('frameForceWeight')
      if(softContactModel.nc == 3):
        forceRef = np.asarray(self.frameForceRef)[:3]
      else:
        forceRef = np.array([np.asarray(self.frameForceRef)[softContactModel.mask]])
      runningModel.differential.with_force_cost = True
      runningModel.differential.f_des = forceRef
      runningModel.differential.f_weight = np.asarray(self.frameForceWeight)
    # Frame force rate reg cost
    if('forceRateReg' in self.WHICH_COSTS):
      self.check_attribute('forceRateRegWeight')
      runningModel.differential.with_force_rate_reg_cost = True
      runningModel.differential.f_rate_reg_weight = np.asarray(self.forceRateRegWeight)
    if('forceBox' in self.WHICH_CONSTRAINTS and node_id != 0): 
      self.check_attribute('forceLowerLimit')
      self.check_attribute('forceUpperLimit')
      runningModel.with_force_constraint = True
      if(softContactModel.nc == 3):
        runningModel.force_lb = np.asarray(self.forceLowerLimit)[:3]
        runningModel.force_ub = np.asarray(self.forceUpperLimit)[:3]
      else:
        runningModel.force_lb = np.array([ np.asarray(self.forceLowerLimit)[softContactModel.mask] ])
        runningModel.force_ub = np.array([ np.asarray(self.forceUpperLimit)[softContactModel.mask] ])
    if('frictionCone' in self.WHICH_CONSTRAINTS): 
      # print("node_id = ", node_id)
      self.check_attribute('frictionCoefficient')
      self.check_attribute('frictionConeFrameName')
      try: 
        assert(softContactModel.nc == 3)
      except:
        logger.error("Soft contact model must be of dimension 3 to use the friction cone constraint")
      fid = self.rmodel.getFrameId(self.frictionConeFrameName)
      residual = force_feedback_mpc.ResidualModelFrictionConeAugmented(state, fid, self.frictionCoefficient, actuation.nu)
      # import pdb; pdb.set_trace()
      runningModel.friction_constraints = [residual]

  def finalize_terminal_model(self, terminalModel, softContactModel):
    ''' 
    Populate terminal model with hard-coded costs 
      All cost weights are scaled by the OCP integration step dt
      because the terminal model is not integrated in calc
    '''
  # Create and add terminal cost models to terminal IAM
    # Frame force cost
    if('force' in self.WHICH_COSTS):
      self.check_attribute('frameForceRef')
      self.check_attribute('frameForceWeight')
      if(softContactModel.nc == 3):
        forceRef = np.asarray(self.frameForceRef)[:3]
      else:
        forceRef = np.array([np.asarray(self.frameForceRef)[softContactModel.mask]])
      terminalModel.differential.with_force_cost = True  
      terminalModel.differential.f_des = forceRef
      terminalModel.differential.f_weight = np.asarray(self.frameForceWeightTerminal)*self.dt
    # Frame force rate reg cost
    if('forceRateReg' in self.WHICH_COSTS):
      self.check_attribute('forceRateRegWeight')
      terminalModel.differential.with_force_rate_reg_cost = True
      terminalModel.differential.f_rate_reg_weight = np.asarray(self.forceRateRegWeight)*self.dt
    if('forceBox' in self.WHICH_CONSTRAINTS):
      self.check_attribute('forceLowerLimit')
      self.check_attribute('forceUpperLimit')
      terminalModel.with_force_constraint = True
      # terminalModel.force_lb = np.asarray(self.forceLowerLimit)
      # terminalModel.force_ub = np.asarray(self.forceUpperLimit)

  def success_log(self, softContactModel):
    logger.info("OCP (SOFT) is ready !")
    logger.info("    COSTS   = "+str(self.WHICH_COSTS))
    logger.info("    SOFT CONTACT MODEL [ oPc="+str(softContactModel.oPc)+\
      " , Kp="+str(softContactModel.Kp)+\
        ', Kv='+str(softContactModel.Kv)+\
        ', pinRefFrame='+str(softContactModel.pinRefFrame)+']')
    if(self.nb_constraints > 0):
      logger.info("    CONSTRAINTS   = "+str(self.WHICH_CONSTRAINTS))
      
  def initialize(self, y0, softContactModel):
    '''
    Initializes OCP and  solver from config parameters and initial state
    Soft contact (visco-elastic) augmented formulation, i.e. visco-elastic
    contact force is part of the state . Supported 3D formulation only for now
      INPUT: 
          y0                : initial state of shooting problem
          softContactModel  : SoftContactModel3D (see in utils)
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
    state = crocoddyl.StateMultibody(self.rmodel)
    actuation = crocoddyl.ActuationModelFull(state)
    
  # Constraints or not ?
    self.parse_constraints()

  # Create IAMs
    runningModels = []
    for i in range(self.N_h):  
        print("NODE ", i)
      # Create DAM (Contact or FreeFwd), IAM LPF and initialize costs+contacts
        costModelSum = self.create_running_cost_model(state, actuation)
        if(self.nb_constraints == 0):
          dam = self.create_differential_action_model(state, actuation, costModelSum, softContactModel) 
        else:
        # Create constraint manager and constraints
          constraintModelManager = self.create_constraint_model_manager(state, actuation, i)
        # Create DAM & IAM and initialize costs+contacts+constraints
          dam = self.create_differential_action_model(state, actuation, costModelSum, softContactModel, constraintModelManager) 
        runningModels.append(force_feedback_mpc.IAMSoftContactAugmented( dam, self.dt ))
        self.finalize_running_model(state, actuation, runningModels[i], softContactModel, i)
        # self.init_running_model(state, actuation, runningModels[i], softContactModel)
        
    # Terminal model
    costModelSum_t = self.create_terminal_cost_model(state, actuation)
    if(self.nb_constraints == 0):
      dam_t = self.create_differential_action_model(state, actuation, costModelSum_t, softContactModel)  
    else:
      constraintModelManager = self.create_constraint_model_manager(state, actuation, self.N_h)
      dam_t = self.create_differential_action_model(state, actuation, costModelSum_t, softContactModel, constraintModelManager)  
    terminalModel = force_feedback_mpc.IAMSoftContactAugmented( dam_t, 0. )
    self.finalize_terminal_model(terminalModel, softContactModel)
    # self.init_terminal_model(state, actuation, terminalModel, softContactModel)
    
    logger.info("Created IAMs.")  


  # Create the shooting problem
    problem = crocoddyl.ShootingProblem(y0, runningModels, terminalModel)

  # Finish
    self.success_log(softContactModel)
    
    return problem

  # # Warm start : initial state + gravity compensation
  #   ddp.xs = [y0 for i in range(self.N_h+1)]
  #   fext0 = softContactModel.computeExternalWrench_(self.rmodel, y0[:self.nq], y0[:self.nv])
  #   ddp.us = [pin_utils.get_tau(y0[:self.nq], y0[:self.nv], np.zeros(self.nv), fext0, self.rmodel, np.zeros(self.nq)) for i in range(self.N_h)] #ddp.problem.quasiStatic(xs_init[:-1])
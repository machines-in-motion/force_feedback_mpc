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
    # Contact force 
    if('forceBox' in self.WHICH_CONSTRAINTS and node_id != 0 and node_id != self.N_h):
      forceBoxConstraint = self.create_force_constraint(state, actuation)
      constraintModelManager.addConstraint('forceBox', forceBoxConstraint)
    if('collisionBox' in self.WHICH_CONSTRAINTS):
        collisionBoxConstraints = self.create_collision_constraints(state, actuation)
        for i, collisionBoxConstraint in enumerate(collisionBoxConstraints):
          constraintModelManager.addConstraint('collisionBox_' + str(i), collisionBoxConstraint)

    return constraintModelManager
    
  def create_differential_action_model(self, state, actuation, softContactModel, constraintModelManager=None):
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
                                softContactModel.oPc,
                                softContactModel.pinRefFrame )
      else:
        dam = force_feedback_mpc.DAMSoftContact3DAugmentedFwdDynamics(state, 
                                actuation, 
                                crocoddyl.CostModelSum(state, nu=actuation.nu),
                                softContactModel.frameId, 
                                softContactModel.Kp,
                                softContactModel.Kv,
                                softContactModel.oPc,
                                softContactModel.pinRefFrame,
                                constraintModelManager )
    elif(softContactModel.nc == 1):
      if(constraintModelManager is None):
        dam = force_feedback_mpc.DAMSoftContact1DAugmentedFwdDynamics(state, 
                                actuation, 
                                crocoddyl.CostModelSum(state, nu=actuation.nu),
                                softContactModel.frameId, 
                                softContactModel.Kp,
                                softContactModel.Kv,
                                softContactModel.oPc,
                                softContactModel.pinRefFrame,
                                softContactModel.maskType )
      else:
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


  def init_running_model(self, state, actuation, runningModel, softContactModel):
    '''
  Populate running model with costs and contacts
    '''
  # Create and add cost function terms to current IAM
    # State regularization 
    if('stateReg' in self.WHICH_COSTS):
      xRegCost = self.create_state_reg_cost(state, actuation)
      runningModel.differential.costs.addCost("stateReg", xRegCost, self.stateRegWeight)
    # Control regularization
    if('ctrlReg' in self.WHICH_COSTS):
      uRegCost = self.create_ctrl_reg_cost(state)
      runningModel.differential.costs.addCost("ctrlReg", uRegCost, self.ctrlRegWeight)
    # Control regularization (gravity)
    if('ctrlRegGrav' in self.WHICH_COSTS):
      runningModel.differential.with_gravity_torque_reg = True
      runningModel.differential.tau_grav_weight = self.ctrlRegGravWeight
    # State limits penalizationself.
    if('stateLim' in self.WHICH_COSTS):
      xLimitCost = self.create_state_limit_cost(state, actuation)
      runningModel.differential.costs.addCost("stateLim", xLimitCost, self.stateLimWeight)
    # Control limits penalization
    if('ctrlLim' in self.WHICH_COSTS):
      uLimitCost = self.create_ctrl_limit_cost(state)
      runningModel.differential.costs.addCost("ctrlLim", uLimitCost, self.ctrlLimWeight)
    # End-effector placement 
    if('placement' in self.WHICH_COSTS):
      framePlacementCost = self.create_frame_placement_cost(state, actuation)
      runningModel.differential.costs.addCost("placement", framePlacementCost, self.framePlacementWeight)
    # End-effector velocity
    if('velocity' in self.WHICH_COSTS): 
      frameVelocityCost = self.create_frame_velocity_cost(state, actuation)
      runningModel.differential.costs.addCost("velocity", frameVelocityCost, self.frameVelocityWeight)
    # Frame translation cost
    if('translation' in self.WHICH_COSTS):
      frameTranslationCost = self.create_frame_translation_cost(state, actuation)
      runningModel.differential.costs.addCost("translation", frameTranslationCost, self.frameTranslationWeight)
    # End-effector orientation 
    if('rotation' in self.WHICH_COSTS):
      frameRotationCost = self.create_frame_rotation_cost(state, actuation)
      runningModel.differential.costs.addCost("rotation", frameRotationCost, self.frameRotationWeight)
    # Frame force cost
    if('force' in self.WHICH_COSTS):
      if(softContactModel.nc == 3):
        forceRef = np.asarray(self.frameForceRef)[:3]
      else:
        forceRef = np.array([np.asarray(self.frameForceRef)[softContactModel.mask]])
      runningModel.differential.f_des = forceRef
      runningModel.differential.f_weight = np.asarray(self.frameForceWeight)
      runningModel.differential.with_force_cost = True
    # Frame force rate reg cost
    if('forceRateReg' in self.WHICH_COSTS):
      runningModel.differential.with_force_rate_reg_cost = True
      runningModel.differential.f_rate_reg_weight = np.asarray(self.forceRateRegWeight)

  def init_terminal_model(self, state, actuation, terminalModel, softContactModel):
    ''' 
    Populate terminal model with costs and contacts 
    '''
  # Create and add terminal cost models to terminal IAM
    # State regularization
    if('stateReg' in self.WHICH_COSTS):
      xRegCost = self.create_state_reg_cost(state, actuation)
      terminalModel.differential.costs.addCost("stateReg", xRegCost, self.stateRegWeightTerminal*self.dt)
    # State limits
    if('stateLim' in self.WHICH_COSTS):
      xLimitCost = self.create_state_limit_cost(state, actuation)
      terminalModel.differential.costs.addCost("stateLim", xLimitCost, self.stateLimWeightTerminal*self.dt)
    # EE placement
    if('placement' in self.WHICH_COSTS):
      framePlacementCost = self.create_frame_placement_cost(state, actuation)
      terminalModel.differential.costs.addCost("placement", framePlacementCost, self.framePlacementWeightTerminal*self.dt)
    # EE velocity
    if('velocity' in self.WHICH_COSTS):
      frameVelocityCost = self.create_frame_velocity_cost(state, actuation)
      terminalModel.differential.costs.addCost("velocity", frameVelocityCost, self.frameVelocityWeightTerminal*self.dt)
    # EE translation
    if('translation' in self.WHICH_COSTS):
      frameTranslationCost = self.create_frame_translation_cost(state, actuation)
      terminalModel.differential.costs.addCost("translation", frameTranslationCost, self.frameTranslationWeightTerminal*self.dt)
    # End-effector orientation 
    if('rotation' in self.WHICH_COSTS):
      frameRotationCost = self.create_frame_rotation_cost(state, actuation)
      terminalModel.differential.costs.addCost("rotation", frameRotationCost, self.frameRotationWeightTerminal*self.dt)
    # Frame force cost
    if('force' in self.WHICH_COSTS):
      if(softContactModel.nc == 3):
        forceRef = np.asarray(self.frameForceRef)[:3]
      else:
        forceRef = np.array([np.asarray(self.frameForceRef)[softContactModel.mask]])
      terminalModel.differential.f_des = forceRef
      terminalModel.differential.f_weight = np.asarray(self.frameForceWeightTerminal)*self.dt
      terminalModel.differential.with_force_cost = True  
    # Frame force rate reg cost
    if('forceRateReg' in self.WHICH_COSTS):
      terminalModel.differential.with_force_rate_reg_cost = True
      terminalModel.differential.f_rate_reg_weight = np.asarray(self.forceRateRegWeight)*self.dt

  def success_log(self, softContactModel):
    logger.info("OCP (SOFT) is ready !")
    logger.info("    COSTS   = "+str(self.WHICH_COSTS))
    logger.info("    SOFT CONTACT MODEL [ oPc="+str(softContactModel.oPc)+\
      " , Kp="+str(softContactModel.Kp)+\
        ', Kv='+str(softContactModel.Kv)+\
        ', pinRefFrame='+str(softContactModel.pinRefFrame)+']')

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
      # Create DAM (Contact or FreeFwd), IAM LPF and initialize costs+contacts
        if(self.nb_constraints == 0):
          dam = self.create_differential_action_model(state, actuation, softContactModel) 
        else:
        # Create constraint manager and constraints
          constraintModelManager = self.create_constraint_model_manager(state, actuation, i)
        # Create DAM (Contact or FreeFwd), IAM Euler and initialize costs+contacts+constraints
          dam = self.create_differential_action_model(state, actuation, softContactModel, constraintModelManager) 
        runningModels.append(force_feedback_mpc.IAMSoftContactAugmented( dam, self.dt ))
        self.init_running_model(state, actuation, runningModels[i], softContactModel)
        
    # Terminal model
    if(self.nb_constraints == 0):
      dam_t = self.create_differential_action_model(state, actuation, softContactModel)  
    else:
      constraintModelManager = self.create_constraint_model_manager(state, actuation, self.N_h)
      dam_t = self.create_differential_action_model(state, actuation, softContactModel, constraintModelManager)  
    terminalModel = force_feedback_mpc.IAMSoftContactAugmented( dam_t, 0. )
    self.init_terminal_model(state, actuation, terminalModel, softContactModel)
    
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



# class OptimalControlProblemSoftContactAugmentedWithConstraints(OptimalControlProblemAbstract):
#   '''
#   Helper class for soft contact (augmented) OCP setup with Crocoddyl
#   '''
#   def __init__(self, robot, config):
#     '''
#     Override base class constructor if necessary
#     '''
#     super().__init__(robot, config)
  
#   def check_config(self):
#     '''
#     Override base class checks if necessary
#     '''
#     super().check_config()
#     self.check_attribute('Kp')
#     self.check_attribute('Kv')
#     self.check_attribute('oPc_offset')
#     self.check_attribute('pinRefFrame')
#     self.check_attribute('contactType')

#   def initialize(self, y0, softContactModel, callbacks=False):
#     '''
#     Initializes OCP and  solver from config parameters and initial state
#     Soft contact (visco-elastic) augmented formulation, i.e. visco-elastic
#     contact force is part of the state . Supported 3D formulation only for now
#       INPUT: 
#           y0                : initial state of shooting problem
#           softContactModel  : SoftContactModel3D (see in utils)
#           callbacks         : display Crocoddyl's DDP solver callbacks
#       OUTPUT:
#          solver

#      A cost term on a variable z(x,u) has the generic form w * a( r( z(x,u) - z0 ) )
#      where w <--> cost weight, e.g. 'stateRegWeight' in config file
#            r <--> residual model depending on some reference z0, e.g. 'stateRegRef'
#                   When ref is set to 'DEFAULT' in YAML file, default references hard-coded here are used
#            a <--> weighted activation, with weights e.g. 'stateRegWeights' in config file 
#            z <--> can be state x, control u, frame position or velocity, contact force, etc.
#     ''' 
    
#   # State and actuation models
#     state = crocoddyl.StateMultibody(self.rmodel)
#     actuation = crocoddyl.ActuationModelFull(state)
    
    
#   # Create IAMs
#     runningModels = []
#     for i in range(self.N_h):  
#         # Create DAMContactDyn     
#         if(softContactModel.nc == 3):
#           dam = DAMSoft3DAugmented(state, 
#                                   actuation, 
#                                   crocoddyl.CostModelSum(state, nu=actuation.nu),
#                                   softContactModel.frameId, 
#                                   softContactModel.Kp,
#                                   softContactModel.Kv,
#                                   softContactModel.oPc,
#                                   softContactModel.pinRefFrame )
#         elif(softContactModel.nc == 1):
#           dam = DAMSoft1DAugmented(state, 
#                                   actuation, 
#                                   crocoddyl.CostModelSum(state, nu=actuation.nu),
#                                   softContactModel.frameId, 
#                                   softContactModel.Kp,
#                                   softContactModel.Kv,
#                                   softContactModel.oPc,
#                                   softContactModel.pinRefFrame,
#                                   softContactModel.sobecType )
#         else:
#           logger.error("softContactModel.nc = 3 or 1")

#       # Create IAM from DAM
#         runningModels.append(IAMSoftAugmented(dam, self.dt))
        
#       # Create and add cost function terms to current IAM
#         # State regularization 
#         if('stateReg' in self.WHICH_COSTS):
#           xRegCost = self.create_state_reg_cost(state, actuation)
#           runningModels[i].differential.costs.addCost("stateReg", xRegCost, self.stateRegWeight)
#         # Control regularization
#         if('ctrlReg' in self.WHICH_COSTS):
#           uRegCost = self.create_ctrl_reg_cost(state)
#           runningModels[i].differential.costs.addCost("ctrlReg", uRegCost, self.ctrlRegWeight)
#         # Control regularization (gravity)
#         if('ctrlRegGrav' in self.WHICH_COSTS):
#           runningModels[i].differential.with_gravity_torque_reg = True
#           runningModels[i].differential.tau_grav_weight = self.ctrlRegGravWeight
#         # State limits penalizationself.
#         if('stateLim' in self.WHICH_COSTS):
#           xLimitCost = self.create_state_limit_cost(state, actuation)
#           runningModels[i].differential.costs.addCost("stateLim", xLimitCost, self.stateLimWeight)
#         # Control limits penalization
#         if('ctrlLim' in self.WHICH_COSTS):
#           uLimitCost = self.create_ctrl_limit_cost(state)
#           runningModels[i].differential.costs.addCost("ctrlLim", uLimitCost, self.ctrlLimWeight)
#         # End-effector placement 
#         if('placement' in self.WHICH_COSTS):
#           framePlacementCost = self.create_frame_placement_cost(state, actuation)
#           runningModels[i].differential.costs.addCost("placement", framePlacementCost, self.framePlacementWeight)
#         # End-effector velocity
#         if('velocity' in self.WHICH_COSTS): 
#           frameVelocityCost = self.create_frame_velocity_cost(state, actuation)
#           runningModels[i].differential.costs.addCost("velocity", frameVelocityCost, self.frameVelocityWeight)
#         # Frame translation cost
#         if('translation' in self.WHICH_COSTS):
#           frameTranslationCost = self.create_frame_translation_cost(state, actuation)
#           runningModels[i].differential.costs.addCost("translation", frameTranslationCost, self.frameTranslationWeight)
#         # End-effector orientation 
#         if('rotation' in self.WHICH_COSTS):
#           frameRotationCost = self.create_frame_rotation_cost(state, actuation)
#           runningModels[i].differential.costs.addCost("rotation", frameRotationCost, self.frameRotationWeight)
#         # Frame force cost
#         if('force' in self.WHICH_COSTS):
#           if(softContactModel.nc == 3):
#             forceRef = np.asarray(self.frameForceRef)[:3]
#           else:
#             forceRef = np.array([np.asarray(self.frameForceRef)[softContactModel.mask]])
#           runningModels[i].differential.f_des = forceRef
#           runningModels[i].differential.f_weight = np.asarray(self.frameForceWeight)
#           runningModels[i].differential.with_force_cost = True
#         # Frame force rate reg cost
#         if('forceRateReg' in self.WHICH_COSTS):
#           runningModels[i].differential.with_force_rate_reg_cost = True
#           runningModels[i].differential.f_rate_reg_weight = np.asarray(self.forceRateRegWeight)

#       # # Armature 
#       #   # Add armature to current IAM
#       #   if(self.armature == 'DEFAULT'):
#       #     self.armature = self.rmodel.rotorInertia / (self.rmodel.rotorGearRatio**2)
#       #   runningModels[i].differential.armature = np.asarray(self.armature)

#         # Constraint model 
#         nc = 0
#         constraint_models_stack_list = []
#         # State limits
#         if('stateBox' in self.WHICH_CONSTRAINTS):
#           if(i == 0):
#             stateBoxConstraint = self.create_no_constraint(state, 'None', actuation)
#           else:
#             stateBoxConstraint = self.create_state_constraint(state, 'stateBox', actuation)   
#             nc += stateBoxConstraint.nc
#           constraint_models_stack_list.append(stateBoxConstraint)
#         # Control limits
#         if('ctrlBox' in self.WHICH_CONSTRAINTS):
#           ctrlBoxConstraint = self.create_ctrl_constraint(state, 'ctrlBox', actuation)
#           nc += ctrlBoxConstraint.nc
#           constraint_models_stack_list.append(ctrlBoxConstraint)
#         # # End-effector position limits 
#         # Unsupported because crocoddyl.FrameTranslationContraints casts DAM into DAMContact
#         # whereas sobec.DAMSoftAugmented inherits from crocoddyl.DAMFree 
#         # Yet there it should be possible to access data.oMf from both !
#         # if('translationBox' in self.WHICH_CONSTRAINTS):
#         #   if(i == 0):
#         #     translationBoxConstraint = self.create_no_constraint(state, 'None', actuation)
#         #   else:
#         #     translationBoxConstraint = self.create_translation_constraint(state, 'translationBox', actuation)
#         #     nc += translationBoxConstraint.nc
#         #   constraint_models_stack_list.append(translationBoxConstraint)
#         # No constraints
#         if('None' in self.WHICH_CONSTRAINTS):
#           noConstraintModel = self.create_no_constraint(state, 'None', actuation)
#           constraint_models_stack_list.append(noConstraintModel)

#         # Running constraint model stack
#         runningConstraintModel = crocoddyl.ConstraintStack(constraint_models_stack_list, state, nc, actuation.nu, 'constraint_'+str(i))


#   # Terminal DAM (Contact or FreeFwd)
#     # Create terminal DAMContactDyn
#     if(softContactModel.nc == 3):
#       dam_t = DAMSoft3DAugmented(state, 
#                                 actuation, 
#                                 crocoddyl.CostModelSum(state, nu=actuation.nu),
#                                 softContactModel.frameId, 
#                                 softContactModel.Kp,
#                                 softContactModel.Kv,
#                                 softContactModel.oPc,
#                                 softContactModel.pinRefFrame )
#     elif(softContactModel.nc == 1):
#       dam_t = DAMSoft1DAugmented(state, 
#                                 actuation, 
#                                 crocoddyl.CostModelSum(state, nu=actuation.nu),
#                                 softContactModel.frameId, 
#                                 softContactModel.Kp,
#                                 softContactModel.Kv,
#                                 softContactModel.oPc,
#                                 softContactModel.pinRefFrame,
#                                 softContactModel.sobecType )
#     else:
#       logger.error("softContactModel.nc = 3 or 1")

#   # Create terminal IAM from terminal DAM
#     terminalModel = IAMSoftAugmented( dam_t, 0. )

#   # Create and add terminal cost models to terminal IAM
#     # State regularization
#     if('stateReg' in self.WHICH_COSTS):
#       xRegCost = self.create_state_reg_cost(state, actuation)
#       terminalModel.differential.costs.addCost("stateReg", xRegCost, self.stateRegWeightTerminal*self.dt)
#     # State limits
#     if('stateLim' in self.WHICH_COSTS):
#       xLimitCost = self.create_state_limit_cost(state, actuation)
#       terminalModel.differential.costs.addCost("stateLim", xLimitCost, self.stateLimWeightTerminal*self.dt)
#     # EE placement
#     if('placement' in self.WHICH_COSTS):
#       framePlacementCost = self.create_frame_placement_cost(state, actuation)
#       terminalModel.differential.costs.addCost("placement", framePlacementCost, self.framePlacementWeightTerminal*self.dt)
#     # EE velocity
#     if('velocity' in self.WHICH_COSTS):
#       frameVelocityCost = self.create_frame_velocity_cost(state, actuation)
#       terminalModel.differential.costs.addCost("velocity", frameVelocityCost, self.frameVelocityWeightTerminal*self.dt)
#     # EE translation
#     if('translation' in self.WHICH_COSTS):
#       frameTranslationCost = self.create_frame_translation_cost(state, actuation)
#       terminalModel.differential.costs.addCost("translation", frameTranslationCost, self.frameTranslationWeightTerminal*self.dt)
#     # End-effector orientation 
#     if('rotation' in self.WHICH_COSTS):
#       frameRotationCost = self.create_frame_rotation_cost(state, actuation)
#       terminalModel.differential.costs.addCost("rotation", frameRotationCost, self.frameRotationWeightTerminal*self.dt)
#     # Frame force cost
#     if('force' in self.WHICH_COSTS):
#       if(softContactModel.nc == 3):
#         forceRef = np.asarray(self.frameForceRef)[:3]
#       else:
#         forceRef = np.array([np.asarray(self.frameForceRef)[softContactModel.mask]])
#       terminalModel.differential.f_des = forceRef
#       terminalModel.differential.f_weight = np.asarray(self.frameForceWeightTerminal)*self.dt
#       terminalModel.differential.with_force_cost = True  
#     # Frame force rate reg cost
#     if('forceRateReg' in self.WHICH_COSTS):
#       terminalModel.differential.with_force_rate_reg_cost = True
#       terminalModel.differential.f_rate_reg_weight = np.asarray(self.forceRateRegWeight)*self.dt

#       # Add armature
#     # terminalModel.differential.armature = np.asarray(self.armature)   


#     # Constraint model 
#     nc = 0
#     constraint_models_stack_list_terminal = []
#     # State limits
#     if('stateBox' in self.WHICH_CONSTRAINTS):
#       stateBoxConstraint = self.create_state_constraint(state, 'stateBox', actuation) 
#       nc += stateBoxConstraint.nc 
#       constraint_models_stack_list_terminal.append(stateBoxConstraint)
#     # Control limits
#     if('ctrlBox' in self.WHICH_CONSTRAINTS):
#       ctrlBoxConstraint = self.create_ctrl_constraint(state, 'ctrlBox', actuation)
#       nc += ctrlBoxConstraint.nc
#       constraint_models_stack_list_terminal.append(ctrlBoxConstraint)
#     # End-effector position limits
#     # Unsupported because crocoddyl.FrameTranslationContraints casts DAM into DAMContact
#     # whereas sobec.DAMSoftAugmented inherits from crocoddyl.DAMFree 
#     # Yet there it should be possible to access data.oMf from both !
#     # if('translationBox' in self.WHICH_CONSTRAINTS):
#     #   translationBoxConstraint = self.create_translation_constraint(state, 'translationBox', actuation)
#     #   nc += translationBoxConstraint.nc
#     #   constraint_models_stack_list_terminal.append(translationBoxConstraint)
#     # No constraint
#     if('None' in self.WHICH_CONSTRAINTS):
#       noConstraintModel = self.create_no_constraint(state, 'None', actuation)
#       constraint_models_stack_list_terminal.append(noConstraintModel)

#     # Terminal constraint model stack
#     terminalConstraintModel = crocoddyl.ConstraintStack(constraint_models_stack_list_terminal, state, nc, actuation.nu, 'constraint_terminal')
    

#     constraintModels = [runningConstraintModel]*(self.N_h) + [terminalConstraintModel] 
#     # logger.warning("Constraint models = \n")
#     # logger.warning(constraintModels)

#     logger.info("Created IAMs.")  



#   # Create the shooting problem
#     problem = crocoddyl.ShootingProblem(y0, runningModels, terminalModel)
  
#   # Creating the DDP solver 
#     self.check_attribute('USE_PROXQP')
#     if(self.USE_PROXQP):
#       logger.warning('Using PROXQP solver')
#       ddp = crocoddyl.SolverPROXQP(problem, constraintModels) 
#     else:
#       logger.warning('Using FADMM solver')
#       ddp = crocoddyl.SolverFADMM(problem, constraintModels)

#   # Callbacks & solver parameters
#     self.check_attribute('with_callbacks')
#     self.check_attribute('use_filter_ls')
#     self.check_attribute('filter_size')
#     self.check_attribute('warm_start')
#     self.check_attribute('termination_tol')
#     self.check_attribute('max_qp_iters')
#     self.check_attribute('qp_termination_tol_abs')
#     self.check_attribute('qp_termination_tol_rel')
#     self.check_attribute('warm_start_y')
#     self.check_attribute('reset_rho')
#     ddp.with_callbacks = self.with_callbacks
#     ddp.use_filter_ls = self.use_filter_ls
#     ddp.filter_size = self.filter_size
#     ddp.warm_start = self.warm_start
#     ddp.termination_tol = self.termination_tol
#     ddp.max_qp_iters = self.max_qp_iters
#     ddp.eps_abs = self.qp_termination_tol_abs
#     ddp.eps_rel = self.qp_termination_tol_rel
#     ddp.warm_start_y = self.warm_start_y
#     ddp.reset_rho = self.reset_rho
  
#   # Callbacks
#     if(callbacks):
#       ddp.setCallbacks([crocoddyl.CallbackLogger(),
#                         crocoddyl.CallbackVerbose()])
  
#   # Warm start : initial state + gravity compensation
#     ddp.xs = [y0 for i in range(self.N_h+1)]
#     fext0 = softContactModel.computeExternalWrench_(self.rmodel, y0[:self.nq], y0[:self.nv])
#     ddp.us = [pin_utils.get_tau(y0[:self.nq], y0[:self.nv], np.zeros(self.nv), fext0, self.rmodel, np.zeros(self.nq)) for i in range(self.N_h)] #ddp.problem.quasiStatic(xs_init[:-1])

#   # Finish
#     logger.info("OCP is ready !")
#     # logger.info(  "USE_SOBEC_BINDINGS = "+str(USE_SOBEC_BINDINGS))
#     logger.info("    COSTS        = "+str(self.WHICH_COSTS))
#     logger.info("    CONSTRAINTS  = "+str(self.WHICH_CONSTRAINTS))
#     logger.info("    SOFT CONTACT MODEL [ oPc="+str(softContactModel.oPc)+\
#       " , Kp="+str(softContactModel.Kp)+\
#         ', Kv='+str(softContactModel.Kv)+\
#         ', pinRefFrame='+str(softContactModel.pinRefFrame)+']')
    
#     return ddp
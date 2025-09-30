"""
@package force_feedback
@file init_ocp.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft.
@date 2022-05-12
@brief Initializes the OCP + solver
"""

import crocoddyl
import numpy as np

import force_feedback_mpc

from croco_mpc_utils.ocp_core import OptimalControlProblemAbstract
from croco_mpc_utils import pinocchio_utils as pin_utils

from croco_mpc_utils.utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger


def getJointAndStateIds(rmodel, jointNames):
  '''
  Determine the joint ids and state ids of the input joint names
  '''
  jointIds = [] ; stateIds = []
  for name in jointNames:
    jid = rmodel.getJointId(name)
    if(rmodel.joints[jid].nv ==1):
      jointIds.append(jid)
      if(rmodel.joints[1].nv > 1):
        stateIds.append(rmodel.idx_vs[jid] - 6)
      else:
        stateIds.append(rmodel.idx_vs[jid])
  jointIds.sort()
  stateIds.sort()
  return jointIds, stateIds

class OptimalControlProblemLPF(OptimalControlProblemAbstract):
  '''
  Helper class for Low-Pass Filter (LPF) OCP setup with Crocoddyl
   to allow joint torque feedback in the MPC 
  '''
  def __init__(self, robot, config, lpf_joint_names):
    '''
    Override base class constructor if necessary
    '''
    super().__init__(robot, config)

    self.lpf_joint_names = lpf_joint_names
    self.n_lpf = len(self.lpf_joint_names)
    self.lpf_joint_ids, self.lpf_state_ids = getJointAndStateIds(self.rmodel, self.lpf_joint_names)

  def check_config(self):
    '''
    Override base class checks if necessary
    '''
    super().check_config()
    self.check_attribute('f_c')
    self.check_attribute('LPF_TYPE')
    self.check_attribute('tau_plus_integration')
    self.check_attribute('wRegWeight')
    self.check_attribute('wLimWeight')

  def parse_contacts(self):
    '''
    Parses the YAML dict of contacts and count them
    '''
    if(not hasattr(self, 'contacts')):
      self.nb_contacts = 0
    else:
      self.nb_contacts = len(self.contacts)
      self.contact_types = [ct['contactModelType'] for ct in self.contacts]
      logger.debug("Detected "+str(len(self.contacts))+" contacts with types = "+str(self.contact_types))

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
    if('ctrlBox' in self.WHICH_CONSTRAINTS and node_id != self.N_h):
      ctrlBoxConstraint = self.create_ctrl_constraint(state, actuation)
      constraintModelManager.addConstraint('ctrlBox', ctrlBoxConstraint)
      logger.debug("constraint manager ctrl box "+str(node_id)+" lb = : "+str(constraintModelManager.g_lb))
      logger.debug("constraint manager ctrl box "+str(node_id)+" ub = : "+str(constraintModelManager.g_ub))
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


  def create_differential_action_model(self, state, actuation, constraintModelManager=None):
    '''
    Initialize a differential action model with or without contacts
    '''
    # If there are contacts, defined constrained DAM
    contactModels = []
    if(self.nb_contacts > 0):
      for ct in self.contacts:
        contactModels.append(self.create_contact_model(ct, state, actuation))   
      if(constraintModelManager is None):
        dam = crocoddyl.DifferentialActionModelContactFwdDynamics(state, 
                                                                  actuation, 
                                                                  crocoddyl.ContactModelMultiple(state, actuation.nu), 
                                                                  crocoddyl.CostModelSum(state, nu=actuation.nu), 
                                                                  inv_damping=0., 
                                                                  enable_force=True)
      else:
        dam = crocoddyl.DifferentialActionModelContactFwdDynamics(state, 
                                                                  actuation, 
                                                                  crocoddyl.ContactModelMultiple(state, actuation.nu), 
                                                                  crocoddyl.CostModelSum(state, nu=actuation.nu), 
                                                                  constraintModelManager,
                                                                  inv_damping=0., 
                                                                  enable_force=True)
    # Otherwise just create free DAM
    else:
      if(constraintModelManager is None):
        dam = crocoddyl.DifferentialActionModelFreeFwdDynamics(state, 
                                                                actuation, 
                                                                crocoddyl.CostModelSum(state, nu=actuation.nu))
      else:
        dam = crocoddyl.DifferentialActionModelFreeFwdDynamics(state, 
                                                                actuation, 
                                                                crocoddyl.CostModelSum(state, nu=actuation.nu),
                                                                constraintModelManager)
    return dam, contactModels

  def init_running_model(self, state, actuation, runningModel, contactModels, w_reg_ref):
    '''
  Populate running model with costs and contacts
    '''
  # Create and add cost function terms to current IAM
    # Add cost on unfiltered control torque (reg + lim)
    runningModel.set_control_reg_cost(self.wRegWeight, w_reg_ref[self.lpf_state_ids]) 
    runningModel.set_control_lim_cost(self.wLimWeight) 
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
      uRegGravCost = self.create_ctrl_reg_grav_cost(state)
      runningModel.differential.costs.addCost("ctrlRegGrav", uRegGravCost, self.ctrlRegGravWeight)
    # State limits penalization
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
      frameForceCost = self.create_frame_force_cost(state, actuation)
      runningModel.differential.costs.addCost("force", frameForceCost, self.frameForceWeight)
    # Friction cone 
    if('friction' in self.WHICH_COSTS):
      frictionConeCost = self.create_friction_force_cost(state, actuation)
      runningModel.differential.costs.addCost("friction", frictionConeCost, self.frictionConeWeight)
    if('collision' in self.WHICH_COSTS):
      collisionCost = self.create_collision_cost(state, actuation)
      runningModel.differential.costs.addCost("collision", collisionCost, self.collisionCostWeight)

    # # Armature 
    # runningModel.differential.armature = np.asarray(self.armature)
    
    # Contact model
    if(len(contactModels) > 0):
      for k,contactModel in enumerate(contactModels):
        runningModel.differential.contacts.addContact(self.contacts[k]['contactModelFrameName'], contactModel, active=self.contacts[k]['active'])

  def init_terminal_model(self, state, actuation, terminalModel, contactModels):
    ''' 
    Populate terminal model with costs and contacts 
    '''
    # State regularization
    if('stateReg' in self.WHICH_COSTS):
      xRegCost = self.create_state_reg_cost(state, actuation)
      terminalModel.differential.costs.addCost("stateReg", xRegCost, self.stateRegWeightTerminal*self.dt)
    # Ctrl regularization
    if('ctrlReg' in self.WHICH_COSTS):
      uRegCost = self.create_ctrl_reg_cost(state)
      terminalModel.differential.costs.addCost("ctrlReg", uRegCost, self.ctrlRegWeightTerminal*self.dt)
    # Control regularization (gravity)
    if('ctrlRegGrav' in self.WHICH_COSTS):
      uRegGravCost = self.create_ctrl_reg_grav_cost(state)
      terminalModel.differential.costs.addCost("ctrlRegGrav", uRegGravCost, self.ctrlRegGravWeightTerminal*self.dt)
    # Control limit
    if('ctrlLim' in self.WHICH_COSTS):
      uLimCost = self.create_ctrl_limit_cost(state)
      terminalModel.differential.costs.addCost("ctrlLim", uLimCost, self.ctrlLimWeightTerminal*self.dt)
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

    # # Add armature
    # terminalModel.differential.armature = np.asarray(self.armature)   
  
    # Add contact model
    if(len(contactModels)):
      for k,contactModel in enumerate(contactModels):
        terminalModel.differential.contacts.addContact(self.contacts[k]['contactModelFrameName'], contactModel, active=self.contacts[k]['active'])

  def lpf_parameter_log(self):
      '''
      Log of the type of LPF and its parameter
      '''
    # LPF parameters (a.k.a simplified actuation model)
      # Approx. LPF obtained from Z.O.H. discretization on CT LPF 
      if(self.LPF_TYPE==0):
          alpha = np.exp(-2*np.pi*self.f_c*self.dt)
      # Approx. LPF obtained from 1st order Euler int. on CT LPF
      if(self.LPF_TYPE==1):
          alpha = 1./float(1+2*np.pi*self.f_c*self.dt)
      # Exact LPF obtained from E.M.A model (IIR)
      if(self.LPF_TYPE==2):
          y = np.cos(2*np.pi*self.f_c*self.dt)
          alpha = 1-(y-1+np.sqrt(y**2 - 4*y +3)) 
      logger.info("Setup Low-Pass Filter (LPF)")
      logger.info("          self.f_c   = "+str(self.f_c))
      logger.info("          alpha = "+str(alpha))

  def success_log(self):
    '''
    Log of successful OCP initialization + important information
    '''
    logger.info("OCP (LPF) is ready !")
    logger.info("    COSTS         = "+str(self.WHICH_COSTS))
    if(self.nb_contacts > 0):
      logger.info("    self.nb_contacts = "+str(self.nb_contacts))
      for ct in self.contacts:
        logger.info("      Found [ "+str(ct['contactModelType'])+" ] (Baumgarte stab. gains = "+str(ct['contactModelGains'])+" , active = "+str(ct['active'])+" )")
    else:
      logger.info("    self.nb_contacts = "+str(self.nb_contacts))
    if(self.nb_constraints > 0):
      logger.info("    CONSTRAINTS   = "+str(self.WHICH_CONSTRAINTS))

  def parse_unfiltered_torque_reg_reference(self, y0):
    '''
    Detects the type of computed (unfiltered) torque regularization used 
    '''
    if(not hasattr(self, 'wRegRef')):
      logger.error("Need to specify 'wRegRef' in YAML config. Please select in ['zero', 'tau0', 'gravity']")
    elif(self.wRegRef == 'gravity'):
      # If no reference is provided, assume default reg w.r.t. gravity torque
      w_reg_ref = np.zeros(self.nq) # dummy reference not used
      log_msg_w_reg = 'gravity torque'
    else:
      # Otherwise, take the user-provided constant torque reference for w_reg
      if(self.wRegRef== 'zero'):
        w_reg_ref = np.zeros(self.nq)
      elif(self.wRegRef== 'tau0'):
        w_reg_ref = pin_utils.get_u_grav(y0[:self.nq], self.rmodel)
      else:
        logger.error("Unknown 'wRegRef' in YAML config. Please select in ['zero', 'tau0', 'gravity']")
      # w_reg_ref = np.asarray(self.wRegRef'])
      log_msg_w_reg = 'constant reference '+self.wRegRef
    logger.warning("Unfiltered torque regularization w.r.t. "+log_msg_w_reg)
    return w_reg_ref
  

  def initialize(self, y0):
    '''
    Initializes OCP and solver from config parameters and initial state
      INPUT: 
          robot       : pinocchio robot wrapper
          config      : dict from YAML config file describing task and MPC params
          y0          : initial augmented state of shooting problem (q0, v0, tau0)
      OUTPUT:
          solver

     A cost term on a variable z(x,u) has the generic form w * r( a( z(x,u) ) )
     where w <--> cost weight, e.g. 'stateRegWeight' in config file
           r <--> residual model depending on some reference, e.g. 'stateRegRef'
                  Wen set to 'None' in config file, default references are hard-coded here
           a <--> weighted activation, with weights e.g. 'stateRegWeights' in config file 
           z <--> can be state x, control u, frame position or velocity, contact force, etc.
    '''
    
  # State and actuation models
    state = crocoddyl.StateMultibody(self.rmodel)
    actuation = crocoddyl.ActuationModelFull(state)

  # Contact or not ?
    self.parse_contacts()
  # Constraints or not ?
    self.parse_constraints()

  # LPF parameters (a.k.a simplified actuation model)
    self.lpf_parameter_log()

  # Regularization cost of unfiltered torque (inside IAM_LPF in Crocoddyl)
    w_reg_ref = self.parse_unfiltered_torque_reg_reference(y0)

  # Create IAMs
    runningModels = []
    for i in range(self.N_h):  
      # Create DAM (Contact or FreeFwd), IAM Euler and initialize costs+contacts
        if(self.nb_constraints == 0):
          dam, contactModels = self.create_differential_action_model(state, actuation) 
        else:
        # Create constraint manager and constraints
          constraintModelManager = self.create_constraint_model_manager(state, actuation, i)

        # Create DAM (Contact or FreeFwd), IAM Euler and initialize costs+contacts+constraints
          dam, contactModels = self.create_differential_action_model(state, actuation, constraintModelManager) 
        runningModels.append(force_feedback_mpc.IntegratedActionModelLPF( dam,
                                                             LPFJointNames=self.lpf_joint_names, 
                                                             stepTime=self.dt, 
                                                             withCostResidual=True, 
                                                             fc=self.f_c, 
                                                             tau_plus_integration=self.tau_plus_integration,
                                                             filter=self.LPF_TYPE))
        self.init_running_model(state, actuation, runningModels[i], contactModels, w_reg_ref)

    # Terminal model
    if(self.nb_constraints == 0):
      dam_t, contactModels = self.create_differential_action_model(state, actuation)  
    else:
      constraintModelManager = self.create_constraint_model_manager(state, actuation, self.N_h)
      dam_t, contactModels = self.create_differential_action_model(state, actuation, constraintModelManager)   
    terminalModel = force_feedback_mpc.IntegratedActionModelLPF( dam_t, 
                                                    LPFJointNames=self.lpf_joint_names, 
                                                    stepTime=0., 
                                                    withCostResidual=False, 
                                                    fc=self.f_c, 
                                                    tau_plus_integration=self.tau_plus_integration,
                                                    filter=self.LPF_TYPE)
    self.init_terminal_model(state, actuation, terminalModel, contactModels)
    
    logger.info("Created IAMs.")  

  # Create the shooting problem
    problem = crocoddyl.ShootingProblem(y0, runningModels, terminalModel)

  # Finish
    self.success_log()
    return problem
"""
@package force_feedback
@file classical_mpc/init_data.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft.
@date 2020-05-18
@brief Initialize / extract data for MPC simulation (soft contact)
"""

import numpy as np
from classical_mpc.data import DDPDataHandlerClassical, MPCDataHandlerClassical

from core_mpc_utils.misc_utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger


# Classical OCP data handler : extract data + generate fancy plots
class DDPDataHandlerSoftContact(DDPDataHandlerClassical):

  def __init__(self, ddp, softContactModel):
    super().__init__(ddp)
    self.softContactModel = softContactModel

  def extract_data(self, ee_frame_name, ct_frame_name):
    '''
    Extract data from ddp solver
    '''
    ddp_data = super().extract_data(ee_frame_name, ct_frame_name)
    # Compute the visco-elastic contact force & extract the reference force from DAM
    xs = np.array(ddp_data['xs'])
    nq = ddp_data['nq']
    nv = ddp_data['nv']
    if(self.softContactModel.nc == 3):
        fs_lin = np.array([self.softContactModel.computeForce_(ddp_data['pin_model'], xs[i,:nq], xs[i,nq:nq+nv]) for i in range(ddp_data['T'])])
        fdes_lin = np.array([self.ddp.problem.runningModels[i].differential.f_des for i in range(ddp_data['T'])])
    else:
        fs_lin = np.zeros((ddp_data['T'],3))
        fs_lin[:,self.softContactModel.mask] = np.array([self.softContactModel.computeForce_(ddp_data['pin_model'], xs[i,:nq], xs[i,nq:]) for i in range(ddp_data['T'])])
        fdes_lin = np.zeros((ddp_data['T'],3))
        fs_lin[:,self.softContactModel.mask] = np.array([self.ddp.problem.runningModels[i].differential.f_des for i in range(ddp_data['T'])])
    fs_ang = np.zeros((ddp_data['T'], 3))
    fdes_ang = np.zeros((ddp_data['T'], 3))
    ddp_data['fs'] = np.hstack([fs_lin, fs_ang])
    ddp_data['force_ref'] = np.hstack([fdes_lin, fdes_ang])
    return ddp_data

  # Temporary patch for augmented soft ddp  --> need to clean it up
  def extract_data_augmented(self, ee_frame_name, ct_frame_name):
    '''
    Extract data from DDP solver 
    Patch for augmented soft contact formulation 
    extracting the contact force from the state 
    and desired force from augmented DAM.
    Set 0 angular force by default.
    '''
    ddp_data = super().extract_data(ee_frame_name, ct_frame_name)
    ddp_data['nq'] = 7
    ddp_data['nv'] = 7
    ddp_data['nx'] = 14
    # Compute the visco-elastic contact force & extract the reference force from DAM
    xs = np.array(ddp_data['xs'])
    nq = ddp_data['nq']
    nv = ddp_data['nv']
    if(self.softContactModel.nc == 3):
        fs_lin = np.array([xs[i,-3:] for i in range(ddp_data['T'])])
        fdes_lin = np.array([self.ddp.problem.runningModels[i].differential.f_des for i in range(ddp_data['T'])])
    # else:
    #     fs_lin = np.zeros((ddp_data['T'],3))
    #     fs_lin[:,self.softContactModel.mask] = np.array([self.softContactModel.computeForce_(ddp_data['pin_model'], xs[i,:nq], xs[i,nq:nq+nv]) for i in range(ddp_data['T'])])
    #     fdes_lin = np.zeros((ddp_data['T'],3))
    #     fs_lin[:,self.softContactModel.mask] = np.array([self.ddp.problem.runningModels[i].differential.f_des for i in range(ddp_data['T'])])
    fs_ang = np.zeros((ddp_data['T'], 3))
    fdes_ang = np.zeros((ddp_data['T'], 3))
    ddp_data['fs'] = np.hstack([fs_lin, fs_ang])
    ddp_data['force_ref'] = np.hstack([fdes_lin, fdes_ang])
    return ddp_data




# Classical MPC data handler : initialize, extract data + generate fancy plots
class MPCDataHandlerSoftContact(MPCDataHandlerClassical):

  def __init__(self, config, robot):
    super().__init__(config, robot)

  def record_predictions(self, nb_plan, ddpSolver, softContactModel):
    '''
    - Records the MPC prediction of at the current step (state, control and forces if contact is specified)
    '''
    self.state_pred[nb_plan, :, :] = np.array(ddpSolver.xs)
    self.ctrl_pred[nb_plan, :, :] = np.array(ddpSolver.us)
    # Extract relevant predictions for interpolations to MPC frequency
    self.x_curr = self.state_pred[nb_plan, 0, :]    # x0* = measured state    (q^,  v^ )
    self.x_pred = self.state_pred[nb_plan, 1, :]    # x1* = predicted state   (q1*, v1*) 
    self.u_curr = self.ctrl_pred[nb_plan, 0, :]     # u0* = optimal control   
    # Record forces in the right frame
    # id_endeff = softContactModel.frameId
    # Extract soft force
    xs = np.array(ddpSolver.xs)
    # Force in WORLD aligned frame
    if(softContactModel.nc == 3):
        fs_lin = np.array([softContactModel.computeForce_(self.rmodel, xs[i,:self.nq], xs[i,self.nq:]) for i in range(self.N_h)])
    else:
        fs_lin = np.zeros((self.N_h,3))
        fs_lin[:,softContactModel.mask] = np.array([softContactModel.computeForce_(self.rmodel, xs[i,:self.nq], xs[i,self.nq:]) for i in range(self.N_h)])
    fs_ang = np.zeros((self.N_h, 3))
    fs = np.hstack([fs_lin, fs_ang])
    # fref = [np.zeros(6) for i in range(self.N_h) ]
    self.force_pred[nb_plan, :, :] = fs
    # if(softContactModel.pinRefFrame == pin.LOCAL):
    #     self.force_pred[nb_plan, :, :] = \
    #         np.array([ddpSolver.problem.runningDatas[i].differential.multibody.contacts.contacts[self.contactFrameName].f.vector for i in range(self.N_h)])
    # else:
    #     self.force_pred[nb_plan, :, :] = \
    #         np.array([self.rdata.oMf[id_endeff].action @ ddpSolver.problem.runningDatas[i].differential.multibody.contacts.contacts[self.contactFrameName].f.vector for i in range(self.N_h)])
    self.f_curr = self.force_pred[nb_plan, 0, :]
    self.f_pred = self.force_pred[nb_plan, 1, :]

  
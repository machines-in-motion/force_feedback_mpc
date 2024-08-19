"""
@package force_feedback
@file soft_mpc/utils.py
@author Sebastien Kleff
@license License BSD-3-Clause
@copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft.
@date 2020-05-18
@brief Initializes the OCP + DDP solver (visco-elastic contact)
"""

import numpy as np
import pinocchio as pin

from croco_mpc_utils.utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger

import force_feedback_mpc

class SoftContactModel3D:
    def __init__(self, Kp, Kv, oPc, frameId, pinRef):
        '''
          Kp, Kv      : stiffness and damping coefficient of the visco-elastic contact model
          oPc         : anchor point of the contact 
          frameId     : frame at which the soft contact is defined
          pinRefFrame : reference frame in which the contact model is expressed (pin.LOCAL or pin.LWA)
        '''
        self.nc = 3
        self.Kp = Kp
        self.Kv = Kv
        self.oPc = oPc
        self.pinRefFrame = self.setPinRef(pinRef)
        self.frameId = frameId 

    def setPinRef(self, pinRef):
        '''
        Sets pinocchio reference frame from string or pin.ReferenceFrame
        '''
        if(type(pinRef) == str):
            if(pinRef == 'LOCAL'):
                return pin.LOCAL
            elif(pinRef == 'LOCAL_WORLD_ALIGNED'):
                return pin.LOCAL_WORLD_ALIGNED
            else:
                logger.error("yaml config file : pinRefFrame must be in either LOCAL or LOCAL_WORLD_ALIGNED !")
        else:
            return pinRef

    def computeForce(self, rmodel, rdata):
        '''
        Compute the 3D visco-elastic contact force 
          rmodel : robot model
          rdata  : robot data
        '''
        oRf = rdata.oMf[self.frameId].rotation
        oPf = rdata.oMf[self.frameId].translation
        lv = pin.getFrameVelocity(rmodel, rdata, self.frameId, pin.LOCAL).linear
        f = -self.Kp * oRf.T @ (oPf - self.oPc) - self.Kv * lv
        # if(self.pinRefFrame == pin.LOCAL):
            # f = -self.Kp * oRf.T @ (oPf - self.oPc) - self.Kv * lv
        if(self.pinRefFrame == pin.LOCAL_WORLD_ALIGNED):
            f = oRf @ f
        return f

    def computeForce_(self, rmodel, q, v):
        '''
        Compute the 3D visco-elastic contact force from (q, v)
          rmodel : robot model
          rdata  : robot data
        '''
        rdata = rmodel.createData()
        pin.forwardKinematics(rmodel, rdata, q, v)
        pin.updateFramePlacements(rmodel, rdata)
        return self.computeForce(rmodel, rdata)

    def computeExternalWrench(self, rmodel, rdata):
        '''
        Compute the vector for pin.Force (external wrenches) due to
        the 3D visco-elastic contact force
          rmodel  : robot model
          rdata   : robot data
        '''
        f3D = self.computeForce(rmodel, rdata)
        oRf = rdata.oMf[self.frameId].rotation
        wrench = [pin.Force.Zero() for _ in range(rmodel.njoints)]
        f6D = pin.Force(f3D, np.zeros(3))
        parentId = rmodel.frames[self.frameId].parent
        jMf = rmodel.frames[self.frameId].placement
        if(self.pinRefFrame == pin.LOCAL):
            wrench[parentId] = jMf.act(f6D)
        elif(self.pinRefFrame == pin.LOCAL_WORLD_ALIGNED):
            lwaXf = pin.SE3.Identity() ; lwaXf.rotation = oRf ; lwaXf.translation = np.zeros(3)
            wrench[parentId] = jMf.act(lwaXf.actInv(f6D))
        return wrench

    def computeExternalWrench_(self, rmodel, q, v):
        '''
        Compute the vector for pin.Force (external wrenches) due to
        the 3D visco-elastic contact force from (q, v)
          rmodel : robot model
          rdata  : robot data
        '''
        rdata = rmodel.createData()
        pin.forwardKinematics(rmodel, rdata, q, v)
        pin.updateFramePlacements(rmodel, rdata)
        return self.computeExternalWrench(rmodel, rdata)

    def print(self):
        logger.debug("- - - - - - - - - - - - - -")
        logger.debug("Contact model parameters")
        logger.debug(" -> frameId : "+str(self.frameId))
        logger.debug(" -> Kp      : "+str(self.Kp))
        logger.debug(" -> Kv      : "+str(self.Kv))
        logger.debug(" -> oPc     : "+str(self.oPc))
        logger.debug(" -> pinRef  : "+str(self.pinRefFrame))
        logger.debug("- - - - - - - - - - - - - -")

    # def getExternalWrenchFromForce(self, rmodel, rdata, f3D):
    #     '''
    #     Compute the vector for pin.Force (external wrenches) due to
    #     the 3D visco-elastic contact force
    #       rmodel  : robot model
    #       rdata   : robot data
    #       f3D     : measured 3D force at contact point
    #     '''
    #     oRf = rdata.oMf[self.frameId].rotation
    #     wrench = [pin.Force.Zero() for _ in range(rmodel.njoints)]
    #     f6D = pin.Force(f3D, np.zeros(3))
    #     parentId = rmodel.frames[self.frameId].parent
    #     jMf = rmodel.frames[self.frameId].placement
    #     if(self.pinRefFrame == pin.LOCAL):
    #         wrench[parentId] = jMf.act(f6D)
    #     elif(self.pinRefFrame == pin.LOCAL_WORLD_ALIGNED):
    #         lwaXf = pin.SE3.Identity() ; lwaXf.rotation = oRf ; lwaXf.translation = np.zeros(3)
    #         wrench[parentId] = jMf.act(lwaXf.actInv(f6D))
    #     return wrench


class SoftContactModel1D:
    def __init__(self, Kp, Kv, oPc, frameId, contactType, pinRef):
        '''
          Kp, Kv      : stiffness and damping coefficient of the visco-elastic contact model
          oPc         : anchor point of the contact 
          frameId     : frame at which the soft contact is defined
          contactType : 1D contact type : 1Dx, 1Dy or 1Dz
          pinRefFrame : reference frame in which the contact model is expressed (pin.LOCAL or pin.LWA)
        '''
        self.nc = 1
        self.Kp = Kp
        self.Kv = Kv
        self.oPc = oPc
        self.pinRefFrame = self.setPinRef(pinRef)
        self.frameId = frameId 
        self.set_contactType(contactType)
        self.contactType = contactType

    def set_contactType(self, contactType):
        assert(contactType in ['1Dx', '1Dy', '1Dz'])
        self.contact_type = contactType
        if(contactType == '1Dx'):
            self.mask = 0
            self.maskType = force_feedback_mpc.Vector3MaskType.x
        if(contactType == '1Dy'):
            self.mask = 1
            self.maskType = force_feedback_mpc.Vector3MaskType.y
        if(contactType == '1Dz'):
            self.mask = 2
            self.maskType = force_feedback_mpc.Vector3MaskType.z
       
    def setPinRef(self, pinRef):
        if(type(pinRef) == str):
            if(pinRef == 'LOCAL'):
                return pin.LOCAL
            elif(pinRef == 'LOCAL_WORLD_ALIGNED'):
                return pin.LOCAL_WORLD_ALIGNED
            else:
                logger.error("yaml config file : pinRefFrame must be in either LOCAL or LOCAL_WORLD_ALIGNED !")
        else:
            return pinRef

    def computeForce(self, rmodel, rdata):
        oRf = rdata.oMf[self.frameId].rotation
        oPf = rdata.oMf[self.frameId].translation
        lv = pin.getFrameVelocity(rmodel, rdata, self.frameId, pin.LOCAL).linear
        # print(lv)
        if(self.pinRefFrame == pin.LOCAL):
            f = (-self.Kp * oRf.T @ (oPf - self.oPc) - self.Kv * lv)[self.mask]
        elif(self.pinRefFrame == pin.LOCAL_WORLD_ALIGNED):
            f = (-self.Kp * (oPf - self.oPc) - self.Kv * oRf @ lv)[self.mask]
        return f

    def computeForce_(self, rmodel, q, v):
        rdata = rmodel.createData()
        pin.forwardKinematics(rmodel, rdata, q, v)
        pin.updateFramePlacements(rmodel, rdata)
        return self.computeForce(rmodel, rdata)

    def computeExternalWrench(self, rmodel, rdata):
        f1D = self.computeForce(rmodel, rdata)
        oRf = rdata.oMf[self.frameId].rotation
        wrench = [pin.Force.Zero() for _ in range(rmodel.njoints)]
        f3D = np.zeros(3) ; f3D[self.mask] = f1D
        f6D = pin.Force(f3D, np.zeros(3))
        parentId = rmodel.frames[self.frameId].parent
        jMf = rmodel.frames[self.frameId].placement
        if(self.pinRefFrame == pin.LOCAL):
            wrench[parentId] = jMf.act(f6D)
        elif(self.pinRefFrame == pin.LOCAL_WORLD_ALIGNED):
            lwaXf = pin.SE3.Identity() ; lwaXf.rotation = oRf ; lwaXf.translation = np.zeros(3)
            wrench[parentId] = jMf.act(lwaXf.actInv(f6D))
        return wrench

    def computeExternalWrench_(self, rmodel, q, v):
        '''
        Compute the vector for pin.Force (external wrenches) due to
        the 3D visco-elastic contact force from (q, v)
          rmodel : robot model
          rdata  : robot data
        '''
        rdata = rmodel.createData()
        pin.forwardKinematics(rmodel, rdata, q, v)
        pin.updateFramePlacements(rmodel, rdata)
        return self.computeExternalWrench(rmodel, rdata)
    
    def print(self):
        logger.debug("- - - - - - - - - - - - - -")
        logger.debug("Contact model parameters")
        logger.debug(" -> frameId : "+str(self.frameId))
        logger.debug(" -> Kp      : "+str(self.Kp))
        logger.debug(" -> Kv      : "+str(self.Kv))
        logger.debug(" -> oPc     : "+str(self.oPc))
        logger.debug(" -> pinRef  : "+str(self.pinRefFrame))
        logger.debug(" -> mask    : "+str(self.maskType))
        logger.debug("- - - - - - - - - - - - - -")
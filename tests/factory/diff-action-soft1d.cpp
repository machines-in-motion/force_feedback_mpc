///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, University of Edinburgh, CTU, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "diff-action-soft1d.hpp"

#include <crocoddyl/core/costs/residual.hpp>
#include <crocoddyl/core/residuals/control.hpp>
#include <crocoddyl/core/utils/exception.hpp>
#include <crocoddyl/multibody/actuations/floating-base.hpp>
#include <crocoddyl/multibody/actuations/full.hpp>
#include <crocoddyl/multibody/states/multibody.hpp>

#include "contact1d.hpp"
#include "cost.hpp"
// #include "sobec/crocomplements/softcontact/dam1d.hpp"

namespace sobec {
namespace unittest {

const std::vector<DAMSoftContact1DTypes::Type>
    DAMSoftContact1DTypes::all(DAMSoftContact1DTypes::init_all());

std::ostream& operator<<(std::ostream& os,
                         DAMSoftContact1DTypes::Type dam_type) {
  switch (dam_type) {
    case DAMSoftContact1DTypes::DAMSoftContact1DAugmentedFwdDynamics_TalosArm:
      os << "DAMSoftContact1DAugmentedFwdDynamics_TalosArm";
      break;
    case DAMSoftContact1DTypes::DAMSoftContact1DAugmentedFwdDynamics_HyQ:
      os << "DAMSoftContact1DAugmentedFwdDynamics_HyQ";
      break;
    case DAMSoftContact1DTypes::DAMSoftContact1DAugmentedFwdDynamics_RandomHumanoid:
      os << "DAMSoftContact1DAugmentedFwdDynamics_RandomHumanoid";
      break;
    case DAMSoftContact1DTypes::DAMSoftContact1DAugmentedFwdDynamics_Talos:
      os << "DAMSoftContact1DAugmentedFwdDynamics_Talos";
      break;
    default:
      break;
  }
  return os;
}

DAMSoftContact1DFactory::DAMSoftContact1DFactory() {}
DAMSoftContact1DFactory::~DAMSoftContact1DFactory() {}


boost::shared_ptr<sobec::DAMSoftContact1DAugmentedFwdDynamics>
DAMSoftContact1DFactory::create(DAMSoftContact1DTypes::Type dam_type,
                              PinocchioReferenceTypes::Type ref_type,
                              ContactModelMaskTypes::Type mask_type) const {
  boost::shared_ptr<sobec::DAMSoftContact1DAugmentedFwdDynamics> action;
  switch (dam_type) {
    // TalosArm 
    case DAMSoftContact1DTypes::
        DAMSoftContact1DAugmentedFwdDynamics_TalosArm:
      action = create_augmentedDAMSoft1D(
          StateModelTypes::StateMultibody_TalosArm,
          ActuationModelTypes::ActuationModelFull, ref_type, mask_type);
      break;
    // HyQ  
    case DAMSoftContact1DTypes::
        DAMSoftContact1DAugmentedFwdDynamics_HyQ:
      action = create_augmentedDAMSoft1D(
          StateModelTypes::StateMultibody_HyQ,
          ActuationModelTypes::ActuationModelFloatingBase, ref_type, mask_type);
      break;
    // RandmHumanoid  
    case DAMSoftContact1DTypes::
        DAMSoftContact1DAugmentedFwdDynamics_RandomHumanoid:
      action = create_augmentedDAMSoft1D(
          StateModelTypes::StateMultibody_RandomHumanoid,
          ActuationModelTypes::ActuationModelFloatingBase, ref_type, mask_type);
      break;
    // Talos  
    case DAMSoftContact1DTypes::
        DAMSoftContact1DAugmentedFwdDynamics_Talos:
      action = create_augmentedDAMSoft1D(
          StateModelTypes::StateMultibody_Talos,
          ActuationModelTypes::ActuationModelFloatingBase, ref_type, mask_type);
      break;
    default:
      throw_pretty(__FILE__ ": Wrong DAMSoftContact1DTypes::Type given");
      break;
  }
  return action;
}


boost::shared_ptr<sobec::DAMSoftContact1DAugmentedFwdDynamics>
DAMSoftContact1DFactory::create_augmentedDAMSoft1D(StateModelTypes::Type state_type,
                                                 ActuationModelTypes::Type actuation_type,
                                                 PinocchioReferenceTypes::Type ref_type,
                                                 ContactModelMaskTypes::Type mask_type) const {
  boost::shared_ptr<sobec::DAMSoftContact1DAugmentedFwdDynamics> action;
  boost::shared_ptr<crocoddyl::StateMultibody> state;
  boost::shared_ptr<crocoddyl::ActuationModelAbstract> actuation;
  boost::shared_ptr<crocoddyl::ContactModelMultiple> contact;
  boost::shared_ptr<crocoddyl::CostModelSum> cost;
  state = boost::static_pointer_cast<crocoddyl::StateMultibody>(StateModelFactory().create(state_type));
  actuation = ActuationModelFactory().create(actuation_type, state_type);
  cost = boost::make_shared<crocoddyl::CostModelSum>(state, actuation->get_nu());
  std::string frameName = "";

  pinocchio::ReferenceFrame pinRefFrame;
  switch(ref_type){
    case PinocchioReferenceTypes::Type::LOCAL:
      pinRefFrame = pinocchio::LOCAL;
      break;
    case PinocchioReferenceTypes::Type::LOCAL_WORLD_ALIGNED:
      pinRefFrame = pinocchio::LOCAL_WORLD_ALIGNED;
      break;
    case PinocchioReferenceTypes::Type::WORLD:
      pinRefFrame = pinocchio::LOCAL_WORLD_ALIGNED;
      break;
    default:
      throw_pretty(__FILE__ ": Wrong PinocchioReferenceTypes::Type given");
      break;
  }
  // get 1D mask type
  sobec::Vector3MaskType mask;
  switch (mask_type) {
    case ContactModelMaskTypes::X: {
      mask = sobec::Vector3MaskType::x;
      break;
    }
    case ContactModelMaskTypes::Y: {
      mask = sobec::Vector3MaskType::y;
      break;
    }
    case ContactModelMaskTypes::Z: {
      mask = sobec::Vector3MaskType::z;
      break;
    }
    default:
      throw_pretty(__FILE__ ": Wrong ContactModelMaskTypes::Type given");
      break;
  }
  switch (state_type) {
    case StateModelTypes::StateMultibody_TalosArm: {
      frameName = "gripper_left_fingertip_1_link";
      break;
    }
    case StateModelTypes::StateMultibody_HyQ: {
      frameName = "rh_haa_joint";
      break;
    }
    case StateModelTypes::StateMultibody_RandomHumanoid: {
      frameName = "rleg6_body";
      break;
    }
    case StateModelTypes::StateMultibody_Talos: {
      frameName = "arm_right_7_link";
      break;
    }
    default:
      throw_pretty(__FILE__ ": Wrong soft contact frame name given");
      break;
  }

  cost->addCost(
      "control",
      CostModelFactory().create(
          CostModelTypes::CostModelResidualControl, state_type,
          ActivationModelTypes::ActivationModelQuad, actuation->get_nu()),
      0.1);
  Eigen::VectorXd Kp = Eigen::VectorXd::Ones(1)*100;
  Eigen::VectorXd Kv = Eigen::VectorXd::Ones(1)*10;
  Eigen::Vector3d oPc = Eigen::Vector3d::Zero();
  action = boost::make_shared<sobec::DAMSoftContact1DAugmentedFwdDynamics>(
      state, 
      actuation, 
      cost, 
      state->get_pinocchio()->getFrameId(frameName), 
      Kp, Kv, oPc, pinRefFrame, mask);
  action->set_force_des(Eigen::VectorXd::Zero(1));
  action->set_force_weight(Eigen::VectorXd::Ones(1)*0.01);
  action->set_with_force_cost(true);
  action->set_tau_grav_weight(0.01);
  action->set_with_gravity_torque_reg(true);
  action->set_with_force_rate_reg_cost(true);
  action->set_force_rate_reg_weight(1e-6*Eigen::VectorXd::Ones(1));
  pinocchio::ReferenceFrame cost_ref = pinocchio::LOCAL;
  action->set_cost_ref(cost_ref);
  return action;
}



}  // namespace unittest
}  // namespace sobec

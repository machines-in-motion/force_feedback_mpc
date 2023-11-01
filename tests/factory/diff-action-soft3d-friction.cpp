///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, University of Edinburgh, CTU, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "diff-action-soft3d-friction.hpp"

#include <crocoddyl/core/costs/residual.hpp>
#include <crocoddyl/core/residuals/control.hpp>
#include <crocoddyl/core/utils/exception.hpp>
#include <crocoddyl/multibody/actuations/floating-base.hpp>
#include <crocoddyl/multibody/actuations/full.hpp>
#include <crocoddyl/multibody/states/multibody.hpp>

#include "contact3d.hpp"
#include "cost.hpp"


namespace sobec {
namespace unittest {

const std::vector<DAMSoftContact3DFrictionTypes::Type>
    DAMSoftContact3DFrictionTypes::all(DAMSoftContact3DFrictionTypes::init_all());

std::ostream& operator<<(std::ostream& os,
                         DAMSoftContact3DFrictionTypes::Type dam_type) {
  switch (dam_type) {
    case DAMSoftContact3DFrictionTypes::
        DAMSoftContact3DAugmentedFrictionFwdDynamics_TalosArm:
      os << "DAMSoftContact3DAugmentedFrictionFwdDynamics_TalosArm";
      break;
    case DAMSoftContact3DFrictionTypes::
        DAMSoftContact3DAugmentedFrictionFwdDynamics_HyQ:
      os << "DAMSoftContact3DAugmentedFrictionFwdDynamics_HyQ";
      break;
    case DAMSoftContact3DFrictionTypes::
        DAMSoftContact3DAugmentedFrictionFwdDynamics_RandomHumanoid:
      os << "DAMSoftContact3DAugmentedFrictionFwdDynamics_RandomHumanoid";
      break;
    case DAMSoftContact3DFrictionTypes::
        DAMSoftContact3DAugmentedFrictionFwdDynamics_Talos:
      os << "DAMSoftContact3DAugmentedFrictionFwdDynamics_Talos";
      break;
    default:
      break;
  }
  return os;
}

DAMSoftContact3DFrictionFactory::DAMSoftContact3DFrictionFactory() {}
DAMSoftContact3DFrictionFactory::~DAMSoftContact3DFrictionFactory() {}

boost::shared_ptr<sobec::DAMSoftContact3DAugmentedFrictionFwdDynamics>
DAMSoftContact3DFrictionFactory::create(DAMSoftContact3DFrictionTypes::Type dam_type,
                              PinocchioReferenceTypes::Type ref_type) const {
  boost::shared_ptr<sobec::DAMSoftContact3DAugmentedFrictionFwdDynamics> action;
  switch (dam_type) {
    // TalosArm 
    case DAMSoftContact3DFrictionTypes::
        DAMSoftContact3DAugmentedFrictionFwdDynamics_TalosArm:
      action = create_augmentedDAMSoft3DFriction(
          StateModelTypes::StateMultibody_TalosArm,
          ActuationModelTypes::ActuationModelFull, ref_type);
      break;
    // HyQ  
    case DAMSoftContact3DFrictionTypes::
        DAMSoftContact3DAugmentedFrictionFwdDynamics_HyQ:
      action = create_augmentedDAMSoft3DFriction(
          StateModelTypes::StateMultibody_HyQ,
          ActuationModelTypes::ActuationModelFloatingBase, ref_type);
      break;
    // RandmHumanoid  
    case DAMSoftContact3DFrictionTypes::
        DAMSoftContact3DAugmentedFrictionFwdDynamics_RandomHumanoid:
      action = create_augmentedDAMSoft3DFriction(
          StateModelTypes::StateMultibody_RandomHumanoid,
          ActuationModelTypes::ActuationModelFloatingBase, ref_type);
      break;
    // Talos  
    case DAMSoftContact3DFrictionTypes::
        DAMSoftContact3DAugmentedFrictionFwdDynamics_Talos:
      action = create_augmentedDAMSoft3DFriction(
          StateModelTypes::StateMultibody_Talos,
          ActuationModelTypes::ActuationModelFloatingBase, ref_type);
      break;
    default:
      throw_pretty(__FILE__ ": Wrong DAMSoftContact3DFrictionTypes::Type given");
      break;
  }
  return action;
}

boost::shared_ptr<sobec::DAMSoftContact3DAugmentedFrictionFwdDynamics>
DAMSoftContact3DFrictionFactory::create_augmentedDAMSoft3DFriction(StateModelTypes::Type state_type,
                                                 ActuationModelTypes::Type actuation_type,
                                                 PinocchioReferenceTypes::Type ref_type) const {
  boost::shared_ptr<sobec::DAMSoftContact3DAugmentedFrictionFwdDynamics> action;
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
  Eigen::VectorXd Kp = Eigen::VectorXd::Ones(3)*100;
  Eigen::VectorXd Kv = Eigen::VectorXd::Ones(3)*10;
  Eigen::Vector3d oPc = Eigen::Vector3d::Zero();
  action = boost::make_shared<sobec::DAMSoftContact3DAugmentedFrictionFwdDynamics>(
      state, 
      actuation, 
      cost, 
      state->get_pinocchio()->getFrameId(frameName), 
      Kp, Kv, oPc, pinRefFrame);
  action->set_force_des(Eigen::Vector3d::Zero());
  action->set_force_weight(Eigen::Vector3d::Ones());
  action->set_with_force_cost(true);
  action->set_tau_grav_weight(0.01);
  action->set_with_gravity_torque_reg(true);
  action->set_with_force_rate_reg_cost(true);
  action->set_force_rate_reg_weight(1e-6*Eigen::Vector3d::Ones());
  pinocchio::ReferenceFrame cost_ref = pinocchio::LOCAL;
  action->set_cost_ref(cost_ref);
  action->set_mu(0.1);
  action->set_eps(10.);
  return action;
}



}  // namespace unittest
}  // namespace sobec

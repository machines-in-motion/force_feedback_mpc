///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, University of Edinburgh, CTU, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef FORCE_FEEDBACK_MPC_DIFF_ACTION_SOFT3D_FACTORY_HPP_
#define FORCE_FEEDBACK_MPC_DIFF_ACTION_SOFT3D_FACTORY_HPP_

#include <crocoddyl/core/diff-action-base.hpp>
#include <crocoddyl/core/numdiff/diff-action.hpp>
#include <crocoddyl/multibody/actions/free-fwddyn.hpp>

#include "crocoddyl/actuation.hpp"
#include "crocoddyl/contact.hpp"
#include "crocoddyl/state.hpp"
#include "crocoddyl/cost.hpp"
#include "force_feedback_mpc/softcontact/dam3d-augmented.hpp"

namespace force_feedback_mpc {
namespace unittest {


struct DAMSoftContact3DTypes {
  enum Type {
    DAMSoftContact3DAugmentedFwdDynamics_TalosArm,
    DAMSoftContact3DAugmentedFwdDynamics_HyQ,
    DAMSoftContact3DAugmentedFwdDynamics_RandomHumanoid,
    DAMSoftContact3DAugmentedFwdDynamics_Talos,
    NbDAMSoftContact3DTypes
  };
  static std::vector<Type> init_all() {
    std::vector<Type> v;
    v.clear();
    for (int i = 0; i < NbDAMSoftContact3DTypes; ++i) {
      v.push_back((Type)i);
    }
    return v;
  }
  static const std::vector<Type> all;
};

std::ostream& operator<<(std::ostream& os,
                         DAMSoftContact3DTypes::Type type);

class DAMSoftContact3DFactory {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit DAMSoftContact3DFactory();
  ~DAMSoftContact3DFactory();

  boost::shared_ptr<force_feedback_mpc::softcontact::DAMSoftContact3DAugmentedFwdDynamics> create(
      DAMSoftContact3DTypes::Type type,
      pinocchio::ReferenceFrame ref_type = pinocchio::LOCAL) const;

  // Soft contact 3D dynamics
  boost::shared_ptr<force_feedback_mpc::softcontact::DAMSoftContact3DAugmentedFwdDynamics>
  create_augmentedDAMSoft3D(StateModelTypes::Type state_type,
                            ActuationModelTypes::Type actuation_type,
                            pinocchio::ReferenceFrame ref_type) const;
};

}  // namespace unittest
}  // namespace force_feedback_mpc

#endif  // FORCE_FEEDBACK_MPC_DIFF_ACTION_SOFT3D_FACTORY_HPP_

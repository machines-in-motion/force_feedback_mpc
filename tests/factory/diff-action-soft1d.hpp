///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, University of Edinburgh, CTU, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef FORCE_FEEDBACK_MPC_DIFF_ACTION_SOFT1D_FACTORY_HPP_
#define FORCE_FEEDBACK_MPC_DIFF_ACTION_SOFT1D_FACTORY_HPP_

#include <crocoddyl/core/diff-action-base.hpp>
#include <crocoddyl/core/numdiff/diff-action.hpp>
#include <crocoddyl/multibody/actions/free-fwddyn.hpp>

#include "crocoddyl/actuation.hpp"
#include "crocoddyl/contact.hpp"
#include "crocoddyl/state.hpp"
#include "crocoddyl/cost.hpp"
#include "force_feedback_mpc/softcontact/dam1d-augmented.hpp"

namespace force_feedback_mpc {
namespace unittest {


struct DAMSoftContact1DTypes {
  enum Type {
    DAMSoftContact1DAugmentedFwdDynamics_TalosArm,
    DAMSoftContact1DAugmentedFwdDynamics_HyQ,
    DAMSoftContact1DAugmentedFwdDynamics_RandomHumanoid,
    DAMSoftContact1DAugmentedFwdDynamics_Talos,
    NbDAMSoftContact1DTypes
  };
  static std::vector<Type> init_all() {
    std::vector<Type> v;
    v.clear();
    for (int i = 0; i < NbDAMSoftContact1DTypes; ++i) {
      v.push_back((Type)i);
    }
    return v;
  }
  static const std::vector<Type> all;
};

std::ostream& operator<<(std::ostream& os,
                         DAMSoftContact1DTypes::Type type);

class DAMSoftContact1DFactory {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef typename force_feedback_mpc::softcontact::Vector3MaskType Vector3MaskType;

  explicit DAMSoftContact1DFactory();
  ~DAMSoftContact1DFactory();

  std::shared_ptr<force_feedback_mpc::softcontact::DAMSoftContact1DAugmentedFwdDynamics> create(
      DAMSoftContact1DTypes::Type type,
      pinocchio::ReferenceFrame ref_type = pinocchio::LOCAL,
      Vector3MaskType mask_type = Vector3MaskType::z) const;

  // Soft contact 1D dynamics
  std::shared_ptr<force_feedback_mpc::softcontact::DAMSoftContact1DAugmentedFwdDynamics>
  create_augmentedDAMSoft1D(StateModelTypes::Type state_type,
                            ActuationModelTypes::Type actuation_type,
                            pinocchio::ReferenceFrame ref_type,
                            Vector3MaskType mask_type) const;
};

}  // namespace unittest
}  // namespace force_feedback_mpc

#endif  // FORCE_FEEDBACK_MPC_DIFF_ACTION_SOFT1D_FACTORY_HPP_

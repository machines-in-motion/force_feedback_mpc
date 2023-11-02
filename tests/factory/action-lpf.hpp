///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef FORCE_FEEDBACK_MPC_ACTION_LPF_FACTORY_HPP_
#define FORCE_FEEDBACK_MPC_ACTION_LPF_FACTORY_HPP_

#include <iterator>

#include "crocoddyl/core/action-base.hpp"
#include "crocoddyl/core/numdiff/action.hpp"
#include "diff-action.hpp"
#include "force_feedback_mpc/lowpassfilter/action.hpp"
#include "statelpf.hpp"

namespace force_feedback_mpc {
namespace unittest {

struct ActionModelLPFTypes {
  enum Type {
    IntegratedActionModelLPF_ALL,
    IntegratedActionModelLPF_RAND,
    IntegratedActionModelLPF_NONE,
    IntegratedActionModelLPF_alpha0,
    // IntegratedActionModelLPF_terminal,
    NbActionModelLPFTypes
  };
  static std::vector<Type> init_all() {
    std::vector<Type> v;
    v.clear();
    for (int i = 0; i < NbActionModelLPFTypes; ++i) {
      v.push_back((Type)i);
    }
    return v;
  }
  static const std::vector<Type> all;
};

std::ostream& operator<<(std::ostream& os, ActionModelLPFTypes::Type type);

class ActionModelLPFFactory {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit ActionModelLPFFactory();
  ~ActionModelLPFFactory();

  boost::shared_ptr<force_feedback_mpc::lpf::IntegratedActionModelLPF> create(
      ActionModelLPFTypes::Type iam_type,
      DifferentialActionModelTypes::Type dam_type,
      PinocchioReferenceTypes::Type ref_type = PinocchioReferenceTypes::LOCAL,
      ContactModelMaskTypes::Type mask_type = ContactModelMaskTypes::Z) const;
  // LPFJointMaskType lpf_mask_type = LPFJointMaskType::ALL) const;
};

}  // namespace unittest
}  // namespace force_feedback_mpc

#endif  // FORCE_FEEDBACK_MPC_ACTION_LPF_FACTORY_HPP_

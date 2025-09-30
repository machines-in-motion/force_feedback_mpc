///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, University of Edinburgh, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "action-soft.hpp"
#include <crocoddyl/core/utils/exception.hpp>

namespace force_feedback_mpc {
namespace unittest {

const std::vector<IAMSoftContactTypes::Type> IAMSoftContactTypes::all(
    IAMSoftContactTypes::init_all());

std::ostream& operator<<(std::ostream& os, IAMSoftContactTypes::Type type) {
  switch (type) {
    case IAMSoftContactTypes::IAMSoftContactAugmented:
      os << "IAMSoftContactAugmented";
      break;
    case IAMSoftContactTypes::IAMSoftContact1DAugmented:
      os << "IAMSoftContact1DAugmented";
      break;
    default:
      break;
  }
  return os;
}

IAMSoftContactFactory::IAMSoftContactFactory() {}
IAMSoftContactFactory::~IAMSoftContactFactory() {}

std::shared_ptr<force_feedback_mpc::softcontact::IAMSoftContactAugmented>
IAMSoftContactFactory::create(IAMSoftContactTypes::Type iam_type,
                              DAMSoftContactAbstractTypes::Type dam_type,
                              pinocchio::ReferenceFrame ref_type,
                              Vector3MaskType mask_type) const {
  std::shared_ptr<force_feedback_mpc::softcontact::IAMSoftContactAugmented> iam;
  switch (iam_type) {
    case IAMSoftContactTypes::IAMSoftContactAugmented: {
      std::shared_ptr<force_feedback_mpc::softcontact::DAMSoftContact3DAugmentedFwdDynamics> dam =
          DAMSoftContact3DFactory().create(mapDAMSoftAbstractTo3D.at(dam_type), ref_type);
      double time_step = 1e-3;
      bool with_cost_residual = true;
      iam = std::make_shared<force_feedback_mpc::softcontact::IAMSoftContactAugmented>(
          dam, time_step, with_cost_residual);
      break;
    }
    case IAMSoftContactTypes::IAMSoftContact1DAugmented: {
      std::shared_ptr<force_feedback_mpc::softcontact::DAMSoftContact1DAugmentedFwdDynamics> dam =
          DAMSoftContact1DFactory().create(mapDAMSoftAbstractTo1D.at(dam_type), ref_type, mask_type);
      double time_step = 1e-3;
      bool with_cost_residual = true;
      iam = std::make_shared<force_feedback_mpc::softcontact::IAMSoftContactAugmented>(
          dam, time_step, with_cost_residual);
      break;
    }
    default:
      throw_pretty(__FILE__ ": Wrong IAMSoftContactTypes::Type given");
      break;
  }
  return iam;
}

}  // namespace unittest
}  // namespace force_feedback_mpc

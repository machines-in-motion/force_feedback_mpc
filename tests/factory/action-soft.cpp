///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, University of Edinburgh, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "action-soft.hpp"
#include <crocoddyl/core/utils/exception.hpp>

namespace sobec {
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

boost::shared_ptr<sobec::IAMSoftContactAugmented>
IAMSoftContactFactory::create(IAMSoftContactTypes::Type iam_type,
                              DAMSoftContactAbstractTypes::Type dam_type,
                              PinocchioReferenceTypes::Type ref_type,
                              ContactModelMaskTypes::Type mask_type) const {
  boost::shared_ptr<sobec::IAMSoftContactAugmented> iam;
  switch (iam_type) {
    case IAMSoftContactTypes::IAMSoftContactAugmented: {
      boost::shared_ptr<sobec::DAMSoftContact3DAugmentedFwdDynamics> dam =
          DAMSoftContact3DFactory().create(mapDAMSoftAbstractTo3D.at(dam_type), ref_type);
      double time_step = 1e-3;
      bool with_cost_residual = true;
      iam = boost::make_shared<sobec::IAMSoftContactAugmented>(
          dam, time_step, with_cost_residual);
      break;
    }
    case IAMSoftContactTypes::IAMSoftContact1DAugmented: {
      boost::shared_ptr<sobec::DAMSoftContact1DAugmentedFwdDynamics> dam =
          DAMSoftContact1DFactory().create(mapDAMSoftAbstractTo1D.at(dam_type), ref_type, mask_type);
      double time_step = 1e-3;
      bool with_cost_residual = true;
      iam = boost::make_shared<sobec::IAMSoftContactAugmented>(
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
}  // namespace sobec

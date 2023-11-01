///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, University of Edinburgh, CTU, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "diff-action-soft-abstract.hpp"


namespace sobec {
namespace unittest {

const std::vector<DAMSoftContactAbstractTypes::Type>
    DAMSoftContactAbstractTypes::all(DAMSoftContactAbstractTypes::init_all());

std::ostream& operator<<(std::ostream& os,
                         DAMSoftContactAbstractTypes::Type dam_type) {
  switch (dam_type) {
    case DAMSoftContactAbstractTypes::DAMSoftContactAbstractAugmentedFwdDynamics_TalosArm:
      os << "DAMSoftContactAbstractAugmentedFwdDynamics_TalosArm";
      break;
    case DAMSoftContactAbstractTypes::DAMSoftContactAbstractAugmentedFwdDynamics_HyQ:
      os << "DAMSoftContactAbstractAugmentedFwdDynamics_HyQ";
      break;
    case DAMSoftContactAbstractTypes::DAMSoftContactAbstractAugmentedFwdDynamics_RandomHumanoid:
      os << "DAMSoftContactAbstractAugmentedFwdDynamics_RandomHumanoid";
      break;
    case DAMSoftContactAbstractTypes::DAMSoftContactAbstractAugmentedFwdDynamics_Talos:
      os << "DAMSoftContactAbstractAugmentedFwdDynamics_Talos";
      break;
    default:
      break;
  }
  return os;
}


}  // namespace unittest
}  // namespace sobec

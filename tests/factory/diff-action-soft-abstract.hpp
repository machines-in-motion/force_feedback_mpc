///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, University of Edinburgh, CTU, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef SOBEC_DIFF_ACTION_SOFT_FACTORY_HPP_
#define SOBEC_DIFF_ACTION_SOFT_FACTORY_HPP_

#include "sobec/crocomplements/softcontact/dam-augmented.hpp"

namespace sobec {
namespace unittest {

struct DAMSoftContactAbstractTypes {
  enum Type {
    DAMSoftContactAbstractAugmentedFwdDynamics_TalosArm,
    DAMSoftContactAbstractAugmentedFwdDynamics_HyQ,
    DAMSoftContactAbstractAugmentedFwdDynamics_RandomHumanoid,
    DAMSoftContactAbstractAugmentedFwdDynamics_Talos,
    NbDAMSoftContactAbstractTypes
  };
  static std::vector<Type> init_all() {
    std::vector<Type> v;
    v.clear();
    for (int i = 0; i < NbDAMSoftContactAbstractTypes; ++i) {
      v.push_back((Type)i);
    }
    return v;
  }
  static const std::vector<Type> all;
};

std::ostream& operator<<(std::ostream& os,
                         DAMSoftContactAbstractTypes::Type type);
                         
}  // namespace unittest
}  // namespace sobec

#endif  // SOBEC_DIFF_ACTION_SOFT_FACTORY_HPP_

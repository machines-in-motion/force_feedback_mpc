///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, University of Edinburgh, CTU, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef SOBEC_DIFF_ACTION_SOFT1D_FACTORY_HPP_
#define SOBEC_DIFF_ACTION_SOFT1D_FACTORY_HPP_

#include <crocoddyl/core/diff-action-base.hpp>
#include <crocoddyl/core/numdiff/diff-action.hpp>
#include <crocoddyl/multibody/actions/free-fwddyn.hpp>

#include "actuation.hpp"
#include "contact1d.hpp"
#include "sobec/crocomplements/softcontact/dam1d-augmented.hpp"
#include "state.hpp"

namespace sobec {
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

  explicit DAMSoftContact1DFactory();
  ~DAMSoftContact1DFactory();

  boost::shared_ptr<sobec::DAMSoftContact1DAugmentedFwdDynamics> create(
      DAMSoftContact1DTypes::Type type,
      PinocchioReferenceTypes::Type ref_type = PinocchioReferenceTypes::LOCAL,
      ContactModelMaskTypes::Type mask_type = ContactModelMaskTypes::Type::Z) const;

  // Soft contact 1D dynamics
  boost::shared_ptr<sobec::DAMSoftContact1DAugmentedFwdDynamics>
  create_augmentedDAMSoft1D(StateModelTypes::Type state_type,
                            ActuationModelTypes::Type actuation_type,
                            PinocchioReferenceTypes::Type ref_type,
                            ContactModelMaskTypes::Type mask_type) const;
};

}  // namespace unittest
}  // namespace sobec

#endif  // SOBEC_DIFF_ACTION_SOFT1D_FACTORY_HPP_

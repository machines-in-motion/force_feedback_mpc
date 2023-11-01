///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, University of Edinburgh, CTU, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef SOBEC_DIFF_ACTION_SOFT3D_FACTORY_HPP_
#define SOBEC_DIFF_ACTION_SOFT3D_FACTORY_HPP_

#include <crocoddyl/core/diff-action-base.hpp>
#include <crocoddyl/core/numdiff/diff-action.hpp>
#include <crocoddyl/multibody/actions/free-fwddyn.hpp>

#include "actuation.hpp"
#include "contact1d.hpp"
#include "sobec/crocomplements/softcontact/dam3d-augmented.hpp"
#include "state.hpp"

namespace sobec {
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

  boost::shared_ptr<sobec::DAMSoftContact3DAugmentedFwdDynamics> create(
      DAMSoftContact3DTypes::Type type,
      PinocchioReferenceTypes::Type ref_type = PinocchioReferenceTypes::LOCAL) const;

  // Soft contact 3D dynamics
  boost::shared_ptr<sobec::DAMSoftContact3DAugmentedFwdDynamics>
  create_augmentedDAMSoft3D(StateModelTypes::Type state_type,
                            ActuationModelTypes::Type actuation_type,
                            PinocchioReferenceTypes::Type ref_type) const;
};

}  // namespace unittest
}  // namespace sobec

#endif  // SOBEC_DIFF_ACTION_SOFT3D_FACTORY_HPP_

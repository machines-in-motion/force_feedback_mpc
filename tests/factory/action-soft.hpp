///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef SOBEC_ACTION_IAM3D_AUGMENTED_FACTORY_HPP_
#define SOBEC_ACTION_IAM3D_AUGMENTED_FACTORY_HPP_

#include <iterator>

#include "crocoddyl/core/action-base.hpp"
#include "crocoddyl/core/numdiff/action.hpp"

#include "sobec/crocomplements/softcontact/iam-augmented.hpp"
#include "statesoft.hpp"

#include "diff-action-soft-abstract.hpp"
#include "diff-action-soft3d.hpp"
#include "diff-action-soft1d.hpp"

namespace sobec {
namespace unittest {

struct IAMSoftContactTypes {
  enum Type {
    IAMSoftContactAugmented,
    IAMSoftContact1DAugmented,
    NbIAMSoftContactTypes
  };
  static std::vector<Type> init_all() {
    std::vector<Type> v;
    v.clear();
    for (int i = 0; i < NbIAMSoftContactTypes; ++i) {
      v.push_back((Type)i);
    }
    return v;
  }
  static const std::vector<Type> all;
};

const std::map<DAMSoftContactAbstractTypes::Type, DAMSoftContact3DTypes::Type>
    mapDAMSoftAbstractTo3D{
        {DAMSoftContactAbstractTypes::DAMSoftContactAbstractAugmentedFwdDynamics_TalosArm,
            DAMSoftContact3DTypes::DAMSoftContact3DAugmentedFwdDynamics_TalosArm},
        {DAMSoftContactAbstractTypes::DAMSoftContactAbstractAugmentedFwdDynamics_HyQ, 
            DAMSoftContact3DTypes::DAMSoftContact3DAugmentedFwdDynamics_HyQ},
        {DAMSoftContactAbstractTypes::DAMSoftContactAbstractAugmentedFwdDynamics_RandomHumanoid, 
            DAMSoftContact3DTypes::DAMSoftContact3DAugmentedFwdDynamics_RandomHumanoid},
        {DAMSoftContactAbstractTypes::DAMSoftContactAbstractAugmentedFwdDynamics_Talos, 
            DAMSoftContact3DTypes::DAMSoftContact3DAugmentedFwdDynamics_Talos}};

const std::map<DAMSoftContactAbstractTypes::Type, DAMSoftContact1DTypes::Type>
    mapDAMSoftAbstractTo1D{
        {DAMSoftContactAbstractTypes::DAMSoftContactAbstractAugmentedFwdDynamics_TalosArm,
            DAMSoftContact1DTypes::DAMSoftContact1DAugmentedFwdDynamics_TalosArm},
        {DAMSoftContactAbstractTypes::DAMSoftContactAbstractAugmentedFwdDynamics_HyQ, 
            DAMSoftContact1DTypes::DAMSoftContact1DAugmentedFwdDynamics_HyQ},
        {DAMSoftContactAbstractTypes::DAMSoftContactAbstractAugmentedFwdDynamics_RandomHumanoid, 
            DAMSoftContact1DTypes::DAMSoftContact1DAugmentedFwdDynamics_RandomHumanoid},
        {DAMSoftContactAbstractTypes::DAMSoftContactAbstractAugmentedFwdDynamics_Talos, 
            DAMSoftContact1DTypes::DAMSoftContact1DAugmentedFwdDynamics_Talos}};

std::ostream& operator<<(std::ostream& os, IAMSoftContactTypes::Type type);

class IAMSoftContactFactory {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit IAMSoftContactFactory();
  ~IAMSoftContactFactory();

  boost::shared_ptr<sobec::IAMSoftContactAugmented> create(
      IAMSoftContactTypes::Type iam_type,
      DAMSoftContactAbstractTypes::Type dam_type,
      PinocchioReferenceTypes::Type ref_type = PinocchioReferenceTypes::LOCAL,
      ContactModelMaskTypes::Type mask_type = ContactModelMaskTypes::Type::Z) const;
};

}  // namespace unittest
}  // namespace sobec

#endif  // SOBEC_ACTION_IAM3D_AUGMENTED_FACTORY_HPP_

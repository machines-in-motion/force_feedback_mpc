///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef FORCE_FEEDBACK_MPC_ACTION_IAM3D_AUGMENTED_FACTORY_HPP_
#define FORCE_FEEDBACK_MPC_ACTION_IAM3D_AUGMENTED_FACTORY_HPP_

#include <iterator>

#include <crocoddyl/core/diff-action-base.hpp>
#include <crocoddyl/core/numdiff/diff-action.hpp>

#include "force_feedback_mpc/softcontact/iam-augmented.hpp"
#include "state-soft.hpp"
#include "diff-action-soft-abstract.hpp"
#include "diff-action-soft3d.hpp"
#include "diff-action-soft1d.hpp"


namespace force_feedback_mpc {
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

  typedef typename force_feedback_mpc::softcontact::Vector3MaskType Vector3MaskType;

  explicit IAMSoftContactFactory();
  ~IAMSoftContactFactory();

  boost::shared_ptr<force_feedback_mpc::softcontact::IAMSoftContactAugmented> create(
      IAMSoftContactTypes::Type iam_type,
      DAMSoftContactAbstractTypes::Type dam_type,
        pinocchio::ReferenceFrame ref_type,
        Vector3MaskType mask_type) const;
};

}  // namespace unittest
}  // namespace force_feedback_mpc
#endif  // FORCE_FEEDBACK_MPC_ACTION_IAM3D_AUGMENTED_FACTORY_HPP_

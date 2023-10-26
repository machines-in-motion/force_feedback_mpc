///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef FORCE_FEEDBACK_MPC_FWD_HPP_
#define FORCE_FEEDBACK_MPC_FWD_HPP_

#include <crocoddyl/core/action-base.hpp>
#include <crocoddyl/core/fwd.hpp>
#include <crocoddyl/core/integrator/euler.hpp>
#include <crocoddyl/core/utils/exception.hpp>
#include <crocoddyl/multibody/actions/contact-fwddyn.hpp>
#include <crocoddyl/multibody/fwd.hpp>
#include <crocoddyl/multibody/states/multibody.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/fwd.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>


namespace force_feedback_mpc {

// State LPF
template <typename Scalar>
class StateLPFTpl;
typedef StateLPFTpl<double> StateLPF;

// IAM LPF
template <typename Scalar>
class IntegratedActionModelLPFTpl;
typedef IntegratedActionModelLPFTpl<double> IntegratedActionModelLPF;
template <typename Scalar>
class IntegratedActionDataLPFTpl;
typedef IntegratedActionDataLPFTpl<double> IntegratedActionDataLPF;

// // Soft contact3D DAM  
//     // 3D
// template <typename Scalar>
// class DifferentialActionModelSoftContact3DFwdDynamicsTpl;
// typedef DifferentialActionModelSoftContact3DFwdDynamicsTpl<double> DifferentialActionModelSoftContact3DFwdDynamics;
// template <typename Scalar>
// class DifferentialActionDataSoftContact3DFwdDynamicsTpl;
// typedef DifferentialActionDataSoftContact3DFwdDynamicsTpl<double> DifferentialActionDataSoftContact3DFwdDynamics;
//     // 1D
// template <typename Scalar>
// class DifferentialActionModelSoftContact1DFwdDynamicsTpl;
// typedef DifferentialActionModelSoftContact1DFwdDynamicsTpl<double> DifferentialActionModelSoftContact1DFwdDynamics;
// template <typename Scalar>
// class DifferentialActionDataSoftContact1DFwdDynamicsTpl;
// typedef DifferentialActionDataSoftContact1DFwdDynamicsTpl<double> DifferentialActionDataSoftContact1DFwdDynamics;


// // Soft contact3D DAM (augmented state)
//     // Abstract
// template <typename Scalar>
// class DAMSoftContactAbstractAugmentedFwdDynamicsTpl;
// typedef DAMSoftContactAbstractAugmentedFwdDynamicsTpl<double> DAMSoftContactAbstractAugmentedFwdDynamics;
// template <typename Scalar>
// class DADSoftContactAbstractAugmentedFwdDynamicsTpl;
// typedef DADSoftContactAbstractAugmentedFwdDynamicsTpl<double> DADSoftContactAbstractAugmentedFwdDynamics;
//     // 3D
// template <typename Scalar>
// class DAMSoftContact3DAugmentedFwdDynamicsTpl;
// typedef DAMSoftContact3DAugmentedFwdDynamicsTpl<double> DAMSoftContact3DAugmentedFwdDynamics;
// template <typename Scalar>
// class DADSoftContact3DAugmentedFwdDynamicsTpl;
// typedef DADSoftContact3DAugmentedFwdDynamicsTpl<double> DADSoftContact3DAugmentedFwdDynamics;
//     // 1D
// template <typename Scalar>
// class DAMSoftContact1DAugmentedFwdDynamicsTpl;
// typedef DAMSoftContact1DAugmentedFwdDynamicsTpl<double> DAMSoftContact1DAugmentedFwdDynamics;
// template <typename Scalar>
// class DADSoftContact1DAugmentedFwdDynamicsTpl;
// typedef DADSoftContact1DAugmentedFwdDynamicsTpl<double> DADSoftContact1DAugmentedFwdDynamics;
//     // 3D
// template <typename Scalar>
// class DAMSoftContact3DAugmentedFrictionFwdDynamicsTpl;
// typedef DAMSoftContact3DAugmentedFrictionFwdDynamicsTpl<double> DAMSoftContact3DAugmentedFrictionFwdDynamics;
// template <typename Scalar>
// class DADSoftContact3DAugmentedFrictionFwdDynamicsTpl;
// typedef DADSoftContact3DAugmentedFrictionFwdDynamicsTpl<double> DADSoftContact3DAugmentedFrictionFwdDynamics;
// // State soft contact
// template <typename Scalar>
// class StateSoftContactTpl;
// typedef StateSoftContactTpl<double> StateSoftContact;
// // IAM Soft contact 3D
// template <typename Scalar>
// class IAMSoftContactAugmentedTpl;
// typedef IAMSoftContactAugmentedTpl<double> IAMSoftContactAugmented;
// template <typename Scalar>
// class IADSoftContactAugmentedTpl;
// typedef IADSoftContactAugmentedTpl<double> IADSoftContactAugmented;

// enum Vector3MaskType { x = 0, y = 1, z = 2 };

}  // namespace force_feedback_mpc

#endif  // FORCE_FEEDBACK_MPC_FWD_HPP_

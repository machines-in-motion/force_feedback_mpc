///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <crocoddyl/core/utils/exception.hpp>
#include <crocoddyl/core/utils/math.hpp>
#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/algorithm/contact-dynamics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>
#include <pinocchio/algorithm/rnea.hpp>

#include "force_feedback_mpc/softcontact/dam-augmented.hpp"

using namespace crocoddyl;


namespace force_feedback_mpc {
namespace softcontact {


DAMSoftContactAbstractAugmentedFwdDynamics::DAMSoftContactAbstractAugmentedFwdDynamics(
    boost::shared_ptr<StateMultibody> state, 
    boost::shared_ptr<ActuationModelAbstract> actuation,
    boost::shared_ptr<CostModelSum> costs,
    const pinocchio::FrameIndex frameId,
    const VectorXs& Kp, 
    const VectorXs& Kv,
    const Vector3s& oPc,
    const std::size_t nc,
    boost::shared_ptr<ConstraintModelManager> constraints)
    : DAMBase(state, actuation->get_nu(), costs->get_nr()),
      actuation_(actuation),
      costs_(costs),
      constraints_(constraints),
      pinocchio_(*state->get_pinocchio().get()),
      without_armature_(true),
      armature_(VectorXs::Zero(state->get_nv())) {
  std::cout << "[DAM-augmented.cpp] START OF INIT " << std::endl;
  if (this->get_costs()->get_nu() != this->get_nu()) {
    throw_pretty("Invalid argument: "
                 << "Costs doesn't have the same control dimension (it should be " + std::to_string(this->get_nu()) + ")");
  }
  // std::cout << "pin.effortLimit  = " << this->get_pinocchio().effortLimit.tail(this->get_nu()) << std::endl;
  DAMBase::set_u_lb(double(-1.) * this->get_pinocchio().effortLimit.tail(this->get_nu()));
  DAMBase::set_u_ub(double(+1.) * this->get_pinocchio().effortLimit.tail(this->get_nu()));
  // Soft contact model parameters
  if(Kp.maxCoeff() < double(0.) || Kv.maxCoeff() < double(0.)){
     throw_pretty("Invalid argument: "
                << "Kp and Kv must be positive "); 
  }
  if(Kv.size() != nc || Kv.size() != nc){
     throw_pretty("Invalid argument: "
                << "Kp and Kv must have size " << nc); 
  }
  Kp_ = Kp;
  Kv_ = Kv;
  oPc_ = oPc;
  frameId_ = frameId;
  // By default the cost is expressed in the same frame as the dynamics
  // and the dynamics is expressed in LOCAL
  ref_ = pinocchio::ReferenceFrame::LOCAL; 
  cost_ref_ = ref_;
  // If gains are too small, set contact to inactive
  if(Kp.maxCoeff() <= double(1e-9) && Kv.maxCoeff() <= double(1e-9)){
    active_contact_ = false;
  } else {
    active_contact_ = true;
  }
  nc_ = nc;
  parentId_ = this->get_pinocchio().frames[frameId_].parent;
  jMf_ = this->get_pinocchio().frames[frameId_].placement;
  with_armature_ = false;
  armature_ = VectorXs::Zero(this->get_state()->get_nv());
  // Hard-coded cost on force and gravity reg
  with_force_cost_ = false;
  force_weight_ = VectorXs::Zero(nc_);
  force_rate_reg_weight_ = VectorXs::Zero(nc_);
  force_des_ = VectorXs::Zero(nc_);
  with_gravity_torque_reg_ = false;
  tau_grav_weight_ = double(0.);
  with_force_constraint_ = false;
  std::cout << "[DAM-augmented.cpp] Kp_ = " << Kp_.size() << std::endl;
  std::cout << "[DAM-augmented.cpp] Kv_ = " << Kv_.size() << std::endl;
  std::cout << "[DAM-augmented.cpp] oPc_ = " << oPc_.size() << std::endl;
  std::cout << "[DAM-augmented.cpp] frameId_ = " << frameId_ << std::endl;
  std::cout << "[DAM-augmented.cpp] parentId_ = " << Kp_.size() << std::endl;
  std::cout << "[DAM-augmented.cpp] ref_ = " << ref_ << std::endl;
  std::cout << "[DAM-augmented.cpp] cost_ref_ = " << cost_ref_ << std::endl;
  std::cout << "[DAM-augmented.cpp] active_contact_ = " << active_contact_ << std::endl;
  std::cout << "[DAM-augmented.cpp] nc_ = " << nc_ << std::endl;
  std::cout << "[DAM-augmented.cpp] jMf_ = " << jMf_ << std::endl;
  std::cout << "[DAM-augmented.cpp] with_armature_ = " << with_armature_ << std::endl;
  std::cout << "[DAM-augmented.cpp] armature_.size() = " << armature_.size() << std::endl;
  std::cout << "[DAM-augmented.cpp] with_force_cost_ = " << with_force_cost_ << std::endl;
  std::cout << "[DAM-augmented.cpp] with_force_rate_reg_cost_ = " << with_force_rate_reg_cost_ << std::endl;
  std::cout << "[DAM-augmented.cpp] force_des_.size() = " << force_des_.size() << std::endl;
  std::cout << "[DAM-augmented.cpp] force_weight_.size() = " << force_weight_.size() << std::endl;
  std::cout << "[DAM-augmented.cpp] force_rate_reg_weight_.size() = " << force_rate_reg_weight_.size() << std::endl;
  std::cout << "[DAM-augmented.cpp] with_gravity_torque_reg_ = " << with_gravity_torque_reg_ << std::endl;
  std::cout << "[DAM-augmented.cpp] g_lb_.size() = " << g_lb_.size() << std::endl;
  std::cout << "[DAM-augmented.cpp] g_ub_.size() = " << g_ub_.size() << std::endl;
  std::cout << "[DAM-augmented.cpp] with_force_constraint_ = " << with_force_constraint_ << std::endl;
  std::cout << "[DAM-augmented.cpp] END OF INIT" << std::endl;

}


DAMSoftContactAbstractAugmentedFwdDynamics::~DAMSoftContactAbstractAugmentedFwdDynamics() {}


void DAMSoftContactAbstractAugmentedFwdDynamics::calc(
    const boost::shared_ptr<DifferentialActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x,
    const Eigen::Ref<const VectorXs>& f) {
  calc(data, x, f, unone_);
}

void DAMSoftContactAbstractAugmentedFwdDynamics::calcDiff(
    const boost::shared_ptr<DifferentialActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x,
    const Eigen::Ref<const VectorXs>& f) {
  calcDiff(data, x, f, unone_);
}


// void DAMSoftContactAbstractAugmentedFwdDynamics::calc(
//                 const boost::shared_ptr<DifferentialActionDataAbstract>&, 
//                 const Eigen::Ref<const VectorXs>& x,
//                 const Eigen::Ref<const VectorXs>& f,
//                 const Eigen::Ref<const VectorXs>& u) {
//   std::cout << "[DAM-augmented.cpp] START OF CALC" << std::endl;
//   if (static_cast<std::size_t>(x.size()) != this->get_state()->get_nx()) {
//     throw_pretty("Invalid argument: "
//                  << "x has wrong dimension (it should be " + std::to_string(this->get_state()->get_nx()) + ")");
//   }
//   if (static_cast<std::size_t>(f.size()) != this->get_nc()) {
//     throw_pretty("Invalid argument: "
//                  << "f has wrong dimension (it should be 3)");
//   }
//   if (static_cast<std::size_t>(u.size()) != this->get_nu()) {
//     throw_pretty("Invalid argument: "
//                  << "u has wrong dimension (it should be " + std::to_string(this->get_nu()) + ")");
//   }
// }


// void DAMSoftContactAbstractAugmentedFwdDynamics::calc(
//                 const boost::shared_ptr<DifferentialActionDataAbstract>&, 
//                 const Eigen::Ref<const VectorXs>& x,
//                 const Eigen::Ref<const VectorXs>& f) {
//   if (static_cast<std::size_t>(x.size()) != this->get_state()->get_nx()) {
//     throw_pretty("Invalid argument: "
//                  << "x has wrong dimension (it should be " + std::to_string(this->get_state()->get_nx()) + ")");
//   }
//   if (static_cast<std::size_t>(f.size()) != this->get_nc()) {
//     throw_pretty("Invalid argument: "
//                  << "f has wrong dimension (it should be 3)");
//   }
//   std::cout << "[DAM-augmented.cpp] END OF CALC" << std::endl;
// }


// void DAMSoftContactAbstractAugmentedFwdDynamics::calcDiff(
//                 const boost::shared_ptr<DifferentialActionDataAbstract>&, 
//                 const Eigen::Ref<const VectorXs>& x,
//                 const Eigen::Ref<const VectorXs>& f,
//                 const Eigen::Ref<const VectorXs>& u) {
//   if (static_cast<std::size_t>(x.size()) != this->get_state()->get_nx()) {
//     throw_pretty("Invalid argument: "
//                  << "x has wrong dimension (it should be " + std::to_string(this->get_state()->get_nx()) + ")");
//   }
//   if (static_cast<std::size_t>(f.size()) != this->get_nc()) {
//     throw_pretty("Invalid argument: "
//                  << "f has wrong dimension (it should be 3)");
//   }
//   if (static_cast<std::size_t>(u.size()) != this->get_nu()) {
//     throw_pretty("Invalid argument: "
//                  << "u has wrong dimension (it should be " + std::to_string(this->get_nu()) + ")");
//   }
// }


// void DAMSoftContactAbstractAugmentedFwdDynamics::calcDiff(
//                 const boost::shared_ptr<DifferentialActionDataAbstract>&, 
//                 const Eigen::Ref<const VectorXs>& x,
//                 const Eigen::Ref<const VectorXs>& f) {
//   if (static_cast<std::size_t>(x.size()) != this->get_state()->get_nx()) {
//     throw_pretty("Invalid argument: "
//                  << "x has wrong dimension (it should be " + std::to_string(this->get_state()->get_nx()) + ")");
//   }
//   if (static_cast<std::size_t>(f.size()) != this->get_nc()) {
//     throw_pretty("Invalid argument: "
//                  << "f has wrong dimension (it should be 3)");
//   }
// }





boost::shared_ptr<DifferentialActionDataAbstractTpl<double> >
DAMSoftContactAbstractAugmentedFwdDynamics::createData() {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
}



std::size_t DAMSoftContactAbstractAugmentedFwdDynamics::get_ng() const {
  if (constraints_ != nullptr) {
    return constraints_->get_ng();
  } else {
    return DAMBase::get_ng();
  }
}

std::size_t DAMSoftContactAbstractAugmentedFwdDynamics::get_nh() const {
  if (constraints_ != nullptr) {
    return constraints_->get_nh();
  } else {
    return DAMBase::get_nh();
  }
}

const typename crocoddyl::MathBaseTpl<double>::VectorXs&
DAMSoftContactAbstractAugmentedFwdDynamics::get_g_lb() const {
  if (constraints_ != nullptr) {
    return constraints_->get_lb();
  } else {
    return g_lb_;
  }
}

const typename crocoddyl::MathBaseTpl<double>::VectorXs&
DAMSoftContactAbstractAugmentedFwdDynamics::get_g_ub() const {
  if (constraints_ != nullptr) {
    return constraints_->get_ub();
  } else {
    return g_lb_;
  }
}

void DAMSoftContactAbstractAugmentedFwdDynamics::print(
    std::ostream& os) const {
  os << "DifferentialActionModelFreeFwdDynamics {nx=" << state_->get_nx()
     << ", ndx=" << state_->get_ndx() << ", nu=" << nu_ << "}";
}

pinocchio::ModelTpl<double>&
DAMSoftContactAbstractAugmentedFwdDynamics::get_pinocchio() const {
  return pinocchio_;
}

const boost::shared_ptr<ActuationModelAbstract >&
DAMSoftContactAbstractAugmentedFwdDynamics::get_actuation() const {
  return actuation_;
}

const boost::shared_ptr<CostModelSum >&
DAMSoftContactAbstractAugmentedFwdDynamics::get_costs() const {
  return costs_;
}

const boost::shared_ptr<ConstraintModelManager >&
DAMSoftContactAbstractAugmentedFwdDynamics::get_constraints() const {
  return constraints_;
}



void DAMSoftContactAbstractAugmentedFwdDynamics::set_Kp(const VectorXs& inKp) {
  if (inKp.maxCoeff() < 0.) {
    throw_pretty("Invalid argument: "
                 << "Stiffness should be positive");
  }
  Kp_ = inKp;
  if(Kp_.maxCoeff() <= double(1e-9) && Kv_.maxCoeff() <= double(1e-9)){
    active_contact_ = false;
  } else {
    active_contact_ = true;
  }
}


void DAMSoftContactAbstractAugmentedFwdDynamics::set_Kv(const VectorXs& inKv) {
  if (inKv.maxCoeff() < 0.) {
    throw_pretty("Invalid argument: "
                 << "Damping should be positive");
  }
  Kv_ = inKv;
  if(Kp_.maxCoeff() <= double(1e-9) && Kv_.maxCoeff() <= double(1e-9)){
    active_contact_ = false;
  } else {
    active_contact_ = true;
  }
}


void DAMSoftContactAbstractAugmentedFwdDynamics::set_oPc(const Vector3s& inoPc) {
  if (inoPc.size() != 3) {
    throw_pretty("Invalid argument: "
                 << "Anchor point position should have size 3");
  }
  oPc_ = inoPc;
}



void DAMSoftContactAbstractAugmentedFwdDynamics::set_ref(const pinocchio::ReferenceFrame inRef) {
  ref_ = inRef;
}


void DAMSoftContactAbstractAugmentedFwdDynamics::set_cost_ref(const pinocchio::ReferenceFrame inRef) {
  cost_ref_ = inRef;
}


void DAMSoftContactAbstractAugmentedFwdDynamics::set_id(const pinocchio::FrameIndex inId) {
  frameId_ = inId;
}


const typename MathBaseTpl<double>::VectorXs& DAMSoftContactAbstractAugmentedFwdDynamics::get_Kp() const {
  return Kp_;
}


const typename MathBaseTpl<double>::VectorXs& DAMSoftContactAbstractAugmentedFwdDynamics::get_Kv() const {
  return Kv_;
}


const typename MathBaseTpl<double>::Vector3s& DAMSoftContactAbstractAugmentedFwdDynamics::get_oPc() const {
  return oPc_;
}


const pinocchio::ReferenceFrame& DAMSoftContactAbstractAugmentedFwdDynamics::get_ref() const {
  return ref_;
}


const pinocchio::ReferenceFrame& DAMSoftContactAbstractAugmentedFwdDynamics::get_cost_ref() const {
  return cost_ref_;
}



const pinocchio::FrameIndex& DAMSoftContactAbstractAugmentedFwdDynamics::get_id() const {
  return frameId_;
}


// armature

const typename MathBaseTpl<double>::VectorXs& DAMSoftContactAbstractAugmentedFwdDynamics::get_armature() const {
  return armature_;
}



bool DAMSoftContactAbstractAugmentedFwdDynamics::get_with_armature() const {
  return with_armature_;
}


void DAMSoftContactAbstractAugmentedFwdDynamics::set_with_armature(const bool inBool) {
  with_armature_ = inBool;
}


void DAMSoftContactAbstractAugmentedFwdDynamics::set_armature(const VectorXs& armature) {
  if (static_cast<std::size_t>(armature.size()) != this->get_state()->get_nv()) {
    throw_pretty("Invalid argument: "
                 << "The armature dimension is wrong (it should be " + std::to_string(this->get_state()->get_nv()) + ")");
  }
  armature_ = armature;
}


bool DAMSoftContactAbstractAugmentedFwdDynamics::get_active_contact() const {
  return active_contact_;
}


void DAMSoftContactAbstractAugmentedFwdDynamics::set_active_contact(const bool inActive) {
  active_contact_ = inActive;
}




bool DAMSoftContactAbstractAugmentedFwdDynamics::get_with_force_cost() const {
  return with_force_cost_;
}


void DAMSoftContactAbstractAugmentedFwdDynamics::set_with_force_cost(const bool inBool) {
  with_force_cost_ = inBool;
}


void DAMSoftContactAbstractAugmentedFwdDynamics::set_force_des(const VectorXs& inForceDes) {
  if (std::size_t(inForceDes.size()) != nc_) {
    throw_pretty("Invalid argument: "
                 << "Desired force should be have size " << nc_);
  }
  force_des_ = inForceDes;
}


void DAMSoftContactAbstractAugmentedFwdDynamics::set_force_weight(const VectorXs& inForceWeights) {
  if (inForceWeights.maxCoeff() < 0.) {
    throw_pretty("Invalid argument: "
                 << "Force cost weights should be positive");
  }
  force_weight_ = inForceWeights;
}


const typename MathBaseTpl<double>::VectorXs& DAMSoftContactAbstractAugmentedFwdDynamics::get_force_des() const {
  return force_des_;
}


const typename MathBaseTpl<double>::VectorXs& DAMSoftContactAbstractAugmentedFwdDynamics::get_force_weight() const {
  return force_weight_;
}


//Force rate reg cost 

void DAMSoftContactAbstractAugmentedFwdDynamics::set_with_force_rate_reg_cost(const bool inBool) {
  with_force_rate_reg_cost_ = inBool;
}


void DAMSoftContactAbstractAugmentedFwdDynamics::set_force_rate_reg_weight(const VectorXs& inForceRegWeights) {
  if (inForceRegWeights.maxCoeff() < 0.) {
    throw_pretty("Invalid argument: "
                 << "Force rate cost weights should be positive");
  }
  force_rate_reg_weight_ = inForceRegWeights;
}


bool DAMSoftContactAbstractAugmentedFwdDynamics::get_with_force_rate_reg_cost() const {
  return with_force_rate_reg_cost_;
}


const typename MathBaseTpl<double>::VectorXs& DAMSoftContactAbstractAugmentedFwdDynamics::get_force_rate_reg_weight() const {
  return force_rate_reg_weight_;
}





bool DAMSoftContactAbstractAugmentedFwdDynamics::get_with_gravity_torque_reg() const {
  return with_gravity_torque_reg_;
}


void DAMSoftContactAbstractAugmentedFwdDynamics::set_with_gravity_torque_reg(const bool inBool) {
  with_gravity_torque_reg_ = inBool;
}


double DAMSoftContactAbstractAugmentedFwdDynamics::get_tau_grav_weight() const {
  return tau_grav_weight_;
}


void DAMSoftContactAbstractAugmentedFwdDynamics::set_tau_grav_weight(const double inWeight) {
  if (inWeight < 0.) {
    throw_pretty("Invalid argument: "
                 << "Gravity torque regularization weight should be positive");
  }
  tau_grav_weight_ = inWeight;
}


}  // namespace softcontact
}  // namespace force_feedback_mpc

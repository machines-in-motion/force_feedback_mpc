///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2022, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_FORCE_FEEDBACK_MPC_CORE_DIFF_ACTION_BASE_HPP_
#define BINDINGS_PYTHON_FORCE_FEEDBACK_MPC_CORE_DIFF_ACTION_BASE_HPP_

#include "force_feedback_mpc/softcontact/dam-augmented.hpp"

namespace force_feedback_mpc {
namespace softcontact {

namespace bp = boost::python;

class DAMSoftContactAbstractAugmentedFwdDynamics_wrap : public DAMSoftContactAbstractAugmentedFwdDynamics,
                                                        public bp::wrapper<DAMSoftContactAbstractAugmentedFwdDynamics> {
 public:
  DAMSoftContactAbstractAugmentedFwdDynamics_wrap(boost::shared_ptr<crocoddyl::StateMultibody> state, 
                                                  boost::shared_ptr<crocoddyl::ActuationModelAbstract> actuation,
                                                  boost::shared_ptr<crocoddyl::CostModelSum> costs,
                                                  const pinocchio::FrameIndex frameId,
                                                  const Eigen::VectorXd& Kp,
                                                  const Eigen::VectorXd& Kv,
                                                  const Eigen::Vector3d& oPc, 
                                                  const std::size_t nc,
                                                  boost::shared_ptr<crocoddyl::ConstraintModelManager> constraints = nullptr)
      : DAMSoftContactAbstractAugmentedFwdDynamics(state, actuation, costs, frameId, Kp, Kv, oPc, nc, constraints), bp::wrapper<DAMSoftContactAbstractAugmentedFwdDynamics>() {}
  
  void calc(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
            const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& f,
            const Eigen::Ref<const Eigen::VectorXd>& u) {
    if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
      throw_pretty("Invalid argument: "
                   << "x has wrong dimension (it should be " +
                          std::to_string(state_->get_nx()) + ")");
    }
    if (static_cast<std::size_t>(f.size()) != this->get_nc()) {
      throw_pretty("Invalid argument: "
                   << "f has wrong dimension (it should be " +
                          std::to_string(this->get_nc()) + ")");
    }
    if (static_cast<std::size_t>(u.size()) != nu_) {
      throw_pretty("Invalid argument: "
                   << "u has wrong dimension (it should be " +
                          std::to_string(nu_) + ")");
    }
    if (std::isnan(u.lpNorm<Eigen::Infinity>())) {
      return bp::call<void>(this->get_override("calc").ptr(), data,
                            (Eigen::VectorXd)x, (Eigen::VectorXd)f);
    } else {
      return bp::call<void>(this->get_override("calc").ptr(), data,
                            (Eigen::VectorXd)x, (Eigen::VectorXd)f, (Eigen::VectorXd)u);
    }
  }

  void calcDiff(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                const Eigen::Ref<const Eigen::VectorXd>& x,
                const Eigen::Ref<const Eigen::VectorXd>& f,
                const Eigen::Ref<const Eigen::VectorXd>& u) {
    if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
      throw_pretty("Invalid argument: "
                   << "x has wrong dimension (it should be " +
                          std::to_string(state_->get_nx()) + ")");
    }
    if (static_cast<std::size_t>(f.size()) != this->get_nc()) {
      throw_pretty("Invalid argument: "
                   << "f has wrong dimension (it should be " +
                          std::to_string(this->get_nc()) + ")");
    }
    if (static_cast<std::size_t>(u.size()) != nu_) {
      throw_pretty("Invalid argument: "
                   << "u has wrong dimension (it should be " +
                          std::to_string(nu_) + ")");
    }
    if (std::isnan(u.lpNorm<Eigen::Infinity>())) {
      return bp::call<void>(this->get_override("calcDiff").ptr(), data,
                            (Eigen::VectorXd)x, (Eigen::VectorXd)f);
    } else {
      return bp::call<void>(this->get_override("calcDiff").ptr(), data,
                            (Eigen::VectorXd)x, (Eigen::VectorXd)f, (Eigen::VectorXd)u);
    }
  }

  boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract> createData() {
    crocoddyl::enableMultithreading() = false;
    if (boost::python::override createData = this->get_override("createData")) {
      return bp::call<boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract> >(createData.ptr());
    }
    return DAMSoftContactAbstractAugmentedFwdDynamics::createData();
  }

  boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract> default_createData() {
    return this->DAMSoftContactAbstractAugmentedFwdDynamics::createData();
  }

};

// BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(DifferentialActionModel_quasiStatic_wraps,
//                                        DAMSoftContactAbstractAugmentedFwdDynamics::quasiStatic_x, 2, 4)

}  // namespace softcontact
}  // namespace force_feedback_mpc

#endif  // BINDINGS_PYTHON_FORCE_FEEDBACK_MPCCORE_DIFF_ACTION_BASE_HPP_

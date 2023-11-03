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

// #include <boost/python.hpp>
// #include <boost/python/enum.hpp>
// #include <eigenpy/eigenpy.hpp>

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
                                                  const pinocchio::ReferenceFrame ref)
      : DAMSoftContactAbstractAugmentedFwdDynamics(state, actuation, costs, frameId, Kp, Kv, oPc, nc, ref), bp::wrapper<DAMSoftContactAbstractAugmentedFwdDynamics>() {}

  void calc(const boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract>& data, 
            const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& f,
            const Eigen::Ref<const Eigen::VectorXd>& u) {
  return bp::call<void>(this->get_override("calc").ptr(), data, (Eigen::VectorXd)x, (Eigen::VectorXd)f, (Eigen::VectorXd)u);
  }

  void calc(const boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract>& data, 
            const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& f) {
    return bp::call<void>(this->get_override("calc").ptr(), data, (Eigen::VectorXd)x, (Eigen::VectorXd)f);
  }


  void calcDiff(const boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract>& data,
                const Eigen::Ref<const Eigen::VectorXd>& x, 
                const Eigen::Ref<const Eigen::VectorXd>& f, 
                const Eigen::Ref<const Eigen::VectorXd>& u) {
    return bp::call<void>(this->get_override("calcDiff").ptr(), data, (Eigen::VectorXd)x, (Eigen::VectorXd)f, (Eigen::VectorXd)u);
  }

  void calcDiff(const boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract>& data,
                const Eigen::Ref<const Eigen::VectorXd>& x, 
                const Eigen::Ref<const Eigen::VectorXd>& f) {
    return bp::call<void>(this->get_override("calcDiff").ptr(), data, (Eigen::VectorXd)x, (Eigen::VectorXd)f);
  }

  // boost::shared_ptr<DADSoftContactAbstractAugmentedFwdDynamics> createData() {
  //   enableMultithreading() = false;
  //   if (boost::python::override createData = this->get_override("createData")) {
  //     return bp::call<boost::shared_ptr<crocoddyl::DifferentialActionModelAbstract> >(createData.ptr());
  //   }
  //   return DAMSoftContactAbstractAugmentedFwdDynamics::createData();
  // }

//   boost::shared_ptr<DADSoftContactAbstractAugmentedFwdDynamics> default_createData() {
//     return this->DAMSoftContactAbstractAugmentedFwdDynamics::createData();
//   }

};

// BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(DifferentialActionModel_quasiStatic_wraps,
//                                        DAMSoftContactAbstractAugmentedFwdDynamics::quasiStatic_x, 2, 4)

}  // namespace softcontact
}  // namespace force_feedback_mpc

#endif  // BINDINGS_PYTHON_FORCE_FEEDBACK_MPCCORE_DIFF_ACTION_BASE_HPP_

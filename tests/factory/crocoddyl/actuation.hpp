///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef FORCE_FEEDBACK_MPC_ACTUATION_FACTORY_HPP_
#define FORCE_FEEDBACK_MPC_ACTUATION_FACTORY_HPP_

#include <crocoddyl/core/actuation-base.hpp>
#include <crocoddyl/core/numdiff/actuation.hpp>
#include "state.hpp"

namespace force_feedback_mpc {
namespace unittest {

struct ActuationModelTypes {
  enum Type {
    ActuationModelFull,
    ActuationModelFloatingBase,
    ActuationModelSquashingFull,
    NbActuationModelTypes
  };
  static std::vector<Type> init_all() {
    std::vector<Type> v;
    v.clear();
    for (int i = 0; i < NbActuationModelTypes; ++i) {
      v.push_back((Type)i);
    }
    return v;
  }
  static const std::vector<Type> all;
};

std::ostream& operator<<(std::ostream& os, ActuationModelTypes::Type type);

class ActuationModelFactory {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit ActuationModelFactory();
  ~ActuationModelFactory();

  std::shared_ptr<crocoddyl::ActuationModelAbstract> create(
      ActuationModelTypes::Type actuation_type,
      StateModelTypes::Type state_type) const;
};

/**
 * @brief Update the actuation model needed for numerical differentiation.
 * We use the address of the object to avoid a copy from the
 * "boost::bind".
 *
 * @param model[in]  Pinocchio model
 * @param data[out]  Pinocchio data
 * @param x[in]      State vector
 * @param u[in]      Control vector
 */
void updateActuation(
    const std::shared_ptr<crocoddyl::ActuationModelAbstract>& model,
    const std::shared_ptr<crocoddyl::ActuationDataAbstract>& data,
    const Eigen::VectorXd& x, const Eigen::VectorXd& u);

}  // namespace unittest
}  // namespace force_feedback_mpc

#endif  // FORCE_FEEDBACK_MPC_ACTUATION_FACTORY_HPP_

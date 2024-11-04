///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2024, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef FORCE_FEEDBACK_MPC_FRICTIONCONE_AUGMENTED_HPP_
#define FORCE_FEEDBACK_MPC_FRICTIONCONE_AUGMENTED_HPP_

#include <crocoddyl/core/residual-base.hpp>
#include <crocoddyl/core/utils/exception.hpp>
#include <crocoddyl/multibody/data/contacts.hpp>
#include <crocoddyl/multibody/fwd.hpp>
#include <crocoddyl/multibody/states/multibody.hpp>
#include <crocoddyl/multibody/data/impulses.hpp>

namespace force_feedback_mpc {
namespace frictioncone {

struct ResidualDataFrictionConeAugmented
    : public crocoddyl::ResidualDataAbstractTpl<double> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef crocoddyl::MathBaseTpl<double> MathBase;
  typedef crocoddyl::ResidualDataAbstractTpl<double> Base;
  typedef crocoddyl::DataCollectorAbstractTpl<double> DataCollectorAbstract;
  typedef crocoddyl::StateMultibodyTpl<double> StateMultibody;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <class Model>
  ResidualDataFrictionConeAugmented(Model* const model,
                           DataCollectorAbstract* const data)
      : Base(model, data) {
    dcone_df.resize(3);
    dcone_df.setZero();
    residual = 0;
}
  VectorXs dcone_df;
  double residual;  
};



/**
 * @brief Contact friction cone residual
 *
 * Nonlinear friction cone 3d
 */
class ResidualModelFrictionConeAugmented
    : public crocoddyl::ResidualModelAbstractTpl<double> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef crocoddyl::MathBaseTpl<double> MathBase;
  typedef crocoddyl::ResidualModelAbstractTpl<double> Base;
  typedef ResidualDataFrictionConeAugmented Data;
  typedef crocoddyl::StateMultibodyTpl<double> StateMultibody;
  typedef crocoddyl::DataCollectorAbstractTpl<double> DataCollectorAbstract;
  typedef crocoddyl::ResidualDataAbstractTpl<double> ResidualDataAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename MathBase::MatrixX3s MatrixX3s;

  /**
   * @brief Initialize the contact friction cone residual model
   *
   * Note that for the inverse-dynamic cases, the control vector contains the
   * generalized accelerations, torques, and all the contact forces.
   *
   * @param[in] state   State of the multibody system
   * @param[in] id      Reference frame id
   * @param[in] fref    Reference friction cone
   * @param[in] nu      Dimension of the control vector
   * @param[in] fwddyn  Indicates that we have a forward dynamics problem (true)
   * or inverse dynamics (false)
   */
  ResidualModelFrictionConeAugmented(boost::shared_ptr<StateMultibody> state,
                            const pinocchio::FrameIndex id,
                            const double coef,
                            const std::size_t nu);

  /**
   * @brief Initialize the contact friction cone residual model
   *
   * The default `nu` value is obtained from `StateAbstractTpl::get_nv()`. Note
   * that this constructor can be used for forward-dynamics cases only.
   *
   * @param[in] state  State of the multibody system
   * @param[in] id     Reference frame id
   * @param[in] fref   Reference friction cone
   */
  ResidualModelFrictionConeAugmented(boost::shared_ptr<StateMultibody> state,
                                      const pinocchio::FrameIndex id,
                                      const double coef);
  virtual ~ResidualModelFrictionConeAugmented();

  /**
   * @brief Compute the contact friction cone residual
   *
   * @param[in] data  Contact friction cone residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const boost::shared_ptr<ResidualDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Compute the residual vector for nodes that depends only on the state
   *
   * It updates the residual vector based on the state only (i.e., it ignores
   * the contact forces). This function is used in the terminal nodes of an
   * optimal control problem.
   *
   * @param[in] data  Residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  virtual void calc(const boost::shared_ptr<ResidualDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Compute the Jacobians of the contact friction cone residual
   *
   * @param[in] data  Contact friction cone residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<ResidualDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Compute the Jacobian of the residual functions with respect to the
   * state only
   *
   * It updates the Jacobian of the residual function based on the state only
   * (i.e., it ignores the contact forces). This function is used in the
   * terminal nodes of an optimal control problem.
   *
   * @param[in] data  Residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<ResidualDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Create the contact friction cone residual data
   */
  virtual boost::shared_ptr<ResidualDataAbstract> createData(
      DataCollectorAbstract* const data);

  /**
   * @brief Return the reference frame id
   */
  pinocchio::FrameIndex get_id() const;

  /**
   * @brief Print relevant information of the contact-friction-cone residual
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const;

 protected:
  using Base::nu_;
  using Base::state_;

 private:
  pinocchio::FrameIndex id_;  //!< Reference frame id
  double coef_;
};


}  // namespace frictioncone
}  // namespace force_feedback_mpc

#endif  // FORCE_FEEDBACK_MPC_FRICTIONCONE_AUGMENTED_HPP_

#ifndef FORCE_FEEDBACK_MPC_SOFTCONTACT_RESIDUAL_GRAVITY_TORQUE_CONTACT_HPP_
#define FORCE_FEEDBACK_MPC_SOFTCONTACT_RESIDUAL_GRAVITY_TORQUE_CONTACT_HPP_

#include <crocoddyl/core/residual-base.hpp>
#include <crocoddyl/multibody/data/multibody.hpp>
#include <crocoddyl/multibody/fwd.hpp>
#include <crocoddyl/multibody/states/multibody.hpp>

namespace force_feedback_mpc {
namespace softcontact {

using namespace crocoddyl;

/**
 * @brief Gravity Torque with Contact residual
 *
 * As described in `ResidualModelAbstractTpl()`, the residual value and its
 * Jacobians are calculated by `calc` and `calcDiff`, respectively.
 *
 * \sa `ResidualModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class ResidualModelGravityTorqueContactTpl
    : public ResidualModelAbstractTpl<_Scalar> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualModelAbstractTpl<Scalar> Base;
  typedef ResidualDataGravityTorqueContactTpl<Scalar> Data;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ResidualDataAbstractTpl<Scalar> ResidualDataAbstract;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::Vector3s Vector3s;
  typedef typename MathBase::VectorXs VectorXs;

  /**
   * @brief Initialize the Gravity Torque with Contact residual model
   *
   * @param[in] state    State of the multibody system
   * @param[in] ref      Reference frame in which contact forces are represented
   * @param[in] frame_id Frame if of the contact point
   * @param[in] nu       Dimension of the control vector
   */
  ResidualModelGravityTorqueContactTpl(std::shared_ptr<StateMultibody> state,
                                       const pinocchio::ReferenceFrame ref,
                                       const pinocchio::FrameIndex frame_id,
                                       const std::size_t nu);

  virtual ~ResidualModelGravityTorqueContactTpl();

  /**
   * @brief Compute the Gravity Torque with Contact residual
   *
   * @param[in] data  Gravity Torque with Contact residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const std::shared_ptr<ResidualDataAbstract> &data,
                    const Eigen::Ref<const VectorXs> &x,
                    const Eigen::Ref<const VectorXs> &u);

  /**
   * @brief Compute the derivatives of the Gravity Torque with Contact residual
   *
   * @param[in] data  Gravity Torque with Contact residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const std::shared_ptr<ResidualDataAbstract> &data,
                        const Eigen::Ref<const VectorXs> &x,
                        const Eigen::Ref<const VectorXs> &u);

  virtual std::shared_ptr<ResidualDataAbstract>
  createData(DataCollectorAbstract *const data);

  /**
   * @brief Return the reference frame of the contact
   */
  pinocchio::ReferenceFrame get_reference() const;

  /**
   * @brief Modify reference frame for the contact
   */
  void set_reference(const pinocchio::ReferenceFrame ref);

  /**
   * @brief Return frame id of the contact
   */
  pinocchio::FrameIndex get_frame_id() const;

  /**
   * @brief Modify frame id of the contact
   */
  void set_frame_id(const pinocchio::FrameIndex frame_id);

  /**
   * @brief Return if the contact is active
   */
  bool get_active_contact() const;

  /**
   * @brief Modify if the contact is active
   */
  void set_active_contact(const bool active_contact);

  /**
   * @brief Print relevant information of the com-position residual
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream &os) const;

protected:
  using Base::nu_;
  using Base::state_;
  using Base::u_dependent_;
  using Base::v_dependent_;

  bool active_contact_;
  pinocchio::FrameIndex frame_id_;
  pinocchio::ReferenceFrame ref_;

private:
  Vector3s cref_; //!< Reference Gravity Torque with Contact
};

template <typename _Scalar>
struct ResidualDataGravityTorqueContactTpl
    : public ResidualDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualDataAbstractTpl<Scalar> Base;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef pinocchio::DataTpl<Scalar> PinocchioData;

  template <template <typename Scalar> class Model>
  ResidualDataGravityTorqueContactTpl(Model<Scalar> *const model,
                                      DataCollectorAbstract *const data)
      : Base(model, data), lJ(6, model->get_state()->get_nv()) {
    // Check that proper shared data has been passed
    DataCollectorActMultibodyTpl<Scalar> *d =
        dynamic_cast<DataCollectorActMultibodyTpl<Scalar> *>(shared);
    if (d == NULL) {
      throw_pretty("Invalid argument: the shared data should be derived from "
                   "DataCollectorActMultibodyTpl");
    }
    // Avoids data casting at runtime
    StateMultibody *sm =
        static_cast<StateMultibody *>(model->get_state().get());
    pinocchio = PinocchioData(*(sm->get_pinocchio().get()));
    actuation = d->actuation;

    lJ.setZero();
  }

  using Base::r;
  using Base::Ru;
  using Base::Rx;
  using Base::shared;

  pinocchio::DataTpl<Scalar> *pinocchio; //!< Pinocchio data
  std::shared_ptr<ActuationDataAbstractTpl<Scalar>>
      actuation; //!< Actuation data
  MatrixXs lJ;   //!< Contact frame LOCAL Jacobian matrix
};

} // namespace softcontact
} // namespace force_feedback_mpc

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/residuals/com-position.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(
    crocoddyl::ResidualGravityTorqueContactTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::ResidualDataGravityTorqueContactTpl)

#endif // FORCE_FEEDBACK_MPC_SOFTCONTACT_RESIDUAL_GRAVITY_TORQUE_CONTACT_HPP_
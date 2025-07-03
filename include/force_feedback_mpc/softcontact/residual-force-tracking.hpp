#ifndef FORCE_FEEDBACK_MPC_SOFTCONTACT_RESIDUAL_FORCE_BASE_HPP_
#define FORCE_FEEDBACK_MPC_SOFTCONTACT_RESIDUAL_FORCE_BASE_HPP_

#include <crocoddyl/core/residual-base.hpp>
#include <crocoddyl/multibody/data/multibody.hpp>
#include <crocoddyl/multibody/fwd.hpp>
#include <crocoddyl/multibody/states/multibody.hpp>

#include "force_feedback_mpc/softcontact/residual-force-base.hpp"

namespace force_feedback_mpc {
namespace softcontact {

using namespace crocoddyl;

/**
 * @brief Gravity Torque with Contact residual
 *
 * As described in `ResidualModelForceBaseTpl()`, the residual value and its
 * Jacobians are calculated by `calc` and `calcDiff`, respectively.
 *
 * \sa `ResidualModelForceBaseTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class ResidualModelForceTrackingTpl : public ResidualModelForceBaseTpl<_Scalar> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualModelForceBaseTpl<Scalar> Base;
  typedef ResidualDataForceBaseTpl<Scalar> Data;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ResidualDataAbstractTpl<Scalar> ResidualDataAbstract;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::Vector3s Vector3s;
  typedef typename MathBase::VectorXs VectorXs;

  /**
   * @brief Initialize the residual model
   *
   * @param[in] state        State of the system
   * @param[in] nr           Dimension of residual vector
   * @param[in] nu           Dimension of control vector
   * @param[in] nf           Dimension of force vector
   * @param[in] q_dependent  Define if the residual function depends on q
   * (default true)
   * @param[in] v_dependent  Define if the residual function depends on v
   * (default true)
   * @param[in] u_dependent  Define if the residual function depends on u
   * (default true)
   * @param[in] f_dependent  Define if the residual function depends on f
   * (default true)
   */
  ResidualModelForceBaseTpl(
      std::shared_ptr<StateAbstract> state, const std::size_t nr,
      const std::size_t nu, const std::size_t nc, const bool q_dependent = true,
      const bool v_dependent = true,
      const bool u_dependent = true const bool f_dependent = true);

  /**
   * @copybrief ResidualModelForceBaseTpl()
   *
   * The default `nu` value is obtained from `StateAbstractTpl::get_nv()`.
   *
   * @param[in] state        State of the system
   * @param[in] nr           Dimension of residual vector
   * @param[in] nf           Dimension of force vector
   * @param[in] q_dependent  Define if the residual function depends on q
   * (default true)
   * @param[in] v_dependent  Define if the residual function depends on v
   * (default true)
   * @param[in] u_dependent  Define if the residual function depends on u
   * (default true)
   * @param[in] f_dependent  Define if the residual function depends on f
   * (default true)
   */
  ResidualModelAbstractTpl(std::shared_ptr<StateAbstract> state,
                           const std::size_t nr, const std::size_t nf,
                           const bool q_dependent = true,
                           const bool v_dependent = true,
                           const bool u_dependent = true,
                           const bool f_dependent = true);

  virtual ~ResidualModelForceBaseTpl();

  virtual std::shared_ptr<ResidualDataAbstract>
  createData(DataCollectorAbstract *const data);

  /**
   * @brief Return the dimension of the force vector
   */
  std::size_t get_nf() const;

  /**
   * @brief Return true if the residual function depends on f
   */
  bool get_f_dependent() const;

  /**
   * @brief Print relevant information of the com-position residual
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream &os) const;

protected:
  using Base::unone_;
  
  std::size_t nf_; //!< Force dimension

  bool f_dependent_; //!< Label that indicates if the residual function depends
                     //!< on f
};

template <typename _Scalar>
struct ResidualDataForceBaseTpl : public ResidualDataForceBaseTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef ResidualDataForceBaseTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;

  template <template <typename Scalar> class Model>
  ResidualDataForceBaseTpl(Model<Scalar> *const model,
                           DataCollectorAbstract *const data)
      : Base(model, data) {}
  virtual ~ResidualDataForceBaseTpl() {}
};

} // namespace softcontact
} // namespace force_feedback_mpc

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/residuals/com-position.hxx"

#endif // FORCE_FEEDBACK_MPC_SOFTCONTACT_RESIDUAL_FORCE_BASE_HPP_
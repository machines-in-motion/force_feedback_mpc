///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, CTU, INRIA,
// University of Oxford Copyright note valid unless otherwise stated in
// individual files. All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef SOBEC_SOFTCONTACT3D_AUGMENTED_FWDDYN_HPP_
#define SOBEC_SOFTCONTACT3D_AUGMENTED_FWDDYN_HPP_

#include <stdexcept>

#include "crocoddyl/core/actuation-base.hpp"
#include "crocoddyl/core/costs/cost-sum.hpp"
#include "crocoddyl/core/diff-action-base.hpp"
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/actions/free-fwddyn.hpp"

#include "sobec/fwd.hpp"
#include "dam-augmented.hpp"

namespace sobec {

/**
 * @brief Differential action model for visco-elastic contact forward dynamics in multibody
 * systems (augmented dynamics including contact force as a state)
 *
 * Derived class with 3D force (linear)
 * Maths here : https://www.overleaf.com/read/xdpymjfhqqhn
 *
 * \sa `DAMSoftContact3DAugmentedFwdDynamicsTpl`, `calc()`, `calcDiff()`,
 * `createData()`
 */
template <typename _Scalar>
class DAMSoftContact3DAugmentedFwdDynamicsTpl
    : public sobec::DAMSoftContactAbstractAugmentedFwdDynamicsTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef sobec::DAMSoftContactAbstractAugmentedFwdDynamicsTpl<Scalar> Base;
  typedef DADSoftContact3DAugmentedFwdDynamicsTpl<Scalar> Data;
  typedef crocoddyl::MathBaseTpl<Scalar> MathBase;
  typedef crocoddyl::CostModelSumTpl<Scalar> CostModelSum;
  typedef crocoddyl::StateMultibodyTpl<Scalar> StateMultibody;
  typedef crocoddyl::ActuationModelAbstractTpl<Scalar> ActuationModelAbstract;
  typedef crocoddyl::DifferentialActionDataAbstractTpl<Scalar> DifferentialActionDataAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::Vector3s Vector3s;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename MathBase::Matrix3s Matrix3s;

  /**
   * @brief Initialize the soft contact forward-dynamics action model
   *
   * It describes the dynamics evolution of a multibody system under
   * visco-elastic contact (linear spring damper force)
   *
   * @param[in] state            State of the multibody system
   * @param[in] actuation        Actuation model
   * @param[in] costs            Stack of cost functions
   * @param[in] frameId          Pinocchio frame id of the frame in contact
   * @param[in] Kp               Soft contact model stiffness
   * @param[in] Kv               Soft contact model damping
   * @param[in] oPc              Anchor point of the contact model in WORLD coordinates
   * @param[in] ref              Pinocchio reference frame in which the contact force is to be expressed
   * 
   */
  DAMSoftContact3DAugmentedFwdDynamicsTpl(
      boost::shared_ptr<StateMultibody> state,
      boost::shared_ptr<ActuationModelAbstract> actuation,
      boost::shared_ptr<CostModelSum> costs,
      const pinocchio::FrameIndex frameId,
      const VectorXs& Kp, 
      const VectorXs& Kv,
      const Vector3s& oPc,
      const pinocchio::ReferenceFrame ref = pinocchio::LOCAL);
  virtual ~DAMSoftContact3DAugmentedFwdDynamicsTpl();

  /**
   * @brief Compute the system acceleration, and cost value
   *
   * It computes the system acceleration using the soft contact forward-dynamics.
   *
   * @param[in] data  Soft contact forward-dynamics data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] f     Force point \f$\mathbf{f}\in\mathbb{R}^{nc}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const boost::shared_ptr<DifferentialActionDataAbstract>& data, 
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& f,
                    const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Compute the system acceleration, and cost value
   *
   * It computes the system acceleration using the soft contact forward-dynamics.
   *
   * @param[in] data  Soft contact forward-dynamics data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] f     Force point \f$\mathbf{f}\in\mathbb{R}^{nc}\f$
   */
  virtual void calc(const boost::shared_ptr<DifferentialActionDataAbstract>& data, 
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& f);

  /**
   * @brief Compute the derivatives of the contact dynamics, and cost function
   *
   * @param[in] data  Contact forward-dynamics data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] f     Force point \f$\mathbf{f}\in\mathbb{R}^{nc}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(
      const boost::shared_ptr<DifferentialActionDataAbstract>& data,
      const Eigen::Ref<const VectorXs>& x, 
      const Eigen::Ref<const VectorXs>& f, 
      const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Compute the derivatives of the contact dynamics, and cost function
   *
   * @param[in] data  Contact forward-dynamics data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] f     Force point \f$\mathbf{f}\in\mathbb{R}^{nc}\f$
   */
  virtual void calcDiff(
      const boost::shared_ptr<DifferentialActionDataAbstract>& data,
      const Eigen::Ref<const VectorXs>& x,
      const Eigen::Ref<const VectorXs>& f);
  
    /**
   * @brief Create the soft contact forward-dynamics data
   *
   * @return soft contact forward-dynamics data
   */
  virtual boost::shared_ptr<DifferentialActionDataAbstract> createData();
  
  protected:
    using Base::Kp_;
    using Base::Kv_;
    using Base::oPc_;
    using Base::frameId_;
    using Base::parentId_;
    using Base::ref_;
    using Base::with_force_cost_;
    using Base::active_contact_;
    using Base::nc_;
    using Base::jMf_;
    using Base::with_armature_;
    using Base::with_gravity_torque_reg_;
    using Base::armature_;
    using Base::force_des_;                    
    using Base::force_weight_;                   
    using Base::tau_grav_weight_;
    using Base::with_force_rate_reg_cost_;
    using Base::force_rate_reg_weight_;
    using Base::cost_ref_;
    
};

template <typename _Scalar>
struct DADSoftContact3DAugmentedFwdDynamicsTpl : public sobec::DADSoftContactAbstractAugmentedFwdDynamicsTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef sobec::DADSoftContactAbstractAugmentedFwdDynamicsTpl<Scalar> Base;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::Vector3s Vector3s;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename MathBase::Matrix3s Matrix3s;

  template <template <typename Scalar> class Model>
  explicit DADSoftContact3DAugmentedFwdDynamicsTpl(Model<Scalar>* const model)
      : Base(model) {}

  using Base::pinocchio;
  using Base::multibody;
  using Base::costs;
  using Base::Minv;
  using Base::u_drift;
  using Base::tmp_xstatic;

  using Base::dtau_dx;
  // Contact frame rotation and Jacobians
  using Base::oRf;
  using Base::lJ;
  using Base::oJ;
  // Partials of ABA w.r.t. state and control
  using Base::aba_dq;
  using Base::aba_dv;
  using Base::aba_dx;
  using Base::aba_dtau;
  using Base::aba_df;
  // Frame linear velocity and acceleration in LOCAL and LOCAL_WORLD_ALIGNED frames
  using Base::lv;
  using Base::la;
  using Base::ov;
  using Base::oa;
  // Partials of frame spatial velocity w.r.t. joint pos, vel, acc
  using Base::lv_dq;
  using Base::lv_dv;
  using Base::lv_dx;
  // Partials of frame spatial acceleration w.r.t. joint pos, vel, acc
  using Base::v_dv;
  using Base::a_dq;
  using Base::a_dv;
  using Base::a_da;
  // Partial of frame spatial acc w.r.t. state 
  using Base::da_dx;
  using Base::da_du;
  using Base::da_df;
  // Time-derivative of contact force
  using Base::fout;
  using Base::fout_copy;
  // Spatial wrench due to contact force
  using Base::pinForce;
  using Base::fext;       //!< External spatial forces in body coordinates (joint level)
  using Base::fext_copy;  //!< External spatial forces in body coordinates (joint level)
  // Partial derivatives of next force w.r.t. augmented state
  using Base::dfdt_dx;
  using Base::dfdt_du;
  using Base::dfdt_df;
  using Base::dfdt_dx_copy;
  using Base::dfdt_du_copy;
  using Base::dfdt_df_copy;
  // Partials of cost w.r.t. force 
  using Base::Lf;
  using Base::Lff;
  // Force residual for hard coded tracking cost
  using Base::f_residual;
  using Base::f_residual_x;
  // Gravity reg residual
  using Base::tau_grav_residual;
  using Base::tau_grav_residual_x;
  using Base::tau_grav_residual_u;
  using Base::tau_grav_residual_f;

  using Base::cost;
  using Base::Fu;
  using Base::Fx;
  using Base::Lu;
  using Base::Luu;
  using Base::Lx;
  using Base::Lxu;
  using Base::Lxx;
  using Base::r;
  using Base::xout;
};

}  // namespace sobec

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include <sobec/crocomplements/softcontact/dam3d-augmented.hxx>

#endif  // SOBEC_SOFTCONTACT3D_AUGMENTED_FWDDYN_HPP_

///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, CTU, INRIA,
// University of Oxford Copyright note valid unless otherwise stated in
// individual files. All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef SOBEC_SOFTCONTACT1D_FWDDYN_HPP_
#define SOBEC_SOFTCONTACT1D_FWDDYN_HPP_

#include <stdexcept>

#include "crocoddyl/core/actuation-base.hpp"
#include "crocoddyl/core/costs/cost-sum.hpp"
#include "crocoddyl/core/diff-action-base.hpp"
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/actions/free-fwddyn.hpp"

#include "sobec/fwd.hpp"

namespace sobec {


/**
 * @brief Differential action model for visco-elastic contact forward dynamics in multibody
 * systems.
 *
 * Maths here : https://www.overleaf.com/read/xdpymjfhqqhn
 *
 * \sa `DifferentialActionModelFreeFwdDynamicsTpl`, `calc()`, `calcDiff()`,
 * `createData()`
 */
template <typename _Scalar>
class DifferentialActionModelSoftContact1DFwdDynamicsTpl
    : public crocoddyl::DifferentialActionModelFreeFwdDynamicsTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef crocoddyl::DifferentialActionModelFreeFwdDynamicsTpl<Scalar> Base;
  typedef DifferentialActionDataSoftContact1DFwdDynamicsTpl<Scalar> Data;
  typedef crocoddyl::MathBaseTpl<Scalar> MathBase;
  typedef crocoddyl::CostModelSumTpl<Scalar> CostModelSum;
  typedef crocoddyl::StateMultibodyTpl<Scalar> StateMultibody;
  typedef crocoddyl::ActuationModelAbstractTpl<Scalar> ActuationModelAbstract;
  typedef crocoddyl::DifferentialActionDataAbstractTpl<Scalar> DifferentialActionDataAbstract;
  typedef crocoddyl::DifferentialActionDataFreeFwdDynamicsTpl<Scalar> DifferentialActionDataFreeFwdDynamics;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::Vector3s Vector3s;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename sobec::Vector3MaskType Vector3MaskType; 
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
  DifferentialActionModelSoftContact1DFwdDynamicsTpl(
      boost::shared_ptr<StateMultibody> state,
      boost::shared_ptr<ActuationModelAbstract> actuation,
      boost::shared_ptr<CostModelSum> costs,
      const pinocchio::FrameIndex frameId,
      const double Kp, 
      const double Kv,
      const Vector3s& oPc,
      const pinocchio::ReferenceFrame ref = pinocchio::LOCAL,
      const Vector3MaskType& type = Vector3MaskType::z);
  virtual ~DifferentialActionModelSoftContact1DFwdDynamicsTpl();

  /**
   * @brief Compute the system acceleration, and cost value
   *
   * It computes the system acceleration using the free forward-dynamics.
   *
   * @param[in] data  Free forward-dynamics data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const boost::shared_ptr<DifferentialActionDataAbstract>& data, 
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Compute the system acceleration, and cost value
   *
   * It computes the system acceleration using the free forward-dynamics.
   *
   * @param[in] data  Free forward-dynamics data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  virtual void calc(const boost::shared_ptr<DifferentialActionDataAbstract>& data, 
                    const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Compute the derivatives of the contact dynamics, and cost function
   *
   * @param[in] data  Contact forward-dynamics data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(
      const boost::shared_ptr<DifferentialActionDataAbstract>& data,
      const Eigen::Ref<const VectorXs>& x, 
      const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Compute the derivatives of the contact dynamics, and cost function
   *
   * @param[in] data  Contact forward-dynamics data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  virtual void calcDiff(
      const boost::shared_ptr<DifferentialActionDataAbstract>& data,
      const Eigen::Ref<const VectorXs>& x);
  
    /**
   * @brief Create the soft contact forward-dynamics data
   *
   * @return soft contact forward-dynamics data
   */
  virtual boost::shared_ptr<DifferentialActionDataAbstract> createData();
  
  void set_force_cost(const Scalar force_des, const Scalar force_weight);

  void set_force_des(const Scalar inForceDes);

  void set_force_weight(const Scalar inForceWeight);

  void set_Kp(const Scalar inKp);

  void set_Kv(const Scalar inKv);

  void set_oPc(const Vector3s& oPc);

  void set_ref(const pinocchio::ReferenceFrame inRef);
  
  void set_id(const pinocchio::FrameIndex inId);

  const Scalar get_Kp() const;

  const Scalar get_Kv() const;

  const Vector3s& get_oPc() const;

  const Scalar get_force_des() const;

  const Scalar get_force_weight() const;

  const pinocchio::ReferenceFrame& get_ref() const;
  
  const pinocchio::FrameIndex& get_id() const;

  std::size_t get_nc() {return nc_;};

  const Vector3MaskType& get_type() {return type_;};

  void set_type(const Vector3MaskType& inType);


  protected:
    Scalar Kp_;                             //!< Contact model stiffness
    Scalar Kv_;                             //!< Contact model damping
    Vector3s oPc_;                          //!< Contact model anchor point
    pinocchio::FrameIndex frameId_;         //!< Frame id of the contact
    pinocchio::FrameIndex parentId_;        //!< Parent id of the contact
    pinocchio::ReferenceFrame ref_;         //!< Pinocchio reference frame
    bool with_force_cost_;                  //!< Force cost ?
    bool active_contact_;                   //!< Active contact ?
    std::size_t nc_;                        //!< Contact model dimension = 1
    Scalar force_des_;                    //!< Desired force 1D
    Scalar force_weight_;                   //!< Force cost weight
    pinocchio::SE3Tpl<Scalar> jMf_;         //!< Placement of contact frame w.r.t. parent frame
    bool with_armature_;                    //!< Indicate if we have defined an armature
    VectorXs armature_;                     //!< Armature vector
    Vector3MaskType type_;
};

template <typename _Scalar>
struct DifferentialActionDataSoftContact1DFwdDynamicsTpl : public crocoddyl::DifferentialActionDataFreeFwdDynamicsTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef crocoddyl::DifferentialActionDataFreeFwdDynamicsTpl<Scalar> Base;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::Vector3s Vector3s;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename MathBase::Matrix3s Matrix3s;

  template <template <typename Scalar> class Model>
  explicit DifferentialActionDataSoftContact1DFwdDynamicsTpl(Model<Scalar>* const model)
      : Base(model),
        lJ(6, model->get_state()->get_nv()),
        oJ(6, model->get_state()->get_nv()),
        lv_partial_dq(6, model->get_state()->get_nv()),
        lv_partial_dv(6, model->get_state()->get_nv()),
        aba_dq(model->get_state()->get_nv(), model->get_state()->get_nv()),
        aba_dv(model->get_state()->get_nv(), model->get_state()->get_nv()),
        aba_dtau(model->get_state()->get_nv(), model->get_state()->get_nv()),
        df_dx(model->get_nc(), model->get_state()->get_ndx()),
        df_dx_copy(model->get_nc(), model->get_state()->get_ndx()),
        fWORLD(3),
        fLOCAL(3),
        f3d(3),
        f3d_copy(3),
        df3d_dx(3, model->get_state()->get_ndx()),
        df3d_dx_copy(3, model->get_state()->get_ndx()),
        pinForce(pinocchio::ForceTpl<Scalar>::Zero()),
        fext(model->get_pinocchio().njoints, pinocchio::ForceTpl<Scalar>::Zero()),
        fext_copy(model->get_pinocchio().njoints, pinocchio::ForceTpl<Scalar>::Zero()) {
    costs->shareMemory(this);
    Minv.setZero();
    u_drift.setZero();
    dtau_dx.setZero();
    tmp_xstatic.setZero();
    oRf.setZero();
    lv.setZero();
    lJ.setZero();
    oJ.setZero();
    lv_partial_dv.setZero();
    lv_partial_dq.setZero();
    aba_dq.setZero();
    aba_dv.setZero();
    aba_dtau.setZero();
    df_dx.setZero();
    df_dx_copy.setZero();
    fWORLD.setZero();
    fLOCAL.setZero();
    f3d.setZero();
    f3d_copy.setZero();
    df3d_dx.setZero();
    df3d_dx_copy.setZero();
  }

  using Base::pinocchio;
  using Base::multibody;
  using Base::costs;
  using Base::Minv;
  using Base::u_drift;
  using Base::dtau_dx;
  using Base::tmp_xstatic;

  Matrix3s oRf;
  Vector3s lv;
  MatrixXs lJ;
  MatrixXs oJ;
  MatrixXs lv_partial_dq;
  MatrixXs lv_partial_dv;
  MatrixXs aba_dq;
  MatrixXs aba_dv;
  MatrixXs aba_dtau;
  // force cost & derivatives
  Scalar f_residual;
  Scalar f;
  Scalar f_copy;
  MatrixXs df_dx; 
  MatrixXs df_dx_copy;

  Vector3s fLOCAL;
  Vector3s fWORLD;
  Vector3s f3d;
  Vector3s f3d_copy;
  MatrixXs df3d_dx; 
  MatrixXs df3d_dx_copy; 

  pinocchio::ForceTpl<Scalar> pinForce;
  pinocchio::container::aligned_vector<pinocchio::ForceTpl<Scalar> > fext;  //!< External spatial forces in body coordinates (joint level)
  pinocchio::container::aligned_vector<pinocchio::ForceTpl<Scalar> > fext_copy;  //!< External spatial forces in body coordinates (joint level)

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
#include <sobec/crocomplements/softcontact/dam1d.hxx>

#endif  // SOBEC_SOFTCONTACT1D_FWDDYN_HPP_

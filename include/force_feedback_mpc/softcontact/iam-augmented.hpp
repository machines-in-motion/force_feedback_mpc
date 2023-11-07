///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef FORCE_FEEDBACK_MPC_IAM3D_AUGMENTED_HPP_
#define FORCE_FEEDBACK_MPC_IAM3D_AUGMENTED_HPP_

#include <crocoddyl/core/action-base.hpp>
#include <crocoddyl/core/diff-action-base.hpp>
#include <crocoddyl/core/fwd.hpp>
#include <crocoddyl/multibody/states/multibody.hpp>
#include <pinocchio/multibody/model.hpp>

#include "state.hpp"
#include "dam3d-augmented.hpp"

namespace force_feedback_mpc {
namespace softcontact {

struct IADSoftContactAugmented : public crocoddyl::ActionDataAbstractTpl<double> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef crocoddyl::MathBaseTpl<double> MathBase;
  typedef crocoddyl::ActionDataAbstractTpl<double> Base;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef pinocchio::DataTpl<double> PinocchioData;
  typedef crocoddyl::DifferentialActionDataAbstractTpl<double> DifferentialActionDataAbstract;

  template <class Model>
  explicit IADSoftContactAugmented(Model* const model)
      : Base(model), tau_tmp(model->get_nu()) {
    tau_tmp.setZero();
    differential = model->get_differential()->createData();
    const std::size_t& ndy = model->get_state()->get_ndx();
    dy = VectorXs::Zero(ndy);
  }
  virtual ~IADSoftContactAugmented() {}

  boost::shared_ptr<DifferentialActionDataAbstract> differential;
  VectorXs dy;

  using Base::cost;
  using Base::r;
  VectorXs tau_tmp;
  // use refs to "alias" base class member names
  VectorXs& ynext = Base::xnext;
  MatrixXs& Fy = Base::Fx;
//   MatrixXs& Fw = Base::Fu;
  VectorXs& Ly = Base::Lx;
//   VectorXs& Lw = Base::Lu;
  MatrixXs& Lyy = Base::Lxx;
  MatrixXs& Lyu = Base::Lxu;
//   MatrixXs& Lww = Base::Luu;
};

class IAMSoftContactAugmented : public crocoddyl::ActionModelAbstractTpl<double> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef crocoddyl::MathBaseTpl<double> MathBase;
  typedef crocoddyl::ActionModelAbstractTpl<double> Base;
  typedef IADSoftContactAugmented Data;
  typedef crocoddyl::ActionDataAbstractTpl<double> ActionDataAbstract;
  typedef crocoddyl::DifferentialActionModelAbstractTpl<double> DifferentialActionModelAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef crocoddyl::StateMultibodyTpl<double> StateMultibody;
  typedef pinocchio::ModelTpl<double> PinocchioModel;

  IAMSoftContactAugmented(
      boost::shared_ptr<DAMSoftContactAbstractAugmentedFwdDynamics> model,
      const double& time_step = double(1e-3),
      const bool& with_cost_residual = true);
  virtual ~IAMSoftContactAugmented();

  virtual void calc(const boost::shared_ptr<ActionDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& y,
                    const Eigen::Ref<const VectorXs>& w);

  virtual void calc(const boost::shared_ptr<ActionDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& y);

  virtual void calcDiff(const boost::shared_ptr<ActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& y,
                        const Eigen::Ref<const VectorXs>& w);

  virtual void calcDiff(const boost::shared_ptr<ActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& y);

  virtual boost::shared_ptr<ActionDataAbstract> createData();

  virtual bool checkData(const boost::shared_ptr<ActionDataAbstract>& data);

//   virtual void quasiStatic(const boost::shared_ptr<ActionDataAbstract>& data,
//                            Eigen::Ref<VectorXs> u,
//                            const Eigen::Ref<const VectorXs>& x,
//                            const std::size_t& maxiter = 100,
//                            const double& tol = double(1e-9));

  const boost::shared_ptr<DAMSoftContactAbstractAugmentedFwdDynamics>& get_differential()
      const;
  const double& get_dt() const;

  const std::size_t& get_nc() const { return nc_; };
  const std::size_t& get_ny() const { return ny_; };

  void set_dt(const double& dt);
  void set_differential(boost::shared_ptr<DAMSoftContactAbstractAugmentedFwdDynamics> model);

 protected:
  using Base::has_control_limits_;  //!< Indicates whether any of the control
                                    //!< limits are active
  using Base::nr_;                  //!< Dimension of the cost residual
  using Base::nu_;                  //!< Control dimension
  using Base::u_lb_;                //!< Lower control limits
  using Base::u_ub_;                //!< Upper control limits
  using Base::unone_;               //!< Neutral state
  std::size_t nc_;                  //!< Contact model dimension
  std::size_t ny_;                  //!< Augmented state dimension : nq+nv+ntau
  using Base::state_;               //!< Model of the state

 private:
  boost::shared_ptr<DAMSoftContactAbstractAugmentedFwdDynamics> differential_;
  double time_step_;
  double time_step2_;
  bool with_cost_residual_;
  boost::shared_ptr<PinocchioModel> pin_model_;  //!< for reg cost
  bool is_terminal_;  //!< is it a terminal model or not ? (deactivate cost on w
                      //!< if true)
};

}  // namespace softcontact
}  // namespace force_feedback_mpc

#endif  // FORCE_FEEDBACK_MPC_IAM3D_AUGMENTED_HPP_

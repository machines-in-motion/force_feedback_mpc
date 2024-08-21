///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef FORCE_FEEDBACK_MPC_LPF_HPP_
#define FORCE_FEEDBACK_MPC_LPF_HPP_

#include <crocoddyl/core/action-base.hpp>
#include <crocoddyl/core/activations/quadratic-barrier.hpp>
#include <crocoddyl/core/diff-action-base.hpp>
#include <crocoddyl/core/fwd.hpp>
#include <crocoddyl/multibody/states/multibody.hpp>
#include <pinocchio/multibody/model.hpp>

#include "state.hpp"

namespace force_feedback_mpc {
namespace lpf {


class IntegratedActionDataLPF : public crocoddyl::ActionDataAbstract {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef crocoddyl::MathBaseTpl<double> MathBase;
  typedef crocoddyl::ActionDataAbstractTpl<double> Base;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef pinocchio::DataTpl<double> PinocchioData;
  typedef crocoddyl::DifferentialActionDataAbstractTpl<double> DifferentialActionDataAbstract;
  typedef crocoddyl::ActivationDataQuadraticBarrierTpl<double> ActivationDataQuadraticBarrier; 

  template <class Model>
  explicit IntegratedActionDataLPF(Model* const model)
      : Base(model), tau_tmp(model->get_nu()) {
    tau_tmp.setZero();
    differential = model->get_differential()->createData();
    const std::size_t& ndy = model->get_state()->get_ndx();
    dy = VectorXs::Zero(ndy);
    // for wlim cost
    activation = boost::static_pointer_cast<ActivationDataQuadraticBarrier>(
        model->activation_model_tauLim_->createData());
  }
  virtual ~IntegratedActionDataLPF() {}

  boost::shared_ptr<DifferentialActionDataAbstract> differential;
  VectorXs dy;

  // PinocchioData pinocchio;                                       // for reg
  // cost
  boost::shared_ptr<ActivationDataQuadraticBarrier> activation;  // for lim cost

  using Base::cost;
  using Base::r;
  VectorXs tau_tmp;
  // use refs to "alias" base class member names
  VectorXs& ynext = Base::xnext;
  MatrixXs& Fy = Base::Fx;
  MatrixXs& Fw = Base::Fu;
  VectorXs& Ly = Base::Lx;
  VectorXs& Lw = Base::Lu;
  MatrixXs& Lyy = Base::Lxx;
  MatrixXs& Lyw = Base::Lxu;
  MatrixXs& Lww = Base::Luu;
  MatrixXs& Gy = Base::Gx;
};


class IntegratedActionModelLPF : public crocoddyl::ActionModelAbstractTpl<double> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef crocoddyl::MathBaseTpl<double> MathBase;
  typedef crocoddyl::ActionModelAbstractTpl<double> Base;
  typedef IntegratedActionDataLPF Data;
  typedef crocoddyl::ActionDataAbstractTpl<double> ActionDataAbstract;
  typedef crocoddyl::DifferentialActionModelAbstractTpl<double> DifferentialActionModelAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef crocoddyl::StateMultibodyTpl<double> StateMultibody;
  typedef pinocchio::ModelTpl<double> PinocchioModel;
  typedef crocoddyl::ActivationModelQuadraticBarrierTpl<double> ActivationModelQuadraticBarrier;
  typedef crocoddyl::ActivationBoundsTpl<double> ActivationBounds;

  IntegratedActionModelLPF(
      boost::shared_ptr<DifferentialActionModelAbstract> model,
      std::vector<std::string> lpf_joint_names = {},
      const double& time_step = double(1e-3),
      const bool& with_cost_residual = true, const double& fc = 0,
      const bool& tau_plus_integration = true, const int& filter = 0);
  virtual ~IntegratedActionModelLPF();

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

  virtual void quasiStatic(const boost::shared_ptr<ActionDataAbstract>& data,
                           Eigen::Ref<VectorXs> u,
                           const Eigen::Ref<const VectorXs>& x,
                           const std::size_t maxiter = 100,
                           const double tol = double(1e-9));

  const boost::shared_ptr<DifferentialActionModelAbstract>& get_differential()
      const;
  const double& get_dt() const;
  const double& get_fc() const;
  const double& get_alpha() const { return alpha_; };

  const std::size_t& get_nw() const { return nw_; };
  const std::size_t& get_ntau() const { return ntau_; };
  const std::size_t& get_ny() const { return ny_; };

  const std::vector<std::string>& get_lpf_joint_names() const {
    return lpf_joint_names_;
  };

  const std::vector<int>& get_lpf_torque_ids() const {
    return lpf_torque_ids_;
  };
  const std::vector<int>& get_non_lpf_torque_ids() const {
    return non_lpf_torque_ids_;
  };

  void set_dt(const double& dt);
  void set_fc(const double& fc);
  void set_alpha(const double& alpha);
  void set_differential(
      boost::shared_ptr<DifferentialActionModelAbstract> model);

  void set_with_lpf_torque_constraint(const bool inBool) {with_lpf_torque_constraint_ = inBool; };
  const bool& get_with_lpf_torque_constraint() const { return with_lpf_torque_constraint_; };

  void set_lpf_torque_lb(const VectorXs& inVec);
  const VectorXs& get_lpf_torque_lb() const { return lpf_torque_lb_; };

  void set_lpf_torque_ub(const VectorXs& inVec);
  const VectorXs& get_lpf_torque_ub() const { return lpf_torque_ub_; };
  
  // hard-coded costs
  void set_control_reg_cost(const double& cost_weight_w_reg,
                            const VectorXs& cost_ref_w_reg);
  void set_control_lim_cost(const double& cost_weight_w_lim);

  void compute_alpha(const double& fc);

 protected:
  using Base::has_control_limits_;  //!< Indicates whether any of the control
                                    //!< limits are active
  using Base::nr_;                  //!< Dimension of the cost residual
  using Base::ng_;                  //!< Number of inequality constraints
  using Base::g_lb_;                //!< Lower bound of the inequality constraints
  using Base::g_ub_;                //!< Upper bound of the inequality constraints
  using Base::nu_;                  //!< Control dimension
  using Base::u_lb_;                //!< Lower control limits
  using Base::u_ub_;                //!< Upper control limits
  using Base::unone_;               //!< Neutral state
  std::size_t nw_;                  //!< Input torque dimension (unfiltered)
  std::size_t ntau_;   //!< Filtered torque dimension ("lpf" dimension)
  std::size_t ny_;     //!< Augmented state dimension : nq+nv+ntau
  using Base::state_;  //!< Model of the state

 public:
  boost::shared_ptr<ActivationModelQuadraticBarrier>
      activation_model_tauLim_;  //!< for lim cost

 private:
  boost::shared_ptr<DifferentialActionModelAbstract> differential_;
  double time_step_;
  double time_step2_;
  double alpha_;
  bool with_cost_residual_;
  double fc_;
  double tauReg_weight_;  //!< Cost weight for unfiltered torque regularization
  VectorXs tauReg_reference_;  //!< Cost reference for unfiltered torque
                               //!< regularization
  VectorXs tauReg_residual_,
      tauLim_residual_;  //!< Residuals for LPF torques reg and lim
  double tauLim_weight_;       //!< Cost weight for unfiltered torque limits
  bool tau_plus_integration_;  //!< Use tau+ = LPF(tau,w) in acceleration
                               //!< computation, or tau
  int filter_;                 //!< Type of LPF used>
  boost::shared_ptr<PinocchioModel> pin_model_;  //!< for reg cost
  bool is_terminal_;  //!< is it a terminal model or not ? (deactivate cost on w
                      //!< if true)
  std::vector<std::string>
      lpf_joint_names_;  //!< Vector of joint names that are low-pass filtered
  std::vector<int>
      lpf_joint_ids_;  //!< Vector of joint ids that are low-pass filtered
  std::vector<int>
      lpf_torque_ids_;  //!< Vector of torque ids that are low-passs filtered

  //   std::vector<std::string> non_lpf_joint_names_;  //!< Vector of joint
  //   names that are NOT low-pass filtered
  std::vector<int> non_lpf_joint_ids_;   //!< Vector of joint ids that are NOT
                                         //!< low-pass filtered
  std::vector<int> non_lpf_torque_ids_;  //!< Vector of torque ids that are NOT
                                         //!< low-passs filtered
  bool with_lpf_torque_constraint_; // Add box constraint on the LPF torques dimensions
  VectorXs lpf_torque_lb_;
  VectorXs lpf_torque_ub_;
  VectorXs g_lb_new_;
  VectorXs g_ub_new_;
};


}  // namespace lpf
}  // namespace force_feedback_mpc

#endif  // FORCE_FEEDBACK_MPC_LPF_HPP_

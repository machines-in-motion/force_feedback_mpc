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

#include "force_feedback_mpc/frictioncone/residual-friction-cone-augmented.hpp"

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
      : Base(model) {
    differential = model->get_differential()->createData();
    const std::size_t& ndy = model->get_state()->get_ndx();
    dy = VectorXs::Zero(ndy);
    Gy.resize(model->get_ng(), model->get_state()->get_ndx());
    Gy.setZero();
    Gu.resize(model->get_ng(), model->get_nu());
    Gu.setZero();
    // if(model->get_with_friction_cone_constraint() && model->get_nc() == 3){
    friction_cone_residual.resize(model->get_nf());
    // std::cout << "model get_nf() = " << model->get_nf() << std::endl;
    // dcone_df.resize(model->get_nf(), 3);
    dcone_df.resize(3);
    dcone_df.setZero();
    // }
  }
  virtual ~IADSoftContactAugmented() {}

  std::shared_ptr<DifferentialActionDataAbstract> differential;
  VectorXs dy;

  using Base::cost;
  using Base::Fu;
  using Base::Fx;
  using Base::Lu;
  using Base::Luu;
  using Base::Lx;
  using Base::Lxu;
  using Base::Lxx;
  using Base::r;
  using Base::xnext;

  // use refs to "alias" base class member names
  VectorXs& ynext = Base::xnext;
  MatrixXs& Fy = Base::Fx;
  VectorXs& Ly = Base::Lx;
  MatrixXs& Lyy = Base::Lxx;
  MatrixXs& Lyu = Base::Lxu;
  MatrixXs& Gy = Base::Gx;
  MatrixXs& Gu = Base::Gu;
  // friction cone specific
  VectorXs friction_cone_residual;
  VectorXs dcone_df;

  // Re-size the constraint size according to friction constraints size
  template <class Model>
  void resizeIneqConstraint(Model* const model) {
    // std::cout << "BEFORE friction_cone_residual = " << friction_cone_residual.size() << std::endl;
    // std::cout << "BEFORE dcone_df = " << dcone_df.size() << std::endl;
    // std::cout << "BEFORE g = " << g.size() << std::endl;
    // std::cout << "BEFORE G = " << Gy.size() << std::endl;
    // std::cout << "BEFORE G = " << Gu.size() << std::endl;
    VectorXs g_lb_old_ =  model->get_g_lb();
    VectorXs g_ub_old_ =  model->get_g_ub();
    // std::cout << "BEFORE g_lb = " << g_lb_old_ << std::endl;
    // std::cout << "BEFORE g_ub = " << g_ub_old_ << std::endl;
    const std::size_t ndx = model->get_differential()->get_state()->get_ndx();
    const std::size_t nu = model->get_differential()->get_nu();
    const std::size_t nf = model->get_nf();
    const std::size_t ng = model->get_ng() + nf;
    g.conservativeResize(ng);
    Gy.conservativeResize(ng, ndx);
    Gu.conservativeResize(ng, nu);
    // std::cout << "Size after = " << Gy.size() << std::endl;
    // new bounds (add friction cone bounds)
    VectorXs g_lb_new_ =  model->get_g_lb();
    VectorXs g_ub_new_ =  model->get_g_ub();
    // std::cout << "AFTER g_lb = " << g_lb_new_ << std::endl;
    // std::cout << "AFTER g_ub = " << g_ub_new_ << std::endl;
    // this->set_g_lb(-std::numeric_limits<double>::infinity()*VectorXs::Ones(this->get_ng()));
    // this->set_g_ub(std::numeric_limits<double>::infinity()*VectorXs::Ones(this->get_ng()));
    g_lb_new_.tail(nf) = 0.*VectorXs::Ones(nf);
    g_ub_new_.tail(nf) = std::numeric_limits<double>::infinity()*VectorXs::Ones(nf);
    // std::cout << "AFTER g_lb_new_ = " << g_lb_new_ << std::endl;
    // std::cout << "AFTER g_ub_new_ = " << g_ub_new_ << std::endl;
  // temp variable used to update the force bounds


    // new (&g) Eigen::Map<VectorXs>(data->g.data(), ng);
    // new (&Gx) Eigen::Map<MatrixXs>(data->Gx.data(), ng, ndx);
    // new (&Gu) Eigen::Map<MatrixXs>(data->Gu.data(), ng, nu);
  };

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
  typedef force_feedback_mpc::frictioncone::ResidualModelFrictionConeAugmented ResidualModelFrictionConeAugmented;
  typedef force_feedback_mpc::frictioncone::ResidualDataFrictionConeAugmented ResidualDataFrictionConeAugmented;

  IAMSoftContactAugmented(
      std::shared_ptr<DAMSoftContactAbstractAugmentedFwdDynamics> model,
      const double& time_step = double(1e-3),
      const bool& with_cost_residual = true,
      const std::vector<std::shared_ptr<ResidualModelFrictionConeAugmented>> friction_constraints = {});
  virtual ~IAMSoftContactAugmented();

  virtual void calc(const std::shared_ptr<ActionDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& y,
                    const Eigen::Ref<const VectorXs>& w);

  virtual void calc(const std::shared_ptr<ActionDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& y);

  virtual void calcDiff(const std::shared_ptr<ActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& y,
                        const Eigen::Ref<const VectorXs>& w);

  virtual void calcDiff(const std::shared_ptr<ActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& y);

  virtual std::shared_ptr<crocoddyl::ActionModelBase> cloneAsDouble() const {
    return std::make_shared<IAMSoftContactAugmented>(*this);
  }
  virtual std::shared_ptr<crocoddyl::ActionModelBase> cloneAsFloat() const {
    return std::make_shared<IAMSoftContactAugmented>(*this);
  }

  virtual std::shared_ptr<ActionDataAbstract> createData();

  virtual bool checkData(const std::shared_ptr<ActionDataAbstract>& data);

//   virtual void quasiStatic(const std::shared_ptr<ActionDataAbstract>& data,
//                            Eigen::Ref<VectorXs> u,
//                            const Eigen::Ref<const VectorXs>& x,
//                            const std::size_t& maxiter = 100,
//                            const double& tol = double(1e-9));

  const std::shared_ptr<DAMSoftContactAbstractAugmentedFwdDynamics>& get_differential()
      const;
  const double& get_dt() const;

  const std::size_t& get_nc() const { return nc_; };
  const std::size_t& get_ny() const { return ny_; };

  void set_dt(const double& dt);
  void set_differential(std::shared_ptr<DAMSoftContactAbstractAugmentedFwdDynamics> model);

  void set_with_force_constraint(const bool inBool) {with_force_constraint_ = inBool; };
  bool get_with_force_constraint() const { return with_force_constraint_; };

  void set_force_lb(const VectorXs& inVec);
  const VectorXs& get_force_lb() const { return force_lb_; };

  void set_force_ub(const VectorXs& inVec);
  const VectorXs& get_force_ub() const { return force_ub_; };

  // void set_with_friction_cone_constraint(const bool inBool) {with_friction_cone_constraint_ = inBool; };
  bool get_with_friction_cone_constraint() const { return with_friction_cone_constraint_; };

  void set_friction_cone_constraints(const std::vector<std::shared_ptr<ResidualModelFrictionConeAugmented>>& frictionConstraints);
  const std::vector<std::shared_ptr<ResidualModelFrictionConeAugmented>>& get_friction_cone_constraints() const { return friction_constraints_; };

  /**
   * @brief Modify the lower bound of the inequality constraints
   */
  void set_g_lb(const VectorXs& g_lb);

  /**
   * @brief Modify the upper bound of the inequality constraints
   */
  void set_g_ub(const VectorXs& g_ub);

  std::size_t get_nf() const { return nf_; };

  // Re-size the constraint size according to friction constraints size
  void resizeIneqConstraint(std::shared_ptr<crocoddyl::ActionDataAbstract>& data);

 protected:
  using Base::has_control_limits_;  //!< Indicates whether any of the control
                                    //!< limits are active
  using Base::nr_;                  //!< Dimension of the cost residual
  using Base::nu_;                  //!< Control dimension
  using Base::state_;               //!< Model of the state
  using Base::u_lb_;                //!< Lower control limits
  using Base::u_ub_;                //!< Upper control limits
  using Base::ng_;                  //!< Number of inequality constraints
  using Base::nh_;                  //!< Number of inequality constraints
  using Base::g_lb_;                //!< Lower bound of the inequality constraints
  using Base::g_ub_;                //!< Upper bound of the inequality constraints
  using Base::unone_;               //!< Neutral state
  std::size_t nc_;                  //!< Contact model dimension
  std::size_t ny_;                  //!< Augmented state dimension : nq+nv+ntau

  std::size_t nf_;                  //!< Number of friction cone constraints

 private:
  std::shared_ptr<DAMSoftContactAbstractAugmentedFwdDynamics> differential_;
  double time_step_;
  double time_step2_;
  bool with_cost_residual_;
  std::shared_ptr<PinocchioModel> pin_model_;  //!< for reg cost
  bool is_terminal_;  //!< is it a terminal model or not ? (deactivate cost on w
                      //!< if true)
  bool with_force_constraint_; // Add box constraint on the contact force
  VectorXs force_lb_;
  VectorXs force_ub_;
  VectorXs g_lb_new_;
  VectorXs g_ub_new_;
  bool with_friction_cone_constraint_; // Add friction cone constraint on the force
  double friction_coef_;              // Friction coefficient

  std::vector<std::shared_ptr<ResidualModelFrictionConeAugmented>> friction_constraints_;
  std::vector<std::shared_ptr<ResidualDataFrictionConeAugmented>> friction_datas_;
};

}  // namespace softcontact
}  // namespace force_feedback_mpc

#endif  // FORCE_FEEDBACK_MPC_IAM3D_AUGMENTED_HPP_

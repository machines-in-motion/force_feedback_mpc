///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, CTU, INRIA,
// University of Oxford Copyright note valid unless otherwise stated in
// individual files. All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef FORCE_FEEDBACK_MPC_SOFTCONTACT_AUGMENTED_FWDDYN_HPP_
#define FORCE_FEEDBACK_MPC_SOFTCONTACT_AUGMENTED_FWDDYN_HPP_

#include <stdexcept>

#include <crocoddyl/core/actuation-base.hpp>
#include <crocoddyl/core/constraints/constraint-manager.hpp>
#include <crocoddyl/core/costs/cost-sum.hpp>
#include <crocoddyl/core/diff-action-base.hpp>
#include <crocoddyl/core/utils/exception.hpp>
#include <crocoddyl/multibody/data/multibody.hpp>
#include <crocoddyl/multibody/fwd.hpp>
#include <crocoddyl/multibody/states/multibody.hpp>


namespace force_feedback_mpc {
namespace softcontact {


struct DADSoftContactAbstractAugmentedFwdDynamics : 
    public crocoddyl::DifferentialActionDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef crocoddyl::MathBaseTpl<double> MathBase;
  typedef crocoddyl::DifferentialActionDataAbstractTpl<double> DADBase;
  typedef crocoddyl::JointDataAbstractTpl<double> JointDataAbstract;
  typedef crocoddyl::DataCollectorJointActMultibodyTpl<double>
      DataCollectorJointActMultibody;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::Vector3s Vector3s;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename MathBase::Matrix3s Matrix3s;

  template <class DAModel>
  explicit DADSoftContactAbstractAugmentedFwdDynamics(DAModel* const model)
      // : DADBase(static_cast<crocoddyl::DifferentialActionModelAbstractTpl<double>*>(model)),
      : DADBase(model), // this complains ( no conversion error )
        pinocchio(pinocchio::DataTpl<Scalar>(model->get_pinocchio())),
        multibody(
            &pinocchio, model->get_actuation()->createData(),
            boost::make_shared<JointDataAbstract>(
                model->get_state(), model->get_actuation(), model->get_nu())),
        costs(model->get_costs()->createData(&multibody)),
        Minv(model->get_state()->get_nv(), model->get_state()->get_nv()),
        u_drift(model->get_state()->get_nv()),
        dtau_dx(model->get_state()->get_nv(), model->get_state()->get_ndx()),
        tmp_xstatic(model->get_state()->get_nx()),
        // custom
        lJ(6, model->get_state()->get_nv()),
        oJ(6, model->get_state()->get_nv()),
        aba_dq(model->get_state()->get_nv(), model->get_state()->get_nv()),
        aba_dv(model->get_state()->get_nv(), model->get_state()->get_nv()),
        aba_dx(model->get_state()->get_nv(), model->get_state()->get_ndx()),
        aba_dtau(model->get_state()->get_nv(), model->get_state()->get_nv()),
        aba_df(model->get_state()->get_nv(), model->get_nc()),
        lv_dq(6, model->get_state()->get_nv()),
        lv_dv(6, model->get_state()->get_nv()),
        lv_dx(6, model->get_state()->get_ndx()),
        v_dv(6, model->get_state()->get_nv()),
        a_dq(6, model->get_state()->get_nv()),
        a_dv(6, model->get_state()->get_nv()),
        a_da(6, model->get_state()->get_nv()),
        da_dx(6,model->get_state()->get_ndx()),
        da_du(6,model->get_nu()),
        da_df(6,model->get_nc()),
        fout(model->get_nc()),
        fout_copy(model->get_nc()),
        pinForce(pinocchio::ForceTpl<double>::Zero()),
        fext(model->get_pinocchio().njoints, pinocchio::ForceTpl<double>::Zero()),
        fext_copy(model->get_pinocchio().njoints, pinocchio::ForceTpl<double>::Zero()),
        dfdt_dx(model->get_nc(), model->get_state()->get_ndx()),
        dfdt_du(model->get_nc(), model->get_nu()),
        dfdt_df(model->get_nc(), model->get_nc()),
        dfdt_dx_copy(model->get_nc(), model->get_state()->get_ndx()),
        dfdt_du_copy(model->get_nc(), model->get_nu()),
        dfdt_df_copy(model->get_nc(), model->get_nc()),
        Lf(model->get_nc()),
        Lff(model->get_nc(), model->get_nc()),
        f_residual(model->get_nc()),
        f_residual_x(model->get_nc(), model->get_state()->get_ndx()),
        f_residual_f(model->get_nc(), model->get_nc()),
        tau_grav_residual(model->get_state()->get_nv()),
        tau_grav_residual_x(model->get_state()->get_nv(), model->get_state()->get_ndx()),
        tau_grav_residual_u(model->get_state()->get_nv(), model->get_actuation()->get_nu()),
        tau_grav_residual_f(model->get_state()->get_nv(), model->get_nc()),
        residual(model->get_nresidual()) {
          // costs residuals (nr) + grav reg (nv) + force (nc) + force rate reg (nc)
    multibody.joint->dtau_du.diagonal().setOnes();
    costs->shareMemory(this);
    if (model->get_constraints() != nullptr) {
      constraints = model->get_constraints()->createData(&multibody);
      constraints->shareMemory(this);
    }
    Minv.setZero();
    u_drift.setZero();
    dtau_dx.setZero();
    tmp_xstatic.setZero();
    // Custom
    oRf.setZero();
    lJ.setZero();
    oJ.setZero();
    aba_dq.setZero();
    aba_dv.setZero();
    aba_dx.setZero();
    aba_dtau.setZero();
    aba_df.setZero();
    lv.setZero();
    la.setZero();
    ov.setZero();
    oa.setZero();
    lv_dq.setZero();
    lv_dv.setZero();
    lv_dx.setZero();
    v_dv.setZero();
    a_dq.setZero();
    a_dv.setZero();
    a_da.setZero();
    da_dx.setZero();
    da_du.setZero();
    da_df.setZero();
    fout.setZero();
    fout_copy.setZero();
    dfdt_dx.setZero();
    dfdt_du.setZero();
    dfdt_df.setZero();
    dfdt_dx_copy.setZero();
    dfdt_du_copy.setZero();
    dfdt_df_copy.setZero();
    Lx.setZero();
    Lu.setZero();
    Lf.setZero();
    Lff.setZero();
    f_residual.setZero();
    f_residual_x.setZero();
    f_residual_f.setZero();
    tau_grav_residual.setZero();
    tau_grav_residual_x.setZero();
    tau_grav_residual_u.setZero();
    tau_grav_residual_f.setZero();
    residual.setZero();
  }
  
  pinocchio::DataTpl<double> pinocchio;
  DataCollectorJointActMultibody multibody;
  boost::shared_ptr<crocoddyl::CostDataSumTpl<double> > costs;
  boost::shared_ptr<crocoddyl::ConstraintDataManagerTpl<double> > constraints;
  MatrixXs Minv;
  VectorXs u_drift;
  MatrixXs dtau_dx;
  VectorXs tmp_xstatic;

  // Contact frame rotation and Jacobians
  Matrix3s oRf;       //!< Contact frame rotation matrix 
  MatrixXs lJ;        //!< Contact frame LOCAL Jacobian matrix
  MatrixXs oJ;        //!< Contact frame WORLD Jacobian matrix
  // Partials of ABA w.r.t. state and control
  MatrixXs aba_dq;    //!< Partial derivative of ABA w.r.t. joint positions
  MatrixXs aba_dv;    //!< Partial derivative of ABA w.r.t. joint velocities
  MatrixXs aba_dx;    //!< Partial derivative of ABA w.r.t. joint state (positions, velocities)
  MatrixXs aba_dtau;  //!< Partial derivative of ABA w.r.t. joint torques 
  MatrixXs aba_df;    //!< Partial derivative of ABA w.r.t. contact force
  // Frame linear velocity and acceleration in LOCAL and LOCAL_WORLD_ALIGNED frames
  Vector3s lv;        //!< Linear spatial velocity of the contact frame in LOCAL
  Vector3s la;        //!< Linear spatial acceleration of the contact frame in LOCAL
  Vector3s ov;        //!< Linear spatial velocity of the contact frame in LOCAL_WORLD_ALIGNED
  Vector3s oa;        //!< Linear spatial acceleration of the contact frame in LOCAL_WORLD_ALIGNED
  // Partials of frame spatial velocity w.r.t. joint pos, vel, acc
  MatrixXs lv_dq;     //!< Partial derivative of spatial velocity of the contact frame w.r.t. joint positions in LOCAL
  MatrixXs lv_dv;     //!< Partial derivative of spatial velocity of the contact frame w.r.t. joint velocities in LOCAL
  MatrixXs lv_dx;     //!< Partial derivative of spatial velocity of the contact frame w.r.t. joint state (positions, velocities) in LOCAL
  // Partials of frame spatial acceleration w.r.t. joint pos, vel, acc
  MatrixXs v_dv;      //!< Partial derivative of spatial velocity of the contact frame w.r.t. joint velocity in LOCAL (not used)
  MatrixXs a_dq;      //!< Partial derivative of spatial acceleration of the contact frame w.r.t. joint positions in LOCAL
  MatrixXs a_dv;      //!< Partial derivative of spatial acceleration of the contact frame w.r.t. joint velocities in LOCAL
  MatrixXs a_da;      //!< Partial derivative of spatial acceleration of the contact frame w.r.t. joint accelerations in LOCAL
  // Partial of frame spatial acc w.r.t. state 
  MatrixXs da_dx;     //!< Partial derivative of spatial acceleration of the contact frame w.r.t. joint state (positions, velocities) in LOCAL
  MatrixXs da_du;     //!< Partial derivative of spatial acceleration of the contact frame w.r.t. joint torques in LOCAL
  MatrixXs da_df;     //!< Partial derivative of spatial acceleration of the contact frame w.r.t. contact force in LOCAL
  // Current force and next force
  VectorXs fout;      //!< Contact force time-derivative (output of the soft contact forward dynamics)
  VectorXs fout_copy; //!< Contact force time-derivative (output of the soft contact forward dynamics) (copy)
  // Spatial wrench due to contact force
  pinocchio::ForceTpl<double> pinForce;                                         //!< External spatial force in body coordinates (at parent joint level)
  pinocchio::container::aligned_vector<pinocchio::ForceTpl<double> > fext;      //!< External spatial forces in body coordinates (joint level)
  pinocchio::container::aligned_vector<pinocchio::ForceTpl<double> > fext_copy; //!< External spatial forces in body coordinates (joint level) (copy)
  // Partial derivatives of next force w.r.t. augmented state
  MatrixXs dfdt_dx;         //!< Partial derivative of fout w.r.t. joint state (positions, velocities)
  MatrixXs dfdt_du;         //!< Partial derivative of fout w.r.t. joint torques  
  MatrixXs dfdt_df;         //!< Partial derivative of fout w.r.t. contact force
  MatrixXs dfdt_dx_copy;    //!< Partial derivative of fout w.r.t. joint state (copy)
  MatrixXs dfdt_du_copy;    //!< Partial derivative of fout w.r.t. joint torques (copy)
  MatrixXs dfdt_df_copy;    //!< Partial derivative of fout w.r.t. contact force (copy)
  // Partials of cost w.r.t. force 
  VectorXs Lf;              //!< Gradient of the cost w.r.t. contact force
  MatrixXs Lff;             //!< Hessian of the cost w.r.t. contact force
  // Force residual for hard coded tracking cost
  VectorXs f_residual;      //!< Contact force residual
  MatrixXs f_residual_x;      //!< Contact force residual partial w.r.t. x
  MatrixXs f_residual_f;      //!< Contact force residual partial w.r.t. f
  // Gravity reg residual
  double tau_grav_weight_;
  VectorXs tau_grav_residual;
  MatrixXs tau_grav_residual_x;
  MatrixXs tau_grav_residual_u;
  MatrixXs tau_grav_residual_f;
  VectorXs residual;

  using DADBase::cost;
  using DADBase::Fu;
  using DADBase::Fx;
  using DADBase::Lu;
  using DADBase::Luu;
  using DADBase::Lx;
  using DADBase::Lxu;
  using DADBase::Lxx;
  using DADBase::r;
  using DADBase::xout;

  using DADBase::g;
  using DADBase::Gx;
  using DADBase::Gu;
  using DADBase::h;
  using DADBase::Hx;
  using DADBase::Hu;
};



/**
 * @brief Differential action model for visco-elastic contact forward dynamics in multibody
 * systems (augmented dynamics including the contact force as a state)
 * 
 * Abstract class designed specifically for cartesian force feedback MPC
 * Maths here : https://www.overleaf.com/read/xdpymjfhqqhn
 *
 * \sa `DAMSoftContactAbstractAugmentedFwdDynamics`, `calc()`, `calcDiff()`,
 * `createData()`
 */

class DAMSoftContactAbstractAugmentedFwdDynamics
    : public crocoddyl::DifferentialActionModelAbstractTpl<double> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef crocoddyl::DifferentialActionModelAbstractTpl<double> DAMBase;
  typedef DADSoftContactAbstractAugmentedFwdDynamics Data;
  typedef crocoddyl::MathBaseTpl<double> MathBase;
  typedef crocoddyl::CostModelSumTpl<double> CostModelSum;
  typedef crocoddyl::StateMultibodyTpl<double> StateMultibody;
  typedef crocoddyl::ActuationModelAbstractTpl<double> ActuationModelAbstract;
  typedef crocoddyl::DifferentialActionDataAbstractTpl<double> DifferentialActionDataAbstract;
  typedef crocoddyl::ConstraintModelManagerTpl<double> ConstraintModelManager;
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
   * 
   */
  DAMSoftContactAbstractAugmentedFwdDynamics(
      boost::shared_ptr<StateMultibody> state,
      boost::shared_ptr<ActuationModelAbstract> actuation,
      boost::shared_ptr<CostModelSum> costs,
      const pinocchio::FrameIndex frameId,
      const VectorXs& Kp, 
      const VectorXs& Kv,
      const Vector3s& oPc,
      const std::size_t nc,
      boost::shared_ptr<ConstraintModelManager> constraints = nullptr);
  virtual ~DAMSoftContactAbstractAugmentedFwdDynamics();

  /**
   * @brief Compute the system acceleration, and cost value
   *
   * It computes the system acceleration using soft contact forward-dynamics.
   *
   * @param[in] data  Soft contact forward-dynamics data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] f     Force point \f$\mathbf{f}\in\mathbb{R}^{nc}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const boost::shared_ptr<DifferentialActionDataAbstract>& data, 
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& f,
                    const Eigen::Ref<const VectorXs>& u) = 0;

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
      const Eigen::Ref<const VectorXs>& u) = 0;

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
   * @brief Return the number of inequality constraints
   */
  virtual std::size_t get_ng() const;

  /**
   * @brief Return the number of equality constraints
   */
  virtual std::size_t get_nh() const;

  /**
   * @brief Return the lower bound of the inequality constraints
   */
  virtual const VectorXs& get_g_lb() const;

  /**
   * @brief Return the upper bound of the inequality constraints
   */
  virtual const VectorXs& get_g_ub() const;

  /**
   * @brief Return the actuation model
   */
  const boost::shared_ptr<ActuationModelAbstract>& get_actuation() const;

  /**
   * @brief Return the cost model
   */
  const boost::shared_ptr<CostModelSum>& get_costs() const;

  /**
   * @brief Return the constraint model manager
   */
  const boost::shared_ptr<ConstraintModelManager>& get_constraints() const;

  /**
   * @brief Return the Pinocchio model
   */
  pinocchio::ModelTpl<Scalar>& get_pinocchio() const;

  /**
   * @brief Print relevant information of the free forward-dynamics model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const;

    /**
   * @brief Create the soft contact forward-dynamics data
   *
   * @return soft contact forward-dynamics data
   */
  virtual boost::shared_ptr<DifferentialActionDataAbstract> createData();

  /**
   * @brief Checks that a specific data belongs to this model
   */
  virtual bool checkData(
      const boost::shared_ptr<DifferentialActionDataAbstract>& data);
      
  void set_Kp(const VectorXs& inKp);
  void set_Kv(const VectorXs& inKv);
  void set_oPc(const Vector3s& oPc);
  void set_ref(const pinocchio::ReferenceFrame inRef);
  void set_id(const pinocchio::FrameIndex inId);

  const VectorXs& get_Kp() const;
  const VectorXs& get_Kv() const;
  const Vector3s& get_oPc() const;
  const pinocchio::ReferenceFrame& get_ref() const;
  const pinocchio::FrameIndex& get_id() const;

  // Set cost reference frame
  const pinocchio::ReferenceFrame& get_cost_ref() const;
  void set_cost_ref(const pinocchio::ReferenceFrame inRef);
  
  bool get_active_contact() const;
  void set_active_contact(const bool);

  // Force cost
  // void set_force_cost(const VectorXs& force_des, const double force_weight);
  void set_with_force_cost(const bool);
  void set_force_des(const VectorXs& inForceDes);
  void set_force_weight(const VectorXs& inForceWeights);
  const VectorXs& get_force_des() const;
  const VectorXs& get_force_weight() const;
  bool get_with_force_cost() const;

  // force rate regularization cost
  void set_with_force_rate_reg_cost(const bool);
  void set_force_rate_reg_weight(const VectorXs& inForceWeights);
  const VectorXs& get_force_rate_reg_weight() const;
  bool get_with_force_rate_reg_cost() const;

  // Gravity cost
  bool get_with_gravity_torque_reg() const;
  void set_with_gravity_torque_reg(const bool);
  double get_tau_grav_weight() const;
  void set_tau_grav_weight(const double);

  std::size_t get_nc() {return nc_;};
  std::size_t get_nresidual() {return this->get_nr() + this->get_state()->get_nv() + 2*this->get_nc();};

  // armature 
  bool get_with_armature() const;
  void set_with_armature(const bool);
  const VectorXs& get_armature() const;
  void set_armature(const VectorXs& armature);

  void set_with_force_constraint(const bool inBool) {with_force_constraint_ = inBool; };
  const bool& get_with_force_constraint() const { return with_force_constraint_; };

  protected:
    using DAMBase::g_lb_;   //!< Lower bound of the inequality constraints
    using DAMBase::g_ub_;   //!< Upper bound of the inequality constraints
    using DAMBase::nu_;     //!< Control dimension
    using DAMBase::state_;  //!< Model of the state

    VectorXs Kp_;                             //!< Contact model stiffness
    VectorXs Kv_;                             //!< Contact model damping
    Vector3s oPc_;                          //!< Contact model anchor point
    pinocchio::FrameIndex frameId_;         //!< Frame id of the contact
    pinocchio::FrameIndex parentId_;        //!< Parent id of the contact
    pinocchio::ReferenceFrame ref_;         //!< Pinocchio reference frame
    pinocchio::ReferenceFrame cost_ref_;         //!< Pinocchio reference frame
    bool active_contact_;                   //!< Active contact ?
    std::size_t nc_;                        //!< Contact model dimension
    pinocchio::SE3Tpl<double> jMf_;         //!< Placement of contact frame w.r.t. parent frame
    bool with_armature_;                    //!< Indicate if we have defined an armature
    VectorXs armature_;                     //!< Armature vector
    bool with_force_cost_;                  //!< Force cost ?
    bool with_force_rate_reg_cost_;              //!< Force rate cost ?
    VectorXs force_des_;                    //!< Desired force 3D
    VectorXs force_weight_;                   //!< Force cost weight
    VectorXs force_rate_reg_weight_;          //!< Force rate cost weight
    bool with_gravity_torque_reg_;          //!< Control regularization w.r.t. gravity torque
    double tau_grav_weight_;                //!< Weight on regularization w.r.t. gravity torque
    bool with_force_constraint_; // Add box constraint on the contact force
  
    boost::shared_ptr<ActuationModelAbstract> actuation_;    //!< Actuation model
    boost::shared_ptr<CostModelSum> costs_;                  //!< Cost model
    boost::shared_ptr<ConstraintModelManager> constraints_;  //!< Constraint model
    pinocchio::ModelTpl<double>& pinocchio_;                 //!< Pinocchio model
    bool without_armature_;  //!< Indicate if we have defined an armature
};


}  // namespace softcontact
}  // namespace force_feedback_mpc


#endif  // FORCE_FEEDBACK_MPC_SOFTCONTACT_AUGMENTED_FWDDYN_HPP_

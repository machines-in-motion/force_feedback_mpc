///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, CTU, INRIA,
// University of Oxford Copyright note valid unless otherwise stated in
// individual files. All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "force_feedback_mpc/python.hpp"
#include "force_feedback_mpc/softcontact/dam-augmented.hpp"
#include "dam-augmented.hpp"

namespace force_feedback_mpc {
namespace softcontact {

namespace bp = boost::python;


void exposeDAMSoftContactAbstractAugmentedFwdDyn() {

  bp::register_ptr_to_python<boost::shared_ptr<DAMSoftContactAbstractAugmentedFwdDynamics>>();

  bp::class_<DAMSoftContactAbstractAugmentedFwdDynamics_wrap, boost::noncopyable>(
      "DAMSoftContactAbstractAugmentedFwdDynamics", 
      "Differential action model for visco-elastic contact forward dynamics in multibody systems.",
      bp::init<boost::shared_ptr<crocoddyl::StateMultibody>,
               boost::shared_ptr<crocoddyl::ActuationModelAbstract>,
               boost::shared_ptr<crocoddyl::CostModelSum>,
               pinocchio::FrameIndex, 
               Eigen::VectorXd, 
               Eigen::VectorXd, 
               Eigen::Vector3d, 
               int, 
               bp::optional< boost::shared_ptr<crocoddyl::ConstraintModelManager> > >(
          bp::args("self", "state", "actuation", "costs", "frameId", "Kp", "Kv", "oPc", "nc", "constraints"),
          "Initialize the constrained forward-dynamics action model.\n\n"
          ":param state: multibody state\n"
          ":param actuation: actuation model\n"
          ":param costs: stack of cost functions\n"
          ":param frameId: Frame id of the contact model\n"
          ":param Kp: Stiffness of the visco-elastic contact model\n"
          ":param Kv: Damping of the visco-elastic contact model\n"
          ":param oPc: Anchor point of the contact model\n"
          ":param nc: Dimension of the contact model\n"
          ":param constraints: stack of constraint functions"))
      .def("calc", bp::pure_virtual(&DAMSoftContactAbstractAugmentedFwdDynamics_wrap::calc), bp::args("self", "data", "x", "f", "u"),
           "Compute the system acceleration and cost value.\n\n"
           ":param data: differential action data\n"
           ":param x: state point (dim. state.nx)\n"
           ":param f: force point (dim. state.nx)\n"
           ":param u: control input (dim. nu)")
      .def<void (DAMSoftContactAbstractAugmentedFwdDynamics::*)(
          const boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &DAMSoftContactAbstractAugmentedFwdDynamics::calc,
          bp::args("self", "data", "x", "f"),
          "Compute the next state and cost value.\n\n"
          "It describes the time-continuous evolution of the multibody system under a visco-elastic contact.\n"
          "Additionally it computes the cost value associated to this state and control pair.\n"
          ":param data: soft contact forward-dynamics action data\n"
          ":param x: continuous-time state vector\n"
          ":param f: continuous-time force vector")
      .def("calcDiff", bp::pure_virtual(&DAMSoftContactAbstractAugmentedFwdDynamics_wrap::calcDiff),
           bp::args("self", "data", "x", "f", "u"),
           "Compute the derivatives of the dynamics and cost functions.\n\n"
           "It computes the partial derivatives of the dynamical system and the cost\n"
           "function. It assumes that calc has been run first.\n"
           "This function builds a quadratic approximation of the\n"
           "time-continuous action model (i.e. dynamical system and cost function).\n"
           ":param data: differential action data\n"
           ":param x: state point (dim. state.nx)\n"
           ":param f: force point (dim. state.nc)\n"
           ":param u: control input (dim. nu)")
      .def<void (DAMSoftContactAbstractAugmentedFwdDynamics::*)(
          const boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &DAMSoftContactAbstractAugmentedFwdDynamics::calcDiff,
          bp::args("self", "data", "x", "f"),
          "Compute the derivatives of the differential multibody system and\n"
          "its cost functions.\n\n"
          "It computes the partial derivatives of the differential multibody system and the\n"
          "cost function. It assumes that calc has been run first.\n"
          "This function builds a quadratic approximation of the\n"
          "action model (i.e. dynamical system and cost function).\n"
          ":param data: soft contact differential forward-dynamics action data\n"
          ":param x: time-continuous state vector\n"
          ":param f: time-continuous force vector")
      .def("createData", &DAMSoftContactAbstractAugmentedFwdDynamics_wrap::createData,
           &DAMSoftContactAbstractAugmentedFwdDynamics_wrap::default_createData,
           bp::args("self"),
           "Create the differential action data.\n\n"
           "Each differential action model has its own data that needs to be\n"
           "allocated. This function returns the allocated data for a "
           "predefined\n"
           "DAM.\n"
           ":return DAM data.")
      .add_property("pinocchio",
                    bp::make_function(
                        &DAMSoftContactAbstractAugmentedFwdDynamics::get_pinocchio,
                        bp::return_internal_reference<>()),
                    "multibody model (i.e. pinocchio model)")
      .add_property("actuation",
                    bp::make_function(
                        &DAMSoftContactAbstractAugmentedFwdDynamics::get_actuation,
                        bp::return_value_policy<bp::return_by_value>()),
                    "actuation model")
      .add_property(
          "costs",
          bp::make_function(&DAMSoftContactAbstractAugmentedFwdDynamics::get_costs,
                            bp::return_value_policy<bp::return_by_value>()),
          "total cost model")
      .add_property(
          "constraints",
          bp::make_function(
              &DAMSoftContactAbstractAugmentedFwdDynamics::get_constraints,
              bp::return_value_policy<bp::return_by_value>()),
          "constraint model manager")
      .add_property(
          "Kp",
          bp::make_function(&DAMSoftContactAbstractAugmentedFwdDynamics::get_Kp,
                            bp::return_value_policy<bp::return_by_value>()),
          &DAMSoftContactAbstractAugmentedFwdDynamics::set_Kp,
          "Stiffness of the contact model")
      .add_property(
          "Kv",
          bp::make_function(&DAMSoftContactAbstractAugmentedFwdDynamics::get_Kv,
                            bp::return_value_policy<bp::return_by_value>()),
          &DAMSoftContactAbstractAugmentedFwdDynamics::set_Kv,
          "Damping of the contact model")
      .add_property(
          "oPc",
          bp::make_function(&DAMSoftContactAbstractAugmentedFwdDynamics::get_oPc,
                            bp::return_value_policy<bp::return_by_value>()),
          &DAMSoftContactAbstractAugmentedFwdDynamics::set_oPc,
          "Anchor point of the contact model")
      .add_property(
          "id",
          bp::make_function(&DAMSoftContactAbstractAugmentedFwdDynamics::get_id,
                            bp::return_value_policy<bp::return_by_value>()),
          &DAMSoftContactAbstractAugmentedFwdDynamics::set_id,
          "Contact frame id")
      .add_property(
          "ref",
          bp::make_function(&DAMSoftContactAbstractAugmentedFwdDynamics::get_ref,
                            bp::return_value_policy<bp::return_by_value>()),
          &DAMSoftContactAbstractAugmentedFwdDynamics::set_ref,
          "Pinocchio reference frame")
      .add_property(
          "active_contact",
          bp::make_function(&DAMSoftContactAbstractAugmentedFwdDynamics::get_active_contact,
                            bp::return_value_policy<bp::return_by_value>()),
          &DAMSoftContactAbstractAugmentedFwdDynamics::set_active_contact,
          "Activeness of the soft contact")
      .add_property(
          "with_gravity_torque_reg",
          bp::make_function(&DAMSoftContactAbstractAugmentedFwdDynamics::get_with_gravity_torque_reg,
                            bp::return_value_policy<bp::return_by_value>()),
          &DAMSoftContactAbstractAugmentedFwdDynamics::set_with_gravity_torque_reg,
          "Is control regularization around gravity vector active?")
      .add_property(
          "tau_grav_weight",
          bp::make_function(&DAMSoftContactAbstractAugmentedFwdDynamics::get_tau_grav_weight,
                            bp::return_value_policy<bp::return_by_value>()),
          &DAMSoftContactAbstractAugmentedFwdDynamics::set_tau_grav_weight,
          "Weight of the control regularization w.r.t. gravity torque")
      .add_property(
          "f_des",
          bp::make_function(&DAMSoftContactAbstractAugmentedFwdDynamics::get_force_des,
                            bp::return_value_policy<bp::return_by_value>()),
          &DAMSoftContactAbstractAugmentedFwdDynamics::set_force_des,
          "Desired force in the cost")
      .add_property(
          "f_weight",
          bp::make_function(&DAMSoftContactAbstractAugmentedFwdDynamics::get_force_weight,
                            bp::return_value_policy<bp::return_by_value>()),
          &DAMSoftContactAbstractAugmentedFwdDynamics::set_force_weight,
          "Force cost weight")
      .add_property(
          "with_force_cost",
          bp::make_function(&DAMSoftContactAbstractAugmentedFwdDynamics::get_with_force_cost,
                          bp::return_value_policy<bp::return_by_value>()),
            &DAMSoftContactAbstractAugmentedFwdDynamics::set_with_force_cost,
          "Is the contact force cost active?")
      .add_property(
          "f_rate_reg_weight",
          bp::make_function(&DAMSoftContactAbstractAugmentedFwdDynamics::get_force_rate_reg_weight,
                            bp::return_value_policy<bp::return_by_value>()),
          &DAMSoftContactAbstractAugmentedFwdDynamics::set_force_rate_reg_weight,
          "Force rate cost weight")
      .add_property(
          "with_force_rate_reg_cost",
          bp::make_function(&DAMSoftContactAbstractAugmentedFwdDynamics::get_with_force_rate_reg_cost,
                          bp::return_value_policy<bp::return_by_value>()),
            &DAMSoftContactAbstractAugmentedFwdDynamics::set_with_force_rate_reg_cost,
          "Is the contact force rate cost active?")
      .add_property(
          "cost_ref_",
          bp::make_function(&DAMSoftContactAbstractAugmentedFwdDynamics::get_cost_ref,
                            bp::return_value_policy<bp::return_by_value>()),
          &DAMSoftContactAbstractAugmentedFwdDynamics::set_cost_ref,
          "Reference frame in which the contact force cost residual is expressed (LOCAL or LOCAL_WORLD_ALIGNED) ")
      .add_property(
          "with_armature",
          bp::make_function(&DAMSoftContactAbstractAugmentedFwdDynamics::get_with_armature,
                            bp::return_value_policy<bp::return_by_value>()),
          &DAMSoftContactAbstractAugmentedFwdDynamics::set_with_armature,
          "Is armature considered ? (true if armature values are > 1e-9, false otherwise) ")
      .add_property(
          "armature",
          bp::make_function(&DAMSoftContactAbstractAugmentedFwdDynamics::get_armature,
                            bp::return_value_policy<bp::return_by_value>()),
          &DAMSoftContactAbstractAugmentedFwdDynamics::set_armature,
          "Armature")
      .add_property(
          "g_lb",
          bp::make_function(&DAMSoftContactAbstractAugmentedFwdDynamics_wrap::get_g_lb,
                            bp::return_internal_reference<>()),
          &DAMSoftContactAbstractAugmentedFwdDynamics_wrap::set_g_lb,
          "lower bound of the inequality constraints")
      .add_property(
          "g_ub",
          bp::make_function(&DAMSoftContactAbstractAugmentedFwdDynamics_wrap::get_g_ub,
                            bp::return_internal_reference<>()),
          &DAMSoftContactAbstractAugmentedFwdDynamics_wrap::set_g_ub,
          "upper bound of the inequality constraints")
      .add_property("ng",
                    bp::make_function(
                        &DAMSoftContactAbstractAugmentedFwdDynamics::get_ng,
                        bp::return_value_policy<bp::return_by_value>()),
                    "number of inequality constraints")
      .add_property("nh",
                    bp::make_function(
                        &DAMSoftContactAbstractAugmentedFwdDynamics::get_nh,
                        bp::return_value_policy<bp::return_by_value>()),
                    "number of equality constraints"); 

  bp::register_ptr_to_python<boost::shared_ptr<DADSoftContactAbstractAugmentedFwdDynamics> >();

  bp::class_<DADSoftContactAbstractAugmentedFwdDynamics>(
      "DADSoftContactAbstractAugmentedFwdDynamics", "Action data for the soft contact forward dynamics system",
      bp::init<DAMSoftContactAbstractAugmentedFwdDynamics*>(
          bp::args("self", "model"),
          "Create soft contact forward-dynamics action data.\n\n"
          ":param model: soft contact model"))
      .add_property(
          "oRf",
          bp::make_getter(&DADSoftContactAbstractAugmentedFwdDynamics::oRf,
                          bp::return_internal_reference<>()),
          "Contact frame rotation matrix in WORLD coordinates")
      .add_property(
          "lJ",
          bp::make_getter(&DADSoftContactAbstractAugmentedFwdDynamics::lJ,
                          bp::return_internal_reference<>()),
          "Jacobian of the contact frame in LOCAL")
      .add_property(
          "oJ",
          bp::make_getter(&DADSoftContactAbstractAugmentedFwdDynamics::oJ,
                          bp::return_internal_reference<>()),
          "Jacobian of the contact frame in LOCAL_WORLD_ALIGNED")
      .add_property(
          "aba_dq",
          bp::make_getter(&DADSoftContactAbstractAugmentedFwdDynamics::aba_dq,
                          bp::return_internal_reference<>()),
          "Partial derivative of joint acceleration w.r.t. joint positions")
      .add_property(
          "aba_dv",
          bp::make_getter(&DADSoftContactAbstractAugmentedFwdDynamics::aba_dv,
                          bp::return_internal_reference<>()),
          "Partial derivative of joint accelerations w.r.t. joint velocities")
      .add_property(
          "aba_dx",
          bp::make_getter(&DADSoftContactAbstractAugmentedFwdDynamics::aba_dx,
                          bp::return_internal_reference<>()),
          "Partial derivative of joint accelerations w.r.t. state")
      .add_property(
          "aba_dtau",
          bp::make_getter(&DADSoftContactAbstractAugmentedFwdDynamics::aba_dtau,
                          bp::return_internal_reference<>()),
          "Partial derivative of joint accelerations w.r.t. joint torques")
      .add_property(
          "aba_df",
          bp::make_getter(&DADSoftContactAbstractAugmentedFwdDynamics::aba_df,
                          bp::return_internal_reference<>()),
          "Partial derivative of joint accelerations w.r.t. contact force")
      .add_property(
          "lv",
          bp::make_getter(&DADSoftContactAbstractAugmentedFwdDynamics::lv,
                          bp::return_internal_reference<>()),
          "Linear velocity of the contact frame expressed in LOCAL coordinates")
      .add_property(
          "la",
          bp::make_getter(&DADSoftContactAbstractAugmentedFwdDynamics::la,
                          bp::return_internal_reference<>()),
          "Linear acceleration of the contact frame expressed in LOCAL coordinates")
      .add_property(
          "ov",
          bp::make_getter(&DADSoftContactAbstractAugmentedFwdDynamics::ov,
                          bp::return_internal_reference<>()),
          "Linear velocity of the contact frame expressed in LOCAL_WORLD_ALIGNED coordinates")
      .add_property(
          "oa",
          bp::make_getter(&DADSoftContactAbstractAugmentedFwdDynamics::oa,
                          bp::return_internal_reference<>()),
          "Linear acceleration of the contact frame expressed in LOCAL_WORLD_ALIGNED coordinates")
      .add_property(
          "lv_dq",
          bp::make_getter(&DADSoftContactAbstractAugmentedFwdDynamics::lv_dq,
                          bp::return_internal_reference<>()),
          "Partial derivative of LOCAL contact frame velocity w.r.t. joint positions")
      .add_property(
          "lv_dv",
          bp::make_getter(&DADSoftContactAbstractAugmentedFwdDynamics::lv_dv,
                          bp::return_internal_reference<>()),
          "Partial derivative of LOCAL contact frame velocity w.r.t. joint velocities")
      .add_property(
          "lv_dx",
          bp::make_getter(&DADSoftContactAbstractAugmentedFwdDynamics::lv_dx,
                          bp::return_internal_reference<>()),
          "Partial derivative of LOCAL contact frame velocity w.r.t. state")
      .add_property(
          "a_dq",
          bp::make_getter(&DADSoftContactAbstractAugmentedFwdDynamics::a_dq,
                          bp::return_internal_reference<>()),
          "Partial derivative of LOCAL contact frame acceleration w.r.t. joint positions")
      .add_property(
          "a_dv",
          bp::make_getter(&DADSoftContactAbstractAugmentedFwdDynamics::a_dv,
                          bp::return_internal_reference<>()),
          "Partial derivative of LOCAL contact frame acceleration w.r.t. joint velocities")
      .add_property(
          "a_da",
          bp::make_getter(&DADSoftContactAbstractAugmentedFwdDynamics::a_da,
                        //   bp::return_internal_reference<>()),
                          bp::return_value_policy<bp::return_by_value>()),
          "Partial derivative of LOCAL contact frame acceleration w.r.t. joint acceleration")
      .add_property(
          "da_dx",
          bp::make_getter(&DADSoftContactAbstractAugmentedFwdDynamics::da_dx,
                          bp::return_internal_reference<>()),
          "Partial derivative of LOCAL contact frame acceleration w.r.t. state")
      .add_property(
          "da_du",
          bp::make_getter(&DADSoftContactAbstractAugmentedFwdDynamics::da_du,
                          bp::return_internal_reference<>()),
          "Partial derivative of LOCAL contact frame acceleration w.r.t. joint torques")
      .add_property(
          "da_df",
          bp::make_getter(&DADSoftContactAbstractAugmentedFwdDynamics::da_df,
                          bp::return_internal_reference<>()),
          "Partial derivative of LOCAL contact frame acceleration w.r.t. contact force")
      
      .add_property(
          "fout",
          bp::make_getter(&DADSoftContactAbstractAugmentedFwdDynamics::fout,
                          bp::return_internal_reference<>()),
          "Time-derivative of the contact force (output of forward dynamics)")
      .add_property(
          "dfdt_dx",
          bp::make_getter(&DADSoftContactAbstractAugmentedFwdDynamics::dfdt_dx,
                          bp::return_internal_reference<>()),
          "Partial derivative of fout w.r.t. joint state")
      .add_property(
          "dfdt_du",
          bp::make_getter(&DADSoftContactAbstractAugmentedFwdDynamics::dfdt_du,
                          bp::return_internal_reference<>()),
          "Partial derivative of fout w.r.t. joint torques")
      .add_property(
          "dfdt_df",
          bp::make_getter(&DADSoftContactAbstractAugmentedFwdDynamics::dfdt_df,
                          bp::return_internal_reference<>()),
          "Partial derivative of fout w.r.t. contact force")
      .add_property(
          "Lf",
          bp::make_getter(&DADSoftContactAbstractAugmentedFwdDynamics::Lf,
                          bp::return_internal_reference<>()),
          "Gradient of the cost w.r.t. contact force")
      .add_property(
          "Lff",
          bp::make_getter(&DADSoftContactAbstractAugmentedFwdDynamics::Lff,
                          bp::return_internal_reference<>()),
          "Hessian of the cost w.r.t. contact force")
      .add_property(
          "f_residual",
          bp::make_getter(&DADSoftContactAbstractAugmentedFwdDynamics::f_residual,
                          bp::return_internal_reference<>()),
          "Residual of the contact force");
}

}  // namespace softcontact
}  // namespace force_feedback_mpc

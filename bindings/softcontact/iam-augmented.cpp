///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, CTU, INRIA,
// University of Oxford Copyright note valid unless otherwise stated in
// individual files. All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "force_feedback_mpc/python.hpp"
#include "force_feedback_mpc/softcontact/iam-augmented.hpp"

namespace force_feedback_mpc {
namespace softcontact {

namespace bp = boost::python;

void exposeIAMSoftContactAugmented() {
  
  bp::register_ptr_to_python<boost::shared_ptr<IAMSoftContactAugmented> >();
  
  bp::class_<IAMSoftContactAugmented, bp::bases<crocoddyl::ActionModelAbstract> >(
      "IAMSoftContactAugmented",
      "Sympletic Euler integrator for differential action models.\n\n"
      "This class implements a sympletic Euler integrator (a.k.a "
      "semi-implicit\n"
      "integrator) give a differential action model, i.e.:\n"
      "  [q+, v+, tau+] = StateLPF.integrate([q, v], [v + a * dt, a * dt] * "
      "dt, [alpha*tau + (1-alpha)*w]).",
      bp::init<boost::shared_ptr<DAMSoftContactAbstractAugmentedFwdDynamics>,
               bp::optional<double, bool, std::vector<boost::shared_ptr<force_feedback_mpc::frictioncone::ResidualModelFrictionConeAugmented>> > >(
          bp::args("self", "diffModel", "stepTime", "withCostResidual", "friction_constraints"),
          "Initialize the sympletic Euler integrator.\n\n"
          ":param diffModel: differential action model\n"
          ":param stepTime: step time\n"
          ":param withCostResidual: includes the cost residuals and derivatives computation, or tau\n"
          ":param friction_constraints: list of friction cone constraint residual models"))
      .def<void (IAMSoftContactAugmented::*)(
          const boost::shared_ptr<crocoddyl::ActionDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &IAMSoftContactAugmented::calc,
          bp::args("self", "data", "x", "u"),
          "Compute the time-discrete evolution of a differential action "
          "model.\n\n"
          "It describes the time-discrete evolution of action model.\n"
          ":param data: action data\n"
          ":param x: state vector\n"
          ":param u: control input")
      .def<void (IAMSoftContactAugmented::*)(
          const boost::shared_ptr<crocoddyl::ActionDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &crocoddyl::ActionModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (IAMSoftContactAugmented::*)(
          const boost::shared_ptr<crocoddyl::ActionDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &IAMSoftContactAugmented::calcDiff,
          bp::args("self", "data", "x", "u"),
          "Computes the derivatives of the integrated action model wrt state "
          "and control. \n\n"
          "This function builds a quadratic approximation of the\n"
          "action model (i.e. dynamical system and cost function).\n"
          "It assumes that calc has been run first.\n"
          ":param data: action data\n"
          ":param x: state vector\n"
          ":param u: control input\n")
      .def<void (IAMSoftContactAugmented::*)(
          const boost::shared_ptr<crocoddyl::ActionDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &crocoddyl::ActionModelAbstract::calcDiff,
          bp::args("self", "data", "x"))
      .def("createData", &IAMSoftContactAugmented::createData,
           bp::args("self"), "Create the Euler integrator data.")
      .add_property(
          "differential",
          bp::make_function(&IAMSoftContactAugmented::get_differential,
                            bp::return_value_policy<bp::return_by_value>()),
          &IAMSoftContactAugmented::set_differential,
          "differential action model")
      .add_property(
          "dt",
          bp::make_function(&IAMSoftContactAugmented::get_dt,
                            bp::return_value_policy<bp::return_by_value>()),
          &IAMSoftContactAugmented::set_dt, "step time")

      .add_property(
          "nc",
          bp::make_function(&IAMSoftContactAugmented::get_nc,
                            bp::return_value_policy<bp::return_by_value>()),
          "Contact model dimension")
      .add_property(
          "ny",
          bp::make_function(&IAMSoftContactAugmented::get_ny,
                            bp::return_value_policy<bp::return_by_value>()),
          "augmented state dimension (nx+ntau)")
      .add_property(
          "force_lb",
          bp::make_function(&IAMSoftContactAugmented::get_force_lb,
                            bp::return_value_policy<bp::return_by_value>()),
          &IAMSoftContactAugmented::set_force_lb,
          "lower bound on the box constraint on the contact force")
      .add_property(
          "force_ub",
          bp::make_function(&IAMSoftContactAugmented::get_force_ub,
                            bp::return_value_policy<bp::return_by_value>()),
          &IAMSoftContactAugmented::set_force_ub,
          "upper bound on the box constraint on the contact force")
      .add_property(
          "with_force_constraint",
          bp::make_function(&IAMSoftContactAugmented::get_with_force_constraint,
                            bp::return_value_policy<bp::return_by_value>()),
          &IAMSoftContactAugmented::set_with_force_constraint,
          "activate box constraint on the contact force (default: False)")
      .add_property(
          "with_friction_cone_constraint",
          bp::make_function(&IAMSoftContactAugmented::get_with_friction_cone_constraint,
                            bp::return_value_policy<bp::return_by_value>()),
          "activate friction cone (Lorentz) constraint on the contact force (default: False)")
      .add_property(
          "friction_constraints",
          bp::make_function(&IAMSoftContactAugmented::set_friction_cone_constraints,
                            bp::return_value_policy<bp::return_by_value>()),
          &IAMSoftContactAugmented::get_friction_cone_constraints,
          "friction cone constraints");

  bp::register_ptr_to_python<boost::shared_ptr<IADSoftContactAugmented> >();

  bp::class_<IADSoftContactAugmented, bp::bases<crocoddyl::ActionDataAbstract> >(
      "IADSoftContactAugmented", "Sympletic Euler integrator data.",
      bp::init<IAMSoftContactAugmented*>(
          bp::args("self", "model"),
          "Create sympletic Euler integrator data.\n\n"
          ":param model: sympletic Euler integrator model"))
      .add_property(
          "differential",
          bp::make_getter(&IADSoftContactAugmented::differential,
                          bp::return_value_policy<bp::return_by_value>()),
          "differential action data")
      .add_property("dy",
                    bp::make_getter(&IADSoftContactAugmented::dy,
                                    bp::return_internal_reference<>()),
                    "state rate.")
      .add_property(
          "friction_cone_residual",
          bp::make_getter(&IADSoftContactAugmented::friction_cone_residual,
                          bp::return_value_policy<bp::return_by_value>()),
          "friction cone residual")
      .add_property(
          "dcone_df",
          bp::make_getter(&IADSoftContactAugmented::dcone_df,
                          bp::return_value_policy<bp::return_by_value>()),
          "friction cone residual derivative w.r.t. f");
}

}  // namespace softcontact
}  // namespace force_feedback_mpc

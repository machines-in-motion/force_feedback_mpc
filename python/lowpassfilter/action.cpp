///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "force_feedback_mpc/lowpassfilter/action.hpp"

#include <crocoddyl/core/action-base.hpp>
#include <eigenpy/eigenpy.hpp>
#include <pinocchio/fwd.hpp>  // to avoid compilation error (https://github.com/loco-3d/crocoddyl/issues/205)

#include "force_feedback_mpc/fwd.hpp"

namespace bp = boost::python;

namespace force_feedback_mpc {


void exposeIntegratedActionModelLPF() {
  bp::register_ptr_to_python<boost::shared_ptr<force_feedback_mpc::IntegratedActionModelLPF> >();
  bp::class_<force_feedback_mpc::IntegratedActionModelLPF, bp::bases<crocoddyl::ActionModelAbstract> >(
      "IntegratedActionModelLPF",
      "Sympletic Euler integrator for differential action models.\n\n"
      "This class implements a sympletic Euler integrator (a.k.a "
      "semi-implicit\n"
      "integrator) give a differential action model, i.e.:\n"
      "  [q+, v+, tau+] = StateLPF.integrate([q, v], [v + a * dt, a * dt] * "
      "dt, [alpha*tau + (1-alpha)*w]).",
      bp::init<boost::shared_ptr<crocoddyl::DifferentialActionModelAbstract>,
               bp::optional<std::vector<std::string>, double, bool, double,
                            bool, int> >(
          bp::args("self", "diffModel", "LPFJointNames", "stepTime",
                   "withCostResidual", "fc", "tau_plus_integration", "filter"),
          "Initialize the sympletic Euler integrator.\n\n"
          ":param diffModel: differential action model\n"
          ":param LPFJointNames: names of joints that are low-pass filtered\n"
          ":param stepTime: step time\n"
          ":param withCostResidual: includes the cost residuals and "
          "derivatives\n"
          ":param fc: LPF parameter depending on cut-off frequency "
          "alpha=1/(1+2*Pi*dt*fc)\n"
          ":param tau_plus_integration: use tau+=LPF(tau,w) in acceleration "
          "computation, or tau\n"
          ":param filter: type of low-pass filter (0 = Expo Moving Avg, 1 = "
          "Classical, 2 = Exact)"))
      .def<void (force_feedback_mpc::IntegratedActionModelLPF::*)(
          const boost::shared_ptr<crocoddyl::ActionDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &force_feedback_mpc::IntegratedActionModelLPF::calc,
          bp::args("self", "data", "x", "u"),
          "Compute the time-discrete evolution of a differential action "
          "model.\n\n"
          "It describes the time-discrete evolution of action model.\n"
          ":param data: action data\n"
          ":param x: state vector\n"
          ":param u: control input")
      .def<void (force_feedback_mpc::IntegratedActionModelLPF::*)(
          const boost::shared_ptr<crocoddyl::ActionDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &crocoddyl::ActionModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (force_feedback_mpc::IntegratedActionModelLPF::*)(
          const boost::shared_ptr<crocoddyl::ActionDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &force_feedback_mpc::IntegratedActionModelLPF::calcDiff,
          bp::args("self", "data", "x", "u"),
          "Computes the derivatives of the integrated action model wrt state "
          "and control. \n\n"
          "This function builds a quadratic approximation of the\n"
          "action model (i.e. dynamical system and cost function).\n"
          "It assumes that calc has been run first.\n"
          ":param data: action data\n"
          ":param x: state vector\n"
          ":param u: control input\n")
      .def<void (force_feedback_mpc::IntegratedActionModelLPF::*)(
          const boost::shared_ptr<crocoddyl::ActionDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &crocoddyl::ActionModelAbstract::calcDiff,
          bp::args("self", "data", "x"))
      .def("createData", &force_feedback_mpc::IntegratedActionModelLPF::createData,
           bp::args("self"), "Create the Euler integrator data.")
      .add_property(
          "differential",
          bp::make_function(&force_feedback_mpc::IntegratedActionModelLPF::get_differential,
                            bp::return_value_policy<bp::return_by_value>()),
          &force_feedback_mpc::IntegratedActionModelLPF::set_differential,
          "differential action model")
      .add_property(
          "dt",
          bp::make_function(&force_feedback_mpc::IntegratedActionModelLPF::get_dt,
                            bp::return_value_policy<bp::return_by_value>()),
          &force_feedback_mpc::IntegratedActionModelLPF::set_dt, "step time")
      .add_property(
          "fc",
          bp::make_function(&force_feedback_mpc::IntegratedActionModelLPF::get_fc,
                            bp::return_value_policy<bp::return_by_value>()),
          &IntegratedActionModelLPF::set_fc,
          "cut-off frequency of low-pass filter")
      .add_property(
          "alpha",
          bp::make_function(&force_feedback_mpc::IntegratedActionModelLPF::get_alpha,
                            bp::return_value_policy<bp::return_by_value>()),
          &force_feedback_mpc::IntegratedActionModelLPF::set_alpha,
          "discrete parameter of the low-pass filter")

      .add_property(
          "nw",
          bp::make_function(&force_feedback_mpc::IntegratedActionModelLPF::get_nw,
                            bp::return_value_policy<bp::return_by_value>()),
          "torque actuation dimension (nu)")
      .add_property(
          "ntau",
          bp::make_function(&force_feedback_mpc::IntegratedActionModelLPF::get_ntau,
                            bp::return_value_policy<bp::return_by_value>()),
          "low-pass filtered actuation dimension")
      .add_property(
          "ny",
          bp::make_function(&force_feedback_mpc::IntegratedActionModelLPF::get_ny,
                            bp::return_value_policy<bp::return_by_value>()),
          "augmented state dimension (nx+ntau)")

      .add_property(
          "lpf_joint_names",
          bp::make_function(&force_feedback_mpc::IntegratedActionModelLPF::get_lpf_joint_names,
                            bp::return_value_policy<bp::return_by_value>()),
          "names of the joints that are low-pass filtered")
      .add_property(
          "lpf_torque_ids",
          bp::make_function(&force_feedback_mpc::IntegratedActionModelLPF::get_lpf_torque_ids,
                            bp::return_value_policy<bp::return_by_value>()),
          "ids in the torque vector of dimensions that are low-pass filtered")
      .add_property(
          "non_lpf_torque_ids",
          bp::make_function(&force_feedback_mpc::IntegratedActionModelLPF::get_non_lpf_torque_ids,
                            bp::return_value_policy<bp::return_by_value>()),
          "ids in the torque vector of dimensions that are NOT low-pass "
          "filtered (perfect actuators)")

      .def("set_control_reg_cost",
           &force_feedback_mpc::IntegratedActionModelLPF::set_control_reg_cost,
           bp::args("self", "weight", "ref"),
           "Initialize cost weight and reference for unfiltered torque "
           "regularization (2-norm residual).")

      .def("set_control_lim_cost",
           &force_feedback_mpc::IntegratedActionModelLPF::set_control_lim_cost,
           bp::args("self", "weight"),
           "Initialize cost weight unfiltered torque limit penalization "
           "(quadratic barrier).");

  bp::register_ptr_to_python<boost::shared_ptr<force_feedback_mpc::IntegratedActionDataLPF> >();

  bp::class_<force_feedback_mpc::IntegratedActionDataLPF, bp::bases<crocoddyl::ActionDataAbstract> >(
      "IntegratedActionDataLPF", "Sympletic Euler integrator data.",
      bp::init<IntegratedActionModelLPF*>(
          bp::args("self", "model"),
          "Create sympletic Euler integrator data.\n\n"
          ":param model: sympletic Euler integrator model"))
      .add_property(
          "differential",
          bp::make_getter(&force_feedback_mpc::IntegratedActionDataLPF::differential,
                          bp::return_value_policy<bp::return_by_value>()),
          "differential action data")
      .add_property("dy",
                    bp::make_getter(&force_feedback_mpc::IntegratedActionDataLPF::dy,
                                    bp::return_internal_reference<>()),
                    "state rate.");
}

}  // namespace force_feedback_mpc

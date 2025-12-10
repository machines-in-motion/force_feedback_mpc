///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, CTU, INRIA,
// University of Oxford Copyright note valid unless otherwise stated in
// individual files. All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "force_feedback_mpc/force_feedback_mpc_python.hpp"
#include "force_feedback_mpc/softcontact/dam3d-augmented.hpp"

namespace force_feedback_mpc {
namespace softcontact {

namespace bp = boost::python;

void exposeDAMSoftContact3DAugmentedFwdDyn() {

  bp::register_ptr_to_python<std::shared_ptr<DAMSoftContact3DAugmentedFwdDynamics>>();

  bp::class_<DAMSoftContact3DAugmentedFwdDynamics, bp::bases<DAMSoftContactAbstractAugmentedFwdDynamics>>(
      "DAMSoftContact3DAugmentedFwdDynamics", 
      "Differential action model for 3D visco-elastic contact forward dynamics in multibody systems.",
      bp::init<std::shared_ptr<crocoddyl::StateMultibody>,
               std::shared_ptr<crocoddyl::ActuationModelAbstract>,
               std::shared_ptr<crocoddyl::CostModelSum>,
               pinocchio::FrameIndex, Eigen::VectorXd, Eigen::VectorXd, Eigen::Vector3d,
               bp::optional<std::shared_ptr<crocoddyl::ConstraintModelManager>> >(
          bp::args("self", "state", "actuation", "costs", "frameId", "Kp", "Kv", "oPc", "constraints"),
          "Initialize the constrained forward-dynamics action model.\n\n"
          ":param state: multibody state\n"
          ":param actuation: actuation model\n"
          ":param costs: stack of cost functions\n"
          ":param frameId: Frame id of the contact model "
          ":param Kp: Stiffness of the visco-elastic contact model "
          ":param Kv: Damping of the visco-elastic contact model "
          ":param oPc: Anchor point of the contact model "
          ":param constraints: stack of constraint functions"))
      .def<void (DAMSoftContact3DAugmentedFwdDynamics::*)(
          const std::shared_ptr<crocoddyl::DifferentialActionDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &DAMSoftContact3DAugmentedFwdDynamics::calc,
          bp::args("self", "data", "x", "f", "u"),
          "Compute the next state and cost value.\n\n"
          "It describes the time-continuous evolution of the multibody system under a visco-elastic contact.\n"
          "Additionally it computes the cost value associated to this state and control pair.\n"
          ":param data: soft contact 3d forward-dynamics action data\n"
          ":param x: continuous-time state vector\n"
          ":param f: continuous-time force vector\n"
          ":param u: continuous-time control input")
      .def<void (DAMSoftContact3DAugmentedFwdDynamics::*)(
          const std::shared_ptr<crocoddyl::DifferentialActionDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&, 
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &DAMSoftContact3DAugmentedFwdDynamics::calc, bp::args("self", "data", "x", "f"))
      
      .def<void (DAMSoftContact3DAugmentedFwdDynamics::*)(
          const std::shared_ptr<crocoddyl::DifferentialActionDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &DAMSoftContact3DAugmentedFwdDynamics::calcDiff,
          bp::args("self", "data", "x", "f", "u"),
          "Compute the derivatives of the differential multibody system and\n"
          "its cost functions.\n\n"
          "It computes the partial derivatives of the differential multibody system and the\n"
          "cost function. It assumes that calc has been run first.\n"
          "This function builds a quadratic approximation of the\n"
          "action model (i.e. dynamical system and cost function).\n"
          ":param data: soft contact 3d differential forward-dynamics action data\n"
          ":param x: time-continuous state vector\n"
          ":param x: time-continuous force vector\n"
          ":param u: time-continuous control input\n")
      .def<void (DAMSoftContact3DAugmentedFwdDynamics::*)(
          const std::shared_ptr<crocoddyl::DifferentialActionDataAbstract>&, 
          const Eigen::Ref<const Eigen::VectorXd>&, 
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &DAMSoftContact3DAugmentedFwdDynamics::calcDiff, bp::args("self", "data", "x", "f"))
      .def("createData", &DAMSoftContact3DAugmentedFwdDynamics::createData,
           bp::args("self"),
           "Create the forward dynamics differential action data.");

  bp::register_ptr_to_python<std::shared_ptr<DADSoftContact3DAugmentedFwdDynamics> >();

  bp::class_<DADSoftContact3DAugmentedFwdDynamics, bp::bases<DADSoftContactAbstractAugmentedFwdDynamics> >(
      "DADSoftContact3DAugmentedFwdDynamics", "Action data for the soft contact 3D forward dynamics system",
      bp::init<DAMSoftContact3DAugmentedFwdDynamics*>(
          bp::args("self", "model"),
          "Create soft contact 3D forward-dynamics action data.\n\n"
          ":param model: soft contact 3D model"))
      .add_property(
          "constraints",
          bp::make_getter(&DADSoftContact3DAugmentedFwdDynamics::constraints,
                          bp::return_value_policy<bp::return_by_value>()),
          "constraint data")
      .add_property(
          "pinocchio",
          bp::make_getter(&DADSoftContact3DAugmentedFwdDynamics::pinocchio,
                          bp::return_internal_reference<>()),
          "pinocchio data")
      .add_property(
          "multibody",
          bp::make_getter(&DADSoftContact3DAugmentedFwdDynamics::multibody,
                          bp::return_internal_reference<>()),
          "multibody data")
      .add_property(
          "costs",
          bp::make_getter(&DADSoftContact3DAugmentedFwdDynamics::costs,
                          bp::return_value_policy<bp::return_by_value>()),
          "total cost data")
      .add_property(
          "tau_grav_residual",
          bp::make_getter(&DADSoftContact3DAugmentedFwdDynamics::tau_grav_residual,
                          bp::return_internal_reference<>()),
          "tau_grav_residual data")
      .add_property(
           "fout",
           bp::make_getter(&DADSoftContact3DAugmentedFwdDynamics::fout,
                           bp::return_internal_reference<>()),
           "fout data");
}

}  // namespace softcontact
}  // namespace force_feedback_mpc

///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2023, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "force-feedback-mpc-python.hpp"
#include "force_feedback_mpc/frictioncone/residual-friction-cone.hpp"

namespace force_feedback_mpc {
namespace frictioncone {

namespace bp = boost::python;

void exposeResidualFrictionCone() {
  bp::register_ptr_to_python<
      std::shared_ptr<ResidualModelFrictionCone> >();

  bp::class_<ResidualModelFrictionCone,
             bp::bases<crocoddyl::ResidualModelAbstract> >(
      "ResidualModelFrictionCone",
      "Nonlinear (Lorentz) friction cone.",
      bp::init<std::shared_ptr<crocoddyl::StateMultibody>, pinocchio::FrameIndex,
               double, std::size_t >(
          bp::args("self", "state", "id", "coef", "nu"),
          "Initialize the contact friction cone residual model.\n\n"
          ":param state: state of the multibody system\n"
          ":param id: reference frame id\n"
          ":param coef: friction coefficient mu\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<std::shared_ptr<crocoddyl::StateMultibody>, pinocchio::FrameIndex, double>(
          bp::args("self", "state", "id", "coef"),
          "Initialize the contact friction cone residual model.\n\n"
          "The default nu is obtained from state.nv. Note that this "
          "constructor can be used for forward-dynamics\n"
          "cases only.\n"
          ":param state: state of the multibody system\n"
          ":param id: reference frame id\n"
          ":param coef: friction coefficient mu"))
      .def<void (ResidualModelFrictionCone::*)(
          const std::shared_ptr<crocoddyl::ResidualDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualModelFrictionCone::calc,
          bp::args("self", "data", "x", "u"),
          "Compute the contact friction cone residual.\n\n"
          ":param data: residual data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: control input (dim. nu)")
      .def<void (ResidualModelFrictionCone::*)(
          const std::shared_ptr<crocoddyl::ResidualDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &crocoddyl::ResidualModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (ResidualModelFrictionCone::*)(
          const std::shared_ptr<crocoddyl::ResidualDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualModelFrictionCone::calcDiff,
          bp::args("self", "data", "x", "u"),
          "Compute the Jacobians of the contact friction cone residual.\n\n"
          "It assumes that calc has been run first.\n"
          ":param data: action data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: control input (dim. nu)")
      .def<void (ResidualModelFrictionCone::*)(
          const std::shared_ptr<crocoddyl::ResidualDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &crocoddyl::ResidualModelAbstract::calcDiff,
          bp::args("self", "data", "x"))
      .def(
          "createData", &ResidualModelFrictionCone::createData,
          bp::with_custodian_and_ward_postcall<0, 2>(),
          bp::args("self", "data"),
          "Create the contact friction cone residual data.\n\n"
          "Each residual model has its own data that needs to be allocated. "
          "This function\n"
          "returns the allocated data for the contact friction cone residual.\n"
          ":param data: shared data\n"
          ":return residual data.")
      .add_property("id",
                    bp::make_function(
                        &ResidualModelFrictionCone::get_id,
                        bp::return_value_policy<bp::return_by_value>()),
                    "frame id")
      .add_property(
          "coef",
          bp::make_function(&ResidualModelFrictionCone::get_friction_coef,
                            bp::return_value_policy<bp::return_by_value>()),
          &ResidualModelFrictionCone::set_friction_coef,
          "Friction cone coefficient"); 
    //   .def(CopyableVisitor<ResidualModelFrictionCone>());

  bp::register_ptr_to_python<
      std::shared_ptr<ResidualDataFrictionCone> >();

  bp::class_<ResidualDataFrictionCone, bp::bases<crocoddyl::ResidualDataAbstract> >(
      "ResidualDataFrictionCone",
      "Data for contact friction cone residual.\n\n",
      bp::init<ResidualModelFrictionCone*, crocoddyl::DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create contact friction cone residual data.\n\n"
          ":param model: contact friction cone residual model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<
          1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property(
          "contact",
          bp::make_getter(&ResidualDataFrictionCone::contact,
                          bp::return_value_policy<bp::return_by_value>()),
          bp::make_setter(&ResidualDataFrictionCone::contact),
          "contact data associated with the current residual")
      .add_property(
          "f3d",
          bp::make_getter(&ResidualDataFrictionCone::f3d,
                          bp::return_internal_reference<>()),
          "Contact force")
      .add_property(
          "dcone_df",
          bp::make_getter(&ResidualDataFrictionCone::dcone_df,
                          bp::return_internal_reference<>()),
          "Derivative of the friction cone w.r.t. contact force")
      .add_property(
          "df_dx",
          bp::make_getter(&ResidualDataFrictionCone::df_dx,
                          bp::return_internal_reference<>()),
          "Derivative of the contact force w.r.t. state")
      .add_property(
          "df_du",
          bp::make_getter(&ResidualDataFrictionCone::df_du,
                          bp::return_internal_reference<>()),
          "Derivative of the contact force w.r.t. control");
    //   .def(CopyableVisitor<ResidualDataFrictionCone>());
}

}  // namespace softcontact
}  // namespace force_feedback_mpc

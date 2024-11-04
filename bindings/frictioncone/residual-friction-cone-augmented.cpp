///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2023, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "force_feedback_mpc/python.hpp"
#include "force_feedback_mpc/frictioncone/residual-friction-cone-augmented.hpp"

namespace force_feedback_mpc {
namespace frictioncone {

namespace bp = boost::python;

void exposeResidualFrictionConeAugmented() {
  bp::register_ptr_to_python<
      boost::shared_ptr<ResidualModelFrictionConeAugmented> >();

  bp::class_<ResidualModelFrictionConeAugmented,
             bp::bases<crocoddyl::ResidualModelAbstract> >(
      "ResidualModelFrictionConeAugmented",
      "Nonlinear (Lorentz) friction cone.",
      bp::init<boost::shared_ptr<crocoddyl::StateMultibody>, pinocchio::FrameIndex,
               double, std::size_t >(
          bp::args("self", "state", "id", "coef", "nu"),
          "Initialize the contact friction cone residual model.\n\n"
          ":param state: state of the multibody system\n"
          ":param id: reference frame id\n"
          ":param coef: friction coefficient mu\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<crocoddyl::StateMultibody>, pinocchio::FrameIndex, double>(
          bp::args("self", "state", "id", "coef"),
          "Initialize the contact friction cone residual model.\n\n"
          "The default nu is obtained from state.nv. Note that this "
          "constructor can be used for forward-dynamics\n"
          "cases only.\n"
          ":param state: state of the multibody system\n"
          ":param id: reference frame id\n"
          ":param coef: friction coefficient mu"))
      .def<void (ResidualModelFrictionConeAugmented::*)(
          const boost::shared_ptr<crocoddyl::ResidualDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualModelFrictionConeAugmented::calc,
          bp::args("self", "data", "f"),
          "Compute the contact friction cone residual.\n\n"
          ":param data: residual data\n"
          ":param f: force (dim 3)")
      .def<void (ResidualModelFrictionConeAugmented::*)(
          const boost::shared_ptr<crocoddyl::ResidualDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualModelFrictionConeAugmented::calcDiff,
          bp::args("self", "data", "f"),
          "Compute the Jacobians of the contact friction cone residual.\n\n"
          "It assumes that calc has been run first.\n"
          ":param data: action data\n"
          ":param f: force (dim 3)")
      .def(
          "createData", &ResidualModelFrictionConeAugmented::createData,
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
                        &ResidualModelFrictionConeAugmented::get_id,
                        bp::return_value_policy<bp::return_by_value>()),
                    "frame id")
      .add_property(
          "coef",
          bp::make_function(&ResidualModelFrictionConeAugmented::get_friction_coef,
                            bp::return_value_policy<bp::return_by_value>()),
          &ResidualModelFrictionConeAugmented::set_friction_coef,
          "Friction cone coefficient"); 
    //   .def(CopyableVisitor<ResidualModelFrictionConeAugmented>());

  bp::register_ptr_to_python<
      boost::shared_ptr<ResidualDataFrictionConeAugmented> >();

  bp::class_<ResidualDataFrictionConeAugmented, bp::bases<crocoddyl::ResidualDataAbstract> >(
      "ResidualDataFrictionConeAugmented",
      "Data for contact friction cone residual.\n\n",
      bp::init<ResidualModelFrictionConeAugmented*, crocoddyl::DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create contact friction cone residual data.\n\n"
          ":param model: contact friction cone residual model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<
          1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property(
          "dcone_df",
          bp::make_getter(&ResidualDataFrictionConeAugmented::dcone_df,
                          bp::return_value_policy<bp::return_by_value>()),
          "Derivative of the friction cone residual w.r.t. soft contact force")
      .add_property(
          "residual",
          bp::make_getter(&ResidualDataFrictionConeAugmented::residual,
                          bp::return_value_policy<bp::return_by_value>()),
          "Friction cone residual");
    //   .def(CopyableVisitor<ResidualDataFrictionConeAugmented>());
}

}  // namespace softcontact
}  // namespace force_feedback_mpc

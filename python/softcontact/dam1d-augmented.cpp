///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, CTU, INRIA,
// University of Oxford Copyright note valid unless otherwise stated in
// individual files. All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "sobec/crocomplements/softcontact/dam1d-augmented.hpp"

#include <boost/python.hpp>
#include <boost/python/enum.hpp>
#include <eigenpy/eigenpy.hpp>

namespace sobec {
namespace python {
namespace bp = boost::python;

void exposeDAMSoftContact1DAugmentedFwdDyn() {

  bp::register_ptr_to_python<boost::shared_ptr<DAMSoftContact1DAugmentedFwdDynamics>>();

  bp::class_<DAMSoftContact1DAugmentedFwdDynamics, bp::bases<sobec::DAMSoftContactAbstractAugmentedFwdDynamics>>(
      "DAMSoftContact1DAugmentedFwdDynamics", 
      "Differential action model for 1D visco-elastic contact forward dynamics in multibody systems.",
      bp::init<boost::shared_ptr<crocoddyl::StateMultibody>,
               boost::shared_ptr<crocoddyl::ActuationModelAbstract>,
               boost::shared_ptr<crocoddyl::CostModelSum>,
               pinocchio::FrameIndex, Eigen::VectorXd, Eigen::VectorXd, Eigen::Vector3d, pinocchio::ReferenceFrame, sobec::Vector3MaskType>(
          bp::args("self", "state", "actuation", "costs", "frameId", "Kp", "Kv", "oPc", "ref", "type"),
          "Initialize the constrained forward-dynamics action model.\n\n"
          ":param state: multibody state\n"
          ":param actuation: actuation model\n"
          ":param costs: stack of cost functions\n"
          ":param frameId: Frame id of the contact model\n"
          ":param Kp: Stiffness of the visco-elastic contact model\n"
          ":param Kv: Damping of the visco-elastic contact model\n"
          ":param oPc: Anchor point of the contact model\n"
          ":param ref: Pinocchio reference frame of the contact\n"
          ":param type: Contact 1D mask type "))
      .def<void (DAMSoftContact1DAugmentedFwdDynamics::*)(
          const boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &DAMSoftContact1DAugmentedFwdDynamics::calc,
          bp::args("self", "data", "x", "f", "u"),
          "Compute the next state and cost value.\n\n"
          "It describes the time-continuous evolution of the multibody system under a visco-elastic contact.\n"
          "Additionally it computes the cost value associated to this state and control pair.\n"
          ":param data: soft contact 1d forward-dynamics action data\n"
          ":param x: continuous-time state vector\n"
          ":param f: continuous-time force vector\n"
          ":param u: continuous-time control input")
      .def<void (DAMSoftContact1DAugmentedFwdDynamics::*)(
          const boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&, 
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &DAMSoftContact1DAugmentedFwdDynamics::calc, bp::args("self", "data", "x", "f"))
      .def<void (DAMSoftContact1DAugmentedFwdDynamics::*)(
          const boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &DAMSoftContact1DAugmentedFwdDynamics::calcDiff,
          bp::args("self", "data", "x", "f", "u"),
          "Compute the derivatives of the differential multibody system and\n"
          "its cost functions.\n\n"
          "It computes the partial derivatives of the differential multibody system and the\n"
          "cost function. It assumes that calc has been run first.\n"
          "This function builds a quadratic approximation of the\n"
          "action model (i.e. dynamical system and cost function).\n"
          ":param data: soft contact 1d differential forward-dynamics action data\n"
          ":param x: time-continuous state vector\n"
          ":param x: time-continuous force vector\n"
          ":param u: time-continuous control input\n")
      .def<void (DAMSoftContact1DAugmentedFwdDynamics::*)(
          const boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract>&, 
          const Eigen::Ref<const Eigen::VectorXd>&, 
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &DAMSoftContact1DAugmentedFwdDynamics::calcDiff, bp::args("self", "data", "x", "f"))

      .def("createData", &DAMSoftContact1DAugmentedFwdDynamics::createData,
           bp::args("self"), "Create the Euler integrator data.")

      .add_property(
          "type",
          bp::make_function(&DAMSoftContact1DAugmentedFwdDynamics::get_type,
                            bp::return_value_policy<bp::return_by_value>()),
          &DAMSoftContact1DAugmentedFwdDynamics::set_type,
          "1D mask type");

  bp::register_ptr_to_python<boost::shared_ptr<DADSoftContact1DAugmentedFwdDynamics> >();

  bp::class_<DADSoftContact1DAugmentedFwdDynamics, bp::bases<sobec::DADSoftContactAbstractAugmentedFwdDynamics> >(
      "DADSoftContact1DAugmentedFwdDynamics", "Action data for the soft contact 1D forward dynamics system",
      bp::init<DAMSoftContact1DAugmentedFwdDynamics*>(
          bp::args("self", "model"),
          "Create soft contact 1D forward-dynamics action data.\n\n"
          ":param model: soft contact 1D model"))
      .add_property(
          "aba_df3d",
          bp::make_getter(&sobec::DADSoftContact1DAugmentedFwdDynamics::aba_df3d,
                          bp::return_internal_reference<>()),
          "Partial derivative of joint acceleration w.r.t. 3D contact force")
      .add_property(
          "da_df3d",
          bp::make_getter(&sobec::DADSoftContact1DAugmentedFwdDynamics::da_df3d,
                          bp::return_internal_reference<>()),
          "Partial derivative of LOCAL contact frame acceleration w.r.t. 3D contact force")
      .add_property(
          "f3d",
          bp::make_getter(&sobec::DADSoftContact1DAugmentedFwdDynamics::f3d,
                          bp::return_internal_reference<>()),
          "LOCAL 3D contact force")
      .add_property(
          "fout3d",
          bp::make_getter(&sobec::DADSoftContact1DAugmentedFwdDynamics::f3d,
                          bp::return_internal_reference<>()),
          "Time-derivative of the LOCAL 3D contact force")
      .add_property(
          "dfdt3d_dx",
          bp::make_getter(&sobec::DADSoftContact1DAugmentedFwdDynamics::dfdt3d_dx,
                          bp::return_internal_reference<>()),
          "Partial derivative of the time derivative of the contact force (3D, LOCAL) w.r.t. state")
      .add_property(
          "dfdt3d_du",
          bp::make_getter(&sobec::DADSoftContact1DAugmentedFwdDynamics::dfdt3d_du,
                          bp::return_internal_reference<>()),
          "Partial derivative of the time derivative of the contact force (3D, LOCAL) w.r.t. joint torques")
      .add_property(
          "dfdt3d_df",
          bp::make_getter(&sobec::DADSoftContact1DAugmentedFwdDynamics::dfdt3d_df,
                          bp::return_internal_reference<>()),
          "Partial derivative of the time derivative of the contact force (3D, LOCAL) w.r.t. contact force");
}

}  // namespace python
}  // namespace sobec

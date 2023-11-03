///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, CTU, INRIA,
// University of Oxford Copyright note valid unless otherwise stated in
// individual files. All rights reserved.
///////////////////////////////////////////////////////////////////////////////

// #include "dam-augmented.cpp"
// #include "sobec/crocomplements/softcontact/dam-augmented.hpp"
#include "sobec/crocomplements/softcontact/dam3d-augmented.hpp"

#include <boost/python.hpp>
#include <boost/python/enum.hpp>
#include <eigenpy/eigenpy.hpp>

namespace sobec {
namespace python {
namespace bp = boost::python;

void exposeDAMSoftContact3DAugmentedFwdDyn() {

  bp::register_ptr_to_python<boost::shared_ptr<DAMSoftContact3DAugmentedFwdDynamics>>();

  bp::class_<DAMSoftContact3DAugmentedFwdDynamics, bp::bases<sobec::DAMSoftContactAbstractAugmentedFwdDynamics>>(
      "DAMSoftContact3DAugmentedFwdDynamics", 
      "Differential action model for 3D visco-elastic contact forward dynamics in multibody systems.",
      bp::init<boost::shared_ptr<crocoddyl::StateMultibody>,
               boost::shared_ptr<crocoddyl::ActuationModelAbstract>,
               boost::shared_ptr<crocoddyl::CostModelSum>,
               pinocchio::FrameIndex, Eigen::VectorXd, Eigen::VectorXd, Eigen::Vector3d, pinocchio::ReferenceFrame>(
          bp::args("self", "state", "actuation", "costs", "frameId", "Kp", "Kv", "oPc", "ref"),
          "Initialize the constrained forward-dynamics action model.\n\n"
          ":param state: multibody state\n"
          ":param actuation: actuation model\n"
          ":param costs: stack of cost functions\n"
          ":param frameId: Frame id of the contact model "
          ":param Kp: Stiffness of the visco-elastic contact model "
          ":param Kv: Damping of the visco-elastic contact model "
          ":param oPc: Anchor point of the contact model "
          ":param ref: Pinocchio reference frame of the contact"))
      .def<void (DAMSoftContact3DAugmentedFwdDynamics::*)(
          const boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract>&,
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
          const boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&, 
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &DAMSoftContact3DAugmentedFwdDynamics::calc, bp::args("self", "data", "x", "f"))
      
      .def<void (DAMSoftContact3DAugmentedFwdDynamics::*)(
          const boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract>&,
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
          const boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract>&, 
          const Eigen::Ref<const Eigen::VectorXd>&, 
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &DAMSoftContact3DAugmentedFwdDynamics::calcDiff, bp::args("self", "data", "x", "f"))
      .def("createData", &DAMSoftContact3DAugmentedFwdDynamics::createData,
           bp::args("self"), "Create the Euler integrator data.");

  bp::register_ptr_to_python<boost::shared_ptr<DADSoftContact3DAugmentedFwdDynamics> >();

  bp::class_<DADSoftContact3DAugmentedFwdDynamics, bp::bases<sobec::DADSoftContactAbstractAugmentedFwdDynamics> >(
      "DADSoftContact3DAugmentedFwdDynamics", "Action data for the soft contact 3D forward dynamics system",
      bp::init<DAMSoftContact3DAugmentedFwdDynamics*>(
          bp::args("self", "model"),
          "Create soft contact 3D forward-dynamics action data.\n\n"
          ":param model: soft contact 3D model"));
}

}  // namespace python
}  // namespace sobec

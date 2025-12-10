#include "force_feedback_mpc/force_feedback_mpc_python.hpp"

BOOST_PYTHON_MODULE(force_feedback_mpc_pywrap) { 

    namespace bp = boost::python;

    bp::import("pinocchio");
    bp::import("crocoddyl");

    eigenpy::enableEigenPy();
    eigenpy::enableEigenPySpecific<Eigen::VectorXi>();

    force_feedback_mpc::lpf::exposeStateLPF(); 
    force_feedback_mpc::lpf::exposeIntegratedActionModelLPF(); 

    force_feedback_mpc::softcontact::exposeStateSoftContact(); 
    force_feedback_mpc::softcontact::exposeDAMSoftContactAbstractAugmentedFwdDyn(); 
    force_feedback_mpc::softcontact::exposeDAMSoftContact3DAugmentedFwdDyn(); 
    force_feedback_mpc::softcontact::exposeDAMSoftContact1DAugmentedFwdDyn(); 
    force_feedback_mpc::softcontact::exposeIAMSoftContactAugmented(); 

    force_feedback_mpc::frictioncone::exposeResidualFrictionCone(); 
    force_feedback_mpc::frictioncone::exposeResidualFrictionConeAugmented(); 
}

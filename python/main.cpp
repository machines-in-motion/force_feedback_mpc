#include "force_feedback_mpc/python.hpp"

BOOST_PYTHON_MODULE(force_feedback_mpc_pywrap) { 

    namespace bp = boost::python;

    bp::import("pinocchio");
    bp::import("crocoddyl");

    force_feedback_mpc::lpf::exposeStateLPF(); 
    force_feedback_mpc::lpf::exposeIntegratedActionModelLPF(); 
}

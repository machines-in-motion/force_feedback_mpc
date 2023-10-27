#ifndef __force_feedback_mpc_python__
#define __force_feedback_mpc_python__

#include <pinocchio/multibody/fwd.hpp>  // Must be included first!
#include <boost/python.hpp>

#include "force_feedback_mpc/lowpassfilter/state.hpp"
// #include "force_feedback_mpc/lowpassfilter/action.hpp"


namespace force_feedback_mpc{
    void exposeStateLPF();
    // void exposeIntegratedActionModelLPF();
} // namespace force_feedback_mpc

#endif

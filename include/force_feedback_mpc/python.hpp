#ifndef __force_feedback_mpc_python__
#define __force_feedback_mpc_python__

#include <boost/python.hpp>

#include "force_feedback_mpc/lowpassfilter/state.hpp"
#include "force_feedback_mpc/lowpassfilter/action.hpp"


namespace force_feedback_mpc{
namespace lpf{
    void exposeStateLPF();
    void exposeIntegratedActionModelLPF();
} // namespace lpf
} // namespace force_feedback_mpc

#endif

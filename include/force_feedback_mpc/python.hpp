#ifndef __force_feedback_mpc_python__
#define __force_feedback_mpc_python__

#include <pinocchio/fwd.hpp>
#include <boost/python.hpp>

#include "force_feedback_mpc/lowpassfilter/state.hpp"
#include "force_feedback_mpc/lowpassfilter/action.hpp"

#include "force_feedback_mpc/softcontact/state.hpp"
#include "force_feedback_mpc/softcontact/dam-augmented.hpp"
#include "force_feedback_mpc/softcontact/dam3d-augmented.hpp"

namespace force_feedback_mpc{
namespace lpf{
    void exposeStateLPF();
    void exposeIntegratedActionModelLPF();
} // namespace lpf
} // namespace force_feedback_mpc


namespace force_feedback_mpc{
namespace softcontact{
    void exposeStateSoftContact();
    void exposeDAMSoftContactAbstractAugmentedFwdDyn();
    void exposeDAMSoftContact3DAugmentedFwdDyn();
} // namespace softcontact
} // namespace force_feedback_mpc

#endif

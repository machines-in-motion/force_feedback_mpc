#ifndef __force_feedback_mpc_python__
#define __force_feedback_mpc_python__

#include <pinocchio/fwd.hpp>
#include <boost/python.hpp>
#include <eigenpy/eigenpy.hpp>

#include "force_feedback_mpc/lowpassfilter/state.hpp"
#include "force_feedback_mpc/lowpassfilter/action.hpp"

#include "force_feedback_mpc/softcontact/state.hpp"
#include "force_feedback_mpc/softcontact/dam-augmented.hpp"
#include "force_feedback_mpc/softcontact/dam3d-augmented.hpp"
#include "force_feedback_mpc/softcontact/dam1d-augmented.hpp"

#include "force_feedback_mpc/jerk/state.hpp"
// #include "force_feedback_mpc/jerk/action.hpp"

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
    void exposeDAMSoftContact1DAugmentedFwdDyn();
    void exposeIAMSoftContactAugmented();
} // namespace softcontact
} // namespace force_feedback_mpc


namespace force_feedback_mpc{
namespace jerk{
    void exposeStateJerk();
    // void exposeIntegratedActionModelJerk();
} // namespace jerk
} // namespace force_feedback_mpc


#endif

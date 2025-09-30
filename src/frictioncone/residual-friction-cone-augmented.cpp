///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2023, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "force_feedback_mpc/frictioncone/residual-friction-cone-augmented.hpp"

using namespace crocoddyl;

namespace force_feedback_mpc {
namespace frictioncone {


ResidualModelFrictionConeAugmented::
    ResidualModelFrictionConeAugmented(std::shared_ptr<StateMultibody> state,
                                        const pinocchio::FrameIndex id,
                                        const double coef,
                                        const std::size_t nu)
    : Base(state, 1, nu),
      id_(id),
      coef_(coef) {
  if (static_cast<pinocchio::FrameIndex>(state->get_pinocchio()->nframes) <=
      id) {
    throw_pretty(
        "Invalid argument: "
        << "the frame index is wrong (it does not exist in the robot)");
  }
  active_ = true;
}


ResidualModelFrictionConeAugmented::
    ResidualModelFrictionConeAugmented(std::shared_ptr<StateMultibody> state,
                                        const pinocchio::FrameIndex id,
                                        const double coef)
    : Base(state, 1),
      id_(id),
      coef_(coef){
  if (static_cast<pinocchio::FrameIndex>(state->get_pinocchio()->nframes) <=
      id) {
    throw_pretty(
        "Invalid argument: "
        << "the frame index is wrong (it does not exist in the robot)");
  }
  active_ = true;
}

ResidualModelFrictionConeAugmented::~ResidualModelFrictionConeAugmented() {}


void ResidualModelFrictionConeAugmented::calc(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& f) {
  Data* d = static_cast<Data*>(data.get());
  // Compute the residual of the friction cone
  d->residual = coef_ * f(2) - sqrt(f(0)*f(0) + f(1)*f(1));
}

void ResidualModelFrictionConeAugmented::calcDiff(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& f) {
  Data* d = static_cast<Data*>(data.get());
  d->dcone_df[0] = -f[0] / sqrt(f(0)*f(0) + f(1)*f(1));
  d->dcone_df[1] = -f[1] / sqrt(f(0)*f(0) + f(1)*f(1));
  d->dcone_df[2] = coef_;
}

std::shared_ptr<ResidualDataAbstractTpl<double> >
ResidualModelFrictionConeAugmented::createData(
    DataCollectorAbstract* const data) {
  std::shared_ptr<ResidualDataAbstract> d = std::allocate_shared<Data>(
      Eigen::aligned_allocator<Data>(), this, data);
  return d;
}


void ResidualModelFrictionConeAugmented::print(
    std::ostream& os) const {
  std::shared_ptr<StateMultibody> s =
      std::static_pointer_cast<StateMultibody>(state_);
  os << "ResidualModelFrictionConeAugmented {frame="
     << s->get_pinocchio()->frames[id_].name << ", mu=" << coef_
     << "}";
}



pinocchio::FrameIndex ResidualModelFrictionConeAugmented::get_id()
    const {
  return id_;
}



}  // namespace frictioncone
}  // namespace force_feedback_mpc
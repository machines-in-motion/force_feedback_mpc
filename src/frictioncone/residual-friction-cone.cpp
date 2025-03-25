///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2023, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "force_feedback_mpc/frictioncone/residual-friction-cone.hpp"

using namespace crocoddyl;

namespace force_feedback_mpc {
namespace frictioncone {


ResidualModelFrictionCone::
    ResidualModelFrictionCone(std::shared_ptr<StateMultibody> state,
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
}


ResidualModelFrictionCone::
    ResidualModelFrictionCone(std::shared_ptr<StateMultibody> state,
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
}

ResidualModelFrictionCone::~ResidualModelFrictionCone() {}


void ResidualModelFrictionCone::calc(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, 
    const Eigen::Ref<const VectorXs>& u) {
  Data* d = static_cast<Data*>(data.get());
  // Compute the residual of the friction cone
  d->f3d = d->contact->f.linear();
  data->r[0] = coef_ * d->f3d(2) - sqrt(d->f3d(0)*d->f3d(0) + d->f3d(1)*d->f3d(1));
}


void ResidualModelFrictionCone::calc(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  Data* d = static_cast<Data*>(data.get());
  // Compute the residual of the friction cone
  d->f3d = d->contact->f.linear();
  data->r[0] = coef_ * d->f3d(2) - sqrt(d->f3d(0)*d->f3d(0) + d->f3d(1)*d->f3d(1));

}

void ResidualModelFrictionCone::calcDiff(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, 
    const Eigen::Ref<const VectorXs>& u) {
  Data* d = static_cast<Data*>(data.get());
  d->f3d = d->contact->f.linear();
  d->dcone_df[0, 0] = -d->f3d[0] / sqrt(d->f3d(0)*d->f3d(0) + d->f3d(1)*d->f3d(1));
  d->dcone_df[0, 1] = -d->f3d[1] / sqrt(d->f3d(0)*d->f3d(0) + d->f3d(1)*d->f3d(1));
  d->dcone_df[0, 2] = coef_;
  d->df_dx = d->df_dx.topRows(3);  
  d->df_du = d->df_du.topRows(3);
  data->Rx = d->dcone_df * d->df_dx; 
  data->Ru = d->dcone_df * d->df_du;
//   data->Rx.setZero();
}


void ResidualModelFrictionCone::calcDiff(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  Data* d = static_cast<Data*>(data.get());
  d->f3d = d->contact->f.linear();
  d->dcone_df[0, 0] = -d->f3d[0] / sqrt(d->f3d(0)*d->f3d(0) + d->f3d(1)*d->f3d(1));
  d->dcone_df[0, 1] = -d->f3d[1] / sqrt(d->f3d(0)*d->f3d(0) + d->f3d(1)*d->f3d(1));
  d->dcone_df[0, 2] = coef_;
  d->df_dx = d->df_dx.topRows(3);  
  d->df_du = d->df_du.topRows(3);
  data->Rx = d->dcone_df * d->df_dx; 
  data->Ru = d->dcone_df * d->df_du;
//   data->Rx.setZero();
}

std::shared_ptr<ResidualDataAbstractTpl<double> >
ResidualModelFrictionCone::createData(
    DataCollectorAbstract* const data) {
  std::shared_ptr<ResidualDataAbstract> d = std::allocate_shared<Data>(
      Eigen::aligned_allocator<Data>(), this, data);
  return d;
}

// 
// void ResidualModelFrictionCone::updateJacobians(
//     const std::shared_ptr<ResidualDataAbstract>& data) {
//   Data* d = static_cast<Data*>(data.get());

//   const MatrixXs& df_dx = d->contact->df_dx;
//   const MatrixXs& df_du = d->contact->df_du;
//   const MatrixX3s& A = fref_.get_A();
//   switch (d->contact_type) {
//     case Contact2D: {
//       // Valid for xz plane
//       data->Rx.noalias() = A.col(0) * df_dx.row(0) + A.col(2) * df_dx.row(1);
//       data->Ru.noalias() = A.col(0) * df_du.row(0) + A.col(2) * df_du.row(1);
//       break;
//     }
//     case Contact3D:
//       data->Rx.noalias() = A * df_dx;
//       data->Ru.noalias() = A * df_du;
//       break;
//     case Contact6D:
//       data->Rx.noalias() = A * df_dx.template topRows<3>();
//       data->Ru.noalias() = A * df_du.template topRows<3>();
//       break;
//     default:
//       break;
//   }
//   update_jacobians_ = false;
// }


void ResidualModelFrictionCone::print(
    std::ostream& os) const {
  std::shared_ptr<StateMultibody> s =
      std::static_pointer_cast<StateMultibody>(state_);
  os << "ResidualModelFrictionCone {frame="
     << s->get_pinocchio()->frames[id_].name << ", mu=" << coef_
     << "}";
}

// 
// bool ResidualModelFrictionCone::is_fwddyn() const {
//   return fwddyn_;
// }


pinocchio::FrameIndex ResidualModelFrictionCone::get_id()
    const {
  return id_;
}

// 
// const FrictionConeTpl&
// ResidualModelFrictionCone::get_reference() const {
//   return fref_;
// }


// 
// void ResidualModelFrictionCone::set_reference(
//     const FrictionCone& reference) {
//   fref_ = reference;
//   update_jacobians_ = true;
// }


}  // namespace frictioncone
}  // namespace force_feedback_mpc
namespace force_feedback_mpc {
namespace softcontact {

using namespace crocoddyl;

template <typename Scalar>
ResidualModelForceTrackingTpl<Scalar>::ResidualModelForceTrackingTpl(
    std::shared_ptr<StateMultibody> state, const pinocchio::ReferenceFrame ref,
    const pinocchio::FrameIndex frame_id, const std::size_t nu)
    : Base(state, 3, nu, true, false), pin_model_(*state->get_pinocchio()) {
  if (nu_ == 0) {
    throw_pretty("Invalid argument: "
                 << "it seems to be an autonomous system, if so, don't add "
                    "this residual function");
  }
}

template <typename Scalar>
void ResidualModelForceTrackingTpl<Scalar>::calc(
    const std::shared_ptr<ResidualDataAbstract> &data,
    const Eigen::Ref<const VectorXs> &x, const Eigen::Ref<const VectorXs> &) {
  Data *d = static_cast<Data *>(data.get());

  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> f =
      x.template tail<3>();

  if (active_contact_) {
    if (cost_ref_ != ref_) {
      Eigen::Ref<Matrix3s> oRf = d->pinocchio.oMf[frame_id_].rotation();
      d->r = oRf.transpose() * f - force_des_;
    } else {
      d->r = f - force_des_;
    }
  } else {
    d->r.setZero();
  }
}

template <typename Scalar>
void ResidualModelForceTrackingTpl<Scalar>::calc(
    const std::shared_ptr<ResidualDataAbstract> &data,
    const Eigen::Ref<const VectorXs> &x) {
  calc(data, x, unone_);
}

template <typename Scalar>
void ResidualModelForceTrackingTpl<Scalar>::calcDiff(
    const std::shared_ptr<ResidualDataAbstract> &data,
    const Eigen::Ref<const VectorXs> &x, const Eigen::Ref<const VectorXs> &) {
  Data *d = static_cast<Data *>(data.get());

  if (active_contact_) {
    if (cost_ref_ != ref_) {
      Eigen::Ref<Matrix3s> oRf = d->pinocchio.oMf[frame_id_].rotation();
      if (cost_ref_ == pinocchio::LOCAL) {
        d->Lx.template tail<3>() = d->r.transpose() * oRf.transpose();
      } else {
        d->Lx.template tail<3>() = d->r.transpose() * oRf;
      }
    } else {
      d->Lx.template tail<3>() = d->r.transpose();
    }
  } else {
    d->Lx.template tail<3>().setZero();
  }
}

template <typename Scalar>
void ResidualModelForceTrackingTpl<Scalar>::calcDiff(
    const std::shared_ptr<ResidualDataAbstract> &data,
    const Eigen::Ref<const VectorXs> &x) {
  calcDiff(data, x, unone_);
}

template <typename Scalar>
std::shared_ptr<ResidualDataAbstractTpl<Scalar>>
ResidualModelForceTrackingTpl<Scalar>::createData(
    DataCollectorAbstract *const data) {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                    data);
}

template <typename Scalar>
template <typename NewScalar>
ResidualModelForceTrackingTpl<NewScalar>
ResidualModelForceTrackingTpl<Scalar>::cast() const {
  typedef ResidualModelForceTrackingTpl<NewScalar> ReturnType;
  typedef StateMultibodyTpl<NewScalar> StateType;
  ReturnType ret(
      std::static_pointer_cast<StateType>(state_->template cast<NewScalar>()),
      nu_);
  return ret;
}

template <typename Scalar>
pinocchio::ReferenceFrame
ResidualModelForceTrackingTpl<Scalar>::get_reference() const;

template <typename Scalar>
void ResidualModelForceTrackingTpl<Scalar>::set_reference(
    const pinocchio::ReferenceFrame ref);

template <typename Scalar>
pinocchio::FrameIndex
ResidualModelForceTrackingTpl<Scalar>::get_frame_id() const;

template <typename Scalar>
void ResidualModelForceTrackingTpl<Scalar>::set_frame_id(
    const pinocchio::FrameIndex frame_id);

template <typename Scalar>
bool ResidualModelForceTrackingTpl<Scalar>::get_active_contact() const;

template <typename Scalar>
void ResidualModelForceTrackingTpl<Scalar>::set_active_contact(
    const bool active_contact);

template <typename Scalar>
void ResidualModelForceTrackingTpl<Scalar>::print(std::ostream &os) const {
  os << "ResidualModelForceTracking";
}

} // namespace softcontact
} // namespace force_feedback_mpc
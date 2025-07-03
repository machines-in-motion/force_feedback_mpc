namespace force_feedback_mpc {
namespace softcontact {

using namespace crocoddyl;

template <typename Scalar>
ResidualModelGravityTorqueContactTpl<Scalar>::
    ResidualModelGravityTorqueContactTpl(std::shared_ptr<StateMultibody> state,
                                         const pinocchio::ReferenceFrame ref,
                                         const pinocchio::FrameIndex frame_id,
                                         const std::size_t nu)
    : Base(state, state->get_nv(), nu, true, false),
      pin_model_(*state->get_pinocchio()) {
  if (nu_ == 0) {
    throw_pretty("Invalid argument: "
                 << "it seems to be an autonomous system, if so, don't add "
                    "this residual function");
  }
}

template <typename Scalar>
void ResidualModelGravityTorqueContactTpl<Scalar>::calc(
    const std::shared_ptr<ResidualDataAbstract> &data,
    const Eigen::Ref<const VectorXs> &x, const Eigen::Ref<const VectorXs> &) {
  Data *d = static_cast<Data *>(data.get());

  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
      x.head(state_->get_nq());

  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> f =
      x.template tail<3>();

  if (!active_contact_) {
    data->r = d->actuation->tau -
              pinocchio::computeGeneralizedGravity(pin_model_, d->pinocchio, q);
  } else {
    data->r = -pinocchio::computeStaticTorque(pin_model_, d->pinocchio, q, f);
  }
}

template <typename Scalar>
void ResidualModelGravityTorqueContactTpl<Scalar>::calc(
    const std::shared_ptr<ResidualDataAbstract> &data,
    const Eigen::Ref<const VectorXs> &x) {
  Data *d = static_cast<Data *>(data.get());

  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
      x.head(state_->get_nq());

  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> f =
      x.template tail<3>();

  if (!active_contact_) {
    data->r = d->actuation->tau -
              pinocchio::computeGeneralizedGravity(pin_model_, d->pinocchio, q);
  } else {
    data->r = -pinocchio::computeStaticTorque(pin_model_, d->pinocchio, q, f);
  }
}

template <typename Scalar>
void ResidualModelGravityTorqueContactTpl<Scalar>::calcDiff(
    const std::shared_ptr<ResidualDataAbstract> &data,
    const Eigen::Ref<const VectorXs> &x, const Eigen::Ref<const VectorXs> &) {
  Data *d = static_cast<Data *>(data.get());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
      x.head(state_->get_nq());

  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> f =
      x.template tail<3>();

  Eigen::Block<MatrixXs, Eigen::Dynamic, Eigen::Dynamic, true> Rq =
      data->Rx.leftCols(state_->get_nv());

  // Compute the derivatives of the residual residual
  if (!active_contact_) {
    pinocchio::computeGeneralizedGravityDerivatives(pin_model_, d->pinocchio, q,
                                                    Rq);
    Rq *= Scalar(-1);
  } else {
    pinocchio::computeStaticTorqueDerivatives(pin_model_, d->pinocchio, q, f,
                                              Rq);
    Rq *= Scalar(-1);
    Rq += d->actuation->dtau_dx;
    data->Rx.template rightCols<3>() = d->lJ.template topRows<3>().transpose();
    if (ref_ != pinocchio::LOCAL) {
      Eigen::Ref<Matrix3s> oRf = d->pinocchio.oMf[frame_id_].rotation();
      pinocchio::getFrameJacobian(pin_model_, d->pinocchio, frame_id_,
                                  pinocchio::LOCAL, d->lJ);

      d->tau_grav_residual_f = d->tau_grav_residual_f * d->oRf.transpose();
      Rq += d->lJ.template topRows<3>().transpose() *
            pinocchio::skew(oRf.transpose() * f) *
            d->lJ.template bottomRows<3>();
    }
  }
}

template <typename Scalar>
void ResidualModelGravityTorqueContactTpl<Scalar>::calcDiff(
    const std::shared_ptr<ResidualDataAbstract> &data,
    const Eigen::Ref<const VectorXs> &x) {
  Data *d = static_cast<Data *>(data.get());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
      x.head(state_->get_nq());

  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> f =
      x.template tail<3>();

  Eigen::Block<MatrixXs, Eigen::Dynamic, Eigen::Dynamic, true> Rq =
      data->Rx.leftCols(state_->get_nv());

  // Compute the derivatives of the residual residual
  if (!active_contact_) {
    pinocchio::computeGeneralizedGravityDerivatives(pin_model_, d->pinocchio, q,
                                                    Rq);
  } else {
    pinocchio::computeStaticTorqueDerivatives(pin_model_, d->pinocchio, q, f,
                                              Rq);
  }
  Rq *= Scalar(-1);
}

template <typename Scalar>
std::shared_ptr<ResidualDataAbstractTpl<Scalar>>
ResidualModelGravityTorqueContactTpl<Scalar>::createData(
    DataCollectorAbstract *const data) {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                    data);
}

template <typename Scalar>
template <typename NewScalar>
ResidualModelGravityTorqueContactTpl<NewScalar>
ResidualModelGravityTorqueContactTpl<Scalar>::cast() const {
  typedef ResidualModelGravityTorqueContactTpl<NewScalar> ReturnType;
  typedef StateMultibodyTpl<NewScalar> StateType;
  ReturnType ret(
      std::static_pointer_cast<StateType>(state_->template cast<NewScalar>()),
      nu_);
  return ret;
}

template <typename Scalar>
pinocchio::ReferenceFrame
ResidualModelGravityTorqueContactTpl<Scalar>::get_reference() const;

template <typename Scalar>
void ResidualModelGravityTorqueContactTpl<Scalar>::set_reference(
    const pinocchio::ReferenceFrame ref);

template <typename Scalar>
pinocchio::FrameIndex
ResidualModelGravityTorqueContactTpl<Scalar>::get_frame_id() const;

template <typename Scalar>
void ResidualModelGravityTorqueContactTpl<Scalar>::set_frame_id(
    const pinocchio::FrameIndex frame_id);

template <typename Scalar>
bool ResidualModelGravityTorqueContactTpl<Scalar>::get_active_contact() const;

template <typename Scalar>
void ResidualModelGravityTorqueContactTpl<Scalar>::set_active_contact(
    const bool active_contact);

template <typename Scalar>
void ResidualModelGravityTorqueContactTpl<Scalar>::print(
    std::ostream &os) const {
  os << "ResidualModelGravityTorqueContact";
}

} // namespace softcontact
} // namespace force_feedback_mpc
namespace force_feedback_mpc {
namespace softcontact {

using namespace crocoddyl;

template <typename Scalar>
ResidualModelForceBaseTpl<Scalar>::ResidualModelForceBaseTpl(
    std::shared_ptr<StateAbstract> state, const std::size_t nr,
    const std::size_t nu, const bool q_dependent, const bool v_dependent,
    const bool u_dependent)
    : state_(state), nr_(nr), nu_(nu), unone_(VectorXs::Zero(nu)),
      q_dependent_(q_dependent), v_dependent_(v_dependent),
      u_dependent_(u_dependent) {}

template <typename Scalar>
ResidualModelForceBaseTpl<Scalar>::ResidualModelForceBaseTpl(
    std::shared_ptr<StateAbstract> state, const std::size_t nr,
    const bool q_dependent, const bool v_dependent, const bool u_dependent)
    : state_(state), nr_(nr), nu_(state->get_nv()),
      unone_(VectorXs::Zero(state->get_nv())), q_dependent_(q_dependent),
      v_dependent_(v_dependent), u_dependent_(u_dependent) {}

template <typename Scalar>
void ResidualModelForceBaseTpl<Scalar>::calcCostDiff(
    const std::shared_ptr<CostDataAbstract> &cdata,
    const std::shared_ptr<ResidualDataAbstract> &rdata,
    const std::shared_ptr<ActivationDataAbstract> &adata, const bool update_u) {
  // This function computes the derivatives of the cost function based on a
  // Gauss-Newton approximation
  const bool is_ru = u_dependent_ && nu_ != 0 && update_u;
  const std::size_t nv = state_->get_nv();
  const std::size_t nx = state_->get_nx();
  if (is_ru) {
    cdata->Lu.noalias() = rdata->Ru.transpose() * adata->Ar;
    rdata->Arr_Ru.noalias() = adata->Arr.diagonal().asDiagonal() * rdata->Ru;
    cdata->Luu.noalias() = rdata->Ru.transpose() * rdata->Arr_Ru;
  }
  if (q_dependent_ && v_dependent_) {
    Eigen::Block<MatrixXs, Eigen::Dynamic, Eigen::Dynamic, true> Rx =
        rdata->Rx.leftCols(nx);
    cdata->Lx.head(nx).noalias() = Rx.transpose() * adata->Ar;
    rdata->Arr_Rx.leftCols(nx).noalias() =
        adata->Arr.diagonal().asDiagonal() * Rx;
    cdata->Lxx.topLeftCorner(nx, nx).noalias() =
        Rx.transpose() * rdata->Arr_Rx.leftCols(nx);
    if (is_ru) {
      cdata->Lxu.noalias() = rdata->Rx.transpose() * rdata->Arr_Ru;
    }
  } else if (q_dependent_) {
    Eigen::Block<MatrixXs, Eigen::Dynamic, Eigen::Dynamic, true> Rq =
        rdata->Rx.leftCols(nv);
    cdata->Lx.head(nv).noalias() = Rq.transpose() * adata->Ar;
    rdata->Arr_Rx.leftCols(nv).noalias() =
        adata->Arr.diagonal().asDiagonal() * Rq;
    cdata->Lxx.topLeftCorner(nv, nv).noalias() =
        Rq.transpose() * rdata->Arr_Rx.leftCols(nv);
    if (is_ru) {
      cdata->Lxu.topRows(nv).noalias() = Rq.transpose() * rdata->Arr_Ru;
    }
  } else if (v_dependent_) {
    Eigen::Block<MatrixXs, Eigen::Dynamic, Eigen::Dynamic, true> Rv =
        rdata->Rx.middleCols(nv, nv);
    cdata->Lx.segment(nv, nv).noalias() = Rv.transpose() * adata->Ar;
    rdata->Arr_Rx.middleCols(nv, nv).noalias() =
        adata->Arr.diagonal().asDiagonal() * Rv;
    cdata->Lxx.block(nv, nv, nv, nv).noalias() =
        Rv.transpose() * rdata->Arr_Rx.middleCols(nv, nv);
    if (is_ru) {
      cdata->Lxu.middleRows(nv, nv).noalias() = Rv.transpose() * rdata->Arr_Ru;
    }
  }
  if (f_dependent_) {
    Eigen::Block<MatrixXs, Eigen::Dynamic, Eigen::Dynamic, true> Rf =
        rdata->Rx.rightCols(nf_);
    cdata->Lx.tail(nf_).noalias() = Rf.transpose() * adata->Ar;
    rdata->Arr_Rx.rightCols(nf_).noalias() =
        adata->Arr.diagonal().asDiagonal() * Rf;
    cdata->Lxx.bottomRightCorner(nf_, nf_).noalias() =
        Rf.transpose() * rdata->Arr_Rx.rightCols(nf_);
    if (is_ru) {
      cdata->Lxu.bottomRows(nf_).noalias() = Rv.transpose() * rdata->Arr_Ru;
    }
  }
}

} // namespace softcontact
} // namespace force_feedback_mpc
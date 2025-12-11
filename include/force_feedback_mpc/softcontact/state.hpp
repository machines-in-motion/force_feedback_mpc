///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef FORCE_FEEDBACK_MPC_STATE_SOFT_CONTACT_AUGMENTED_HPP_
#define FORCE_FEEDBACK_MPC_STATE_SOFT_CONTACT_AUGMENTED_HPP_

#include <pinocchio/multibody/model.hpp>
#include "crocoddyl/core/state-base.hpp"

namespace force_feedback_mpc {
namespace softcontact {

class StateSoftContact : public crocoddyl::StateAbstract {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef crocoddyl::MathBaseTpl<double> MathBase;
  typedef crocoddyl::StateAbstractTpl<double> Base;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  enum JointType { FreeFlyer = 0, Spherical, Simple };

  explicit StateSoftContact(std::shared_ptr<pinocchio::ModelTpl<double> > model, std::size_t nc);
  virtual ~StateSoftContact();

  virtual VectorXs zero() const;
  virtual VectorXs rand() const;
  virtual void diff(const Eigen::Ref<const VectorXs>& y0,
                    const Eigen::Ref<const VectorXs>& y1,
                    Eigen::Ref<VectorXs> dyout) const;
  virtual void integrate(const Eigen::Ref<const VectorXs>& y,
                         const Eigen::Ref<const VectorXs>& dy,
                         Eigen::Ref<VectorXs> yout) const;
  virtual void Jdiff(const Eigen::Ref<const VectorXs>&,
                     const Eigen::Ref<const VectorXs>&,
                     Eigen::Ref<MatrixXs> Jfirst, Eigen::Ref<MatrixXs> Jsecond,
                     const crocoddyl::Jcomponent firstsecond = crocoddyl::both) const;

  virtual void Jintegrate(const Eigen::Ref<const VectorXs>& y,
                          const Eigen::Ref<const VectorXs>& dy,
                          Eigen::Ref<MatrixXs> Jfirst,
                          Eigen::Ref<MatrixXs> Jsecond,
                          const crocoddyl::Jcomponent firstsecond = crocoddyl::both,
                          const crocoddyl::AssignmentOp = crocoddyl::setto) const;
  virtual void JintegrateTransport(const Eigen::Ref<const VectorXs>& y,
                                   const Eigen::Ref<const VectorXs>& dy,
                                   Eigen::Ref<MatrixXs> Jin,
                                   const crocoddyl::Jcomponent firstsecond) const;
  virtual std::shared_ptr<crocoddyl::StateBase> cloneAsDouble() const {
    return std::make_shared<StateSoftContact>(*this);
  }
  virtual std::shared_ptr<crocoddyl::StateBase> cloneAsFloat() const {
    return std::make_shared<StateSoftContact>(*this);
  }

  const std::shared_ptr<pinocchio::ModelTpl<double> >& get_pinocchio() const;
  const std::size_t& get_nc() const;
  const std::size_t& get_ny() const;
  const std::size_t& get_ndy() const;

 protected:
  using Base::has_limits_;
  using Base::lb_;
  using Base::ndx_;
  using Base::nq_;
  using Base::nv_;
  using Base::nx_;
  using Base::ub_;
  std::shared_ptr<pinocchio::ModelTpl<double> > pinocchio_;
  std::size_t nc_;
  std::size_t ny_;
  std::size_t ndy_;

 private:
  // std::shared_ptr<pinocchio::ModelTpl<double> > pinocchio_;
  VectorXs y0_;
  JointType joint_type_;
  
};

}  // namespace softcontact
}  // namespace force_feedback_mpc

#endif  // FORCE_FEEDBACK_MPC_STATE_SOFT_CONTACT_AUGMENTED_HPP_

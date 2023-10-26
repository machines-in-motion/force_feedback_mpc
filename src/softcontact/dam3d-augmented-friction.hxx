///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <crocoddyl/core/utils/exception.hpp>
#include <crocoddyl/core/utils/math.hpp>
#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/algorithm/contact-dynamics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>
#include <pinocchio/algorithm/rnea.hpp>

#include "dam3d-augmented-friction.hpp"

namespace sobec {

template <typename Scalar>
DAMSoftContact3DAugmentedFrictionFwdDynamicsTpl<Scalar>::DAMSoftContact3DAugmentedFrictionFwdDynamicsTpl(
    boost::shared_ptr<StateMultibody> state, 
    boost::shared_ptr<ActuationModelAbstract> actuation,
    boost::shared_ptr<CostModelSum> costs,
    const pinocchio::FrameIndex frameId,
    const VectorXs& Kp, 
    const VectorXs& Kv,
    const Vector3s& oPc,
    const pinocchio::ReferenceFrame ref)
    : Base(state, actuation, costs, frameId, Kp, Kv, oPc, 3, ref) {}

template <typename Scalar>
DAMSoftContact3DAugmentedFrictionFwdDynamicsTpl<Scalar>::~DAMSoftContact3DAugmentedFrictionFwdDynamicsTpl() {}

template <typename Scalar>
void DAMSoftContact3DAugmentedFrictionFwdDynamicsTpl<Scalar>::calc(
            const boost::shared_ptr<DifferentialActionDataAbstract>& data, 
            const Eigen::Ref<const VectorXs>& x,
            const Eigen::Ref<const VectorXs>& f,
            const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != this->get_state()->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(this->get_state()->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(f.size()) != 3) {
    throw_pretty("Invalid argument: "
                 << "f has wrong dimension (it should be 3)");
  }
  if (static_cast<std::size_t>(u.size()) != this->get_nu()) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(this->get_nu()) + ")");
  }

  Data* d = static_cast<Data*>(data.get());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q = x.head(this->get_state()->get_nq());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v = x.tail(this->get_state()->get_nv());
  pinocchio::computeAllTerms(this->get_pinocchio(), d->pinocchio, q, v);
  pinocchio::updateFramePlacements(this->get_pinocchio(), d->pinocchio);
  d->oRf = d->pinocchio.oMf[frameId_].rotation();
  // Actuation calc
  this->get_actuation()->calc(d->multibody.actuation, x, u);
  
  // If contact is active, compute aq = ABA(q,v,tau,fext)
  if(active_contact_){
    // Compute external wrench for LOCAL f
    d->f3d = f;
    d->pinForce = pinocchio::ForceTpl<Scalar>(f, Vector3s::Zero());
    // Rotate if not f not in LOCAL
    if(ref_ != pinocchio::LOCAL){
        d->pinForce = pinocchio::ForceTpl<Scalar>(d->oRf.transpose() * f, Vector3s::Zero());
    }
    d->fext[parentId_] = jMf_.act(d->pinForce);

    // ABA with armature
    if(with_armature_){
      d->pinocchio.M.diagonal() += armature_;
      pinocchio::cholesky::decompose(this->get_pinocchio(), d->pinocchio);
      d->Minv.setZero();
      pinocchio::cholesky::computeMinv(this->get_pinocchio(), d->pinocchio, d->Minv);
      d->u_drift = d->multibody.actuation->tau - d->pinocchio.nle;
      //  Compute jacobian transpose lambda
      pinocchio::getFrameJacobian(this->get_pinocchio(), d->pinocchio, frameId_, pinocchio::LOCAL, d->lJ);
      d->xout.noalias() = d->Minv * d->u_drift + d->Minv * d->lJ.topRows(3).transpose() * d->pinForce.linear(); 
     
    // ABA without armature
    } else {
      d->xout = pinocchio::aba(this->get_pinocchio(), d->pinocchio, q, v, d->multibody.actuation->tau, d->fext); 
    }
    // Compute time derivative of contact force : need to forward kin with current acc
    pinocchio::forwardKinematics(this->get_pinocchio(), d->pinocchio, q, v, d->xout);
    d->la = pinocchio::getFrameAcceleration(this->get_pinocchio(), d->pinocchio, frameId_, pinocchio::LOCAL).linear();     
    d->lv = pinocchio::getFrameVelocity(this->get_pinocchio(), d->pinocchio, frameId_, pinocchio::LOCAL).linear();
    d->fout3d = -(Kp_.asDiagonal() * d->lv) - (Kv_.asDiagonal() * d->la);
    d->fout[2] = d->fout3d[2];
    // Dynamic friction on (x,y) tangential components
    d->fout[0] = -mu_*(1./std::sqrt(2))*( sign_smooth(d->lv[0])*d->fout[2] + sign_smooth_diff(d->lv[0])*d->la[0]*d->f3d[2] );
    d->fout[1] = -mu_*(1./std::sqrt(2))*( sign_smooth(d->lv[1])*d->fout[2] + sign_smooth_diff(d->lv[1])*d->la[1]*d->f3d[2] );
    d->fout_copy = d->fout;
    d->fout3d_copy = d->fout3d;
    // Rotate if not f not in LOCAL
    if(ref_ != pinocchio::LOCAL){
        d->oa = pinocchio::getFrameAcceleration(this->get_pinocchio(), d->pinocchio, frameId_, pinocchio::LOCAL_WORLD_ALIGNED).linear();
        d->ov = pinocchio::getFrameVelocity(this->get_pinocchio(), d->pinocchio, frameId_, pinocchio::LOCAL_WORLD_ALIGNED).linear();
        d->fout3d = -(Kp_.asDiagonal() * d->ov) - (Kv_.asDiagonal() * d->oa);
        d->fout[2] = d->fout3d[2];
        // Dynamic friction on (x,y) tangential components
        d->fout[0] = -mu_*(1./std::sqrt(2))*( sign_smooth(d->ov[0])*d->fout[2] + sign_smooth_diff(d->ov[0])*d->oa[0]*d->f3d[2] );
        d->fout[1] = -mu_*(1./std::sqrt(2))*( sign_smooth(d->ov[1])*d->fout[2] + sign_smooth_diff(d->ov[1])*d->oa[1]*d->f3d[2] );
    } 
  }

  // If contact NOT active : compute aq = ABA(q,v,tau)
  else {
    if (with_armature_) {
      // pinocchio::computeAllTerms(this->get_pinocchio(), d->pinocchio, q, v);
      d->pinocchio.M.diagonal() += armature_;
      pinocchio::cholesky::decompose(this->get_pinocchio(), d->pinocchio);
      d->Minv.setZero();
      pinocchio::cholesky::computeMinv(this->get_pinocchio(), d->pinocchio, d->Minv);
      d->u_drift = d->multibody.actuation->tau - d->pinocchio.nle;
      d->xout.noalias() = d->Minv * d->u_drift;
    } else {
      d->xout = pinocchio::aba(this->get_pinocchio(), d->pinocchio, q, v, d->multibody.actuation->tau);
    }
  }

  pinocchio::updateGlobalPlacements(this->get_pinocchio(), d->pinocchio);
  
  // Computing the cost value and residuals
  this->get_costs()->calc(d->costs, x, u);
  d->cost = d->costs->cost;

  // hard coded cost not in contact
  if(!active_contact_){
    if(with_gravity_torque_reg_){
      d->tau_grav_residual = d->multibody.actuation->tau - pinocchio::computeGeneralizedGravity(this->get_pinocchio(), d->pinocchio, q);
      d->cost += 0.5*tau_grav_weight_*d->tau_grav_residual.transpose()*d->tau_grav_residual;
    }
  }

  // Add hard-coded cost in contact
  if(active_contact_){
    if(with_force_cost_){
      if(cost_ref_ != ref_){
        if(cost_ref_ == pinocchio::LOCAL){
          d->f_residual = d->oRf.transpose() * f - force_des_;
        }
        else{
          d->f_residual = d->oRf * f - force_des_;
        }
      }
      else{
        d->f_residual = f - force_des_;
      }
      d->cost += 0.5 * d->f_residual.transpose() * force_weight_.asDiagonal() * d->f_residual;
    }
    if(with_gravity_torque_reg_){
      d->tau_grav_residual = (d->multibody.actuation->tau - pinocchio::computeStaticTorque(this->get_pinocchio(), d->pinocchio, q, d->fext));
      d->cost += 0.5*tau_grav_weight_*d->tau_grav_residual.transpose()*d->tau_grav_residual;
    }
    if(with_force_rate_reg_cost_){
      d->cost += 0.5 * d->fout.transpose() * force_rate_reg_weight_.asDiagonal() * d->fout;  // penalize time derivative of the force 
    }
  }
}


template <typename Scalar>
void DAMSoftContact3DAugmentedFrictionFwdDynamicsTpl<Scalar>::calc(
            const boost::shared_ptr<DifferentialActionDataAbstract>& data, 
            const Eigen::Ref<const VectorXs>& x,
            const Eigen::Ref<const VectorXs>& f) {
  if (static_cast<std::size_t>(x.size()) != this->get_state()->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(this->get_state()->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(f.size()) != 3) {
    throw_pretty("Invalid argument: "
                 << "f has wrong dimension (it should be 3)");
  }
  Data* d = static_cast<Data*>(data.get());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q = x.head(this->get_state()->get_nq());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v = x.tail(this->get_state()->get_nv());
  pinocchio::computeAllTerms(this->get_pinocchio(), d->pinocchio, q, v);
  this->get_costs()->calc(d->costs, x);
  d->cost = d->costs->cost;
  // hard coded cost not in contact
  if(!active_contact_){
    if(with_gravity_torque_reg_){
      d->tau_grav_residual = -pinocchio::computeGeneralizedGravity(this->get_pinocchio(), d->pinocchio, q);
      d->cost += 0.5*tau_grav_weight_*d->tau_grav_residual.transpose()*d->tau_grav_residual;
    }
  }
  else{
    if(with_force_cost_){
      if(cost_ref_ != ref_){
        if(cost_ref_ == pinocchio::LOCAL){
          d->f_residual = d->oRf.transpose() * f - force_des_;
        }
        else{
          d->f_residual = d->oRf * f - force_des_;
        }
      }
      else{
        d->f_residual = f - force_des_;
      }
      d->cost += 0.5 * d->f_residual.transpose() * force_weight_.asDiagonal() * d->f_residual;
    }
    if(with_gravity_torque_reg_){
      d->tau_grav_residual = -pinocchio::computeStaticTorque(this->get_pinocchio(), d->pinocchio, q, d->fext);
      d->cost += 0.5*tau_grav_weight_*d->tau_grav_residual.transpose()*d->tau_grav_residual;
    }
    if(with_force_rate_reg_cost_){
      d->cost += 0.5* d->fout.transpose() * force_rate_reg_weight_.asDiagonal() * d->fout;  // penalize time derivative of the force 
    }
  }
}



template <typename Scalar>
void DAMSoftContact3DAugmentedFrictionFwdDynamicsTpl<Scalar>::calcDiff(
    const boost::shared_ptr<DifferentialActionDataAbstract>& data, 
    const Eigen::Ref<const VectorXs>& x,
    const Eigen::Ref<const VectorXs>& f,
    const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != this->get_state()->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(this->get_state()->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(f.size()) != 3) {
    throw_pretty("Invalid argument: "
                 << "f has wrong dimension (it should be 3)");
  }
  if (static_cast<std::size_t>(u.size()) != this->get_nu()) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(this->get_nu()) + ")");
  }

  const std::size_t nv = this->get_state()->get_nv();
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q = x.head(this->get_state()->get_nq());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v = x.tail(nv);
  Data* d = static_cast<Data*>(data.get());
  d->oRf = d->pinocchio.oMf[frameId_].rotation();
  // Actuation calcDiff
  this->get_actuation()->calcDiff(d->multibody.actuation, x, u);
  
  // If contact is active, compute ABA derivatives + force
  if(active_contact_){
    // Compute Jacobian
    // pinocchio::framesForwardKinematics(this->get_pinocchio(), d->pinocchio, q);
    pinocchio::getFrameJacobian(this->get_pinocchio(), d->pinocchio, frameId_, pinocchio::LOCAL, d->lJ);

    // Derivatives of d->xout (ABA) w.r.t. x and u in LOCAL (same in WORLD)
    // No armature
    if(!with_armature_){
      pinocchio::computeABADerivatives(this->get_pinocchio(), d->pinocchio, q, v, d->multibody.actuation->tau, d->fext, 
                                                                  d->aba_dq, d->aba_dv, d->aba_dtau);
      d->Fx.leftCols(nv) = d->aba_dq;
      d->Fx.rightCols(nv) = d->aba_dv; 
      d->Fx += d->aba_dtau * d->multibody.actuation->dtau_dx;
      d->Fu = d->aba_dtau * d->multibody.actuation->dtau_du;
      // Compute derivatives of d->xout (ABA) w.r.t. f in LOCAL 
      d->aba_df = d->aba_dtau * d->lJ.topRows(3).transpose() * jMf_.rotation() * Matrix3s::Identity();
      // Skew term added to RNEA derivatives when force is expressed in LWA
      if(ref_ != pinocchio::LOCAL){
          d->Fx.leftCols(nv)+= d->aba_dtau * d->lJ.topRows(3).transpose() * pinocchio::skew(d->oRf.transpose() * d->f3d) * d->lJ.bottomRows(3);
          // Rotate dABA/df
          d->aba_df = d->aba_df * d->oRf.transpose();
      }
    // With armature
    } else {
        pinocchio::computeRNEADerivatives(this->get_pinocchio(), d->pinocchio, q, v, d->xout, d->fext);
        d->dtau_dx.leftCols(nv) = d->multibody.actuation->dtau_dx.leftCols(nv) - d->pinocchio.dtau_dq;
        d->dtau_dx.rightCols(nv) = d->multibody.actuation->dtau_dx.rightCols(nv) - d->pinocchio.dtau_dv;
        d->Fx.noalias() = d->Minv * d->dtau_dx;
        d->Fu.noalias() = d->Minv * d->multibody.actuation->dtau_du;
        // Compute derivatives of d->xout (ABA) w.r.t. f in LOCAL 
        d->aba_df = d->Minv * d->lJ.topRows(3).transpose() * jMf_.rotation() * Matrix3s::Identity();
        // Skew term added to RNEA derivatives when force is expressed in LWA
        if(ref_ != pinocchio::LOCAL){
            d->Fx.leftCols(nv)+= d->Minv * d->lJ.topRows(3).transpose() * pinocchio::skew(d->oRf.transpose() * d->f3d) * d->lJ.bottomRows(3);
            // Rotate dABA/df
            d->aba_df = d->aba_df * d->oRf.transpose();
        }
      }

    // Derivatives of d->fout in LOCAL : important >> UPDATE FORWARD KINEMATICS with d->xout
    pinocchio::getFrameVelocityDerivatives(this->get_pinocchio(), d->pinocchio, frameId_, pinocchio::LOCAL, 
                                                    d->lv_dq, d->lv_dv);
    d->lv_dx.leftCols(nv) = d->lv_dq;
    d->lv_dx.rightCols(nv) = d->lv_dv;
    // Derivatives of spatial acc w.r.t. (x, f, u)
    pinocchio::getFrameAccelerationDerivatives(this->get_pinocchio(), d->pinocchio, frameId_, pinocchio::LOCAL, 
                                                    d->v_dv, d->a_dq, d->a_dv, d->a_da);
    d->da_dx.topRows(3).leftCols(nv) = d->a_dq.topRows(3) + d->a_da.topRows(3) * d->Fx.leftCols(nv); 
    d->da_dx.topRows(3).rightCols(nv) = d->a_dv.topRows(3) + d->a_da.topRows(3) * d->Fx.rightCols(nv); 
    d->da_du.topRows(3) = d->a_da.topRows(3) * d->Fu;
    d->da_df.topRows(3) = d->a_da.topRows(3) * d->aba_df;
    // Derivatives of fdot w.r.t. (x,f,u)
    d->dfdt3d_dx = -(Kp_.asDiagonal()*d->lv_dx.topRows(3)) - (Kv_.asDiagonal()*d->da_dx.topRows(3));
    d->dfdt3d_du = -(Kv_.asDiagonal()*d->da_du.topRows(3));
    d->dfdt3d_df = -(Kv_.asDiagonal()*d->da_df.topRows(3));
    d->dfdt3d_dx_copy = d->dfdt3d_dx;
    d->dfdt3d_du_copy = d->dfdt3d_du;
    d->dfdt3d_df_copy = d->dfdt3d_df;
    d->dfdt_dx.row(2) = d->dfdt3d_dx_copy.row(2);
    d->dfdt_du.row(2) = d->dfdt3d_du_copy.row(2);
    d->dfdt_df.row(2) = d->dfdt3d_df_copy.row(2);

    // Derivatives of fdot w.r.t. (x,f,u)
    d->dfdt_dx.row(0) = mu_*d->f3d[2]*(2*eps_tanh_*(1./std::sqrt(2))*sign_smooth_diff(d->lv[0])*sign_smooth(d->lv[0])*d->lv_dx.row(0)*d->la[0] - (1./std::sqrt(2))*sign_smooth_diff(d->lv[0])*d->da_dx.row(0)) 
                      - mu_*(1./std::sqrt(2))*(sign_smooth_diff(d->lv[0])*d->lv_dx.row(0)*d->fout_copy[2] + sign_smooth(d->lv[0])*d->dfdt3d_dx_copy.row(2));
    d->dfdt_dx.row(1) = mu_*d->f3d[2]*(2*eps_tanh_*(1./std::sqrt(2))*sign_smooth_diff(d->lv[1])*sign_smooth(d->lv[1])*d->lv_dx.row(1)*d->la[1] - (1./std::sqrt(2))*sign_smooth_diff(d->lv[1])*d->da_dx.row(1)) 
                      - mu_*(1./std::sqrt(2))*(sign_smooth_diff(d->lv[1])*d->lv_dx.row(1)*d->fout_copy[2] + sign_smooth(d->lv[1])*d->dfdt3d_dx_copy.row(2));
    d->dfdt_du.row(0) = -mu_*(1./std::sqrt(2))*(d->f3d[2]*sign_smooth_diff(d->lv[0])*d->da_du.row(0) + sign_smooth(d->lv[0])*d->dfdt3d_du_copy.row(2));
    d->dfdt_du.row(1) = -mu_*(1./std::sqrt(2))*(d->f3d[2]*sign_smooth_diff(d->lv[1])*d->da_du.row(1) + sign_smooth(d->lv[1])*d->dfdt3d_du_copy.row(2));
    d->dfdt_df.row(0) = -mu_*(1./std::sqrt(2))*(d->f3d[2]*sign_smooth_diff(d->lv[0])*d->da_df.row(0) + sign_smooth(d->lv[0])*d->dfdt3d_df_copy.row(2));
    d->dfdt_df.row(1) = -mu_*(1./std::sqrt(2))*(d->f3d[2]*sign_smooth_diff(d->lv[1])*d->da_df.row(1) + sign_smooth(d->lv[1])*d->dfdt3d_df_copy.row(2));
    d->dfdt_df.row(0)[2] += -mu_*(1./std::sqrt(2))*sign_smooth_diff(d->lv[0])*d->la[0];
    d->dfdt_df.row(1)[2] += -mu_*(1./std::sqrt(2))*sign_smooth_diff(d->lv[1])*d->la[1];

    //Rotate dfout_dx if not LOCAL 
    if(ref_ != pinocchio::LOCAL){
        // Only for z
        pinocchio::getFrameJacobian(this->get_pinocchio(), d->pinocchio, frameId_, pinocchio::LOCAL_WORLD_ALIGNED, d->oJ);
        d->dfdt3d_dx.leftCols(nv) = d->oRf * d->dfdt3d_dx_copy.leftCols(nv)- pinocchio::skew(d->oRf * d->fout3d_copy) * d->oJ.bottomRows(3);
        d->dfdt3d_dx.rightCols(nv) = d->oRf * d->dfdt3d_dx_copy.rightCols(nv);
        d->dfdt3d_du = d->oRf * d->dfdt3d_du_copy;
        d->dfdt3d_df = d->oRf * d->dfdt3d_df_copy;
        d->dfdt_dx.row(2) = d->dfdt3d_dx.row(2);
        d->dfdt_du.row(2) = d->dfdt3d_du.row(2);
        d->dfdt_df.row(2) = d->dfdt3d_df.row(2);
        // For tangential directions x,y we need frame derivatives in LWA
        pinocchio::getFrameVelocityDerivatives(this->get_pinocchio(), d->pinocchio, frameId_, pinocchio::LOCAL_WORLD_ALIGNED, 
                                                        d->lv_dq, d->lv_dv);
        d->ovw = pinocchio::getFrameVelocity(this->get_pinocchio(), d->pinocchio, frameId_, pinocchio::LOCAL_WORLD_ALIGNED).angular();
        d->lv_dx.leftCols(nv) = d->lv_dq;
        d->lv_dx.rightCols(nv) = d->lv_dv;
        d->lv_dx.leftCols(nv).topRows(3) += pinocchio::skew(d->ovw) * d->oJ.topRows(3); // need correction term !
        d->da_dx.topRows(3).leftCols(nv) =  d->oRf * (d->a_dq.topRows(3) + d->a_da.topRows(3) * d->Fx.leftCols(nv)) - pinocchio::skew(d->oa) * d->oJ.bottomRows(3); // need correction term !
        d->da_dx.topRows(3).rightCols(nv) = d->oRf * (d->a_dv.topRows(3) + d->a_da.topRows(3) * d->Fx.rightCols(nv));  
        d->da_du.topRows(3) = d->oRf * (d->a_da.topRows(3) * d->Fu);
        d->da_df.topRows(3) = d->oRf * (d->a_da.topRows(3) * d->aba_df);
        // Derivatives of fdot w.r.t. (x,f,u)
        d->dfdt_dx.row(0) = mu_*d->f3d[2]*(2*eps_tanh_*(1./std::sqrt(2))*sign_smooth_diff(d->ov[0])*sign_smooth(d->ov[0])*d->lv_dx.row(0)*d->oa[0] - (1./std::sqrt(2))*sign_smooth_diff(d->ov[0])*d->da_dx.row(0)) 
                          - mu_*(1./std::sqrt(2))*(sign_smooth_diff(d->ov[0])*d->lv_dx.row(0)*d->fout[2] + sign_smooth(d->ov[0])*d->dfdt3d_dx.row(2));
        d->dfdt_dx.row(1) = mu_*d->f3d[2]*(2*eps_tanh_*(1./std::sqrt(2))*sign_smooth_diff(d->ov[1])*sign_smooth(d->ov[1])*d->lv_dx.row(1)*d->oa[1] - (1./std::sqrt(2))*sign_smooth_diff(d->ov[1])*d->da_dx.row(1)) 
                          - mu_*(1./std::sqrt(2))*(sign_smooth_diff(d->ov[1])*d->lv_dx.row(1)*d->fout[2] + sign_smooth(d->ov[1])*d->dfdt3d_dx.row(2));
        d->dfdt_du.row(0) = -mu_*(1./std::sqrt(2))*(d->f3d[2]*sign_smooth_diff(d->ov[0])*d->da_du.row(0) + sign_smooth(d->ov[0])*d->dfdt3d_du.row(2));
        d->dfdt_du.row(1) = -mu_*(1./std::sqrt(2))*(d->f3d[2]*sign_smooth_diff(d->ov[1])*d->da_du.row(1) + sign_smooth(d->ov[1])*d->dfdt3d_du.row(2));
        d->dfdt_df.row(0) = -mu_*(1./std::sqrt(2))*(d->f3d[2]*sign_smooth_diff(d->ov[0])*d->da_df.row(0) + sign_smooth(d->ov[0])*d->dfdt3d_df.row(2));
        d->dfdt_df.row(1) = -mu_*(1./std::sqrt(2))*(d->f3d[2]*sign_smooth_diff(d->ov[1])*d->da_df.row(1) + sign_smooth(d->ov[1])*d->dfdt3d_df.row(2));
        d->dfdt_df.row(0)[2] += -mu_*(1./std::sqrt(2))*sign_smooth_diff(d->ov[0])*d->oa[0];
        d->dfdt_df.row(1)[2] += -mu_*(1./std::sqrt(2))*sign_smooth_diff(d->ov[1])*d->oa[1];
    }
  }
  else {
    // Computing the dynamics derivatives
    if (!with_armature_) {
      // Computing the free forward dynamics with ABA derivatives
      pinocchio::computeABADerivatives(this->get_pinocchio(), d->pinocchio, q, v, d->multibody.actuation->tau, 
                                                      d->aba_dq, d->aba_dv, d->aba_dtau);
      d->Fx.leftCols(nv) = d->aba_dq;
      d->Fx.rightCols(nv) = d->aba_dv;
      d->pinocchio.Minv = d->aba_dtau;
      d->Fx += d->aba_dtau * d->multibody.actuation->dtau_dx;
      d->Fu = d->aba_dtau * d->multibody.actuation->dtau_du;
    } else {
      pinocchio::computeRNEADerivatives(this->get_pinocchio(), d->pinocchio, q, v, d->xout);
      d->dtau_dx.leftCols(nv) = d->multibody.actuation->dtau_dx.leftCols(nv) - d->pinocchio.dtau_dq;
      d->dtau_dx.rightCols(nv) = d->multibody.actuation->dtau_dx.rightCols(nv) - d->pinocchio.dtau_dv;
      d->Fx.noalias() = d->Minv * d->dtau_dx;
      d->Fu.noalias() = d->Minv * d->multibody.actuation->dtau_du;
    }
  }

  this->get_costs()->calcDiff(d->costs, x, u);
  d->Lx = d->costs->Lx;
  d->Lu = d->costs->Lu;
  d->Lxx = d->costs->Lxx;
  d->Lxu = d->costs->Lxu;
  d->Luu = d->costs->Luu;

  // Add hard-coded gravity reg cost partials (no contact)
  if(!active_contact_){
    if(with_gravity_torque_reg_){
      d->tau_grav_residual = (d->multibody.actuation->tau - pinocchio::computeGeneralizedGravity(this->get_pinocchio(), d->pinocchio, q));
      Eigen::Block<MatrixXs, Eigen::Dynamic, Eigen::Dynamic, false> Rq = d->tau_grav_residual_x.topLeftCorner(nv, nv);
      pinocchio::computeGeneralizedGravityDerivatives(this->get_pinocchio(), d->pinocchio, q, Rq);
      Rq *= -1;
      d->tau_grav_residual_x += d->multibody.actuation->dtau_dx;
      d->tau_grav_residual_u = d->multibody.actuation->dtau_du;
      d->Lx += tau_grav_weight_ * d->tau_grav_residual.transpose() * d->tau_grav_residual_x;
      d->Lu += tau_grav_weight_ * d->tau_grav_residual.transpose() * d->tau_grav_residual_u;
      d->Lxx += tau_grav_weight_ * d->tau_grav_residual_x.transpose() * d->tau_grav_residual_x;
      d->Lxu += tau_grav_weight_ * d->tau_grav_residual_x.transpose() * d->tau_grav_residual_u;
      d->Luu += tau_grav_weight_ * d->tau_grav_residual_u.transpose() * d->tau_grav_residual_u;
    }
  }

  // Add hard-coded costs partials (in contact)
  if(active_contact_){
    if(with_force_cost_){
      if(cost_ref_ != ref_){
        if(cost_ref_ == pinocchio::LOCAL){
          d->f_residual = d->oRf.transpose() * f - force_des_;
          d->Lf = d->f_residual.transpose() * force_weight_.asDiagonal() * d->oRf.transpose();
          Eigen::Block<MatrixXs, Eigen::Dynamic, Eigen::Dynamic, false> Rq = d->f_residual_x.topLeftCorner(nc_, nv);
          Rq = pinocchio::skew(d->oRf.transpose() * f) * d->lJ.bottomRows(3);
          d->Lx += d->f_residual.transpose() * force_weight_.asDiagonal() * d->f_residual_x;
          d->Lff = force_weight_.asDiagonal() * d->oRf * d->oRf.transpose();
        }
        else{
          d->f_residual = d->oRf * f - force_des_;
          d->Lf = d->f_residual.transpose() * force_weight_.asDiagonal() * d->oRf;
          Eigen::Block<MatrixXs, Eigen::Dynamic, Eigen::Dynamic, false> Rq = d->f_residual_x.topLeftCorner(nc_, nv);
          Rq = pinocchio::skew(d->oRf * f) * d->oJ.bottomRows(3);
          d->Lx += d->f_residual.transpose() * force_weight_.asDiagonal() * pinocchio::skew(d->oRf * f) * d->f_residual_x;
          d->Lff = force_weight_.asDiagonal() * d->oRf.transpose() * d->oRf;
        }
      }
      else{
        d->f_residual = f - force_des_;
        d->Lf = d->f_residual.transpose() * force_weight_.asDiagonal();
        d->Lff = force_weight_.asDiagonal() * Matrix3s::Identity();
      }
    }
    if(with_gravity_torque_reg_){
      // Compute residual derivatives w.r.t. x, u and f
      d->tau_grav_residual = (d->multibody.actuation->tau - pinocchio::computeStaticTorque(this->get_pinocchio(), d->pinocchio, q, d->fext));
      Eigen::Block<MatrixXs, Eigen::Dynamic, Eigen::Dynamic, false> Rq = d->tau_grav_residual_x.topLeftCorner(nv, nv);
      pinocchio::computeStaticTorqueDerivatives(this->get_pinocchio(), d->pinocchio, q, d->fext, Rq);
      Rq *= -1;
      d->tau_grav_residual_x += d->multibody.actuation->dtau_dx;
      d->tau_grav_residual_u = d->multibody.actuation->dtau_du;
      d->tau_grav_residual_f = d->lJ.topRows(3).transpose(); 
      if(ref_ != pinocchio::LOCAL){
        d->tau_grav_residual_f = d->tau_grav_residual_f * d->oRf.transpose();  
        d->tau_grav_residual_x.topLeftCorner(nv, nv) += d->lJ.topRows(3).transpose() * pinocchio::skew(d->oRf.transpose() * f) * d->lJ.bottomRows(3);
      }
      // Add cost partials (approx. Hessian with jac^T jac)
      d->Lf += tau_grav_weight_ * d->tau_grav_residual.transpose() * d->tau_grav_residual_f; 
      d->Lff += tau_grav_weight_ * d->tau_grav_residual_f.transpose() * d->tau_grav_residual_f; 
      d->Lx += tau_grav_weight_ * d->tau_grav_residual.transpose() * d->tau_grav_residual_x;
      d->Lu += tau_grav_weight_ * d->tau_grav_residual.transpose() * d->tau_grav_residual_u;
      d->Lxx += tau_grav_weight_ * d->tau_grav_residual_x.transpose() * d->tau_grav_residual_x;
      d->Lxu += tau_grav_weight_ * d->tau_grav_residual_x.transpose() * d->tau_grav_residual_u;
      d->Luu += tau_grav_weight_ * d->tau_grav_residual_u.transpose() * d->tau_grav_residual_u;
    }
    if(with_force_rate_reg_cost_){
      d->Lf += d->fout.transpose() * force_rate_reg_weight_.asDiagonal() * d->dfdt_df ;    
      d->Lff +=  d->dfdt_df.transpose() * force_rate_reg_weight_.asDiagonal() * d->dfdt_df;  
      d->Lx += d->fout.transpose() * force_rate_reg_weight_.asDiagonal() * d->dfdt_dx;
      d->Lxx +=  d->dfdt_dx.transpose() * force_rate_reg_weight_.asDiagonal() * d->dfdt_dx;
      d->Lu += d->fout.transpose() * force_rate_reg_weight_.asDiagonal() * d->dfdt_du;
      d->Luu +=  d->dfdt_du.transpose() * force_rate_reg_weight_.asDiagonal() * d->dfdt_du;
    }
  }
}


template <typename Scalar>
void DAMSoftContact3DAugmentedFrictionFwdDynamicsTpl<Scalar>::calcDiff(
    const boost::shared_ptr<DifferentialActionDataAbstract>& data, 
    const Eigen::Ref<const VectorXs>& x,
    const Eigen::Ref<const VectorXs>& f) {
  if (static_cast<std::size_t>(x.size()) != this->get_state()->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(this->get_state()->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(f.size()) != 3) {
    throw_pretty("Invalid argument: "
                 << "f has wrong dimension (it should be 3)");
  }
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q = x.head(this->get_state()->get_nq());
  const std::size_t nv = this->get_state()->get_nv();
  Data* d = static_cast<Data*>(data.get());
  this->get_costs()->calcDiff(d->costs, x);
  // Add hard-coded costs partials (in contact)
  if(!active_contact_){
    if(with_gravity_torque_reg_){
      d->tau_grav_residual = -pinocchio::computeGeneralizedGravity(this->get_pinocchio(), d->pinocchio, q);
      Eigen::Block<MatrixXs, Eigen::Dynamic, Eigen::Dynamic, false> Rq = d->tau_grav_residual_x.topLeftCorner(nv, nv);
      pinocchio::computeGeneralizedGravityDerivatives(this->get_pinocchio(), d->pinocchio, q, Rq);
      Rq *= -1;
      d->Lx += tau_grav_weight_ * d->tau_grav_residual.transpose() * d->tau_grav_residual_x;
      d->Lxx += tau_grav_weight_ * d->tau_grav_residual_x.transpose() * d->tau_grav_residual_x;
    }
  }
  // Add hard-coded costs partials (in contact) 
  if(active_contact_){
    if(with_force_cost_){
      if(cost_ref_ != ref_){
        if(cost_ref_ == pinocchio::LOCAL){
          d->f_residual = d->oRf.transpose() * f - force_des_;
          d->Lf = d->f_residual.transpose() * force_weight_.asDiagonal() * d->oRf.transpose();
          Eigen::Block<MatrixXs, Eigen::Dynamic, Eigen::Dynamic, false> Rq = d->f_residual_x.topLeftCorner(nc_, nv);
          Rq = pinocchio::skew(d->oRf.transpose() * f) * d->lJ.bottomRows(3);
          d->Lx += d->f_residual.transpose() * force_weight_.asDiagonal() * d->f_residual_x;
          d->Lff = force_weight_.asDiagonal() * d->oRf * d->oRf.transpose();
        }
        else{
          d->f_residual = d->oRf * f - force_des_;
          d->Lf = d->f_residual.transpose() * force_weight_.asDiagonal() * d->oRf;
          Eigen::Block<MatrixXs, Eigen::Dynamic, Eigen::Dynamic, false> Rq = d->f_residual_x.topLeftCorner(nc_, nv);
          Rq = pinocchio::skew(d->oRf * f) * d->oJ.bottomRows(3);
          d->Lx += d->f_residual.transpose() * force_weight_.asDiagonal() * d->f_residual_x;
          d->Lff = force_weight_.asDiagonal() * d->oRf.transpose() * d->oRf;
        }
      }
      else{
        d->f_residual = f - force_des_;
        d->Lf = d->f_residual.transpose() * force_weight_.asDiagonal();
        d->Lff = force_weight_.asDiagonal() * Matrix3s::Identity();
      }
    }
    if(with_gravity_torque_reg_){
      // Compute residual derivatives w.r.t. x and f
      d->tau_grav_residual = -pinocchio::computeStaticTorque(this->get_pinocchio(), d->pinocchio, q, d->fext);
      Eigen::Block<MatrixXs, Eigen::Dynamic, Eigen::Dynamic, false> Rq = d->tau_grav_residual_x.topLeftCorner(nv, nv);
      pinocchio::computeStaticTorqueDerivatives(this->get_pinocchio(), d->pinocchio, q, d->fext, Rq);
      Rq *= -1;
      d->tau_grav_residual_x += d->multibody.actuation->dtau_dx;
      d->tau_grav_residual_f = d->lJ.topRows(3).transpose(); 
      if(ref_ != pinocchio::LOCAL){
        d->tau_grav_residual_f = d->tau_grav_residual_f * d->oRf.transpose();  
        d->tau_grav_residual_x.topLeftCorner(nv, nv) += d->lJ.topRows(3).transpose() * pinocchio::skew(d->oRf.transpose() * f) * d->lJ.bottomRows(3);
      }
      // Add cost partials (approx. Hessian with jac^T jac)
      d->Lf += tau_grav_weight_ * d->tau_grav_residual.transpose() * d->tau_grav_residual_f; 
      d->Lff += tau_grav_weight_ * d->tau_grav_residual_f.transpose() * d->tau_grav_residual_f; 
      d->Lx += tau_grav_weight_ * d->tau_grav_residual.transpose() * d->tau_grav_residual_x;
      d->Lxx += tau_grav_weight_ * d->tau_grav_residual_x.transpose() * d->tau_grav_residual_x;
    }
    if(with_force_rate_reg_cost_){
      d->Lf += d->fout.transpose() * force_rate_reg_weight_.asDiagonal() * d->dfdt_df ;    
      d->Lff += d->dfdt_df.transpose() * force_rate_reg_weight_.asDiagonal() * d->dfdt_df;  
      d->Lx += d->fout.transpose() * force_rate_reg_weight_.asDiagonal() * d->dfdt_dx;
      d->Lxx += d->dfdt_dx.transpose() * force_rate_reg_weight_.asDiagonal() * d->dfdt_dx;
    }
  }
}

template <typename Scalar>
boost::shared_ptr<DifferentialActionDataAbstractTpl<Scalar> >
DAMSoftContact3DAugmentedFrictionFwdDynamicsTpl<Scalar>::createData() {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
}

}  // namespace sobec

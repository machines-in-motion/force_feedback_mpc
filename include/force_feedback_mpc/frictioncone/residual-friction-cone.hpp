///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2024, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef FORCE_FEEDBACK_MPC_FRICTIONCONE_HPP_
#define FORCE_FEEDBACK_MPC_FRICTIONCONE_HPP_

#include <crocoddyl/core/residual-base.hpp>
#include <crocoddyl/core/utils/exception.hpp>
#include <crocoddyl/multibody/contact-base.hpp>
#include <crocoddyl/multibody/contacts/contact-2d.hpp>
#include <crocoddyl/multibody/contacts/contact-3d.hpp>
#include <crocoddyl/multibody/contacts/contact-6d.hpp>
#include <crocoddyl/multibody/contacts/multiple-contacts.hpp>
#include <crocoddyl/multibody/data/contacts.hpp>
#include <crocoddyl/multibody/fwd.hpp>
#include <crocoddyl/multibody/impulse-base.hpp>
#include <crocoddyl/multibody/impulses/impulse-3d.hpp>
#include <crocoddyl/multibody/impulses/impulse-6d.hpp>
#include <crocoddyl/multibody/impulses/multiple-impulses.hpp>
#include <crocoddyl/multibody/states/multibody.hpp>
#include <crocoddyl/multibody/data/impulses.hpp>

namespace force_feedback_mpc {
namespace frictioncone {

enum Vector3MaskType { x = 0, y = 1, z = 2, Last };

struct ResidualDataFrictionCone
    : public crocoddyl::ResidualDataAbstractTpl<double> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef crocoddyl::MathBaseTpl<double> MathBase;
  typedef crocoddyl::ResidualDataAbstractTpl<double> Base;
  typedef crocoddyl::DataCollectorAbstractTpl<double> DataCollectorAbstract;
  typedef crocoddyl::ContactModelMultipleTpl<double> ContactModelMultiple;
  typedef crocoddyl::ImpulseModelMultipleTpl<double> ImpulseModelMultiple;
  typedef crocoddyl::StateMultibodyTpl<double> StateMultibody;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <class Model>
  ResidualDataFrictionCone(Model* const model,
                           DataCollectorAbstract* const data)
      : Base(model, data),
        dcone_df(3),
        df_dx(3, model->get_state()->get_ndx()),
        df_du(3, model->get_nu()) {
    f3d.setZero();
    dcone_df.setZero();
    df_dx.setZero();
    df_du.setZero();
    contact_type = crocoddyl::ContactUndefined;
    // Check that proper shared data has been passed
    bool is_contact = true;
    crocoddyl::DataCollectorContactTpl<double>* d1 =
        dynamic_cast<crocoddyl::DataCollectorContactTpl<double>*>(shared);
    crocoddyl::DataCollectorImpulseTpl<double>* d2 =
        dynamic_cast<crocoddyl::DataCollectorImpulseTpl<double>*>(shared);
    if (d1 == NULL && d2 == NULL) {
      throw_pretty(
          "Invalid argument: the shared data should be derived from "
          "DataCollectorContact or DataCollectorImpulse");
    }
    if (d2 != NULL) {
      is_contact = false;
    }

    // Avoids data casting at runtime
    const pinocchio::FrameIndex id = model->get_id();
    const boost::shared_ptr<StateMultibody>& state =
        boost::static_pointer_cast<StateMultibody>(model->get_state());
    std::string frame_name = state->get_pinocchio()->frames[id].name;
    bool found_contact = false;
    if (is_contact) {
      for (typename ContactModelMultiple::ContactDataContainer::iterator it =
               d1->contacts->contacts.begin();
           it != d1->contacts->contacts.end(); ++it) {
        if (it->second->frame == id) {
          crocoddyl::ContactData2DTpl<double>* d2d =
              dynamic_cast<crocoddyl::ContactData2DTpl<double>*>(it->second.get());
          if (d2d != NULL) {
            contact_type = crocoddyl::Contact2D;
            found_contact = true;
            contact = it->second;
            break;
          }
          crocoddyl::ContactData3DTpl<double>* d3d =
              dynamic_cast<crocoddyl::ContactData3DTpl<double>*>(it->second.get());
          if (d3d != NULL) {
            contact_type = crocoddyl::Contact3D;
            found_contact = true;
            contact = it->second;
            break;
          }
          crocoddyl::ContactData6DTpl<double>* d6d =
              dynamic_cast<crocoddyl::ContactData6DTpl<double>*>(it->second.get());
          if (d6d != NULL) {
            contact_type = crocoddyl::Contact6D;
            found_contact = true;
            contact = it->second;
            break;
          }
          throw_pretty(
              "Domain error: there isn't defined at least a 2d contact for " +
              frame_name);
          break;
        }
      }
    } 
    else {
      for (typename ImpulseModelMultiple::ImpulseDataContainer::iterator it =
               d2->impulses->impulses.begin();
           it != d2->impulses->impulses.end(); ++it) {
        if (it->second->frame == id) {
          crocoddyl::ImpulseData3DTpl<double>* d3d =
              dynamic_cast<crocoddyl::ImpulseData3DTpl<double>*>(it->second.get());
          if (d3d != NULL) {
            contact_type = crocoddyl::Contact3D;
            found_contact = true;
            contact = it->second;
            break;
          }
          crocoddyl::ImpulseData6DTpl<double>* d6d =
              dynamic_cast<crocoddyl::ImpulseData6DTpl<double>*>(it->second.get());
          if (d6d != NULL) {
            contact_type = crocoddyl::Contact6D;
            found_contact = true;
            contact = it->second;
            break;
          }
          throw_pretty(
              "Domain error: there isn't defined at least a 3d contact for " +
              frame_name);
          break;
        }
      }
    }
    if (!found_contact) {
      throw_pretty("Domain error: there isn't defined contact data for " +
                   frame_name);
    }
  }

  boost::shared_ptr<crocoddyl::ForceDataAbstractTpl<double> >
      contact;               //!< Contact force data
  crocoddyl::ContactType contact_type;  //!< Type of contact (2D / 3D / 6D)
  using Base::r;
  using Base::Ru;
  using Base::Rx;
  using Base::shared;
  
  Eigen::Vector3d f3d;
  VectorXs dcone_df;
  MatrixXs df_dx;
  MatrixXs df_du;
};



/**
 * @brief Contact friction cone residual
 *
 * Nonlinear friction cone 3d
 */
class ResidualModelFrictionCone
    : public crocoddyl::ResidualModelAbstractTpl<double> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef crocoddyl::MathBaseTpl<double> MathBase;
  typedef crocoddyl::ResidualModelAbstractTpl<double> Base;
  typedef ResidualDataFrictionCone Data;
  typedef crocoddyl::StateMultibodyTpl<double> StateMultibody;
  typedef crocoddyl::ResidualDataAbstractTpl<double> ResidualDataAbstract;
  typedef crocoddyl::DataCollectorAbstractTpl<double> DataCollectorAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename MathBase::MatrixX3s MatrixX3s;

  /**
   * @brief Initialize the contact friction cone residual model
   *
   * Note that for the inverse-dynamic cases, the control vector contains the
   * generalized accelerations, torques, and all the contact forces.
   *
   * @param[in] state   State of the multibody system
   * @param[in] id      Reference frame id
   * @param[in] fref    Reference friction cone
   * @param[in] nu      Dimension of the control vector
   * @param[in] fwddyn  Indicates that we have a forward dynamics problem (true)
   * or inverse dynamics (false)
   */
  ResidualModelFrictionCone(boost::shared_ptr<StateMultibody> state,
                            const pinocchio::FrameIndex id,
                            const double coef,
                            const std::size_t nu);

  /**
   * @brief Initialize the contact friction cone residual model
   *
   * The default `nu` value is obtained from `StateAbstractTpl::get_nv()`. Note
   * that this constructor can be used for forward-dynamics cases only.
   *
   * @param[in] state  State of the multibody system
   * @param[in] id     Reference frame id
   * @param[in] fref   Reference friction cone
   */
  ResidualModelFrictionCone(boost::shared_ptr<StateMultibody> state,
                                      const pinocchio::FrameIndex id,
                                      const double coef);
  virtual ~ResidualModelFrictionCone();

  /**
   * @brief Compute the contact friction cone residual
   *
   * @param[in] data  Contact friction cone residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const boost::shared_ptr<ResidualDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Compute the residual vector for nodes that depends only on the state
   *
   * It updates the residual vector based on the state only (i.e., it ignores
   * the contact forces). This function is used in the terminal nodes of an
   * optimal control problem.
   *
   * @param[in] data  Residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  virtual void calc(const boost::shared_ptr<ResidualDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Compute the Jacobians of the contact friction cone residual
   *
   * @param[in] data  Contact friction cone residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<ResidualDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Compute the Jacobian of the residual functions with respect to the
   * state only
   *
   * It updates the Jacobian of the residual function based on the state only
   * (i.e., it ignores the contact forces). This function is used in the
   * terminal nodes of an optimal control problem.
   *
   * @param[in] data  Residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<ResidualDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Create the contact friction cone residual data
   */
  virtual boost::shared_ptr<ResidualDataAbstract> createData(
      DataCollectorAbstract* const data);

//   /**
//    * @brief Update the Jacobians of the contact friction cone residual
//    *
//    * @param[in] data  Contact friction cone residual data
//    */
//   void updateJacobians(const boost::shared_ptr<ResidualDataAbstract>& data);

//   /**
//    * @brief Indicates if we are using the forward-dynamics (true) or
//    * inverse-dynamics (false)
//    */
//   bool is_fwddyn() const;

  /**
   * @brief Return the reference frame id
   */
  pinocchio::FrameIndex get_id() const;

  void set_friction_coef(const double inDouble) {coef_ = inDouble; };
  const double& get_friction_coef() const { return coef_; };

  /**
   * @brief Print relevant information of the contact-friction-cone residual
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const;

 protected:
  using Base::nu_;
  using Base::state_;

 private:
//   bool fwddyn_;  //!< Indicates if we are using this function for forward
                 //!< dynamics
//   bool update_jacobians_;     //!< Indicates if we need to update the Jacobians
                              //!< (used for inverse dynamics case)
  pinocchio::FrameIndex id_;  //!< Reference frame id
  double coef_;
//   FrictionCone fref_;         //!< Reference contact friction cone
};


}  // namespace frictioncone
}  // namespace force_feedback_mpc

#endif  // FORCE_FEEDBACK_MPC_FRICTIONCONE_HPP_

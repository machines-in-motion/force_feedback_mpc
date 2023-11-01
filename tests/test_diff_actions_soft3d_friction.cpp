///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, New York University, Max Planck
// Gesellschaft, INRIA, University of Oxford Copyright note valid unless
// otherwise stated in individual files. All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include "common.hpp"
#include "factory/diff-action-soft3d-friction.hpp"
#include <crocoddyl/multibody/residuals/control-gravity.hpp>

using namespace boost::unit_test;
using namespace sobec::unittest;

//----------------------------------------------------------------------------//

void test_check_data(DAMSoftContact3DFrictionTypes::Type action_type,
                     PinocchioReferenceTypes::Type ref_type) {
  // create the model
  DAMSoftContact3DFrictionFactory factory;
  boost::shared_ptr<sobec::DAMSoftContact3DAugmentedFrictionFwdDynamics> model =
      factory.create(action_type, ref_type);
  // Run the print function
  std::ostringstream tmp;
  tmp << *model;
  // create the corresponding data object
  boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract> data =
      model->createData();
  BOOST_CHECK(model->checkData(data));
}

void test_calc_returns_state(DAMSoftContact3DFrictionTypes::Type action_type,
                             PinocchioReferenceTypes::Type ref_type) {
  // create the model
  DAMSoftContact3DFrictionFactory factory;
  boost::shared_ptr<sobec::DAMSoftContact3DAugmentedFrictionFwdDynamics> model = factory.create(action_type, ref_type);
  // create the corresponding data object
  boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract> data = model->createData();
  // Generating random state and control vectors
  const Eigen::VectorXd x = model->get_state()->rand();
  const Eigen::VectorXd f = Eigen::Vector3d::Random();
  const Eigen::VectorXd u = Eigen::VectorXd::Random(model->get_nu());
  // Getting the state dimension from calc() call
  model->calc(data, x, f, u);
  BOOST_CHECK(static_cast<std::size_t>(data->xout.size()) == model->get_state()->get_nv());
}

void test_calc_returns_a_cost(DAMSoftContact3DFrictionTypes::Type action_type,
                              PinocchioReferenceTypes::Type ref_type) {
  // create the model
  DAMSoftContact3DFrictionFactory factory;
  boost::shared_ptr<sobec::DAMSoftContact3DAugmentedFrictionFwdDynamics> model =
      factory.create(action_type, ref_type);
  // create the corresponding data object and set the cost to nan
  boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract> data =
      model->createData();
  data->cost = nan("");
  // Getting the cost value computed by calc()
  const Eigen::VectorXd x = model->get_state()->rand();
  const Eigen::VectorXd f = Eigen::Vector3d::Random();
  const Eigen::VectorXd u = Eigen::VectorXd::Random(model->get_nu());
  model->calc(data, x, f, u);
  // Checking that calc returns a cost value
  BOOST_CHECK(!std::isnan(data->cost));
}

void test_attributes(DAMSoftContact3DFrictionTypes::Type action_type,
                     PinocchioReferenceTypes::Type ref_type) {
  // create the model
  DAMSoftContact3DFrictionFactory factory;
  boost::shared_ptr<sobec::DAMSoftContact3DAugmentedFrictionFwdDynamics> model =
      factory.create(action_type, ref_type);
  double disturbance = std::sqrt(2.0 * std::numeric_limits<double>::epsilon());
  double tol = sqrt(disturbance);
  // Test default values set for test 
    // Contact model
  BOOST_CHECK( (model->get_Kp() - Eigen::VectorXd::Ones(model->get_nc()) * 100).isZero(NUMDIFF_MODIFIER*tol) );
  BOOST_CHECK( (model->get_Kv() - Eigen::VectorXd::Ones(model->get_nc()) * 10 ).isZero(NUMDIFF_MODIFIER*tol) );
  BOOST_CHECK(model->get_oPc().isZero(NUMDIFF_MODIFIER * tol));
  BOOST_CHECK(model->get_active_contact());
    // Gravity torque reg
  BOOST_CHECK(model->get_with_gravity_torque_reg());
  BOOST_CHECK(model->get_tau_grav_weight() - 0.01 <= NUMDIFF_MODIFIER * tol);
    // Force cost
  BOOST_CHECK((model->get_force_weight() - Eigen::Vector3d::Ones()).isZero(NUMDIFF_MODIFIER * tol));
  BOOST_CHECK(model->get_force_des().isZero(NUMDIFF_MODIFIER * tol));
  BOOST_CHECK(model->get_with_force_cost());
  BOOST_CHECK(model->get_cost_ref() == pinocchio::ReferenceFrame::LOCAL);
    // Force rate cost
  BOOST_CHECK((model->get_force_rate_reg_weight() - 1e-6*Eigen::Vector3d::Ones()).isZero(NUMDIFF_MODIFIER * tol));
  BOOST_CHECK(model->get_with_force_rate_reg_cost());

  // Test class default values
  BOOST_CHECK(model->get_nc() == 3 );
  BOOST_CHECK(!model->get_with_armature() );
  BOOST_CHECK(model->get_armature().isZero(NUMDIFF_MODIFIER * tol) );

  // Test setters 
  Eigen::VectorXd Kp = Eigen::VectorXd::Ones(model->get_nc()) * (rand() % 100);
  model->set_Kp(Kp);
  BOOST_CHECK( (model->get_Kp() - Kp).isZero(NUMDIFF_MODIFIER*tol) );

  Eigen::VectorXd Kv = Eigen::VectorXd::Ones(model->get_nc()) * (rand() % 100);
  model->set_Kv(Kv);
  BOOST_CHECK( (model->get_Kv() - Kv).isZero(NUMDIFF_MODIFIER*tol));

  Eigen::Vector3d oPc = Eigen::Vector3d::Random();
  model->set_oPc(oPc);
  BOOST_CHECK( (model->get_oPc() - oPc).isZero( NUMDIFF_MODIFIER * tol ) );

  pinocchio::ReferenceFrame ref = pinocchio::LOCAL ;
  model->set_ref(ref);
  BOOST_CHECK( model->get_ref() == ref);

  pinocchio::ReferenceFrame cost_ref = pinocchio::LOCAL_WORLD_ALIGNED ;
  model->set_cost_ref(cost_ref);
  BOOST_CHECK( model->get_cost_ref() == cost_ref);

  bool active_contact = false;
  model->set_active_contact(active_contact);
  BOOST_CHECK(model->get_active_contact() == active_contact);

  bool with_force_cost = false;
  model->set_with_force_cost(with_force_cost);
  BOOST_CHECK(model->get_with_force_cost() == with_force_cost);

  Eigen::Vector3d force_weight = rand() % 100 * Eigen::Vector3d::Ones();
  model->set_force_weight(force_weight);
  BOOST_CHECK( (model->get_force_weight() - force_weight).isZero(NUMDIFF_MODIFIER*tol) );

  Eigen::Vector3d force_des = Eigen::Vector3d::Random();
  model->set_force_des(force_des);
  BOOST_CHECK( (model->get_force_des() - force_des).isZero( NUMDIFF_MODIFIER * tol ) );
  
  Eigen::Vector3d force_rate_reg_weight = rand() % 100 * Eigen::Vector3d::Ones();
  model->set_force_rate_reg_weight(force_rate_reg_weight);
  BOOST_CHECK( (model->get_force_rate_reg_weight() - force_rate_reg_weight).isZero(NUMDIFF_MODIFIER*tol) );

  bool with_grav_cost = false;
  model->set_with_gravity_torque_reg(with_grav_cost);
  BOOST_CHECK(model->get_with_gravity_torque_reg() == with_grav_cost);

  double tau_grav_weight = rand() % 100;
  model->set_tau_grav_weight(tau_grav_weight);
  BOOST_CHECK(model->get_tau_grav_weight() == tau_grav_weight);

  bool with_armature = true;
  model->set_with_armature(with_armature);
  BOOST_CHECK(model->get_with_armature() == with_armature);

  Eigen::VectorXd armature = Eigen::VectorXd::Random(model->get_state()->get_nv());
  model->set_armature(armature);
  BOOST_CHECK((model->get_armature()-armature).isZero(NUMDIFF_MODIFIER*tol));

  double mu = rand() % 100;
  model->set_mu(mu);
  BOOST_CHECK(model->get_mu() == mu);

  double eps = rand() % 100;
  model->set_eps(eps);
  BOOST_CHECK(model->get_eps() == eps);

  double val = rand() % 100;
  BOOST_CHECK(model->sign_smooth(val) == std::tanh(model->get_eps()*val));
  BOOST_CHECK(model->sign_smooth_diff(val) == model->get_eps()*( 1-std::tanh(model->get_eps()*val)*std::tanh(model->get_eps()*val) ) );
}


// Test partials against numdiff
void test_partials_numdiff(boost::shared_ptr<sobec::DAMSoftContact3DAugmentedFrictionFwdDynamics> model){
  // create the corresponding data object
  boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract> data = model->createData();
  // Generating random values for the state and control
  std::size_t ndx = model->get_state()->get_ndx();
  std::size_t nx = model->get_state()->get_nx();
  std::size_t nc = model->get_nc();
  std::size_t nu = model->get_nu();
  Eigen::VectorXd x = model->get_state()->rand();
  Eigen::VectorXd f = Eigen::VectorXd::Random(nc);
  Eigen::VectorXd u = Eigen::VectorXd::Random(nu);
  // Computing the action derivatives
  model->calc(data, x, f, u);
  model->calcDiff(data, x, f, u);

  // numdiff by hand because ND not adapted to augmented calc and calcDiff
  const Eigen::VectorXd& xn0 = data->xout;
  const Eigen::VectorXd& fn0 = boost::static_pointer_cast<sobec::DADSoftContact3DAugmentedFrictionFwdDynamics>(data)->fout;
  const double c0 = data->cost;
  // perturbations
  Eigen::VectorXd dx = Eigen::VectorXd::Zero(ndx);
  Eigen::VectorXd df = Eigen::VectorXd::Zero(nc);
  Eigen::VectorXd du = Eigen::VectorXd::Zero(nu);
  Eigen::VectorXd xp = Eigen::VectorXd::Zero(nx);
  Eigen::VectorXd fp = Eigen::VectorXd::Zero(nc);
  double disturbance = std::sqrt(2.0 * std::numeric_limits<double>::epsilon());
  // data
  boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract> data_num_diff = model->createData();
  boost::shared_ptr<sobec::DADSoftContact3DAugmentedFrictionFwdDynamics> data_num_diff_cast = boost::static_pointer_cast<sobec::DADSoftContact3DAugmentedFrictionFwdDynamics>(data_num_diff);
  std::vector<boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract>> data_x;
  std::vector<boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract>> data_f;
  std::vector<boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract>> data_u;
  for (std::size_t i = 0; i < ndx; ++i) {
      data_x.push_back(model->createData());
  }
  for (std::size_t i = 0; i < nc; ++i) {
      data_f.push_back(model->createData());
  }
  for (std::size_t i = 0; i < nu; ++i) {
      data_u.push_back(model->createData());
  }

  // Computing the d action(x,f,u) / dx
  for (std::size_t ix = 0; ix < ndx; ++ix) {
    dx(ix) = disturbance;
    model->get_state()->integrate(x, dx, xp);
    model->calc(data_x[ix], xp, f, u);
    boost::shared_ptr<sobec::DADSoftContact3DAugmentedFrictionFwdDynamics> data_ix_cast = boost::static_pointer_cast<sobec::DADSoftContact3DAugmentedFrictionFwdDynamics>(data_x[ix]);
    const Eigen::VectorXd& xn = data_ix_cast->xout;
    const Eigen::VectorXd& fn = data_ix_cast->fout;
    const double c = data_ix_cast->cost;
    data_num_diff_cast->Fx.col(ix) = (xn - xn0) / disturbance;
    data_num_diff_cast->dfdt_dx.col(ix) = (fn - fn0) / disturbance;
    data_num_diff_cast->Lx(ix) = (c - c0) / disturbance;
    dx(ix) = 0.0;
  }

  // Computing the d action(x,f,u) / df
  for (std::size_t idf = 0; idf < nc; ++idf) {
    df(idf) = disturbance;
    model->calc(data_f[idf], x, f + df, u);
    boost::shared_ptr<sobec::DADSoftContact3DAugmentedFrictionFwdDynamics> data_idf_cast = boost::static_pointer_cast<sobec::DADSoftContact3DAugmentedFrictionFwdDynamics>(data_f[idf]);
    const Eigen::VectorXd& xn = data_idf_cast->xout;
    const Eigen::VectorXd& fn = data_idf_cast->fout;
    const double c = data_idf_cast->cost;
    data_num_diff_cast->aba_df.col(idf) = (xn - xn0) / disturbance;
    data_num_diff_cast->dfdt_df.col(idf) = (fn - fn0) / disturbance;
    data_num_diff_cast->Lf(idf) = (c - c0) / disturbance;
    df(idf) = 0.0;
  }

  // Computing the d action(x,f,u) / du
  for (unsigned iu = 0; iu < nu; ++iu) {
    du(iu) = disturbance;
    model->calc(data_u[iu], x, f, u + du);
    boost::shared_ptr<sobec::DADSoftContact3DAugmentedFrictionFwdDynamics> data_iu_cast = boost::static_pointer_cast<sobec::DADSoftContact3DAugmentedFrictionFwdDynamics>(data_u[iu]);
    const Eigen::VectorXd& xn = data_iu_cast->xout;
    const Eigen::VectorXd& fn = data_iu_cast->fout;
    const double c = data_iu_cast->cost;
    data_num_diff_cast->Fu.col(iu) = (xn - xn0) / disturbance;
    data_num_diff_cast->dfdt_du.col(iu) = (fn - fn0) / disturbance;
    data_num_diff_cast->Lu(iu) = (c - c0) / disturbance;
    du(iu) = 0.0;
  }

  // Second-order cost derivatives 

  // Checking the partial derivatives against NumDiff
  boost::shared_ptr<sobec::DADSoftContact3DAugmentedFrictionFwdDynamics> datacast = boost::static_pointer_cast<sobec::DADSoftContact3DAugmentedFrictionFwdDynamics>(data);
  double tol = 1e-3; //sqrt(disturbance);
  BOOST_CHECK((datacast->Fx - data_num_diff_cast->Fx).isZero(NUMDIFF_MODIFIER * tol));
  BOOST_CHECK((datacast->Fu - data_num_diff_cast->Fu).isZero(NUMDIFF_MODIFIER * tol));
  BOOST_CHECK((datacast->aba_df - data_num_diff_cast->aba_df).isZero(NUMDIFF_MODIFIER * tol));
  // BOOST_CHECK((datacast->aba_dtau - data_num_diff_cast->aba_dtau).isZero(NUMDIFF_MODIFIER * tol));
  if((datacast->dfdt_dx - data_num_diff_cast->dfdt_dx).isZero(NUMDIFF_MODIFIER * tol)==false){
    std::cout << "dfdt_dx error  " << std::endl;
    std::cout << datacast->dfdt_dx - data_num_diff_cast->dfdt_dx << std::endl;
  }
  if((datacast->dfdt_du - data_num_diff_cast->dfdt_du).isZero(NUMDIFF_MODIFIER * tol)==false){
    std::cout << "dfdt_du error = " << std::endl;
    std::cout << datacast->dfdt_du - data_num_diff_cast->dfdt_du << std::endl;
  }
  if((datacast->dfdt_df - data_num_diff_cast->dfdt_df).isZero(NUMDIFF_MODIFIER * tol)==false){
    std::cout << "dfdt_df error = " << std::endl;
    std::cout << datacast->dfdt_df - data_num_diff_cast->dfdt_df << std::endl;
  }
  if((datacast->Lx - data_num_diff_cast->Lx).isZero(NUMDIFF_MODIFIER * tol)==false){
    std::cout << "Lx error = " << std::endl;
    std::cout << datacast->Lx - data_num_diff_cast->Lx << std::endl;
  }
  BOOST_CHECK((datacast->dfdt_dx - data_num_diff_cast->dfdt_dx).isZero(NUMDIFF_MODIFIER * tol));
  BOOST_CHECK((datacast->dfdt_df - data_num_diff_cast->dfdt_df).isZero(NUMDIFF_MODIFIER * tol));
  BOOST_CHECK((datacast->dfdt_du - data_num_diff_cast->dfdt_du).isZero(NUMDIFF_MODIFIER * tol));
  BOOST_CHECK((datacast->Lx - data_num_diff_cast->Lx).isZero(NUMDIFF_MODIFIER * tol));
  BOOST_CHECK((datacast->Lu - data_num_diff_cast->Lu).isZero(NUMDIFF_MODIFIER * tol));
  BOOST_CHECK((datacast->Lf - data_num_diff_cast->Lf).isZero(NUMDIFF_MODIFIER * tol));
}

void test_partial_derivatives_against_numdiff(
    DAMSoftContact3DFrictionTypes::Type action_type,
    PinocchioReferenceTypes::Type ref_type) {
  // create the model
  DAMSoftContact3DFrictionFactory factory;
  boost::shared_ptr<sobec::DAMSoftContact3DAugmentedFrictionFwdDynamics> model = factory.create(action_type, ref_type);
  test_partials_numdiff(model);
}

void test_partial_derivatives_against_numdiff_armature(
    DAMSoftContact3DFrictionTypes::Type action_type,
    PinocchioReferenceTypes::Type ref_type) {
  // create the model
  DAMSoftContact3DFrictionFactory factory;
  boost::shared_ptr<sobec::DAMSoftContact3DAugmentedFrictionFwdDynamics> model = factory.create(action_type, ref_type);
  Eigen::VectorXd armature = 1e-3*Eigen::VectorXd::Ones(model->get_state()->get_nv());
  // optional armature
  int nbase = model->get_state()->get_nv() - model->get_nu();
  armature.head(nbase) = Eigen::VectorXd::Zero(nbase);
  model->set_armature(armature);
  model->set_with_armature(true);
  test_partials_numdiff(model);
}




// Test equivalence with free dynamics when Kp,Kv=0
void test_calc_free(boost::shared_ptr<sobec::DAMSoftContact3DAugmentedFrictionFwdDynamics> modelsoft,
                    Eigen::VectorXd armature) {
  // Create DAM free
  boost::shared_ptr<crocoddyl::StateMultibody> statemb = boost::static_pointer_cast<crocoddyl::StateMultibody>(modelsoft->get_state()); 
  boost::shared_ptr<crocoddyl::DifferentialActionModelFreeFwdDynamics> modelfree = boost::make_shared<crocoddyl::DifferentialActionModelFreeFwdDynamics>(
          statemb, modelsoft->get_actuation(), boost::make_shared<crocoddyl::CostModelSum>(*modelsoft->get_costs()));
  // Add gravity cost on free model
  if(modelsoft->get_with_gravity_torque_reg()){
    boost::shared_ptr<crocoddyl::CostModelAbstract> cost = boost::make_shared<crocoddyl::CostModelResidual>(
            statemb, boost::make_shared<crocoddyl::ResidualModelControlGrav>( statemb, modelfree->get_actuation()->get_nu() ));
    modelfree->get_costs()->addCost( "grav_reg", cost, modelsoft->get_tau_grav_weight());
  }
  // optional armature
  if(armature.isZero(1e-9)){
    modelfree->set_armature(Eigen::VectorXd::Zero(modelsoft->get_state()->get_nv()));
    modelsoft->set_armature(Eigen::VectorXd::Zero(modelsoft->get_state()->get_nv()));
    modelsoft->set_with_armature(false);
  } else {
    int nbase = modelsoft->get_state()->get_nv() - modelsoft->get_nu();
    armature.head(nbase) = Eigen::VectorXd::Zero(nbase);
    modelfree->set_armature(armature);
    modelsoft->set_armature(armature);
    modelsoft->set_with_armature(true);
  }
  boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract> datafree = modelfree->createData();
  boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract> datasoft = modelsoft->createData();
  // Generating random state and control vectors
  const Eigen::VectorXd x = modelsoft->get_state()->rand();
  const Eigen::VectorXd f = Eigen::Vector3d::Zero();
  const Eigen::VectorXd u = Eigen::VectorXd::Random(modelsoft->get_nu());
  // Getting the state dimension from calc() call
  modelsoft->calc(datasoft, x, f, u);
  modelfree->calc(datafree, x, u);
  BOOST_CHECK((datasoft->xout - datafree->xout).isZero(1e-4));
}

void test_calc_equivalent_free(DAMSoftContact3DFrictionTypes::Type action_type,
                               PinocchioReferenceTypes::Type ref_type) {
  // create the model
  DAMSoftContact3DFrictionFactory factory;
  boost::shared_ptr<sobec::DAMSoftContact3DAugmentedFrictionFwdDynamics> modelsoft =
      factory.create(action_type, ref_type);
  
  // Set 0 stiffness and damping
  modelsoft->set_Kp(Eigen::VectorXd::Zero(modelsoft->get_nc()));
  modelsoft->set_Kv(Eigen::VectorXd::Zero(modelsoft->get_nc()));
  // Set active contact
  modelsoft->set_active_contact(true);
  // Test that freeFwdDyn = SoftContact(Kp=0;Kv=0), without armature
  test_calc_free(modelsoft, Eigen::VectorXd::Zero(modelsoft->get_state()->get_nv()));
  // Test that freeFwdDyn = SoftContact(Kp=0;Kv=0), WITH armature
  test_calc_free(modelsoft, 1e-3*Eigen::VectorXd::Ones(modelsoft->get_state()->get_nv()));

  boost::shared_ptr<sobec::DAMSoftContact3DAugmentedFrictionFwdDynamics> modelsoft2 =
      factory.create(action_type, ref_type);
  // Set non-zero stiffness and damping
  modelsoft2->set_Kp(Eigen::VectorXd::Ones(modelsoft->get_nc()) * 100.);
  modelsoft2->set_Kv(Eigen::VectorXd::Ones(modelsoft->get_nc()) * 10.);
  // Set inactive contact
  modelsoft2->set_active_contact(false);
  // Test that freeFwdDyn = SoftContact(active=False) without armature
  test_calc_free(modelsoft2, Eigen::VectorXd::Zero(modelsoft2->get_state()->get_nv()));
  // Test that freeFwdDyn = SoftContact(active=False) WITH armature
  test_calc_free(modelsoft2, 1e-3*Eigen::VectorXd::Ones(modelsoft2->get_state()->get_nv()));
}


void test_calcDiff_free(boost::shared_ptr<sobec::DAMSoftContact3DAugmentedFrictionFwdDynamics> modelsoft, 
                        Eigen::VectorXd armature){
  // Create DAM free
  boost::shared_ptr<crocoddyl::StateMultibody> statemb = boost::static_pointer_cast<crocoddyl::StateMultibody>(modelsoft->get_state()); 
  boost::shared_ptr<crocoddyl::DifferentialActionModelFreeFwdDynamics> modelfree =
      boost::make_shared<crocoddyl::DifferentialActionModelFreeFwdDynamics>(
          statemb, modelsoft->get_actuation(), boost::make_shared<crocoddyl::CostModelSum>(*modelsoft->get_costs()));
  // Add gravity cost on free model
  if(modelsoft->get_with_gravity_torque_reg()){
    boost::shared_ptr<crocoddyl::CostModelAbstract> cost = boost::make_shared<crocoddyl::CostModelResidual>(
            statemb, boost::make_shared<crocoddyl::ResidualModelControlGrav>( statemb, modelfree->get_actuation()->get_nu() ));
    modelfree->get_costs()->addCost( "grav_reg", cost, modelsoft->get_tau_grav_weight());
  }
  // optional armature
  if(armature.isZero(1e-9)){
    modelfree->set_armature(Eigen::VectorXd::Zero(modelsoft->get_state()->get_nv()));
    modelsoft->set_armature(Eigen::VectorXd::Zero(modelsoft->get_state()->get_nv()));
    modelsoft->set_with_armature(false);
  } else {
    int nbase = modelsoft->get_state()->get_nv() - modelsoft->get_nu();
    armature.head(nbase) = Eigen::VectorXd::Zero(nbase);
    modelfree->set_armature(armature);
    modelsoft->set_armature(armature);
    modelsoft->set_with_armature(true);
  }
  boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract> datafree = modelfree->createData();
  boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract> datasoft = modelsoft->createData();
  // Generating random state and control vectors
  const Eigen::VectorXd x = modelsoft->get_state()->rand();
  const Eigen::VectorXd f = Eigen::Vector3d::Zero();
  const Eigen::VectorXd u = Eigen::VectorXd::Random(modelsoft->get_nu());
  // Getting the state dimension from calc() call
  modelsoft->calc(datasoft, x, f, u);
  modelsoft->calcDiff(datasoft, x, f, u);
  modelfree->calc(datafree, x, u);
  modelfree->calcDiff(datafree, x, u);
  // Checking the partial derivatives against NumDiff
  double tol = 1e-2;
  BOOST_CHECK((datasoft->Fx - datafree->Fx).isZero(tol));
  BOOST_CHECK((datasoft->Fu - datafree->Fu).isZero(tol));
  BOOST_CHECK((datasoft->Lx - datafree->Lx).isZero(tol));
  BOOST_CHECK((datasoft->Lu - datafree->Lu).isZero(tol));
  BOOST_CHECK((datasoft->Lxx - datafree->Lxx).isZero(tol));
  BOOST_CHECK((datasoft->Lxu - datafree->Lxu).isZero(tol));
  BOOST_CHECK((datasoft->Luu - datafree->Luu).isZero(tol));
  // if(!(datasoft->Fx - datafree->Fx).isZero(tol)){
  //   // std::cout << "Test = " << action_type << "_" << ref_type << std::endl;
  //   // std::cout << " armature free = " << std::endl; 
  //   // std::cout << modelfree->get_armature() << std::endl;
  //   std::cout << " -------------------------------------  " << std::endl;
  //   std::cout << " -------------------------------------  " << std::endl;
  //   std::cout << " armature = " << modelsoft->get_with_armature() << std::endl;
  //   std::cout << " fout = " << boost::static_pointer_cast<sobec::DADSoftContact3DAugmentedFrictionFwdDynamics>(datasoft)->fout << std::endl;
  //   // std::cout << modelsoft->get_with_armature() << std::endl;
  //   // std::cout << " Kp, Kv = " << std::endl;
  //   // std::cout << modelsoft->get_Kp() << ", " << modelsoft->get_Kp() << std::endl;
  //   std::cout << " contact_active = " << std::endl;
  //   std::cout << modelsoft->get_active_contact() << std::endl;
  //   std::cout << " Fq error " << std::endl;
  //   std::cout << datasoft->Fx.leftCols(modelsoft->get_state()->get_nv()) - datafree->Fx.leftCols(modelsoft->get_state()->get_nv())<< std::endl;
  //   std::cout << " -------------------------------------  " << std::endl;
  //   std::cout << " -------------------------------------  " << std::endl;
  //   // std::cout << " Fv error " << std::endl;
  //   // std::cout << datafree->Fx.rightCols(modelsoft->get_state()->get_nv()) - datafree->Fx.rightCols(modelsoft->get_state()->get_nv()) << std::endl;
  // }
}

void test_calcDiff_equivalent_free(DAMSoftContact3DFrictionTypes::Type action_type,
                               PinocchioReferenceTypes::Type ref_type) {
  // create the model
  std::cout << "Test = " << action_type << "_" << ref_type << std::endl;
  DAMSoftContact3DFrictionFactory factory;
  boost::shared_ptr<sobec::DAMSoftContact3DAugmentedFrictionFwdDynamics> modelsoft =
      factory.create(action_type, ref_type);
  // Set 0 stiffness and damping
  modelsoft->set_Kp(Eigen::VectorXd::Zero(modelsoft->get_nc()));
  modelsoft->set_Kv(Eigen::VectorXd::Zero(modelsoft->get_nc()));
  // Set active contact
  modelsoft->set_active_contact(false);
  // Test that freeFwdDyn = SoftContact(Kp=0;Kv=0), without armature
  test_calcDiff_free(modelsoft, Eigen::VectorXd::Zero(modelsoft->get_state()->get_nv()));
  // Test that freeFwdDyn = SoftContact(Kp=0;Kv=0), WITH armature
  test_calcDiff_free(modelsoft, 1e-3*Eigen::VectorXd::Ones(modelsoft->get_state()->get_nv()));
  boost::shared_ptr<sobec::DAMSoftContact3DAugmentedFrictionFwdDynamics> modelsoft2 =
      factory.create(action_type, ref_type);
  // Set non-zero stiffness and damping
  modelsoft2->set_Kp(Eigen::VectorXd::Ones(modelsoft->get_nc()) * 100.);
  modelsoft2->set_Kv(Eigen::VectorXd::Ones(modelsoft->get_nc()) * 10.);
  // // Set inactive contact
  modelsoft2->set_active_contact(false);
  // Test that freeFwdDyn = SoftContact(active=False) without armature
  test_calcDiff_free(modelsoft2, Eigen::VectorXd::Zero(modelsoft2->get_state()->get_nv()));
  // Test that freeFwdDyn = SoftContact(active=False) WITH armature
  test_calcDiff_free(modelsoft2, 1e-3*Eigen::VectorXd::Ones(modelsoft2->get_state()->get_nv()));
}

//----------------------------------------------------------------------------//

void register_action_model_unit_tests(DAMSoftContact3DFrictionTypes::Type action_type,
                                      PinocchioReferenceTypes::Type ref_type = PinocchioReferenceTypes::LOCAL) {
  boost::test_tools::output_test_stream test_name;
  test_name << "test_" << action_type << "_" << ref_type;
  std::cout << "Running " << test_name.str() << std::endl;
  test_suite* ts = BOOST_TEST_SUITE(test_name.str());
  ts->add(BOOST_TEST_CASE(boost::bind(&test_check_data, action_type, ref_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_calc_returns_state, action_type, ref_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_calc_returns_a_cost, action_type, ref_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_attributes, action_type, ref_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_partial_derivatives_against_numdiff, action_type, ref_type)));
  // Test equivalence with Euler for soft contact when Kp, Kv = 0 and f=0, or active_contact = 0
  ts->add(BOOST_TEST_CASE(boost::bind(&test_calc_equivalent_free, action_type, ref_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_calcDiff_equivalent_free, action_type, ref_type)));
  // test partials with armature (Careful : error of size of u_drift and dtau_dx in croco::DAMFree - need change from nu to nv !!!)
  ts->add(BOOST_TEST_CASE(boost::bind(&test_partial_derivatives_against_numdiff_armature, action_type, ref_type)));
  framework::master_test_suite().add(ts);
}

bool init_function() {

  for (size_t i = 0; i < DAMSoftContact3DFrictionTypes::all.size(); ++i) {
    for (size_t j = 0; j < PinocchioReferenceTypes::all.size(); ++j) {
    register_action_model_unit_tests(DAMSoftContact3DFrictionTypes::all[i], PinocchioReferenceTypes::all[j]);
    }
  }

  return true;
}

int main(int argc, char** argv) {
  return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}

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
#include "factory/diff-action.hpp"

using namespace boost::unit_test;
using namespace sobec::unittest;

//----------------------------------------------------------------------------//

void test_check_data(DifferentialActionModelTypes::Type action_type,
                     PinocchioReferenceTypes::Type ref_type,
                     ContactModelMaskTypes::Type mask_type) {
  // create the model
  DifferentialActionModelFactory factory;
  boost::shared_ptr<crocoddyl::DifferentialActionModelAbstract> model =
      factory.create(action_type, ref_type, mask_type);

  // Run the print function
  std::ostringstream tmp;
  tmp << *model;

  // create the corresponding data object
  boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract> data =
      model->createData();

  BOOST_CHECK(model->checkData(data));
}

void test_calc_returns_state(DifferentialActionModelTypes::Type action_type,
                             PinocchioReferenceTypes::Type ref_type,
                             ContactModelMaskTypes::Type mask_type) {
  // create the model
  DifferentialActionModelFactory factory;
  boost::shared_ptr<crocoddyl::DifferentialActionModelAbstract> model =
      factory.create(action_type, ref_type, mask_type);

  // create the corresponding data object
  boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract> data =
      model->createData();

  // Generating random state and control vectors
  const Eigen::VectorXd x = model->get_state()->rand();
  const Eigen::VectorXd u = Eigen::VectorXd::Random(model->get_nu());

  // Getting the state dimension from calc() call
  model->calc(data, x, u);

  BOOST_CHECK(static_cast<std::size_t>(data->xout.size()) ==
              model->get_state()->get_nv());
}

void test_calc_returns_a_cost(DifferentialActionModelTypes::Type action_type,
                              PinocchioReferenceTypes::Type ref_type,
                              ContactModelMaskTypes::Type mask_type) {
  // create the model
  DifferentialActionModelFactory factory;
  boost::shared_ptr<crocoddyl::DifferentialActionModelAbstract> model =
      factory.create(action_type, ref_type, mask_type);

  // create the corresponding data object and set the cost to nan
  boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract> data =
      model->createData();
  data->cost = nan("");

  // Getting the cost value computed by calc()
  const Eigen::VectorXd x = model->get_state()->rand();
  const Eigen::VectorXd u = Eigen::VectorXd::Random(model->get_nu());
  model->calc(data, x, u);

  // Checking that calc returns a cost value
  BOOST_CHECK(!std::isnan(data->cost));
}

void test_quasi_static(DifferentialActionModelTypes::Type action_type,
                       PinocchioReferenceTypes::Type ref_type,
                       ContactModelMaskTypes::Type mask_type) {
  if (action_type ==
      DifferentialActionModelTypes::
          DifferentialActionModelFreeFwdDynamics_TalosArm_Squashed)
    return;
  // create the model
  DifferentialActionModelFactory factory;
  boost::shared_ptr<crocoddyl::DifferentialActionModelAbstract> model =
      factory.create(action_type, ref_type, mask_type);

  // create the corresponding data object and set the cost to nan
  boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract> data =
      model->createData();

  // Getting the cost value computed by calc()
  Eigen::VectorXd x = model->get_state()->rand();
  x.tail(model->get_state()->get_nv()).setZero();
  Eigen::VectorXd u = Eigen::VectorXd::Zero(model->get_nu());
  model->quasiStatic(data, u, x);
  model->calc(data, x, u);

  BOOST_CHECK(data->xout.norm() <= 1e-8);
}

void test_partial_derivatives_against_numdiff(
    DifferentialActionModelTypes::Type action_type,
    PinocchioReferenceTypes::Type ref_type,
    ContactModelMaskTypes::Type mask_type) {
  // create the model
  DifferentialActionModelFactory factory;
  boost::shared_ptr<crocoddyl::DifferentialActionModelAbstract> model =
      factory.create(action_type, ref_type, mask_type);

  // create the corresponding data object and set the cost to nan
  boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract> data =
      model->createData();

  crocoddyl::DifferentialActionModelNumDiff model_num_diff(model);
  boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract> data_num_diff =
      model_num_diff.createData();

  // Generating random values for the state and control
  Eigen::VectorXd x = model->get_state()->rand();
  Eigen::VectorXd u = Eigen::VectorXd::Random(model->get_nu());

  // Computing the action derivatives
  model->calc(data, x, u);
  model->calcDiff(data, x, u);

  model_num_diff.calc(data_num_diff, x, u);
  model_num_diff.calcDiff(data_num_diff, x, u);


  // Checking the partial derivatives against NumDiff
  double tol = 0.01; 
  BOOST_CHECK((data->Fx - data_num_diff->Fx).isZero(tol));
  BOOST_CHECK((data->Fu - data_num_diff->Fu).isZero(tol));
  BOOST_CHECK((data->Lx - data_num_diff->Lx).isZero(tol));
  BOOST_CHECK((data->Lu - data_num_diff->Lu).isZero(tol));
  if (model_num_diff.get_with_gauss_approx()) {
    BOOST_CHECK((data->Lxx - data_num_diff->Lxx).isZero(tol));
    BOOST_CHECK((data->Lxu - data_num_diff->Lxu).isZero(tol));
    BOOST_CHECK((data->Luu - data_num_diff->Luu).isZero(tol));
  } else {
    BOOST_CHECK((data_num_diff->Lxx).isZero(tol));
    BOOST_CHECK((data_num_diff->Lxu).isZero(tol));
    BOOST_CHECK((data_num_diff->Luu).isZero(tol));
  }
  // if(!(data->Lx - data_num_diff->Lx).isZero(tol)){
  //   std::cout << "Test = " << action_type << "_" << ref_type << std::endl;
  //   std::cout << " tol = " << tol << std::endl;
  //   std::cout << " Lx - Lx_nd " << std::endl;
  //   std::cout << data->Lx - data_num_diff->Lx << std::endl;
  // }
  // if(!(data->Lu - data_num_diff->Lu).isZero(tol)){
  //   std::cout << "Test = " << action_type << "_" << ref_type << std::endl;
  //   std::cout << " tol = " << tol << std::endl;
  //   std::cout << " Lu - Lu_nd " << std::endl;
  //   std::cout << data->Lu - data_num_diff->Lu << std::endl;
  // }
}



void test_calc_equivalent_free(DifferentialActionModelTypes::Type action_type,
                               PinocchioReferenceTypes::Type ref_type,
                               ContactModelMaskTypes::Type mask_type) {
  // create the model
  DifferentialActionModelFactory factory;
  boost::shared_ptr<crocoddyl::DifferentialActionModelAbstract> model =
      factory.create(action_type, ref_type, mask_type);
  boost::shared_ptr<sobec::DifferentialActionModelSoftContact3DFwdDynamics> modelsoft = boost::static_pointer_cast<sobec::DifferentialActionModelSoftContact3DFwdDynamics>(model); 
  boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract> data = modelsoft->createData();
  
  // Create DAM free
  boost::shared_ptr<crocoddyl::StateMultibody> statemb = boost::static_pointer_cast<crocoddyl::StateMultibody>(modelsoft->get_state()); 
  boost::shared_ptr<crocoddyl::DifferentialActionModelFreeFwdDynamics> modelfree =
      boost::make_shared<crocoddyl::DifferentialActionModelFreeFwdDynamics>(
          statemb, modelsoft->get_actuation(), modelsoft->get_costs());
  const boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract>& datafree = modelfree->createData();

  // Generating random state and control vectors
  const Eigen::VectorXd x = modelsoft->get_state()->rand();
  const Eigen::VectorXd u = Eigen::VectorXd::Random(modelsoft->get_nu());
  // Set 0 stiffness and damping
  modelsoft->set_Kp(0.);
  modelsoft->set_Kv(0.);

  // Getting the state dimension from calc() call
  modelsoft->calc(data, x, u);
  modelfree->calc(datafree, x, u);

  BOOST_CHECK((data->xout - datafree->xout).norm() <= 1e-8);
  BOOST_CHECK((data->xout - datafree->xout).isZero(1e-6));
}


void test_calcDiff_equivalent_free(DifferentialActionModelTypes::Type action_type,
                               PinocchioReferenceTypes::Type ref_type,
                               ContactModelMaskTypes::Type mask_type) {
  // create the model
  DifferentialActionModelFactory factory;
  boost::shared_ptr<crocoddyl::DifferentialActionModelAbstract> model =
      factory.create(action_type, ref_type, mask_type);
  boost::shared_ptr<sobec::DifferentialActionModelSoftContact3DFwdDynamics> modelsoft = boost::static_pointer_cast<sobec::DifferentialActionModelSoftContact3DFwdDynamics>(model); 
  boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract> data = modelsoft->createData();
  
  // Create DAM free
  boost::shared_ptr<crocoddyl::StateMultibody> statemb = boost::static_pointer_cast<crocoddyl::StateMultibody>(modelsoft->get_state()); 
  boost::shared_ptr<crocoddyl::DifferentialActionModelFreeFwdDynamics> modelfree =
      boost::make_shared<crocoddyl::DifferentialActionModelFreeFwdDynamics>(
          statemb, modelsoft->get_actuation(), modelsoft->get_costs());
  const boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract>& datafree = modelfree->createData();

  // Generating random state and control vectors
  const Eigen::VectorXd x = modelsoft->get_state()->rand();
  const Eigen::VectorXd u = Eigen::VectorXd::Random(modelsoft->get_nu());
  // Set 0 stiffness and damping
  modelsoft->set_Kp(0.);
  modelsoft->set_Kv(0.);

  // Getting the state dimension from calc() call
  modelsoft->calc(data, x, u);
  modelsoft->calcDiff(data, x, u);
  modelfree->calc(datafree, x, u);
  modelfree->calcDiff(datafree, x, u);

  // Checking the partial derivatives against NumDiff
  double tol = 1e-6;
  BOOST_CHECK((data->Fx - datafree->Fx).isZero(tol));
  BOOST_CHECK((data->Fu - datafree->Fu).isZero(tol));
  BOOST_CHECK((data->Lx - datafree->Lx).isZero(tol));
  BOOST_CHECK((data->Lu - datafree->Lu).isZero(tol));
  BOOST_CHECK((data->Lxx - datafree->Lxx).isZero(tol));
  BOOST_CHECK((data->Lxu - datafree->Lxu).isZero(tol));
  BOOST_CHECK((data->Luu - datafree->Luu).isZero(tol));
}



//----------------------------------------------------------------------------//

void register_action_model_unit_tests(
    DifferentialActionModelTypes::Type action_type,
    PinocchioReferenceTypes::Type ref_type = PinocchioReferenceTypes::LOCAL,
    ContactModelMaskTypes::Type mask_type = ContactModelMaskTypes::Z) {
  boost::test_tools::output_test_stream test_name;
  switch (action_type) {
    case DifferentialActionModelTypes::
        DifferentialActionModelContact1DFwdDynamics_TalosArm:
      test_name << "test_" << action_type << "_" << ref_type << "_"
                << mask_type;
      break;
    case DifferentialActionModelTypes::
        DifferentialActionModelContact1DFwdDynamics_HyQ:
      test_name << "test_" << action_type << "_" << ref_type << "_"
                << mask_type;
      break;
    case DifferentialActionModelTypes::
        DifferentialActionModelContact3DFwdDynamics_TalosArm:
      test_name << "test_" << action_type << "_" << ref_type;
      break;
    case DifferentialActionModelTypes::
        DifferentialActionModelContact3DFwdDynamics_HyQ:
      test_name << "test_" << action_type << "_" << ref_type;
      break;
    case DifferentialActionModelTypes::
        DifferentialActionModelContact3DFwdDynamics_Talos:
      test_name << "test_" << action_type << "_" << ref_type;
      break;
    case DifferentialActionModelTypes::
        DifferentialActionModelContact6DFwdDynamics_Talos:
      test_name << "test_" << action_type << "_" << ref_type;
      break;
    case DifferentialActionModelTypes::
        DifferentialActionModelFreeFwdDynamics_TalosArm:
      test_name << "test_" << action_type;
      break;
    case DifferentialActionModelTypes::
        DifferentialActionModelFreeFwdDynamics_TalosArm_Squashed:
      test_name << "test_" << action_type;
      break;
    case DifferentialActionModelTypes::
        DifferentialActionModelSoftContact3DFwdDynamics_TalosArm:
      test_name << "test_" << action_type << "_" << ref_type;
      break;
    case DifferentialActionModelTypes::
        DifferentialActionModelSoftContact3DFwdDynamics_HyQ:
      test_name << "test_" << action_type << "_" << ref_type;
      break;
    case DifferentialActionModelTypes::
        DifferentialActionModelSoftContact1DFwdDynamics_TalosArm:
      test_name << "test_" << action_type << "_" << ref_type << "_" << mask_type;
      break;
    case DifferentialActionModelTypes::
        DifferentialActionModelSoftContact1DFwdDynamics_HyQ:
      test_name << "test_" << action_type << "_" << ref_type << "_" << mask_type;
      break;
    default:
      throw_pretty(__FILE__ ": Wrong DifferentialActionModelTypes::Type given");
      break;
  }
  std::cout << "Running " << test_name.str() << std::endl;
  test_suite* ts = BOOST_TEST_SUITE(test_name.str());
  ts->add(BOOST_TEST_CASE(
      boost::bind(&test_check_data, action_type, ref_type, mask_type)));
  ts->add(BOOST_TEST_CASE(
      boost::bind(&test_calc_returns_state, action_type, ref_type, mask_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_calc_returns_a_cost, action_type,
                                      ref_type, mask_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_partial_derivatives_against_numdiff,
                                      action_type, ref_type, mask_type)));
  // ts->add(BOOST_TEST_CASE(
  //     boost::bind(&test_quasi_static, action_type, ref_type, mask_type)));
  framework::master_test_suite().add(ts);
}

bool init_function() {
  // free (no contact)
  for (size_t i = 0; i < DifferentialActionModelTypes::all.size(); ++i) {
    if (DifferentialActionModelTypes::all[i] ==
            DifferentialActionModelTypes::
                DifferentialActionModelFreeFwdDynamics_TalosArm ||
        DifferentialActionModelTypes::all[i] ==
            DifferentialActionModelTypes::
                DifferentialActionModelFreeFwdDynamics_TalosArm_Squashed) {
      register_action_model_unit_tests(DifferentialActionModelTypes::all[i]);
    }
  }

  // 6D contact
  for (size_t i = 0; i < DifferentialActionModelTypes::all.size(); ++i) {
    if (DifferentialActionModelTypes::all[i] ==
        DifferentialActionModelTypes::
            DifferentialActionModelContact6DFwdDynamics_Talos) {
      for (size_t j = 0; j < PinocchioReferenceTypes::all.size(); ++j) {
        register_action_model_unit_tests(DifferentialActionModelTypes::all[i],
                                         PinocchioReferenceTypes::all[j]);
      }
    }
  }

  // 3D contact
  for (size_t i = 0; i < DifferentialActionModelTypes::all.size(); ++i) {
    if (DifferentialActionModelTypes::all[i] ==
        DifferentialActionModelTypes::DifferentialActionModelContact3DFwdDynamics_TalosArm ||
        DifferentialActionModelTypes::all[i] ==
            DifferentialActionModelTypes::
                DifferentialActionModelContact3DFwdDynamics_HyQ ||
        DifferentialActionModelTypes::all[i] ==
            DifferentialActionModelTypes::
                DifferentialActionModelContact3DFwdDynamics_Talos) {
      for (size_t j = 0; j < PinocchioReferenceTypes::all.size(); ++j) {
        register_action_model_unit_tests(DifferentialActionModelTypes::all[i],
                                         PinocchioReferenceTypes::all[j]);
      }
    }
  }

  // 1D contact (rigid + soft)
  for (size_t i = 0; i < DifferentialActionModelTypes::all.size(); ++i) {
    if (DifferentialActionModelTypes::all[i] ==
        DifferentialActionModelTypes::DifferentialActionModelContact1DFwdDynamics_TalosArm ||
        DifferentialActionModelTypes::all[i] ==
        DifferentialActionModelTypes::DifferentialActionModelContact1DFwdDynamics_HyQ ||
        DifferentialActionModelTypes::all[i] ==
        DifferentialActionModelTypes::DifferentialActionModelSoftContact1DFwdDynamics_TalosArm ||
        DifferentialActionModelTypes::all[i] ==
        DifferentialActionModelTypes::DifferentialActionModelSoftContact1DFwdDynamics_HyQ) {
      for (size_t j = 0; j < PinocchioReferenceTypes::all.size(); ++j) {
        for (size_t k = 0; k < ContactModelMaskTypes::all.size(); ++k) {
          register_action_model_unit_tests(DifferentialActionModelTypes::all[i],
                                           PinocchioReferenceTypes::all[j],
                                           ContactModelMaskTypes::all[k]);
        }
      }
    }
  }

  return true;
}

int main(int argc, char** argv) {
  return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}

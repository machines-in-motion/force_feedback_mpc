///////////////////////////////////////////////////////////////////////////////
// Regression test: soft-contact 1D augmented model on a contact frame whose
// placement w.r.t. its parent joint has a NON-IDENTITY rotation.
//
// The other 1D tests use frames whose jMf rotation is the identity
// (e.g. talos_arm/gripper_left_fingertip_1_link), which makes them blind to any
// spurious frame rotation applied to the force Jacobian in calcDiff: such a term
// cancels when jMf.rotation() == I. On a rotated contact frame it makes the model
// differentiate w.r.t. the wrong contact axis, which is what this test checks.
//
// Two independent checks, for every mask (x,y,z) and reference frame:
//   1. analytical derivatives vs crocoddyl's numerical differentiation;
//   2. the 1D model is the restriction of the 3D model to the masked axis, so its
//      derivatives w.r.t. the force state must equal the corresponding column/entry
//      of the 3D ones (checked at the level of the assembled action derivatives).
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include <crocoddyl/core/numdiff/action.hpp>

#include "common.hpp"
#include "factory/crocoddyl/actuation.hpp"
#include "factory/crocoddyl/state.hpp"
#include "force_feedback_mpc/softcontact/dam1d-augmented.hpp"
#include "force_feedback_mpc/softcontact/dam3d-augmented.hpp"
#include "force_feedback_mpc/softcontact/iam-augmented.hpp"

using namespace boost::unit_test;
using namespace force_feedback_mpc::unittest;

typedef
    typename force_feedback_mpc::softcontact::Vector3MaskType Vector3MaskType;

namespace {

const std::string ROTATED_FRAME = "contact_rotated_frame";
const double TIME_STEP = 1e-3;

// talos_arm + a contact frame rotated w.r.t. its parent joint
struct RotatedFrameFixture {
  std::shared_ptr<crocoddyl::StateMultibody> state;
  std::shared_ptr<crocoddyl::ActuationModelAbstract> actuation;
  std::shared_ptr<crocoddyl::CostModelSum> cost;
  pinocchio::FrameIndex frameId;

  RotatedFrameFixture() {
    state = std::static_pointer_cast<crocoddyl::StateMultibody>(
        StateModelFactory().create(StateModelTypes::StateMultibody_TalosArm));
    actuation = ActuationModelFactory().create(
        ActuationModelTypes::ActuationModelFull,
        StateModelTypes::StateMultibody_TalosArm);
    cost = std::make_shared<crocoddyl::CostModelSum>(state,
                                                     actuation->get_nu());
    // add a frame whose placement rotation is not the identity
    const std::shared_ptr<pinocchio::Model>& pin_model = state->get_pinocchio();
    const pinocchio::FrameIndex ref =
        pin_model->getFrameId("gripper_left_fingertip_1_link");
    const pinocchio::Frame& parent = pin_model->frames[ref];
    // 90 deg about x, then 90 deg about z: a rotation that mixes all three axes
    Eigen::Matrix3d R =
        (Eigen::AngleAxisd(M_PI / 2., Eigen::Vector3d::UnitZ()) *
         Eigen::AngleAxisd(M_PI / 2., Eigen::Vector3d::UnitX()))
            .toRotationMatrix();
    pinocchio::SE3 placement(R, parent.placement.translation());
    BOOST_REQUIRE(!R.isApprox(Eigen::Matrix3d::Identity()));
    frameId = pin_model->addFrame(
        pinocchio::Frame(ROTATED_FRAME, parent.parentJoint, ref, placement,
                         pinocchio::OP_FRAME));
  }
};

std::shared_ptr<force_feedback_mpc::softcontact::IAMSoftContactAugmented>
make_iam_1d(const RotatedFrameFixture& f, pinocchio::ReferenceFrame ref_type,
            Vector3MaskType mask) {
  auto dam = std::make_shared<
      force_feedback_mpc::softcontact::DAMSoftContact1DAugmentedFwdDynamics>(
      f.state, f.actuation, f.cost, f.frameId, Eigen::VectorXd::Ones(1) * 100.,
      Eigen::VectorXd::Ones(1) * 10., Eigen::Vector3d::Zero(), mask);
  dam->set_ref(ref_type);
  return std::make_shared<
      force_feedback_mpc::softcontact::IAMSoftContactAugmented>(dam, TIME_STEP,
                                                                true);
}

std::shared_ptr<force_feedback_mpc::softcontact::IAMSoftContactAugmented>
make_iam_3d(const RotatedFrameFixture& f, pinocchio::ReferenceFrame ref_type) {
  auto dam = std::make_shared<
      force_feedback_mpc::softcontact::DAMSoftContact3DAugmentedFwdDynamics>(
      f.state, f.actuation, f.cost, f.frameId, Eigen::VectorXd::Ones(3) * 100.,
      Eigen::VectorXd::Ones(3) * 10., Eigen::Vector3d::Zero());
  dam->set_ref(ref_type);
  return std::make_shared<
      force_feedback_mpc::softcontact::IAMSoftContactAugmented>(dam, TIME_STEP,
                                                                true);
}

}  // namespace

//----------------------------------------------------------------------------//

// 1. analytical derivatives against numerical differentiation
void test_partial_derivatives_against_numdiff(pinocchio::ReferenceFrame ref_type,
                                              Vector3MaskType mask) {
  RotatedFrameFixture f;
  const auto& model = make_iam_1d(f, ref_type, mask);
  const std::shared_ptr<crocoddyl::ActionDataAbstract>& data =
      model->createData();

  crocoddyl::ActionModelNumDiff model_num_diff(model);
  const std::shared_ptr<crocoddyl::ActionDataAbstract>& data_num_diff =
      model_num_diff.createData();

  Eigen::VectorXd x = model->get_state()->rand();
  const Eigen::VectorXd& u = Eigen::VectorXd::Random(model->get_nu());

  model->calc(data, x, u);
  model->calcDiff(data, x, u);
  model_num_diff.calc(data_num_diff, x, u);
  model_num_diff.calcDiff(data_num_diff, x, u);

  // Tolerance as in the other action tests
  const double tol = std::pow(model_num_diff.get_disturbance(), 1. / 3.);
  BOOST_CHECK((data->Fx - data_num_diff->Fx).isZero(tol));
  BOOST_CHECK((data->Fu - data_num_diff->Fu).isZero(tol));
  BOOST_CHECK((data->Lx - data_num_diff->Lx).isZero(tol));
  BOOST_CHECK((data->Lu - data_num_diff->Lu).isZero(tol));
}

// 2. the 1D model must be the restriction of the 3D model to the masked axis
void test_1d_is_restriction_of_3d(pinocchio::ReferenceFrame ref_type,
                                  Vector3MaskType mask) {
  RotatedFrameFixture f;
  const auto& model1d = make_iam_1d(f, ref_type, mask);
  const auto& model3d = make_iam_3d(f, ref_type);
  const std::shared_ptr<crocoddyl::ActionDataAbstract>& data1d =
      model1d->createData();
  const std::shared_ptr<crocoddyl::ActionDataAbstract>& data3d =
      model3d->createData();

  // same robot state, and a 3d force aligned with the masked axis
  const std::size_t nq = f.state->get_nq(), nv = f.state->get_nv();
  const std::size_t m = static_cast<std::size_t>(mask);
  Eigen::VectorXd x = f.state->rand();
  Eigen::VectorXd u = Eigen::VectorXd::Random(model1d->get_nu());
  const double fm = 12.3;

  Eigen::VectorXd y1 = Eigen::VectorXd::Zero(model1d->get_state()->get_nx());
  y1.head(nq + nv) = x;
  y1.tail(1)(0) = fm;
  Eigen::VectorXd y3 = Eigen::VectorXd::Zero(model3d->get_state()->get_nx());
  y3.head(nq + nv) = x;
  y3.tail(3)(m) = fm;

  model1d->calc(data1d, y1, u);
  model1d->calcDiff(data1d, y1, u);
  model3d->calc(data3d, y3, u);
  model3d->calcDiff(data3d, y3, u);

  const double tol = 1e-9;
  // the dynamics themselves must agree (robot part and masked force component)
  BOOST_CHECK((data1d->xnext.head(nq + nv) - data3d->xnext.head(nq + nv))
                  .isZero(tol));
  BOOST_CHECK(std::abs(data1d->xnext.tail(1)(0) - data3d->xnext.tail(3)(m)) <
              tol);
  // ... hence so must their derivatives w.r.t. the force state:
  //   d(robot state)/df_m  and  d(f_m)/df_m
  const std::size_t ndx1 = model1d->get_state()->get_ndx();
  const std::size_t ndx3 = model3d->get_state()->get_ndx();
  BOOST_CHECK((data1d->Fx.block(0, ndx1 - 1, 2 * nv, 1) -
               data3d->Fx.block(0, ndx3 - 3 + m, 2 * nv, 1))
                  .isZero(tol));
  BOOST_CHECK(std::abs(data1d->Fx(ndx1 - 1, ndx1 - 1) -
                       data3d->Fx(ndx3 - 3 + m, ndx3 - 3 + m)) < tol);
  // and w.r.t. the state and the control
  BOOST_CHECK((data1d->Fx.topLeftCorner(2 * nv, 2 * nv) -
               data3d->Fx.topLeftCorner(2 * nv, 2 * nv))
                  .isZero(tol));
  BOOST_CHECK((data1d->Fu.topRows(2 * nv) - data3d->Fu.topRows(2 * nv))
                  .isZero(tol));
}

//----------------------------------------------------------------------------//

void register_unit_tests(pinocchio::ReferenceFrame ref_type,
                        Vector3MaskType mask) {
  boost::test_tools::output_test_stream test_name;
  test_name << "test_soft1d_rotated_frame_" << ref_type << "_" << mask;
  test_suite* ts = BOOST_TEST_SUITE(test_name.str());
  ts->add(BOOST_TEST_CASE(boost::bind(
      &test_partial_derivatives_against_numdiff, ref_type, mask)));
  ts->add(BOOST_TEST_CASE(
      boost::bind(&test_1d_is_restriction_of_3d, ref_type, mask)));
  framework::master_test_suite().add(ts);
}

bool init_function() {
  for (size_t k = Vector3MaskType::x; k < Vector3MaskType::Last; ++k) {
    register_unit_tests(pinocchio::LOCAL, static_cast<Vector3MaskType>(k));
    register_unit_tests(pinocchio::LOCAL_WORLD_ALIGNED,
                        static_cast<Vector3MaskType>(k));
  }
  return true;
}

int main(int argc, char** argv) {
  return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}

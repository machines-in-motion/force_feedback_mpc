///
/// Benchmark (AuRo revision, reviewer 1 major comment 4): analytical vs numerical
/// derivatives of the augmented soft-contact action model on the iiwa.
///
/// Accuracy : analytical calcDiff vs CENTRAL finite differences (manifold-aware),
///            with a finite-difference step sweep.
/// Timing   : analytical calc+calcDiff, forward finite differences
///            (crocoddyl::ActionModelNumDiff) and central finite differences,
///            per node and over a full MPC horizon at a fixed trajectory.
///
/// This file only benchmarks the production models; it does not modify them.
///
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/model.hpp>

#include <crocoddyl/core/costs/cost-sum.hpp>
#include <crocoddyl/core/costs/residual.hpp>
#include <crocoddyl/core/numdiff/action.hpp>
#include <crocoddyl/core/optctrl/shooting.hpp>
#include <crocoddyl/core/residuals/control.hpp>
#include <crocoddyl/multibody/actuations/full.hpp>
#include <crocoddyl/multibody/residuals/frame-translation.hpp>
#include <crocoddyl/multibody/residuals/state.hpp>
#include <crocoddyl/multibody/states/multibody.hpp>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>

#include "force_feedback_mpc/softcontact/dam1d-augmented.hpp"
#include "force_feedback_mpc/softcontact/dam3d-augmented.hpp"
#include "force_feedback_mpc/softcontact/iam-augmented.hpp"
#include "timings.hpp"

using namespace crocoddyl;
using namespace force_feedback_mpc::softcontact;
typedef Eigen::VectorXd VectorXd;
typedef Eigen::MatrixXd MatrixXd;

// ----------------------------------------------------------------- central FD
// Minimal, benchmark-only central finite differences of an action model.
// Manifold-aware: state perturbations live in the tangent space and the output
// difference is taken with state->diff, exactly like crocoddyl's NumDiff.
struct CentralDiff {
  std::shared_ptr<ActionModelAbstract> model;
  std::shared_ptr<ActionDataAbstract> dp, dm, d0;
  std::size_t ndx, nu, ng;
  double h;
  bool central;  // false: forward differences (same first-order quantities)
  MatrixXd Fx, Fu, Gx, Gu;
  VectorXd Lx, Lu, dx, du, xp, xm, dxnext;

  CentralDiff(std::shared_ptr<ActionModelAbstract> m, double step, bool is_central = true)
      : model(m), dp(m->createData()), dm(m->createData()), d0(m->createData()),
        ndx(m->get_state()->get_ndx()), nu(m->get_nu()), ng(m->get_ng()), h(step),
        central(is_central) {
    Fx.resize(ndx, ndx); Fu.resize(ndx, nu);
    Gx.resize(ng, ndx); Gu.resize(ng, nu);
    Lx.resize(ndx); Lu.resize(nu);
    dx = VectorXd::Zero(ndx); du = VectorXd::Zero(nu);
    xp.resize(m->get_state()->get_nx()); xm.resize(m->get_state()->get_nx());
    dxnext.resize(ndx);
  }

  void calcDiff(const VectorXd& x, const VectorXd& u) {
    const auto& state = model->get_state();
    const double den = central ? 2. * h : h;
    if (!central) model->calc(d0, x, u);  // baseline for forward differences
    for (std::size_t i = 0; i < ndx; ++i) {
      dx(i) = h;
      state->integrate(x, dx, xp);
      model->calc(dp, xp, u);
      if (central) {
        state->integrate(x, -dx, xm);
        model->calc(dm, xm, u);
      }
      const auto& ref = central ? dm : d0;
      state->diff(ref->xnext, dp->xnext, dxnext);
      Fx.col(i) = dxnext / den;
      Lx(i) = (dp->cost - ref->cost) / den;
      if (ng > 0) Gx.col(i) = (dp->g - ref->g) / den;
      dx(i) = 0.;
    }
    for (std::size_t j = 0; j < nu; ++j) {
      du(j) = h;
      model->calc(dp, x, u + du);
      if (central) model->calc(dm, x, u - du);
      const auto& ref = central ? dm : d0;
      state->diff(ref->xnext, dp->xnext, dxnext);
      Fu.col(j) = dxnext / den;
      Lu(j) = (dp->cost - ref->cost) / den;
      if (ng > 0) Gu.col(j) = (dp->g - ref->g) / den;
      du(j) = 0.;
    }
  }
};

// ------------------------------------------------------------------- reporting
struct Err { double max_abs, rel_fro; };

Err compare(const MatrixXd& a, const MatrixXd& b) {
  const double den = std::max(b.norm(), 1e-9);
  return Err{(a - b).cwiseAbs().maxCoeff(), (a - b).norm() / den};
}

struct Stats { double median, mean, p05, p95; std::size_t n; };

Stats stats(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  double sum = 0.; for (double x : v) sum += x;
  auto q = [&](double p) { return v[std::min(v.size() - 1, (std::size_t)(p * v.size())) ]; };
  return Stats{q(0.5), sum / v.size(), q(0.05), q(0.95), v.size()};
}

void print_stats(const std::string& name, const Stats& s) {
  std::cout << "  " << std::left << std::setw(34) << name << std::right << std::fixed
            << std::setprecision(1) << std::setw(10) << s.median << " us (median)"
            << std::setw(10) << s.mean << " (mean)" << std::setw(10) << s.p05 << " (p05)"
            << std::setw(10) << s.p95 << " (p95)   n=" << s.n << std::endl;
}

// --------------------------------------------------------------------- models
struct Setup {
  std::shared_ptr<pinocchio::Model> rmodel;
  std::shared_ptr<StateMultibody> state;
  std::shared_ptr<ActuationModelFull> actuation;
  std::shared_ptr<CostModelSum> costs;
  pinocchio::FrameIndex frameId;
  VectorXd q0;
};

Setup build_iiwa() {
  // iiwa with the FT sensor shell, joint A7 locked: the model used by the
  // polishing / force-tracking experiments (demos/polishing/soft, demos/force_tracking/soft)
  const std::string urdf =
      "/home/skleff/miniconda3/envs/force_feedback/lib/python3.12/site-packages/"
      "mim_robots/robots/kuka/urdf/iiwa_ft_sensor_shell.urdf";
  pinocchio::Model full;
  pinocchio::urdf::buildModel(urdf, full);
  std::vector<pinocchio::JointIndex> locked;
  if (full.existJointName("A7")) locked.push_back(full.getJointId("A7"));
  auto rmodel = std::make_shared<pinocchio::Model>();
  pinocchio::buildReducedModel(full, locked, pinocchio::neutral(full), *rmodel);

  Setup s;
  s.rmodel = rmodel;
  s.state = std::make_shared<StateMultibody>(rmodel);
  s.actuation = std::make_shared<ActuationModelFull>(s.state);
  s.frameId = rmodel->getFrameId("contact");
  // nominal configuration of the polishing experiment (demos/polishing/soft/polishing_soft.yml)
  s.q0 = VectorXd(rmodel->nq);
  s.q0 << 0., 1.0471975511965976, 0., -1.1344640137963142, 0.2, 0.7853981633974483;

  VectorXd x0(s.state->get_nx());
  x0 << s.q0, VectorXd::Zero(rmodel->nv);
  s.costs = std::make_shared<CostModelSum>(s.state, s.actuation->get_nu());
  s.costs->addCost("stateReg",
                   std::make_shared<CostModelResidual>(
                       s.state, std::make_shared<ResidualModelState>(s.state, x0, s.actuation->get_nu())),
                   0.01);
  s.costs->addCost("ctrlReg",
                   std::make_shared<CostModelResidual>(
                       s.state, std::make_shared<ResidualModelControl>(s.state, s.actuation->get_nu())),
                   0.001);
  s.costs->addCost("translation",
                   std::make_shared<CostModelResidual>(
                       s.state, std::make_shared<ResidualModelFrameTranslation>(
                                    s.state, s.frameId, Eigen::Vector3d(0.65, 0., 0.01),
                                    s.actuation->get_nu())),
                   65.);
  return s;
}

std::shared_ptr<IAMSoftContactAugmented> make_iam(const Setup& s, std::size_t nc, double dt) {
  const Eigen::Vector3d oPc(0.65, 0., 0.01);  // contactPosition in the experiment config
  std::shared_ptr<DAMSoftContactAbstractAugmentedFwdDynamics> dam;
  if (nc == 3) {
    auto d = std::make_shared<DAMSoftContact3DAugmentedFwdDynamics>(
        s.state, s.actuation, s.costs, s.frameId, VectorXd::Constant(3, 1000.),
        VectorXd::Constant(3, 100.), oPc);
    d->set_force_des(VectorXd::Zero(3));
    d->set_force_weight(VectorXd::Constant(3, 0.01));
    dam = d;
  } else {
    auto d = std::make_shared<DAMSoftContact1DAugmentedFwdDynamics>(
        s.state, s.actuation, s.costs, s.frameId, VectorXd::Constant(1, 1000.),
        VectorXd::Constant(1, 100.), oPc, Vector3MaskType::z);
    d->set_force_des(VectorXd::Zero(1));
    d->set_force_weight(VectorXd::Constant(1, 0.01));
    dam = d;
  }
  dam->set_with_force_cost(true);
  dam->set_ref(pinocchio::LOCAL_WORLD_ALIGNED);
  return std::make_shared<IAMSoftContactAugmented>(dam, dt, true);
}

// Representative states: the nominal polishing posture, perturbed configurations and
// several contact-force magnitudes (the experiment tracks 50 N along z).
std::vector<std::pair<VectorXd, VectorXd>> make_states(const Setup& s, std::size_t nc,
                                                       std::size_t nx, std::size_t nu) {
  const std::size_t nq = s.rmodel->nq, nv = s.rmodel->nv;
  std::vector<std::pair<VectorXd, VectorXd>> out;
  const std::vector<double> fz = {5., 50., 100., 50.};
  for (std::size_t k = 0; k < fz.size(); ++k) {
    VectorXd x = VectorXd::Zero(nx);
    x.head(nq) = s.q0;
    if (k >= 2) x.head(nq) += VectorXd::Constant(nq, 0.05 * (double)(k - 1));  // other configurations
    if (k == 3) x.segment(nq, nv) = VectorXd::Constant(nv, 0.2);               // moving
    if (nc == 3) {
      x.tail(3) << 0., 0., fz[k];
    } else {
      x.tail(1) << fz[k];
    }
    out.emplace_back(x, VectorXd::Constant(nu, 0.5));
  }
  return out;
}

// ----------------------------------------------------------------------- main
int main() {
  const double dt = 0.003;      // OCP integration step of the polishing experiment
  const std::size_t N = 5;      // horizon length of the polishing experiment
  const std::size_t reps = 2000, horizon_reps = 200;

  for (std::size_t nc : {std::size_t(1), std::size_t(3)}) {
    Setup s = build_iiwa();
    auto iam = make_iam(s, nc, dt);
    auto data = iam->createData();
    const std::size_t nx = iam->get_state()->get_nx(), ndx = iam->get_state()->get_ndx(),
                      nu = iam->get_nu();
    std::cout << "\n=================================================================\n"
              << "iiwa soft-contact " << nc << "D augmented model   nq=" << s.rmodel->nq
              << " nv=" << s.rmodel->nv << " nc=" << nc << " -> nx=" << nx << " ndx=" << ndx
              << " nu=" << nu << " dt=" << dt << " N=" << N
              << "\n=================================================================" << std::endl;

    auto states = make_states(s, nc, nx, nu);

    // ---------------- finite-difference step sweep (accuracy of Fx/Fu)
    std::cout << "\n-- central finite-difference step sweep (state 1: nominal, f_z = 50 N)\n"
              << "     h        max|Fx err|   relFro(Fx)   max|Fu err|   relFro(Fu)" << std::endl;
    const VectorXd& xs0 = states[1].first;
    const VectorXd& us0 = states[1].second;
    iam->calc(data, xs0, us0);
    iam->calcDiff(data, xs0, us0);
    for (double h : {1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7}) {
      CentralDiff cd(iam, h);
      cd.calcDiff(xs0, us0);
      Err ex = compare(data->Fx, cd.Fx), eu = compare(data->Fu, cd.Fu);
      std::cout << std::scientific << std::setprecision(2) << "  " << std::setw(8) << h
                << "   " << std::setw(11) << ex.max_abs << "   " << std::setw(11) << ex.rel_fro
                << "   " << std::setw(11) << eu.max_abs << "   " << std::setw(11) << eu.rel_fro
                << std::endl;
    }

    // ---------------- accuracy at the representative states (selected h)
    const double h_ref = 1e-5;
    std::cout << "\n-- accuracy vs central finite differences (h = " << h_ref << ")\n"
              << "  state   block                      max|err|      relFro" << std::endl;
    for (std::size_t k = 0; k < states.size(); ++k) {
      const VectorXd& x = states[k].first;
      const VectorXd& u = states[k].second;
      iam->calc(data, x, u);
      iam->calcDiff(data, x, u);
      CentralDiff cd(iam, h_ref);
      cd.calcDiff(x, u);
      const std::size_t nr = ndx - nc;  // robot-state rows; the last nc rows are the force state
      struct Blk { const char* name; MatrixXd a, b; };
      std::vector<Blk> blks = {
          {"Fx (all)", data->Fx, cd.Fx},
          {"Fx robot-state rows", data->Fx.topRows(nr), cd.Fx.topRows(nr)},
          {"Fx force-state rows", data->Fx.bottomRows(nc), cd.Fx.bottomRows(nc)},
          {"  dlam_next/dq", data->Fx.bottomLeftCorner(nc, s.rmodel->nv),
           cd.Fx.bottomLeftCorner(nc, s.rmodel->nv)},
          {"  dlam_next/dv", data->Fx.block(nr, s.rmodel->nv, nc, s.rmodel->nv),
           cd.Fx.block(nr, s.rmodel->nv, nc, s.rmodel->nv)},
          {"  dlam_next/dlam", data->Fx.bottomRightCorner(nc, nc), cd.Fx.bottomRightCorner(nc, nc)},
          {"Fu (all)", data->Fu, cd.Fu},
          {"Fu force-state rows (dlam/dtau)", data->Fu.bottomRows(nc), cd.Fu.bottomRows(nc)},
          {"Lx", data->Lx, cd.Lx},
          {"Lu", data->Lu, cd.Lu},
      };
      for (const auto& b : blks) {
        Err e = compare(b.a, b.b);
        std::cout << "  " << std::setw(5) << k << "   " << std::left << std::setw(30) << b.name
                  << std::right << std::scientific << std::setprecision(2) << std::setw(11)
                  << e.max_abs << "   " << std::setw(11) << e.rel_fro << std::endl;
      }
    }

    // ---------------- forward FD (crocoddyl NumDiff) accuracy, for reference
    {
      ActionModelNumDiff nd(iam);
      auto nd_data = nd.createData();
      nd.calc(nd_data, xs0, us0);
      nd.calcDiff(nd_data, xs0, us0);
      iam->calc(data, xs0, us0);
      iam->calcDiff(data, xs0, us0);
      Err ex = compare(data->Fx, nd_data->Fx), eu = compare(data->Fu, nd_data->Fu);
      std::cout << "\n-- forward FD (crocoddyl::ActionModelNumDiff, disturbance "
                << nd.get_disturbance() << "): Fx relFro " << ex.rel_fro << ", Fu relFro "
                << eu.rel_fro << std::endl;
    }

    // ---------------- timing, per node
    std::cout << "\n-- timing per node (" << reps << " repetitions, state 1)" << std::endl;
    force_feedback_mpc::Timer timer;
    {  // analytical: calc + calcDiff
      ActionModelNumDiff nd(iam);
      auto nd_data = nd.createData();
      CentralDiff cd(iam, h_ref);
      CentralDiff fw(iam, 1e-7, false);  // forward FD, first-order quantities only
      // warm-up
      for (std::size_t i = 0; i < 50; ++i) {
        iam->calc(data, xs0, us0); iam->calcDiff(data, xs0, us0);
        nd.calc(nd_data, xs0, us0); nd.calcDiff(nd_data, xs0, us0);
        cd.calcDiff(xs0, us0); fw.calcDiff(xs0, us0);
      }
      std::vector<double> t_ana, t_fwd, t_cen, t_fwd1;
      for (std::size_t i = 0; i < reps; ++i) {
        timer.start();
        iam->calc(data, xs0, us0);
        iam->calcDiff(data, xs0, us0);
        timer.stop(); t_ana.push_back(timer.elapsed().user * 1e3);  // ms -> us
      }
      for (std::size_t i = 0; i < reps; ++i) {
        timer.start();
        nd.calc(nd_data, xs0, us0);
        nd.calcDiff(nd_data, xs0, us0);
        timer.stop(); t_fwd.push_back(timer.elapsed().user * 1e3);
      }
      for (std::size_t i = 0; i < reps; ++i) {
        timer.start();
        cd.calcDiff(xs0, us0);
        timer.stop(); t_cen.push_back(timer.elapsed().user * 1e3);
      }
      for (std::size_t i = 0; i < reps; ++i) {
        timer.start();
        fw.calcDiff(xs0, us0);
        timer.stop(); t_fwd1.push_back(timer.elapsed().user * 1e3);
      }
      Stats a = stats(t_ana), f = stats(t_fwd), c = stats(t_cen), f1 = stats(t_fwd1);
      print_stats("analytical (calc+calcDiff)", a);
      print_stats("forward FD (1st order only)", f1);
      print_stats("central FD (1st order only)", c);
      print_stats("crocoddyl NumDiff (+num Hessians)", f);
      std::cout << std::fixed << std::setprecision(1)
                << "  speedup vs forward FD: " << f1.median / a.median
                << "x   vs central FD: " << c.median / a.median
                << "x   vs crocoddyl NumDiff: " << f.median / a.median << "x" << std::endl;
    }

    // ---------------- timing, full horizon at a fixed trajectory
    {
      std::vector<std::shared_ptr<ActionModelAbstract>> runs_ana, runs_fd;
      for (std::size_t i = 0; i < N; ++i) {
        runs_ana.push_back(iam);
        runs_fd.push_back(std::make_shared<ActionModelNumDiff>(iam));
      }
      auto term = make_iam(s, nc, 0.);
      ShootingProblem prob_ana(xs0, runs_ana, term);
      ShootingProblem prob_fd(xs0, runs_fd, std::make_shared<ActionModelNumDiff>(term));
      std::vector<VectorXd> xs(N + 1, xs0), us(N, us0);
      std::vector<CentralDiff> cds;
      for (std::size_t i = 0; i < N; ++i) cds.emplace_back(iam, h_ref);
      for (std::size_t i = 0; i < 20; ++i) {  // warm-up
        prob_ana.calc(xs, us); prob_ana.calcDiff(xs, us);
        prob_fd.calc(xs, us); prob_fd.calcDiff(xs, us);
      }
      std::vector<double> t_ana, t_fwd, t_cen;
      for (std::size_t i = 0; i < horizon_reps; ++i) {
        timer.start(); prob_ana.calc(xs, us); prob_ana.calcDiff(xs, us);
        timer.stop(); t_ana.push_back(timer.elapsed().user * 1e3);
      }
      for (std::size_t i = 0; i < horizon_reps; ++i) {
        timer.start(); prob_fd.calc(xs, us); prob_fd.calcDiff(xs, us);
        timer.stop(); t_fwd.push_back(timer.elapsed().user * 1e3);
      }
      for (std::size_t i = 0; i < horizon_reps; ++i) {
        timer.start();
        for (std::size_t k = 0; k < N; ++k) cds[k].calcDiff(xs[k], us[k]);
        timer.stop(); t_cen.push_back(timer.elapsed().user * 1e3);
      }
      Stats a = stats(t_ana), f = stats(t_fwd), c = stats(t_cen);
      std::cout << "\n-- timing full horizon N=" << N << " (" << horizon_reps
                << " repetitions, fixed trajectory)" << std::endl;
      print_stats("analytical (problem calc+calcDiff)", a);
      print_stats("forward FD (NumDiff models)", f);
      print_stats("central FD (running nodes)", c);
      std::cout << std::fixed << std::setprecision(1)
                << "  speedup vs forward FD: " << f.median / a.median
                << "x   vs central FD: " << c.median / a.median << "x" << std::endl;
    }
  }
  return 0;
}

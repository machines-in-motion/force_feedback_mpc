import numpy as np
from force_feedback_mpc.core_mpc_utils.path_utils import load_yaml_file
import matplotlib.pyplot as plt

import os
import sys

# Repository root (for the `demos` package)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
# PREFIX = "/home/skleff/Desktop/PUBLICATIONS/TRO-Soft/SECOND_SUBMISSION/EXPERIMENT_DATA/go2+arm/"
PREFIX = "/home/skleff/Desktop/PUBLICATIONS/TRO-Soft/AURO/REVISION/GO2_RERUN/A_foot_mu_default/"
# PREFIX = "/home/skleff/Desktop/PUBLICATIONS/TRO-Soft/AURO/REVISION/GO2_RERUN/B_foot_mu_1.0/"
TYPE = "soft"  # classical or soft

if TYPE == "classical":
    from demos.go2arm.Go2MPC_wrapper_classical import Go2MPCClassical as Go2MPCWrapper

    DATA_PATH = PREFIX+"CONSTANT_F=80/TOL=1e-4/go2_classical_INT=False_Fmin=80_Fmax=80_maxit=1000_fweight=0.0005.npz"
    CONFIG_PATH = REPO_ROOT + "/demos/go2arm/Go2MPC_demo_classical.yml"
else:
    from demos.go2arm.Go2MPC_wrapper_soft import Go2MPCSoft as Go2MPCWrapper

    # DATA_PATH = PREFIX+"GO2_DATA/F=80/go2_soft_Fmax=80_maxit=1000_fweight=0.05.npz"  # go2_soft_Fmax=80_maxit=1000_fweight=0.001_CONSTANT.npz' #'
    DATA_PATH = PREFIX+"CONSTANT_F=80/TOL=1e-4/go2_soft_Fmax=80_maxit=1000_fweight=0.0005.npz"  # go2_soft_Fmax=80_maxit=1000_fweight=0.001_CONSTANT.npz' #'
    CONFIG_PATH = "/home/skleff/CODE/force_feedback_mpc/demos/go2arm/Go2MPC_demo_soft.yml"

DATA_PATH = "/home/skleff/Desktop/PUBLICATIONS/TRO-Soft/AURO/REVISION/GO2_RERUN/B_tuning/go2_soft_Fmin=80_Fmax=80_maxit=1000_fweight=0.002_tol=0.0001.npz"
print("Loading data from: ", DATA_PATH)
print("Loading config from: ", CONFIG_PATH)

# Load data and extract signals
data = np.load(DATA_PATH, allow_pickle=True)
jointPos = data["jointPos"]
jointVel = data["jointVel"]
gap_norm = data["gap_norm"]
constraint_norm = data["constraint_norm"]
kkt_norm = data["kkt_norm"]
joint_torques = data["joint_torques"]
measured_forces_dict = data["measured_forces"].item()
# filtered_forces      = data['filtered_forces'].item()
desired_forces = data["desired_forces"]
predicted_forces = data["predicted_forces"].item()
ee_frame_names = data["ee_frame_names"]
# Load config file
CONFIG = load_yaml_file(CONFIG_PATH)
DT_SIMU = CONFIG["DT_SIMU"]
N_SIMU = CONFIG["N_SIMU"]
MU = CONFIG["MU"]
MPC_FREQ = CONFIG["MPC_FREQ"]


# Compute cost and constraint violation along MPC trajectory
FMIN = CONFIG["FMIN"]
FMAX = CONFIG["FMAX"]
HORIZON = CONFIG["HORIZON"]
DT_OCP = CONFIG["DT_OCP"]
# OCP cost evaluation requires the mim_robots 'go2' model (not available in the
# current environment) and the cost is not printed anyway: disabled by default.
COMPUTE_OCP_COST = False
if COMPUTE_OCP_COST:
    mpc = Go2MPCWrapper(HORIZON=HORIZON, friction_mu=MU, dt=DT_OCP, USE_MUJOCO=False)
    mpc.initialize(FMIN=FMIN)
    m = mpc.ocp.runningModels[0]
    d = m.createData()
cost = 0
violation = 0
err_f_x = 0.0
err_f_y = 0.0
err_f_z = 0.0
f = np.zeros(15)
for i in range(N_SIMU):
    # print("Stage ", i)
    # Get state
    if TYPE == "classical":
        x = np.hstack([jointPos[i], jointVel[i]])
    else:
        for k, fname in enumerate(ee_frame_names):
            f[3 * k : 3 * (k + 1)] = measured_forces_dict[fname][i]
        x = np.hstack([jointPos[i], jointVel[i], f])
    u = joint_torques[i]
    # Compute cost and constaint violation
    if COMPUTE_OCP_COST:
        m = mpc.ocp.runningModels[0]
        m.calc(d, x, u)
        cost += d.cost
    fric_res_l6 = MU * np.abs(measured_forces_dict["Link6"][i][0]) - np.sqrt(
        measured_forces_dict["Link6"][i][1] ** 2
        + measured_forces_dict["Link6"][i][2] ** 2
    )
    fric_res_FL = MU * np.abs(measured_forces_dict["FL_FOOT"][i][2]) - np.sqrt(
        measured_forces_dict["FL_FOOT"][i][0] ** 2
        + measured_forces_dict["FL_FOOT"][i][1] ** 2
    )
    fric_res_FR = MU * np.abs(measured_forces_dict["FR_FOOT"][i][2]) - np.sqrt(
        measured_forces_dict["FR_FOOT"][i][0] ** 2
        + measured_forces_dict["FR_FOOT"][i][1] ** 2
    )
    fric_res_HL = MU * np.abs(measured_forces_dict["HL_FOOT"][i][2]) - np.sqrt(
        measured_forces_dict["HL_FOOT"][i][0] ** 2
        + measured_forces_dict["HL_FOOT"][i][1] ** 2
    )
    fric_res_HR = MU * np.abs(measured_forces_dict["HR_FOOT"][i][2]) - np.sqrt(
        measured_forces_dict["HR_FOOT"][i][0] ** 2
        + measured_forces_dict["HR_FOOT"][i][1] ** 2
    )
    violation += min(fric_res_l6, 0)
    violation += min(fric_res_FL, 0)
    violation += min(fric_res_FR, 0)
    violation += min(fric_res_HL, 0)
    violation += min(fric_res_HR, 0)
    # cstr_lb = 0
    # cstr_ub = 0
    # if(np.linalg.norm(m.g_lb) < np.inf):
    #     cstr_lb = min(0, np.linalg.norm(d.g - m.g_lb, np.inf))
    #     violation += cstr_lb
    # if(np.linalg.norm(m.g_ub) < np.inf):
    #     cstr_ub = max(0, np.linalg.norm(d.g - m.g_ub, np.inf))
    #     violation += cstr_ub
    err_f_x += (measured_forces_dict["Link6"][i][0] - desired_forces[i][0]) ** 2
    err_f_y += (measured_forces_dict["Link6"][i][1] - desired_forces[i][1]) ** 2
    err_f_z += (measured_forces_dict["Link6"][i][2] - desired_forces[i][2]) ** 2
# print("Total cost: ", cost)
print("Total constraint violation: ", violation)
print("RMSE F_ee_x = ", np.sqrt(err_f_x / N_SIMU))
print("RMSE F_ee_y = ", np.sqrt(err_f_y / N_SIMU))
print("RMSE F_ee_z = ", np.sqrt(err_f_z / N_SIMU))


# Visualize the measured force against the desired
time_span = np.linspace(0, (N_SIMU - 1) * DT_SIMU, N_SIMU)
# EE FORCES
fig, axs = plt.subplots(3, 1, constrained_layout=True)
# Fx_lb_mea = (1./MU)*np.sqrt(measured_forces_dict['Link6'][:, 1]**2 + measured_forces_dict['Link6'][:, 1]**2)
# Fx_lb_pred = (1./MU)*np.sqrt(predicted_forces_dict['Link6'][:, 1]**2 + predicted_forces_dict['Link6'][:, 1]**2)
axs[0].plot(
    time_span,
    np.abs(measured_forces_dict["Link6"][:, 0]),
    linewidth=4,
    color="g",
    marker="o",
    alpha=0.5,
    label="Fx mea",
)
axs[0].plot(
    time_span,
    np.abs(desired_forces[:, 0]),
    linewidth=4,
    color="k",
    marker="o",
    alpha=0.25,
    label="Fx des",
)
# axs[0].plot(time_span, np.abs(predicted_forces_dict['Link6'][:,0]), linewidth=4, color='b', marker='o', alpha=0.25, label="Fx predicted")
# axs[0].plot(time_span, Fx_lb_mea, '--', linewidth=4, color='k',  alpha=0.5, label="Fx friction constraint (lower bound)")
axs[0].set_ylim(-10.0, 105)

axs[1].plot(
    time_span,
    measured_forces_dict["Link6"][:, 1],
    linewidth=4,
    color="g",
    marker="o",
    alpha=0.5,
    label="Fy mea",
)
axs[1].plot(
    time_span,
    desired_forces[:, 1],
    linewidth=4,
    color="k",
    marker="o",
    alpha=0.25,
    label="Fy des",
)
# axs[1].plot(time_span, predicted_forces_dict['Link6'][:,1], linewidth=4, color='b', marker='o', alpha=0.25, label="Fy predicted")
axs[1].set_ylim(-10.0, 10)

axs[2].plot(
    time_span,
    measured_forces_dict["Link6"][:, 2],
    linewidth=4,
    color="g",
    marker="o",
    alpha=0.5,
    label="Fz mea",
)
axs[2].plot(
    time_span,
    desired_forces[:, 2],
    linewidth=4,
    color="k",
    marker="o",
    alpha=0.25,
    label="Fz des",
)
# axs[2].plot(time_span, predicted_forces_dict['Link6'][:,2], linewidth=4, color='b', marker='o', alpha=0.25, label="Fz predicted")
axs[2].set_ylim(-25.0, 10)
for i in range(3):
    axs[i].legend()
    axs[i].grid()
fig.suptitle("Contact force at the end-effector", fontsize=16)

# FEET FORCES (measured and predicted, with friction constraint lower bound on Fz)
fig, axs = plt.subplots(3, 4, constrained_layout=True)
for i, fname in enumerate(ee_frame_names[:-1]):
    # x,y
    axs[0, i].plot(
        time_span,
        measured_forces_dict[fname][:, 0],
        linewidth=4,
        color="g",
        marker="o",
        alpha=0.5,
        label="Fx measured",
    )
    # axs[0, i].plot(time_span, predicted_forces_dict[fname][:,0], linewidth=4, color='b', marker='o', alpha=0.25, label="Fx predicted")
    axs[1, i].plot(
        time_span,
        measured_forces_dict[fname][:, 1],
        linewidth=4,
        color="g",
        marker="o",
        alpha=0.5,
        label="Fy measured",
    )
    # axs[1, i].plot(time_span, predicted_forces_dict[fname][:,1], linewidth=4, color='b', marker='o', alpha=0.25, label="Fy predicted")
    axs[0, i].legend()
    # axs[0, i].title(fname)
    axs[0, i].grid()
    axs[1, i].legend()
    axs[1, i].grid()

    # z
    Fz_lb_mea = (1.0 / MU) * np.sqrt(
        measured_forces_dict[fname][:, 0] ** 2 + measured_forces_dict[fname][:, 1] ** 2
    )
    # Fz_lb_pred = (1./MU)*np.sqrt(predicted_forces_dict[fname][:, 0]**2 + predicted_forces_dict[fname][:, 1]**2)
    axs[2, i].plot(
        time_span,
        measured_forces_dict[fname][:, 2],
        linewidth=4,
        color="g",
        marker="o",
        alpha=0.5,
        label="Fz measured",
    )
    # axs[2, i].plot(time_span, predicted_forces_dict[fname][:,2], linewidth=4, color='b', marker='o', alpha=0.25, label="Fz predicted")
    axs[2, i].plot(
        time_span,
        Fz_lb_mea,
        "--",
        linewidth=4,
        color="k",
        alpha=0.5,
        label="Fz friction constraint (lower bound)",
    )
    # axs[2, i].plot(time_span, Fz_lb_pred, '--', linewidth=4, color='b', alpha=0.2, label="Fz friction lb (pred)")
    axs[2, i].legend()
    axs[2, i].grid()

fig.suptitle("Contact forces at feet FL, FR, HL, HR", fontsize=16)


# SOLVER METRICS
N_MPC_STEPS = int(N_SIMU * DT_SIMU * MPC_FREQ)
time_span2 = np.linspace(0, (N_SIMU - 1) * DT_SIMU, N_MPC_STEPS)
# fig, axs = plt.subplots(3, 1, constrained_layout=True)
# axs[0].plot(time_span, gap_norm,linewidth=4, color='g', marker='o', alpha=0.5, label="Gap norm")
# # axs[0].plot(time_span, np.abs(desired_forces[:,0]), linewidth=4, color='k', marker='o', alpha=0.25, label="Fx des")
# # axs[0].set_ylim(0., 105)

# axs[1].plot(time_span, constraint_norm, linewidth=4, color='g', marker='o', alpha=0.5, label="Constraint norm")
# # axs[1].plot(time_span, desired_forces[:,1], linewidth=4, color='k', marker='o', alpha=0.25, label="Fy des")
# # axs[1].set_ylim(-10., 10)

# axs[2].plot(time_span, kkt_norm,linewidth=4, color='g', marker='o', alpha=0.5, label="KKT")
# axs[2].plot(time_span, np.array([1e-4]*N_MPC_STEPS), linewidth=4, color='k', marker='o', alpha=0.25, label="TOL")
# # axs[2].set_ylim(-25., 10)
# for i in range(3):
#     axs[i].legend()
#     axs[i].grid()
# fig.suptitle('Solver convergence', fontsize=16)

fig, axs = plt.subplots(2, 1, constrained_layout=True)
axs[0].plot(
    time_span2,
    gap_norm + constraint_norm,
    linewidth=4,
    color="g",
    marker="o",
    alpha=0.5,
    label="Constraint norm",
)
# axs[0].set_ylim(0., 105)
axs[1].plot(
    time_span,
    np.abs(measured_forces_dict["Link6"][:, 0]),
    linewidth=4,
    color="g",
    marker="o",
    alpha=0.5,
    label="Measured force (Fx)",
)
axs[1].plot(
    time_span,
    np.abs(desired_forces[:, 0]),
    linewidth=4,
    color="k",
    marker="o",
    alpha=0.25,
    label="Desired force (Fx)",
)
axs[1].set_ylim(-1.0, FMAX + 1)
for i in range(2):
    axs[i].legend()
    axs[i].grid()
fig.suptitle("Constraint violation + Force tracking", fontsize=16)

plt.show() 
# FOOT FRICTION CONE / UNILATERALITY ANALYSIS (all controllers, independent of TYPE)
#   Forces are 3D, [Fx, Fy, Fz] in WORLD axes (flat ground: F_N = Fz, F_T = [Fx, Fy])
#     'measured'  : realized PyBullet contact forces (env -> robot, summed over contact points)
#     'predicted' : MPC-predicted forces (soft: force state at node 1). WARNING: classical
#                   predicted forces are constant in the files (logged as an aliased view
#                   of the solver data without copy) and are not usable.
#   Constraints implemented in both OCPs (per foot, mu = MU):
#     friction      : mu*|F_N| - ||F_T|| >= 0
#     unilaterality : F_N >= 0
#   NB: PyBullet enforces Coulomb friction itself, so measured forces cannot leave the
#   *simulator's* cone; reaching it means the foot slides. The ground has mu = MU, but
#   PyBullet multiplies it with the link friction (default 0.5, the URDF sets none), so the
#   effective coefficient appears to be MU_SIM = 0.5*MU: measured ratios saturate at 0.5.
DATASET_DIR = PREFIX + "CONSTANT_F=80/TOL=1e-4/"
CONTROLLERS = {
    "Classical MPC": "go2_classical_INT=False_Fmin=80_Fmax=80_maxit=1000_fweight=0.0005.npz",
    "Classical MPC + Integral": "go2_classical_INT=True_Fmin=80_Fmax=80_maxit=1000_fweight=0.0005.npz",
    "Force-Feedback MPC": "go2_soft_Fmax=80_maxit=1000_fweight=0.0005.npz",
}
SHOW_CONTROLLERS = list(CONTROLLERS.keys())
COLORS = {"Classical MPC": "b", "Classical MPC + Integral": "g", "Force-Feedback MPC": "r"}
FEET = ["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"]
FORCE_SOURCE = "measured"  # 'measured' or 'predicted'
METRIC = "margin"  # 'margin' (mu*F_N - ||F_T||) or 'ratio' (||F_T|| / (mu*F_N))
SEPARATE_FEET = True  # one subplot per foot, or all feet on a single axis
FIGSIZE = (12, 8)
FN_MIN = 0.0  # unilaterality lower bound used in the OCP
FN_EPS = 1e-6  # F_N below this is treated as "no contact" (ratio undefined -> NaN)
MU_SIM = 0.5 * MU  # apparent PyBullet effective friction (inferred from data, see NB above)


def foot_force_metrics(F, mu, fn_min=FN_MIN, fn_eps=FN_EPS):
    """F : (N,3) array of [Fx, Fy, Fz]. Returns the constraint quantities per sample."""
    FN = F[:, 2]
    FT = np.linalg.norm(F[:, :2], axis=1)
    no_contact = FN <= fn_eps
    ratio = np.full(FN.shape, np.nan)
    ratio[~no_contact] = FT[~no_contact] / (mu * FN[~no_contact])
    return {
        "FN": FN,
        "FT": FT,
        "margin": mu * FN - FT,  # >= 0 feasible (negative F_N counts as infeasible)
        "margin_ocp": mu * np.abs(FN) - FT,  # exact OCP friction residual
        "ratio": ratio,  # <= 1 feasible, NaN where F_N <= fn_eps
        "uni": FN - fn_min,  # >= 0 feasible
        "no_contact": no_contact,
    }


def load_foot_metrics(source=FORCE_SOURCE):
    """Returns {controller: {foot: metrics}} for the controllers in CONTROLLERS."""
    metrics = {}
    for label, fname in CONTROLLERS.items():
        forces = np.load(DATASET_DIR + fname, allow_pickle=True)[source + "_forces"].item()
        metrics[label] = {foot: foot_force_metrics(forces[foot], MU) for foot in FEET}
    return metrics


def print_constraint_summary(metrics, source=FORCE_SOURCE):
    print("\nFoot constraint summary (" + source + " forces, mu = " + str(MU) + ", N = " + str(N_SIMU) + ")")
    print(
        "%-26s %-8s %9s %9s %5s %6s %9s %6s %9s %6s %8s"
        % ("controller", "foot", "min F_N", "max F_N", "#FN<0", "#noCt", "min marg", "#m<0", "max ratio", "#nan", "%slide")
    )
    for label, feet in metrics.items():
        for foot, mt in feet.items():
            in_contact = ~mt["no_contact"]
            # % of in-contact samples on the simulator cone (sliding), measured forces only
            sliding = np.nan
            if source == "measured" and in_contact.any():
                sliding = 100.0 * np.mean(mt["ratio"][in_contact] >= 0.999 * MU_SIM / MU)
            print(
                "%-26s %-8s %9.2f %9.2f %5d %6d %9.3g %6d %9.3f %6d %8.1f"
                % (
                    label, foot, mt["FN"].min(), mt["FN"].max(), np.sum(mt["FN"] < 0),
                    np.sum(mt["no_contact"]), mt["margin"].min(), np.sum(mt["margin"] < 0),
                    np.nanmax(mt["ratio"]) if in_contact.any() else np.nan,
                    np.sum(np.isnan(mt["ratio"])), sliding,
                )
            )


def plot_feet(metrics, key, ylabel, ref, title):
    """Plots metrics[controller][foot][key] vs time, with a feasibility reference line."""
    if SEPARATE_FEET:
        fig, axs = plt.subplots(2, 2, figsize=FIGSIZE, sharex=True, constrained_layout=True)
        axs = axs.flatten()
    else:
        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE, constrained_layout=True)
        axs = [ax] * len(FEET)
    for k, foot in enumerate(FEET):
        for label in SHOW_CONTROLLERS:
            lbl = label if SEPARATE_FEET else label + " " + foot
            axs[k].plot(time_span, metrics[label][foot][key], color=COLORS[label], alpha=0.7, label=lbl)
        if SEPARATE_FEET or k == 0:
            axs[k].axhline(ref, color="k", linestyle="--", label="OCP bound")
            if key == "ratio":
                axs[k].axhline(MU_SIM / MU, color="k", linestyle=":", label="PyBullet cone (MU_SIM)")
        if SEPARATE_FEET:
            axs[k].set_title(foot)
        axs[k].set_xlabel("Time (s)")
        axs[k].set_ylabel(ylabel)
        axs[k].grid(True)
    axs[0].legend()
    fig.suptitle(title + " (" + FORCE_SOURCE + " forces)")
    return fig, axs


foot_metrics = load_foot_metrics(FORCE_SOURCE)
print_constraint_summary(foot_metrics, FORCE_SOURCE)
if METRIC == "margin":
    plot_feet(foot_metrics, "margin", r"$\mu F_N - \|F_T\|$ (N)", 0.0, "Friction cone margin")
else:
    plot_feet(foot_metrics, "ratio", r"$\|F_T\| / (\mu F_N)$", 1.0, "Friction ratio")
plot_feet(foot_metrics, "uni", r"$F_N - F_{min}$ (N)", 0.0, "Unilaterality margin")


plt.show()

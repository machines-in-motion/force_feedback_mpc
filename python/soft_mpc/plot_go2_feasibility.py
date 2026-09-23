"""
Go2+arm (AuRo revision): friction-cone and unilaterality satisfaction per contact,
  - in closed loop (realized PyBullet forces, ground-only for the feet, wall-only for the EE)
  - in the OCP prediction (forces planned over the MPC horizon, at each MPC cycle)

Requires the npz files written by demos/go2arm/Go2MPC_demo_{classical,soft}.py with
horizon logging (keys: ground_forces, wall_forces, ocp_forces, sqp_iters, kkt_norm, config).

Conventions (see demos/go2arm/*):
  - forces are applied by the environment on the robot, WORLD / LOCAL_WORLD_ALIGNED axes
  - feet (ground): normal = +z  -> F_N = Fz, F_T = (Fx, Fy)
  - end-effector (wall at x=0.42): normal = -x -> F_N = -Fx, F_T = (Fy, Fz)
  - OCP constraints (feet only): mu*|F_N| - ||F_T|| >= 0 and F_N >= 0, mu = config MU.
    The end-effector has no friction/unilaterality constraint in either OCP.
  - OCP nodes: classical nodes 0..T-1 (node 0 depends on the first control);
    force-feedback nodes 1..T (node 0 is the measured force, not a decision variable)
"""

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------- knobs
DATA_DIR = "/home/skleff/Desktop/PUBLICATIONS/TRO-Soft/AURO/REVISION/GO2_RERUN/B_final/"
FF_FWEIGHT = "0.002"  # force-feedback force-tracking weight (tuned for set B; published value 0.0005)
CONTROLLERS = {
    "Classical MPC": "go2_classical_INT=False_Fmin=80_Fmax=80_maxit=1000_fweight=0.0005.npz",
    "Classical MPC + Integral": "go2_classical_INT=True_Fmin=80_Fmax=80_maxit=1000_fweight=0.0005.npz",
    "Force-feedback MPC": "go2_soft_Fmin=80_Fmax=80_maxit=1000_fweight=" + FF_FWEIGHT + "_tol=0.0001.npz",
}
SHOW_CONTROLLERS = list(CONTROLLERS.keys())
CONTACTS = ["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT", "Link6"]  # drop "Link6" to show feet only
# categorical colors (reference palette slots 1-3, fixed order) + line styles as secondary encoding
COLORS = {"Classical MPC": "#2a78d6", "Classical MPC + Integral": "#eb6834", "Force-feedback MPC": "#1baf7a"}
STYLES = {"Classical MPC": "-", "Classical MPC + Integral": "--", "Force-feedback MPC": "-."}
SOURCE_COLORS = {"closed loop": "#2a78d6", "OCP prediction": "#eb6834"}
FN_EPS = 1e-6  # |F_N| below this: no contact (closed loop)
RATIO_FN_MIN = 0.5  # [N] ratio ||F_T||/F_N only defined above this normal force (near-zero F_N makes it
#                      meaningless); excluded nodes are counted in the summary, where the margin is also given
RATIO_YLIM = (0.0, 1.5)  # ratio axis limits (the printed summary gives the true max ratio, even if clipped)
FIGSIZE = (16, 6)
SAVE_DIR = None  # e.g. DATA_DIR + "figs/" to save PNG/PDF, None to only show
DT_SIMU, DT_MPC = 1e-3, 1e-2


# ----------------------------------------------------------------------------- helpers
def normal_tangential(F, contact):
    """F: (..., 3) forces -> (F_N, ||F_T||) for a foot (ground normal +z) or the EE (wall normal -x)"""
    if contact == "Link6":
        return -F[..., 0], np.linalg.norm(F[..., 1:], axis=-1)
    return F[..., 2], np.linalg.norm(F[..., :2], axis=-1)


def ratio(FN, FT):
    """||F_T|| / F_N, NaN where F_N <= RATIO_FN_MIN (unloaded, no contact or tensile force)"""
    r = np.full(np.shape(FN), np.nan)
    ok = FN > RATIO_FN_MIN
    r[ok] = FT[ok] / FN[ok]
    return r


def load(label):
    d = np.load(DATA_DIR + CONTROLLERS[label], allow_pickle=True)
    cfg = d["config"].item()
    kind = "soft" if "filtered_forces" in d.files else "classical"  # only the soft demo saves filtered_forces
    ocp = d["ocp_forces"]  # (cycles, nodes, contacts, 3)
    ocp = ocp[:, 1:] if kind == "soft" else ocp  # decision-variable nodes only (see header)
    names = list(d["ee_frame_names"])
    out = {"mu": cfg["MU"], "tol": float(d["sqp_tol"]), "kkt": d["kkt_norm"], "names": names, "contacts": {}}
    out["converged"] = d["kkt_norm"] <= out["tol"]
    ground, wall = d["ground_forces"].item(), d["wall_forces"].item()
    for c in CONTACTS:
        k = names.index(c)
        FN_cl, FT_cl = normal_tangential((wall if c == "Link6" else ground)[c], c)
        FN_ocp, FT_ocp = normal_tangential(ocp[:, :, k], c)  # (cycles, nodes)
        out["contacts"][c] = dict(
            FN_cl=FN_cl, ratio_cl=ratio(FN_cl, FT_cl),
            FN_ocp=FN_ocp, ratio_ocp=ratio(FN_ocp, FT_ocp),
            margin_ocp=out["mu"] * FN_ocp - FT_ocp,
            # worst case over the horizon at each MPC cycle
            ratio_ocp_worst=np.nanmax(np.where(np.isnan(ratio(FN_ocp, FT_ocp)), -np.inf, ratio(FN_ocp, FT_ocp)), axis=1),
            FN_ocp_worst=FN_ocp.min(axis=1),
        )
    return out


def print_summary(data):
    print("\nPer contact: closed loop (realized) | OCP prediction (all nodes, all cycles; converged cycles in brackets)")
    print(f"{'controller':26s} {'contact':8s} | {'max ratio':>9s} {'%on cone':>8s} {'min F_N':>8s} {'%no ct':>6s} |"
          f" {'max ratio':>17s} {'min margin [N]':>21s} {'min F_N [N]':>21s} {'#non-conv':>9s} {'#nodes F_N<min':>14s}")
    for label, D in data.items():
        mu, conv = D["mu"], D["converged"]
        for c, q in D["contacts"].items():
            r, FN = q["ratio_cl"][1:], q["FN_cl"][1:]  # sample 0 precedes the first contact update
            on_cone = 100 * np.nanmean(r >= 0.999 * mu) if np.any(~np.isnan(r)) else np.nan
            ro, mo, fo = q["ratio_ocp"], q["margin_ocp"], q["FN_ocp"]
            print(f"{label:26s} {c:8s} | {np.nanmax(r) if np.any(~np.isnan(r)) else np.nan:9.3f} {on_cone:8.1f} {FN.min():8.2f} {100*np.mean(FN <= FN_EPS):6.1f} |"
                  f" {np.nanmax(ro):8.3f} [{np.nanmax(ro[conv]) if conv.any() else np.nan:6.3f}]"
                  f" {mo.min():10.2e} [{mo[conv].min() if conv.any() else np.nan:9.2e}]"
                  f" {fo.min():10.2e} [{fo[conv].min() if conv.any() else np.nan:9.2e}] {np.sum(~conv):9d} {np.sum(fo <= RATIO_FN_MIN):14d}")
        print(f"{'':26s} (mu = {mu}, SQP tol = {D['tol']:.0e}, EE Link6 is not constrained in the OCP)")


def _hold(t, y):
    """zero-order-hold series (one value per MPC cycle) extended to the end of the last cycle"""
    return np.append(t, t[-1] + DT_MPC), np.append(y, y[-1])


def _finish(fig, name):
    if SAVE_DIR is not None:
        import os
        os.makedirs(SAVE_DIR, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(SAVE_DIR + name + "." + ext, dpi=200)


def plot_controller(label, D):
    """One figure per controller: top = friction ratio vs mu, bottom = normal force vs 0; closed loop vs OCP"""
    n = len(CONTACTS)
    fig, axs = plt.subplots(2, n, figsize=FIGSIZE, sharex=True, constrained_layout=True, squeeze=False)
    t_cl = np.arange(len(D["contacts"][CONTACTS[0]]["FN_cl"])) * DT_SIMU
    t_ocp = np.arange(len(D["kkt"])) * DT_MPC
    nc = ~D["converged"]
    for j, c in enumerate(CONTACTS):
        q = D["contacts"][c]
        axs[0, j].plot(t_cl, q["ratio_cl"], color=SOURCE_COLORS["closed loop"], lw=1.5, label="closed loop")
        axs[0, j].step(*_hold(t_ocp, np.where(np.isinf(q["ratio_ocp_worst"]), np.nan, q["ratio_ocp_worst"])), where="post",
                       color=SOURCE_COLORS["OCP prediction"], lw=1.5, ls="--", label="OCP prediction (worst node)")
        axs[1, j].plot(t_cl, q["FN_cl"], color=SOURCE_COLORS["closed loop"], lw=1.5, label="closed loop")
        axs[1, j].step(*_hold(t_ocp, q["FN_ocp_worst"]), where="post", color=SOURCE_COLORS["OCP prediction"], lw=1.5, ls="--",
                       label="OCP prediction (worst node)")
        if nc.any():  # non-converged SQP solves
            axs[1, j].plot(t_ocp[nc], q["FN_ocp_worst"][nc], "x", color="0.3", ms=3, label="SQP not converged")
        if c != "Link6":
            axs[0, j].axhline(D["mu"], color="k", lw=1, ls=":", label=r"$\mu$")
            axs[1, j].axhline(0.0, color="k", lw=1, ls=":", label=r"$F_N = 0$")
        axs[0, j].set_ylim(*RATIO_YLIM)
        axs[0, j].set_title(c + (" (not constrained)" if c == "Link6" else ""))
        axs[1, j].set_xlabel("Time (s)")
        for i in range(2):
            axs[i, j].grid(True, color="0.9")
    axs[0, 0].set_ylabel(r"$\|F_T\| / F_N$")
    axs[1, 0].set_ylabel(r"$F_N$ (N)")
    axs[0, 0].legend(fontsize=8)
    axs[1, 0].legend(fontsize=8)
    fig.suptitle(label + ": friction cone and unilaterality")
    _finish(fig, "feasibility_" + label.replace(" ", "_").replace("+", "plus"))
    return fig, axs


def plot_ocp_comparison(data):
    """OCP prediction only, all controllers on the same axes, one column per contact"""
    n = len(CONTACTS)
    fig, axs = plt.subplots(2, n, figsize=FIGSIZE, sharex=True, constrained_layout=True, squeeze=False)
    for label, D in data.items():
        t_ocp = np.arange(len(D["kkt"])) * DT_MPC
        for j, c in enumerate(CONTACTS):
            q = D["contacts"][c]
            kw = dict(where="post", color=COLORS[label], ls=STYLES[label], lw=1.5, label=label)
            axs[0, j].step(*_hold(t_ocp, np.where(np.isinf(q["ratio_ocp_worst"]), np.nan, q["ratio_ocp_worst"])), **kw)
            axs[1, j].step(*_hold(t_ocp, q["FN_ocp_worst"]), **kw)
    mu = next(iter(data.values()))["mu"]
    for j, c in enumerate(CONTACTS):
        if c != "Link6":
            axs[0, j].axhline(mu, color="k", lw=1, ls=":", label=r"$\mu$")
            axs[1, j].axhline(0.0, color="k", lw=1, ls=":", label=r"$F_N = 0$")
        axs[0, j].set_ylim(*RATIO_YLIM)
        axs[0, j].set_title(c + (" (not constrained)" if c == "Link6" else ""))
        axs[1, j].set_xlabel("Time (s)")
        for i in range(2):
            axs[i, j].grid(True, color="0.9")
    axs[0, 0].set_ylabel(r"worst-node $\|F_T\| / F_N$")
    axs[1, 0].set_ylabel(r"worst-node $F_N$ (N)")
    axs[0, 0].legend(fontsize=8)
    fig.suptitle("OCP prediction: friction cone and unilaterality (worst node over the horizon)")
    _finish(fig, "feasibility_ocp_comparison")
    return fig, axs


if __name__ == "__main__":
    data = {label: load(label) for label in SHOW_CONTROLLERS}
    print_summary(data)
    for label, D in data.items():
        plot_controller(label, D)
    plot_ocp_comparison(data)
    plt.show()

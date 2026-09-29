"""
Quantitative friction-cone / unilaterality analysis of the Go2 multi-contact experiment
(Reviewer 1, Major comment 1).

Distinguishes two different questions:
  * PREDICTED  constraint satisfaction -- how accurately the OCP solution satisfies the
    hard constraints it imposes (optimization accuracy). Computed from the per-node
    contact forces of the accepted SQP iterate (`ocp_forces`).
  * REALIZED   constraint satisfaction -- whether the constrained prediction translates
    into the closed-loop simulation. Computed from the PyBullet contact forces
    (`measured_forces`). Note that the simulator enforces its own Coulomb law, so a
    realized margin of zero means sliding, not constraint violation.

Usage:
    python go2_constraint_analysis.py <run_dir> [out_dir]
"""
import glob
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUN = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/skleff/Desktop/PUBLICATIONS/TRO-Soft/AURO/REVISION/GO2_RERUN/C_hessfix_reg1em2"
OUT = sys.argv[2] if len(sys.argv) > 2 else RUN
MU = 0.75
FN_EPS = 1e-6          # below this the foot is considered unloaded
N_FEET = 4
CTRL = [("classical", "go2_classical_INT=False*.npz", "#4a4a4a"),
        ("classical + integral", "go2_classical_INT=True*.npz", "#c1492b"),
        ("force feedback (proposed)", "go2_soft*.npz", "#1b6ca8")]
FOOT_LABELS = ["FL", "FR", "HL", "HR"]

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True, "legend.frameon": False,
                     "figure.dpi": 150})


def cone_viol(F):
    """violation of mu*Fn - |Ft| >= 0, in N; F has shape (..., 3), z is the normal"""
    return np.maximum(np.linalg.norm(F[..., :2], axis=-1) - MU * F[..., 2], 0.0)


def uni_viol(F):
    return np.maximum(-F[..., 2], 0.0)


def load(run_dir):
    out = {}
    for name, pat, color in CTRL:
        g = glob.glob(os.path.join(run_dir, pat))
        if not g:
            print(f"! missing {pat} in {run_dir}")
            continue
        d = np.load(g[0], allow_pickle=True)
        M = d["measured_forces"].item()
        names = list(d["ee_frame_names"])
        rec = dict(color=color, file=os.path.basename(g[0]), names=names,
                   meas=np.stack([np.asarray(M[n], float) for n in names], axis=1),
                   kkt=np.asarray(d["kkt_norm"], float),
                   cstr=np.asarray(d["constraint_norm"], float),
                   desired=np.asarray(d["desired_forces"], float))
        if "ocp_forces" in d:
            rec["ocp"] = np.asarray(d["ocp_forces"], float)   # (cycles, nodes, ee, 3)
        if "sqp_iters" in d:
            rec["iters"] = np.asarray(d["sqp_iters"], float)
        # the soft OCP carries the measured force at node 0 (not a decision variable)
        rec["first_node"] = 1 if "soft" in rec["file"] else 0
        out[name] = rec
    return out


def boundary_pct(ratio, fn, fn_min=1.0):
    """fraction of MEANINGFULLY LOADED samples sitting at the friction limit.
    At a near-zero normal force the ratio saturates at 1 whatever the tangential
    force, so reporting it there would be physically meaningless."""
    m = fn > fn_min
    return float(100 * np.nanmean(ratio[m] >= 0.99)) if m.any() else float("nan")


def summarize(R):
    rows = []
    for name, r in R.items():
        F = r["meas"][:, :N_FEET]                       # realized foot forces
        fn = F[..., 2]
        loaded = fn > FN_EPS
        ratio = np.full(fn.shape, np.nan)
        ratio[loaded] = (np.linalg.norm(F[..., :2], axis=-1)[loaded] /
                         (MU * fn[loaded]))
        row = dict(name=name, file=r["file"],
                   kkt_med=float(np.median(r["kkt"])), kkt_max=float(np.max(r["kkt"])),
                   n=int(len(r["kkt"])),
                   n_1em4=int((r["kkt"] <= 1e-4).sum()),
                   n_1em3=int((r["kkt"] <= 1e-3).sum()),
                   n_1em2=int((r["kkt"] <= 1e-2).sum()),
                   real_fn_min=float(fn.min()),
                   real_unloaded_pct=[float(100 * (~loaded[:, k]).mean()) for k in range(N_FEET)],
                   real_ratio_max=float(np.nanmax(ratio)),
                   real_sliding_pct=float(100 * np.nanmean(ratio >= 0.99)),
                   front_fn_med=float(np.median(fn[:, :2])),
                   front_boundary_pct=[boundary_pct(ratio[:, k], fn[:, k]) for k in (0, 1)],
                   hind_boundary_pct=[boundary_pct(ratio[:, k], fn[:, k]) for k in (2, 3)])
        if "ocp" in r:
            P = r["ocp"][:, r["first_node"]:, :N_FEET]   # predicted foot forces
            cv, uv = cone_viol(P), uni_viol(P)
            row.update(pred_cone_max=float(cv.max()), pred_uni_max=float(uv.max()),
                       pred_cone_med=float(np.median(cv.max(axis=(1, 2)))),
                       pred_cycle_max=cv.max(axis=(1, 2)))
            conv = r["kkt"] <= 1e-4
            if conv.any():
                row["pred_cone_max_conv"] = float(cv[conv].max())
            if (~conv).any():
                row["pred_cone_max_unconv"] = float(cv[~conv].max())
        if "iters" in r:
            row["it_med"] = float(np.median(r["iters"]))
        rows.append(row)
    return rows


def fig_feet(R, out, t_skip=0.05):
    """per foot: predicted constraint violation (OCP) and realized cone ratio / normal force"""
    fig, ax = plt.subplots(3, N_FEET, figsize=(9.5, 5.4), sharex=True)
    fn_hi = np.zeros(N_FEET)
    for k in range(N_FEET):
        for name, r in R.items():
            F = r["meas"][:, k]
            t = np.arange(len(F)) * 1e-3
            keep = t >= t_skip
            fn = F[:, 2]
            loaded = fn > FN_EPS
            ratio = np.full(len(fn), np.nan)
            ratio[loaded] = np.linalg.norm(F[loaded, :2], axis=-1) / (MU * fn[loaded])
            if "ocp" in r:
                P = r["ocp"][:, r["first_node"]:, k]
                cyc = cone_viol(P).max(axis=1)
                ax[0, k].semilogy(np.arange(len(cyc)) * 1e-2, np.maximum(cyc, 1e-12),
                                  color=r["color"], lw=0.9, label=name)
            ax[1, k].plot(t[keep], ratio[keep], color=r["color"], lw=0.9, label=name)
            ax[2, k].plot(t[keep], fn[keep], color=r["color"], lw=0.9)
            fn_hi[k] = max(fn_hi[k], np.nanpercentile(fn[keep], 99.5))
        ax[1, k].axhline(1.0, color="k", ls="--", lw=0.8)
        ax[2, k].axhline(0.0, color="k", ls="--", lw=0.8)
        ax[0, k].set_title(FOOT_LABELS[k], fontsize=9)
        ax[1, k].set_ylim(-0.05, 1.2)
        ax[2, k].set_ylim(-0.05 * fn_hi[k], 1.15 * fn_hi[k])
        ax[2, k].set_xlabel("time [s]")
        ax[0, k].set_ylim(1e-10, 1.0)
    ax[0, 0].set_ylabel("predicted violation [N]")
    ax[1, 0].set_ylabel(r"$\|\lambda_T\| / (\mu \lambda_N)$")
    ax[2, 0].set_ylabel(r"$\lambda_N$ [N]")
    h, l = ax[1, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig_go2_feet.{ext}"))


def fig_pred(R, out):
    """predicted (OCP) constraint satisfaction and solver accuracy"""
    fig, ax = plt.subplots(1, 3, figsize=(9.5, 2.6))
    for name, r in R.items():
        if "ocp" not in r:
            continue
        P = r["ocp"][:, r["first_node"]:, :N_FEET]
        cyc = cone_viol(P).max(axis=(1, 2))
        t = np.arange(len(cyc)) * 1e-2
        ax[0].semilogy(t, np.maximum(cyc, 1e-12), color=r["color"], lw=0.9, label=name)
        ax[1].semilogy(np.arange(len(r["kkt"])) * 1e-2, r["kkt"], color=r["color"], lw=0.9)
        ee = r["meas"][:, -1]
        ax[2].plot(np.arange(len(ee)) * 1e-3, np.linalg.norm(ee, axis=-1),
                   color=r["color"], lw=0.9)
    ax[2].axhline(np.linalg.norm(list(R.values())[0]["desired"][0]), color="k",
                  ls="--", lw=0.8, label="desired")
    ax[0].set_ylabel("max predicted cone violation [N]")
    ax[1].set_ylabel("KKT residual")
    ax[1].axhline(1e-4, color="k", ls=":", lw=0.8)
    ax[2].set_ylabel(r"$\|\lambda_{\rm ee}\|$ [N]")
    for a, ttl in zip(ax, ("(a) OCP constraint satisfaction", "(b) solver accuracy",
                           "(c) end-effector force")):
        a.set_xlabel("time [s]")
        a.set_title(ttl, fontsize=9, loc="left")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig_go2_predicted.{ext}"))


def sci(v, floor=1e-12):
    v = max(float(v), floor)
    e = int(np.floor(np.log10(v)))
    return r"$%.1f\times10^{%d}$" % (v / 10 ** e, e)


def latex_table(rows, out):
    L = [r"\begin{tabular}{lccccccc}", r"\toprule",
         r" & \multicolumn{3}{c}{optimization (predicted)} & \multicolumn{4}{c}{closed loop (realized)} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-8}",
         r"controller & KKT (med.) & $\le10^{-4}$ & cone [N] & "
         r"front $\lambda_N$ [N] & front unloaded & front at $\mu$ & hind at $\mu$ \\",
         r"\midrule"]
    for r in rows:
        L.append(r"%s & %s & %d/%d & %s & %.1f & %.0f\%% & %.0f\%% & %.0f\%% \\" % (
            r["name"], sci(r["kkt_med"]), r["n_1em4"], r["n"],
            sci(r["pred_cone_max"]) if "pred_cone_max" in r else "--",
            r["front_fn_med"], np.mean(r["real_unloaded_pct"][:2]),
            np.mean(r["front_boundary_pct"]), np.mean(r["hind_boundary_pct"])))
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(out, "table_go2_constraints.tex"), "w").write("\n".join(L))


if __name__ == "__main__":
    R = load(RUN)
    rows = summarize(R)
    os.makedirs(OUT, exist_ok=True)
    fig_feet(R, OUT)
    fig_pred(R, OUT)
    latex_table(rows, OUT)
    json.dump([{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in r.items()} for r in rows],
              open(os.path.join(OUT, "go2_constraint_summary.json"), "w"), indent=1)
    for r in rows:
        print(f"\n=== {r['name']}  ({r['file']})")
        print(f"  KKT median {r['kkt_med']:.2e}  max {r['kkt_max']:.2e}   "
              f"<=1e-4 {r['n_1em4']}/{r['n']}  <=1e-3 {r['n_1em3']}  <=1e-2 {r['n_1em2']}"
              + (f"   iters med {r['it_med']:.0f}" if "it_med" in r else ""))
        if "pred_cone_max" in r:
            print(f"  predicted: max cone {r['pred_cone_max']:.2e} N   "
                  f"max unilaterality {r['pred_uni_max']:.2e} N   "
                  f"per-cycle median {r['pred_cone_med']:.2e} N")
            if "pred_cone_max_conv" in r:
                print(f"             converged solves {r['pred_cone_max_conv']:.2e} N", end="")
            if "pred_cone_max_unconv" in r:
                print(f"   non-converged {r['pred_cone_max_unconv']:.2e} N", end="")
            print()
        print(f"  realized : min Fn {r['real_fn_min']:.3f} N   max ratio {r['real_ratio_max']:.3f}"
              f"   unloaded per foot {['%.0f%%' % p for p in r['real_unloaded_pct']]}"
              f"   at cone boundary {r['real_sliding_pct']:.0f}% of loaded samples")
    print("\nwrote figures + table to", OUT)

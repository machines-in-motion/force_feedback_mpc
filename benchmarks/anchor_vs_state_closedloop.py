"""
Closed-loop comparison of the anchor-update and force-state transcriptions on the Go2
multi-contact task (complement to the Reviewer-1 / Major-2 answer).

Same task, same contact parameters, same costs, same hard constraints, same solver and
same regularization floor; only the transcription differs.

Usage:
    python anchor_vs_state_closedloop.py <force_state_npz_dir> <anchor_npz_dir> [out_dir]
"""
import glob
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MU, N_FEET, FN_EPS = 0.75, 4, 1e-6
STATE_DIR = sys.argv[1]
ANCHOR_DIR = sys.argv[2]
OUT = sys.argv[3] if len(sys.argv) > 3 else ANCHOR_DIR
C_STATE, C_ANCHOR = "#1b6ca8", "#c1492b"

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True, "legend.frameon": False,
                     "figure.dpi": 150})


def cone_viol(F):
    return np.maximum(np.linalg.norm(F[..., :2], axis=-1) - MU * F[..., 2], 0.0)


def load(d, pat, first_node):
    f = glob.glob(os.path.join(d, pat))
    if not f:
        raise SystemExit(f"no {pat} in {d}")
    z = np.load(f[0], allow_pickle=True)
    M = z["measured_forces"].item()
    names = list(z["ee_frame_names"])
    P = np.asarray(z["ocp_forces"], float)[:, first_node:, :N_FEET]
    return dict(file=os.path.basename(f[0]),
                meas=np.stack([np.asarray(M[n], float) for n in names], axis=1),
                kkt=np.asarray(z["kkt_norm"], float), pred=P,
                desired=np.asarray(z["desired_forces"], float),
                iters=np.asarray(z["sqp_iters"], float) if "sqp_iters" in z else None,
                q=np.asarray(z["jointPos"], float))


def stats(r, name):
    cv = cone_viol(r["pred"])
    fn = r["meas"][:, :N_FEET, 2]
    ee = np.linalg.norm(r["meas"][:, -1], axis=-1)
    des = np.linalg.norm(r["desired"], axis=-1)
    return dict(name=name, file=r["file"], n=int(len(r["kkt"])),
                kkt_med=float(np.median(r["kkt"])), kkt_max=float(r["kkt"].max()),
                n_1em4=int((r["kkt"] <= 1e-4).sum()), n_1em3=int((r["kkt"] <= 1e-3).sum()),
                n_1em2=int((r["kkt"] <= 1e-2).sum()),
                pred_cone_max=float(cv.max()), pred_cone_med=float(np.median(cv.max(axis=(1, 2)))),
                pred_uni_max=float(np.maximum(-r["pred"][..., 2], 0).max()),
                front_fn_med=float(np.median(fn[:, :2])),
                front_unloaded=float(100 * (fn[:, :2] <= FN_EPS).mean()),
                ee_err_mean=float(np.mean(np.abs(ee - des))),
                base_z_min=float(r["q"][:, 2].min()),
                it_med=float(np.median(r["iters"])) if r["iters"] is not None else None)


def sci(v, floor=1e-12):
    v = max(float(v), floor)
    e = int(np.floor(np.log10(v)))
    return r"$%.1f\times10^{%d}$" % (v / 10 ** e, e)


if __name__ == "__main__":
    S = load(STATE_DIR, "go2_soft*.npz", 1)        # node 0 is the measured force
    A = load(ANCHOR_DIR, "go2_anchor*.npz", 0)
    rows = [stats(S, "force state (proposed)"), stats(A, "anchor update")]

    fig, ax = plt.subplots(1, 4, figsize=(11.0, 2.5))
    for r, c, nm in ((S, C_STATE, "force state (proposed)"), (A, C_ANCHOR, "anchor update")):
        cv = cone_viol(r["pred"]).max(axis=(1, 2))
        ax[0].semilogy(np.arange(len(cv)) * 1e-2, np.maximum(cv, 1e-12), color=c, lw=0.9, label=nm)
        ax[1].semilogy(np.arange(len(r["kkt"])) * 1e-2, r["kkt"], color=c, lw=0.9)
        t = np.arange(len(r["meas"])) * 1e-3
        keep = t >= 0.05
        ax[2].plot(t[keep], r["meas"][keep, :2, 2].min(axis=1), color=c, lw=0.9)
        ee = np.linalg.norm(r["meas"][:, -1], axis=-1)
        ax[3].plot(t, ee, color=c, lw=0.9)
    ax[2].axhline(0.0, color="k", ls="--", lw=0.8)
    ax[3].axhline(np.linalg.norm(S["desired"][-1]), color="k", ls="--", lw=0.8)
    ax[0].set_ylabel("max predicted cone violation [N]")
    ax[1].set_ylabel("KKT residual")
    ax[1].axhline(1e-4, color="k", ls=":", lw=0.8)
    ax[2].set_ylabel(r"realized front $\lambda_N$ [N]")
    ax[3].set_ylabel(r"realized $\|\lambda_{\rm ee}\|$ [N]")
    for a, t_ in zip(ax, ("(a) OCP constraint satisfaction", "(b) solver accuracy",
                          "(c) front-foot contact", "(d) end-effector force")):
        a.set_xlabel("time [s]")
        a.set_title(t_, fontsize=9, loc="left")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_anchor_vs_state_closedloop.{ext}"))

    L = [r"\begin{tabular}{lcccccc}", r"\toprule",
         r"transcription & KKT (med.) & $\le10^{-2}$ & max pred.\ cone [N] & "
         r"med.\ pred.\ cone [N] & front $\lambda_N$ [N] & SQP it. \\", r"\midrule"]
    for r in rows:
        L.append(r"%s & %s & %d/%d & %s & %s & %.1f & %d \\" % (
            r["name"], sci(r["kkt_med"]), r["n_1em2"], r["n"], sci(r["pred_cone_max"]),
            sci(r["pred_cone_med"]), r["front_fn_med"], r["it_med"] or 0))
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "table_anchor_vs_state_closedloop.tex"), "w").write("\n".join(L))
    json.dump(rows, open(os.path.join(OUT, "anchor_vs_state_closedloop.json"), "w"), indent=1)

    for r in rows:
        print(f"\n=== {r['name']}  ({r['file']})")
        print(f"  KKT median {r['kkt_med']:.2e} max {r['kkt_max']:.2e}  "
              f"<=1e-4 {r['n_1em4']}/{r['n']}  <=1e-2 {r['n_1em2']}/{r['n']}  "
              f"iters med {r['it_med']:.0f}")
        print(f"  predicted cone violation: max {r['pred_cone_max']:.3e} N   "
              f"per-cycle median {r['pred_cone_med']:.3e} N   "
              f"unilaterality max {r['pred_uni_max']:.2e} N")
        print(f"  realized: front lambda_N median {r['front_fn_med']:.2f} N   "
              f"front unloaded {r['front_unloaded']:.0f}%   "
              f"EE force error {r['ee_err_mean']:.1f} N   base z min {r['base_z_min']:.3f} m")
    print("\nwrote figure + table to", OUT)

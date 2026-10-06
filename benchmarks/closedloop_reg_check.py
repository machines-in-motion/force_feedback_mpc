"""
Is the closed-loop comparison of the two transcriptions an artifact of the solver
regularization?

The multi-contact experiment is rerun with both transcriptions at several regularization
floors. For each run we report what the controller achieves physically (whether the front
feet stay loaded, and the end-effector force reached) alongside how well the OCP was
solved, so that a difference in behaviour can be attributed to the transcription rather
than to the solver setting.

Usage:  python closedloop_reg_check.py [out_dir]
"""
import glob
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

B = "/home/skleff/Desktop/PUBLICATIONS/TRO-Soft/AURO/REVISION/GO2_RERUN/"
OUT = sys.argv[1] if len(sys.argv) > 1 else B + "E_reg_check"
FN_EPS = 1e-6
C_STATE, C_ANCHOR = "#1b6ca8", "#c1492b"

RUNS = [
    ("force state", 1e-3, B + "C_hessfix_reg1em3/go2_soft*.npz"),
    ("force state", 1e-2, B + "E_reg_check/go2_soft_reg1em2*.npz"),
    ("force state", 1e-1, B + "E_reg_check/go2_soft_reg1em1*.npz"),
    ("anchor", 1e-3, B + "D_anchor_closedloop/go2_anchor*.npz"),
    ("anchor", 1e-2, B + "E_reg_check/go2_anchor_reg1em2*.npz"),
    ("anchor", 1e-1, B + "E_reg_check/go2_anchor_reg1em1*.npz"),
]

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True, "legend.frameon": False,
                     "figure.dpi": 150})


def summarize(path):
    d = np.load(path, allow_pickle=True)
    M = d["measured_forces"].item()
    n = list(d["ee_frame_names"])
    fn = np.stack([np.asarray(M[k], float)[:, 2] for k in n[:4]], axis=1)
    ee = np.linalg.norm(np.asarray(M[n[-1]], float), axis=1)
    des = np.linalg.norm(np.asarray(d["desired_forces"], float), axis=1)
    kkt = np.asarray(d["kkt_norm"], float)
    it = np.asarray(d["sqp_iters"], float) if "sqp_iters" in d else np.array([np.nan])
    return dict(front_unloaded=float(100 * (fn[:, :2] <= FN_EPS).mean()),
                ee_final=float(ee[-200:].mean()), ee_target=float(des[-1]),
                base_z_min=float(np.asarray(d["jointPos"], float)[:, 2].min()),
                kkt_med=float(np.median(kkt)), n_1em2=int((kkt <= 1e-2).sum()),
                n_cycles=int(len(kkt)), it_med=float(np.median(it)),
                file=os.path.basename(path))


if __name__ == "__main__":
    rows = []
    for form, reg, pat in RUNS:
        g = glob.glob(pat)
        if not g:
            print(f"! missing: {form}, rho={reg:.0e}")
            continue
        r = summarize(g[0])
        r.update(form=form, reg=reg)
        rows.append(r)
    json.dump(rows, open(os.path.join(OUT, "closedloop_reg_check.json"), "w"), indent=1)

    print(f"{'formulation':14s} {'rho':>6s} {'front unloaded':>14s} {'EE |f| [N]':>11s} "
          f"{'base z':>8s} {'KKT med':>9s} {'<=1e-2':>8s} {'it med':>7s}")
    for r in rows:
        print(f"{r['form']:14s} {r['reg']:6.0e} {r['front_unloaded']:13.1f}% "
              f"{r['ee_final']:11.1f} {r['base_z_min']:8.3f} {r['kkt_med']:9.2e} "
              f"{r['n_1em2']:4d}/{r['n_cycles']:<3d} {r['it_med']:7.0f}")

    fig, ax = plt.subplots(1, 3, figsize=(9.5, 2.7))
    for form, c, m in (("force state", C_STATE, "o"), ("anchor", C_ANCHOR, "s")):
        g = sorted([r for r in rows if r["form"] == form], key=lambda r: r["reg"])
        if not g:
            continue
        x = [r["reg"] for r in g]
        ax[0].semilogx(x, [r["front_unloaded"] for r in g], color=c, marker=m, ms=4,
                       lw=1.3, label=form)
        ax[1].semilogx(x, [r["ee_final"] for r in g], color=c, marker=m, ms=4, lw=1.3)
        ax[2].loglog(x, [r["kkt_med"] for r in g], color=c, marker=m, ms=4, lw=1.3)
    if rows:
        ax[1].axhline(rows[0]["ee_target"], color="k", ls="--", lw=0.8)
    ax[2].axhline(1e-2, color="0.35", ls=":", lw=1)
    ax[0].set_ylabel("front feet unloaded [%]")
    ax[1].set_ylabel(r"realized $\|\lambda_{\rm ee}\|$ [N]")
    ax[2].set_ylabel("median KKT residual")
    for a, t in zip(ax, ("(a) stance", "(b) force tracking", "(c) solver accuracy")):
        a.set_xlabel(r"regularization floor $\rho$")
        a.set_title(t, fontsize=9, loc="left")
    ax[0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_closedloop_reg_check.{e}"))
    print("\nwrote", OUT)

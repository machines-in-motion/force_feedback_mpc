"""Regime-map figure: behaviour of both transcriptions as the prediction approaches the
nonsmooth apex of the friction cone (default solver regularization)."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/skleff/Desktop/PUBLICATIONS/TRO-Soft/AURO/REVISION/SMOOTH_REGIME"
import glob
R = json.load(open(os.path.join(OUT, "regime_map.json")))
DAMP = []
for f in glob.glob(os.path.join(OUT, "regime_map_reg1e-03.json")):
    DAMP = json.load(open(f))
F = sorted({r["f_des"] for r in R})
C_S, C_A = "#1b6ca8", "#c1492b"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True, "legend.frameon": False,
                     "figure.dpi": 150})

def med(form, key):
    return [np.median([r[key] for r in R if r["f_des"] == f and r["form"] == form]) for f in F]

fig, ax = plt.subplots(1, 3, figsize=(9.5, 2.7))
apex = [f for f in F if med("force state", "min_fn")[F.index(f)] <= 1e-3]
for a in ax:
    if apex:
        a.axvspan(min(apex) - 5, max(F) + 5, color="0.90", zorder=0)
    a.set_xlabel("end-effector force target [N]")
    a.set_xlim(min(F) - 3, max(F) + 3)

def medd(form, key, data, targets):
    return [np.median([r[key] for r in data if r["f_des"] == f and r["form"] == form])
            for f in targets]

for form, c, m in (("force state", C_S, "o"), ("anchor", C_A, "s")):
    ax[0].semilogy(F, med(form, "kkt"), color=c, marker=m, ms=4, lw=1.3,
                   label=rf"{form}, $\rho=10^{{-9}}$")
    ax[1].semilogy(F, np.maximum(med(form, "cone_viol"), 1e-9), color=c, marker=m, ms=4, lw=1.3)
    ax[2].plot(F, med(form, "min_fn"), color=c, marker=m, ms=4, lw=1.3)
if DAMP:
    FD = sorted({r["f_des"] for r in DAMP})
    for form, c, m in (("force state", C_S, "o"), ("anchor", C_A, "s")):
        ax[0].semilogy(FD, medd(form, "kkt", DAMP, FD), color=c, marker=m, ms=4, lw=1.3,
                       ls="--", mfc="white", label=rf"{form}, $\rho=10^{{-3}}$")
        ax[1].semilogy(FD, np.maximum(medd(form, "cone_viol", DAMP, FD), 1e-9), color=c,
                       marker=m, ms=4, lw=1.3, ls="--", mfc="white")
ax[0].axhline(1e-4, color="0.35", ls=":", lw=1)
ax[0].set_ylabel("KKT residual at termination")
ax[0].legend(loc="upper left", fontsize=6.5)
ax[0].set_title("(a) solver accuracy", fontsize=9, loc="left")
ax[1].set_ylabel("max predicted cone violation [N]")
ax[1].set_title("(b) constraint satisfaction", fontsize=9, loc="left")
ax[2].axhline(0.0, color="k", ls="--", lw=0.8)
ax[2].set_ylabel(r"min predicted $\lambda_N$ [N]")
ax[2].set_title("(c) distance to the cone apex", fontsize=9, loc="left")
ax[1].text(0.97, 0.06, "apex regime", transform=ax[1].transAxes, ha="right",
           fontsize=7.5, color="0.35")
fig.tight_layout()
for e in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"fig_regime_map.{e}"))
print("wrote", os.path.join(OUT, "fig_regime_map.pdf"))

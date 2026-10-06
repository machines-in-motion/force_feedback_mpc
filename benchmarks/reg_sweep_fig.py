"""Figure: sensitivity of each transcription to the solver's Riccati regularization,
at one smooth and one apex operating point."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/skleff/Desktop/PUBLICATIONS/TRO-Soft/AURO/REVISION/SMOOTH_REGIME"
R = json.load(open(os.path.join(OUT, "reg_sweep.json")))
REGIMES = list(dict.fromkeys(r["regime"] for r in R))
REGS = sorted({r["reg"] for r in R})
C_S, C_A = "#1b6ca8", "#c1492b"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True, "legend.frameon": False,
                     "figure.dpi": 150})

def frac_conv(regime, form):
    out = []
    for g in REGS:
        v = [r["converged"] for r in R if r["regime"] == regime and r["form"] == form
             and r["reg"] == g]
        out.append(sum(v) / max(len(v), 1))
    return np.array(out)


def med(regime, form, key):
    out = []
    for g in REGS:
        v = [r[key] for r in R if r["regime"] == regime and r["form"] == form and r["reg"] == g]
        out.append(np.median(v) if v else np.nan)
    return np.array(out)

fig, ax = plt.subplots(2, len(REGIMES), figsize=(7.0, 4.6), sharex=True)
ax = np.atleast_2d(ax)
for j, regime in enumerate(REGIMES):
    for form, c, m in (("force state", C_S, "o"), ("anchor", C_A, "s")):
        k, v, fc = med(regime, form, "kkt"), med(regime, form, "cone_viol"), frac_conv(regime, form)
        ax[0, j].loglog(REGS, k, color=c, lw=1.3, label=form, zorder=2)
        ax[1, j].loglog(REGS, np.maximum(v, 1e-10), color=c, lw=1.3, zorder=2)
        for a, y in ((ax[0, j], k), (ax[1, j], np.maximum(v, 1e-10))):
            sel = fc == 1.0
            a.plot(np.array(REGS)[sel], y[sel], ls="none", marker=m, ms=4.5, color=c, zorder=3)
            a.plot(np.array(REGS)[~sel], y[~sel], ls="none", marker=m, ms=4.5, color=c,
                   mfc="white", zorder=3)
    ax[0, j].axhline(1e-4, color="0.35", ls=":", lw=1)
    ax[0, j].set_title(regime, fontsize=9, loc="left")
    ax[1, j].set_xlabel(r"regularization floor $\rho$")
ax[0, 0].set_ylabel("KKT residual")
ax[1, 0].set_ylabel("max cone violation [N]")
ax[0, 0].legend(loc="upper left", fontsize=8)
ax[0, 0].plot([], [], ls="none", marker="o", ms=4.5, color="0.4", label="reached tolerance")
ax[0, 0].plot([], [], ls="none", marker="o", ms=4.5, color="0.4", mfc="white",
              label="did not converge")
ax[0, 0].legend(loc="upper left", fontsize=6.5)
fig.tight_layout()
for e in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"fig_reg_sweep.{e}"))
print("wrote", os.path.join(OUT, "fig_reg_sweep.pdf"))

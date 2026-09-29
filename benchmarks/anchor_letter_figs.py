"""
Figures and LaTeX table for the Reviewer-1 / Major-2 answer (anchor-update vs force-state).

Reads only saved data, so it is cheap to re-run while tweaking the plots:
    <RESULTS>/letter_data.json     produced by anchor_letter_data.py
    <RESULTS>/basin_summary.json   produced by anchor_basin_sweep.py (parsed summary)

Usage:
    python anchor_letter_figs.py [results_dir]
"""
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ----------------------------------------------------------------- customization
RESULTS = (sys.argv[1] if len(sys.argv) > 1 else
           "/home/skleff/Desktop/PUBLICATIONS/TRO-Soft/AURO/REVISION/ANCHOR_VS_STATE/")
SAMPLES = [80, 210, 330]          # the three frozen OCPs, in figure order
REGS = [1e-3, 1e-1]               # matched regularization floors to show
OCP_LABELS = ["OCP 1", "OCP 2", "OCP 3"]
BASIN_REG = 1e-1                  # regularization shown in the basin figure
C_STATE, C_ANCHOR = "#1b6ca8", "#c1492b"
FIG1 = "fig_anchor_vs_state_convergence"   # kept for compatibility with the letter
FIG2 = "fig_anchor_vs_state_basin"

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True, "legend.frameon": False,
                     "figure.dpi": 150})

D = json.load(open(os.path.join(RESULTS, "letter_data.json")))
RUNS = D["runs"]


def get(sample, form, reg):
    for r in RUNS:
        if r["sample"] == sample and r["form"] == form and abs(r["reg"] - reg) < 1e-12:
            return r
    raise KeyError((sample, form, reg))


# ------------------------------------------------- Figure 1: constraint violation
fig, ax = plt.subplots(figsize=(3.6, 2.7))
w = 0.2
xs = np.arange(len(SAMPLES))
for i, (reg, hatch) in enumerate(zip(REGS, ("", "///"))):
    for j, (form, c) in enumerate((("force-state", C_STATE), ("anchor", C_ANCHOR))):
        vals = [get(s, form, reg)["cone_max"] for s in SAMPLES]
        lbl = ("force state" if form == "force-state" else "anchor update")
        exp = int(np.round(np.log10(reg)))
        ax.bar(xs + (2 * i + j - 1.5) * w, vals, w, color=c, hatch=hatch,
               edgecolor="white", linewidth=0.5,
               label=rf"{lbl}, $\rho=10^{{{exp}}}$")
ax.set_yscale("log")
ax.set_xticks(xs)
ax.set_xticklabels(OCP_LABELS)
ax.set_ylabel("max friction-cone violation [N]")
ax.set_ylim(top=4.0)
ax.legend(fontsize=6.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.03),
          columnspacing=1.0, handlelength=1.6)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(RESULTS, f"{FIG1}.{ext}"))

# --------------------------------------------------------- Figure 2: basin sweep
basin_path = os.path.join(RESULTS, "basin_summary.json")
if os.path.exists(basin_path):
    B = [r for r in json.load(open(basin_path)) if abs(r["reg"] - BASIN_REG) < 1e-12]
    amps = sorted({r["amp"] for r in B})
    fig2, ax2 = plt.subplots(1, 2, figsize=(7.0, 2.7))
    x = np.arange(len(amps))
    for form, c, m in (("force-state", C_STATE, "o"), ("anchor", C_ANCHOR, "s")):
        med, frac = [], []
        for a in amps:
            g = [r for r in B if r["form"] == form and abs(r["amp"] - a) < 1e-12]
            med.append(np.median([r["kkt_med"] for r in g]))
            frac.append(100 * sum(r["n_le_1em2"] for r in g) / sum(r["n_tot"] for r in g))
        lbl = "force state" if form == "force-state" else "anchor update"
        ax2[0].semilogy(x, med, color=c, lw=1.4, marker=m, ms=4, label=lbl)
        ax2[1].plot(x, frac, color=c, lw=1.4, marker=m, ms=4, label=lbl)
    for a in ax2:
        a.set_xticks(x)
        a.set_xticklabels([f"{v:g}" for v in amps])
        a.set_xlabel("initial-guess perturbation amplitude")
    ax2[0].set_ylabel("median KKT residual")
    ax2[0].legend(loc="upper left", fontsize=8)
    ax2[0].set_title("(a) solution quality", fontsize=9, loc="left")
    ax2[1].set_ylabel(r"runs reaching KKT $\leq 10^{-2}$ [%]")
    ax2[1].set_ylim(-5, 105)
    ax2[1].legend(loc="lower left", fontsize=8)
    ax2[1].set_title("(b) robustness of the initialization", fontsize=9, loc="left")
    fig2.tight_layout()
    for ext in ("pdf", "png"):
        fig2.savefig(os.path.join(RESULTS, f"{FIG2}.{ext}"))
else:
    print(f"! {basin_path} missing, skipping the basin figure")


# ------------------------------------------------------------------------- table
def sci(v, p=1):
    e = int(np.floor(np.log10(v)))
    return r"$%.*f\times10^{%d}$" % (p, v / 10 ** e, e)


L = [r"\begin{tabular}{llcccccc}", r"\toprule",
     r"$\rho$ & transcription & $\dim x$ & KKT & cone viol.\ [N] & SQP it. & ms/it. & total [s] \\",
     r"\midrule"]
for reg in REGS:
    exp = int(np.round(np.log10(reg)))
    for i, (form, name) in enumerate((("force-state", "force state (proposed)"),
                                      ("anchor", "anchor update"))):
        g = [get(s, form, reg) for s in SAMPLES]
        L.append(r"%s & %s & %d & %s & %s & %d & %.0f & %.1f \\" % (
            rf"${{10^{{{exp}}}}}$" if i == 0 else "", name, g[0]["ndx"],
            sci(float(np.median([r["kkt"] for r in g]))),
            sci(max(r["cone_max"] for r in g)),
            np.median([r["iters"] for r in g]),
            np.median([r["ms_per_it"] for r in g]),
            np.median([r["ms"] for r in g]) / 1e3))
    L.append(r"\midrule" if reg != REGS[-1] else r"\bottomrule")
L.append(r"\end{tabular}")
open(os.path.join(RESULTS, "table_anchor_vs_state.tex"), "w").write("\n".join(L))

print("results directory:", RESULTS)
print("  figures:", f"{FIG1}.pdf/.png", f"{FIG2}.pdf/.png")
print("  table  :", "table_anchor_vs_state.tex")
print("\nconstraint-Jacobian scaling (from letter_data.json):")
for k, v in D["jac"].items():
    print(f"  OCP at sample {k}: |dg/dlambda|={v['g_lambda']:.3f}  |dg/dx|={v['g_x']:.1f}"
          f"  ratio={v['g_x'] / v['g_lambda']:.0f}x")

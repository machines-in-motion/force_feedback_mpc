"""
Figure and LaTeX table for the analytical vs numerical derivative benchmark.

Inputs (both produced by the benchmarks, not recomputed here):
    <OUT>/deriv_timing_go2.json     from deriv_timing_go2.py
    <OUT>/deriv_iiwa.log            stdout of build/benchmarks/soft_derivatives
                                    (its RANDTIME lines carry the per-point medians)

Usage:
    python deriv_timing_figs.py [out_dir]
"""
import json
import os
import re
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/skleff/Desktop/PUBLICATIONS/TRO-Soft/AURO/REVISION/DERIVATIVES"
METHODS = ["analytical", "forward FD", "central FD", "crocoddyl NumDiff"]
COLORS = ["#1b6ca8", "#6c8ebf", "#c1492b", "#7a7a7a"]
IIWA_NC = 3          # contact dimension of the iiwa model to display

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True, "legend.frameon": False,
                     "figure.dpi": 150})


def load_iiwa(path):
    """per-operating-point medians, in us, keyed by contact dimension"""
    out = {}
    for line in open(path, errors="ignore"):
        if line.startswith("RANDTIME"):
            t = line.split()
            out.setdefault(int(t[1]), []).append([float(v) for v in t[3:7]])
    return {k: np.array(v) for k, v in out.items()}


def load_iiwa_solve(path):
    """the FDDP solve block of the last (highest nc) section"""
    txt = open(path, errors="ignore").read()
    blocks = re.findall(
        r"analytical models : ([\d.]+) ms, (\d+) iterations, ([\d.]+) ms/it.*?"
        r"NumDiff models    : ([\d.]+) ms, (\d+) iterations, ([\d.]+) ms/it", txt, re.S)
    if not blocks:
        return None
    b = blocks[-1]
    return dict(ana_ms=float(b[0]), ana_it=int(b[1]), ana_ms_it=float(b[2]),
                nd_ms=float(b[3]), nd_it=int(b[4]), nd_ms_it=float(b[5]))


def sci(v, p=1):
    e = int(np.floor(np.log10(abs(v)))) if v else 0
    return r"$%.*f\times10^{%d}$" % (p, v / 10 ** e, e)


def main():
    iiwa = load_iiwa(os.path.join(OUT, "deriv_iiwa.log"))
    go2 = json.load(open(os.path.join(OUT, "deriv_timing_go2.json")))
    gper = go2["per_node_ms"]

    # ------------------------------------------------------------------ figure
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for a, (title, means, stds, unit) in zip(ax, [
        ("(a) KUKA iiwa (C++), per node",
         [iiwa[IIWA_NC][:, j].mean() for j in range(4)],
         [iiwa[IIWA_NC][:, j].std() for j in range(4)], r"time [$\mu$s]"),
        ("(b) Go2 + arm (Python), per node",
         [np.mean(gper[m]) for m in METHODS],
         [np.std(gper[m]) for m in METHODS], "time [ms]"),
    ]):
        x = np.arange(len(METHODS))
        a.bar(x, means, 0.62, yerr=stds, color=COLORS, edgecolor="white", linewidth=0.5,
              error_kw=dict(ecolor="0.25", capsize=2.5, lw=0.9))
        for i, (m, s_) in enumerate(zip(means, stds)):
            a.text(i, m * 1.6, f"{m / means[0]:.0f}x" if i else "1x",
                   ha="center", fontsize=7.5, color="0.25")
        a.set_yscale("log")
        a.set_xticks(x)
        a.set_xticklabels(["analytical", "forward\nFD", "central\nFD", "crocoddyl\nNumDiff"],
                          fontsize=7.5)
        a.set_ylabel(unit)
        a.set_title(title, fontsize=9, loc="left")
        a.set_ylim(top=max(means) * 12)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_deriv_timing.{ext}"))

    # ------------------------------------------------------------------- table
    sv = go2.get("solve", {})
    isv = load_iiwa_solve(os.path.join(OUT, "deriv_iiwa.log"))
    L = [r"\begin{tabular}{llccc}", r"\toprule",
         r"model & quantity & analytical & forward FD & central FD \\", r"\midrule"]
    im = [iiwa[IIWA_NC][:, j] for j in range(4)]
    L.append(r"\multirow{2}{*}{iiwa (C++)} & one node [$\mu$s] & $%.1f\pm%.2f$ & "
             r"$%.1f\pm%.2f$ & $%.1f\pm%.2f$ \\" %
             (im[0].mean(), im[0].std(), im[1].mean(), im[1].std(), im[2].mean(), im[2].std()))
    L.append(r" & speedup & -- & $%.1f\times$ & $%.1f\times$ \\" %
             (im[1].mean() / im[0].mean(), im[2].mean() / im[0].mean()))
    L.append(r"\midrule")
    gm = {m: np.array(gper[m]) for m in METHODS}
    L.append(r"\multirow{3}{*}{Go2+arm (Python)} & one node [ms] & $%.2f\pm%.2f$ & "
             r"$%.1f\pm%.1f$ & $%.1f\pm%.1f$ \\" %
             (gm["analytical"].mean(), gm["analytical"].std(),
              gm["forward FD"].mean(), gm["forward FD"].std(),
              gm["central FD"].mean(), gm["central FD"].std()))
    hz = go2["horizon_ms"]
    L.append(r" & horizon $N=20$ [ms] & %.0f & %.0f & %.0f \\" %
             (hz["analytical"], hz["forward FD"], hz["central FD"]))
    L.append(r" & speedup & -- & $%.1f\times$ & $%.1f\times$ \\" %
             (gm["forward FD"].mean() / gm["analytical"].mean(),
              gm["central FD"].mean() / gm["analytical"].mean()))
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "table_deriv_timing.tex"), "w").write("\n".join(L))

    # --------------------------------------------------------------- printout
    print(f"{'':26s} {'analytical':>14s} {'forward FD':>14s} {'central FD':>14s} {'NumDiff':>14s}")
    print(f"{'iiwa, per node [us]':26s} " +
          " ".join(f"{im[j].mean():8.1f}+-{im[j].std():4.2f}" for j in range(4)))
    print(f"{'Go2, per node [ms]':26s} " +
          " ".join(f"{gm[m].mean():8.2f}+-{gm[m].std():4.2f}" for m in METHODS))
    print(f"{'Go2, horizon N=20 [ms]':26s} " +
          " ".join(f"{hz[m]:14.0f}" for m in METHODS))
    if isv:
        print(f"\niiwa full solve (FDDP, N=5): analytical {isv['ana_ms']:.2f} ms "
              f"({isv['ana_it']} it, {isv['ana_ms_it']:.2f} ms/it)  |  NumDiff "
              f"{isv['nd_ms']:.2f} ms ({isv['nd_it']} it, {isv['nd_ms_it']:.2f} ms/it)  "
              f"-> {isv['nd_ms'] / isv['ana_ms']:.0f}x total, "
              f"{isv['nd_ms_it'] / isv['ana_ms_it']:.0f}x per iteration")
    if sv:
        print(f"\nGo2 full solve (CSQP): analytical {sv['ms']:.0f} ms, {sv['iters']} it, "
              f"{sv['ms_per_iter']:.1f} ms/it = {sv['deriv_ms']:.1f} derivatives + "
              f"{sv['other_ms']:.1f} solver")
        for m in METHODS[1:]:
            k = f"proj_{m}"
            if k in sv:
                print(f"   projected with {m:20s}: {sv[k]['ms_per_iter']:9.1f} ms/it "
                      f"({sv[k]['speedup']:.1f}x slower)")
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()

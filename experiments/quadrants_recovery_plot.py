"""Quadrants recovery figure -- plotting step (reads quadrants_recovery_summary.csv).

Grouped bar chart over the four scenarios, two panels: (a) rho (KL), (b) gamma (odds ratio), each
comparing the observed-covariate benchmark against the per-x worst-case value U induces. rho_bench
>= rho_true in every scenario -> the IB belief holds. Run quadrants_recovery_run.py first.
Writes quadrants_recovery.png.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = os.path.join(os.path.dirname(__file__), "quadrants_recovery_summary.csv")
OUT = os.path.join(os.path.dirname(__file__), "quadrants_recovery.png")

SHORT = {
    "both_robust":          "weak $X$,\nweak $U$",
    "rho_robust_gamma_not": "rare-strong\n$X$",
    "gamma_robust_rho_not": "common-\nmoderate $X$",
    "both_not_robust":      "strong $X$,\nstrong $U$",
}
C_BENCH, C_TRUE = "#4878a8", "#555555"   # observed-covariate benchmark / true value of U


def _panel(ax, x, w, labels, bench, bench_sd, true_v, title, ylab):
    ax.bar(x - w / 2, bench, w, yerr=bench_sd, color=C_BENCH, capsize=2.5,
           error_kw=dict(lw=0.8), label=r"benchmark (observed $X$)")
    ax.bar(x + w / 2, true_v, w, color=C_TRUE, alpha=0.85,
           label=r"true value (per-$x$ worst case)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.margins(y=0.16)


def main():
    if not os.path.exists(CSV):
        print(f"No data at {CSV}.\nRun:  python {os.path.join('experiments', 'quadrants_recovery_run.py')}")
        sys.exit(1)
    df = pd.read_csv(CSV)
    labels = [SHORT.get(s, s) for s in df["scenario"]]
    x = np.arange(len(df))
    w = 0.38

    plt.rcParams.update({
        "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
        "xtick.labelsize": 6.2, "ytick.labelsize": 6, "legend.fontsize": 6.2,
        "axes.titlepad": 4, "axes.linewidth": 0.7,
    })
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.4, 2.9))

    _panel(axA, x, w, labels, df["rho_bench"], df["rho_bench_std"], df["rho_true"],
           r"(a) f-sensitivity $\rho$ (KL)", r"$\rho$")
    _panel(axB, x, w, labels, df["gamma_bench"], df["gamma_bench_std"], df["gamma_true"],
           r"(b) MSM $\Gamma$ (odds ratio)", r"$\Gamma$")
    axB.axhline(1.0, color="0.6", lw=0.7, ls=":")   # no-confounding reference on the MSM scale

    handles, lbls = axA.get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.9, bottom=0.27, wspace=0.22)
    fig.savefig(OUT, dpi=600)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

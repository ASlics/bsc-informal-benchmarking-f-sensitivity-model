"""
Heterogeneous-confounding failure -- plotting step (the mirror of spike_tail_plot.py).

Renders the three-panel figure from the results written by heterogeneous_run.py
(experiments/data/heterogeneous.json). If that file is absent, it prints how to produce it and
exits. This step is light (no DGP), so the figure can be re-styled cheaply.

Panels: (a) measured OR(x,u) for X0's bump at a few marching centres over g(u), with the in-bulk
centre as the within-covariate baseline; (b) Gamma_j vs centre (stays matched); (c) rho_j vs
centre (collapses as overlap shrinks). The first bar (centre 0.60) is the in-bulk baseline.

Run:  python experiments/heterogeneous_plot.py        (after heterogeneous_run.py)
Writes heterogeneous_experiment.png next to this file.
"""
import os
import sys
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "heterogeneous.json")
OUT_PATH = os.path.join(os.path.dirname(__file__), "heterogeneous_experiment.png")


def _arr(seq):
    """JSON list (with nulls for NaN) -> float array with np.nan."""
    return np.array([np.nan if v is None else v for v in seq], dtype=float)


def main():
    if not os.path.exists(DATA_PATH):
        print(f"No data at {DATA_PATH}.\n"
              f"Run the DGP first:  python {os.path.join('experiments', 'heterogeneous_run.py')}")
        sys.exit(1)

    with open(DATA_PATH) as fh:
        d = json.load(fh)

    meta = d["meta"]
    U = _arr(meta["U"])
    G = _arr(meta["G"])
    col_w = meta.get("col_w", 3.35)

    rows = d["rows"]
    centers = np.array([r["center"] for r in rows], dtype=float)
    overlaps = np.array([r["overlap"] for r in rows], dtype=float)
    rho_x0 = np.array([r["rho_x0"] for r in rows], dtype=float)
    rho_x0_sd = np.array([r["rho_x0_sd"] for r in rows], dtype=float)
    gam_x0 = np.array([r["gam_x0"] for r in rows], dtype=float)
    gam_x0_sd = np.array([r["gam_x0_sd"] for r in rows], dtype=float)
    or_x0 = [_arr(c) for c in d["or_x0"]]
    n = len(centers)

    # --- consistent, paper-friendly typography (matches spike_tail_plot.py) -- #
    plt.rcParams.update({
        "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
        "axes.titlepad": 4, "axes.linewidth": 0.7,
    })

    # muted single-hue gradient, dark -> light; the in-bulk baseline (centre 0.60, the first
    # sweep point) is a neutral grey -- mirrors the no-spike baseline of the spike figure.
    base_color = "0.5"
    cmap = plt.get_cmap("Blues")
    march = np.arange(1, n)                          # centres after the in-bulk baseline
    span = cmap(np.linspace(0.9, 0.45, len(march)))  # darker near bulk, lighter into the tail
    cen_color = {0: base_color}
    for k, idx in enumerate(march):
        cen_color[int(idx)] = span[k]

    # ---------------------------- figure ----------------------------------- #
    fig, (axD, axG, axR) = plt.subplots(
        3, 1, figsize=(col_w, 5.2),
        gridspec_kw=dict(hspace=0.5, height_ratios=[1.25, 1.0, 1.0]))

    # (a) distributions: X0 bump at a few marching centres ------------------- #
    for i in range(n):
        if i == 0:
            axD.plot(U, or_x0[i], "--", color=base_color, lw=1.6, zorder=4)
        else:
            axD.plot(U, or_x0[i], "-", color=cen_color[i], lw=1.4, zorder=3)
    axD.axhline(1.0, color="black", lw=0.6, alpha=0.35, zorder=2)
    axD.set_xlim(U[0], U[-1])
    axD.set_ylim(bottom=0)
    axD.set_xlabel(r"covariate value  $u$")
    axD.set_ylabel(r"measured OR$(x, u)$")
    axD.set_title(r"(a) measured OR$(x, u)$: a bump marching into g's tail")

    handles = [Line2D([0], [0], color=base_color, ls="--", lw=1.6,
                      label=f"{centers[0]:.2f} (baseline, in bulk)")]
    handles += [Line2D([0], [0], color=cen_color[int(i)], lw=1.4, label=f"{centers[i]:.2f}")
                for i in march]
    leg = axD.legend(handles=handles, loc="upper right", ncol=2, handlelength=1.3,
                     columnspacing=1.0, labelspacing=0.3, framealpha=0.9,
                     title="bump centre", title_fontsize=5.5)
    leg.get_frame().set_linewidth(0.5)

    pos = np.arange(n)
    xlabels = [f"{centers[i]:.2f}\n{overlaps[i]:.3f}" for i in range(n)]
    bar_colors = [cen_color[i] for i in range(n)]

    # (b) Gamma_j stays matched as the centre marches left -------------------- #
    axG.bar(pos, gam_x0, 0.68, yerr=gam_x0_sd, color=bar_colors,
            capsize=2.5, edgecolor="black", linewidth=0.5, error_kw=dict(lw=0.8))
    for p in pos:
        axG.text(p, gam_x0[p] + gam_x0_sd[p] + 0.06, f"{gam_x0[p]:.2f}",
                 ha="center", va="bottom", fontsize=5.5)
    axG.set_xticks(pos)
    axG.set_xticklabels(xlabels, fontsize=5.5)
    axG.set_xlabel(r"bump centre  /  overlap")
    axG.set_ylabel(r"$X_0$ worst-case $\Gamma_j$")
    axG.set_title(r"(b) MSM benchmark $\Gamma_j$ stays matched")
    axG.margins(y=0.22)

    # (c) rho_j collapses as overlap shrinks ---------------------------------- #
    axR.bar(pos, rho_x0, 0.68, yerr=rho_x0_sd, color=bar_colors,
            capsize=2.5, edgecolor="black", linewidth=0.5, error_kw=dict(lw=0.8))
    for p in pos:
        axR.text(p, rho_x0[p] + rho_x0_sd[p] + 0.0015, f"{rho_x0[p]:.3f}",
                 ha="center", va="bottom", fontsize=5.5)
    axR.set_xticks(pos)
    axR.set_xticklabels(xlabels, fontsize=5.5)
    axR.set_xlabel(r"bump centre  /  overlap")
    axR.set_ylabel(r"$X_0$ average $\rho_j$")
    axR.set_title(r"(c) f-sensitivity benchmark $\rho_j$ collapses")
    axR.margins(y=0.24)

    fig.subplots_adjust(left=0.16, right=0.86, top=0.95, bottom=0.07)
    fig.savefig(OUT_PATH, dpi=200)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()

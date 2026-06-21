"""Three-covariate IB demo (Gamma flat, rho rises) -- plotting step (reads
experiments/data/three_cov.json).

Three panels: (a) measured OR(x,u) for X0's bump at its marching centres (X0-at-0.8 baseline
dashed); (b) Gamma_bench vs X0 centre (matched); (c) rho_bench vs X0 centre (rises). Run
three_cov_run.py first. Writes three_cov_experiment.png.
"""
import os
import sys
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "three_cov.json")
OUT_PATH = os.path.join(os.path.dirname(__file__), "three_cov_experiment.png")


def _arr(seq):
    return np.array([np.nan if v is None else v for v in seq], dtype=float)


def main():
    if not os.path.exists(DATA_PATH):
        print(f"No data at {DATA_PATH}.\n"
              f"Run the DGP first:  python {os.path.join('experiments', 'three_cov_run.py')}")
        sys.exit(1)

    with open(DATA_PATH) as fh:
        d = json.load(fh)
    meta = d["meta"]
    U = _arr(meta["U"])
    col_w = meta.get("col_w", 3.35)

    rows = d["rows"]
    centers = np.array([r["center"] for r in rows], dtype=float)
    rho_bench = np.array([r["rho_bench"] for r in rows], dtype=float)
    rho_bench_sd = np.array([r["rho_bench_sd"] for r in rows], dtype=float)
    gam_bench = np.array([r["gam_bench"] for r in rows], dtype=float)
    gam_bench_sd = np.array([r["gam_bench_sd"] for r in rows], dtype=float)
    or_x0 = [_arr(c) for c in d["or_x0"]]
    n = len(centers)

    plt.rcParams.update({
        "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
        "axes.titlepad": 4, "axes.linewidth": 0.7,
    })

    # single-hue gradient dark -> light; the X0-at-0.8 step (= X1/X2 anchor) is neutral grey
    base_color = "0.5"
    cmap = plt.get_cmap("Blues")
    march = np.arange(1, n)
    span = cmap(np.linspace(0.9, 0.45, len(march)))
    cen_color = {0: base_color}
    for k, idx in enumerate(march):
        cen_color[int(idx)] = span[k]

    fig, (axD, axG, axR) = plt.subplots(
        3, 1, figsize=(col_w, 5.2),
        gridspec_kw=dict(hspace=0.5, height_ratios=[1.25, 1.0, 1.0]))

    # (a) distributions
    for i in range(n):
        if i == 0:
            axD.plot(U, or_x0[i], "--", color=base_color, lw=1.6, zorder=4)
        else:
            axD.plot(U, or_x0[i], "-", color=cen_color[i], lw=1.4, zorder=3)
    axD.axhline(1.0, color="black", lw=0.6, alpha=0.35, zorder=2)
    axD.set_xlim(U[0], U[-1])
    ytop = float(np.nanmax([np.nanmax(c) for c in or_x0]))
    axD.set_ylim(0, ytop * 1.7)
    axD.set_xlabel(r"covariate value  $u$")
    axD.set_ylabel(r"measured OR$(x, u)$")
    axD.set_title(r"(a) measured OR$(x, u)$")

    handles = [Line2D([0], [0], color=base_color, ls="--", lw=1.6, label=f"{centers[0]:.1f}")]
    handles += [Line2D([0], [0], color=cen_color[int(i)], lw=1.4, label=f"{centers[i]:.1f}")
                for i in march]
    leg = axD.legend(handles=handles, loc="upper center", ncol=4, handlelength=1.1,
                     columnspacing=0.8, labelspacing=0.25, handletextpad=0.4,
                     borderpad=0.4, framealpha=0.9, title=r"$X_0$ centre", title_fontsize=5.5)
    leg.get_frame().set_linewidth(0.5)

    pos = np.arange(n)
    xlabels = [f"{centers[i]:.2f}" for i in range(n)]
    bar_colors = [cen_color[i] for i in range(n)]

    # (b) Gamma_bench stays matched
    axG.bar(pos, gam_bench, 0.68, yerr=gam_bench_sd, color=bar_colors,
            capsize=2.5, edgecolor="black", linewidth=0.5, error_kw=dict(lw=0.8))
    for p in pos:
        axG.text(p, gam_bench[p] + gam_bench_sd[p] + 0.06, f"{gam_bench[p]:.2f}",
                 ha="center", va="bottom", fontsize=5.5)
    axG.set_xticks(pos); axG.set_xticklabels(xlabels, fontsize=5.5)
    axG.set_xlabel(r"$X_0$ bump centre")
    axG.set_ylabel(r"$\Gamma_{\mathrm{bench}}$")
    axG.set_title(r"(b) MSM benchmark $\Gamma_{\mathrm{bench}}$")
    axG.margins(y=0.22)

    # (c) rho_bench rises
    axR.bar(pos, rho_bench, 0.68, yerr=rho_bench_sd, color=bar_colors,
            capsize=2.5, edgecolor="black", linewidth=0.5, error_kw=dict(lw=0.8))
    for p in pos:
        axR.text(p, rho_bench[p] + rho_bench_sd[p] + 0.0015, f"{rho_bench[p]:.3f}",
                 ha="center", va="bottom", fontsize=5.5)
    axR.set_xticks(pos); axR.set_xticklabels(xlabels, fontsize=5.5)
    axR.set_xlabel(r"$X_0$ bump centre")
    axR.set_ylabel(r"$\rho_{\mathrm{bench}}$")
    axR.set_title(r"(c) f-sensitivity benchmark $\rho_{\mathrm{bench}}$")
    axR.margins(y=0.24)

    fig.subplots_adjust(left=0.16, right=0.86, top=0.95, bottom=0.07)
    fig.savefig(OUT_PATH, dpi=600, bbox_inches="tight", pad_inches=0.03)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()

"""Informal-benchmarking (IB) library: the faithful f-sensitivity benchmark rho, the
Figure-1 empirical-curve experiment, and the quadrant data generators.

Benchmark rho (rho_bench) — a yardstick from the OBSERVED covariates and treatment: dropping
X_j, the rho it consumes is the symmetric KL between the treated/control distributions of X_j
measured over the WHOLE sample (POOLED, not conditioned on the other covariates X_{-j}):
    rho_j     = max(D_KL(P(X_j|T=1)||P(X_j|T=0)), D_KL(P(X_j|T=0)||P(X_j|T=1)))
    rho_bench = mean_j rho_j           (the AVERAGE observed covariate; Gamma_bench = max_j Gamma_j)
Mirrors the MSM benchmark Gamma_j = max_v OR(v) on the same likelihood-ratio object, but
f-sensitivity reads an average divergence where the MSM takes the worst-case level. There is no
stratification (no max_s / mean_s): the pooled imbalance is confounded by any kept covariate
correlated with X_j, so the DGP keeps the observed covariates independent.

This is the shared library imported by the paper's experiment runners (heterogeneous_run,
spike_tail_run, quadrants_multiseed); it needs only numpy/pandas/matplotlib. Run it directly
to (re)generate the Figure-1 empirical-curve figure.
"""
import os
import sys
import tempfile
import warnings

# Force UTF-8 stdout so the Greek rho printed by callers survives redirection on Windows.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import numpy as np
import pandas as pd
import matplotlib
if __name__ == "__main__":          # headless script run: render to file, never to a display
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Compact single-column style for the two-column layout (\columnwidth ~3in).
plt.rcParams.update({
    "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6,
    "savefig.dpi": 600, "savefig.bbox": "tight", "figure.constrained_layout.use": False,
})
COL_W = 3.35   # target single-column width in inches

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_generation.Generator import Generator

# Generator.generate shares one global NumPy RNG across n_jobs threads, so n_jobs>1 makes the
# draws non-reproducible; passing N_JOBS=1 lets a single seed fully determine the data.
N_JOBS = 1


# =========================================================================== #
# Data-generating process
# =========================================================================== #
# Binary X_j ~ Bern(p_x[j]) independent of each other and of U ~ Bern(u_prob); logistic
# treatment P(T=1|X,U) = sigmoid(t_base + sum_j beta_t[j] X_j + t_u*U) and outcome
# Y = mean(X) + y_effect*U + true_ate*T + noise (noise ~ Normal(-1, 0.1)). beta_t sets the
# per-covariate confounding the benchmark reads (larger -> larger rho_j). Used here to build
# the quadrant generators. ROLE labels are retained only for figure layout.
ROLE = {"X0": "strong", "X1": "high", "X2": "moderate", "X3": "mild", "X4": "weak"}


def make_generator_ib(p_x=(0.50, 0.45, 0.40, 0.55, 0.50), beta_t=(0.5, 0.5, 0.5, 0.5, 0.5),
                      u_prob=0.40, t_base=-0.85, t_u=-0.8, y_effect=0.7, true_ate=1.0):
    p_x, beta_t = list(p_x), list(beta_t)
    dim = len(p_x)
    sizes = {"U": 1, "X": dim, "T": 1, "Y": 1}

    def _x_gen(u, noise):
        return [1 if np.random.rand() < p_x[i] else 0 for i in range(dim)]

    def _t_gen(u, x, noise):
        logit = t_base + sum(beta_t[i] * x[i] for i in range(dim)) + t_u * u[0]
        p = 1.0 / (1.0 + np.exp(-logit))
        return [1 if np.random.rand() < p else 0]

    generators = {
        "U": lambda noise: [1 if np.random.rand() < u_prob else 0],
        "X": _x_gen,
        "T": _t_gen,
        "Y": lambda u, x, t, noise: [round(sum(x) / dim + y_effect * u[0] + true_ate * t[0] + noise, 1)],
    }
    noise = {"U": lambda: 0, "X": lambda: 0, "T": lambda: 0,
             "Y": lambda: 0.1 * np.random.randn() - 1}
    return Generator(generators=generators, noise_generators=noise, sizes=sizes)


def x_cols_for(p_x):
    return [f"X{i}" for i in range(len(p_x))]


# =========================================================================== #
# The faithful benchmark rho (from observed covariates and treatment)
# =========================================================================== #
def kl_discrete(p, q, eps=1e-12):
    """D_KL(p || q) for discrete distributions. Terms with p_i == 0 contribute 0; the
    denominator is floored at eps so an empty cell does not produce inf."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / p.sum()
    q = np.clip(q / q.sum(), eps, None)
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def benchmark_per_covariate(df, x_cols, t_col="T"):
    """Drop each X_j and measure the arm shift over the WHOLE sample (POOLED): compare X_j's
    treated/control distributions directly, without conditioning on the other covariates X_{-j}.
        rho_j   = symmetric-KL(P(X_j|T=1), P(X_j|T=0))   (f-sensitivity, KL)
        gamma_j = max_v max(OR(v), 1/OR(v))               (MSM / Tan companion, worst level)
    No stratification, so there is no max_s / mean_s -- each covariate yields a single rho_j and
    gamma_j. The pooled imbalance is confounded by any kept covariate correlated with X_j, so the
    DGP must keep the observed covariates independent for rho_j to read X_j alone. Outcome Y is
    never read. Returns a per-covariate frame."""
    rows = []
    for j in x_cols:
        levels = sorted(df[j].unique())
        tr = df.loc[df[t_col] == 1, j]
        ct = df.loc[df[t_col] == 0, j]
        tr_cnt = np.array([(tr == lv).sum() for lv in levels], dtype=float)
        ct_cnt = np.array([(ct == lv).sum() for lv in levels], dtype=float)
        p1 = tr_cnt / tr_cnt.sum()                       # P(X_j | T=1)
        p0 = ct_cnt / ct_cnt.sum()                       # P(X_j | T=0)
        # rho_j is an average-divergence object (symmetric KL); gamma_j the worst-case level OR.
        ratio = p1 / np.clip(p0, 1e-12, None)
        rows.append({"covariate": j,
                     "rho_j": max(kl_discrete(p1, p0), kl_discrete(p0, p1)),
                     "gamma_j": float(np.max(np.maximum(ratio, 1.0 / ratio)))})
    return pd.DataFrame(rows)


def compute_benchmark(gen, x_cols, n_rows=5000, n_seeds=20, verbose=True):
    """Run the pooled benchmark over many seeds and aggregate. Returns (summary_df, rho_bench,
    gamma_argmax, gamma_bench); rho_bench = mean_j mean_seed rho_j averages over the observed
    covariates (the f-sensitivity budget), while gamma_bench = max_j mean_seed gamma_j keeps the
    MSM worst-case covariate and gamma_argmax names it."""
    tmp = tempfile.mkdtemp()
    per_seed = []
    for seed in range(n_seeds):
        np.random.seed(seed)
        data_obj, _ = gen.generate(n_rows, N_JOBS, os.path.join(tmp, f"bench_{seed}.csv"))
        d = benchmark_per_covariate(data_obj.data, x_cols)
        d["seed"] = seed
        per_seed.append(d)
    allres = pd.concat(per_seed, ignore_index=True)

    summary = (allres.groupby("covariate")
               .agg(rho_mean=("rho_j", "mean"), rho_std=("rho_j", "std"),
                    gamma_mean=("gamma_j", "mean"), gamma_std=("gamma_j", "std"))
               .reindex(x_cols))
    summary["role"] = [ROLE.get(c, "") for c in summary.index]

    rho_bench = float(summary["rho_mean"].mean())            # mean_j: the average observed covariate
    gamma_argmax = summary["gamma_mean"].idxmax()
    gamma_bench = float(summary.loc[gamma_argmax, "gamma_mean"])   # max_j: the worst-case covariate

    # Seed-to-seed spread of the benchmark, for the paper's error bars. The point estimates above
    # aggregate per-covariate seed-means (mean_j / max_j); the spread re-applies the SAME
    # aggregation within each seed (rho: mean_j rho_j(seed); gamma: max_j gamma_j(seed)) and takes
    # the std over seeds. (rho_bench equals the mean of its per-seed values; gamma_bench, a max of
    # seed-means, is a slightly more stable summary than the per-seed max used only for the spread.)
    per_seed_bench = allres.groupby("seed").agg(rho=("rho_j", "mean"), gamma=("gamma_j", "max"))
    summary.attrs["rho_bench_std"] = float(per_seed_bench["rho"].std(ddof=0))
    summary.attrs["gamma_bench_std"] = float(per_seed_bench["gamma"].std(ddof=0))

    if verbose:
        print(f"\nStep 1 - benchmark rho per dropped covariate "
              f"({n_seeds} seeds, N={n_rows}, DIM={len(x_cols)}):")
        print(summary.round(4).to_string())
        print(f"\n  rho_bench   = mean_j rho_j = {rho_bench:.4f}  (averaged over {len(x_cols)} covariates)")
        print(f"  gamma_bench = max_j gamma_j = {gamma_bench:.3f}  (worst covariate: {gamma_argmax})")
    return summary, rho_bench, gamma_argmax, gamma_bench


# =========================================================================== #
# Empirical OR(x,U) curves (Figure-1 on simulated data, the paper's fig1-curves).
# A binary covariate gives only a 2-point OR, so the dropped confounder is MULTI-VALUED (a
# K-level grid over u in [0,1]). We shape the per-level propensity e(u); the realized
# OR(u) = odds(T|marginal)/odds(T|u) is then measured from the data.
# =========================================================================== #
EMP_K = 81        # confounder levels (fine grid so narrow dips are resolved)
EMP_N = 80000     # samples per seed (large so the measured OR line is smooth)


def _gauss_np(u, c, w):
    return np.exp(-((u - c) / w) ** 2)


def _confounder_propensities():
    """Per-level treatment propensity e(u) for each confounder, shaped to give:
    matched_gamma -> spike (one narrow dip -> tall narrow OR, small rho) vs broad
                     (wide dip -> same peak OR, large rho);
    matched_rho   -> no-spike (moderate wide dip) vs tail-spike (same body + a deep narrow
                     tail dip -> OR explodes there, average ~unchanged)."""
    u = np.linspace(0.0, 1.0, EMP_K)
    spike = 0.5 - 0.235 * _gauss_np(u, 0.08, 0.032)
    broad = 0.5 - 0.285 * _gauss_np(u, 0.50, 0.20)
    no_spike = 0.5 - 0.20 * _gauss_np(u, 0.45, 0.22)
    tail_spike = no_spike - 0.40 * _gauss_np(u, 0.0625, 0.016)
    clip = lambda e: np.clip(e, 0.03, 0.97)
    return u, {
        "matched_gamma": [("dotted (spike, rare)", clip(spike)),
                          ("dashed (broad, common)", clip(broad))],
        "matched_rho": [("dashed (no-spike, common)", clip(no_spike)),
                        ("dotted (tail-spike, rare)", clip(tail_spike))],
    }


def _make_confounder_generator(propensity):
    """Single multi-valued confounder Z ~ Uniform{0..K-1}; T ~ Bernoulli(propensity[Z]);
    trivial U and Y. Z is the dropped covariate we treat as the confounder U."""
    K = len(propensity)
    sizes = {"U": 1, "X": 1, "T": 1, "Y": 1}
    generators = {
        "U": lambda noise: [0],
        "X": lambda u, noise: [int(np.random.randint(K))],
        "T": lambda u, x, noise: [1 if np.random.rand() < propensity[int(x[0])] else 0],
        "Y": lambda u, x, t, noise: [t[0]],
    }
    noise = {c: (lambda: 0) for c in ("U", "X", "T", "Y")}
    return Generator(generators=generators, noise_generators=noise, sizes=sizes)


def _empirical_or(df, K, min_count=20):
    """Measure OR(u) = odds(T=1|marginal)/odds(T=1|Z=u) per level, plus Gamma=max OR (MSM
    worst-case). rho is intentionally NOT computed here (the faithful benchmark rho is the
    symmetric KL from benchmark_per_covariate); this OR-curve view only reports Gamma."""
    p1 = float((df["T"] == 1).mean())
    odds_marg = p1 / (1 - p1)
    OR = np.full(K, np.nan)
    for z in range(K):
        sub = df[df["X0"] == z]
        if len(sub) < min_count:
            continue
        e = float(np.clip((sub["T"] == 1).mean(), 1e-3, 1 - 1e-3))
        OR[z] = odds_marg / (e / (1 - e))
    v = ~np.isnan(OR)
    gamma = float(np.nanmax(OR[v]))
    return OR, gamma


def run_empirical_curves(out_png=None, n=EMP_N, n_seeds=5, verbose=True):
    """Generate data from the multi-valued versions of both experiments over n_seeds and
    plot the MEASURED OR(u) lines with +/-1 std error bands; Gamma is reported as mean +/-
    std across seeds. Writes ib_figure1_empirical_curves.png."""
    out_png = out_png or os.path.join(os.path.dirname(__file__), "ib_figure1_empirical_curves.png")
    u, specs = _confounder_propensities()
    tmp = tempfile.mkdtemp()
    styles = {"dotted": dict(ls=":", color="crimson", lw=1.3, marker="o", ms=2.3),
              "dashed": dict(ls="--", color="steelblue", lw=1.3, marker="o", ms=2.3)}
    band_color = {"dotted": "crimson", "dashed": "steelblue"}
    fig, axes = plt.subplots(2, 1, figsize=(COL_W, 4.6))
    titles = {"matched_gamma": r"(a) similar $\Gamma$, different $\rho$",
              "matched_rho": r"(b) similar $\rho$, $\Gamma$ explodes"}
    for ax, name in zip(axes, ("matched_gamma", "matched_rho")):
        if verbose:
            print(f"\n{name} (empirical, N={n}, K={EMP_K}, {n_seeds} seeds):")
        for label, prop in specs[name]:
            ors, gammas = [], []
            for seed in range(n_seeds):
                np.random.seed(seed)
                gen = _make_confounder_generator(prop)
                data_obj, _ = gen.generate(n, N_JOBS,
                                           os.path.join(tmp, f"emp_{name}_{label[:3]}_{seed}.csv"))
                OR, gamma = _empirical_or(data_obj.data, len(prop))
                ors.append(OR); gammas.append(gamma)
            with warnings.catch_warnings():   # empty slice at sparse levels -> NaN (plotted as a gap)
                warnings.simplefilter("ignore", RuntimeWarning)
                OR_mean = np.nanmean(ors, axis=0)
                OR_std = np.nanstd(ors, axis=0)
            g_mean, g_std = float(np.mean(gammas)), float(np.std(gammas))
            key = label.split()[0]
            ax.plot(u, OR_mean, label=fr"{label}:  $\Gamma$={g_mean:.2f}$\pm${g_std:.2f}",
                    **styles[key])
            ax.fill_between(u, OR_mean - OR_std, OR_mean + OR_std,
                            color=band_color[key], alpha=0.18, linewidth=0)
            if verbose:
                print(f"  {label:28s} Gamma={g_mean:.3f} +/- {g_std:.3f}")
        ax.axhline(1.0, color="gray", lw=0.8, alpha=0.6)
        ax.set_ylabel(r"measured OR$(x,u)$")
        ax.set_title(titles[name])
        ax.legend(loc="upper right")
        ax.set_ylim(bottom=0)
    axes[-1].set_xlabel(r"$u$  (dropped confounder level)")
    plt.tight_layout()
    fig.savefig(out_png)
    print(f"\nwrote {out_png}")
    return fig


# =========================================================================== #
# Quadrant data generators (the breakdown/conclusion runner lives in quadrants_multiseed.py,
# which imports make_quadrant_generator, compute_benchmark and QUADRANT_SPECS from here).
# =========================================================================== #
# Each scenario sets the SHAPE of all three observed covariates via per-covariate (p_x, beta_t)
# lists, because rho_bench = mean_j rho_j now averages over covariates while gamma_bench =
# max_j gamma_j keeps the worst one. The two budgets therefore read the covariate SET
# differently, which is the lever for disagreement:
#   - a single RARE-STRONG covariate among weak ones lifts max_j gamma (large gamma_bench) while
#     mean_j rho stays small (rho-robust, Gamma-NOT);
#   - a UNIFORMLY common-moderate set lifts mean_j rho (large rho_bench) while no single gamma is
#     large (Gamma-robust, rho-NOT);
#   - all-weak vs all-strong covariates, with weak/strong U, give both-robust / both-not.
# Covariates are drawn independently, so the pooled per-covariate rho_j reads each one alone.
QUADRANT_TBASE = -0.6
QUADRANT_SPECS = {
    "both_robust":          dict(p_x=(0.45, 0.45, 0.45), beta_t=(0.40, 0.40, 0.40),
                                 u_prob=0.35, t_u=-0.35, y_effect=0.5,
                                 target="both robust (all-weak covariates)"),
    "rho_robust_gamma_not": dict(p_x=(0.16, 0.45, 0.45), beta_t=(1.75, 0.40, 0.40),
                                 u_prob=0.40, t_u=-0.70, y_effect=1.2,
                                 target="rho robust, Gamma NOT (one rare-strong covariate, rest weak)"),
    "gamma_robust_rho_not": dict(p_x=(0.50, 0.50, 0.50), beta_t=(1.10, 1.10, 1.10),
                                 u_prob=0.40, t_u=-0.70, y_effect=3.5,
                                 target="Gamma robust, rho NOT (all common-moderate covariates)"),
    "both_not_robust":      dict(p_x=(0.50, 0.50, 0.50), beta_t=(1.65, 1.65, 1.65),
                                 u_prob=0.45, t_u=-1.15, y_effect=4.5,
                                 target="both NOT robust (all-strong covariates)"),
}


def make_quadrant_generator(spec):
    """3 binary covariates whose per-covariate (p_x, beta_t) shapes are set by the spec, so the
    averaged f-sensitivity budget rho_bench = mean_j rho_j and the worst-case MSM budget
    gamma_bench = max_j gamma_j can disagree (see the scenario table above); hidden U is set by
    (u_prob, t_u, y_effect)."""
    return make_generator_ib(p_x=spec["p_x"], beta_t=spec["beta_t"], u_prob=spec["u_prob"],
                             t_base=QUADRANT_TBASE, t_u=spec["t_u"], y_effect=spec["y_effect"],
                             true_ate=1.0)


if __name__ == "__main__":
    run_empirical_curves()

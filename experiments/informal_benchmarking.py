import os
import sys
import tempfile
import warnings

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import numpy as np
import pandas as pd
import matplotlib
if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6,
    "savefig.dpi": 600, "savefig.bbox": "tight", "figure.constrained_layout.use": False,
})
COL_W = 3.35

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_generation.Generator import Generator

N_JOBS = 1

ROLE = {"X0": "strong", "X1": "high", "X2": "moderate", "X3": "mild", "X4": "weak"}


def jsonable(a):
    return [None if (v is None or np.isnan(v)) else float(v) for v in np.asarray(a, float)]


def measure_or(df, col, K):
    p1 = float((df["T"] == 1).mean())
    odds_marg = p1 / (1.0 - p1)
    OR = np.full(K, np.nan)
    for z in range(K):
        sub = df[df[col] == z]
        if len(sub) < 20:
            continue
        e = float(np.clip((sub["T"] == 1).mean(), 1e-3, 1 - 1e-3))
        OR[z] = odds_marg / (e / (1.0 - e))
    return OR


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


def f_kl(t, eps=1e-12):
    t = np.clip(np.asarray(t, dtype=float), eps, None)
    return t * np.log(t)


def benchmark_per_covariate(df, x_cols, t_col="T", eps=1e-3):
    rows = []
    e_marg = float((df[t_col] == 1).mean())
    odds_marg = e_marg / (1.0 - e_marg)
    for j in x_cols:
        levels = sorted(df[j].unique())
        tr = df.loc[df[t_col] == 1, j]
        ct = df.loc[df[t_col] == 0, j]
        tr_cnt = np.array([(tr == lv).sum() for lv in levels], dtype=float)
        ct_cnt = np.array([(ct == lv).sum() for lv in levels], dtype=float)
        p1 = tr_cnt / tr_cnt.sum()
        p0 = ct_cnt / ct_cnt.sum()
        e_v = np.clip(tr_cnt / np.clip(tr_cnt + ct_cnt, 1.0, None), eps, 1.0 - eps)
        OR = odds_marg / (e_v / (1.0 - e_v))
        rho_j = max(float(np.sum(p1 * f_kl(OR))),
                    float(np.sum(p0 * f_kl(1.0 / OR))))
        rows.append({"covariate": j,
                     "rho_j": rho_j,
                     "gamma_j": float(np.max(np.maximum(OR, 1.0 / OR)))})
    return pd.DataFrame(rows)


def compute_benchmark(gen, x_cols, n_rows=5000, n_seeds=20, verbose=True):
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

    rho_argmax = summary["rho_mean"].idxmax()
    rho_bench = float(summary.loc[rho_argmax, "rho_mean"])
    gamma_argmax = summary["gamma_mean"].idxmax()
    gamma_bench = float(summary.loc[gamma_argmax, "gamma_mean"])

    per_seed_bench = allres.groupby("seed").agg(rho=("rho_j", "max"), gamma=("gamma_j", "max"))
    summary.attrs["rho_bench_std"] = float(per_seed_bench["rho"].std(ddof=0))
    summary.attrs["gamma_bench_std"] = float(per_seed_bench["gamma"].std(ddof=0))

    if verbose:
        print(f"\nStep 1 - benchmark rho per dropped covariate "
              f"({n_seeds} seeds, N={n_rows}, DIM={len(x_cols)}):")
        print(summary.round(4).to_string())
        print(f"\n  rho_bench   = max_j rho_j = {rho_bench:.4f}  (strongest of {len(x_cols)} covariates)")
        print(f"  gamma_bench = max_j gamma_j = {gamma_bench:.3f}  (worst covariate: {gamma_argmax})")
    return summary, rho_bench, gamma_argmax, gamma_bench


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def true_rho_from_U(p_x=(0.50, 0.45, 0.40, 0.55, 0.50), beta_t=(0.5, 0.5, 0.5, 0.5, 0.5),
                    u_prob=0.40, t_base=-0.85, t_u=-0.8, eps=1e-3):
    p_x, beta_t = np.asarray(p_x, float), np.asarray(beta_t, float)
    dim = len(p_x)
    pu = np.array([1.0 - u_prob, u_prob])
    u_lvl = np.array([0.0, 1.0])
    odds = lambda e: np.clip(e, eps, 1 - eps) / (1.0 - np.clip(e, eps, 1 - eps))
    rho_perx = 0.0
    pT1_givenU = np.zeros(2)
    for mask in range(2 ** dim):
        x = np.array([(mask >> j) & 1 for j in range(dim)], float)
        px = float(np.prod(np.where(x == 1, p_x, 1.0 - p_x)))
        e_star = _sigmoid(t_base + float(beta_t @ x) + t_u * u_lvl)
        e_x = float(pu @ e_star)
        OR = np.clip(odds(e_x) / odds(e_star), eps, None)
        wT1 = pu * e_star / e_x
        wT0 = pu * (1 - e_star) / (1 - e_x)
        rho_perx = max(rho_perx,
                       max(float(wT1 @ f_kl(OR)), float(wT0 @ f_kl(1.0 / OR))))
        pT1_givenU += px * e_star
    e_marg = float(pu @ pT1_givenU)
    OR_p = np.clip(odds(e_marg) / odds(pT1_givenU), eps, None)
    wT1_p = pu * pT1_givenU / e_marg
    wT0_p = pu * (1 - pT1_givenU) / (1 - e_marg)
    rho_pooled = max(float(wT1_p @ f_kl(OR_p)), float(wT0_p @ f_kl(1.0 / OR_p)))
    return rho_perx, rho_pooled


def true_gamma_from_U(p_x=(0.50, 0.45, 0.40, 0.55, 0.50), beta_t=(0.5, 0.5, 0.5, 0.5, 0.5),
                      u_prob=0.40, t_base=-0.85, t_u=-0.8, eps=1e-3):
    p_x, beta_t = np.asarray(p_x, float), np.asarray(beta_t, float)
    dim = len(p_x)
    pu = np.array([1.0 - u_prob, u_prob])
    u_lvl = np.array([0.0, 1.0])
    odds = lambda e: np.clip(e, eps, 1 - eps) / (1.0 - np.clip(e, eps, 1 - eps))
    gamma_perx = 1.0
    pT1_givenU = np.zeros(2)
    for mask in range(2 ** dim):
        x = np.array([(mask >> j) & 1 for j in range(dim)], float)
        px = float(np.prod(np.where(x == 1, p_x, 1.0 - p_x)))
        e_star = _sigmoid(t_base + float(beta_t @ x) + t_u * u_lvl)
        e_x = float(pu @ e_star)
        OR = np.clip(odds(e_x) / odds(e_star), eps, None)
        gamma_perx = max(gamma_perx, float(np.max(np.maximum(OR, 1.0 / OR))))
        pT1_givenU += px * e_star
    e_marg = float(pu @ pT1_givenU)
    OR_p = np.clip(odds(e_marg) / odds(pT1_givenU), eps, None)
    gamma_pooled = float(np.max(np.maximum(OR_p, 1.0 / OR_p)))
    return gamma_perx, gamma_pooled


EMP_K = 81
EMP_N = 80000


def _gauss_np(u, c, w):
    return np.exp(-((u - c) / w) ** 2)


def _confounder_propensities():
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
    gamma = float(np.nanmax(np.maximum(OR[v], 1.0 / OR[v])))
    return OR, gamma


def run_empirical_curves(out_png=None, n=EMP_N, n_seeds=5, verbose=True):
    out_png = out_png or os.path.join(os.path.dirname(__file__), "ib_figure1_empirical_curves.png")
    u, specs = _confounder_propensities()
    tmp = tempfile.mkdtemp()
    styles = {"dotted": dict(ls=":", color="crimson", lw=1.3, marker="o", ms=2.3),
              "dashed": dict(ls="--", color="steelblue", lw=1.3, marker="o", ms=2.3)}
    band_color = {"dotted": "crimson", "dashed": "steelblue"}
    fig, axes = plt.subplots(2, 1, figsize=(COL_W, 4.6))
    titles = {"matched_gamma": r"(a) similar $\Gamma$, different $\rho$",
              "matched_rho": r"(b) similar $\rho$, different $\Gamma$"}
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
            with warnings.catch_warnings():
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
    axes[-1].set_xlabel(r"$u$  (dropped confounder value)")
    plt.tight_layout()
    fig.savefig(out_png)
    print(f"\nwrote {out_png}")
    return fig


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
    return make_generator_ib(p_x=spec["p_x"], beta_t=spec["beta_t"], u_prob=spec["u_prob"],
                             t_base=QUADRANT_TBASE, t_u=spec["t_u"], y_effect=spec["y_effect"],
                             true_ate=1.0)


if __name__ == "__main__":
    run_empirical_curves()

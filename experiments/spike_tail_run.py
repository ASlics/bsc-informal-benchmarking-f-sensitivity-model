"""
Figure-1 RIGHT-panel SPIKE experiment -- DGP / computation step.

Runs the data-generating process and benchmark sweep, then writes the results to
experiments/data/spike_tail.json. It produces NO figure; run spike_tail_plot.py afterwards to
render the plot from the saved data. Splitting compute from plotting lets the figure be
re-styled without re-running the (~minutes) simulation.

SETUP. A SINGLE multi-valued covariate X0 is drawn from a peaked level distribution g(u). X0
carries a moderate "body" treatment-propensity bump in the bulk of g, plus a tight spike in the
sparse left tail of g whose height we grow. Because the spike sits where g has little mass,
growing it changes X0's average divergence rho_j little but raises its worst-case ratio
Gamma_j. We benchmark X0 directly; the no-spike case (spike height 0) is the within-covariate
baseline.

Run:  python experiments/spike_tail_run.py
Writes experiments/data/spike_tail.json.
"""
import os
import sys
import json
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_generation.Generator import Generator
import informal_benchmarking as ib   # benchmark_per_covariate, N_JOBS, COL_W

# --------------------------- design knobs ---------------------------------- #
K = 25                       # confounder levels on [0,1]
N = 50000                    # samples per seed
N_SEEDS = 6
FLOOR = 0.08                 # uniform floor on g(u): every level stays two-armed (positivity)
G_CENTER, G_WIDTH = 0.60, 0.22   # g concentrates mass in the bulk; its LEFT tail is sparse

# the "body" bump -- a moderate confounder sitting in the bulk of g
BODY_CENTER, BODY_WIDTH, BODY_AMP = 0.55, 0.20, 0.95
# X0's extra feature: a tight spike in the SPARSE LEFT TAIL of g (little probability mass)
SPIKE_CENTER, SPIKE_WIDTH = 0.22, 0.030
# logit height of the spike, swept from none -> very tall
SPIKE_AMPS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]

U = np.linspace(0.0, 1.0, K)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DATA_PATH = os.path.join(DATA_DIR, "spike_tail.json")


def _gauss(c, w):
    return np.exp(-((U - c) / w) ** 2)


_graw = _gauss(G_CENTER, G_WIDTH)
G = (1.0 - FLOOR) * _graw / _graw.sum() + FLOOR * np.ones(K) / K


def _dip(center, width, amp):
    """A mean-zero (over g) logit dip -- a bump in the odds ratio, baseline propensity ~0.5."""
    raw = _gauss(center, width)
    return -amp * (raw - float((G * raw).sum()))


def spike_mass(width=SPIKE_WIDTH):
    """Probability mass under the spike = E_g[spike shape]; small because the spike sits in the
    sparse left tail of g, which is why it barely moves the average divergence rho."""
    return float((G * _gauss(SPIKE_CENTER, width)).sum())


S_BODY = _dip(BODY_CENTER, BODY_WIDTH, BODY_AMP)


def shape_x0(spike_amp):
    """X0 = the body bump + a tight tail spike of the given logit height."""
    return S_BODY + _dip(SPIKE_CENTER, SPIKE_WIDTH, spike_amp)


def make_gen(s_x0):
    """A single covariate X0 ~ g(u) carrying shape s_x0; treatment is logistic in X0 alone
    (columns X0, T)."""
    cum = np.cumsum(G.astype(float))
    cum /= cum[-1]
    sizes = {"U": 1, "X": 1, "T": 1, "Y": 1}

    def _x(u, noise):
        return [min(int(np.searchsorted(cum, np.random.rand(), side="right")), K - 1)]

    def _t(u, x, noise):
        logit = s_x0[min(int(x[0]), K - 1)]
        p = 1.0 / (1.0 + np.exp(-logit))
        return [1 if np.random.rand() < p else 0]

    generators = {"U": lambda noise: [0], "X": _x, "T": _t,
                  "Y": lambda u, x, t, noise: [t[0]]}
    noise = {c: (lambda: 0) for c in ("U", "X", "T", "Y")}
    return Generator(generators=generators, noise_generators=noise, sizes=sizes)


def measure_or(df, col):
    """Measured OR(u) = odds(T|marg)/odds(T|col=u) per level -- the distribution-panel object."""
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


def empirical(s_x0, label):
    """Generate the single-covariate data over N_SEEDS seeds and benchmark X0. Returns
    seed-means of rho_j, Gamma_j and the OR(u) curve."""
    tmp = tempfile.mkdtemp()
    gen = make_gen(s_x0)
    rx, gx, orx = [], [], []
    for s in range(N_SEEDS):
        np.random.seed(s)
        data_obj, _ = gen.generate(N, ib.N_JOBS, os.path.join(tmp, f"{label}_{s}.csv"))
        df = data_obj.data
        dx = ib.benchmark_per_covariate(df, ["X0"])   # pooled (single covariate: nothing to condition on)
        rx.append(float(dx["rho_j"].iloc[0]))
        gx.append(float(dx["gamma_j"].iloc[0]))
        orx.append(measure_or(df, "X0"))
    return dict(rho_x0=float(np.nanmean(rx)), rho_x0_sd=float(np.nanstd(rx)),
                gam_x0=float(np.nanmean(gx)), gam_x0_sd=float(np.nanstd(gx)),
                or_x0=np.nanmean(np.array(orx), axis=0))


def _jsonable(a):
    """Convert a 1-D float array to a JSON-safe list, mapping NaN -> null."""
    return [None if (v is None or np.isnan(v)) else float(v) for v in np.asarray(a, float)]


def main():
    print(f"Single-covariate Figure-1 tail-spike experiment  (K={K}, N={N}, seeds={N_SEEDS}, "
          f"spike centre={SPIKE_CENTER}, spike mass={spike_mass():.4f})\n")
    rows, or_x0 = [], []
    print(f"{'spike amp':>9}  {'rho_x0':>16}  {'Gamma_x0':>14}")
    for amp in SPIKE_AMPS:
        r = empirical(shape_x0(amp), f"tspk{amp:g}")
        rows.append(dict(amp=float(amp), **{k: float(v) for k, v in r.items()
                                            if k != "or_x0"}))
        or_x0.append(r["or_x0"])
        print(f"{amp:9.2f}  {r['rho_x0']:7.4f} +/- {r['rho_x0_sd']:6.4f}  "
              f"{r['gam_x0']:6.2f} +/- {r['gam_x0_sd']:5.2f}")

    payload = {
        "meta": {
            "experiment": "spike_tail",
            "K": K, "N": N, "N_SEEDS": N_SEEDS,
            "spike_center": SPIKE_CENTER, "spike_mass": spike_mass(),
            "col_w": float(ib.COL_W),
            "U": _jsonable(U), "G": _jsonable(G),
        },
        "rows": rows,
        "or_x0": [_jsonable(c) for c in or_x0],
    }
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(DATA_PATH, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {DATA_PATH}")


if __name__ == "__main__":
    main()

"""
Heterogeneous-confounding failure on a SINGLE-COVARIATE setup -- DGP / computation step
(the mirror of spike_tail_run.py).

Runs the data-generating process and benchmark sweep, then writes the results to
experiments/data/heterogeneous.json. It produces NO figure; run heterogeneous_plot.py afterwards
to render the plot from the saved data. Splitting compute from plotting lets the figure be
re-styled without re-running the (~minutes) simulation.

SETUP. A single multi-valued covariate X0 is drawn from a peaked level distribution g(u). X0
carries a treatment-propensity bump of fixed shape whose peak logit depth is held constant -- so
its worst-case ratio Gamma is matched across the sweep -- but the bump's centre MARCHES LEFT, out
of the bulk of g and into its sparse tail, so the treated/control overlap shrinks. Treatment is logistic in X0
alone. We benchmark X0 directly; the in-bulk centre (0.60) is the within-covariate baseline.

THE FAILURE. Gamma_j stays matched to the baseline across the march (same amplitude), while the
f-sensitivity average rho_j COLLAPSES as the bump leaves the data mass. This is the converse of
the spike-magnitude sweep (where Gamma explodes but rho barely moves).

Run:  python experiments/heterogeneous_run.py
Writes experiments/data/heterogeneous.json.
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
G_CENTER, G_WIDTH = 0.60, 0.22

BUMP_WIDTH = 0.06            # fixed narrow bump width (the 'same shape')
BUMP_AMP = 1.30             # sets the in-bulk baseline peak logit depth (held constant per centre)
CENTERS = [0.60, 0.50, 0.40, 0.30, 0.20, 0.12]   # X0's centre marches LEFT into the sparse tail

U = np.linspace(0.0, 1.0, K)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DATA_PATH = os.path.join(DATA_DIR, "heterogeneous.json")


def _gauss(c, w):
    return np.exp(-((U - c) / w) ** 2)


_graw = _gauss(G_CENTER, G_WIDTH)
G = (1.0 - FLOOR) * _graw / _graw.sum() + FLOOR * np.ones(K) / K


def _peak_depth(center, amp):
    """Magnitude of the bump's deepest logit dip (at its centre) for a given pre-mean-zero amp.
    The mean-zeroing subtracts E_g[raw], which SHRINKS as the bump leaves g's mass, so a fixed
    `amp` would make the peak dip -- and hence the worst-case ratio Gamma -- DRIFT upward as the
    centre marches into the tail. We hold this peak dip constant instead (see shape_x0)."""
    raw = _gauss(center, BUMP_WIDTH)
    return amp * (float(raw.max()) - float((G * raw).sum()))


def overlap(center):
    """Data mass under the bump = E_g[bump]; shrinks as the centre marches into g's tail."""
    return float((G * _gauss(center, BUMP_WIDTH)).sum())


# Anchor every centre to the in-bulk baseline's peak dip so Gamma is genuinely MATCHED across the
# march; the upward drift under a fixed pre-mean-zero amplitude was an artifact of mean-zeroing.
TARGET_DEPTH = _peak_depth(CENTERS[0], BUMP_AMP)


def shape_x0(center):
    """X0 = a bump relocated to `center`, rescaled so its peak logit dip matches the in-bulk
    baseline -> the worst-case ratio Gamma stays matched while only the overlap (and the
    mass-weighted rho) changes."""
    raw = _gauss(center, BUMP_WIDTH)
    mean = float((G * raw).sum())
    amp = TARGET_DEPTH / (float(raw.max()) - mean)
    return -amp * (raw - mean)


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
        dx = ib.benchmark_per_covariate(df, ["X0"], min_arm_count=40, min_cell_count=10)
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
    print(f"Single-covariate heterogeneous-shift experiment  (K={K}, N={N}, seeds={N_SEEDS}, "
          f"bump width={BUMP_WIDTH}, amp={BUMP_AMP})\n")
    rows, or_x0 = [], []
    print(f"{'centre':>7}  {'overlap':>8}  {'rho_x0':>16}  {'Gamma_x0':>14}")
    for c in CENTERS:
        r = empirical(shape_x0(c), f"het{c:g}")
        rows.append(dict(center=float(c), overlap=overlap(c),
                         **{k: float(v) for k, v in r.items() if k != "or_x0"}))
        or_x0.append(r["or_x0"])
        print(f"{c:7.2f}  {overlap(c):8.4f}  {r['rho_x0']:7.4f} +/- {r['rho_x0_sd']:6.4f}  "
              f"{r['gam_x0']:6.2f} +/- {r['gam_x0_sd']:5.2f}")

    payload = {
        "meta": {
            "experiment": "heterogeneous",
            "K": K, "N": N, "N_SEEDS": N_SEEDS,
            "bump_width": BUMP_WIDTH, "bump_amp": BUMP_AMP,
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

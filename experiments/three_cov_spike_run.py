"""Three-covariate IB demo (Gamma_bench rises, rho_bench flat) -- DGP / compute step
(the multi-covariate mirror of spike_tail_run.py).

X0, X1, X2 ~ g(u) independently. X1, X2 carry a moderate body bump; X0 carries the same body plus a
tight spike in g's sparse left tail whose height we grow. As the spike grows, Gamma_0 (and so
Gamma_bench) climbs while rho_0 barely moves and rho_bench stays flat -- the MSM benchmark reacts to
a rare confounder the f-sensitivity benchmark scores as small. Writes
experiments/data/three_cov_spike.json; render with three_cov_spike_plot.py.
"""
import os
import sys
import json
import tempfile

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_generation.Generator import Generator
import informal_benchmarking as ib

K = 25                       # covariate levels on [0,1]
N = 50000                    # samples per seed
N_SEEDS = 5
FLOOR = 0.08                 # uniform floor on g(u): every level stays two-armed (positivity)
G_CENTER, G_WIDTH = 0.60, 0.22   # g concentrates mass in the bulk; its left tail is sparse
BODY_CENTER, BODY_WIDTH, BODY_AMP = 0.55, 0.20, 0.95   # shared body bump (sets the flat rho baseline)
SPIKE_CENTER, SPIKE_WIDTH = 0.22, 0.030                # X0's tight spike in the sparse left tail
SPIKE_AMPS = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0]            # X0 spike logit height: none -> tall

U = np.linspace(0.0, 1.0, K)
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DATA_PATH = os.path.join(DATA_DIR, "three_cov_spike.json")


def _gauss(c, w):
    return np.exp(-((U - c) / w) ** 2)


_graw = _gauss(G_CENTER, G_WIDTH)
G = (1.0 - FLOOR) * _graw / _graw.sum() + FLOOR * np.ones(K) / K


def _dip(center, width, amp):
    """Mean-zero (over g) logit dip -- a bump in the odds ratio, baseline propensity ~0.5."""
    raw = _gauss(center, width)
    return -amp * (raw - float((G * raw).sum()))


def spike_mass(width=SPIKE_WIDTH):
    """Probability mass under the spike = E_g[spike shape] (small: it sits in g's sparse tail)."""
    return float((G * _gauss(SPIKE_CENTER, width)).sum())


S_BODY = _dip(BODY_CENTER, BODY_WIDTH, BODY_AMP)   # shared body bump (X1, X2, and X0's base)


def shape_x0(spike_amp):
    """X0 = the shared body bump + a tail spike of the given logit height."""
    return S_BODY + _dip(SPIKE_CENTER, SPIKE_WIDTH, spike_amp)


def make_gen(s0):
    """X0 carries s0; X1, X2 carry the plain body bump. All three ~ g(u) independently; treatment
    logistic in the sum of shapes."""
    cum = np.cumsum(G.astype(float))
    cum /= cum[-1]
    shapes = [s0, S_BODY, S_BODY]
    sizes = {"U": 1, "X": 3, "T": 1, "Y": 1}

    def _draw():
        return min(int(np.searchsorted(cum, np.random.rand(), side="right")), K - 1)

    def _x(u, noise):
        return [_draw(), _draw(), _draw()]

    def _t(u, x, noise):
        logit = sum(shapes[i][min(int(x[i]), K - 1)] for i in range(3))
        p = 1.0 / (1.0 + np.exp(-logit))
        return [1 if np.random.rand() < p else 0]

    generators = {"U": lambda noise: [0], "X": _x, "T": _t,
                  "Y": lambda u, x, t, noise: [t[0]]}
    noise = {c: (lambda: 0) for c in ("U", "X", "T", "Y")}
    return Generator(generators=generators, noise_generators=noise, sizes=sizes)


def run_step(spike_amp):
    """Grow X0's spike to `spike_amp` (X1, X2 stay plain). Generate over N_SEEDS seeds, benchmark
    all three, return seed-means of per-covariate rho_j / Gamma_j, the bench maxes, Gamma argmax, OR(u)."""
    gen = make_gen(shape_x0(spike_amp))
    tmp = tempfile.mkdtemp()
    rho = {j: [] for j in ("X0", "X1", "X2")}
    gam = {j: [] for j in ("X0", "X1", "X2")}
    rho_bench, gam_bench, or_x0 = [], [], []
    for s in range(N_SEEDS):
        np.random.seed(s)
        data_obj, _ = gen.generate(N, ib.N_JOBS, os.path.join(tmp, f"tcs{spike_amp:g}_{s}.csv"))
        df = data_obj.data
        d = ib.benchmark_per_covariate(df, ["X0", "X1", "X2"]).set_index("covariate")
        for j in ("X0", "X1", "X2"):
            rho[j].append(float(d.loc[j, "rho_j"]))
            gam[j].append(float(d.loc[j, "gamma_j"]))
        rho_bench.append(float(d["rho_j"].max()))
        gam_bench.append(float(d["gamma_j"].max()))
        or_x0.append(ib.measure_or(df, "X0", K))
    gam_argmax = max(("X0", "X1", "X2"), key=lambda j: np.mean(gam[j]))
    return dict(
        amp=float(spike_amp),
        rho_x0=float(np.mean(rho["X0"])), rho_x1=float(np.mean(rho["X1"])),
        rho_x2=float(np.mean(rho["X2"])),
        gam_x0=float(np.mean(gam["X0"])), gam_x1=float(np.mean(gam["X1"])),
        gam_x2=float(np.mean(gam["X2"])),
        rho_bench=float(np.mean(rho_bench)), rho_bench_sd=float(np.std(rho_bench)),
        gam_bench=float(np.mean(gam_bench)), gam_bench_sd=float(np.std(gam_bench)),
        gam_argmax=gam_argmax, _or_x0=np.nanmean(np.array(or_x0), axis=0),
    )


def main():
    print(f"Three-covariate spike IB demo  (K={K}, N={N}, seeds={N_SEEDS}, "
          f"spike centre={SPIKE_CENTER}, spike mass={spike_mass():.4f})\n")
    rows, or_x0 = [], []
    hdr = (f"{'spike':>6} | {'rho X0':>8} {'rho X1':>8} {'rho X2':>8} {'rho_bench':>10} | "
           f"{'gam X0':>8} {'gam_bench':>10} {'argmax':>7}")
    print(hdr); print("-" * len(hdr))
    for amp in SPIKE_AMPS:
        r = run_step(amp)
        or_x0.append(r.pop("_or_x0"))
        rows.append(r)
        print(f"{amp:6.2f} | {r['rho_x0']:8.4f} {r['rho_x1']:8.4f} {r['rho_x2']:8.4f} "
              f"{r['rho_bench']:10.4f} | {r['gam_x0']:8.2f} {r['gam_bench']:10.3f} "
              f"{r['gam_argmax']:>7}")

    payload = {
        "meta": {"experiment": "three_cov_spike", "K": K, "N": N, "N_SEEDS": N_SEEDS,
                 "spike_center": SPIKE_CENTER, "spike_mass": spike_mass(),
                 "body_center": BODY_CENTER, "col_w": float(ib.COL_W),
                 "U": ib.jsonable(U), "G": ib.jsonable(G)},
        "rows": rows,
        "or_x0": [ib.jsonable(c) for c in or_x0],
    }
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(DATA_PATH, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {DATA_PATH}")


if __name__ == "__main__":
    main()

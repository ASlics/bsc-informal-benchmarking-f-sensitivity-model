import os
import sys
import json
import tempfile

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_generation.Generator import Generator
import informal_benchmarking as ib

K = 25
N = 50000
N_SEEDS = 5
FLOOR = 0.06
G_SLOPE = 3.0
BUMP_WIDTH = 0.055
BUMP_AMP = 1.8
ANCHOR = 0.80
X0_CENTERS = [0.80, 0.60, 0.40, 0.20]

U = np.linspace(0.0, 1.0, K)
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DATA_PATH = os.path.join(DATA_DIR, "three_cov.json")


def _gauss(c, w):
    return np.exp(-((U - c) / w) ** 2)


_graw = np.exp(-G_SLOPE * U)
G = (1.0 - FLOOR) * _graw / _graw.sum() + FLOOR * np.ones(K) / K

_ref_raw = _gauss(ANCHOR, BUMP_WIDTH)
TARGET_DEPTH = BUMP_AMP * (float(_ref_raw.max()) - float((G * _ref_raw).sum()))


def shape(center):
    raw = _gauss(center, BUMP_WIDTH)
    mean = float((G * raw).sum())
    amp = TARGET_DEPTH / (float(raw.max()) - mean)
    return -amp * (raw - mean)


def overlap(center):
    return float((G * _gauss(center, BUMP_WIDTH)).sum())


def make_gen(s0, s1, s2):
    cum = np.cumsum(G.astype(float))
    cum /= cum[-1]
    shapes = [s0, s1, s2]
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


def run_step(c_x0):
    s0, s1, s2 = shape(c_x0), shape(ANCHOR), shape(ANCHOR)
    gen = make_gen(s0, s1, s2)
    tmp = tempfile.mkdtemp()
    rho = {j: [] for j in ("X0", "X1", "X2")}
    gam = {j: [] for j in ("X0", "X1", "X2")}
    rho_bench, gam_bench, or_x0 = [], [], []
    for s in range(N_SEEDS):
        np.random.seed(s)
        data_obj, _ = gen.generate(N, ib.N_JOBS, os.path.join(tmp, f"tc{c_x0:g}_{s}.csv"))
        df = data_obj.data
        d = ib.benchmark_per_covariate(df, ["X0", "X1", "X2"]).set_index("covariate")
        for j in ("X0", "X1", "X2"):
            rho[j].append(float(d.loc[j, "rho_j"]))
            gam[j].append(float(d.loc[j, "gamma_j"]))
        rho_bench.append(float(d["rho_j"].max()))
        gam_bench.append(float(d["gamma_j"].max()))
        or_x0.append(ib.measure_or(df, "X0", K))
    argmax = max(("X0", "X1", "X2"), key=lambda j: np.mean(rho[j]))
    return dict(
        center=float(c_x0), overlap=overlap(c_x0),
        rho_x0=float(np.mean(rho["X0"])), rho_x1=float(np.mean(rho["X1"])),
        rho_x2=float(np.mean(rho["X2"])),
        gam_x0=float(np.mean(gam["X0"])), gam_x1=float(np.mean(gam["X1"])),
        gam_x2=float(np.mean(gam["X2"])),
        rho_bench=float(np.mean(rho_bench)), rho_bench_sd=float(np.std(rho_bench)),
        gam_bench=float(np.mean(gam_bench)), gam_bench_sd=float(np.std(gam_bench)),
        rho_argmax=argmax, _or_x0=np.nanmean(np.array(or_x0), axis=0),
    )


def main():
    print(f"Three-covariate IB demo  (K={K}, N={N}, seeds={N_SEEDS}, "
          f"bump width={BUMP_WIDTH}, anchor={ANCHOR})")
    print(f"  g(u) left-heavy (slope {G_SLOPE}); overlap at 0.8={overlap(0.8):.3f}, "
          f"0.2={overlap(0.2):.3f}\n")
    rows, or_x0 = [], []
    hdr = (f"{'X0 ctr':>7} {'overlap':>8} | {'rho X0':>8} {'rho X1':>8} {'rho X2':>8} "
           f"{'rho_bench':>10} {'argmax':>7} | {'gam_bench':>10}")
    print(hdr); print("-" * len(hdr))
    for c in X0_CENTERS:
        r = run_step(c)
        or_x0.append(r.pop("_or_x0"))
        rows.append(r)
        print(f"{c:7.2f} {r['overlap']:8.4f} | {r['rho_x0']:8.4f} {r['rho_x1']:8.4f} "
              f"{r['rho_x2']:8.4f} {r['rho_bench']:10.4f} {r['rho_argmax']:>7} | "
              f"{r['gam_bench']:10.3f}")

    payload = {
        "meta": {"experiment": "three_cov", "K": K, "N": N, "N_SEEDS": N_SEEDS,
                 "bump_width": BUMP_WIDTH, "bump_amp": BUMP_AMP, "anchor": ANCHOR,
                 "g_slope": G_SLOPE, "col_w": float(ib.COL_W),
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

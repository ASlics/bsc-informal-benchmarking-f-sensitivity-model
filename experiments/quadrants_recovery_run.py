"""Quadrants recovery table: does the benchmark recover the rho (and gamma) the hidden U induces,
across the four covariate scenarios?

U is binary and the propensity is known, so the (f,rho) value and worst-case ratio U induces are
closed form (no solver). For each scenario we read, on the same data: rho_bench / gamma_bench (the
benchmarks over the observed covariates, max_j, seed-averaged) and rho_true / gamma_true (the value
U induces, per-x worst case -- the strict definition). The IB belief holds where rho_bench >=
rho_true. Writes quadrants_recovery_summary.csv.

Usage:  python experiments/quadrants_recovery_run.py [seeds_bench=15] [n_rows=4000]
"""
import os
import sys

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))                 # experiments/
from informal_benchmarking import (make_quadrant_generator, compute_benchmark,  # noqa: E402
                                    true_rho_from_U, true_gamma_from_U,
                                    QUADRANT_SPECS, QUADRANT_TBASE)

LABELS = {
    "both_robust":          "weak covariates, weak U",
    "rho_robust_gamma_not": "one rare-strong, rest weak",
    "gamma_robust_rho_not": "all common-moderate",
    "both_not_robust":      "strong covariates, strong U",
}


def main():
    seeds_bench = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    n_rows = int(sys.argv[2]) if len(sys.argv) > 2 else 4000

    rows = []
    for name, spec in QUADRANT_SPECS.items():
        print("=" * 78)
        print(f"{name}  ({LABELS[name]})  target={spec['target']}")
        print("=" * 78, flush=True)

        gen = make_quadrant_generator(spec)
        bench_summary, rho_bench, _, gamma_bench = compute_benchmark(
            gen, ["X0", "X1", "X2"], n_rows=n_rows, n_seeds=seeds_bench, verbose=False)
        rho_bench_std = bench_summary.attrs["rho_bench_std"]
        gamma_bench_std = bench_summary.attrs["gamma_bench_std"]

        rho_true, _ = true_rho_from_U(
            spec["p_x"], spec["beta_t"], spec["u_prob"], QUADRANT_TBASE, spec["t_u"])
        gamma_true, _ = true_gamma_from_U(
            spec["p_x"], spec["beta_t"], spec["u_prob"], QUADRANT_TBASE, spec["t_u"])

        recovered = rho_bench >= rho_true
        rows.append(dict(
            scenario=name, label=LABELS[name],
            rho_bench=rho_bench, rho_bench_std=rho_bench_std, rho_true=rho_true,
            gamma_bench=gamma_bench, gamma_bench_std=gamma_bench_std, gamma_true=gamma_true,
            ib_belief_holds=bool(recovered)))

        print(f"  rho   : bench={rho_bench:.4f}+-{rho_bench_std:.4f}  true(U)={rho_true:.4f}  "
              f"=> IB belief {'HOLDS' if recovered else 'FAILS (under-reports)'}")
        print(f"  gamma : bench={gamma_bench:.3f}+-{gamma_bench_std:.3f}  true(U)={gamma_true:.3f}",
              flush=True)

    out = os.path.join(os.path.dirname(__file__), "quadrants_recovery_summary.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("\n" + "=" * 78)
    print(f"wrote {out}  ({seeds_bench} benchmark seeds, n={n_rows})")


if __name__ == "__main__":
    main()

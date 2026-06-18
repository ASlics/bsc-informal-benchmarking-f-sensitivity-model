"""Multi-seed stress test of the two-framework quadrants table (tab:quadrants).

The paper's Table 1 reports the dual breakdown on a SINGLE seed (break_seeds=1) while the
benchmark uses many seeds. This script recomputes, for each of the four scenarios, the
breakdown (rho_break via f-sensitivity EVaR, gamma_break via MSM CVaR) INDEPENDENTLY on
each of N seeds, and reports per-seed values, mean +/- std, and how often each framework's
conclusion (robust <=> break > bench) matches the paper's table conclusion. The benchmark is
seed-averaged exactly as in the paper.

Usage:  python quadrants_multiseed.py [n_seeds=5] [seeds_bench=15] [n_rows=4000] [scenario_filter]
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))                 # experiments/
from informal_benchmarking import (make_quadrant_generator, compute_benchmark,  # noqa: E402
                                    QUADRANT_SPECS, N_JOBS)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_generation.DataObject import DataObject                               # noqa: E402

# Conclusions as printed in the paper's Table 1, keyed by scenario.
PAPER = {
    "both_robust":          dict(label="weak covariate, weak U",     rho=True,  gamma=True),
    "rho_robust_gamma_not": dict(label="rare strong covariate",      rho=True,  gamma=False),
    "gamma_robust_rho_not": dict(label="common moderate covariate",  rho=False, gamma=True),
    "both_not_robust":      dict(label="strong covariate, strong U", rho=False, gamma=False),
}

# Same ladders compute_breakdown_dual uses, so the ceiling semantics match the table.
RHO_LADDER = (0.05, 0.1, 0.2, 0.35, 0.6, 1.0)
GAMMA_LADDER = (1.25, 1.5, 2.0, 3.0, 5.0, 8.0)


def per_seed_dual(gen, seeds, n_rows):
    """One independent dual breakdown per seed (U hidden). Returns a per-seed DataFrame."""
    from models.FSensitivity import FSensitivity
    from models.MarginalSensitivityModel import MSM
    fs, msm = FSensitivity(), MSM()
    tmp = tempfile.mkdtemp()
    rows = []
    for seed in seeds:
        np.random.seed(seed)
        data_obj, _ = gen.generate(n_rows, N_JOBS, os.path.join(tmp, f"dual_{seed}.csv"))
        df = data_obj.data.drop(columns=[c for c in data_obj.data.columns
                                         if c.upper().startswith("U")])
        data_full = DataObject(df)
        r = fs.find_breakdown_rho(data_full, target=0.0, approach="evar",
                                  rho_ladder=RHO_LADDER, refine_steps=1, verbose=False)
        g = msm.find_breakdown_gamma(data_full, target=0.0, approach="cvar",
                                     gamma_ladder=GAMMA_LADDER, refine_steps=1, verbose=False)
        rows.append(dict(seed=seed, rho_break=r["rho"], rho_reached=bool(r["reached"]),
                         gamma_break=g["gamma"], gamma_reached=bool(g["reached"])))
        print(f"    seed {seed}: rho_break={r['rho']:.4f}{'' if r['reached'] else '+'}  "
              f"gamma_break={g['gamma']:.3f}{'' if g['reached'] else '+'}", flush=True)
    return pd.DataFrame(rows)


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    seeds_bench = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    n_rows = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
    only = sys.argv[4] if len(sys.argv) > 4 else None
    seeds = list(range(n_seeds))

    specs = {only: QUADRANT_SPECS[only]} if only else QUADRANT_SPECS
    allrows, summary = [], []
    for name, spec in specs.items():
        print("=" * 78)
        print(f"{name}  ({PAPER[name]['label']})  target={spec['target']}")
        print("=" * 78, flush=True)
        gen = make_quadrant_generator(spec)
        _, rho_bench, _, gamma_bench = compute_benchmark(
            gen, ["X0", "X1", "X2"], n_rows=n_rows, n_seeds=seeds_bench, verbose=False)
        print(f"  benchmark ({seeds_bench} seeds): rho_bench={rho_bench:.4f}  "
              f"gamma_bench={gamma_bench:.3f}", flush=True)

        df = per_seed_dual(gen, seeds, n_rows)
        df.insert(0, "scenario", name)
        df["rho_bench"], df["gamma_bench"] = rho_bench, gamma_bench
        df["rho_robust"] = df["rho_break"] > rho_bench
        df["gamma_robust"] = df["gamma_break"] > gamma_bench
        allrows.append(df)

        rfrac, gfrac = df["rho_robust"].mean(), df["gamma_robust"].mean()
        summary.append(dict(
            scenario=name, label=PAPER[name]["label"],
            rho_bench=rho_bench, gamma_bench=gamma_bench,
            rho_break_mean=df["rho_break"].mean(), rho_break_std=df["rho_break"].std(ddof=0),
            gamma_break_mean=df["gamma_break"].mean(), gamma_break_std=df["gamma_break"].std(ddof=0),
            rho_robust_frac=rfrac, gamma_robust_frac=gfrac,
            paper_rho=PAPER[name]["rho"], paper_gamma=PAPER[name]["gamma"]))
        print(f"  => rho_break   {df['rho_break'].mean():.4f}+-{df['rho_break'].std(ddof=0):.4f}  "
              f"robust on {rfrac*100:.0f}% of seeds (paper: {PAPER[name]['rho']})")
        print(f"  => gamma_break {df['gamma_break'].mean():.3f}+-{df['gamma_break'].std(ddof=0):.3f}  "
              f"robust on {gfrac*100:.0f}% of seeds (paper: {PAPER[name]['gamma']})", flush=True)

    outdir = os.path.dirname(__file__)
    sfx = f"_{only}" if only else ""
    pd.concat(allrows, ignore_index=True).to_csv(
        os.path.join(outdir, f"quadrants_multiseed_perseed{sfx}.csv"), index=False)
    pd.DataFrame(summary).to_csv(
        os.path.join(outdir, f"quadrants_multiseed_summary{sfx}.csv"), index=False)

    print("\n" + "=" * 78)
    print(f"MULTI-SEED SUMMARY  ({n_seeds} breakdown seeds, {seeds_bench} benchmark seeds, n={n_rows})")
    print("=" * 78)
    for a in summary:
        rflag = "STABLE" if a["rho_robust_frac"] in (0.0, 1.0) else "FLIPS "
        gflag = "STABLE" if a["gamma_robust_frac"] in (0.0, 1.0) else "FLIPS "
        rmatch = (a["rho_robust_frac"] == 1.0) == a["paper_rho"] and a["rho_robust_frac"] in (0.0, 1.0)
        gmatch = (a["gamma_robust_frac"] == 1.0) == a["paper_gamma"] and a["gamma_robust_frac"] in (0.0, 1.0)
        print(f"{a['label']:28s} | f-sens {a['rho_break_mean']:.3f}+-{a['rho_break_std']:.3f} "
              f"vs {a['rho_bench']:.3f} [{a['rho_robust_frac']*100:3.0f}% rob {rflag} "
              f"{'OK ' if rmatch else 'DIFF'}] | "
              f"MSM {a['gamma_break_mean']:.2f}+-{a['gamma_break_std']:.2f} "
              f"vs {a['gamma_bench']:.2f} [{a['gamma_robust_frac']*100:3.0f}% rob {gflag} "
              f"{'OK ' if gmatch else 'DIFF'}]")


if __name__ == "__main__":
    main()

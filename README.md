# Informal benchmarking for f-sensitivity

Experiments comparing the **f-sensitivity** model to the **marginal sensitivity model (MSM)**.
The MSM bounds a worst-case propensity odds ratio Γ; f-sensitivity bounds an average divergence ρ.
Informal benchmarking drops each observed covariate and reads the confounding it would have induced;
because Γ reacts to the worst level while ρ weighs the mass of the confounding, the two can rank the
same covariates differently — which is what these experiments demonstrate.

## Setup

```bash
pip install numpy pandas matplotlib tqdm
```

Tested on Python 3.14.

## Experiments

Run from the repository root. Each `*_run.py` runs the (slow) simulation and caches results to
`experiments/data/*.json`; the matching `*_plot.py` renders the figure from that cache.

| Experiment | What it shows | Run | Plot |
| --- | --- | --- | --- |
| Figure-1 curves | similar Γ / different ρ, and ρ similar / Γ explodes | `python experiments/informal_benchmarking.py` | (same command) |
| Three-covariate | Γ flat, ρ rises | `python experiments/three_cov_run.py` | `python experiments/three_cov_plot.py` |
| Three-covariate spike | Γ rises, ρ flat | `python experiments/three_cov_spike_run.py` | `python experiments/three_cov_spike_plot.py` |
| Quadrants recovery | benchmark ρ/Γ vs the value the hidden confounder induces | `python experiments/quadrants_recovery_run.py` | `python experiments/quadrants_recovery_plot.py` |

`experiments/informal_benchmarking.py` is the shared library imported by the runners.

Contact: a.slics@student.tudelft.nl

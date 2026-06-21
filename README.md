# f-sensitivity-through-evar

Informal-benchmarking experiments for the **f-sensitivity** model, comparing it to the **marginal
sensitivity model (MSM)**, developed for my TU Delft research project.

Both models bound the strength of an unobserved confounder, but on different scales. The MSM bounds
a **worst-case** propensity odds ratio Γ; f-sensitivity bounds an **average** divergence ρ
(here the symmetric KL). *Informal benchmarking* calibrates these bounds by dropping each observed
covariate and reading the confounding it would have induced. Because Γ reacts to the worst level
while ρ weighs the mass of the confounding, the two can rank the same covariates differently — which
is what these experiments demonstrate.

## Setup

```bash
pip install numpy pandas matplotlib tqdm
```

Tested on Python 3.14. No external solver is required.

## Repository layout

```
data_generation/    Generator (causal data with a hidden confounder) + thin DataObject wrapper
experiments/        the shared library, experiment runners/plotters, and their cached data
  informal_benchmarking.py   shared lib: benchmark ρ/Γ, Figure-1 curves, quadrant generators, helpers
  data/*.json                cached simulation results (git-ignored, regenerable)
```

## Experiments

Run all commands from the repository root. Each `*_run.py` script runs the (slow) simulation and
caches results to `experiments/data/*.json`; the matching `*_plot.py` renders the figure from that
cache (fast, re-styleable). Figures are written next to the scripts in `experiments/`.

| Experiment | What it shows | Run | Plot | Output |
| --- | --- | --- | --- | --- |
| Figure-1 empirical curves | similar Γ / different ρ, and ρ similar / Γ explodes | `python experiments/informal_benchmarking.py` | (same command) | `ib_figure1_empirical_curves.png` |
| Three-covariate | Γ flat, ρ rises (benchmark argmax switches) | `python experiments/three_cov_run.py` | `python experiments/three_cov_plot.py` | `three_cov_experiment.png` |
| Three-covariate spike | Γ rises, ρ flat (MSM reacts to a rare confounder) | `python experiments/three_cov_spike_run.py` | `python experiments/three_cov_spike_plot.py` | `three_cov_spike_experiment.png` |
| Quadrants recovery table | benchmark ρ/Γ vs the closed-form value the hidden U induces | `python experiments/quadrants_recovery_run.py` | `python experiments/quadrants_recovery_plot.py` | `quadrants_recovery.png` |

`experiments/informal_benchmarking.py` is the shared library (the benchmark ρ/Γ, the Figure-1
empirical-curve plot, the quadrant data generators, and small helpers) imported by the runners.

## Attribution

`data_generation/` is adapted from
[f-sensitivity-through-evar](https://github.com/MatejHav/f-sensitivity-through-evar) by Matej
Havelka, used under the MIT License (see `LICENSE`).

Thesis: pending

Contact:
a.slics@student.tudelft.nl

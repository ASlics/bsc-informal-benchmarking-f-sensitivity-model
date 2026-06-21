# f-sensitivity-through-evar

Informal-benchmarking experiments for f-sensitivity, comparing it to the marginal sensitivity
model (MSM), developed for my master thesis.

## Setup

```bash
pip install numpy pandas matplotlib tqdm
```

Tested on Python 3.14. No external solver is required.

## Experiments

Run all commands from the repository root. Each `*_run.py` script runs the (slow) simulation and
caches results to `experiments/data/*.json`; the matching `*_plot.py` renders the figure from that
cache (fast, re-styleable).

| Experiment | Run | Plot | Output |
| --- | --- | --- | --- |
| Figure-1 empirical curves | `python experiments/informal_benchmarking.py` | (same command) | `ib_figure1_empirical_curves.png` |
| Three-covariate (Γ flat, ρ rises) | `python experiments/three_cov_run.py` | `python experiments/three_cov_plot.py` | `three_cov_experiment.png` |
| Three-covariate spike (Γ rises, ρ flat) | `python experiments/three_cov_spike_run.py` | `python experiments/three_cov_spike_plot.py` | `three_cov_spike_experiment.png` |
| Quadrants recovery table | `python experiments/quadrants_recovery_run.py` | `python experiments/quadrants_recovery_plot.py` | `quadrants_recovery.png` |

`experiments/informal_benchmarking.py` is the shared library (benchmark ρ, the Figure-1
empirical-curve plot, the quadrant data generators, and small helpers) imported by the runners.
Figures are written next to the scripts in `experiments/`.

Thesis: pending

Contact:
a.slics@student.tudelft.nl

# f-sensitivity-through-evar

Implementation of f-sensitivity (via an EVaR formulation) and the informal-benchmarking
experiments comparing it to the marginal sensitivity model, developed for my master thesis.

## Setup

```bash
# light tier — benchmark + all figures (no external solver)
pip install numpy pandas matplotlib tqdm

# solver tier — partial-identification bounds (quadrants table only)
pip install scikit-learn joblib pyomo torch   # plus IPOPT: conda install -c conda-forge ipopt
```

Tested on Python 3.14.

## Experiments

Run all commands from the repository root. The `*_run.py` scripts run the (slow) simulation
and cache results to `experiments/data/*.json`; the matching `*_plot.py` scripts render the
figure from that cache (fast, re-styleable).

| Experiment | Run | Plot | Output |
| --- | --- | --- | --- |
| Figure-1 empirical curves | `python experiments/informal_benchmarking.py` | (same command) | `ib_figure1_empirical_curves.png` |
| Spike-tail | `python experiments/spike_tail_run.py` | `python experiments/spike_tail_plot.py` | `spike_tail_experiment.png` |
| Heterogeneous confounding | `python experiments/heterogeneous_run.py` | `python experiments/heterogeneous_plot.py` | `heterogeneous_experiment.png` |
| Two-framework quadrants (solver tier) | `python experiments/quadrants_multiseed.py` | — | `quadrants_multiseed_*.csv` |

`experiments/informal_benchmarking.py` is the shared library (benchmark ρ, the Figure-1
empirical-curve plot, and the quadrant data generators) imported by the runners. Figures are
written next to the scripts in `experiments/`.

Thesis: pending

Contact:
a.slics@student.tudelft.nl

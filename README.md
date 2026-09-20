# ConformalGuard

**A reproducible benchmark for evaluating conformal prediction under distribution shift.**

ConformalGuard studies how classification prediction sets behave when deployment data differs from the data used for training and conformal calibration.

The benchmark currently supports:

- IID evaluation
- Covariate shift
- Label shift
- Concept shift
- Multi-seed experiment grids
- Accuracy and conformal metrics
- IID-relative degradation analysis
- CSV and JSON reporting
- Robustness visualizations

The implementation is designed for reproducible, CPU-friendly tabular classification experiments.

---

## Research Question

> How does conformal prediction reliability change under controlled distribution shift, and how quickly do predictive performance and empirical coverage degrade as shift severity increases?

ConformalGuard focuses on empirical robustness rather than assuming conformal prediction remains calibrated under arbitrary distribution changes.

---

## Why ConformalGuard?

Conformal prediction can provide finite-sample coverage guarantees under appropriate assumptions.

Real deployment conditions may violate those assumptions through changes in feature distributions, class prevalence, or the relationship between features and labels.

ConformalGuard measures these effects using accuracy, macro F1, empirical coverage, coverage gap, prediction-set size, and empty-set rate.


---

## Benchmark Families

### IID

The IID experiment provides the reference condition used for robustness comparisons.

### Covariate Shift

Covariate shift modifies selected test features with configurable severity while preserving the original labels. This stresses changes in the input distribution.

### Label Shift

Label shift reproducibly changes the class proportions of the evaluation population while preserving feature-label pairs. This measures sensitivity to changing class prevalence.

### Concept Shift

Concept shift changes the relationship between features and labels using a controlled, feature-dependent label transformation. This stresses changes in the conditional relationship between X and Y.

---

## Experiment Pipeline

```text
Dataset
  |
  v
Train / calibration / test split
  |
  +--> Train classifier
  |
  +--> Conformal calibration
  |
  +--> IID evaluation
  |
  +--> Covariate-shift grid
  |
  +--> Label-shift grid
  |
  +--> Concept-shift grid
  |
  v
Multi-seed aggregation
  |
  v
Robustness reporting
  |
  +--> CSV / JSON
  +--> absolute-metric plots
  +--> IID-relative degradation
  +--> degradation plots
```

---

## Robustness Degradation

ConformalGuard measures every shifted condition relative to the IID baseline.

```text
delta = shifted metric - IID metric
```

For accuracy, macro F1, and coverage, a negative delta represents degradation relative to IID performance.

The degradation layer supports:

- accuracy
- macro F1
- coverage
- coverage gap
- prediction-set size
- empty-set rate

Results can be returned as a pandas DataFrame or exported to CSV and JSON.


---

## Installation

ConformalGuard requires Python 3.11 or newer.

```bash
python -m pip install -e ".[dev]"
```

---

## Run the Test Suite

```bash
python -m pytest -q
```

Latest verified development checkpoint:

```text
116 passed
```

---

## Run the Robustness Benchmark

```bash
python examples/robustness_benchmark.py
```

The example runs IID, covariate-shift, label-shift, and concept-shift experiments across multiple random seeds.

Generated outputs include:

```text
artifacts/robustness_summary.csv
artifacts/robustness_summary.json
artifacts/robustness_accuracy.png
artifacts/robustness_coverage.png
artifacts/robustness_accuracy_degradation.png
artifacts/robustness_coverage_degradation.png
```

---

## Verified Example Results

One verified run of `examples/robustness_benchmark.py` on scikit-learn's breast-cancer dataset produced:

| Condition | Mean accuracy | Mean coverage |
|---|---:|---:|
| IID | 0.980 | 0.874 |
| Covariate severity 0.5 | 0.889 | 0.792 |
| Covariate severity 1.0 | 0.716 | 0.617 |
| Covariate severity 2.0 | 0.468 | 0.404 |
| Concept severity 0.25 | 0.854 | 0.760 |
| Concept severity 0.50 | 0.743 | 0.664 |
| Concept severity 0.75 | 0.620 | 0.570 |
| Concept severity 1.00 | 0.509 | 0.468 |

These values describe one reproducible example run, not universal performance claims. Results depend on the dataset, split, classifier, random seeds, conformity score, confidence level, and shift configuration.

---

## Python API

Run the combined benchmark:

```python
from conformalguard.experiments import run_robustness_benchmark

result = run_robustness_benchmark(
    X,
    y,
    label_target_proportions=(
        {0: 0.50, 1: 0.50},
        {0: 0.70, 1: 0.30},
        {0: 0.30, 1: 0.70},
    ),
    covariate_severities=(0.0, 0.5, 1.0, 2.0),
    concept_severities=(0.0, 0.25, 0.50, 0.75, 1.0),
    seeds=(11, 42, 73),
    confidence_level=0.90,
)
```

Create a normalized summary table:

```python
from conformalguard.experiments import robustness_summary_frame

summary = robustness_summary_frame(result)
```

Compare shifted conditions against IID:

```python
from conformalguard.experiments import robustness_degradation_frame

degradation = robustness_degradation_frame(result)
```

Plot coverage degradation:

```python
from conformalguard.experiments import plot_robustness_coverage_degradation

ax = plot_robustness_coverage_degradation(result)
```

---

## Reproducibility

The experiment architecture explicitly controls or records:

- train / calibration / test splits
- random seeds
- confidence level
- conformity score
- shift severity
- shifted-feature fraction
- label target proportions
- concept-shift feature
- classification metrics
- conformal metrics

Multi-seed grids aggregate results using means and standard deviations so robustness comparisons do not rely on a single split.

---

## Design Principles

- **Reproducibility first:** experiments use explicit seeds and controlled configurations.
- **CPU-friendly research:** the benchmark is designed to run without specialized accelerator hardware.
- **Controlled distribution shift:** shift severity is an explicit experimental parameter.
- **Coverage is not enough:** prediction-set efficiency and classification quality are evaluated alongside coverage.
- **IID-relative interpretation:** shifted results can be compared directly with the reference condition.
- **Honest benchmarking:** the project reports empirical behavior under specific shift mechanisms rather than claiming arbitrary distribution-shift guarantees.

---

## Limitations

The current benchmark uses controlled synthetic shift mechanisms. These are useful for isolating failure modes but do not reproduce every real deployment environment.

The implementation is centered on tabular classification. Concept shift is implemented through a controlled feature-dependent label transformation rather than a naturally evolving production process.

The example benchmark uses a single public dataset and should not be interpreted as evidence of universal conformal robustness or failure.

Natural temporal drift, online adaptation, recalibration policies, recovery-cost experiments, and broader multi-dataset evaluation remain possible future extensions.

---

## Development Status

The experimental core is implemented and tested.

Current verified capabilities include IID evaluation, three controlled distribution-shift families, multi-seed grids, combined robustness benchmarking, normalized reporting, CSV/JSON export, IID-relative degradation analysis, and visualization.

The repository is now in final documentation and release-hardening work before the first stable release.

---

## Author

**Chinmaya Satyam**

B.Tech - Computer Science and Artificial Intelligence
Sri Venkateswara University College of Engineering

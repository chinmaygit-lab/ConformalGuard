# ConformalGuard

**A reproducible benchmark for evaluating conformal prediction under distribution shift.**

ConformalGuard studies how reliably conformal prediction methods maintain statistical coverage when the data distribution changes between calibration and deployment.

The project focuses on **CPU-friendly tabular classifiers** and evaluates the trade-off between prediction-set reliability, shift severity, and the additional cost required to restore coverage.

---

## Research Question

> **How reliable are conformal prediction sets from CPU-friendly tabular classifiers under covariate, label, and temporal distribution shift, and what is the cost of restoring coverage as shift severity increases?**

The benchmark is designed to answer two related questions:

1. **How much does distribution shift degrade conformal coverage?**
2. **How much additional data, computation, or adaptation is required to recover the desired coverage level?**

---

## Why ConformalGuard?

Conformal prediction provides prediction sets with finite-sample coverage guarantees under appropriate assumptions.

However, real-world deployment often violates those assumptions.

A model may be calibrated on one distribution and deployed on another because of:

* Changes in feature distributions
* Changes in class proportions
* Temporal drift
* Changes in the data-generating process

A conformal predictor that performs well under independent and identically distributed data may therefore provide prediction sets with substantially different empirical coverage after deployment.

**ConformalGuard** provides a controlled benchmark for studying this behavior.

---

## Shift Types

The benchmark is organized around three major forms of distribution shift.

### 1. Covariate Shift

The distribution of input features changes:

```text
P_train(X) ≠ P_test(X)
```

while the relationship between features and labels is intended to remain comparatively stable.

Example:

```text
Training distribution
        ↓
Feature distribution changes
        ↓
Deployment distribution
```

---

### 2. Label Shift

The class distribution changes:

```text
P_train(Y) ≠ P_test(Y)
```

while the class-conditional feature distributions are intended to remain comparatively stable.

This allows the benchmark to investigate how changes in class prevalence affect conformal prediction sets.

---

### 3. Temporal Shift

The deployment data comes from a later period than the calibration/training data.

```text
Past
 │
 ▼
Training ─── Calibration
                │
                │ Time
                ▼
             Deployment
```

Temporal shift is particularly important because it represents a common real-world deployment scenario where future observations differ from historical data.

---

# Core Evaluation Goals

ConformalGuard evaluates more than classification accuracy.

The primary focus is the behavior of **prediction sets** under distribution shift.

Key measurements include:

### Coverage

The fraction of test examples for which the true label is contained in the prediction set.

```text
Empirical Coverage
=
Correctly Covered Examples
──────────────────────────
       Test Examples
```

---

### Target Coverage

A desired coverage level such as:

```text
90%
95%
99%
```

can be specified for the benchmark.

The evaluation then measures how closely the observed coverage matches the target.

---

### Coverage Degradation

The benchmark measures the change in coverage between the reference setting and shifted settings.

```text
Coverage degradation
=
Reference coverage − Shifted coverage
```

This provides a direct measure of robustness to distribution shift.

---

### Prediction-Set Size

Coverage alone is not sufficient.

A predictor could increase coverage simply by returning very large prediction sets.

Therefore, ConformalGuard also measures:

```text
Average prediction-set size
```

This captures the efficiency–coverage trade-off.

---

### Cost of Recovery

When shift causes coverage to fall below the desired level, the benchmark evaluates the cost of restoring coverage.

Depending on the experimental protocol, this may include:

* Additional calibration data
* Additional labeled data
* Recalibration
* Adaptation
* Computational overhead

The goal is to quantify not only **whether coverage can be recovered**, but **what it costs to recover it**.

---

# Experimental Framework

The benchmark follows a reproducible pipeline:

```text
                    Dataset
                       │
                       ▼
                Data Preparation
                       │
                       ▼
              Train Base Classifier
                       │
                       ▼
                Calibration Set
                       │
                       ▼
              Conformal Predictor
                       │
              ┌────────┼────────┐
              │        │        │
              ▼        ▼        ▼
          Covariate  Label   Temporal
            Shift    Shift     Shift
              │        │        │
              └────────┼────────┘
                       ▼
                  Test Sets
                       │
                       ▼
                 Evaluation
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
      Coverage     Set Size     Recovery Cost
```

---

# Benchmark Dimensions

Experiments are intended to vary several dimensions systematically.

| Dimension         | Examples                              |
| ----------------- | ------------------------------------- |
| Classifier        | CPU-friendly tabular models           |
| Conformal method  | Configurable benchmark methods        |
| Target coverage   | 90%, 95%, 99%                         |
| Shift type        | Covariate, label, temporal            |
| Shift severity    | Multiple controlled levels            |
| Dataset           | Reproducible tabular datasets         |
| Random seed       | Fixed/reported seeds                  |
| Calibration size  | Controlled experiment parameter       |
| Recovery strategy | Configurable adaptation/recalibration |

The exact experimental matrix will be documented as the benchmark develops.

---

# Reproducibility

Reproducibility is a central design goal.

Experiments should record:

* Dataset version
* Data split
* Random seed
* Model configuration
* Calibration configuration
* Conformal method
* Target coverage
* Shift type
* Shift severity
* Evaluation metrics
* Software environment

A successful benchmark run should be reproducible from the repository configuration.

---

# Research Environment

The project currently focuses on establishing a controlled research environment before large-scale experiments are introduced.

The environment is intended to support:

```text
Python
   │
   ├── Dataset processing
   ├── Model training
   ├── Conformal prediction
   ├── Distribution-shift generation
   ├── Evaluation
   └── Experiment tracking
```

CPU-friendly methods are prioritized so that experiments can be reproduced without requiring specialized GPU infrastructure.

---

# Metrics

The benchmark will primarily report:

### Statistical Metrics

* Empirical coverage
* Coverage error
* Coverage degradation
* Coverage variability across runs

### Efficiency Metrics

* Average prediction-set size
* Median prediction-set size
* Prediction-set size distribution

### Robustness Metrics

* Performance versus shift severity
* Coverage degradation versus shift severity
* Recovery threshold

### Recovery Cost

Where applicable:

* Additional calibration samples
* Additional labeled samples
* Recalibration cost
* Runtime
* Computational overhead

---

# Experimental Principle

A central principle of ConformalGuard is to evaluate **coverage and efficiency together**.

For example:

```text
Method A
Coverage:        94.8%
Average set size: 1.4

Method B
Coverage:        99.2%
Average set size: 3.8
```

Method B has higher coverage, but it may be substantially less informative.

Therefore, benchmark results should not rank methods using coverage alone.

---

# Proposed Experiment Matrix

A typical experiment can be represented as:

```text
Classifier
    ×
Conformal Method
    ×
Target Coverage
    ×
Shift Type
    ×
Shift Severity
    ×
Random Seed
```

For each configuration:

```text
Train
  ↓
Calibrate
  ↓
Apply distribution shift
  ↓
Generate prediction sets
  ↓
Evaluate coverage
  ↓
Evaluate set size
  ↓
Measure recovery requirements
```

This structure allows results to be compared across different experimental conditions.

---

# Project Structure

The repository is currently in the **project foundation and research-environment setup stage**.

The planned structure is:

```text
ConformalGuard/
│
├── README.md
├── requirements.txt
├── pyproject.toml
│
├── configs/
│   └── experiments/
│
├── src/
│   └── conformalguard/
│       ├── data/
│       ├── models/
│       ├── conformal/
│       ├── shifts/
│       ├── evaluation/
│       └── experiments/
│
├── tests/
│
├── notebooks/
│
├── results/
│
├── figures/
│
└── docs/
```

As implementation progresses, each directory will be populated with independently testable components.

---

# Development Status

### Current

* [x] Project definition
* [x] Research question
* [x] Benchmark scope
* [x] Research environment setup
* [x] Reproducibility requirements defined
* [ ] Dataset pipeline
* [ ] Baseline classifiers
* [ ] Conformal prediction pipeline
* [ ] Covariate-shift experiments
* [ ] Label-shift experiments
* [ ] Temporal-shift experiments
* [ ] Coverage evaluation
* [ ] Recovery-cost experiments
* [ ] Automated experiment runner
* [ ] Benchmark results
* [ ] Final analysis

The project is intentionally **not presenting benchmark results yet**.

---

# Design Principles

## 1. Reproducibility First

Every experiment should be reproducible from explicit configuration and recorded random seeds.

## 2. CPU-Friendly Research

The benchmark prioritizes models and workflows that can be executed on standard CPU hardware.

## 3. Controlled Distribution Shift

Shift severity should be configurable and measurable rather than relying only on naturally occurring drift.

## 4. Coverage Is Not Enough

Prediction-set efficiency is evaluated alongside statistical coverage.

## 5. Honest Benchmarking

Experimental results will be reported together with their assumptions, limitations, and uncertainty rather than presenting conformal prediction as automatically robust to arbitrary distribution shift.

---

# Limitations

Conformal prediction's guarantees depend on assumptions about the relationship between calibration and test data.

Therefore, this project does **not** assume that conformal prediction automatically provides its nominal coverage under arbitrary distribution shift.

Instead, the purpose of the benchmark is to measure how empirical behavior changes when those assumptions are stressed.

The benchmark results should consequently be interpreted within the specific datasets, models, shift mechanisms, and experimental protocols used.

---

# Future Work

Planned development includes:

* Implementing baseline tabular classifiers
* Adding multiple conformal prediction methods
* Creating controlled shift generators
* Building standardized experiment configurations
* Running multi-seed experiments
* Quantifying coverage degradation
* Measuring prediction-set efficiency
* Evaluating recalibration strategies
* Measuring the cost of coverage recovery
* Generating publication-quality plots and tables
* Building a complete reproducibility pipeline

---

# Project Status

**Current status:** Foundation and research environment setup.

The repository is being developed toward a reproducible empirical study of conformal prediction under distribution shift.

---

## Author

**Chinmaya Satyam**

B.Tech – Computer Science and Artificial Intelligence
Sri Venkateswara University College of Engineering

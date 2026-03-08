# AI-Based MAchine Maintenance
### Machine Failure Prediction Using Machine Learning Techniques

> *Predict industrial equipment failures before they happen - using physics-informed feature engineering, synthetic oversampling, and a four-model stacking ensemble.*

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Why Predictive Maintenance Matters](#why-predictive-maintenance-matters)
3. [Applications](#applications)
4. [Dataset](#dataset)
5. [Project Structure](#project-structure)
6. [Methodology](#methodology)
   - [Data Preprocessing](#1-data-preprocessing)
   - [Synthetic Oversampling](#2-synthetic-minority-oversampling)
   - [Feature Engineering](#3-feature-engineering)
   - [Exploratory Data Analysis](#4-exploratory-data-analysis)
   - [Baseline Model](#5-baseline-model)
   - [Advanced Models](#6-advanced-models)
   - [Hyperparameter Tuning](#7-hyperparameter-tuning)
   - [Stacking Ensemble](#8-stacking-ensemble)
   - [Explainability](#9-explainability)
   - [Robustness Analysis](#10-robustness-analysis)
7. [Results](#results)
8. [Key Findings](#key-findings)
9. [Installation](#installation)
10. [Usage](#usage)
11. [Production Deployment](#production-deployment)
12. [Future Improvements](#future-improvements)
13. [References](#references)

---

## Project Overview

This project builds a production-grade machine learning system for **predictive maintenance** — the practice of detecting imminent equipment failures from real-time sensor data before any breakdown occurs.

The system is trained on the **AI4I 2020 Predictive Maintenance Dataset** (UCI ML Repository), a physically grounded synthetic dataset representing a CNC milling machine with five sensor channels. It achieves:

| Metric | Score |
|--------|-------|
| **ROC-AUC** | **> 0.98** |
| **Accuracy** | **> 95%** |
| **F1 Score** | **> 0.90** |

The pipeline spans the complete ML lifecycle: raw sensor ingestion → domain-informed feature engineering → class imbalance correction → multi-model training → hyperparameter optimisation → stacking ensemble → explainability → model persistence.

---

## Why Predictive Maintenance Matters

Industrial equipment failures are among the most costly, disruptive, and preventable events in modern manufacturing. Traditional maintenance strategies fall into two categories, each with significant drawbacks:

**Reactive maintenance** — fix it after it breaks. This is the most expensive approach: it causes unplanned downtime, emergency repair costs, potential safety incidents, and scrapped production runs.

**Preventive maintenance** — service on a fixed schedule regardless of actual condition. This wastes resources (parts replaced while still functional) and still fails to prevent failures that occur between scheduled service intervals.

**Predictive maintenance** solves both problems. By continuously analysing sensor readings and issuing alerts only when failure is genuinely imminent, it enables maintenance teams to act at precisely the right moment — avoiding both unexpected breakdowns and unnecessary interventions.

**Economic impact at a glance:**

- Unplanned downtime costs manufacturing industry an estimated crores globally.
- A single CNC machine stoppage costs massively, depending on the production line.
- Predictive maintenance programs typically yield a **25–30% reduction** in maintenance costs and a **70–75% reduction** in unexpected breakdowns.
---

## Applications

The trained model and methodology in this project are directly applicable across a wide range of industries:

| Industry | Specific Application |
| **CNC Manufacturing** | Spindle wear monitoring, tool breakage prediction, coolant pump health |
| **Aviation & Aerospace** | Jet engine turbofan degradation, hydraulic actuator leaks, landing gear wear |
| **Automotive Production** | Press and stamping tooling wear, conveyor bearing failure, paint line robotics |
| **Industrial Robotics** | Joint servo degradation, gearbox wear, end-effector force anomalies |
| **Energy & Utilities** | Wind turbine gearbox failure, transformer hot-spot detection, pump cavitation |
| **Oil & Gas** | Compressor valve failure, pipeline pump degradation, wellhead equipment health |
| **Rail Transportation** | Wheel bearing degradation, brake wear, traction motor health |
| **Semiconductor Fabrication** | Vacuum pump failure, wafer handler arm wear, plasma etch chamber maintenance |

The core approach — encoding physical failure thresholds as binary features and using an ensemble of complementary classifiers — generalises naturally to any domain where sensor readings have known physical operating bounds.

---

## Dataset

**Name:** AI4I 2020 Predictive Maintenance Dataset  
**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset)  
**Paper:** Matzka, S. (2020). *Explainable Artificial Intelligence for Predictive Maintenance Applications.* IEEE AIKE.

### Raw Features

| Feature | Type | Description |
|---------|------|-------------|
| `Type` | Categorical | Machine quality variant: L (Low), M (Medium), H (High) |
| `Air temperature [K]` | Float | Ambient air temperature in Kelvin |
| `Process temperature [K]` | Float | Machine process temperature in Kelvin |
| `Rotational speed [rpm]` | Integer | Spindle motor rotational speed in RPM |
| `Torque [Nm]` | Float | Torque applied to the spindle in Newton-metres |
| `Tool wear [min]` | Integer | Cumulative tool usage time in minutes |

### Target Variable

`Machine failure` — binary label: `0` = No failure, `1` = Failure.

### Failure Modes (defined in paper)

| Code | Name | Physical Condition |
| TWF | Tool Wear Failure | Tool wear in range [200, 240] min + random trigger |
| HDF | Heat Dissipation Failure | `ΔT < 8.6 K` AND `RPM < 1380` |
| PWF | Power Failure | Mechanical power < 3,500 W or > 9,000 W |
| OSF | Overstrain Failure | `Torque × Tool Wear > 11,000` (L), `12,000` (M), `13,000` (H) |
| RNF | Random Failure | 0.1% independent random chance |

> **Note:** The sub-type columns (TWF, HDF, PWF, OSF, RNF) are **excluded from model inputs** — they are downstream consequences of failure and would constitute data leakage. Only the raw sensor readings are used as predictors.

### Class Imbalance

Only ~3–4% of records represent machine failures, reflecting the real-world rarity of equipment failure events. This severe imbalance (approximately 30:1) requires dedicated handling and is addressed through synthetic oversampling and class-weighted models.

---

## Project Structure

```
Failure_Predictor/
│
├── Failure_Predictor.ipynb   # Complete ML pipeline notebook
├── README.md                      # This file
│
└── model_artifacts/
    ├── machine_failure_model.pkl  # Trained stacking ensemble (joblib)
    ├── hgb_model.pkl              # Tuned HistGradientBoosting (standalone)
    ├── feature_scaler.pkl         # StandardScaler for SVM pipeline
    ├── fit_stats.pkl              # Training statistics for feature engineering
    ├── model_metadata.pkl         # Full metadata: features, metrics, params
    ├── final_dashboard.png        # Summary results dashboard
    └── robustness_dashboard.png   # Robustness analysis charts
```

---

## Methodology

The project follows a rigorous, industry-standard ML pipeline across 15 notebook sections.

---

### 1. Data Preprocessing

**Goal:** Prepare clean, model-ready data without information leakage.

**Steps:**

1. **Remove identifier columns** — `UDI` and `Product ID` carry no predictive signal and are dropped.
2. **Remove downstream leakage columns** — The five failure sub-type columns (TWF, HDF, PWF, OSF, RNF) are consequences of the target label. Including them would allow the model to trivially predict failure by detecting its own cause — a classic data leakage scenario.
3. **Ordinal encode machine type** — `L → 0`, `M → 1`, `H → 2`. This preserves the natural ordering (quality tier) of the three machine variants.
4. **Verify data integrity** — Confirm zero missing values before splitting.
5. **Stratified train/test split (80/20)** — `stratify=y` ensures both training and test sets maintain the same ~3–4% failure rate as the full dataset, preventing accidentally skewed evaluation sets.
6. **Feature scaling** — `StandardScaler` is fit exclusively on the training set and applied to both sets. Scaling is used for SVM (distance-based, scale-sensitive) but not for tree-based models.

---

### 2. Synthetic Minority Oversampling

**Problem:** With a 30:1 class imbalance, even a model that predicts "no failure" for every sample achieves 96%+ accuracy. Class weights partially mitigate this, but the model still sees very few failure examples during training.

**Solution:** A SMOTE-equivalent oversampling procedure implemented in pure NumPy:

1. Identify all minority class (failure) samples in the training set
2. Draw random pairs of minority class neighbours
3. Interpolate linearly between each pair: `x_new = x_A × λ + x_B × (1 − λ)` where `λ ~ Uniform(0, 1)`
4. Add small Gaussian noise (`σ = 2%` of feature standard deviation) to increase synthetic sample diversity
5. Augment the training set to a **10:1 majority:minority ratio** — more balanced than the original 30:1 without the over-representation risks of a perfectly balanced 1:1 ratio

**Critical constraint:** Oversampling is applied **only to the training set**. The test set retains the original class distribution to ensure honest, deployment-realistic evaluation.

**Reference:** Chawla, N.V. et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique.* JAIR 16, 321–357.

---

### 3. Feature Engineering

The most impactful component of the pipeline. Features are organised into three tiers based on their origin and purpose.

#### Tier 1 — Physics-Based Interactions

Derived from the known physical relationships governing milling machine operation:

| Feature | Formula | Physical Meaning |
| `temp_diff` | `Process Temp − Air Temp` | Thermal differential; low ΔT indicates poor heat dissipation |
| `power_W` | `2π × (RPM/60) × Torque` | Actual mechanical power in Watts; defined in paper's PWF condition |
| `torque_x_wear` | `Torque × Tool Wear` | Overstrain composite; the exact quantity in the paper's OSF formula |
| `rpm_x_torque` | `RPM × Torque` | Combined mechanical load proxy |
| `wear_per_rpm` | `Tool Wear / (RPM + 1)` | Wear velocity relative to speed |
| `temp_ratio` | `Process Temp / Air Temp` | Thermal process efficiency |
| `power_deviation` | `|power_W − 6250|` | Distance from midpoint of the safe power band [3500, 9000 W] |
| `power_deficit` | `power_W − 6250` | Signed deviation — distinguishes under- from over-powered states |
| `high_torque` | `Torque > 60 Nm` | Binary flag for empirically high-torque operation |
| `wear_stage` | `cut(Tool Wear, [0,100,180,300])` | Ordinal wear stage: early / mid / critical |

#### Tier 2 — Failure Zone Indicators

The single most impactful tier. Rather than forcing the model to rediscover the paper's threshold conditions from raw sensor values, these binary flags directly encode the known physical failure boundaries:

| Feature | Condition | Failure Mode Captured |
| `in_twf_zone` | `200 ≤ Tool Wear ≤ 240` | Tool Wear Failure zone |
| `in_hdf_zone` | `ΔT < 8.6 K AND RPM < 1380` | Heat Dissipation Failure zone |
| `in_pwf_zone` | `power_W < 3500 OR > 9000` | Power Failure zone |
| `in_osf_zone_L` | `torque_x_wear > 11,000` | OSF zone for Low-quality machines |
| `in_osf_zone_M` | `torque_x_wear > 12,000` | OSF zone for Medium-quality machines |
| `in_osf_zone_H` | `torque_x_wear > 13,000` | OSF zone for High-quality machines |
| `in_osf_zone_type` | Type-conditional OSF threshold | Precise per-machine OSF flag |
| `failure_zone_count` | Sum of all zone flags | How many failure zones are simultaneously triggered |
| `any_zone` | `failure_zone_count > 0` | Binary: at least one failure zone active |

The `in_osf_zone_type` feature is particularly powerful — it applies the correct overstrain threshold based on the machine's quality tier, matching the paper's specification exactly.

#### Tier 3 — Statistical Deviation Scores

Captures readings that are anomalous relative to the training distribution:

| Feature | Formula | Purpose |
| `torque_zscore` | `(Torque − μ) / σ` | Standardised torque anomaly |
| `rpm_zscore` | `(RPM − μ) / σ` | Standardised RPM anomaly |
| `wear_zscore` | `(Wear − μ) / σ` | Standardised wear anomaly |
| `anomaly_score` | `√(z_τ² + z_ω² + z_w²)` | Diagonal Mahalanobis-like distance from normal operating point |
| `log_torque` | `log(1 + Torque)` | Log transform to reduce right-skew |
| `sqrt_wear` | `√Tool Wear` | Square-root to stabilise wear variance |
| `log_power` | `log(1 + power_W)` | Log-normalised power |
| `power_sq` | `power_W² / 10⁸` | Quadratic power term for nonlinear boundary fitting |
| `wear_stage_x_torque` | `wear_stage × Torque` | Interaction between wear severity and mechanical load |

> **Leakage prevention:** All z-score statistics (`μ`, `σ`) are computed exclusively from the training set and saved to `fit_stats.pkl`. They are then applied — not re-fitted — to the test set and any production inference requests.

**Final feature count:** 6 raw sensors + 1 encoded type = 7 inputs → **30 features** after engineering.

---

### 4. Exploratory Data Analysis

A structured visual investigation of the dataset before modelling:

- **Class distribution** — bar chart and pie chart confirming the ~3–4% failure rate
- **Correlation heatmap** — lower-triangular Pearson correlations with annotated values
- **Feature distributions by class** — overlaid histograms for all five sensor channels, stratified by failure/no-failure
- **Boxplots** — median and quartile comparison between failure and non-failure groups, annotated with Mann-Whitney U test p-values to confirm statistical significance
- **Zone effectiveness analysis** — bar charts showing failure rates inside vs. outside each Tier 2 zone, demonstrating that samples inside failure zones have dramatically elevated failure rates
- **Feature-target correlation ranking** — horizontal bar chart of all 30 engineered features sorted by absolute correlation with the target, colour-coded by tier

---

### 5. Baseline Model

**Logistic Regression** is trained on the full engineered feature set (with StandardScaler) as a performance baseline. It serves as the lower bound against which all subsequent models are compared.

Key settings: `class_weight='balanced'`, `solver='lbfgs'`, `max_iter=1000`.

This baseline also validates that the engineered features alone carry strong signal — if Logistic Regression performs well, it confirms the features are well-constructed before introducing model complexity.

---

### 6. Advanced Models

Four model families are trained and compared using **Stratified 5-Fold Cross-Validation** (scoring: ROC-AUC):

#### Random Forest
An ensemble of decision trees trained via bagging (bootstrap sampling). Each tree sees a random subset of features at each split, reducing correlation between trees and lowering variance. Uses `class_weight='balanced'` and `n_estimators=300`.

#### Extra Trees (Extremely Randomised Trees)
Similar to Random Forest but uses fully random split thresholds rather than finding the optimal split — further reducing variance at a slight bias cost. Typically faster to train and more robust to noisy features.

#### HistGradientBoostingClassifier
Sklearn's native histogram-based gradient boosting implementation, equivalent in design to LightGBM. Trees are built sequentially — each corrects the residuals of the previous — using histogram binning (default 255 bins) for fast, memory-efficient training. Supports `class_weight='balanced'` natively and handles missing values without imputation.

#### Support Vector Machine (SVM, RBF Kernel)
Maps the feature space into a high-dimensional Hilbert space via the RBF kernel `K(x, x') = exp(−γ‖x − x'‖²)` and finds the maximum-margin hyperplane. Particularly effective for the roughly elliptical failure regions in power-torque space. Requires StandardScaler preprocessing. Wrapped in `CalibratedClassifierCV` (Platt scaling) to produce calibrated probability estimates for the meta-learner.

---

### 7. Hyperparameter Tuning

`RandomizedSearchCV` is used for all four models — more efficient than grid search while exploring a wide parameter space. All searches use the same **Stratified 5-Fold CV** and optimise for **ROC-AUC**.

#### HistGradientBoosting — 60 iterations

| Parameter | Search Range | Effect |
| `max_iter` | [200, 800] | Number of boosting rounds |
| `max_leaf_nodes` | [20, 127] | Controls tree complexity |
| `max_depth` | None, 4, 6, 8, 10, 12 | Maximum depth per tree |
| `min_samples_leaf` | [10, 80] | Leaf size regularisation |
| `learning_rate` | Uniform(0.01, 0.21) | Step size per boosting round |
| `l2_regularization` | Uniform(0, 1.0) | Weight regularisation strength |
| `max_bins` | 63, 127, 255 | Histogram bin resolution |
| `colsample_bytree` | Uniform(0.5, 1.0) | Feature subsampling per tree |

#### Random Forest — 60 iterations

| Parameter | Search Range | Effect |
| `n_estimators` | [200, 800] | Number of trees in ensemble |
| `max_depth` | None, 8, 12, 18, 25, 35 | Tree depth cap |
| `min_samples_split` | [2, 15] | Minimum samples to split a node |
| `min_samples_leaf` | [1, 8] | Minimum samples per leaf |
| `max_features` | sqrt, log2, 0.4, 0.6, 0.8 | Features per split |
| `bootstrap` | True, False | Whether to use bootstrap sampling |
| `max_samples` | None, 0.7, 0.8, 0.9 | Fraction of samples per tree |

#### Extra Trees — 40 iterations

| Parameter | Search Range | Effect |
| `n_estimators` | [200, 600] | Number of trees |
| `max_depth` | None, 10, 15, 20, 30 | Tree depth cap |
| `min_samples_split` | [2, 12] | Minimum split samples |
| `min_samples_leaf` | [1, 6] | Minimum leaf samples |
| `max_features` | sqrt, log2, 0.5, 0.7 | Features per split |

#### SVM — 30 iterations

| Parameter | Search Range | Effect |
| `C` | Uniform(0.1, 100) | Regularisation: high C = tighter fit, lower margin |
| `gamma` | scale, auto, 1e-4–1.0 | RBF bandwidth: high γ = localised, spiky boundary |
| `kernel` | rbf, poly | Decision boundary type |
| `degree` | 2, 3 | Polynomial degree (only used when `kernel='poly'`) |
| `class_weight` | balanced, None | Whether to up-weight minority class |
| `shrinking` | True, False | Heuristic for faster training convergence |

The SVM is wrapped in `CalibratedClassifierCV(method='sigmoid')` — Platt scaling fits a sigmoid function to map raw decision scores to calibrated probability estimates. The entire `[StandardScaler → CalibratedSVC]` chain is wrapped in a `sklearn.pipeline.Pipeline` so it integrates cleanly with the `StackingClassifier` without leaking test-set statistics into the scaler fit.

---

### 8. Stacking Ensemble

The final model combines all four tuned base learners into a **two-level stacking architecture**:

```
Level 0 — Base Learners (trained on 4/5 folds each, predict on held-out fold):
  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────────┐
  │  HGB (Tuned)     │  │  RF (Tuned)      │  │  ET (Tuned)      │  │  SVM Pipeline     │
  │  Gradient boost  │  │  Bagging         │  │  Random splits   │  │  Scaled + Kernel  │
  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘  └────────┬──────────┘
           │                     │                      │                     │
           └─────────────────────┴──────────────────────┴─────────────────────┘
                                               │
                              (out-of-fold probability predictions)
                                               │
Level 1 — Meta-Learner:
  ┌────────────────────────────────────────────────────────────────────────┐
  │  Logistic Regression (C=1.0, class_weight='balanced')                  │
  │  Learns to optimally combine the four base learner probability outputs │
  └────────────────────────────────────────────────────────────────────────┘
                                               │
                                  Final P(failure) ∈ [0, 1]
```

**Why each base learner contributes differently:**

- **HGB** — excels at learning sharp nonlinear decision boundaries around the failure zone thresholds. Sequential boosting iteratively reduces residuals, converging to highly accurate probability estimates in complex regions.
- **RF** — provides stable, well-calibrated probability estimates through variance reduction via bagging. Robust to individual outlier samples.
- **ET** — high randomisation introduces prediction diversity that is poorly correlated with both HGB and RF, which is exactly what the meta-learner needs to improve upon any single model.
- **SVM** — the maximum-margin geometric approach captures failure boundaries defined by circular/elliptical regions in the power-torque-wear feature space that gradient methods may slightly overfit.

**Out-of-fold predictions:** The `StackingClassifier` uses `cv=StratifiedKFold(5)` to generate base learner predictions. Each base learner is trained on 4 folds and predicts on the remaining fold, cycling through all 5 folds. This gives the meta-learner training data that was never seen by the base learners that generated it — preventing base learner overfitting from propagating to the meta-learner.

**Meta-learner:** Logistic Regression is intentionally simple — its role is to learn the optimal linear combination of four probability estimates, not to model additional complexity. A simple meta-learner reduces the risk of overfitting at Level 1.

---

### 9. Explainability

#### Permutation Importance

For every feature, the test set ROC-AUC is measured before and after randomly shuffling that feature's values (repeated 15 times). The mean drop in AUC when a feature is shuffled is its permutation importance. This method:

- Measures actual predictive contribution on **held-out data**, not training data
- Is unbiased toward high-cardinality features (unlike impurity-based importance)
- Works with any model type, including the stacking ensemble
- Reports uncertainty (standard deviation across repeats)

Both permutation importance and RF impurity importance are computed and compared — high agreement between the two methods validates that important features are genuinely predictive.

#### Partial Dependence Plots (PDP) + ICE

**1D PDPs** show how the predicted failure probability changes as a single feature varies from its minimum to maximum value, with all other features held at their marginal distribution (averaged out). This answers: *"What is the average effect of increasing tool wear on failure probability?"*

**Individual Conditional Expectation (ICE) plots** show one line per test sample instead of a single average. Divergence between ICE lines reveals heterogeneous effects — for example, the impact of tool wear on failure probability differs based on machine type and torque level.

**2D PDPs** visualise the joint interaction between two features as a colour-mapped surface, identifying the high-risk combination region (e.g., the joint zone of high torque × high tool wear).

#### Zone Contribution Analysis

For correctly predicted failures vs. missed failures (false negatives), the fraction of samples inside each failure zone is compared. This reveals whether missed failures are predominantly Random Failures (RNF) — which are by definition unpredictable — or whether specific failure modes are being systematically missed.

---

### 10. Robustness Analysis

- **Repeated 5×5 cross-validation** — 5 repetitions of 5-fold CV (25 total fits) to estimate the true mean and variance of the ROC-AUC. Provides a 95% confidence interval for generalisation performance.
- **Prediction probability distribution** — overlaid histograms of predicted failure probabilities for the two classes. A well-calibrated model produces a bimodal distribution with clear separation between the classes.
- **Precision-Recall curve** — more informative than ROC for imbalanced problems. The area under the PR curve (average precision) quantifies model quality independent of the decision threshold.
- **Threshold sensitivity analysis** — precision, recall, and F1 are plotted as a function of the decision threshold (0 to 1), allowing operators to choose the operating point that matches their cost structure.
- **False Negative deep-dive** — the sensor readings for missed failures are examined to understand the operating conditions in which the model fails.

---

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| Logistic Regression (baseline) | ~0.92 | ~0.65 | ~0.80 | ~0.72 | ~0.92 |
| Random Forest (tuned) | ~0.97 | ~0.88 | ~0.89 | ~0.88 | ~0.98 |
| Extra Trees (tuned) | ~0.97 | ~0.87 | ~0.90 | ~0.88 | ~0.98 |
| HistGradientBoosting (tuned) | ~0.97 | ~0.89 | ~0.91 | ~0.90 | ~0.99 |
| SVM — RBF (tuned) | ~0.95 | ~0.82 | ~0.84 | ~0.83 | ~0.97 |
| **Stacking Ensemble** | **~0.97** | **~0.90** | **~0.92** | **~0.91** | **~0.989** |

> Exact values will vary slightly depending on whether the dataset is loaded from UCI directly or generated via the synthetic fallback. The figures above reflect the synthetic fallback with `RANDOM_SEED=42`.

**Repeated 5×5 CV ROC-AUC:** `0.987 +/- 0.004` — confirming stability across different data splits.

---

## Key Findings

**1. Domain knowledge beats blind feature search.**
The Tier 2 failure zone flags — which directly encode the paper's physical failure conditions — contributed more AUC improvement than any other single component. This validates the principle that understanding the failure physics is more valuable than adding more generic features.

**2. The type-conditional OSF threshold is the most important single feature.**
`in_osf_zone_type` (which applies the correct torque × wear threshold based on machine quality: 11,000 for L, 12,000 for M, 13,000 for H) consistently ranked first in both permutation importance and RF impurity importance. Generic thresholds lose predictive signal by treating all machine types identically.

**3. Class imbalance correction matters more than model choice.**
The gap between an unweighted model and a SMOTE-oversampled model was larger than the gap between Logistic Regression and Random Forest, given the same features. Handling the 30:1 imbalance correctly is the single most important preprocessing decision.

**4. Diversity among base learners drives ensemble improvement.**
The SVM, despite having a lower standalone ROC-AUC than the tree models, improved the stacking ensemble's performance. This is because SVM errors are uncorrelated with the boosting/bagging errors — the meta-learner learns to use the SVM's correct predictions to override ensemble uncertainty.

**5. Missed failures are predominantly unpredictable.**
False Negative analysis shows that the vast majority of missed failures are Random Failures (RNF) — defined in the dataset as 0.1% independent chance events with no physical precursor. These are intrinsically unpredictable from sensor readings alone. The model captures nearly all physically-grounded failure modes.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/paul-dev-28/Machine_Failure_Predictor.git
cd Failure_Predictor

# Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.3.0
scipy>=1.9.0
joblib>=1.2.0
jupyter>=1.0.0
```

> The pipeline uses only standard libraries. No XGBoost, LightGBM, or imbalanced-learn installation is required — `HistGradientBoostingClassifier` and the custom SMOTE implementation are built on scikit-learn and NumPy respectively.

---

## Usage

### Running the Full Notebook

```bash
jupyter notebook Failure_Predictor.ipynb
```

Run all cells from top to bottom. The notebook will:
1. Attempt to load the AI4I dataset from the UCI URL
2. Fall back to a high-fidelity synthetic generator if the URL is unavailable
3. Execute the full pipeline and save model artifacts to `model_artifacts/`

### Loading the Trained Model

```python
import joblib
import pandas as pd
import numpy as np

# Load artifacts
model     = joblib.load('model_artifacts/machine_failure_model.pkl')
fit_stats = joblib.load('model_artifacts/fit_stats.pkl')
metadata  = joblib.load('model_artifacts/model_metadata.pkl')

print(f"Model type:  {metadata['model_type']}")
print(f"Test AUC:    {metadata['test_roc_auc']:.5f}")
print(f"Features:    {len(metadata['feature_names'])}")
```

### Making a Prediction

```python
# Define the feature engineering function (copy from notebook Section 6)
def engineer_features(df, fit_stats):
    d = df.copy()
    # Tier 1: Physics
    d['temp_diff']      = d['Process temperature [K]'] - d['Air temperature [K]']
    d['power_W']        = (2 * np.pi * d['Rotational speed [rpm]'] / 60) * d['Torque [Nm]']
    d['torque_x_wear']  = d['Torque [Nm]'] * d['Tool wear [min]']
    # ... (full function in notebook Section 6)
    return d

# Single sensor reading prediction
sensor_reading = {
    'Type':                       1,      # 0=L, 1=M, 2=H
    'Air temperature [K]':        298.1,
    'Process temperature [K]':    308.6,
    'Rotational speed [rpm]':     1408,
    'Torque [Nm]':                62.1,
    'Tool wear [min]':            215,
}

df_input = pd.DataFrame([sensor_reading])
df_fe    = engineer_features(df_input, fit_stats)

failure_prob = model.predict_proba(df_fe)[0, 1]

result = {
    'failure_probability': round(float(failure_prob), 4),
    'alert':               failure_prob > 0.5,
    'urgency':             'HIGH'   if failure_prob > 0.80 else
                           'MEDIUM' if failure_prob > 0.50 else 'LOW',
}
print(result)
# -> {'failure_probability': 0.9341, 'alert': True, 'urgency': 'HIGH'}
```

---

## Production Deployment

The model is designed to integrate into a real-time industrial monitoring system:

```
IoT Sensor Layer          Processing Layer          Alert Layer
──────────────────        ────────────────          ────────────
CNC Machine               Feature Engineering       SCADA System
  ↓ (5-second poll)          (~1 ms)                 
MQTT / OPC-UA ->  ->  ->    Stacking Inference  ->  ->  Maintenance
                             (~5 ms)               Scheduling
                          Threshold Check             
                                                  SMS / Email
                          Logging (DB)              Alert
```

**Recommended decision thresholds by cost structure:**

| Operational Context | Threshold | Expected Recall | Expected Precision |
| Catastrophic failure cost (aviation, nuclear) | 0.20 | ~98% | ~70% |
| Balanced default (general manufacturing) | 0.50 | ~92% | ~90% |
| High false-alarm cost (frequent false stops) | 0.70 | ~80% | ~96% |

**Inference latency:** The stacking ensemble requires four base learner inferences plus one meta-learner inference per prediction. On a modern CPU this completes in under 10 milliseconds per sample — well within the latency requirements of typical sensor polling intervals (seconds to minutes).

---

## Future Improvements

### 1. Time-Series Modelling
The current model treats each sensor reading as independent. Real machine degradation follows temporal trajectories — tool wear accumulates, temperatures drift, vibration signatures evolve. Incorporating sequence models (LSTM, Temporal Convolutional Networks, Transformer) over rolling windows would capture degradation trends that snapshot models cannot see.

### 2. Threshold Calibration
The default decision threshold of 0.50 is not optimal for all cost structures. Calibrating the threshold on a held-out validation set using the actual cost ratio of false negatives (missed failure) to false positives (false alarm) would maximise economic value rather than F1 score.

### 3. Prediction Drift Monitoring
In production, sensor distributions shift over time due to machine aging, environmental changes, and component replacements. Implementing drift detection (Population Stability Index, Kolmogorov-Smirnov tests on feature distributions) and triggering retraining when drift is detected would maintain model performance over the machine lifecycle.

### 4. Online / Incremental Learning
Periodically retraining the model on newly labelled failure events (with a sliding window or forgetting factor) would allow the model to adapt to gradual changes in machine behaviour without full retraining from scratch.

### 5. Multi-Label Failure Mode Prediction
Currently the model predicts binary failure/no-failure. Predicting *which* failure mode is imminent (TWF vs. HDF vs. PWF vs. OSF) would allow maintenance teams to prepare the correct spare parts and procedures before arrival. This is a multi-label classification problem since multiple failure modes can co-occur.

### 6. Remaining Useful Life (RUL) Estimation
Beyond binary failure prediction, estimating the number of minutes/cycles remaining before failure would enable even more precise maintenance scheduling. This is a regression problem and could be modelled with survival analysis (Cox Proportional Hazards, Weibull regression) or deep learning approaches.

### 7. Uncertainty Quantification
The current model outputs a point estimate (probability) but not its uncertainty. Bayesian methods (Monte Carlo Dropout, Conformal Prediction, Deep Ensembles) would produce calibrated confidence intervals — critical for high-stakes decisions where "I'm 80% confident with ±15% uncertainty" is more actionable than a single number.

### 8. Anomaly Detection Pre-Filter
Before the failure classifier, a one-class anomaly detector (Isolation Forest, Autoencoder) trained only on normal operation data could flag unusual readings that fall outside the normal operating envelope — providing an additional safety layer independent of the supervised classifier.

### 9. Federated Learning
In a multi-plant scenario, each plant's machine data is sensitive and cannot be centralised. Federated learning would allow model updates to be trained locally at each plant and aggregated centrally (e.g., via FedAvg) without sharing raw sensor data.

### 10. Explainability Enhancement
Replacing Partial Dependence Plots with SHAP (SHapley Additive exPlanations) would provide per-prediction local explanations — critical for operator trust and regulatory compliance in safety-critical environments. SHAP waterfall plots show exactly which sensor reading pushed a specific alert, making the model's reasoning transparent to maintenance engineers.

---

## References

1. Matzka, S. (2020). *Explainable Artificial Intelligence for Predictive Maintenance Applications.* 3rd IEEE International Conference on Artificial Intelligence and Knowledge Engineering (AIKE), pp. 69–74.

2. Chawla, N.V., Bowyer, K.W., Hall, L.O., & Kegelmeyer, W.P. (2002). *SMOTE: Synthetic Minority Over-sampling Technique.* Journal of Artificial Intelligence Research, 16, 321–357.

3. Wolpert, D.H. (1992). *Stacked Generalisation.* Neural Networks, 5(2), 241–259.

4. Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.Y. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* Advances in Neural Information Processing Systems, 30.

5. Breiman, L. (2001). *Random Forests.* Machine Learning, 45(1), 5–32.

6. Geurts, P., Ernst, D., & Wehenkel, L. (2006). *Extremely Randomised Trees.* Machine Learning, 63(1), 3–42.

7. Cortes, C., & Vapnik, V. (1995). *Support-vector networks.* Machine Learning, 20(3), 273–297.

8. Platt, J. (1999). *Probabilistic Outputs for Support Vector Machines and Comparisons to Regularised Likelihood Methods.* Advances in Large Margin Classifiers, 10(3), 61–74.

9. Biau, G., & Scornet, E. (2016). *A Random Forest Guided Tour.* TEST, 25(2), 197–227.

10. Goldstein, A., Kapelner, A., Bleich, J., & Pitkin, E. (2015). *Peeking Inside the Black Box: Visualising Statistical Learning With Plots of Individual Conditional Expectation.* Journal of Computational and Graphical Statistics, 24(1), 44–65.

---

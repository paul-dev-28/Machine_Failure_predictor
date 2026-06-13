## AI-Based Predictive Maintenance

Machine failure prediction using machine learning on the AI4I 2020 Predictive Maintenance Dataset.

## Overview

Developed an end-to-end machine learning pipeline to predict industrial machine failures from sensor data. The project includes feature engineering, class imbalance handling, model tuning, explainability, and ensemble learning.

## Dataset

* AI4I 2020 Predictive Maintenance Dataset
* Source: UCI Machine Learning Repository
* Target: Binary machine failure prediction

## Features

* Data preprocessing and feature engineering
* Custom oversampling for class imbalance
* Random Forest, Extra Trees, HistGradientBoosting, and SVM models
* Stacking ensemble
* SHAP-based explainability
* Cross-validation and hyperparameter tuning

## Results

| Metric | Score |
|---------|---------|
| ROC-AUC | 0.98 |
| Accuracy | 97% |
| F1 Score | 0.91 |

## Tech Stack

* Python
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn
* SHAP

## Repository Structure

```text
Machine_Failure_Predictor/
│
├── Failure_Predictor.ipynb
├── README.md
├── requirements.txt
└── model_artifacts/
```

## How to Run

```bash
git clone https://github.com/paul-dev-28/Machine_Failure_Predictor.git

cd Machine_Failure_Predictor

pip install -r requirements.txt

jupyter notebook Failure_Predictor.ipynb
```

## Future Work

* Time-series modelling
* Remaining Useful Life (RUL) estimation
* Real-time deployment
* Drift monitoring

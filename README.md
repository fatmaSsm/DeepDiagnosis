# DeepDiagnosis || 🧬

**Genomic Variant Pathogenicity Classification | TEKNOFEST 2026 – Healthcare AI**

DeepDiagnosis is a machine learning research project developed by the **DeepDiagnosis team** for the **TEKNOFEST 2026 Healthcare Artificial Intelligence Competition**.

The project investigates computational approaches to genomic variant pathogenicity classification through data preprocessing, ensemble learning, stratified cross-validation, model evaluation, and decision threshold optimization.

Four dataset groups were examined: **MASTER, KANSER, CFTR, and PAH**.

> ⚠️ **Disclaimer:** This project is intended exclusively for research and educational purposes. It has not been clinically validated and must not be used for medical diagnosis, treatment, or clinical decision-making.

---

## 🎯 Project Objectives

The primary objective of DeepDiagnosis is to investigate machine learning approaches for identifying potentially pathogenic genomic variants while accounting for the challenges associated with genomic classification datasets.

The project focuses on:

- Developing and comparing supervised machine learning models.
- Processing incomplete and heterogeneous tabular data.
- Addressing class imbalance and evaluating minority-class performance.
- Investigating ensemble learning methods.
- Optimizing classification decision thresholds.
- Evaluating false negatives and false positives across different variant groups.
- Analyzing model performance using multiple evaluation metrics.

The research emphasizes the importance of balancing sensitivity and specificity rather than relying solely on accuracy.

## 🛠️ Technologies & Tools

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Algorithms | HistGradientBoosting, RandomForest, LogisticRegression |
| Ensemble Learning | Soft Voting |
| Model Evaluation | Stratified Cross-Validation, Confusion Matrix, ROC-AUC, PR-AUC |
| Performance Metrics | Recall, Precision, Specificity, F1-Score, MCC |
| Visualization | Matplotlib |
| Exploratory Analysis | Orange Data Mining |

## 🧠 Machine Learning Methodology

### 1. Data Preparation

The original competition experiments involved preprocessing genomic variant datasets to support machine learning analysis.

The workflow included:

- Inspecting dataset characteristics and feature distributions.
- Identifying missing values.
- Handling numerical and categorical variables.
- Investigating feature quality and class distributions.
- Preparing datasets for model training and evaluation.

The historical competition pipeline contains known preprocessing leakage risks, documented in `docs/METHODOLOGY.md`.

### 2. Model Development

Three primary classification algorithms were investigated:

**HistGradientBoosting Classifier**

A gradient boosting method designed for efficient learning from tabular data.

**Random Forest Classifier**

An ensemble of decision trees that captures nonlinear relationships and feature interactions.

**Logistic Regression**

A linear classification algorithm used as a baseline and as a component of the original experimental approach.

### 3. Ensemble Learning

The original competition experiments investigated soft-voting ensembles to combine predictions from multiple classifiers.

The purpose was to explore whether combining different modeling approaches could improve classification performance and robustness.

### 4. Cross-Validation and Threshold Optimization

The experimental workflow explored stratified cross-validation to preserve class proportions across folds.

Classification thresholds were also investigated to evaluate trade-offs between:

- Recall (Sensitivity)
- Specificity
- Precision
- F1-Score
- Matthews Correlation Coefficient (MCC)
- False Negative Rate

Threshold selection is especially important in imbalanced classification problems because different thresholds can substantially change false-positive and false-negative rates.

The original experiments should be interpreted as exploratory because the historical implementation has identified methodological limitations.

---

## 📊 Historical Model Performance

The following results were recorded during the team's internal competition experiments using optimized decision thresholds.

| Dataset Group | Final Threshold | Recall (%) | Specificity (%) | F1-Score (%) | MCC | False Negative Rate (%) |
|---|---:|---:|---:|---:|---:|---:|
| MASTER | 0.578 | 89.1 | 50.9 | 71.7 | 0.424 | 10.9 |
| KANSER | 0.654 | 85.0 | 81.7 | 80.0 | 0.656 | 15.0 |
| CFTR | 0.654 | 100.0 | 40.0 | 81.2 | 0.523 | 0.0 |
| PAH | 0.714 | 93.5 | 29.0 | 77.5 | 0.304 | 6.5 |

**Interpretation**

The historical results demonstrate varying trade-offs between recall and specificity across dataset groups.

In particular, high recall values should not be interpreted as evidence of clinical reliability. Lower specificity values indicate increased false-positive classifications.

**Important:** These metrics originate from an internal, non-independent test simulation. They have not been independently reproduced or clinically validated. The historical pipeline has known preprocessing leakage risks, so the results should be treated as exploratory rather than as verified estimates of generalization performance.

These scores belong to the original competition experiments and **must not be attributed to the public baseline implementation**.

---

## 📁 Repository Structure

| Location | Description |
|---|---|
| `src/pipeline.py` | Leakage-aware Logistic Regression baseline for model training and CSV prediction |
| `src/make_demo_data.py` | Synthetic dataset generator for demonstrating the public pipeline |
| `legacy/` | Original competition scripts retained for historical reference |
| `results/historical_results.json` | Recorded historical experimental results |
| `docs/METHODOLOGY.md` | Methodology, implementation differences, and known limitations |
| `tests/` | Unit tests for the public machine learning pipeline |
| `data/README.md` | Dataset format and usage instructions |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |

### Public Baseline vs. Original Competition Models

This repository distinguishes between two implementations.

**Original Competition Experiments**

The original DeepDiagnosis experiments investigated multiple machine learning algorithms, ensemble predictions, and decision threshold optimization.

These scripts are retained under `legacy/` for historical and educational reference.

**Public Reproducible Baseline**

The `src/pipeline.py` implementation provides a separate, leakage-aware Logistic Regression baseline with training and CSV prediction functionality.

It is intended to demonstrate a clearer machine learning workflow and does not reproduce the original ensemble architecture or its reported performance.

---

## 🚀 Installation & Usage

**Requirements:** Python 3.10 or later.

### 1. Clone the Repository

```bash
git clone https://github.com/fatmaSsm/DeepDiagnosis.git
cd DeepDiagnosis
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment.

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Unit Tests

```bash
python -m unittest discover -s tests -v
```

### 5. Generate Synthetic Demo Data

```bash
python -m src.make_demo_data --output-dir demo_data
```

### 6. Train Baseline Models

```bash
python -m src.pipeline train --data-dir demo_data --group ALL
```

### 7. Generate Predictions

```bash
python -m src.pipeline predict --group MASTER --input demo_data/YARISMA_TRAIN_MASTER.csv --output outputs/demo_predictions.csv
```

**Note:** Synthetic data are generated only to demonstrate the implementation. They must not be used to make scientific or clinical performance claims.

### Using Authorized Competition Data

Researchers with authorized access to the original competition datasets may place the required CSV files in the `data/` directory and execute:

```bash
python -m src.pipeline train --data-dir data --group ALL
```

Dataset formats and usage instructions are documented in `data/README.md`.

---

## 🔒 Dataset Availability

The original genomic variant datasets used in the TEKNOFEST 2026 Healthcare AI Competition are **not publicly distributed in this repository**.

These datasets were provided for competition purposes, and permission to publish the project source code does not automatically grant redistribution rights for the underlying data.

Therefore, the original training datasets have been intentionally excluded to respect data-use conditions and avoid unauthorized distribution.

To support reproducibility of the public software workflow, the repository provides a **synthetic demonstration dataset generator**.

The original trained model artifacts and competition reports are also excluded.

The repository does not currently assert an open-source license for the historical team code. Any future licensing requires agreement on the applicable rights and licensing terms.

---

## 👥 Team & Acknowledgments

DeepDiagnosis was collaboratively developed by the **DeepDiagnosis team** as part of the TEKNOFEST 2026 Healthcare Artificial Intelligence Competition.

The project reflects the collective efforts of its team members in machine learning experimentation, data analysis, model evaluation, and competition research.

The team has approved the publication of the project code on GitHub. Individual contributions remain attributable to their respective authors, and this repository does not claim sole authorship of the original competition work.

---

## 📚 Documentation

For additional technical information, refer to:

- `docs/METHODOLOGY.md` — Experimental methodology, methodological limitations, and implementation differences.
- `data/README.md` — Data requirements and usage guidelines.
- `results/historical_results.json` — Historical experiment metrics.

---

## 📬 Contact 

Fatma Susam 

[![GitHub](https://img.shields.io/badge/GitHub-fatmaSsm-181717?style=for-the-badge&logo=github)](https://github.com/fatmaSsm)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/fatma-susam/)

---

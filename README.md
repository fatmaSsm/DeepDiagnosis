# DeepDiagnosis || 🧬

**Genomic variant pathogenicity classification | TEKNOFEST 2026 – Healthcare AI**

A research/competition project exploring machine learning approaches to genomic variant pathogenicity classification across four groups: **MASTER, KANSER, CFTR and PAH**. The original experiments explored **HistGradientBoosting, RandomForest, LogisticRegression, soft-voting ensembles, stratified cross-validation and decision threshold optimization**.

> **Research and educational use only.** Not clinically validated; not intended for diagnosis, treatment, or clinical decision-making.

## Repository overview

| Location | Contents |
|---|---|
| `src/pipeline.py` | Clean leakage-aware **Logistic Regression baseline** for training and CSV prediction |
| `src/make_demo_data.py` | Synthetic demonstration dataset generator (no genomic/competition records) |
| `legacy/` | Unmodified original v2 competition scripts (historical reference; known leakage caveats) |
| `results/historical_results.json` | Historical competition-run metrics; **not independently reproduced** |
| `docs/METHODOLOGY.md` | Methodological differences and limitations |
| `tests/` | Pipeline unit tests |
| `data/README.md` | Input format and data usage guidance |

## Getting started

Python 3.10+ recommended. Run from the repository root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.make_demo_data --output-dir demo_data
python -m src.pipeline train --data-dir demo_data --group ALL
python -m src.pipeline predict --group MASTER --input demo_data/YARISMA_TRAIN_MASTER.csv --output outputs/demo_predictions.csv
```

For authorized original datasets, place the four named CSV files in `data/`, then run:

```bash
python -m src.pipeline train --data-dir data --group ALL
```

Trained model artifacts and private data stay **local and git-ignored**. The example is illustrative and generated entirely from random synthetic features.

## Historical results

The metrics JSON in `results/` preserves the team's historical run outputs for transparency. They reflect an internal, non-independent test simulation with different preprocessing/algorithm from `src/pipeline.py`; results are not external or clinical validation. The original pipeline has known preprocessing leakage risks; treat historical scores as **exploratory**, not as an independently verified benchmark. We intentionally do not claim that the publication baseline achieves those scores.

## Motivation and research considerations

Genomic variant classification involves class imbalance, incomplete annotation and uncertainty. We examine recall, specificity, precision, F1, MCC, ROC-AUC and PR-AUC; recall alone should not be treated as clinical safety. Class label 1 represents the positive class in the supplied datasets, with its clinical interpretation dependent on data definitions.

## Team and attribution

Developed collaboratively by the **DeepDiagnosis team** for TEKNOFEST 2026. **The team has approved publishing the project code on GitHub.** Individual contributions belong to their respective team members; this repository does not claim sole authorship of the competition work.

## Dataset availability and licensing

**Why are the original datasets not included?** The training CSV files used in the TEKNOFEST 2026 Healthcare AI competition were provided for the competition. Although the DeepDiagnosis team has approved publishing the project code, this does not automatically grant redistribution rights for the underlying datasets. To respect data-use conditions and avoid distributing data without verified permission, **the original competition datasets are not published in this repository**.

The repository provides a synthetic-data generator so that readers can inspect and run the public baseline without access to the competition records. **Synthetic data are for software demonstration only**, not for scientific or clinical performance claims. Anyone with separately authorized access to the original data may place their local CSVs in `data/` following [`data/README.md`](data/README.md).

The original competition-trained model files and competition reports are also not distributed. Historical aggregate evaluation results are retained with explicit methodological caveats. No open-source license is asserted for the legacy team code at this time; licensing requires a separate agreement on rights and terms.

## Further reading

See [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for known limitations and the distinction between competition code and public baseline.

---

## 📬 Contact 

Fatma Susam 

[![GitHub](https://img.shields.io/badge/GitHub-fatmaSsm-181717?style=for-the-badge&logo=github)](https://github.com/fatmaSsm)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/fatma-susam/)

---

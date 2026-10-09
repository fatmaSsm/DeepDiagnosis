# Input data (not distributed)

Place authorized local copies of the competition data here:

- `YARISMA_TRAIN_MASTER.csv`
- `YARISMA_TRAIN_KANSER.csv`
- `YARISMA_TRAIN_CFTR.csv`
- `YARISMA_TRAIN_PAH.csv`

Expected columns: `Label` (0 or 1), optional `Variant_ID`, and numeric/categorical features. The baseline automatically detects categoricals from object dtype and `CAT_` / `AA_` prefixes.

**The original competition datasets are intentionally excluded because permission to redistribute those datasets has not been verified. Team approval to publish the code does not itself authorize dataset redistribution. Do not commit original competition data, reports containing personal details, or generated private predictions.** For a no-data demonstration, use `python -m src.make_demo_data --output-dir demo_data`. Synthetic data have no biological significance.

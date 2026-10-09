# Methodology and limitations

## Original competition implementation

`legacy/` contains the source snapshot of the team's original v2 training/prediction/validation scripts for research traceability. It is **not** the recommended executable entry point. Known limitations include categorical encoding fitted using both training and holdout categories and preprocessing fitted before CV; therefore reported cross-validation estimates may be optimistic. Its reference to a benign-weighted test split is a local simulation, not external clinical validation. The original ensemble used HistGradientBoosting, RandomForest and LogisticRegression and threshold selection.

## Publication baseline

`src/pipeline.py` is a newly organized **logistic-regression baseline**, not an exact reproduction of the competition ensemble. It uses stratified 80/20 holdout; a ColumnTransformer inside a Pipeline; out-of-fold probabilities from training folds to select a recall-constrained threshold; and a single fit on the training portion before holdout evaluation. The holdout is not used for threshold selection. Its results cannot be substituted for the historical results.

## Important limitations

- No original competition datasets or trained models are distributed.
- No training/test results have been reproduced here on original data.
- Demo data are synthetic, only for testing program behavior.
- Random split could overestimate generalization if related variants, patients or batches cross splits. Group-aware external validation should be done when metadata permit.
- The learned category preprocessing is technical, not a biologically validated representation.
- Never use these predictions for medical decision-making or patient care.

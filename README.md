<div align="center">

# PriceTrack

**A rigorous, ensemble-driven mobile price-range classifier.**
Built with scikit-learn, PCA dimensionality reduction, and statistically validated model comparison.

</div>

---

PriceTrack takes raw phone specs (RAM, battery, camera, screen, etc.) and predicts a price-range class. It doesn't stop at "train a model and report accuracy" — it tunes six base learners with grid search, builds two competing ensembles (weighted soft voting and stacking), and runs paired significance tests to prove the ensemble actually beats the best single model rather than just looking better by chance.

## Features

- **Automated data pipeline** — CSV validation, stratified train/test split, feature scaling
- **PCA dimensionality reduction** — compresses 20 raw features while reporting exact variance retained
- **Six tuned base models** — Logistic Regression, KNN, Naive Bayes, SVM, Random Forest, Gradient Boosting, each optimized via `GridSearchCV`
- **Weighted Soft Voting ensemble** — per-model weights derived from cross-validation accuracy, not fixed guesses
- **Stacking ensemble** — a logistic-regression meta-learner trained on out-of-fold `predict_proba` outputs
- **Statistical significance testing** — paired t-test and Wilcoxon signed-rank test comparing the best single model against the best ensemble
- **Full visual reporting** — accuracy comparison charts, confusion-matrix heatmaps, and boxplots of the significance test, all auto-saved as PNGs
- **Bilingual logging** — structured, timestamped run logs (Persian-language messages) at every pipeline stage
- **Synthetic dataset generator** — reproducible sample data generator (`generate_dataset.py`) for local testing without a real dataset

## Reports

| Base Models | All Models vs Ensembles | Significance Test |
|---|---|---|
| _add your screenshot_ | _add your screenshot_ | _add your screenshot_ |

## Design system

PriceTrack's plots use a single brand-gradient-inspired palette (`main.py`) instead of ad hoc colors:

- **Signature palette** — a purple → blue pairing (`#8C7CFF` / `#6C5CE7` for test accuracy bars, `#3E7BFA` for CV mean/std markers), the same diagonal purple-to-blue identity used across the comparison charts
- **Confusion-matrix heatmaps** — Blues for the Weighted Voting ensemble, Greens for Stacking, so the two are visually distinguishable at a glance
- **Clean, minimal chart chrome** — no gridline clutter, annotated bar values, and a single legend per figure

All of it lives inside `plot_all_results` and `plot_significance` in `main.py`, so changing the palette is a one-file edit.

## Architecture

A layered pipeline with a strict responsibility split: `data → model → ensemble → reporting`.

```
.
├── data_loader.py           # DataLoader — CSV validation, stratified split, StandardScaler
├── model_service.py          # DimensionalityReducer (PCA) + ModelService (6 tuned base models)
├── ensemble_service.py       # WeightedVotingBuilder, StackingBuilder, EnsembleEvaluator, SignificanceTester
├── generate_dataset.py       # synthetic dataset generator for local testing
├── main.py                   # orchestrates the full pipeline + all chart generation
├── mobile_price_data.csv     # dataset
└── mobile_price_data.xlsx    # dataset (Excel copy)
```

**Why this split:** `DataLoader` owns nothing but ingestion and scaling, so it can be swapped for a different data source without touching model code. `ModelService` and `DimensionalityReducer` are pure scikit-learn wrappers with no I/O or plotting logic, which is what makes them independently testable. `ensemble_service.py` composes the tuned `best_estimators` from `ModelService` rather than retraining from scratch, so the ensembles always build on the exact same tuned hyperparameters used for the base-model comparison. `main.py` is the only file that touches `matplotlib`/`seaborn` — every other module stays plot-free and importable on its own.

## Requirements

| Tool | Version |
|---|---|
| Python | 3.10+ |
| scikit-learn | 1.3+ |
| pandas / numpy | latest |
| matplotlib / seaborn | latest |
| scipy | latest (for `ttest_rel`, `wilcoxon`) |

## Setup

```bash
git clone <this-repo>
cd pricetrack
pip install -r requirements.txt
```

No dataset generation step is required if `mobile_price_data.csv` is already present — `generate_dataset.py` is only needed to regenerate a fresh synthetic dataset.

## Run

```bash
# (optional) regenerate the synthetic dataset
python generate_dataset.py

# run the full pipeline: PCA → tuning → ensembles → significance test → charts
python main.py
```

## Output

Running `main.py` produces:

- `base_models_accuracy.png` — test vs. CV accuracy for all six tuned base models
- `all_models_accuracy.png` — base models plus both ensembles, side by side
- `ensemble_confusion_matrices.png` — Voting vs. Stacking confusion matrices
- `significance_test.png` — boxplot of CV scores for the best single model vs. the best ensemble, with t-test and Wilcoxon p-values in the title
- Full console/log output at every stage: PCA variance retained, per-model accuracy, ensemble weights, and the final significance verdict

## How the ensemble comparison actually works

1. All six base models are tuned independently via `GridSearchCV` on PCA-reduced, scaled features.
2. `WeightedVotingBuilder` assigns each model a soft-voting weight proportional to its own cross-validation accuracy — stronger models get more say.
3. `StackingBuilder` trains a logistic-regression meta-learner on the base models' out-of-fold probability outputs (`stack_method="predict_proba"`), so the meta-learner never sees leaked training predictions.
4. `EnsembleEvaluator` scores every model — base and ensemble — on both test accuracy and 5-fold CV accuracy for a fair, consistent comparison.
5. `SignificanceTester` re-runs stratified 5-fold CV for the best single model and the best ensemble on identical folds, then applies a paired t-test and a Wilcoxon signed-rank test to check whether the improvement is statistically real, not just noise.

## Troubleshooting

| Issue | Fix |
|---|---|
| `DataLoadError: فایل دیتاست ... یافت نشد` | Run `python generate_dataset.py` first, or place `mobile_price_data.csv` in the project root |
| `DataLoadError: ستون هدف ... وجود ندارد` | Confirm the CSV has a `price_range` column, or pass a different `target_column` to `DataLoader` |
| `GridSearchCV` runs very slowly | Reduce the parameter grids in `model_service.py`, or lower `n_jobs` if running on a memory-constrained machine |
| `wilcoxon` returns `None` | Happens when the paired CV score differences are all zero (identical folds) — not a bug, just an edge case with too little variance to test |
| Charts look empty or cut off | Make sure `matplotlib`'s backend can write files headlessly (`Agg`) if running on a server without a display |

For anything else, run `python main.py` directly (not through a wrapper) and read the actual traceback — the log messages are Persian, but exceptions are standard Python.

## License

MIT — do whatever you'd like, just keep the license notice.

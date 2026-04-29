# CS506 Final Report — Boston Housing Violations

> **10-minute presentation video:** _link to be added once the recording is uploaded_
>
> **Final deliverable notebook:** [`modeling_final.ipynb`](modeling_final.ipynb) — single Random Forest, two metrics (ROC-AUC and accuracy), trained on parcel-grouped splits to avoid the leakage that happened in the v1 baseline.

## Table of contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [How to Build and Run](#3-how-to-build-and-run)
4. [Data Collection](#4-data-collection)
5. [Data Cleaning](#5-data-cleaning)
6. [Feature Extraction](#6-feature-extraction)
7. [Modeling](#7-modeling)
8. [Results](#8-results)
9. [Limitations](#9-limitations)
10. [Tests and CI](#10-tests-and-ci)

---

## 1. Project Overview

We use Boston's open-data records on housing-code violations to (a) find patterns in who is repeatedly violating the code, and (b) predict whether a given property will receive a *repeat* violation.

The three goals we set at the start of the project:

- **Offending patterns** — rank owners and management companies by violation frequency.
- **Building characteristics** — see whether build year, property type, or condition correlate with violation rates.
- **Complaint mix** — categorize the kinds of complaints that come in and identify the most common.

The modeling question, added once we had the cleaned dataset: given a parcel's attributes and the violation description, predict whether that parcel will appear in the dataset more than once. This is binary classification on parcel-grouped data — the parcel is the unit of analysis, not the row.

---

## 2. Repository Structure

```
CS506-FINAL/
├── README.md                  ← this report
├── Makefile                   ← `make install / train / test / all`
├── requirements.txt
├── .github/workflows/test.yml ← CI: runs pytest on every push/PR
│
├── merged_violations.csv      ← cleaned merged dataset (16,277 rows)
├── data_processing.ipynb      ← raw → merged_violations.csv
├── visualizations.ipynb       ← 8 EDA charts answering the project goals
│
├── modeling.ipynb             ← v1: leaky baseline (kept as reference)
├── modeling_v2.ipynb          ← v2: Random Forest with group-aware split
├── modeling_v3.ipynb          ← v3: 5 models + tuning + voting ensemble
├── modeling_v4.ipynb          ← v4: TF-IDF + CatBoost + stacking
├── modeling_final.ipynb       ← FINAL deliverable
│
└── tests/test_pipeline.py     ← five sanity tests
```

Each modeling notebook is a deliberate experiment on top of the previous one. See [Modeling](#7-modeling) for the chain of reasoning.

---

## 3. How to Build and Run

```bash
make install   # install Python dependencies from requirements.txt
make all       # install + execute modeling_final.ipynb + run tests
```

Per-target detail:

| Command | Purpose |
|---|---|
| `make install` | Install dependencies. |
| `make data` | Re-run `data_processing.ipynb` to regenerate `merged_violations.csv`. Requires the raw datasets in `./data/`. The repo already ships the cleaned output, so this is only needed to regenerate from scratch. |
| `make train` | Execute `modeling_final.ipynb` in place — outputs land in the notebook itself. |
| `make test` | Run the pytest suite under `tests/`. |
| `make all` | `install` + `train` + `test`. |

**Reproducibility:** all randomness is seeded by `RANDOM_STATE = 42`. A fresh `make all` reproduces the headline numbers in [Results](#8-results) bit-for-bit.

---

## 4. Data Collection

All data comes from Boston's open-data portal, [Analyze Boston](https://data.boston.gov/) — the canonical source for the records we are studying.

| Dataset | Used for |
|---|---|
| [Building and Property Violations](https://data.boston.gov/dataset/building-and-property-violations1) | Primary dataset — every violation record. |
| [Property Assessment Data](https://data.boston.gov/dataset/property-assessment) | Owner, build year, building type, condition, parcel ID. |
| [SAM (Street Address Management)](https://data.boston.gov/dataset/live-street-address-management-sam-addresses) | Bridge between violations (`sam_id`) and parcel IDs. |

**Join strategy** (in [`data_processing.ipynb`](data_processing.ipynb)): `building_violations.sam_id → SAM.SAM_ADDRESS_ID → SAM.PARCEL → property_assessment.PID`. This chain links each violation row to its parcel and the parcel's ownership record.

---

## 5. Data Cleaning

The cleaning pipeline lives in [`data_processing.ipynb`](data_processing.ipynb) and produces `merged_violations.csv` (16,277 rows). The steps:

1. **Per-source cleaning before any joins** — drop rows missing the relevant join key, deduplicate, strip whitespace, parse dates, uppercase string fields. Cast `YR_BUILT` to numeric and clip nonsensical values.
2. **Two left joins** — Building Violations on `sam_id = SAM_ADDRESS_ID`, then on `PARCEL = PID`.
3. **Column selection** — keep the columns needed for analysis and modeling, drop bookkeeping fields.

Missing values are handled at the modeling stage rather than during cleaning: numeric columns are imputed with the column median, categoricals get an explicit `"__missing__"` token, and missing `description` becomes an empty string. This keeps the cleaned CSV faithful to the raw data and pushes imputation choices into the modeling pipeline.

---

## 6. Feature Extraction

The final model uses 14 features assembled from the merged dataset.

**Target:** `repeat_violation = 1` if the parcel appears in more than one row, else `0`. This is purely structural — no per-case information leaks into the label.

**Feature buckets:**

- **Numeric** (`POINT_X`, `POINT_Y`, `YR_BUILT`, `month`, `year`) — passthrough; Random Forest does not need scaling.
- **Low-cardinality categorical** (`MAILING_NEIGHBORHOOD`, `LU_DESC`, `BLDG_TYPE`, `OVERALL_COND`, `OWN_OCC`, `ward`) — dense one-hot encoding with rare levels (<20 occurrences) collapsed.
- **High-cardinality categorical** (`OWNER`, `code`) — fold-aware target encoding. One-hot would explode the feature space (`OWNER` has thousands of distinct values); target encoding replaces each value with a smoothed conditional mean of the target, refit per CV fold so test labels never leak.
- **Text** (`description`) — TF-IDF on unigrams and bigrams, then TruncatedSVD down to 20 dimensions. Lets the model see word patterns ("Permit", "Unsafe") rather than treating each unique description string as a separate category. This is the most informative feature in our permutation-importance analysis.

We tried `weekday`, `property_age`, KMeans-based geographic clusters, and a corporate-owner regex flag in earlier versions; all had near-zero permutation importance and were dropped to keep the model lean.

---

## 7. Modeling

The modeling effort proceeded in five iterations, each motivated by the previous one's findings.

**v1 (`modeling.ipynb`)** — decision tree with a *random row-level split*. Reported ~0.97 accuracy, but most parcels appear in many rows so a random split puts most of any parcel's records in both train and test. The model was memorizing parcels, not predicting. We keep this notebook as a cautionary baseline.

**v2 (`modeling_v2.ipynb`)** — Random Forest with a **group-aware split by parcel**. The single most important fix in the project. The honest held-out ROC-AUC dropped to 0.611 — far below v1's misleading number, but a real measure of how well the model generalizes to *new* properties.

**v3 (`modeling_v3.ipynb`)** — five models (Logistic Regression, RF, HistGradientBoosting, XGBoost, LightGBM), `HalvingRandomSearchCV` tuning, voting ensemble. Five models performed within ~0.03 of each other in CV. Aggressive tuning improved CV by 0.015 but did not move held-out test AUC. The voting ensemble added ~0.01 over the best single model.

**v4 (`modeling_v4.ipynb`)** — TF-IDF text features on `description`, CatBoost, stacking ensemble, threshold tuning. Stacking (0.620 AUC) beat lone Random Forest (0.616) by ~0.004 — within noise. Confirmed that complexity isn't paying for itself on this dataset.

**Final (`modeling_final.ipynb`)** — single Random Forest with the smallest feature set that retained signal in v4. Trained with default hyperparameters; threshold for accuracy chosen from out-of-fold training predictions (no test-set peeking) at the (1 − prior) quantile so predicted positive rate matches the class prior.

**Why Random Forest as the final choice.** Across v3 and v4 every model — Logistic Regression, RF, XGBoost, CatBoost, voting, stacking — landed in a 0.62 ± 0.02 band on held-out AUC. Model complexity is not the bottleneck. Random Forest handles mixed numeric and categorical features without scaling assumptions, captures non-linearities, and is the simplest defensible choice in that band.

**Evaluation strategy.** Group-aware 80/20 split by parcel (no parcel in both train and test), 5-fold group-aware CV for sanity-checking, and two complementary metrics: ROC-AUC (threshold-invariant ranking) and accuracy at the OOF-picked threshold. Threshold selection uses *only* training-fold predictions, so test data is never seen until final evaluation.

---

## 8. Results

On the held-out test set (1,958 parcels never seen during training):

| Metric | Value |
|---|---|
| ROC-AUC | **0.622** |
| Accuracy | **0.605** |

For context, v1's leaky model reported ~0.97 accuracy on its random row-level split — that number was memorization of parcels, not predictive skill. The 0.622 AUC on truly unseen parcels is an honest estimate of how the model would perform on new properties, and it matches the most complex stacking ensemble we built (0.620, v4) within noise.

**Where the predictive signal lives.** Permutation importance from v2 and v4 agreed on the ranking: `description` (TF-IDF on the violation type) is by far the strongest, followed by `OWNER`, `code`, `ward`, and `LU_DESC`. Numeric features and `MAILING_NEIGHBORHOOD` carry only weak marginal signal once `ward` is in the model.

**Visualization findings** (from [`visualizations.ipynb`](visualizations.ipynb)) align with the modeling result:

- Violations are geographically concentrated — Dorchester alone is 27% of the dataset, and a small number of LLC owners hold 20–43 violations each.
- Building age is a strong correlate: median build year of violating properties is 1910, and multi-family residential dominates.
- Two violation types make up nearly half the dataset: "Failure to Obtain Permit" (24.5%) and "Unsafe and Dangerous" (21.1%).

These match where the model's permutation importance points: `description`, `OWNER`, and `ward` carry the signal that the EDA suggested.

---

## 9. Limitations

- **There is a signal ceiling around AUC ≈ 0.62.** Five different modeling approaches landed in 0.60–0.62. This is not a tuning problem — it is an upper bound on what these features can predict.
- **The target depends on the calendar window.** A parcel that received its first violation late in the data has artificially few opportunities to repeat. We do not model this censoring.
- **Owner identity is brittle.** The same physical owner can appear under different string variants ("ABC LLC" vs. "ABC, LLC"). Our cleaning is whitespace + case normalization only; deeper entity resolution would likely help.
- **Random Forest probabilities skew low for unseen parcels** (target encoding falls back to the global mean for any owner or code never seen in training). We address accuracy with threshold selection rather than with calibration, since threshold selection is much simpler and the AUC is what we report as the headline number.

To do meaningfully better the project would likely need a different *target* (severity instead of binary repeat), a different *unit of analysis* (per-parcel regression), or new data sources (census, 311 calls, building permits) — not a fancier model on the same features.

---

## 10. Tests and CI

[`tests/test_pipeline.py`](tests/test_pipeline.py) contains five tests, all passing in ~3 seconds:

- target column is binary and matches the "parcel appears > once" rule
- group-aware split has zero parcel overlap between train and test (the v1 leak fix)
- preprocessor produces a finite numeric matrix of the expected shape
- Random Forest trains and `predict_proba` is in [0, 1]
- held-out AUC clears a sanity floor of 0.55 on a fixed-seed subsample

[`.github/workflows/test.yml`](.github/workflows/test.yml) runs the suite on every push and pull request via Python 3.12 on `ubuntu-latest`. Run locally with `make test`.

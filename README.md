# CS506 Final Report — Boston Housing Violations

> **10-minute presentation video:** _link to be added once the recording is uploaded_
>
> **Final deliverable notebook:** [`modeling_final.ipynb`](modeling_final.ipynb) — single Random Forest, two metrics (ROC-AUC and accuracy), trained on parcel-grouped splits to avoid the leakage that plagued the v1 baseline.

---

## Table of contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [How to Build and Run](#3-how-to-build-and-run)
4. [Data Collection](#4-data-collection)
5. [Data Cleaning](#5-data-cleaning)
6. [Feature Extraction](#6-feature-extraction)
7. [Modeling](#7-modeling)
8. [Results](#8-results)
9. [Limitations and Failure Cases](#9-limitations-and-failure-cases)
10. [Tests and CI](#10-tests-and-ci)
11. [Process Summary](#11-process-summary)

---

## 1. Project Overview

This project cleans, analyzes, models, and visualizes data on housing-code violations in the City of Boston. The aim is to identify and evaluate patterns in building management, characteristics, locality, and the nature of violations — and to predict, for a given property, whether it is likely to receive a *repeat* violation.

**Project goals:**

- **Goal 1 — Offending patterns.** Identify and rank management companies and individual owners with the highest violation frequencies.
- **Goal 2 — Building characteristics.** Determine whether build year, property type, or building condition correlate with higher violation rates.
- **Goal 3 — Complaint mix.** Categorize and quantify the specific types of building complaints to surface the most prevalent issues.

**Modeling goal (added during the project):** predict whether a parcel will receive a repeat violation given its property attributes and the violation description. This is a binary classification problem on parcel-grouped data — the parcel is the unit of analysis, not the row.

---

## 2. Repository Structure

```
CS506-FINAL/
├── README.md                  ← this report
├── Makefile                   ← `make install / data / train / test / all`
├── requirements.txt           ← Python dependencies
├── .gitignore
├── .github/workflows/test.yml ← CI: runs pytest on every push/PR
│
├── merged_violations.csv      ← cleaned merged dataset (16,277 rows)
├── data_processing.ipynb      ← raw → merged_violations.csv (cleaning + joins)
├── visualizations.ipynb       ← 8 EDA charts answering the three project goals
│
├── modeling.ipynb             ← v1: leaky decision-tree baseline (kept as historical reference)
├── modeling_v2.ipynb          ← v2: Random Forest with group-aware split (the leak fix)
├── modeling_v3.ipynb          ← v3: 5 models + halving search + voting ensemble
├── modeling_v4.ipynb          ← v4: TF-IDF + CatBoost + stacking + threshold tuning
├── modeling_final.ipynb       ← FINAL deliverable — the simple model that just works
│
└── tests/test_pipeline.py     ← five sanity tests (target, group split, preprocessor, model, AUC)
```

**Why so many modeling notebooks?** Each version is a deliberate experiment on top of the previous one. The final notebook consolidates everything we learned. See [Section 7 — Modeling](#7-modeling) for the full narrative and what each version contributed.

---

## 3. How to Build and Run

The repo ships a `Makefile` and a `requirements.txt`; everything is reproducible from a clean clone with two commands.

```bash
# 1) Install Python dependencies
make install

# 2) Train the final model end-to-end and run the test suite
make all
```

Per-target detail (run any of these on its own):

| Command | What it does |
|---|---|
| `make install` | Installs everything from `requirements.txt` into the current Python. |
| `make data` | Re-runs `data_processing.ipynb` to regenerate `merged_violations.csv`. **Requires the four raw Boston datasets** in `./data/` (`BV.csv`, `PW.csv`, `SAM.csv`, `PA.csv`). The repo already ships the cleaned output, so this target is only needed if you want to regenerate from scratch. |
| `make train` | Executes `modeling_final.ipynb` in-place — outputs and plots are saved into the notebook itself. |
| `make test` | Runs the pytest suite under `tests/`. Should complete in under 10 seconds. |
| `make all` | `install` + `train` + `test`. The full reproducibility path. |
| `make clean` | Removes generated artifacts (`predictions.csv`, `__pycache__`, Jupyter checkpoints). |

**Python version:** developed and tested on Python 3.12. Should work on 3.10+.

**Reproducibility:** all randomness goes through `RANDOM_STATE = 42`. The same `make all` on a fresh clone will reproduce the headline numbers in [Section 8 — Results](#8-results) bit-for-bit.

---

## 4. Data Collection

We use publicly available datasets from the City of Boston's open-data portal, [Analyze Boston](https://data.boston.gov/). Rather than collect data ourselves, we combine existing city records into a single dataset that links violations to specific properties, owners, and neighborhoods.

| Dataset | Used for | Source |
|---|---|---|
| **Building and Property Violations** | Primary dataset — every violation record. | [link](https://data.boston.gov/dataset/building-and-property-violations1) |
| **Public Works Violations** | Additional violation records for context. | [link](https://data.boston.gov/dataset/public-works-violations) |
| **Property Assessment Data** | Owner, build year, building type, condition, parcel ID. | [link](https://data.boston.gov/dataset/property-assessment) |
| **SAM (Street Address Management)** | Bridge that maps `sam_id` (in violations) to `PARCEL` (in property assessment). | [link](https://data.boston.gov/dataset/live-street-address-management-sam-addresses) |
| **Neighborhood Boundaries** | Group properties by neighborhood for analysis. | [link](https://data.boston.gov/dataset/bpda-neighborhood-boundaries) |

**Why these sources.** They are the canonical, official record of what we are trying to study. The City of Boston's open-data portal is the only place where violation records, property assessment data, and the SAM address bridge are published together with consistent IDs. Any third-party aggregator would either be derived from these same files or stale.

**Join strategy** (implemented in [`data_processing.ipynb`](data_processing.ipynb)):

- `building_violations.sam_id` → `SAM.SAM_ADDRESS_ID` → `SAM.PARCEL` (string match)
- `SAM.PARCEL` → `property_assessment.PID` (string match)

This chain links each violation row to its physical parcel and the parcel's ownership record.

---

## 5. Data Cleaning

All cleaning lives in [`data_processing.ipynb`](data_processing.ipynb) and produces `merged_violations.csv` (16,277 rows, 14 retained columns). The steps:

1. **Per-source cleaning** — applied independently to each input file before any joins:
   - **Building Violations:** drop rows with no `sam_id` (cannot be joined), drop exact duplicates, strip whitespace from string columns, parse all date-like columns with `pd.to_datetime(errors='coerce')`, uppercase string fields for consistent matching.
   - **SAM Addresses:** drop rows missing either join key, deduplicate on `SAM_ADDRESS_ID`, cast both join keys to stripped strings.
   - **Property Assessment:** drop rows missing `PID`, deduplicate on `PID`, strip and uppercase `OWNER`, cast `YR_BUILT` to numeric and clip nonsensical values (e.g. before 1800 or after the current year).

2. **Joins** — two left joins, in this order:
   - Building Violations ⨝ SAM on `sam_id = SAM_ADDRESS_ID`
   - Result ⨝ Property Assessment on `PARCEL = PID`

3. **Column selection** — keep only the columns needed for analysis and modeling (case ID, dates, location, violation code/description, owner, build year, neighborhood, ward, parcel coordinates, etc.). Drop bookkeeping fields.

4. **Final filter (applied per-notebook, not in the cleaned CSV):** modeling and visualizations drop rows missing `PARCEL`, since both the prediction target (`repeat_violation`) and the group-aware split require a parcel ID.

**Handling of missing / noisy / inconsistent data.**

| Issue | How we handle it |
|---|---|
| Missing join keys (`sam_id`, `SAM_ADDRESS_ID`, `PID`) | Drop the row — without the key it cannot be linked and is useless for analysis. |
| Mixed-case / whitespace in `OWNER`, `description`, etc. | Strip + uppercase, so the same owner does not appear as multiple distinct strings. |
| Bad `YR_BUILT` (e.g. 0, 9999) | Coerce to numeric, clip to a sane historical range. |
| Missing numeric features (`POINT_X`, `POINT_Y`, `YR_BUILT`) | Imputed with the column median in the modeling pipeline. |
| Missing categoricals | Replaced with the explicit token `"__missing__"` so the encoder treats them as a real category instead of silently dropping the row. |
| Missing `description` | Empty string (so TF-IDF returns a zero vector). |
| Duplicate rows | Dropped at each per-source step. |

**Resulting dataset:** 16,277 violation rows covering 9,786 unique parcels, with a 60.4% positive rate for the `repeat_violation` target.

---

## 6. Feature Extraction

The final model uses **14 features** assembled from the merged dataset, plus a derived target.

**Target.** `repeat_violation = 1` if the parcel appears in more than one row, else `0`. Computed once on the full dataset before any train/test split (the rule is purely structural — no information from individual cases leaks into it).

**Feature buckets and their encoders:**

| Group | Columns | Encoder | Why |
|---|---|---|---|
| **Numeric** | `POINT_X`, `POINT_Y`, `YR_BUILT`, `month`, `year` | passthrough | Random Forest does not need scaling. `month`/`year` are extracted from `status_dttm` to capture seasonality and long-term trend. |
| **Low-cardinality categorical** | `MAILING_NEIGHBORHOOD`, `LU_DESC`, `BLDG_TYPE`, `OVERALL_COND`, `OWN_OCC`, `ward` | dense one-hot, rare levels (<20 occurrences) collapsed | Direct one-hot is fine when there are tens of categories, not thousands. Collapsing rare levels prevents per-fold dimensionality blowup. |
| **High-cardinality categorical** | `OWNER` (~thousands of distinct values), `code` (503 distinct violation codes) | fold-aware `TargetEncoder` | One-hot would explode dimensionality. Target encoding replaces each category with a smoothed conditional mean of the target, refit on every CV fold so test labels never leak in. |
| **Text** | `description` (the violation type, e.g. "Failure to Obtain Permit") | TF-IDF (1- and 2-grams, max 500 features) → TruncatedSVD (20 dims) | Exposes actual word patterns instead of treating each unique description as a single category. Permutation importance in v2 and v4 confirms this is the single most informative feature. |

**Features we tried and dropped.** `weekday`, `property_age`, `geo_cluster`, `is_corporate_owner` — all introduced in v3/v4, all near-zero permutation importance, all removed for the final model. Less surface area, same predictive performance.

---

## 7. Modeling

The modeling effort proceeded in five deliberate iterations. Each one is a separate notebook so the reasoning chain stays auditable.

| Version | Notebook | Model(s) | Held-out ROC-AUC | Lesson |
|---|---|---|---|---|
| **v1** | `modeling.ipynb` | Decision tree, **random row-level split** | reported ~0.97 accuracy (misleading) | The same parcel appears in many rows. A random split puts most of a parcel's records in *both* train and test — the model just memorizes the parcel ID. **The reported number is leakage, not skill.** Kept in the repo as the cautionary baseline. |
| **v2** | `modeling_v2.ipynb` | Random Forest, group-aware split by `PARCEL` | **0.611** | The single most important fix. Once parcels can no longer leak between train and test, the honest performance is far more modest. `description` is the most informative feature. |
| **v3** | `modeling_v3.ipynb` | Logistic Regression, RF, HistGradientBoosting, XGBoost, LightGBM + `HalvingRandomSearchCV` + voting ensemble | best single 0.599; voting top-3 0.622 | Five models perform within ~0.03 of each other in CV. Aggressive tuning improved CV by ~0.015 but **did not move held-out test ROC-AUC**. The voting ensemble adds ~0.01 over the best single model. |
| **v4** | `modeling_v4.ipynb` | LR, RF, tuned XGBoost, CatBoost, stacking ensemble + threshold tuning | RF alone 0.616; stacking 0.620 | Stacking beats lone RF by ~0.004 — within noise. TF-IDF on `description` is now its own pipeline branch. `weekday`, `property_age`, LightGBM, and HistGradientBoosting are dropped as noise. Threshold tuning matters a great deal for accuracy when the model is not perfectly calibrated. |
| **FINAL** | `modeling_final.ipynb` | **Single Random Forest**, group-aware split, threshold picked from training-fold OOF predictions | **0.622 ROC-AUC**, **0.605 accuracy** | Consolidates every lesson above into the simplest model that performs at the same level as the most complex one we built. |

**Why we picked Random Forest as the final model.** It is appropriate for this problem because:

- The features are mixed numeric and categorical, with a fold-aware target-encoded high-cardinality column. RF handles this natively without scaling assumptions.
- The decision boundaries needed are non-linear (interactions between owner, neighborhood, and violation type matter), and RF captures non-linearities and interactions without manual feature engineering.
- Across v3 and v4, the held-out ROC-AUC of every model — Logistic Regression, RF, XGBoost, CatBoost, voting, stacking — falls in a 0.62 ± 0.02 band. **Model complexity is not the bottleneck**, the data is. RF is the simplest defensible choice in that band.

**Training procedure** (implemented in `modeling_final.ipynb`):

1. Load `merged_violations.csv`, drop rows missing `PARCEL`, derive the `repeat_violation` target.
2. **Group-aware 80/20 split by `PARCEL`** — `GroupShuffleSplit` so no parcel appears in both train and test.
3. Fit the `ColumnTransformer` preprocessor (one-hot, target encoding, TF-IDF + SVD, passthrough).
4. Train a single `RandomForestClassifier(n_estimators=300, min_samples_leaf=2)` with default hyperparameters. (No tuning: v3/v4 established that tuning does not move the held-out number.)
5. **5-fold group-aware cross-validation** on the training set as a sanity check before evaluating on held-out test.
6. **Threshold selection.** The Random Forest's predicted probabilities on unseen parcels skew low (median ≈ 0.22 even though the true positive rate is 0.60). Picking a threshold of 0.5 gives an artificially low accuracy (~0.44) for a perfectly reasonable AUC. We pick the threshold from **out-of-fold training predictions** so it does not see the test set: take the (1 − prior) quantile of the OOF probabilities, which makes the predicted positive rate match the class prior.
7. Evaluate on the held-out parcels with two metrics: ROC-AUC (threshold-invariant ranking quality) and accuracy at the chosen threshold.

**Evaluation strategy.**

- **Group-aware split + group-aware CV.** Without this, every reported number is leakage.
- **Two complementary metrics.** ROC-AUC is the natural fit for an imbalanced binary classifier and is invariant to the operating point. Accuracy is the most direct comparison to the v1 baseline and the metric a non-specialist intuitively understands.
- **Train-only threshold selection.** No test-set peeking. The threshold is a single scalar derived from out-of-fold training predictions.

---

## 8. Results

### 8.1 Headline modeling result

On the held-out test set (1,958 parcels never seen during training):

| Metric | Value |
|---|---|
| ROC-AUC | **0.622** |
| Accuracy (threshold 0.186, picked from OOF) | **0.605** |

For context, the v1 leaky model reported ~0.97 accuracy on its random row-level split. **That number was memorization of parcels seen in training, not predictive skill.** Our 0.605 on truly unseen parcels is the honest number, and our 0.622 ROC-AUC matches the most complex stacking ensemble we built (0.620, v4) within noise. See the ROC curve and full classification report inside [`modeling_final.ipynb`](modeling_final.ipynb).

### 8.2 Where the predictive signal lives

Permutation importance (from v2 and v4, agreed across both):

1. **`description`** (violation type, TF-IDF) — by far the largest drop in ROC-AUC when shuffled.
2. **`OWNER`** — target-encoded owner identity.
3. **`code`** — fine-grained violation code.
4. **`ward`** — repeat rate ranges from 46% to 69% across Boston wards.
5. **`LU_DESC`** — land-use type.

Numeric features (`POINT_X`/`POINT_Y`, `YR_BUILT`, `month`, `year`) and `MAILING_NEIGHBORHOOD` carry only weak marginal signal once `ward` is in the model.

### 8.3 Exploratory findings (the three project goals)

These are the visualization-driven findings from [`visualizations.ipynb`](visualizations.ipynb), which was our pre-modeling EDA. They answer the original goals directly.

**Goal 1 — Offending patterns.** Violations are geographically concentrated: Dorchester alone accounts for ~27% of all cases (4,526 of 17,075). A small number of private LLCs and limited partnerships are disproportionate repeat offenders — the top 15 private owners each hold 20–43 violations, and most are structured as LLCs, which complicates direct accountability. Resolution rate is uniform city-wide (~94% closed), so the bottleneck is not enforcement response but recurring non-compliance from the same actors.

**Goal 2 — Building characteristics.** Building age is the strongest predictor: the median build year of violating properties is 1910, with the distribution heavily concentrated before 1930. Multi-family residential buildings (2-family, 3-family, condos) account for the majority of violations, suggesting tenant density amplifies the impact of owner negligence. Overall building condition is *not* a reliable predictor — most violations occur in "Average" or "Good" rated buildings, meaning structural decay is not necessary for non-compliance to occur.

**Goal 3 — Complaint mix.** Two categories dominate: "Failure to Obtain Permit" (4,181 cases, 24.5%) and "Unsafe and Dangerous" (3,611 cases, 21.1%) — together nearly half of all violations. This split suggests two distinct problem populations — regulatory non-compliance (permit failures) and genuine structural safety risk — likely requiring different enforcement strategies.

### 8.4 How the visualizations support the model

| Visualization | Insight | How it shows up in the model |
|---|---|---|
| Violations by neighborhood (Dorchester ~27%) | Strong geographic concentration | `ward` and `MAILING_NEIGHBORHOOD` are useful features |
| Top 10 violation types | Two categories dominate, very long tail | TF-IDF on `description` is the most important feature |
| Top 15 private owners (LLCs) | A few owners are repeat offenders | `OWNER` (target-encoded) is the second most important feature |
| Year-built distribution | Old buildings dominate | `YR_BUILT` is a (weak) feature that nonetheless contributes |
| Geographic dot map | Clear east-Boston / Dorchester / Roxbury clustering | Coordinates contribute marginally; `ward` captures most of this signal |

---

## 9. Limitations and Failure Cases

**Inherent ceiling around ROC-AUC ~0.62.** Across five distinct modeling approaches — RF, gradient boosting, XGBoost, CatBoost, stacking — the held-out ROC-AUC clusters in 0.60–0.62. This is not a tuning problem; it is a **signal ceiling on this dataset, framing, and target**.

**The target is structurally hard.** "Has the parcel had more than one violation in the recorded period?" depends on calendar window. A parcel that received its first violation late in the data has artificially few opportunities to repeat. We do not model this censoring.

**Owner identity carries most of the high-cardinality signal, but is brittle.** The same physical owner can appear under many string variants ("ABC LLC" vs. "ABC, LLC" vs. "ABC L.L.C."). Our cleaning is whitespace + case-only; deeper entity resolution would likely help.

**Probability calibration is poor on unseen parcels.** Random Forest probabilities skew low when target encoding falls back to the global mean (which is what happens for any owner / code never seen during training). We fix accuracy via threshold selection, but a properly calibrated probability — e.g. via `CalibratedClassifierCV` — would be a better path for a real triage tool. We did not pursue this since it adds complexity for marginal accuracy gains.

**Class balance is mild (60/40), so accuracy is not a hard metric.** A "predict the majority class" baseline gets 0.604 accuracy. Our 0.605 accuracy is essentially the same number — the value of the model is in the **AUC of 0.622** (i.e., it ranks parcels correctly), not the accuracy. We report both for transparency.

**Failure cases the model genuinely struggles with:**

- **Unseen owners.** A parcel whose owner appears nowhere in the training set has its `OWNER` target-encoded value smoothed to the global mean — the model loses one of its top-2 features.
- **Rare violation codes.** Codes with fewer than 5 occurrences are effectively a single bucket after `min_df=5` on the TF-IDF and rare-level collapsing on the OHE.
- **Brand-new parcels with little property history.** With missing `YR_BUILT` and a generic `description`, the model has very little to work with and its prediction collapses toward the prior.

**What it would take to do meaningfully better.** Likely *not* a fancier model on the same features. Candidates:

- **Different target.** Predicting violation severity, predicting time-to-next-violation, or per-parcel regression (one row per parcel) would all reframe the problem more productively.
- **Richer data.** Census tract demographics, 311 service-request volume, building permits, eviction records — all linkable by parcel.
- **Better entity resolution on `OWNER`.** Even simple normalization of LLC suffixes and trust beneficiaries would consolidate sparse signal.

---

## 10. Tests and CI

The repo includes a small but real pytest suite at [`tests/test_pipeline.py`](tests/test_pipeline.py). Five tests, ~3 seconds locally:

| Test | What it guards |
|---|---|
| `test_target_is_binary_and_matches_repeat_rule` | Target column is 0/1 and consistent with the "parcel appears > once" rule. |
| `test_group_split_has_no_parcel_overlap` | The single most important fix from v1: zero `PARCEL` overlap between train and test. |
| `test_preprocessor_outputs_finite_numeric_matrix` | The `ColumnTransformer` produces a finite numeric matrix of the expected shape. |
| `test_random_forest_trains_and_outputs_valid_probabilities` | End-to-end: the model trains and `predict_proba` is in [0, 1]. |
| `test_held_out_auc_above_random_floor` | Sanity floor: held-out ROC-AUC > 0.55 on a fixed-seed subsample. Not the headline 0.622 — this guards against silent regressions, not for the report number. |

**CI.** [`.github/workflows/test.yml`](.github/workflows/test.yml) runs the suite on every push and pull request via Python 3.12 on `ubuntu-latest`. Run locally with `make test`.

---

## 11. Process Summary

The project ran from week 0 (project planning) through final submission (May 1). Key milestones:

| Phase | What happened |
|---|---|
| Weeks 1–2 | Pulled the four raw datasets from Analyze Boston; built the cleaning + joins in `data_processing.ipynb`. |
| Weeks 3–4 | Computed dataset statistics; first scheduled check-in. |
| Weeks 5–6 | EDA and the eight visualizations in `visualizations.ipynb`; first attempts at a predictive model (`modeling.ipynb`, v1). |
| Weeks 7–8 | Identified the parcel-leakage bug in v1; rebuilt with group-aware split and Random Forest in v2. Second check-in. |
| Weeks 9–10 | Modeling iterations v3 (five models + tuning + voting) and v4 (TF-IDF + CatBoost + stacking). Realized model complexity was not paying for itself. |
| Final week | Consolidated into `modeling_final.ipynb`: a single Random Forest with two metrics. Added Makefile, tests, CI, and this report. |

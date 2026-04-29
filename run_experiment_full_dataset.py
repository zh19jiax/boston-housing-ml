#!/usr/bin/env python3
"""Run A — execute the modeling_final.ipynb pipeline against the FULL BV+PW dataset.

This script mirrors `modeling_final.ipynb` exactly:
- Same USECOLS, same target rule, same date features.
- Same feature buckets (numeric / low-card cat / high-card cat / TF-IDF on description).
- Same group-aware 80/20 split by PARCEL.
- Same RandomForestClassifier(n_estimators=300, min_samples_leaf=2).
- Same threshold rule: pick threshold from out-of-fold training predictions at the
  (1 - prior) quantile so predicted positive rate matches the class prior.

The only difference from the notebook is that this is a `.py` script with print()
output instead of notebook cells, so you can run it as an SCC batch job and paste
the stdout back to me.

Usage on SCC:
    module load academic-ml/fall-2025  # or whatever module gives you Python 3.12 + sklearn
    python3 run_experiment_full_dataset.py --csv path/to/merged_violations.csv

Defaults to ./merged_violations.csv if --csv is not given.

Expected runtime on a typical SCC node (4-8 cores): ~30-60 minutes for the full
905K-row dataset. The 5-fold CV step and the OOF-prediction step each fit the
RandomForest five times.
"""
from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, TargetEncoder

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

USECOLS = [
    "case_no", "PARCEL", "POINT_X", "POINT_Y", "YR_BUILT",
    "MAILING_NEIGHBORHOOD", "LU_DESC", "BLDG_TYPE", "OVERALL_COND",
    "description", "OWNER", "OWN_OCC", "ward", "code", "status_dttm",
]
NUM_FEATURES  = ["POINT_X", "POINT_Y", "YR_BUILT", "month", "year"]
LOW_CARD_CAT  = ["MAILING_NEIGHBORHOOD", "LU_DESC", "BLDG_TYPE",
                 "OVERALL_COND", "OWN_OCC", "ward"]
HIGH_CARD_CAT = ["OWNER", "code"]
TEXT_COL      = "description"


def banner(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main(csv_path: str) -> None:
    t0 = time.time()
    banner("STAGE 1 — load and characterize the dataset")

    # If the CSV happens to include a `source` column (BV vs PW marker), we read it
    # for diagnostics but do NOT filter on it — Run A is the unfiltered pipeline.
    cols_to_try = USECOLS + ["source"]
    df = pd.read_csv(csv_path, usecols=lambda c: c in cols_to_try,
                     parse_dates=["status_dttm"], low_memory=False)
    print(f"Loaded {csv_path}")
    print(f"  raw rows : {len(df):>8,}")
    print(f"  columns  : {df.shape[1]}")
    if "source" in df.columns:
        print("  source breakdown (raw):")
        for src, n in df["source"].value_counts(dropna=False).items():
            print(f"    {src!s:<25} {n:>8,}")

    df = df.dropna(subset=["PARCEL"]).reset_index(drop=True)
    print(f"\nAfter dropna(PARCEL):")
    print(f"  rows           : {len(df):>8,}")
    print(f"  unique parcels : {df['PARCEL'].nunique():>8,}")

    df["repeat_violation"] = (
        df.groupby("PARCEL")["case_no"].transform("count") > 1
    ).astype(int)
    df["month"] = df["status_dttm"].dt.month
    df["year"]  = df["status_dttm"].dt.year

    print(f"\nClass balance (positive = parcel appears > once):")
    print(f"  positive rate  : {df['repeat_violation'].mean():.3f}")
    print(f"  rows-per-parcel describe:")
    print(df.groupby('PARCEL').size().describe().round(2).to_string())

    print(f"\nTop 10 descriptions:")
    print(df["description"].value_counts().head(10).to_string())

    # Build features.
    feature_cols = NUM_FEATURES + LOW_CARD_CAT + HIGH_CARD_CAT + [TEXT_COL]
    X = df[feature_cols].copy()
    y = df["repeat_violation"].values
    groups = df["PARCEL"]

    X[NUM_FEATURES] = X[NUM_FEATURES].fillna(X[NUM_FEATURES].median())
    for c in LOW_CARD_CAT + HIGH_CARD_CAT:
        X[c] = X[c].fillna("__missing__").astype(str)
    X[TEXT_COL] = X[TEXT_COL].fillna("").astype(str)

    print(f"\nX shape: {X.shape}")
    print(f"Time so far: {time.time() - t0:.1f}s")

    banner("STAGE 2 — group-aware train/test split")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train    = groups.iloc[train_idx]
    print(f"Train: {len(X_train):>8,} rows / {groups_train.nunique():>7,} parcels")
    print(f"Test : {len(X_test):>8,} rows / {groups.iloc[test_idx].nunique():>7,} parcels")

    # Build the same preprocessor as modeling_final.ipynb.
    text_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=500)),
        ("svd",   TruncatedSVD(n_components=20, random_state=RANDOM_STATE)),
    ])
    preprocessor = ColumnTransformer([
        ("low_card",  OneHotEncoder(handle_unknown="ignore", min_frequency=20,
                                    sparse_output=False), LOW_CARD_CAT),
        ("high_card", TargetEncoder(target_type="binary", smooth="auto",
                                    random_state=RANDOM_STATE), HIGH_CARD_CAT),
        ("text",      text_pipe, TEXT_COL),
        ("passthrough_num", "passthrough", NUM_FEATURES),
    ])

    def make_model() -> Pipeline:
        return Pipeline([
            ("prep", preprocessor),
            ("clf",  RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                            n_jobs=-1, random_state=RANDOM_STATE)),
        ])

    banner("STAGE 3 — 5-fold group-aware cross-validated AUC on training set")
    t1 = time.time()
    cv_scores = cross_val_score(
        make_model(), X_train, y_train,
        cv=GroupKFold(n_splits=5), groups=groups_train,
        scoring="roc_auc", n_jobs=1,
    )
    print(f"5-fold CV ROC-AUC: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
    print(f"  per-fold        : {[round(s, 3) for s in cv_scores]}")
    print(f"  CV stage time   : {time.time() - t1:.1f}s")

    banner("STAGE 4 — OOF threshold selection (no test peeking)")
    t1 = time.time()
    oof_proba = np.zeros(len(X_train))
    for fold, (tr, va) in enumerate(GroupKFold(n_splits=5).split(X_train, y_train, groups=groups_train)):
        fold_t = time.time()
        fold_model = make_model()
        fold_model.fit(X_train.iloc[tr], y_train[tr])
        oof_proba[va] = fold_model.predict_proba(X_train.iloc[va])[:, 1]
        print(f"  fold {fold + 1}/5 fit in {time.time() - fold_t:.1f}s")
    prior     = y_train.mean()
    threshold = float(np.quantile(oof_proba, 1 - prior))
    print(f"\n  class prior (train positive rate) : {prior:.3f}")
    print(f"  chosen classification threshold   : {threshold:.4f}")
    print(f"  OOF stage time   : {time.time() - t1:.1f}s")

    banner("STAGE 5 — fit on full training set, evaluate on held-out test")
    t1 = time.time()
    final_model = make_model()
    final_model.fit(X_train, y_train)
    y_proba = final_model.predict_proba(X_test)[:, 1]
    auc       = roc_auc_score(y_test, y_proba)
    acc_05    = accuracy_score(y_test, (y_proba >= 0.5).astype(int))
    acc_picked = accuracy_score(y_test, (y_proba >= threshold).astype(int))
    pred_pos_rate_picked = (y_proba >= threshold).mean()
    print(f"  Final fit + predict time: {time.time() - t1:.1f}s")
    print()
    print(f"  Held-out ROC-AUC                          : {auc:.3f}")
    print(f"  Held-out accuracy @ threshold 0.5         : {acc_05:.3f}")
    print(f"  Held-out accuracy @ threshold {threshold:.3f}     : {acc_picked:.3f}")
    print(f"  Predicted positive rate at picked threshold: {pred_pos_rate_picked:.3f}")
    print(f"  Test positive rate                         : {y_test.mean():.3f}")

    banner("STAGE 6 — copy-paste this summary block back")
    print(f"""DATASET                 : {csv_path}
ROWS (post dropna)      : {len(df):,}
PARCELS                 : {df['PARCEL'].nunique():,}
POSITIVE RATE           : {df['repeat_violation'].mean():.3f}
TRAIN ROWS / PARCELS    : {len(X_train):,} / {groups_train.nunique():,}
TEST ROWS / PARCELS     : {len(X_test):,} / {groups.iloc[test_idx].nunique():,}

5-fold CV ROC-AUC (train): {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}
Held-out ROC-AUC         : {auc:.3f}
Held-out accuracy @0.5   : {acc_05:.3f}
Held-out accuracy @picked: {acc_picked:.3f} (threshold = {threshold:.4f})

Comparison to current modeling_final.ipynb on BV-only (16,277 rows):
  CV AUC     0.590 +/- 0.016
  test AUC   0.622
  test acc   0.605

Total runtime: {time.time() - t0:.1f}s
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", default="merged_violations.csv",
        help="Path to the merged_violations.csv to evaluate against.",
    )
    args = parser.parse_args()
    main(args.csv)

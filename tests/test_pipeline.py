"""Sanity tests for the final modeling pipeline.

Each test rebuilds the pieces of `modeling_final.ipynb` it cares about so the test
suite is independent of the notebook execution. We keep the data load + feature
prep in module-scope fixtures so the heavier work happens once.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "merged_violations.csv"
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


@pytest.fixture(scope="module")
def df():
    """Load the cleaned merged dataset and apply the same target/date logic as the final notebook."""
    assert DATA_PATH.exists(), f"merged_violations.csv missing at {DATA_PATH}"
    df = pd.read_csv(DATA_PATH, usecols=USECOLS, parse_dates=["status_dttm"])
    df = df.dropna(subset=["PARCEL"]).reset_index(drop=True)
    df["repeat_violation"] = (
        df.groupby("PARCEL")["case_no"].transform("count") > 1
    ).astype(int)
    df["month"] = df["status_dttm"].dt.month
    df["year"]  = df["status_dttm"].dt.year
    return df


@pytest.fixture(scope="module")
def Xy(df):
    """Build X, y, groups exactly as the final notebook does."""
    feature_cols = NUM_FEATURES + LOW_CARD_CAT + HIGH_CARD_CAT + [TEXT_COL]
    X = df[feature_cols].copy()
    y = df["repeat_violation"].values
    groups = df["PARCEL"]

    X[NUM_FEATURES] = X[NUM_FEATURES].fillna(X[NUM_FEATURES].median())
    for c in LOW_CARD_CAT + HIGH_CARD_CAT:
        X[c] = X[c].fillna("__missing__").astype(str)
    X[TEXT_COL] = X[TEXT_COL].fillna("").astype(str)
    return X, y, groups


def test_target_is_binary_and_matches_repeat_rule(df):
    """`repeat_violation` must be 1 iff the parcel appears in more than one row."""
    target = df["repeat_violation"]
    assert set(target.unique()) <= {0, 1}, "target must be binary"
    # For every parcel, every row's label must equal (count > 1).
    counts = df.groupby("PARCEL")["case_no"].transform("count")
    expected = (counts > 1).astype(int)
    assert (target == expected).all(), "target does not match the repeat-violation rule"
    # Sanity: dataset is mildly imbalanced, mean should be in (0.4, 0.8).
    assert 0.4 < target.mean() < 0.8, f"unexpected class balance {target.mean():.3f}"


def test_group_split_has_no_parcel_overlap(Xy):
    """The single most important fix from v1 → v2: train and test must share zero parcels."""
    from sklearn.model_selection import GroupShuffleSplit

    X, y, groups = Xy
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    train_parcels = set(groups.iloc[train_idx])
    test_parcels  = set(groups.iloc[test_idx])
    assert train_parcels.isdisjoint(test_parcels), "parcels leaked between train and test"
    # Roughly 80/20 split of *parcels*.
    total = len(train_parcels) + len(test_parcels)
    assert 0.75 < len(train_parcels) / total < 0.85


def test_preprocessor_outputs_finite_numeric_matrix(Xy):
    """Preprocessor must produce a finite, fully-numeric matrix the model can train on."""
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, TargetEncoder

    X, y, _ = Xy

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
    Z = preprocessor.fit_transform(X, y)
    assert Z.shape[0] == len(X)
    assert np.issubdtype(np.asarray(Z).dtype, np.number)
    assert np.isfinite(np.asarray(Z)).all(), "preprocessor produced non-finite values"


def test_random_forest_trains_and_outputs_valid_probabilities(Xy):
    """End-to-end smoke test: pipeline trains and predict_proba is in [0, 1]."""
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, TargetEncoder

    X, y, groups = Xy

    # Subsample to ~25% for test speed; the model class is what we're checking, not accuracy.
    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.choice(len(X), size=len(X) // 4, replace=False)
    X_s = X.iloc[sample_idx].reset_index(drop=True)
    y_s = y[sample_idx]
    g_s = groups.iloc[sample_idx].reset_index(drop=True)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    tr, te = next(gss.split(X_s, y_s, groups=g_s))

    text_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=200)),
        ("svd",   TruncatedSVD(n_components=10, random_state=RANDOM_STATE)),
    ])
    pre = ColumnTransformer([
        ("low_card",  OneHotEncoder(handle_unknown="ignore", min_frequency=20,
                                    sparse_output=False), LOW_CARD_CAT),
        ("high_card", TargetEncoder(target_type="binary", smooth="auto",
                                    random_state=RANDOM_STATE), HIGH_CARD_CAT),
        ("text",      text_pipe, TEXT_COL),
        ("passthrough_num", "passthrough", NUM_FEATURES),
    ])
    model = Pipeline([
        ("prep", pre),
        ("clf",  RandomForestClassifier(n_estimators=50, min_samples_leaf=2,
                                        n_jobs=1, random_state=RANDOM_STATE)),
    ])
    model.fit(X_s.iloc[tr], y_s[tr])
    proba = model.predict_proba(X_s.iloc[te])[:, 1]
    assert proba.shape == (len(te),)
    assert ((proba >= 0.0) & (proba <= 1.0)).all()


def test_held_out_auc_above_random_floor(Xy):
    """On a fixed-seed subsample the model should clear a sanity floor (AUC > 0.55).

    Not a check on the headline 0.622 — that requires the full training set. This
    just guards against silent regressions where the model becomes worse than random.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, TargetEncoder

    X, y, groups = Xy

    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.choice(len(X), size=len(X) // 2, replace=False)
    X_s = X.iloc[sample_idx].reset_index(drop=True)
    y_s = y[sample_idx]
    g_s = groups.iloc[sample_idx].reset_index(drop=True)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    tr, te = next(gss.split(X_s, y_s, groups=g_s))

    text_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=300)),
        ("svd",   TruncatedSVD(n_components=15, random_state=RANDOM_STATE)),
    ])
    pre = ColumnTransformer([
        ("low_card",  OneHotEncoder(handle_unknown="ignore", min_frequency=20,
                                    sparse_output=False), LOW_CARD_CAT),
        ("high_card", TargetEncoder(target_type="binary", smooth="auto",
                                    random_state=RANDOM_STATE), HIGH_CARD_CAT),
        ("text",      text_pipe, TEXT_COL),
        ("passthrough_num", "passthrough", NUM_FEATURES),
    ])
    model = Pipeline([
        ("prep", pre),
        ("clf",  RandomForestClassifier(n_estimators=150, min_samples_leaf=2,
                                        n_jobs=1, random_state=RANDOM_STATE)),
    ])
    model.fit(X_s.iloc[tr], y_s[tr])
    proba = model.predict_proba(X_s.iloc[te])[:, 1]
    auc = roc_auc_score(y_s[te], proba)
    assert auc > 0.55, f"held-out AUC {auc:.3f} is below the sanity floor"

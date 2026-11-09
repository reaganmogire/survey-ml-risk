#!/usr/bin/env python
# coding: utf-8
"""
scripts/baseline_model_comparison.py

Trains logistic regression (and, optionally, random forest) baselines
for all six outcomes, using the IDENTICAL preprocessing pipeline,
predictor set, and train/val/test splits as the primary
HistGradientBoostingClassifier models in src/model_ai.py, so that
algorithm choice is the only varying factor. Addresses Reviewer 1
Comment 9, Reviewer 2 Comment 1, and Reviewer 3 Comment 4 (justify
gradient boosting vs. simpler/alternative models).

IMPORTANT -- DUPLICATED LOGIC WARNING: as in
scripts/restricted_years_sensitivity.py, the predictor classification
and preprocessing pipeline below are DUPLICATED from src/model_ai.py,
because model_ai.py runs its training logic at module level and cannot
be safely imported without re-triggering a full retrain. If the
predictor classification in model_ai.py changes, update this script to
match.

Run (from repo root):
  python scripts/baseline_model_comparison.py
  python scripts/baseline_model_comparison.py --include-random-forest

Environment variables (optional):
  DATA_DERIVED_DIR, TABLES_DIR, ARTIFACT_DIR (ARTIFACT_DIR points to
  the FINAL/current HistGradientBoostingClassifier models, used as the
  comparison target -- not overwritten by this script)
"""

from __future__ import annotations

import os
import argparse
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)


# ============================================================
# Path configuration
# ============================================================
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1] if THIS_FILE.parent.name in {"src", "scripts"} else THIS_FILE.parent

DATA_DERIVED_DIR = Path(os.environ.get("DATA_DERIVED_DIR", REPO_ROOT / "data" / "derived"))
TABLES_DIR       = Path(os.environ.get("TABLES_DIR",       REPO_ROOT / "tables"))
ARTIFACT_DIR     = Path(os.environ.get("ARTIFACT_DIR",     REPO_ROOT / "app" / "artifacts"))

BASELINE_TABLES_DIR = TABLES_DIR / "baseline_comparison"
BASELINE_TABLES_DIR.mkdir(parents=True, exist_ok=True)

COMBINED_CSV = Path(os.environ.get(
    "BRFSS_CLEAN_CSV",
    DATA_DERIVED_DIR / "BRFSS_2011_2015_clean_model.csv"
))

RANDOM_STATE = 42

DISEASE_NAME_MAP = {
    "heart_attack": "Heart attack",
    "coronary_hd":  "Coronary heart disease",
    "stroke":       "Stroke",
    "kidney":       "Chronic kidney disease",
    "depression":   "Depression",
    "diabetes":     "Diabetes",
}


# ============================================================
# DUPLICATED from src/model_ai.py -- see module docstring warning.
# ============================================================
predictor_cols = [
    "_STATE", "SEX", "_AGEG5YR", "_EDUCAG", "_INCOMG", "_MRACE1", "_HISPANC",
    "SMOKE100", "SMOKDAY2", "ALCDAY5", "DRNKANY5", "EXERANY2",
    "FRUIT1", "VEGETAB1",
    "HLTHPLN1", "PERSDOC2", "MEDCOST", "CHECKUP1",
    "BPHIGH4", "BPMEDS", "TOLDHI2", "CHOLCHK", "ASTHMA3", "HAVARTH3",
    "GENHLTH", "PHYSHLTH", "MENTHLTH", "POORHLTH",
    "DIFFWALK", "DECIDE",
    "WEIGHT2", "HEIGHT3", "_BMI5"
]

target_cols_binary = [
    "heart_attack", "coronary_hd", "stroke",
    "kidney", "depression", "diabetes"
]

NOMINAL_CATEGORICAL_COLS = [
    "_STATE", "SEX", "_MRACE1", "_HISPANC", "SMOKE100", "SMOKDAY2",
    "DRNKANY5", "EXERANY2", "HLTHPLN1", "PERSDOC2", "MEDCOST",
    "BPHIGH4", "BPMEDS", "TOLDHI2", "ASTHMA3", "HAVARTH3",
    "DIFFWALK", "DECIDE",
]

ORDINAL_KEPT_NUMERIC_COLS = [
    "_AGEG5YR", "_EDUCAG", "_INCOMG", "GENHLTH", "CHECKUP1", "CHOLCHK",
]

CONTINUOUS_COLS = [
    "PHYSHLTH", "MENTHLTH", "POORHLTH",
    "WEIGHT2", "HEIGHT3", "_BMI5",
    "FRUIT1", "VEGETAB1", "ALCDAY5",
]

numeric_features = ORDINAL_KEPT_NUMERIC_COLS + CONTINUOUS_COLS
categorical_features = NOMINAL_CATEGORICAL_COLS


def build_preprocessor() -> ColumnTransformer:
    """Identical to model_ai.py's preprocessor -- see duplication warning above."""
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    # Dense output, identical to model_ai.py (see the note there): keeps the three algorithms on exactly the same inputs.
    ], sparse_threshold=0.0)


def get_full_model_splits(df: pd.DataFrame, disease: str):
    """
    Reproduces model_ai.py's EXACT train/val/test split for a given
    disease (same predictor_cols, same target, same random_state,
    same test_size/stratify arguments), so the baseline models are
    evaluated on the identical held-out test set as the primary
    HistGradientBoostingClassifier models.
    """
    X = df[predictor_cols].copy()
    y = df[target_cols_binary].copy()

    mask = y[disease].notna()
    X_i = X.loc[mask].copy()
    y_i = y.loc[mask, disease].astype(int).copy()

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_i, y_i, test_size=0.2, random_state=RANDOM_STATE, stratify=y_i
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train_full
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_baseline(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    preprocessor = build_preprocessor()

    if model_name == "logistic_regression":
        clf_step = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            random_state=RANDOM_STATE,
        )
    elif model_name == "random_forest":
        clf_step = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("clf", clf_step),
    ])
    pipe.fit(X_train, y_train)

    y_proba = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    brier = brier_score_loss(y_test, y_proba)

    return {"model": pipe, "y_proba": y_proba, "y_test": y_test,
            "auc": auc, "average_precision": ap, "brier": brier}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-random-forest", action="store_true",
                         help="Also train a random forest baseline (slower).")
    args = parser.parse_args()

    print(f"Loading dataset: {COMBINED_CSV}")
    df = pd.read_csv(COMBINED_CSV)
    print(f"Shape: {df.shape}")

    full_models_path = ARTIFACT_DIR / "disease_models.joblib"
    full_models = joblib.load(full_models_path)

    baselines_to_run = ["logistic_regression"]
    if args.include_random_forest:
        baselines_to_run.append("random_forest")

    rows = []

    for disease in target_cols_binary:
        print(f"\n=== {DISEASE_NAME_MAP.get(disease, disease)} ===")
        X_train, X_val, X_test, y_train, y_val, y_test = get_full_model_splits(df, disease)
        print(f"  n_train={len(X_train):,}  n_test={len(X_test):,}")

        # Reference: primary HistGradientBoostingClassifier model,
        # evaluated on the SAME test set (recomputed here to confirm
        # test-set identity, rather than trusting the stored y_test
        # blindly -- this also lets us assert the split matches).
        full_info = full_models[disease]
        y_test_full_stored = np.asarray(full_info["y_test"])
        if len(y_test_full_stored) != len(y_test) or not np.array_equal(
            np.asarray(y_test_full_stored), np.asarray(y_test)
        ):
            print("  WARNING: test set from full model does not match the split "
                  "reproduced here. Baseline comparison may not be on an "
                  "identical held-out set -- verify random_state/predictor_cols "
                  "consistency with model_ai.py.")
        gbm_auc = roc_auc_score(full_info["y_test"], full_info["y_proba"])
        gbm_ap = average_precision_score(full_info["y_test"], full_info["y_proba"])
        gbm_brier = brier_score_loss(full_info["y_test"], full_info["y_proba"])

        row = {
            "disease": disease,
            "disease_pretty": DISEASE_NAME_MAP.get(disease, disease),
            "n_test": len(y_test),
            "AUROC_gbm": round(gbm_auc, 4),
            "AUPRC_gbm": round(gbm_ap, 4),
            "Brier_gbm": round(gbm_brier, 4),
        }

        for model_name in baselines_to_run:
            print(f"  Training {model_name}...")
            result = train_baseline(model_name, X_train, y_train, X_test, y_test)
            short = "lr" if model_name == "logistic_regression" else "rf"
            row[f"AUROC_{short}"] = round(result["auc"], 4)
            row[f"AUPRC_{short}"] = round(result["average_precision"], 4)
            row[f"Brier_{short}"] = round(result["brier"], 4)
            row[f"AUROC_gbm_minus_{short}"] = round(gbm_auc - result["auc"], 4)
            print(f"    {model_name}: AUROC={result['auc']:.4f}  "
                  f"AUPRC={result['average_precision']:.4f}  "
                  f"Brier={result['brier']:.4f}")

        rows.append(row)

    comparison_df = pd.DataFrame(rows)
    print("\n=== Full comparison: GBM (primary) vs. baseline(s) ===")
    print(comparison_df.to_string(index=False))

    out_path = BASELINE_TABLES_DIR / "baseline_model_comparison.csv"
    comparison_df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")

    # Summary sentence data: min/max AUROC gap, for direct use in
    # Methods/Results/response-letter text.
    if "AUROC_gbm_minus_lr" in comparison_df.columns:
        gaps = comparison_df["AUROC_gbm_minus_lr"]
        print(f"\nGBM vs. logistic regression AUROC advantage: "
              f"min={gaps.min():.4f}, max={gaps.max():.4f}, "
              f"mean={gaps.mean():.4f} across 6 outcomes.")
        largest_gap_disease = comparison_df.loc[gaps.idxmax(), "disease_pretty"]
        print(f"Largest advantage for: {largest_gap_disease}")


if __name__ == "__main__":
    main()

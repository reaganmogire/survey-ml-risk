#!/usr/bin/env python
# coding: utf-8
"""
model_ai.py

Publication pipeline: Train and evaluate survey-based machine-learning models for
chronic disease case-finding (detection of prevalent disease) using harmonized BRFSS data (2011-2015).

REVISION UPDATE: corrects the numeric/categorical predictor split.

Original issue: the previous version used pd.api.types.is_numeric_dtype()
to infer which predictors should be one-hot encoded vs. median-imputed
and standardized. Because every BRFSS predictor is stored as a numeric
dtype in the raw CSVs, ALL 33 predictors -- including nominal codes
like SEX, _MRACE1, _HISPANC, and _STATE -- fell into the numeric
branch. The categorical branch was empty (see revision diagnostic,
scripts/revision_analyses.py Phase 0.2).

Fix applied here (targeted fix): genuinely NOMINAL predictors (no
meaningful order -- sex, race, state FIPS code, binary yes/no history
flags) are now explicitly routed through one-hot encoding. Genuinely
ORDINAL predictors (age group, education, income, general health,
time-since-checkup/cholesterol-check) are deliberately left as numeric,
since an ordinal numeric encoding is a defensible, standard choice for
ordered categories. See NOMINAL_CATEGORICAL_COLS / ORDINAL_KEPT_NUMERIC_COLS
below for the explicit classification and rationale per variable.

Note: this script now also expects the cleaned dataset produced by the
updated data_preparation.py, which additionally fixes the compound
frequency encoding for FRUIT1/VEGETAB1/ALCDAY5 and the "88=None" issue
for PHYSHLTH/MENTHLTH/POORHLTH -- both applied upstream, in place,
under the original column names. No column-name changes are needed
here as a result.

This script:
  1) Loads the cleaned BRFSS modeling dataset (produced by data_preparation.py)
  2) Trains one model per outcome using a consistent preprocessing + GBDT pipeline
  3) Evaluates discrimination and calibration metrics
  4) Selects a per-outcome probability threshold maximizing F1 score on a
     held-out validation set (separate from the test set)
  5) Exports artifacts for inference (Streamlit app) and manuscript-ready tables/figures
  6) Compares performance against the frozen pre-revision model artifacts,
     for direct before/after reporting in the manuscript revision

Portability notes (GitHub-friendly):
  - No absolute paths are used.
  - Default paths assume this repo structure:
      data/derived/BRFSS_2011_2015_clean_model.csv
      app/artifacts/
      figures/
      tables/
  - You can override paths via environment variables if needed.
"""

# ============================================================
# 0. Imports
# ============================================================
from __future__ import annotations

import os
import glob
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.calibration import calibration_curve

import joblib
import shap

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)


# ============================================================
# 0b. Path configuration (REPO-RELATIVE; PORTABLE) -- unchanged
# ============================================================
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1] if THIS_FILE.parent.name in {"src", "scripts"} else THIS_FILE.parent

DATA_DERIVED_DIR = Path(os.environ.get("DATA_DERIVED_DIR", REPO_ROOT / "data" / "derived"))
FIGURES_DIR      = Path(os.environ.get("FIGURES_DIR",      REPO_ROOT / "figures"))
TABLES_DIR       = Path(os.environ.get("TABLES_DIR",       REPO_ROOT / "tables"))
ARTIFACT_DIR     = Path(os.environ.get("ARTIFACT_DIR",     REPO_ROOT / "app" / "artifacts"))

# Where the PRE-REVISION artifacts were frozen before this script last
# overwrote app/artifacts/ (see scripts/revision_analyses.py, Phase 0.1).
# Used only for the before/after comparison at the end of this script.
FROZEN_DIR = Path(os.environ.get("FROZEN_V1_DIR", REPO_ROOT / "revision" / "v1_original"))

for d in [DATA_DERIVED_DIR, FIGURES_DIR, TABLES_DIR, ARTIFACT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

COMBINED_CSV = Path(os.environ.get(
    "BRFSS_CLEAN_CSV",
    DATA_DERIVED_DIR / "BRFSS_2011_2015_clean_model.csv"
))


# ============================================================
# 1. Load cleaned, combined BRFSS data (2011-2015)
# ============================================================
if not COMBINED_CSV.exists():
    raise FileNotFoundError(
        f"Cleaned dataset not found:\n  {COMBINED_CSV}\n\n"
        "Expected output of data_preparation.py. Either:\n"
        "  (a) run data_preparation.py to generate it, or\n"
        "  (b) set BRFSS_CLEAN_CSV to the correct path."
    )

df = pd.read_csv(COMBINED_CSV)
print(f"Loaded cleaned dataset: {COMBINED_CSV}")
print("Shape:", df.shape)


# ============================================================
# 2. Define predictors and targets -- unchanged list, same column
#    names (data_preparation.py fixes are applied upstream, in place)
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

target_continuous = ["BMI"]

binary_mask = df[target_cols_binary].notnull().any(axis=1)
df_model = df.loc[binary_mask, predictor_cols + target_cols_binary + target_continuous + ["YEAR"]].copy()

print("Modeling dataset shape:", df_model.shape)


# ============================================================
# 3. Split into features (X) and outcomes (y)
# ============================================================
X = df_model[predictor_cols].copy()
y = df_model[target_cols_binary].copy()

print("\nOutcome prevalence proxy (mean among observed entries):")
print(y.mean(numeric_only=True))


# ============================================================
# 4. CORRECTED numeric vs categorical split
#    ------------------------------------------------------
#    Previously: pd.api.types.is_numeric_dtype(X[col]) was used to
#    infer the split, which put ALL 33 predictors into the numeric
#    branch (see revision/tables/phase0_2_pipeline_diagnostic.txt).
#
#    Fix: explicitly declare which predictors are genuinely NOMINAL
#    and route ONLY those through one-hot encoding. Ordinal predictors
#    are deliberately left numeric.
# ============================================================
NOMINAL_CATEGORICAL_COLS = [
    "_STATE",     # FIPS state code -- no order
    "SEX",        # 1=Male, 2=Female -- no order
    "_MRACE1",    # Race category -- no order
    "_HISPANC",   # Hispanic ethnicity -- binary, no order
    "SMOKE100",   # Ever smoked 100 cigarettes -- yes/no
    "SMOKDAY2",   # Current smoking frequency -- categorical response set
    "DRNKANY5",   # Any alcohol use -- yes/no
    "EXERANY2",   # Any exercise in past month -- yes/no
    "HLTHPLN1",   # Has health insurance -- yes/no
    "PERSDOC2",   # Has personal doctor -- categorical (yes-one/yes-more/no)
    "MEDCOST",    # Could not see doctor due to cost -- yes/no
    "BPHIGH4",    # Told high blood pressure -- categorical
    "BPMEDS",     # On blood pressure medication -- yes/no
    "TOLDHI2",    # Told high cholesterol -- yes/no
    "ASTHMA3",    # Asthma -- yes/no
    "HAVARTH3",   # Arthritis -- yes/no
    "DIFFWALK",   # Difficulty walking/climbing -- yes/no
    "DECIDE",     # Cognitive difficulty -- yes/no
]

# Ordinal predictors -- deliberately kept numeric.
ORDINAL_KEPT_NUMERIC_COLS = [
    "_AGEG5YR",  # Age group -- ordered
    "_EDUCAG",   # Education category -- ordered
    "_INCOMG",   # Income category -- ordered
    "GENHLTH",   # Self-rated general health (1=excellent...5=poor) -- ordered
    "CHECKUP1",  # Time since last routine checkup -- ordered
    "CHOLCHK",   # Time since last cholesterol check -- ordered
]

# Continuous predictors. FRUIT1, VEGETAB1, ALCDAY5, PHYSHLTH, MENTHLTH,
# POORHLTH are genuinely continuous NOW that data_preparation.py has
# corrected their encoding upstream (compound frequency decode and
# "88=None" recode, respectively, applied in place under these same
# column names).
CONTINUOUS_COLS = [
    "PHYSHLTH", "MENTHLTH", "POORHLTH",     # day counts, 0-30 (fixed upstream)
    "WEIGHT2", "HEIGHT3", "_BMI5",          # anthropometric measures
    "FRUIT1", "VEGETAB1", "ALCDAY5",        # frequency rates (fixed upstream)
]

numeric_features = ORDINAL_KEPT_NUMERIC_COLS + CONTINUOUS_COLS
categorical_features = NOMINAL_CATEGORICAL_COLS

# Sanity check: every predictor must be classified exactly once.
_all_classified = set(numeric_features) | set(categorical_features)
_missing = set(predictor_cols) - _all_classified
_unexpected = _all_classified - set(predictor_cols)
if _missing:
    raise ValueError(f"Predictors not classified into numeric or categorical: {_missing}")
if _unexpected:
    raise ValueError(f"Classified columns not present in predictor_cols: {_unexpected}")

print(f"\nCorrected encoding split:")
print(f"  Categorical (one-hot encoded), n={len(categorical_features)}: {categorical_features}")
print(f"  Numeric (median-imputed + standardized), n={len(numeric_features)}: {numeric_features}")


# ============================================================
# 5. Preprocessing pipeline
# ============================================================
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    # HistGradientBoostingClassifier needs DENSE input. By default ColumnTransformer returns a
    # sparse matrix when the one-hot output is sparse enough (density < 0.3), and this model sits
    # right at that threshold (about 33 non-zero values per row out of about 109 columns), so the
    # output flipped to sparse and training failed. Force dense output so it can never flip.
    sparse_threshold=0.0,
)


# (The former "uncertainty helper" u = 1 - 2|p - 0.5| was removed: it is a simple
#  transformation of distance from 0.5, not a valid measure of predictive uncertainty.
#  Uncertainty is conveyed through bootstrap confidence intervals instead.)


# ============================================================
# 7. Train one model per disease
# ============================================================
disease_models = {}

for col in target_cols_binary:
    print("\n==============================")
    print(f"Training model for: {col}")
    print("==============================")

    mask = y[col].notna()
    X_i  = X.loc[mask].copy()
    y_i  = y.loc[mask, col].astype(int).copy()
    print(f"  Samples for {col}: {X_i.shape[0]}")

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_i, y_i, test_size=0.2, random_state=42, stratify=y_i
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )

    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("clf", HistGradientBoostingClassifier(
            learning_rate=0.1,
            max_iter=300,
            max_depth=None,
            l2_regularization=0.0,
            random_state=42
        ))
    ])

    clf.fit(X_train, y_train)

    y_proba_val = clf.predict_proba(X_val)[:, 1]
    thresholds  = np.linspace(0.01, 0.99, 99)
    f1_scores   = []
    for t in thresholds:
        y_pred_t = (y_proba_val >= t).astype(int)
        f1_scores.append(f1_score(y_val, y_pred_t) if y_pred_t.sum() > 0 else 0.0)

    best_idx    = int(np.argmax(f1_scores))
    best_thresh = float(thresholds[best_idx])
    best_f1_val = float(f1_scores[best_idx])

    print(f"  Best threshold (validation set) for {col}: {best_thresh:.3f}  "
          f"(validation F1 = {best_f1_val:.3f})")

    y_proba = clf.predict_proba(X_test)[:, 1]
    auc   = roc_auc_score(y_test, y_proba)
    ap    = average_precision_score(y_test, y_proba)
    brier = brier_score_loss(y_test, y_proba)

    print(f"  AUROC : {auc:.3f}")
    print(f"  AUPRC : {ap:.3f}")
    print(f"  Brier : {brier:.3f}")

    y_pred_05   = (y_proba >= 0.5).astype(int)
    y_pred_best = (y_proba >= best_thresh).astype(int)
    print("\nClassification report (threshold = 0.5):")
    print(classification_report(y_test, y_pred_05, digits=3))
    print(f"\nClassification report (optimal threshold = {best_thresh:.3f}):")
    print(classification_report(y_test, y_pred_best, digits=3))

    disease_models[col] = {
        "model":          clf,
        "X_test":         X_test,
        "y_test":         y_test,
        "y_proba":        y_proba,
        "best_threshold": best_thresh,
        "auc":            auc,
        "average_precision": ap,
        "brier":          brier,
        "f1_val":         best_f1_val,
    }


# ============================================================
# 8. Evaluation helper
# ============================================================
def evaluate_models(disease_models, use_optimal_threshold=True, fixed_threshold=0.5):
    rows = []
    for disease, info in disease_models.items():
        y_test  = info["y_test"]
        y_proba = info["y_proba"]
        t      = info["best_threshold"] if (use_optimal_threshold and "best_threshold" in info) else fixed_threshold
        y_pred = (y_proba >= t).astype(int)

        auc        = roc_auc_score(y_test, y_proba)
        ap         = average_precision_score(y_test, y_proba)
        brier      = brier_score_loss(y_test, y_proba)
        prevalence = y_test.mean()

        cm   = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        ppv  = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        npv  = tn / (tn + fn) if (tn + fn) > 0 else np.nan

        rows.append({
            "disease": disease, "threshold": t, "prevalence": prevalence,
            "AUROC": auc, "AUPRC": ap, "Brier": brier,
            "sensitivity": sens, "specificity": spec, "PPV": ppv, "NPV": npv,
        })
    return pd.DataFrame(rows).set_index("disease")


# ============================================================
# 9. Compare fixed vs optimised thresholds
# ============================================================
eval_05  = evaluate_models(disease_models, use_optimal_threshold=False, fixed_threshold=0.5)
eval_opt = evaluate_models(disease_models, use_optimal_threshold=True)

print("\n=== Evaluation @ threshold 0.5 ===")
print(eval_05)
print("\n=== Evaluation @ per-disease optimal thresholds ===")
print(eval_opt)

eval_05.to_csv(TABLES_DIR  / "model_evaluation_threshold_0p5.csv")
eval_opt.to_csv(TABLES_DIR / "model_evaluation_optimal_thresholds.csv")


# ============================================================
# 9b. Compare against FROZEN pre-revision artifacts (if available)
#     -- NOT against app/artifacts/, since that now holds the new
#     (fixed) results. Reads from revision/v1_original/<timestamp>/,
#     the snapshot saved before this script was last rerun.
# ============================================================
def find_latest_frozen_snapshot(frozen_dir: Path) -> Optional[Path]:
    if not frozen_dir.exists():
        return None
    candidates = sorted(
        [p for p in frozen_dir.iterdir() if p.is_dir()],
        key=lambda p: p.name,
        reverse=True,
    )
    for c in candidates:
        if (c / "disease_models.joblib").exists():
            return c
    return None


def compare_against_frozen(eval_opt_new: pd.DataFrame, frozen_dir: Path = FROZEN_DIR) -> Optional[pd.DataFrame]:
    snapshot_dir = find_latest_frozen_snapshot(frozen_dir)
    if snapshot_dir is None:
        print(f"\n  No frozen pre-revision snapshot found under {frozen_dir}; "
              "skipping before/after comparison.")
        return None

    print(f"\n  Comparing against frozen pre-revision snapshot: {snapshot_dir}")
    frozen_models = joblib.load(snapshot_dir / "disease_models.joblib")

    rows = []
    for disease in eval_opt_new.index:
        if disease not in frozen_models:
            continue
        old_info = frozen_models[disease]
        y_test_old  = np.asarray(old_info["y_test"],  dtype=int)
        y_proba_old = np.asarray(old_info["y_proba"], dtype=float)
        auc_old   = roc_auc_score(y_test_old, y_proba_old)
        ap_old    = average_precision_score(y_test_old, y_proba_old)
        brier_old = brier_score_loss(y_test_old, y_proba_old)

        auc_new   = eval_opt_new.loc[disease, "AUROC"]
        ap_new    = eval_opt_new.loc[disease, "AUPRC"]
        brier_new = eval_opt_new.loc[disease, "Brier"]

        rows.append({
            "disease": disease,
            "AUROC_before": round(auc_old, 4), "AUROC_after": round(auc_new, 4),
            "AUROC_delta": round(auc_new - auc_old, 4),
            "AUPRC_before": round(ap_old, 4), "AUPRC_after": round(ap_new, 4),
            "AUPRC_delta": round(ap_new - ap_old, 4),
            "Brier_before": round(brier_old, 4), "Brier_after": round(brier_new, 4),
            "Brier_delta": round(brier_new - brier_old, 4),
        })

    comp_df = pd.DataFrame(rows).set_index("disease")
    print("\n=== Before (pre-revision, frozen) vs After (post-encoding-fix) ===")
    print(comp_df.to_string())

    out_path = TABLES_DIR / "encoding_fix_before_after_comparison.csv"
    comp_df.to_csv(out_path)
    print(f"\n  Saved -> {out_path}")
    print("  Use this table for the response to the review and")
    print("  the review (model specification, review comment).")
    return comp_df


compare_against_frozen(eval_opt)


# ============================================================
# 10. Predicted probability and case-finding classification for a single user
# ============================================================
CASE_FINDING_POSITIVE = "Above threshold (likely existing condition)"
CASE_FINDING_NEGATIVE = "Below threshold"


def predict_user(input_dict, models=disease_models) -> pd.DataFrame:
    row = pd.DataFrame([input_dict]).reindex(columns=predictor_cols, fill_value=np.nan)
    results = []
    for disease, info in models.items():
        model = info["model"]
        p     = float(model.predict_proba(row)[0, 1])
        t     = float(info.get("best_threshold", 0.5))
        label = CASE_FINDING_POSITIVE if p >= t else CASE_FINDING_NEGATIVE
        results.append({
            "disease": disease, "probability": p, "threshold_used": t,
            "classification": label,
        })
    return pd.DataFrame(results).set_index("disease")


# ============================================================
# 11. Explain a single prediction using SHAP
# ============================================================
def explain_prediction(input_dict, disease, models=disease_models, top_n=10):
    if disease not in models:
        raise ValueError(f"Disease '{disease}' not found in disease_models.")

    info  = models[disease]
    model = info["model"]

    row = pd.DataFrame([input_dict]).reindex(columns=predictor_cols, fill_value=np.nan)

    p     = float(model.predict_proba(row)[0, 1])
    t     = float(info.get("best_threshold", 0.5))
    label = CASE_FINDING_POSITIVE if p >= t else CASE_FINDING_NEGATIVE

    preprocessor_ = model.named_steps["preprocessor"]
    clf_          = model.named_steps["clf"]

    X_t = preprocessor_.transform(row)

    explainer   = shap.TreeExplainer(clf_)
    shap_values = explainer.shap_values(X_t)

    if isinstance(shap_values, list):
        shap_for_pos = shap_values[1][0]
    else:
        shap_for_pos = shap_values[0]

    feature_names = preprocessor_.get_feature_names_out()

    contrib_df = (
        pd.DataFrame({"feature": feature_names, "shap_value": shap_for_pos})
        .assign(abs_shap=lambda d: d["shap_value"].abs())
        .sort_values("abs_shap", ascending=False)
        .head(top_n)
        .drop(columns=["abs_shap"])
    )

    return p, label, contrib_df


# ============================================================
# 12. Example users: predict + explain
#     Note: FRUIT1, VEGETAB1 and ALCDAY5 are given here as the DECODED
#     "times per day" rates that the trained pipeline expects (the compound
#     day/week/month decode runs once, in data_preparation.py, and is NOT
#     part of the saved pipeline). The previous version passed the raw
#     BRFSS codes (e.g. FRUIT1=205, ALCDAY5=201), which the pipeline read
#     as 205 servings / 201 drinks per day. Decode used:
#       FRUIT1 / VEGETAB1 :  1xx -> xx per day ; 2xx -> (xx)/7 per day
#                            (205 -> 0.71 ; 201 -> 0.14 ; 101 -> 1.00)
#       ALCDAY5           :  1xx -> xx/7 per day ; 2xx -> xx/30 per day
#                            (101 -> 0.1429 ; 103 -> 0.4286 ; 201 -> 0.0333)
#     If the deployed app collects raw BRFSS-style codes, app.py must apply
#     the same decode before calling predict_user() / explain_prediction().
#     AGE NOTE: _AGEG5YR is the 5-year band (1 = 18-24, 6 = 45-49, 9 = 60-64,
#     13 = 80+). Check that each example's label matches its band.
# ============================================================
example_users = {
    "Example 1: Older female, multiple CVD risk factors": {
        "_STATE": 13, "SEX": 2, "_AGEG5YR": 13, "_EDUCAG": 2, "_INCOMG": 2,
        "_MRACE1": 2, "_HISPANC": 2, "SMOKE100": 1, "SMOKDAY2": 2,
        "ALCDAY5": 0.0333, "DRNKANY5": 1, "EXERANY2": 2, "FRUIT1": 0.71, "VEGETAB1": 0.71,
        "HLTHPLN1": 1, "PERSDOC2": 1, "MEDCOST": 2, "CHECKUP1": 1,
        "BPHIGH4": 1, "BPMEDS": 1, "TOLDHI2": 1, "CHOLCHK": 1, "ASTHMA3": 2, "HAVARTH3": 1,
        "GENHLTH": 3, "PHYSHLTH": 10, "MENTHLTH": 5, "POORHLTH": 7,
        "DIFFWALK": 1, "DECIDE": 2, "WEIGHT2": 2100, "HEIGHT3": 504, "_BMI5": 3600,
    },
    "Example 2: Middle-aged, relatively low risk": {
        "_STATE": 24, "SEX": 1, "_AGEG5YR": 9, "_EDUCAG": 3, "_INCOMG": 4,
        "_MRACE1": 1, "_HISPANC": 2, "SMOKE100": 2, "SMOKDAY2": 3,
        "ALCDAY5": 0.1429, "DRNKANY5": 2, "EXERANY2": 1, "FRUIT1": 1.0, "VEGETAB1": 1.0,
        "HLTHPLN1": 1, "PERSDOC2": 1, "MEDCOST": 2, "CHECKUP1": 2,
        "BPHIGH4": 2, "BPMEDS": 2, "TOLDHI2": 2, "CHOLCHK": 1, "ASTHMA3": 2, "HAVARTH3": 2,
        "GENHLTH": 2, "PHYSHLTH": 0, "MENTHLTH": 0, "POORHLTH": 0,
        "DIFFWALK": 2, "DECIDE": 2, "WEIGHT2": 1700, "HEIGHT3": 507, "_BMI5": 2650,
    },
    "Example 3: Younger adult with depression risk": {
        "_STATE": 6, "SEX": 2, "_AGEG5YR": 6, "_EDUCAG": 4, "_INCOMG": 5,
        "_MRACE1": 1, "_HISPANC": 1, "SMOKE100": 2, "SMOKDAY2": 3,
        "ALCDAY5": 0.4286, "DRNKANY5": 1, "EXERANY2": 1, "FRUIT1": 1.0, "VEGETAB1": 0.14,
        "HLTHPLN1": 1, "PERSDOC2": 2, "MEDCOST": 1, "CHECKUP1": 3,
        "BPHIGH4": 2, "BPMEDS": 2, "TOLDHI2": 2, "CHOLCHK": 1, "ASTHMA3": 1, "HAVARTH3": 2,
        "GENHLTH": 3, "PHYSHLTH": 3, "MENTHLTH": 15, "POORHLTH": 5,
        "DIFFWALK": 2, "DECIDE": 1, "WEIGHT2": 1500, "HEIGHT3": 505, "_BMI5": 2500,
    },
}

disease_for_explanation = {
    "Example 1: Older female, multiple CVD risk factors":  "coronary_hd",
    "Example 2: Middle-aged, relatively low risk":         "heart_attack",
    "Example 3: Younger adult with depression risk":       "depression",
}

all_results = {}

for label, user in example_users.items():
    print("\n" + "=" * 80)
    print(label)
    print("=" * 80)

    pred_df = predict_user(user)
    print(pred_df)

    disease = disease_for_explanation[label]
    print(f"\nTop feature contributions for: {disease}")
    p, case_label, contrib = explain_prediction(user, disease)
    print(f"  Predicted probability: {p:.3f} -> {case_label}")
    print(contrib)

    all_results[label] = {
        "predictions":          pred_df,
        "explanation_disease":  disease,
        "probability":          p,
        "case_label":           case_label,
        "contrib":              contrib,
    }


# ============================================================
# 13. Calibration curves per disease
# ============================================================
def plot_calibration_for_disease(disease, disease_models=disease_models, n_bins=10):
    if disease not in disease_models:
        raise ValueError(f"Disease '{disease}' not found in disease_models.")

    info    = disease_models[disease]
    y_test  = info["y_test"]
    y_proba = info["y_proba"]

    prob_true, prob_pred = calibration_curve(
        y_test, y_proba, n_bins=n_bins, strategy="quantile"
    )

    plt.figure(figsize=(5, 5))
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(f"Calibration curve: {disease}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"calibration_{disease}.pdf", dpi=300)
    plt.close()

for d in target_cols_binary:
    plot_calibration_for_disease(d)


# ============================================================
# 14. Save artifacts for downstream use (Streamlit app)
#     NOTE: this OVERWRITES app/artifacts/*.joblib in place. The
#     pre-revision versions of these files are preserved separately
#     under revision/v1_original/<timestamp>/ (see Phase 0.1 of
#     scripts/revision_analyses.py) for before/after comparison.
# ============================================================
optimal_thresholds = {
    disease: info["best_threshold"] for disease, info in disease_models.items()
}

path_disease_models = ARTIFACT_DIR / "disease_models.joblib"
path_opt_thresh     = ARTIFACT_DIR / "optimal_thresholds.joblib"
path_predictors     = ARTIFACT_DIR / "predictor_cols.joblib"

joblib.dump(disease_models,     path_disease_models)
joblib.dump(optimal_thresholds, path_opt_thresh)
joblib.dump(predictor_cols,     path_predictors)

print("\nSaved model artifacts for deployment:")
print("  disease_models       ->", path_disease_models)
print("  optimal_thresholds   ->", path_opt_thresh)
print("  predictor_cols       ->", path_predictors)
print("\nNOTE: this overwrote the previous (pre-revision) artifacts in place.")
print("      The pre-revision versions remain available under:")

print(f"      {FROZEN_DIR}/<timestamp>/  (see revision_analyses.py Phase 0.1)")

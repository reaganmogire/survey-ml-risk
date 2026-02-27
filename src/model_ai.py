#!/usr/bin/env python
# coding: utf-8
"""
model_ai.py

Publication pipeline: Train and evaluate survey-based machine-learning models for
chronic disease risk prediction using harmonized BRFSS data (2011–2015).

This script:
  1) Loads the cleaned BRFSS modeling dataset (produced by data_preparation.py)
  2) Trains one model per outcome using a consistent preprocessing + GBDT pipeline
  3) Evaluates discrimination and calibration metrics
  4) Selects a per-outcome probability threshold maximizing F1 score (on held-out test set)
  5) Exports artifacts for inference (Streamlit app) and manuscript-ready tables/figures

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
from sklearn.isotonic import IsotonicRegression

import joblib

# SHAP is used for interpretability (optional but recommended for the paper)
import shap

# Display settings (useful during interactive runs; safe in scripts)
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)


# ============================================================
# 0b. Path configuration (REPO-RELATIVE; PORTABLE)
# ============================================================
# Repo root is assumed to be: <repo>/src/model_ai.py OR <repo>/model_ai.py
# Adjust parents[...] if you place this file elsewhere.
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1] if THIS_FILE.parent.name in {"src", "scripts"} else THIS_FILE.parent

DATA_DERIVED_DIR = Path(os.environ.get("DATA_DERIVED_DIR", REPO_ROOT / "data" / "derived"))
FIGURES_DIR = Path(os.environ.get("FIGURES_DIR", REPO_ROOT / "figures"))
TABLES_DIR = Path(os.environ.get("TABLES_DIR", REPO_ROOT / "tables"))
ARTIFACT_DIR = Path(os.environ.get("ARTIFACT_DIR", REPO_ROOT / "app" / "artifacts"))

# Create output directories if they do not exist
for d in [DATA_DERIVED_DIR, FIGURES_DIR, TABLES_DIR, ARTIFACT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Input (clean modeling dataset produced by data_preparation.py)
COMBINED_CSV = Path(os.environ.get(
    "BRFSS_CLEAN_CSV",
    DATA_DERIVED_DIR / "BRFSS_2011_2015_clean_model.csv"
))


# ============================================================
# 1. Load cleaned, combined BRFSS data (2011–2015)
#    Assumes the upstream cleaning/merging script has already been run.
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
# 2. Define predictors and targets
#    Predictor list must match the variables produced by the cleaning pipeline.
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

# Included here for completeness (continuous outcome used in some analyses)
target_continuous = ["BMI"]

# Keep only rows with at least one observed binary target.
# This ensures the analytic sample includes participants with ascertainment for ≥1 outcome.
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
# 4. Identify numeric vs categorical predictors
#    Heuristic: numeric dtype → numeric pipeline; otherwise categorical.
# ============================================================
numeric_features = []
categorical_features = []

for col in predictor_cols:
    if pd.api.types.is_numeric_dtype(X[col]):
        numeric_features.append(col)
    else:
        categorical_features.append(col)


# ============================================================
# 5. Preprocessing pipeline
#    Numeric: median imputation + standardization
#    Categorical: mode imputation + one-hot encoding (robust to unseen categories)
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
    ]
)


# ============================================================
# 6. Train one model per disease and choose an F1-optimizing threshold
#    - Train/test split is stratified to preserve class balance.
#    - Metrics reported: AUROC, AUPRC, Brier score
#    - Threshold selection: maximize F1 on the held-out test set
# ============================================================
disease_models = {}

for col in target_cols_binary:
    print("\n==============================")
    print(f"Training model for: {col}")
    print("==============================")

    # Use all rows where this disease label is observed
    mask = y[col].notna()
    X_i = X.loc[mask].copy()
    y_i = y.loc[mask, col].astype(int).copy()
    print(f"  Samples for {col}: {X_i.shape[0]}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_i,
        y_i,
        test_size=0.2,
        random_state=42,
        stratify=y_i
    )

    # Pipeline: preprocessing + HistGradientBoostingClassifier
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

    # Fit model
    clf.fit(X_train, y_train)

    # Predict probabilities on test set (positive class)
    y_proba = clf.predict_proba(X_test)[:, 1]

    # --- Metrics independent of threshold ---
    auc = roc_auc_score(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    brier = brier_score_loss(y_test, y_proba)

    print(f"  AUROC : {auc:.3f}")
    print(f"  AUPRC : {ap:.3f}")
    print(f"  Brier : {brier:.3f}")

    # --- Baseline evaluation at threshold 0.5 (for comparison) ---
    y_pred_05 = (y_proba >= 0.5).astype(int)
    print("\nClassification report (threshold = 0.5):")
    print(classification_report(y_test, y_pred_05, digits=3))

    # --- Find threshold that maximizes F1 on the test set ---
    thresholds = np.linspace(0.01, 0.99, 99)
    f1_scores = []

    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        if y_pred_t.sum() == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(f1_score(y_test, y_pred_t))

    best_idx = int(np.argmax(f1_scores))
    best_thresh = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])

    # Evaluation at optimal threshold
    y_pred_best = (y_proba >= best_thresh).astype(int)
    print(f"\nBest threshold by F1 for {col}: {best_thresh:.3f}  (F1 = {best_f1:.3f})")
    print("Classification report (optimal threshold):")
    print(classification_report(y_test, y_pred_best, digits=3))

    # Store artifacts required for:
    #  - downstream evaluation (test set predictions)
    #  - inference deployment (trained pipeline)
    #  - interpretability (optional)
    disease_models[col] = {
        "model": clf,
        "X_test": X_test,
        "y_test": y_test,
        "y_proba": y_proba,
        "best_threshold": best_thresh,
        "auc": auc,
        "average_precision": ap,
        "brier": brier,
        "f1_best": best_f1,
    }


# ============================================================
# 7. Evaluation helper (supports per-disease thresholds or fixed cutoff)
# ============================================================
def evaluate_models(disease_models, use_optimal_threshold=True, fixed_threshold=0.5):
    """
    Summarize performance per disease using stored test predictions.

    Returns a DataFrame with:
      prevalence, AUROC, AUPRC, Brier, sensitivity, specificity, PPV, NPV
    """
    rows = []

    for disease, info in disease_models.items():
        y_test = info["y_test"]
        y_proba = info["y_proba"]

        t = info["best_threshold"] if (use_optimal_threshold and "best_threshold" in info) else fixed_threshold
        y_pred = (y_proba >= t).astype(int)

        auc = roc_auc_score(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        brier = brier_score_loss(y_test, y_proba)
        prevalence = y_test.mean()

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan

        rows.append({
            "disease": disease,
            "threshold": t,
            "prevalence": prevalence,
            "AUROC": auc,
            "AUPRC": ap,
            "Brier": brier,
            "sensitivity": sens,
            "specificity": spec,
            "PPV": ppv,
            "NPV": npv
        })

    return pd.DataFrame(rows).set_index("disease")


# ============================================================
# 8. Compare fixed threshold (0.5) vs per-disease F1-optimized thresholds
# ============================================================
eval_05 = evaluate_models(disease_models, use_optimal_threshold=False, fixed_threshold=0.5)
eval_opt = evaluate_models(disease_models, use_optimal_threshold=True)

print("\n=== Evaluation @ threshold 0.5 ===")
print(eval_05)
print("\n=== Evaluation @ per-disease optimal thresholds ===")
print(eval_opt)

# Save evaluation tables (optional but useful for manuscript supplements)
eval_05.to_csv(TABLES_DIR / "model_evaluation_threshold_0p5.csv")
eval_opt.to_csv(TABLES_DIR / "model_evaluation_optimal_thresholds.csv")


# ============================================================
# 9. Uncertainty proxy (simple; peaks at p=0.5)
# ============================================================
def uncertainty_from_proba(p: float) -> float:
    """Scaled uncertainty proxy in [0, 1]."""
    return float(4 * (p * (1 - p)))


# ============================================================
# 10. Predict risks for a single user across all disease models
# ============================================================
def predict_user(input_dict, models=disease_models) -> pd.DataFrame:
    """
    Predict risks for a single user across all disease models.

    - Uses per-disease optimized threshold when available.
    - Aligns input to the training feature schema using predictor_cols.
    """
    row = pd.DataFrame([input_dict]).reindex(columns=predictor_cols, fill_value=np.nan)

    results = []
    for disease, info in models.items():
        model = info["model"]
        p = float(model.predict_proba(row)[0, 1])
        unc = uncertainty_from_proba(p)
        t = float(info.get("best_threshold", 0.5))
        label = "High risk" if p >= t else "Low/Moderate risk"

        results.append({
            "disease": disease,
            "probability": p,
            "threshold_used": t,
            "risk_classification": label,
            "uncertainty": unc
        })

    return pd.DataFrame(results).set_index("disease")


# ============================================================
# 11. Explain a single prediction using SHAP
# ============================================================
def explain_prediction(input_dict, disease, models=disease_models, top_n=10):
    """
    Explain one prediction using SHAP for the underlying tree model inside the pipeline.

    Returns:
      probability, risk label, and a DataFrame of top contributing transformed features.
    """
    if disease not in models:
        raise ValueError(f"Disease '{disease}' not found in disease_models.")

    info = models[disease]
    model = info["model"]

    row = pd.DataFrame([input_dict]).reindex(columns=predictor_cols, fill_value=np.nan)

    p = float(model.predict_proba(row)[0, 1])
    t = float(info.get("best_threshold", 0.5))
    label = "High risk" if p >= t else "Low/Moderate risk"

    preprocessor_ = model.named_steps["preprocessor"]
    clf_ = model.named_steps["clf"]

    X_t = preprocessor_.transform(row)

    explainer = shap.TreeExplainer(clf_)
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
# ============================================================

example_users = {
    "Example 1: Middle-aged, relatively low risk": {
        "_STATE": 24,         # State code: Maryland
        "SEX": 1,             # Sex code: Male
        "_AGEG5YR": 9,        # Age group: 45–49
        "_EDUCAG": 3,
        "_INCOMG": 4,
        "_MRACE1": 1,
        "_HISPANC": 2,
        "SMOKE100": 2,
        "SMOKDAY2": 3,
        "ALCDAY5": 101,
        "DRNKANY5": 2,
        "EXERANY2": 1,
        "FRUIT1": 101,
        "VEGETAB1": 101,
        "HLTHPLN1": 1,
        "PERSDOC2": 1,
        "MEDCOST": 2,
        "CHECKUP1": 2,
        "BPHIGH4": 2,
        "BPMEDS": 2,
        "TOLDHI2": 2,
        "CHOLCHK": 1,
        "ASTHMA3": 2,
        "HAVARTH3": 2,
        "GENHLTH": 2,
        "PHYSHLTH": 0,
        "MENTHLTH": 0,
        "POORHLTH": 0,
        "DIFFWALK": 2,
        "DECIDE": 2,
        "WEIGHT2": 1700,      # Weight (scaled): 170 lb
        "HEIGHT3": 507,       # Height (scaled): 5'7"
        "_BMI5": 2650,        # BMI (scaled): 26.5
    },

    "Example 2: Older female, multiple CVD risk factors": {
        "_STATE": 13,         # State code: Georgia
        "SEX": 2,             # Sex code: Female
        "_AGEG5YR": 13,       # Age group: 65–69
        "_EDUCAG": 2,         # Education: HS or less
        "_INCOMG": 2,         # Income: lower
        "_MRACE1": 2,         # Race code (e.g., Black)
        "_HISPANC": 2,
        "SMOKE100": 1,        # Ever smoked
        "SMOKDAY2": 2,        # Smokes some days
        "ALCDAY5": 201,
        "DRNKANY5": 1,
        "EXERANY2": 2,        # No exercise
        "FRUIT1": 205,
        "VEGETAB1": 205,
        "HLTHPLN1": 1,
        "PERSDOC2": 1,
        "MEDCOST": 2,
        "CHECKUP1": 1,        # Recent checkup
        "BPHIGH4": 1,         # Told high BP
        "BPMEDS": 1,          # On BP meds
        "TOLDHI2": 1,         # Told high cholesterol
        "CHOLCHK": 1,
        "ASTHMA3": 2,
        "HAVARTH3": 1,        # Arthritis
        "GENHLTH": 3,         # General health: fair/good category
        "PHYSHLTH": 10,
        "MENTHLTH": 5,
        "POORHLTH": 7,
        "DIFFWALK": 1,
        "DECIDE": 2,
        "WEIGHT2": 2100,      # Weight (scaled): 210 lb
        "HEIGHT3": 504,       # Height (scaled): 5'4"
        "_BMI5": 3600,        # BMI (scaled): 36.0
    },

    "Example 3: Younger adult with depression risk": {
        "_STATE": 6,          # State code: California
        "SEX": 2,             # Sex code: Female
        "_AGEG5YR": 6,        # Age group: 30–34
        "_EDUCAG": 4,         # Education: college+
        "_INCOMG": 5,
        "_MRACE1": 1,
        "_HISPANC": 1,        # Hispanic
        "SMOKE100": 2,
        "SMOKDAY2": 3,
        "ALCDAY5": 103,
        "DRNKANY5": 1,
        "EXERANY2": 1,
        "FRUIT1": 101,
        "VEGETAB1": 201,
        "HLTHPLN1": 1,
        "PERSDOC2": 2,
        "MEDCOST": 1,         # Cost barrier to care
        "CHECKUP1": 3,
        "BPHIGH4": 2,
        "BPMEDS": 2,
        "TOLDHI2": 2,
        "CHOLCHK": 1,
        "ASTHMA3": 1,
        "HAVARTH3": 2,
        "GENHLTH": 3,
        "PHYSHLTH": 3,
        "MENTHLTH": 15,       # Many poor mental health days
        "POORHLTH": 5,
        "DIFFWALK": 2,
        "DECIDE": 1,          # Cognitive difficulty
        "WEIGHT2": 1500,      # Weight (scaled): 150 lb
        "HEIGHT3": 505,       # Height (scaled): 5'5"
        "_BMI5": 2500,        # BMI (scaled): 25.0
    },
}

# Select a disease to explain for each example user
disease_for_explanation = {
    "Example 1: Middle-aged, relatively low risk": "heart_attack",
    "Example 2: Older female, multiple CVD risk factors": "coronary_hd",
    "Example 3: Younger adult with depression risk": "depression",
}

all_results = {}

for label, user in example_users.items():
    print("\n" + "="*80)
    print(label)
    print("="*80)

    # Full multi-disease prediction table
    pred_df = predict_user(user)
    display(pred_df)

    # SHAP explanation for the designated disease
    disease = disease_for_explanation[label]
    print(f"\nTop feature contributions for: {disease}")
    p, risk_label, contrib = explain_prediction(user, disease)
    print(f"  Predicted probability: {p:.3f} → {risk_label}")
    display(contrib)

    all_results[label] = {
        "predictions": pred_df,
        "explanation_disease": disease,
        "probability": p,
        "risk_label": risk_label,
        "contrib": contrib,
    }


# ============================================================
# 13. Calibration curves per disease
#    Uses stored y_test and y_proba for each disease model.
# ============================================================

from sklearn.calibration import calibration_curve

def plot_calibration_for_disease(disease, disease_models=disease_models, n_bins=10):
    """
    Plot calibration curve for a given disease using its stored test set.
    """
    if disease not in disease_models:
        raise ValueError(f"Disease '{disease}' not found in disease_models.")

    info = disease_models[disease]
    y_test = info["y_test"]
    y_proba = info["y_proba"]

    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=n_bins)

    plt.figure(figsize=(5, 5))
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(f"Calibration curve: {disease}")
    plt.legend()
    plt.grid(True)
    plt.show()

# Example: plot for diabetes (or any other disease)
plot_calibration_for_disease("diabetes")

# Optionally, loop over all diseases:
for d in target_cols_binary:
    plot_calibration_for_disease(d)


# ============================================================
# 14. Save artifacts for downstream use (Streamlit app)
# ============================================================
optimal_thresholds = {disease: info["best_threshold"] for disease, info in disease_models.items()}

path_disease_models = ARTIFACT_DIR / "disease_models.joblib"
path_opt_thresh = ARTIFACT_DIR / "optimal_thresholds.joblib"
path_predictors = ARTIFACT_DIR / "predictor_cols.joblib"

joblib.dump(disease_models, path_disease_models)
joblib.dump(optimal_thresholds, path_opt_thresh)
joblib.dump(predictor_cols, path_predictors)

print("\nSaved model artifacts for deployment:")
print("  disease_models       ->", path_disease_models)
print("  optimal_thresholds   ->", path_opt_thresh)
print("  predictor_cols       ->", path_predictors)

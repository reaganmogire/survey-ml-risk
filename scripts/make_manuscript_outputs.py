#!/usr/bin/env python3
# coding: utf-8
"""
scripts/make_manuscript_outputs.py

Standalone manuscript-output generator for the Survey-ML Risk project.

This script is intentionally independent of src/model_ai.py and src/data_preparation.py.

It generates:
  - Figure: ROC curves (multi-panel)
  - Figure: Calibration curves (multi-panel)
  - Figure: Mean uncertainty by disease
  - Table 3: Model performance (incl. AUROC/AUPRC/Brier/Sens/Spec/PPV/NPV + calibration errors Emax/ECE/MCE)
  - Uncertainty summary tables:
      * mean/median/Q1/Q3 + mean CI
      * mean(SE) + median[IQR]
  - Example-user outputs:
      * Table A: predicted risks for example users
      * Table B: top SHAP features for a selected disease per example user (if SHAP available)
      * Figure: example predicted risks
      * Figure: example SHAP explanations (if SHAP available)

Optional (requires cleaned dataset):
  - Table 1 (wide): demographics by year + all years
  - Table 2: disease prevalence by year + all years

Inputs:
  - Required artifacts (default locations):
      app/artifacts/disease_models.joblib
      app/artifacts/optimal_thresholds.joblib   (optional but recommended)
      app/artifacts/predictor_cols.joblib       (optional)

  - Optional dataset:
      data/derived/BRFSS_2011_2015_clean_model.csv

Outputs:
  - tables/*.csv
  - figures/*.pdf

Run (from repo root):
  python scripts/make_manuscript_outputs.py

Environment variables (optional):
  ARTIFACT_DIR    (default: <repo>/app/artifacts)
  TABLES_DIR      (default: <repo>/tables)
  FIGURES_DIR     (default: <repo>/figures)
  CLEAN_DATA_CSV  (default: <repo>/data/derived/BRFSS_2011_2015_clean_model.csv)
"""

from __future__ import annotations

import os
import math
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import joblib

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_curve,
)
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression

# Optional: SHAP
try:
    import shap  # type: ignore
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False


# ============================================================
# Paths (repo-relative defaults; override via env vars)
# ============================================================
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]  # <repo>/scripts/make_manuscript_outputs.py

ARTIFACT_DIR = Path(os.environ.get("ARTIFACT_DIR", REPO_ROOT / "app" / "artifacts"))
TABLES_DIR   = Path(os.environ.get("TABLES_DIR", REPO_ROOT / "tables"))
FIGURES_DIR  = Path(os.environ.get("FIGURES_DIR", REPO_ROOT / "figures"))

CLEAN_DATA_CSV = Path(os.environ.get(
    "CLEAN_DATA_CSV",
    REPO_ROOT / "data" / "derived" / "BRFSS_2011_2015_clean_model.csv"
))

for d in [TABLES_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# Disease display names
# ============================================================
DISEASE_NAME_MAP = {
    "heart_attack": "Heart attack",
    "coronary_hd": "Coronary heart disease",
    "stroke": "Stroke",
    "kidney": "Chronic kidney disease",
    "depression": "Depression",
    "diabetes": "Diabetes",
}

def pretty_disease_name(d: str) -> str:
    return DISEASE_NAME_MAP.get(d, d.replace("_", " ").title())


# ============================================================
# Uncertainty helper (match your app’s semantics)
#   0 = confident (p near 0/1), 1 = uncertain (p=0.5)
# ============================================================
def uncertainty_from_proba(p: float) -> float:
    p = float(p)
    return float(1.0 - abs(p - 0.5) * 2.0)


# ============================================================
# Calibration metrics
#   Emax: isotonic-regression calibrated curve max deviation
#   ECE/MCE: binned absolute calibration error
# ============================================================
def compute_calibration_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, float, float]:
    y_true = np.asarray(y_true).astype(float)
    y_proba = np.asarray(y_proba).astype(float)
    n = len(y_true)

    # --- ECE/MCE (fixed bins) ---
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    abs_errors = []
    weights = []

    for j in range(n_bins):
        if j < n_bins - 1:
            mask = (y_proba >= bins[j]) & (y_proba < bins[j + 1])
        else:
            mask = (y_proba >= bins[j]) & (y_proba <= bins[j + 1])

        if not np.any(mask):
            continue

        p_bin = y_proba[mask].mean()
        o_bin = y_true[mask].mean()
        err = abs(o_bin - p_bin)
        abs_errors.append(err)
        weights.append(mask.sum() / n)

    if abs_errors:
        abs_errors = np.array(abs_errors)
        weights = np.array(weights)
        ece = float(np.sum(weights * abs_errors))
        mce = float(np.max(abs_errors))
    else:
        ece = float("nan")
        mce = float("nan")

    # --- Emax (isotonic) ---
    try:
        ir = IsotonicRegression(out_of_bounds="clip")
        order = np.argsort(y_proba)
        p_sorted = y_proba[order]
        y_sorted = y_true[order]
        y_cal = ir.fit_transform(p_sorted, y_sorted)
        emax = float(np.max(np.abs(y_cal - p_sorted)))
    except Exception:
        emax = float("nan")

    return emax, ece, mce


# ============================================================
# Load artifacts (required)
# ============================================================
def load_artifacts() -> Tuple[Dict[str, Any], Dict[str, float], Optional[list]]:
    disease_models_path = ARTIFACT_DIR / "disease_models.joblib"
    if not disease_models_path.exists():
        raise FileNotFoundError(
            f"Missing required artifact: {disease_models_path}\n"
            "Expected: app/artifacts/disease_models.joblib"
        )

    disease_models = joblib.load(disease_models_path)

    opt_path = ARTIFACT_DIR / "optimal_thresholds.joblib"
    if opt_path.exists():
        optimal_thresholds = joblib.load(opt_path)
    else:
        # fallback: read per-model
        optimal_thresholds = {
            d: float(info.get("best_threshold", 0.5)) for d, info in disease_models.items()
        }

    pred_path = ARTIFACT_DIR / "predictor_cols.joblib"
    predictor_cols = joblib.load(pred_path) if pred_path.exists() else None

    return disease_models, optimal_thresholds, predictor_cols


# ============================================================
# Table 3 (performance + calibration)
# ============================================================
def make_table3_model_performance(
    disease_models: Dict[str, Any],
    thresholds: Dict[str, float],
    save_path: Path,
    n_bins: int = 10,
) -> pd.DataFrame:
    rows = []

    for disease, info in disease_models.items():
        if not isinstance(info, dict) or ("y_test" not in info) or ("y_proba" not in info):
            raise ValueError(
                f"disease_models[{disease}] must contain 'y_test' and 'y_proba' "
                "for manuscript outputs. Rebuild artifacts with these stored."
            )

        y_test = np.asarray(info["y_test"]).astype(int)
        y_proba = np.asarray(info["y_proba"]).astype(float)
        thr = float(thresholds.get(disease, info.get("best_threshold", 0.5)))
        y_pred = (y_proba >= thr).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        ppv  = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        npv  = tn / (tn + fn) if (tn + fn) > 0 else np.nan

        emax, ece, mce = compute_calibration_metrics(y_test, y_proba, n_bins=n_bins)

        rows.append({
            "Disease": pretty_disease_name(disease),
            "N_test": int(len(y_test)),
            "Prevalence_%": float(y_test.mean() * 100.0),
            "Threshold": thr,
            "AUROC": float(roc_auc_score(y_test, y_proba)),
            "AUPRC": float(average_precision_score(y_test, y_proba)),
            "Brier": float(brier_score_loss(y_test, y_proba)),
            "Sensitivity": float(sens),
            "Specificity": float(spec),
            "PPV": float(ppv),
            "NPV": float(npv),
            "Emax": float(emax),
            "ECE": float(ece),
            "MCE": float(mce),
        })

    df = pd.DataFrame(rows).set_index("Disease")
    # Round numeric columns for manuscript readiness
    for c in df.columns:
        if c in {"N_test"}:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce").round(3)

    df.to_csv(save_path)
    return df


# ============================================================
# Uncertainty tables
# ============================================================
def compute_uncertainty_tables(
    disease_models: Dict[str, Any],
    save_mean_median_q1q3: Path,
    save_meanse_medianiqr: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    rows1 = []
    rows2 = []

    for disease, info in disease_models.items():
        y_proba = np.asarray(info["y_proba"]).astype(float)
        u = np.array([uncertainty_from_proba(p) for p in y_proba], dtype=float)

        mean_u = float(np.mean(u))
        median_u = float(np.median(u))
        q1 = float(np.percentile(u, 25))
        q3 = float(np.percentile(u, 75))
        sd = float(np.std(u, ddof=1)) if len(u) > 1 else np.nan
        se = float(sd / np.sqrt(len(u))) if len(u) > 1 else np.nan

        ci_low = mean_u - 1.96 * se if np.isfinite(se) else np.nan
        ci_high = mean_u + 1.96 * se if np.isfinite(se) else np.nan

        rows1.append({
            "disease": disease,
            "mean_uncertainty": mean_u,
            "median_uncertainty": median_u,
            "q1_uncertainty": q1,
            "q3_uncertainty": q3,
            "ci_low_mean": ci_low,
            "ci_high_mean": ci_high,
        })

        rows2.append({
            "disease": disease,
            "Mean (SE)": f"{mean_u:.3f} ({se:.3f})" if np.isfinite(se) else f"{mean_u:.3f} (n/a)",
            "Median [Q1–Q3]": f"{median_u:.3f} [{q1:.3f}, {q3:.3f}]",
            "N": int(len(u)),
        })

    df1 = pd.DataFrame(rows1).set_index("disease")
    df1.to_csv(save_mean_median_q1q3)

    df2 = pd.DataFrame(rows2).set_index("disease")
    df2.to_csv(save_meanse_medianiqr)

    return df1, df2


# ============================================================
# Figures: ROC + Calibration + Uncertainty
# ============================================================
def plot_roc_curves(disease_models: Dict[str, Any], save_path: Path) -> None:
    diseases = list(disease_models.keys())
    n = len(diseases)
    n_cols = 3
    n_rows = math.ceil(n / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = np.array(axes).reshape(-1)

    for ax, disease in zip(axes, diseases):
        y_test = np.asarray(disease_models[disease]["y_test"]).astype(int)
        y_proba = np.asarray(disease_models[disease]["y_proba"]).astype(float)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)

        ax.plot(fpr, tpr, label=f"AUROC={auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_title(pretty_disease_name(disease))
        ax.set_xlabel("False positive rate (1 − specificity)")
        ax.set_ylabel("True positive rate (sensitivity)")
        ax.grid(True)
        ax.legend(fontsize=8)

    for j in range(len(diseases), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_calibration_curves(disease_models: Dict[str, Any], save_path: Path, n_bins: int = 10) -> None:
    diseases = list(disease_models.keys())
    n = len(diseases)
    n_cols = 3
    n_rows = math.ceil(n / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = np.array(axes).reshape(-1)

    for ax, disease in zip(axes, diseases):
        y_test = np.asarray(disease_models[disease]["y_test"]).astype(int)
        y_proba = np.asarray(disease_models[disease]["y_proba"]).astype(float)
        prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=n_bins, strategy="quantile")

        ax.plot(prob_pred, prob_true, marker="o", label="Model")
        ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect")
        ax.set_title(pretty_disease_name(disease))
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.grid(True)
        ax.legend(fontsize=8)

    for j in range(len(diseases), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_uncertainty_means(disease_models: Dict[str, Any], save_path: Path) -> None:
    diseases = list(disease_models.keys())
    means = []
    labels = []
    for d in diseases:
        y_proba = np.asarray(disease_models[d]["y_proba"]).astype(float)
        u = np.array([uncertainty_from_proba(p) for p in y_proba], dtype=float)
        means.append(float(np.mean(u)))
        labels.append(pretty_disease_name(d))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, means)
    ax.set_ylabel("Mean uncertainty")
    ax.set_title("Mean uncertainty by disease")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


# ============================================================
# Example users (prediction + SHAP) — fully self-contained
# ============================================================
FEATURE_LABELS = {
    "GENHLTH": "General health",
    "PHYSHLTH": "Physical health days",
    "MENTHLTH": "Mental health days",
    "POORHLTH": "Days poor health limited activities",
    "BPHIGH4": "High blood pressure",
    "BPMEDS": "Blood pressure medication",
    "TOLDHI2": "Told high cholesterol",
    "CHOLCHK": "Cholesterol checked recently",
    "SMOKE100": "Ever smoked 100 cigarettes",
    "SMOKDAY2": "Current smoking frequency",
    "ALCDAY5": "Alcohol drinking days",
    "DRNKANY5": "Any alcohol use",
    "EXERANY2": "Any exercise in past month",
    "FRUIT1": "Fruit intake",
    "VEGETAB1": "Vegetable intake",
    "ASTHMA3": "Asthma",
    "HAVARTH3": "Arthritis",
    "DIFFWALK": "Difficulty walking/climbing",
    "DECIDE": "Cognitive difficulty (memory/concentration)",
    "PERSDOC2": "Has personal doctor",
    "HLTHPLN1": "Has health insurance",
    "MEDCOST": "Could not see doctor (cost)",
    "CHECKUP1": "Time since last checkup",
    "_BMI5": "BMI",
    "WEIGHT2": "Weight (scaled)",
    "HEIGHT3": "Height (scaled)",
    "_AGEG5YR": "Age group",
    "_EDUCAG": "Education category",
    "_INCOMG": "Income category",
    "_MRACE1": "Race",
    "_HISPANC": "Hispanic ethnicity",
    "_STATE": "State",
    "SEX": "Sex",
}

def pretty_feature_from_explainer(name: str) -> str:
    # attempt to recover a BRFSS code from one-hot feature names
    s = str(name)
    # common patterns: num__GENHLTH, cat__GENHLTH_2, etc.
    if s.lower().startswith(("num__", "cat__")):
        s = s[5:]
    if s.lower().startswith(("num", "cat")):
        s = s[3:]
    # try longest all-caps token
    import re
    caps = re.findall(r"[A-Z0-9_]{3,}", s)
    for token in reversed(caps):
        token = token.strip("_")
        if token in FEATURE_LABELS:
            return FEATURE_LABELS[token]
        if f"_{token}" in FEATURE_LABELS:
            return FEATURE_LABELS[f"_{token}"]
    return s.replace("_", " ")

def predict_user_all(input_dict: Dict[str, Any], disease_models: Dict[str, Any], predictor_cols: Optional[list]) -> pd.DataFrame:
    # align to predictor schema if known, else pass as-is
    if predictor_cols is not None:
        row = pd.DataFrame([input_dict]).reindex(columns=predictor_cols, fill_value=np.nan)
    else:
        row = pd.DataFrame([input_dict])

    out = []
    for disease, info in disease_models.items():
        model = info["model"]
        p = float(model.predict_proba(row)[0, 1])
        thr = float(info.get("best_threshold", 0.5))
        out.append({
            "disease": disease,
            "probability": p,
            "threshold_used": thr,
            "risk_classification": "High risk" if p >= thr else "Low/Moderate risk",
            "uncertainty": uncertainty_from_proba(p),
        })
    return pd.DataFrame(out).set_index("disease")

def explain_user_shap(input_dict: Dict[str, Any], disease: str, disease_models: Dict[str, Any], predictor_cols: Optional[list], top_n: int = 10) -> Optional[pd.DataFrame]:
    if not HAS_SHAP:
        return None
    if disease not in disease_models:
        return None

    info = disease_models[disease]
    model = info["model"]

    if predictor_cols is not None:
        row = pd.DataFrame([input_dict]).reindex(columns=predictor_cols, fill_value=np.nan)
    else:
        row = pd.DataFrame([input_dict])

    # must be a Pipeline with preprocessor + clf
    if not hasattr(model, "named_steps"):
        return None
    if "preprocessor" not in model.named_steps or "clf" not in model.named_steps:
        return None

    pre = model.named_steps["preprocessor"]
    clf = model.named_steps["clf"]

    X_t = pre.transform(row)
    try:
        feature_names = pre.get_feature_names_out()
    except Exception:
        feature_names = np.array([f"f{i}" for i in range(X_t.shape[1])], dtype=object)

    try:
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(X_t)
        # binary: list [class0, class1]
        if isinstance(sv, list) and len(sv) == 2:
            sv = sv[1]
        sv = np.asarray(sv).reshape(-1)
    except Exception:
        return None

    df = pd.DataFrame({"feature": feature_names, "shap_value": sv})
    df["abs"] = df["shap_value"].abs()
    df = df.sort_values("abs", ascending=False).head(top_n).drop(columns=["abs"])
    df["feature_pretty"] = df["feature"].apply(pretty_feature_from_explainer)
    return df

def write_example_outputs(
    disease_models: Dict[str, Any],
    predictor_cols: Optional[list],
    tables_dir: Path,
    figures_dir: Path,
) -> None:

    example_users = {
        "Example 1: Middle-aged, relatively low risk": {
            "_STATE": 24, "SEX": 1, "_AGEG5YR": 9, "_EDUCAG": 3, "_INCOMG": 4, "_MRACE1": 1, "_HISPANC": 2,
            "SMOKE100": 2, "SMOKDAY2": 3, "ALCDAY5": 101, "DRNKANY5": 2, "EXERANY2": 1, "FRUIT1": 101, "VEGETAB1": 101,
            "HLTHPLN1": 1, "PERSDOC2": 1, "MEDCOST": 2, "CHECKUP1": 2, "BPHIGH4": 2, "BPMEDS": 2, "TOLDHI2": 2,
            "CHOLCHK": 1, "ASTHMA3": 2, "HAVARTH3": 2, "GENHLTH": 2, "PHYSHLTH": 0, "MENTHLTH": 0, "POORHLTH": 0,
            "DIFFWALK": 2, "DECIDE": 2, "WEIGHT2": 1700, "HEIGHT3": 507, "_BMI5": 2650,
        },
        "Example 2: Older female, multiple CVD risk factors": {
            "_STATE": 13, "SEX": 2, "_AGEG5YR": 13, "_EDUCAG": 2, "_INCOMG": 2, "_MRACE1": 2, "_HISPANC": 2,
            "SMOKE100": 1, "SMOKDAY2": 2, "ALCDAY5": 201, "DRNKANY5": 1, "EXERANY2": 2, "FRUIT1": 205, "VEGETAB1": 205,
            "HLTHPLN1": 1, "PERSDOC2": 1, "MEDCOST": 2, "CHECKUP1": 1, "BPHIGH4": 1, "BPMEDS": 1, "TOLDHI2": 1,
            "CHOLCHK": 1, "ASTHMA3": 2, "HAVARTH3": 1, "GENHLTH": 3, "PHYSHLTH": 10, "MENTHLTH": 5, "POORHLTH": 7,
            "DIFFWALK": 1, "DECIDE": 2, "WEIGHT2": 2100, "HEIGHT3": 504, "_BMI5": 3600,
        },
        "Example 3: Younger adult with depression risk": {
            "_STATE": 6, "SEX": 2, "_AGEG5YR": 6, "_EDUCAG": 4, "_INCOMG": 5, "_MRACE1": 1, "_HISPANC": 1,
            "SMOKE100": 2, "SMOKDAY2": 3, "ALCDAY5": 103, "DRNKANY5": 1, "EXERANY2": 1, "FRUIT1": 101, "VEGETAB1": 201,
            "HLTHPLN1": 1, "PERSDOC2": 2, "MEDCOST": 1, "CHECKUP1": 3, "BPHIGH4": 2, "BPMEDS": 2, "TOLDHI2": 2,
            "CHOLCHK": 1, "ASTHMA3": 1, "HAVARTH3": 2, "GENHLTH": 3, "PHYSHLTH": 3, "MENTHLTH": 15, "POORHLTH": 5,
            "DIFFWALK": 2, "DECIDE": 1, "WEIGHT2": 1500, "HEIGHT3": 505, "_BMI5": 2500,
        },
    }

    disease_for_explanation = {
        "Example 1: Middle-aged, relatively low risk": "heart_attack",
        "Example 2: Older female, multiple CVD risk factors": "coronary_hd",
        "Example 3: Younger adult with depression risk": "depression",
    }

    pred_rows = []
    shap_rows = []

    for label, user in example_users.items():
        pred_df = predict_user_all(user, disease_models, predictor_cols)

        for dis, r in pred_df.iterrows():
            pred_rows.append({
                "Example": label,
                "Disease_internal": dis,
                "Disease": pretty_disease_name(dis),
                "Predicted_probability": float(r["probability"]),
                "Threshold_used": float(r["threshold_used"]),
                "Risk_classification": r["risk_classification"],
                "Uncertainty": float(r["uncertainty"]),
            })

        dis_explain = disease_for_explanation[label]
        contrib = explain_user_shap(user, dis_explain, disease_models, predictor_cols, top_n=10)

        if contrib is not None and not contrib.empty:
            for rank, (_, row) in enumerate(contrib.iterrows(), start=1):
                shap_rows.append({
                    "Example": label,
                    "Explained_disease": pretty_disease_name(dis_explain),
                    "Rank": rank,
                    "Feature": row["feature"],
                    "Feature_pretty": row["feature_pretty"],
                    "SHAP_value": float(row["shap_value"]),
                })

    tableA = pd.DataFrame(pred_rows)
    tableB = pd.DataFrame(shap_rows)

    tableA.to_csv(tables_dir / "tableA_example_predictions.csv", index=False)
    tableB.to_csv(tables_dir / "tableB_example_explanations.csv", index=False)

    # Figure: predicted risks per example
    examples = list(example_users.keys())
    diseases_order = ["heart_attack", "coronary_hd", "stroke", "kidney", "depression", "diabetes"]

    n_cols = 2
    n_rows = int(np.ceil(len(examples) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = np.array(axes).reshape(-1)

    for ax, label in zip(axes, examples):
        pred_df = predict_user_all(example_users[label], disease_models, predictor_cols)
        ordered = [d for d in diseases_order if d in pred_df.index] + [d for d in pred_df.index if d not in diseases_order]
        y_vals = [pred_df.loc[d, "probability"] for d in ordered]
        x_labels = [pretty_disease_name(d) for d in ordered]

        ax.bar(x_labels, y_vals)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Predicted probability")
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=45)

    for j in range(len(examples), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    fig.savefig(figures_dir / "example_predicted_risks.pdf", dpi=300)
    plt.close(fig)

    # Figure: SHAP bar plots (if available)
    if HAS_SHAP and not tableB.empty:
        n_cols = 2
        n_rows = int(np.ceil(len(examples) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
        axes = np.array(axes).reshape(-1)

        for ax, label in zip(axes, examples):
            dis_explain = disease_for_explanation[label]
            contrib = explain_user_shap(example_users[label], dis_explain, disease_models, predictor_cols, top_n=10)
            if contrib is None or contrib.empty:
                ax.axis("off")
                continue

            dfp = contrib.sort_values("shap_value")
            ax.barh(dfp["feature_pretty"], dfp["shap_value"])
            ax.axvline(0, linewidth=0.8)
            ax.set_title(f"{label}\n{pretty_disease_name(dis_explain)} – SHAP")
            ax.set_xlabel("SHAP value")
            ax.tick_params(axis="y", labelsize=8)

        for j in range(len(examples), len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        fig.savefig(figures_dir / "example_shap_explanations.pdf", dpi=300)
        plt.close(fig)


# ============================================================
# Optional: Table 1 wide + Table 2 prevalence (requires dataset)
# ============================================================
def make_table1_wide(df: pd.DataFrame, save_path: Path) -> pd.DataFrame:
    # year column detection
    if "YEAR" in df.columns:
        year_col = "YEAR"
    elif "IYEAR" in df.columns:
        year_col = "IYEAR"
    else:
        raise ValueError("No YEAR or IYEAR column found in cleaned dataset.")

    years = sorted(df[year_col].dropna().unique())
    all_year_labels = [str(int(y)) for y in years] + ["All years"]

    sex_map = {1: "Male", 2: "Female"}
    age_labels = {
        1:"18–24", 2:"25–29", 3:"30–34", 4:"35–39", 5:"40–44",
        6:"45–49", 7:"50–54", 8:"55–59", 9:"60–64", 10:"65–69",
        11:"70–74", 12:"75–79", 13:"80+"
    }
    educ_map = {1:"< High school", 2:"High school graduate", 3:"Some college", 4:"College graduate"}
    income_labels = {1:"<$15k", 2:"$15–25k", 3:"$25–35k", 4:"$35–50k", 5:"$50–75k", 6:"$75k+"}
    race_map = {1:"White", 2:"Black", 3:"Asian", 4:"Native American", 5:"Other / Multiracial", 6:"≥2 races"}
    hisp_map = {1:"Hispanic", 2:"Non-Hispanic"}

    variable_sets = [
        ("Sex", "SEX", sex_map),
        ("Age group", "_AGEG5YR", age_labels),
        ("Education", "_EDUCAG", educ_map),
        ("Income", "_INCOMG", income_labels),
        ("Race", "_MRACE1", race_map),
        ("Hispanic ethnicity", "_HISPANC", hisp_map),
    ]

    out_rows = []

    def cell_value(count: int, total: int) -> str:
        if total == 0:
            return "0 (0.0%)"
        pct = 100 * count / total
        return f"{count:,} ({pct:.1f}%)"

    # categorical
    for section, col, label_map in variable_sets:
        if col not in df.columns:
            continue
        categories = sorted(df[col].dropna().unique())
        for cat in categories:
            cat_name = label_map.get(cat, str(cat))
            row = {"Section": section, "Category": cat_name}
            for yr in years:
                sub = df[df[year_col] == yr]
                total = sub.shape[0]
                count = int((sub[col] == cat).sum())
                row[str(int(yr))] = cell_value(count, total)
            total_all = df.shape[0]
            count_all = int((df[col] == cat).sum())
            row["All years"] = cell_value(count_all, total_all)
            out_rows.append(row)

    # BMI summary if available
    if "BMI" in df.columns:
        def bmi_summary(series: pd.Series) -> str:
            series = series.dropna()
            if series.empty:
                return ""
            m = series.mean()
            sd = series.std()
            med = series.median()
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            return f"{m:.1f} ({sd:.1f}); {med:.1f} [{q1:.1f}–{q3:.1f}]"

        row = {"Section": "BMI (continuous)", "Category": "BMI mean (SD); median [Q1–Q3]"}
        for yr in years:
            sub = df[df[year_col] == yr]
            row[str(int(yr))] = bmi_summary(sub["BMI"])
        row["All years"] = bmi_summary(df["BMI"])
        out_rows.append(row)

    table = pd.DataFrame(out_rows)[["Section", "Category"] + all_year_labels]
    table.to_csv(save_path, index=False)
    return table


def make_table2_disease_prevalence(df: pd.DataFrame, save_path: Path) -> pd.DataFrame:
    # year column detection
    if "YEAR" in df.columns:
        year_col = "YEAR"
    elif "IYEAR" in df.columns:
        year_col = "IYEAR"
    else:
        raise ValueError("No YEAR or IYEAR column found in cleaned dataset.")

    disease_cols = {
        "heart_attack":  "Heart attack",
        "coronary_hd":   "Coronary heart disease",
        "stroke":        "Stroke",
        "kidney":        "Chronic kidney disease",
        "depression":    "Depression",
        "diabetes":      "Diabetes"
    }

    missing = [c for c in disease_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing disease columns in cleaned dataset: {missing}")

    years = sorted(df[year_col].dropna().unique())
    col_labels = [str(int(y)) for y in years] + ["All years"]

    out = pd.DataFrame(index=list(disease_cols.values()), columns=col_labels, dtype=object)

    for dis_code, dis_name in disease_cols.items():
        for yr in years:
            sub = df[df[year_col] == yr]
            nonmiss = int(sub[dis_code].notna().sum())
            cases = float(sub[dis_code].sum())
            prev = (cases / nonmiss) * 100 if nonmiss > 0 else np.nan
            out.loc[dis_name, str(int(yr))] = f"{int(cases):,} / {nonmiss:,} ({prev:.2f}%)" if nonmiss > 0 else "n/a"

        nonmiss_all = int(df[dis_code].notna().sum())
        cases_all = float(df[dis_code].sum())
        prev_all = (cases_all / nonmiss_all) * 100 if nonmiss_all > 0 else np.nan
        out.loc[dis_name, "All years"] = f"{int(cases_all):,} / {nonmiss_all:,} ({prev_all:.2f}%)" if nonmiss_all > 0 else "n/a"

    out.to_csv(save_path, index=True)
    return out


# ============================================================
# Main
# ============================================================
def main() -> None:
    print("== Survey-ML Risk: manuscript outputs ==")
    print("Repo root:", REPO_ROOT)
    print("Artifacts:", ARTIFACT_DIR)
    print("Tables:", TABLES_DIR)
    print("Figures:", FIGURES_DIR)

    disease_models, optimal_thresholds, predictor_cols = load_artifacts()

    # -------------------------
    # Core outputs (artifact-only)
    # -------------------------
    print("\n[1/5] Table 3: model performance + calibration")
    t3 = make_table3_model_performance(
        disease_models=disease_models,
        thresholds=optimal_thresholds,
        save_path=TABLES_DIR / "table3_model_performance_updated.csv",
        n_bins=10,
    )
    print("Saved:", TABLES_DIR / "table3_model_performance_updated.csv")

    print("\n[2/5] Uncertainty tables")
    _u1, _u2 = compute_uncertainty_tables(
        disease_models=disease_models,
        save_mean_median_q1q3=TABLES_DIR / "table_uncertainty_mean_median_q1_q3.csv",
        save_meanse_medianiqr=TABLES_DIR / "table_uncertainty_meanSE_medianIQR.csv",
    )
    print("Saved uncertainty tables.")

    print("\n[3/5] Figures: ROC + calibration + uncertainty")
    plot_roc_curves(disease_models, FIGURES_DIR / "fig_roc_curves.pdf")
    plot_calibration_curves(disease_models, FIGURES_DIR / "fig_calibration_curves.pdf", n_bins=10)
    plot_uncertainty_means(disease_models, FIGURES_DIR / "fig_uncertainty.pdf")
    print("Saved figures in:", FIGURES_DIR)

    print("\n[4/5] Example users: tables + figures (+SHAP if available)")
    write_example_outputs(
        disease_models=disease_models,
        predictor_cols=predictor_cols,
        tables_dir=TABLES_DIR,
        figures_dir=FIGURES_DIR,
    )
    print("Saved example-user outputs.")

    # -------------------------
    # Optional outputs (needs dataset)
    # -------------------------
    print("\n[5/5] Optional: Table 1 wide + Table 2 prevalence (requires cleaned dataset)")
    if CLEAN_DATA_CSV.exists():
        print("Found cleaned dataset:", CLEAN_DATA_CSV)
        df = pd.read_csv(CLEAN_DATA_CSV)

        t1 = make_table1_wide(df, TABLES_DIR / "table1_wide.csv")
        print("Saved:", TABLES_DIR / "table1_wide.csv")

        t2 = make_table2_disease_prevalence(df, TABLES_DIR / "table2_disease_prevalence.csv")
        print("Saved:", TABLES_DIR / "table2_disease_prevalence.csv")
    else:
        print("Cleaned dataset not found. Skipping Table 1 and Table 2.")
        print("If needed, place it at:", CLEAN_DATA_CSV)
        print("or set env var CLEAN_DATA_CSV to its location.")

    print("\nDone.")


if __name__ == "__main__":
    main()

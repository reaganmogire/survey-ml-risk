#!/usr/bin/env python3
# coding: utf-8
"""
Streamlit app: Survey-based chronic disease risk prediction (BRFSS 2011–2015).

Key features
- Loads trained artifacts (joblib) from app/artifacts/
- If artifacts are missing, downloads them from GitHub Releases (model-v1)
- Provides interactive risk prediction for multiple outcomes
- Optional local explanations with SHAP (no external AI/LLM calls)

This application is intended for research and demonstration purposes only.
It is NOT a clinical decision support tool.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import requests

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve


# ------------------------------------------------------------
# Optional dependency: SHAP
# ------------------------------------------------------------
try:
    import shap  # noqa: F401
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False


# ============================================================
# 1) Artifact auto-download (GitHub Release: model-v1)
# ============================================================
REPO_OWNER = "reaganmogire"
REPO_NAME = "survey-ml-risk"
MODEL_TAG = os.environ.get("MODEL_TAG", "model-v1")  # allow override if you ever version-bump

RELEASE_BASE = f"https://github.com/{REPO_OWNER}/{REPO_NAME}/releases/download/{MODEL_TAG}"

APP_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = APP_DIR / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

ARTIFACTS = {
    "disease_models.joblib": f"{RELEASE_BASE}/disease_models.joblib",
    "optimal_thresholds.joblib": f"{RELEASE_BASE}/optimal_thresholds.joblib",
    "predictor_cols.joblib": f"{RELEASE_BASE}/predictor_cols.joblib",
}


def _download_file(url: str, dest: Path) -> None:
    """Download a file with a Streamlit progress indicator."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0

        prog = st.progress(0.0)
        msg = st.empty()

        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    prog.progress(min(downloaded / total, 1.0))
                    msg.caption(f"Downloaded {downloaded/1e6:.1f} / {total/1e6:.1f} MB")

    tmp.replace(dest)
    msg.empty()


def ensure_model_artifacts() -> None:
    """Ensure artifacts exist locally; otherwise download from GitHub Releases."""
    missing = [name for name in ARTIFACTS if not (ARTIFACT_DIR / name).exists()]
    if not missing:
        return

    st.info("Downloading model artifacts from GitHub Releases (first run only)…")
    for name in missing:
        url = ARTIFACTS[name]
        dest = ARTIFACT_DIR / name
        with st.spinner(f"Downloading {name}…"):
            _download_file(url, dest)


@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load artifacts once per Streamlit container session."""
    ensure_model_artifacts()
    disease_models = joblib.load(ARTIFACT_DIR / "disease_models.joblib")
    optimal_thresholds = joblib.load(ARTIFACT_DIR / "optimal_thresholds.joblib")
    predictor_cols = joblib.load(ARTIFACT_DIR / "predictor_cols.joblib")
    return disease_models, optimal_thresholds, predictor_cols


disease_models, optimal_thresholds, predictor_cols = load_artifacts()


# ============================================================
# 2) Utility functions
# ============================================================
def uncertainty_from_proba(p: float) -> float:
    """
    Simple uncertainty proxy:
      0 -> very confident (p near 0 or 1)
      1 -> maximally uncertain (p = 0.5)
    """
    p = float(p)
    return float(1.0 - abs(p - 0.5) * 2.0)


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def predict_all_conditions(model_input: Dict[str, Any]) -> pd.DataFrame:
    """
    Predict risk for each condition.

    model_input must include keys matching predictor_cols.
    Missing predictors are filled with NaN (the pipeline should impute as trained).
    """
    row = pd.DataFrame([model_input]).reindex(columns=predictor_cols, fill_value=np.nan)

    records = []
    for disease, info in disease_models.items():
        model = info["model"] if isinstance(info, dict) and "model" in info else info
        thr = float(optimal_thresholds.get(disease, 0.5))

        proba = float(model.predict_proba(row)[0, 1])
        label = "High risk" if proba >= thr else "Low / moderate risk"
        records.append(
            {
                "Condition": disease,
                "probability": proba,
                "threshold_used": thr,
                "risk_classification": label,
                "uncertainty": uncertainty_from_proba(proba),
            }
        )

    df = pd.DataFrame(records).sort_values("probability", ascending=False)
    return df


def rule_based_guidance(user_inputs: Dict[str, Any], results_df: pd.DataFrame) -> str:
    """
    Educational, rule-based guidance. No external services, no LLM.
    """
    lines = []
    high = results_df.loc[results_df["risk_classification"] == "High risk", "Condition"].tolist()

    if high:
        lines.append("**Higher-risk flags (model-based):** " + ", ".join(high) + ".")
    else:
        lines.append("**Model-based result:** No conditions flagged as higher risk at the stored thresholds.")

    bmi = _safe_float(user_inputs.get("BMI (kg/m²)"))
    if bmi is not None:
        if bmi >= 30:
            lines.append(
                "- BMI suggests obesity. Gradual weight reduction (diet quality + regular activity) "
                "can reduce cardiometabolic risk."
            )
        elif bmi >= 25:
            lines.append(
                "- BMI suggests overweight. Small, sustained changes (daily walking, reduced sugary drinks, "
                "higher fiber/vegetables) can improve risk."
            )

    smoke = str(user_inputs.get("Smoking status", "")).lower()
    if "current" in smoke:
        lines.append(
            "- Current smoking increases cardiovascular and overall risk. Consider evidence-based cessation "
            "support (counseling, nicotine replacement, medications as appropriate)."
        )

    alc = _safe_float(user_inputs.get("Alcohol (drinks/week)"))
    if alc is not None and alc >= 14:
        lines.append(
            "- Reported alcohol intake is relatively high. Reducing intake can lower blood pressure and "
            "improve cardiometabolic health."
        )

    phys = str(user_inputs.get("Any exercise in past month?", "")).lower()
    if phys == "no":
        lines.append(
            "- Increasing physical activity (as medically appropriate) supports cardiometabolic, renal, and "
            "mental health."
        )

    lines.append(
        "\n⚠️ **Disclaimer:** This tool is for research/demonstration only. It does not provide medical advice "
        "or diagnosis. Consult a qualified clinician for personalised guidance."
    )

    return "\n".join(lines)


# ============================================================
# 3) Optional SHAP explainers
# ============================================================
@st.cache_resource(show_spinner=False)
def build_shap_explainers():
    if not _HAS_SHAP:
        return None

    explainers = {}
    for disease, info in disease_models.items():
        model = info["model"] if isinstance(info, dict) and "model" in info else info
        try:
            # Many of your models are sklearn Pipelines with a tree-based classifier
            # SHAP can sometimes handle the estimator directly; if not, we degrade gracefully.
            explainers[disease] = shap.TreeExplainer(model)
        except Exception:
            explainers[disease] = None
    return explainers


shap_explainers = build_shap_explainers()


def explain_instance(model_input: Dict[str, Any], disease: str, top_n: int = 20) -> Optional[pd.DataFrame]:
    if not _HAS_SHAP or shap_explainers is None:
        return None
    explainer = shap_explainers.get(disease)
    if explainer is None:
        return None

    x = pd.DataFrame([model_input]).reindex(columns=predictor_cols, fill_value=np.nan)

    try:
        sv = explainer.shap_values(x)
        # Binary classifiers sometimes return list [class0, class1]
        if isinstance(sv, list) and len(sv) == 2:
            sv = sv[1]
        sv = np.array(sv).reshape(-1)
    except Exception:
        return None

    df = pd.DataFrame({"feature": predictor_cols, "shap_value": sv})
    df["abs"] = df["shap_value"].abs()
    df = df.sort_values("abs", ascending=False).head(top_n).drop(columns=["abs"])
    return df


# ============================================================
# 4) Streamlit UI
# ============================================================
st.set_page_config(page_title="Survey-ML Risk", layout="wide")

st.sidebar.title("Survey-ML Risk")
page = st.sidebar.radio("Navigate", ["Risk prediction", "Model evaluation", "About"])
st.sidebar.markdown("---")
st.sidebar.caption("Research prototype; not for clinical use.")


# ------------------------------------------------------------
# Page: About
# ------------------------------------------------------------
if page == "About":
    st.title("About this app")

    st.markdown(
        """
This repository provides an interpretable machine-learning framework for estimating chronic disease risk
using population survey data (BRFSS 2011–2015).

**Key design goals**
- Scalable risk estimation without EHRs, biomarkers, or laboratory data
- Interpretable feature contributions (SHAP, optional)
- Reproducible, publication-oriented outputs (tables/figures)

**Disclaimer**
This application is intended for research and demonstration purposes only and does not constitute medical advice.
        """
    )

    st.markdown("### Loaded artifact summary")
    st.write(f"Model tag: `{MODEL_TAG}`")
    st.write(f"Artifacts directory: `{ARTIFACT_DIR}`")
    st.write("Predictor columns loaded:", len(predictor_cols))
    st.write("Conditions loaded:", list(disease_models.keys()))


# ------------------------------------------------------------
# Page: Risk prediction
# ------------------------------------------------------------
elif page == "Risk prediction":
    st.title("Risk prediction (survey-based ML)")

    st.markdown(
        """
Enter inputs below to generate predicted risks for multiple conditions.
On the first run, the app downloads model artifacts from GitHub Releases.
        """
    )

    # -------- Inputs (minimal set; extend to cover your full predictor schema) --------
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age (years)", min_value=18, max_value=99, value=35, step=1)
        sex = st.selectbox("Sex", ["Male", "Female"])
        bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=70.0, value=27.0, step=0.1)

    with col2:
        htn = st.selectbox("Ever told you have high blood pressure?", ["No", "Yes"])
        diffwalk = st.selectbox("Difficulty walking/climbing stairs?", ["No", "Yes"])
        genhlth = st.selectbox("General health", ["Excellent", "Very good", "Good", "Fair", "Poor"])

    with col3:
        smoke = st.selectbox("Smoking status", ["Never", "Former", "Current"])
        drinks_pw = st.number_input("Alcohol (drinks/week)", min_value=0.0, max_value=70.0, value=0.0, step=1.0)
        exer = st.selectbox("Any exercise in past month?", ["No", "Yes"])

    # -------- Map user-friendly inputs -> BRFSS-coded predictors --------
    # IMPORTANT: Your trained model expects specific BRFSS-coded columns.
    # This mapping is intentionally minimal and safe; expand it to match your full feature set.
    # Any missing predictors will be set to NaN (your pipeline should impute if trained accordingly).
    model_input: Dict[str, Any] = {}

    # Common BRFSS-style codes (examples). Adjust if your training used different coding.
    if "_AGEG5YR" in predictor_cols:
        # Approximate 5-year age-group bins: 18–24=1, 25–29=2, ... 80+=13
        # This is a standard BRFSS-style mapping; update if your pipeline encoded differently.
        def age_to_ageg5yr(a: int) -> int:
            a = max(18, min(int(a), 99))
            bins = [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
            for i, upper in enumerate(bins, start=1):
                if a < upper:
                    return i
            return 13

        model_input["_AGEG5YR"] = age_to_ageg5yr(age)

    if "SEX" in predictor_cols:
        model_input["SEX"] = 1 if sex == "Male" else 2

    if "_BMI5" in predictor_cols:
        model_input["_BMI5"] = int(round(float(bmi) * 100))

    if "BPHIGH4" in predictor_cols:
        # 1=yes, 2=no
        model_input["BPHIGH4"] = 1 if htn == "Yes" else 2

    if "DIFFWALK" in predictor_cols:
        model_input["DIFFWALK"] = 1 if diffwalk == "Yes" else 2

    if "GENHLTH" in predictor_cols:
        model_input["GENHLTH"] = {"Excellent": 1, "Very good": 2, "Good": 3, "Fair": 4, "Poor": 5}[genhlth]

    if "EXERANY2" in predictor_cols:
        model_input["EXERANY2"] = 1 if exer == "Yes" else 2

    if "SMOKE100" in predictor_cols:
        # crude approximation: never -> 2 (No), former/current -> 1 (Yes)
        model_input["SMOKE100"] = 2 if smoke == "Never" else 1

    if "ALCDAY5" in predictor_cols:
        # Simple encoding: 888=no drinks; else 200+drinks/week capped at 7
        if drinks_pw <= 0:
            model_input["ALCDAY5"] = 888
        else:
            d = int(round(min(float(drinks_pw), 7.0)))
            model_input["ALCDAY5"] = 200 + d

    user_inputs = {
        "Age (years)": age,
        "Sex": sex,
        "BMI (kg/m²)": bmi,
        "High blood pressure history": htn,
        "Difficulty walking/climbing stairs?": diffwalk,
        "General health": genhlth,
        "Smoking status": smoke,
        "Alcohol (drinks/week)": drinks_pw,
        "Any exercise in past month?": exer,
    }

    # Show missing predictors for transparency
    missing_predictors = [c for c in predictor_cols if c not in model_input]
    with st.expander("Predictor coverage (for transparency)"):
        st.write(f"Provided predictors: {len(model_input)} / {len(predictor_cols)}")
        st.write("Missing predictors will be imputed as NaN (per training pipeline).")
        st.write(missing_predictors)

    run_btn = st.button("Run prediction")

    if run_btn:
        results_df = predict_all_conditions(model_input)

        st.subheader("Predicted risks")
        st.dataframe(results_df, use_container_width=True)

        st.markdown(
            """
**Field definitions**
- `probability`: predicted risk (0–1)  
- `threshold_used`: stored threshold for classification  
- `risk_classification`: label using `threshold_used`  
- `uncertainty`: 0 (confident) → 1 (uncertain; highest at p≈0.5)
            """
        )

        st.subheader("Educational guidance (rule-based)")
        st.write(rule_based_guidance(user_inputs, results_df))

        st.subheader("Local feature contributions (optional)")
        if not _HAS_SHAP:
            st.info("Install `shap` to enable local explanations.")
        else:
            top_condition = results_df.iloc[0]["Condition"]
            disease_choice = st.selectbox("Select condition to explain", options=list(results_df["Condition"]),
                                          index=0)

            contrib = explain_instance(model_input, disease_choice, top_n=20)
            if contrib is None or contrib.empty:
                st.info("SHAP explanation not available for this model/configuration.")
            else:
                st.dataframe(contrib, use_container_width=True)

                # Simple bar chart
                df_plot = contrib.sort_values("shap_value")
                fig, ax = plt.subplots(figsize=(7, 5))
                ax.barh(df_plot["feature"], df_plot["shap_value"])
                ax.set_xlabel("SHAP value (impact on model output)")
                ax.set_title(f"Local explanation: {disease_choice}")
                st.pyplot(fig)


# ------------------------------------------------------------
# Page: Model evaluation
# ------------------------------------------------------------
elif page == "Model evaluation":
    st.title("Model evaluation")

    st.markdown(
        """
This page visualizes discrimination and calibration using stored test predictions,
*if* they were included in the serialized artifact.

Expected keys per disease inside `disease_models.joblib`:
- `y_test` (array-like)
- `y_proba` (array-like predicted probabilities)

If these keys are absent, you can still rely on the precomputed tables in `tables/`.
        """
    )

    available = []
    for disease, info in disease_models.items():
        if isinstance(info, dict) and ("y_test" in info and "y_proba" in info):
            available.append(disease)

    if not available:
        st.warning(
            "No stored test predictions found in `disease_models.joblib` (missing `y_test` and/or `y_proba`). "
            "To enable interactive evaluation, update `src/model_ai.py` to store these arrays when serializing."
        )
        st.markdown("### Precomputed outputs")
        st.write("See the `tables/` and `figures/` folders in this repository for manuscript-ready outputs.")
    else:
        disease_choice = st.selectbox("Select condition", options=available)
        info = disease_models[disease_choice]
        y = np.asarray(info["y_test"]).astype(int)
        p = np.asarray(info["y_proba"]).astype(float)

        auroc = roc_auc_score(y, p)
        ap = average_precision_score(y, p)
        brier = brier_score_loss(y, p)

        c1, c2, c3 = st.columns(3)
        c1.metric("AUROC", f"{auroc:.3f}")
        c2.metric("PR-AUC", f"{ap:.3f}")
        c3.metric("Brier", f"{brier:.3f}")

        # ROC curve
        fpr, tpr, _ = roc_curve(y, p)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(fpr, tpr)
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"ROC: {disease_choice}")
        st.pyplot(fig)

        # PR curve
        prec, rec, _ = precision_recall_curve(y, p)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(rec, prec)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision–Recall: {disease_choice}")
        st.pyplot(fig)

        # Calibration
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(mean_pred, frac_pos, marker="o")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title(f"Calibration: {disease_choice}")
        st.pyplot(fig)

        # Confusion matrix at stored threshold
        thr = float(optimal_thresholds.get(disease_choice, 0.5))
        yhat = (p >= thr).astype(int)
        cm = confusion_matrix(y, yhat)
        st.write(f"Confusion matrix at threshold {thr:.3f}")
        st.dataframe(pd.DataFrame(cm, index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"]))

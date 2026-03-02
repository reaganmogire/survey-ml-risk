#!/usr/bin/env python3
# coding: utf-8

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
MODEL_TAG = os.environ.get("MODEL_TAG", "model-v1")  # override if you version-bump

RELEASE_BASE = f"https://github.com/{REPO_OWNER}/{REPO_NAME}/releases/download/{MODEL_TAG}"

APP_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = APP_DIR / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

ARTIFACTS = {
    "disease_models.joblib": f"{RELEASE_BASE}/disease_models.joblib",
    "optimal_thresholds.joblib": f"{RELEASE_BASE}/optimal_thresholds.joblib",
    "predictor_cols.joblib": f"{RELEASE_BASE}/predictor_cols.joblib",
}

DISCLAIMER_TEXT = (
    "Disclaimer: This tool does not provide a diagnosis and is not a substitute for professional medical advice. "
    "If you have health concerns, consult a qualified clinician."
)


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

    lines.append("\n⚠️ " + DISCLAIMER_TEXT)
    return "\n".join(lines)


# ============================================================
# 3) SHAP utilities (Pipeline-safe)
# ============================================================
@st.cache_resource(show_spinner=False)
def build_tree_explainers():
    """
    Build SHAP TreeExplainers for the *classifier* inside each sklearn Pipeline.

    Returns:
      dict[disease] = {"preprocessor": preprocessor, "clf": clf, "explainer": TreeExplainer}
    """
    if not _HAS_SHAP:
        return None

    out = {}
    for disease, info in disease_models.items():
        model = info["model"] if isinstance(info, dict) and "model" in info else info

        # Expect sklearn Pipeline with named steps
        if not hasattr(model, "named_steps"):
            out[disease] = None
            continue

        if "preprocessor" not in model.named_steps or "clf" not in model.named_steps:
            out[disease] = None
            continue

        pre = model.named_steps["preprocessor"]
        clf = model.named_steps["clf"]

        try:
            explainer = shap.TreeExplainer(clf)
            out[disease] = {"preprocessor": pre, "clf": clf, "explainer": explainer}
        except Exception:
            out[disease] = None

    return out


tree_explainers = build_tree_explainers()


def explain_instance_pipeline(model_input: Dict[str, Any], disease: str, top_n: int = 20) -> Optional[pd.DataFrame]:
    """
    Compute SHAP values for a single instance for a given disease model.

    Pipeline workflow:
      X_raw -> preprocessor.transform -> X_transformed
      shap.TreeExplainer(clf).shap_values(X_transformed)

    Returns a DataFrame of the top contributing transformed features.
    """
    if not _HAS_SHAP or tree_explainers is None:
        return None

    bundle = tree_explainers.get(disease)
    if bundle is None:
        return None

    pre = bundle["preprocessor"]
    explainer = bundle["explainer"]

    x_raw = pd.DataFrame([model_input]).reindex(columns=predictor_cols, fill_value=np.nan)

    try:
        X_t = pre.transform(x_raw)
        feature_names = pre.get_feature_names_out()
    except Exception:
        return None

    try:
        sv = explainer.shap_values(X_t)
        # For binary classification, SHAP may return a list [class0, class1]
        if isinstance(sv, list) and len(sv) == 2:
            sv = sv[1]
        sv = np.asarray(sv).reshape(-1)
    except Exception:
        return None

    df = pd.DataFrame({"feature": feature_names, "shap_value": sv})
    df["abs"] = df["shap_value"].abs()
    df = df.sort_values("abs", ascending=False).head(top_n).drop(columns=["abs"])
    return df


# ============================================================
# 4) Streamlit UI
# ============================================================
st.set_page_config(page_title="Survey-ML Risk", layout="wide")

# --- CSS: fix dropdown visibility (selectbox) without changing app logic ---
st.markdown(
    """
<style>
/* Keep general page bright */
html, body, [class*="css"] { background-color: #f6fbff; }

/* Selectbox control: force light background + readable text */
div[data-baseweb="select"] > div {
  background: #ffffff !important;
  color: #0b1f33 !important;
  border: 1px solid #cfe3f5 !important;
}

/* Dropdown menu background */
ul[role="listbox"] {
  background-color: #ffffff !important;
  color: #0b1f33 !important;
  border: 1px solid #cfe3f5 !important;
  box-shadow: 0px 4px 12px rgba(0,0,0,0.08) !important;
}

/* Individual dropdown items */
ul[role="listbox"] li {
  background: #ffffff !important;
  color: #0b1f33 !important;
}

/* Hover effect */
ul[role="listbox"] li:hover {
  background-color: #eaf5ff !important;
  color: #0b1f33 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

st.sidebar.title("Survey-ML Risk")
page = st.sidebar.radio("Navigate", ["Risk prediction", "Model evaluation", "About"])
st.sidebar.markdown("---")
st.sidebar.caption("Research/demonstration only; not yet for clinical use.")


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

⚠️ **{disclaimer}**
        """.format(disclaimer=DISCLAIMER_TEXT)
    )

    st.markdown("### Project repository")
    st.markdown("[View source code on GitHub](https://github.com/reaganmogire/survey-ml-risk)")

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

    # -------- Inputs (minimal set; expand to cover your full predictor schema) --------
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
    # Any missing predictors will be set to NaN (your pipeline should impute as trained).
    model_input: Dict[str, Any] = {}

    # BRFSS-style codes (standard approach; adjust if your training pipeline used different coding)
    if "_AGEG5YR" in predictor_cols:

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
        model_input["BPHIGH4"] = 1 if htn == "Yes" else 2

    if "DIFFWALK" in predictor_cols:
        model_input["DIFFWALK"] = 1 if diffwalk == "Yes" else 2

    if "GENHLTH" in predictor_cols:
        model_input["GENHLTH"] = {"Excellent": 1, "Very good": 2, "Good": 3, "Fair": 4, "Poor": 5}[genhlth]

    if "EXERANY2" in predictor_cols:
        model_input["EXERANY2"] = 1 if exer == "Yes" else 2

    if "SMOKE100" in predictor_cols:
        # Approximation: never -> No; former/current -> Yes
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

    # Show predictor coverage for transparency
    missing_predictors = [c for c in predictor_cols if c not in model_input]
    with st.expander("Predictor coverage (for transparency)"):
        st.write(f"Provided predictors: {len(model_input)} / {len(predictor_cols)}")
        st.write("Missing predictors will be passed as NaN (imputed per training pipeline).")
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

        # Simple plots for quick visual interpretation
        st.subheader("Plots")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(results_df["Condition"], results_df["probability"])
        ax.set_ylabel("Predicted probability")
        ax.set_title("Predicted risk by condition")
        ax.tick_params(axis="x", rotation=30)
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(results_df["Condition"], results_df["uncertainty"])
        ax.set_ylabel("Uncertainty (0=confident, 1=uncertain)")
        ax.set_title("Prediction uncertainty by condition")
        ax.tick_params(axis="x", rotation=30)
        st.pyplot(fig)

        st.subheader("Educational guidance (rule-based)")
        st.write(rule_based_guidance(user_inputs, results_df))

        st.subheader("Local feature contributions (optional)")
        if not _HAS_SHAP:
            st.info("Install `shap` to enable local explanations.")
        else:
            disease_choice = st.selectbox("Select condition to explain", options=list(results_df["Condition"]), index=0)

            contrib = explain_instance_pipeline(model_input, disease_choice, top_n=20)
            if contrib is None or contrib.empty:
                st.info("SHAP explanation not available for this model/configuration.")
            else:
                st.dataframe(contrib, use_container_width=True)

                # Bar plot of top SHAP values
                df_plot = contrib.sort_values("shap_value")
                fig, ax = plt.subplots(figsize=(7, 5))
                ax.barh(df_plot["feature"], df_plot["shap_value"])
                ax.set_xlabel("SHAP value (impact on model output)")
                ax.set_title(f"Local explanation: {disease_choice}")
                st.pyplot(fig)

        st.markdown("\n⚠️ **{}**".format(DISCLAIMER_TEXT))


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
        st.markdown("\n⚠️ **{}**".format(DISCLAIMER_TEXT))
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

        # Calibration curve
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

        st.markdown("\n⚠️ **{}**".format(DISCLAIMER_TEXT))

#!/usr/bin/env python3
# coding: utf-8
"""
Streamlit app: Survey-based chronic disease risk prediction (BRFSS 2011–2015).

Key features
- Loads trained artifacts (joblib) from app/artifacts/
- If artifacts are missing, downloads them from GitHub Releases (model-v1)
- Provides interactive risk prediction for multiple outcomes
- Local explanations with SHAP for sklearn Pipelines (preprocessor + tree model), if available
- No external AI/LLM calls

This application is intended for research only.
"""

from __future__ import annotations

import os
import re
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
# 0) Streamlit page + Bright medical theme (forced readability)
# ============================================================
st.set_page_config(page_title="Survey-ML Risk", page_icon="🩺", layout="wide")

st.markdown(
    """
<style>
  /* --- Page background --- */
  .stApp {
    background: linear-gradient(180deg, #F7FBFF 0%, #FFFFFF 40%, #F3FAFF 100%);
  }

  /* --- Sidebar --- */
  section[data-testid="stSidebar"] {
    background: #EAF5FF !important;
    border-right: 1px solid rgba(11,31,53,0.12);
  }

  /* Make ALL text readable */
  html, body, [class*="css"]  {
    color: #0B1F35 !important;
  }

  /* Headings */
  h1, h2, h3, h4, h5, h6 {
    color: #0B3B66 !important;
  }

  /* Paragraph/help text */
  p, li, span, label, small {
    color: #0B1F35 !important;
  }

  /* Cards */
  .card {
    background: #FFFFFF;
    border: 1px solid rgba(11,31,53,0.12);
    border-radius: 14px;
    padding: 16px 18px;
    box-shadow: 0 6px 20px rgba(11,31,53,0.06);
  }

  /* Buttons */
  .stButton>button {
    background: #2F80ED !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.55rem 1.0rem !important;
  }
  .stButton>button:hover {
    background: #1F6FDC !important;
    color: #FFFFFF !important;
  }

  /* --- Widget labels (the ones that became invisible) --- */
  div[data-testid="stWidgetLabel"] label,
  div[data-testid="stWidgetLabel"] p,
  .stRadio label, .stRadio p,
  .stSelectbox label, .stSelectbox p,
  .stNumberInput label, .stNumberInput p {
    color: #0B1F35 !important;
    font-weight: 600 !important;
  }

  /* --- Input boxes: force light background + dark text --- */
  div[data-baseweb="input"] input,
  div[data-baseweb="base-input"] input,
  div[data-baseweb="textarea"] textarea {
    background: #FFFFFF !important;
    color: #0B1F35 !important;
    border: 1px solid rgba(11,31,53,0.18) !important;
    border-radius: 10px !important;
  }

  /* Selectbox (closed state) */
  div[data-baseweb="select"] > div {
    background: #FFFFFF !important;
    color: #0B1F35 !important;
    border: 1px solid rgba(11,31,53,0.18) !important;
    border-radius: 10px !important;
  }

  /* Selectbox dropdown menu - LIGHT TEXT ON DARK BACKGROUND */
  ul[role="listbox"] > li {
    color: #FFFFFF !important;
  }
  ul[role="listbox"] > li:hover {
    background-color: #374151 !important;
    color: #FFFFFF !important;
  }
  div[role="option"] {
    color: #FFFFFF !important;
  }
  div[role="option"]:hover {
    background-color: #374151 !important;
    color: #FFFFFF !important;
  }

  /* Alerts */
  div[role="alert"] * {
    color: #0B1F35 !important;
  }
</style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 1) Artifact auto-download (GitHub Release: model-v1)
# ============================================================
REPO_OWNER = "reaganmogire"
REPO_NAME = "survey-ml-risk"
MODEL_TAG = os.environ.get("MODEL_TAG", "model-v1")

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


# ============================================================
# 1b) Human-friendly labels
# ============================================================
DISEASE_LABELS = {
    "heart_attack": "Heart attack (myocardial infarction)",
    "coronary_hd": "Coronary heart disease",
    "stroke": "Stroke",
    "kidney": "Chronic kidney disease",
    "depression": "Depression",
    "diabetes": "Diabetes",
}

FEATURE_LABELS = {
    "_STATE": "State",
    "SEX": "Sex",
    "_AGEG5YR": "Age group",
    "_EDUCAG": "Education",
    "_INCOMG": "Household income",
    "_MRACE1": "Race",
    "_HISPANC": "Hispanic ethnicity",
    "SMOKE100": "Ever smoked (100 cigarettes)",
    "SMOKDAY2": "Current smoking frequency",
    "ALCDAY5": "Alcohol use frequency",
    "DRNKANY5": "Any alcohol use (past 30 days)",
    "EXERANY2": "Any exercise (past 30 days)",
    "FRUIT1": "Fruit intake",
    "VEGETAB1": "Vegetable intake",
    "HLTHPLN1": "Has health insurance",
    "PERSDOC2": "Has a personal doctor",
    "MEDCOST": "Cost barrier to care",
    "CHECKUP1": "Time since routine checkup",
    "BPHIGH4": "History of high blood pressure",
    "BPMEDS": "On BP medication",
    "TOLDHI2": "Told high cholesterol",
    "CHOLCHK": "Cholesterol checked recently",
    "ASTHMA3": "History of asthma",
    "HAVARTH3": "History of arthritis",
    "GENHLTH": "Self-rated general health",
    "PHYSHLTH": "Physically unhealthy days",
    "MENTHLTH": "Mentally unhealthy days",
    "POORHLTH": "Days poor health limited activities",
    "DIFFWALK": "Difficulty walking/climbing stairs",
    "DECIDE": "Cognitive difficulty (memory/concentration)",
    "WEIGHT2": "Weight (BRFSS-coded)",
    "HEIGHT3": "Height (BRFSS-coded)",
    "_BMI5": "BMI (×100; BRFSS-coded)",
}

SEX_MAP = {1: "Male", 2: "Female"}


def pretty_disease(d: str) -> str:
    return DISEASE_LABELS.get(d, d.replace("_", " ").title())


def pretty_feature_name(raw: str) -> str:
    """
    Convert transformed feature names into intuitive labels.

    Examples:
      num__MENTHLTH -> Mentally unhealthy days
      cat__SEX_1.0  -> Sex = Male
    """
    s = str(raw)

    if s.startswith("num__"):
        base = s[len("num__"):]
        return FEATURE_LABELS.get(base, base)

    if s.startswith("cat__"):
        base = s[len("cat__"):]
        if "_" in base:
            var, level = base.rsplit("_", 1)
            var_label = FEATURE_LABELS.get(var, var)
            if var == "SEX":
                try:
                    level_i = int(float(level))
                    level_label = SEX_MAP.get(level_i, str(level_i))
                    return f"{var_label} = {level_label}"
                except Exception:
                    return f"{var_label} = {level}"
            return f"{var_label} = {level}"
        return FEATURE_LABELS.get(base, base)

    # Fallback recovery
    s2 = s.replace("Num__", "").replace("Cat__", "")
    s2 = s2.replace("num__", "").replace("cat__", "")
    s2 = s2.replace("__", "")
    s2 = re.sub(r"[^A-Za-z0-9_]", "", s2)
    tokens = re.findall(r"[A-Z0-9_]{3,}", s2.upper())
    if tokens:
        code = tokens[-1]
        return FEATURE_LABELS.get(code, FEATURE_LABELS.get(f"_{code}", code))

    return s


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0

        prog = st.progress(0.0)
        msg = st.empty()

        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
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
    missing = [name for name in ARTIFACTS if not (ARTIFACT_DIR / name).exists()]
    if not missing:
        return

    st.info("Downloading model artifacts from GitHub Releases (first run only)…")
    for name in missing:
        with st.spinner(f"Downloading {name}…"):
            _download_file(ARTIFACTS[name], ARTIFACT_DIR / name)


@st.cache_resource(show_spinner=False)
def load_artifacts():
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
    row = pd.DataFrame([model_input]).reindex(columns=predictor_cols, fill_value=np.nan)

    records = []
    for disease, info in disease_models.items():
        model = info["model"] if isinstance(info, dict) and "model" in info else info
        thr = float(optimal_thresholds.get(disease, 0.5))

        proba = float(model.predict_proba(row)[0, 1])
        label = "Higher risk" if proba >= thr else "Lower / moderate risk"
        records.append(
            {
                "Condition": pretty_disease(disease),
                "Predicted risk (0–1)": proba,
                "Threshold": thr,
                "Risk category": label,
                "Uncertainty (0–1)": uncertainty_from_proba(proba),
                "_disease_code": disease,
            }
        )

    df = pd.DataFrame(records).sort_values("Predicted risk (0–1)", ascending=False).reset_index(drop=True)
    return df


def rule_based_guidance(user_inputs: Dict[str, Any], results_df: pd.DataFrame) -> str:
    lines = []
    high = results_df.loc[results_df["Risk category"] == "Higher risk", "Condition"].tolist()

    if high:
        lines.append("**Higher-risk flags (model-based):** " + ", ".join(high) + ".")
    else:
        lines.append("**Model-based result:** No conditions flagged as higher risk at the stored thresholds.")

    bmi = _safe_float(user_inputs.get("BMI (kg/m²)"))
    if bmi is not None:
        if bmi >= 30:
            lines.append("- BMI suggests obesity. Gradual weight reduction (diet quality + regular activity) can reduce cardiometabolic risk.")
        elif bmi >= 25:
            lines.append("- BMI suggests overweight. Small sustained changes can improve risk.")

    smoke = str(user_inputs.get("Smoking status", "")).lower()
    if "current" in smoke:
        lines.append("- Current smoking increases cardiovascular and overall risk. Consider evidence-based cessation support.")

    alc = _safe_float(user_inputs.get("Alcohol (drinks/week)"))
    if alc is not None and alc >= 14:
        lines.append("- Reported alcohol intake is relatively high. Reducing intake can lower blood pressure and improve cardiometabolic health.")

    phys = str(user_inputs.get("Any exercise in past month?", "")).lower()
    if phys == "no":
        lines.append("- Increasing physical activity (as medically appropriate) supports cardiometabolic, renal, and mental health.")

    lines.append("\n⚠️ " + DISCLAIMER_TEXT)
    return "\n".join(lines)


# ============================================================
# 3) SHAP utilities (Pipeline-safe)
# ============================================================
@st.cache_resource(show_spinner=False)
def build_tree_explainers():
    if not _HAS_SHAP:
        return None

    out = {}
    for disease, info in disease_models.items():
        model = info["model"] if isinstance(info, dict) and "model" in info else info

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
            out[disease] = {"preprocessor": pre, "explainer": explainer}
        except Exception:
            out[disease] = None

    return out


tree_explainers = build_tree_explainers()


def explain_instance_pipeline(model_input: Dict[str, Any], disease: str, top_n: int = 20) -> Optional[pd.DataFrame]:
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
        if isinstance(sv, list) and len(sv) == 2:
            sv = sv[1]
        sv = np.asarray(sv).reshape(-1)
    except Exception:
        return None

    df = pd.DataFrame({"feature_raw": feature_names, "shap_value": sv})
    df["Feature"] = df["feature_raw"].apply(pretty_feature_name)
    df["abs"] = df["shap_value"].abs()
    df = df.sort_values("abs", ascending=False).head(top_n).drop(columns=["abs"]).reset_index(drop=True)
    return df[["Feature", "shap_value"]]


def plot_shap_bar(df: pd.DataFrame, title: str) -> plt.Figure:
    d = df.copy().sort_values("shap_value", ascending=True)
    y = d["Feature"].tolist()
    x = d["shap_value"].values

    colors = np.where(x >= 0, "#D9534F", "#2F80ED")  # red=increase, blue=decrease

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(y, x, color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("SHAP value (impact on predicted risk)")
    ax.grid(axis="x", linestyle="--", alpha=0.25)

    # Move legend to bottom
    ax.text(0.02, -0.15, "← Decrease risk (SHAP < 0)", transform=ax.transAxes,
            ha="left", va="top", fontsize=11, color="#2F80ED")
    ax.text(0.98, -0.15, "Increase risk (SHAP > 0) →", transform=ax.transAxes,
            ha="right", va="top", fontsize=11, color="#D9534F")

    fig.tight_layout()
    return fig


# ============================================================
# 4) Streamlit UI
# ============================================================
st.sidebar.title("Survey-ML Risk")
page = st.sidebar.radio("Navigate", ["Risk prediction", "Model evaluation", "About"])
st.sidebar.markdown("---")
st.sidebar.caption("Research only; not for clinical use.")


if page == "About":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.title("About this app")
    st.markdown(
        """
This repository provides an interpretable machine-learning framework for estimating chronic disease risk
using population survey data (BRFSS 2011–2015).

**Key design goals**
- Scalable risk estimation without EHRs, biomarkers, or laboratory data
- Interpretable feature contributions (SHAP, optional)
- Reproducible, publication-oriented outputs (tables/figures)
        """
    )
    st.warning(DISCLAIMER_TEXT, icon="⚠️")
    st.markdown("### Loaded artifact summary")
    st.write(f"Model tag: `{MODEL_TAG}`")
    st.write(f"Artifacts directory: `{ARTIFACT_DIR}`")
    st.write("Predictor columns loaded:", len(predictor_cols))
    st.write("Conditions loaded:", [pretty_disease(k) for k in disease_models.keys()])
    st.markdown("</div>", unsafe_allow_html=True)


elif page == "Risk prediction":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.title("Risk prediction (survey-based ML)")
    st.write("Enter inputs to generate predicted risks. On first run, the app downloads model artifacts from GitHub Releases.")
    st.warning(DISCLAIMER_TEXT, icon="⚠️")
    st.markdown("</div>", unsafe_allow_html=True)

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

    model_input: Dict[str, Any] = {}

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
        model_input["SMOKE100"] = 2 if smoke == "Never" else 1

    if "ALCDAY5" in predictor_cols:
        if drinks_pw <= 0:
            model_input["ALCDAY5"] = 888
        else:
            d = int(round(min(float(drinks_pw), 7.0)))
            model_input["ALCDAY5"] = 200 + d

    for c in predictor_cols:
        model_input.setdefault(c, np.nan)

    user_inputs = {
        "Age (years)": age,
        "Sex": sex,
        "BMI (kg/m²)": bmi,
        "Smoking status": smoke,
        "Alcohol (drinks/week)": drinks_pw,
        "Any exercise in past month?": exer,
    }

    run_btn = st.button("Run prediction")

    if run_btn:
        results_df = predict_all_conditions(model_input)

        st.subheader("Predicted risks")
        show = results_df.drop(columns=["_disease_code"]).copy()
        show["Predicted risk (0–1)"] = show["Predicted risk (0–1)"].map(lambda x: f"{x:.4f}")
        show["Threshold"] = show["Threshold"].map(lambda x: f"{x:.2f}")
        show["Uncertainty (0–1)"] = show["Uncertainty (0–1)"].map(lambda x: f"{x:.4f}")
        st.dataframe(show, use_container_width=True, hide_index=True)

        st.subheader("Local feature contributions (optional)")
        if not _HAS_SHAP:
            st.info("Install `shap` to enable local explanations.")
        else:
            disease_choice = st.selectbox(
                "Select condition to explain",
                options=list(results_df["_disease_code"].tolist()),
                format_func=lambda d: pretty_disease(d),
                index=0,
            )
            contrib = explain_instance_pipeline(model_input, disease_choice, top_n=20)
            if contrib is None or contrib.empty:
                st.info("SHAP explanation not available for this model/configuration.")
            else:
                st.dataframe(
                    contrib.assign(
                        interpretation=lambda d: np.where(
                            d["shap_value"] >= 0, "Increases predicted risk", "Decreases predicted risk"
                        )
                    ),
                    use_container_width=True,
                    hide_index=True
                )
                st.pyplot(plot_shap_bar(contrib, title=f"Local explanation: {pretty_disease(disease_choice)}"))

        st.subheader("Guidance")
        st.write(rule_based_guidance(user_inputs, results_df))


elif page == "Model evaluation":
    st.title("Model evaluation")

    available = []
    for disease, info in disease_models.items():
        if isinstance(info, dict) and ("y_test" in info and "y_proba" in info):
            available.append(disease)

    if not available:
        st.warning(
            "No stored test predictions found in `disease_models.joblib` (missing `y_test` and/or `y_proba`). "
            "Use precomputed outputs in `tables/` and `figures/`."
        )
        st.markdown("\n⚠️ **{}**".format(DISCLAIMER_TEXT))
    else:
        disease_choice = st.selectbox("Select condition", options=available, format_func=lambda d: pretty_disease(d))
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

        fpr, tpr, _ = roc_curve(y, p)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(fpr, tpr)
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"ROC: {pretty_disease(disease_choice)}")
        st.pyplot(fig)

        prec, rec, _ = precision_recall_curve(y, p)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(rec, prec)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision–Recall: {pretty_disease(disease_choice)}")
        st.pyplot(fig)

        frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(mean_pred, frac_pos, marker="o")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title(f"Calibration: {pretty_disease(disease_choice)}")
        st.pyplot(fig)

        thr = float(optimal_thresholds.get(disease_choice, 0.5))
        yhat = (p >= thr).astype(int)
        cm = confusion_matrix(y, yhat)
        st.write(f"Confusion matrix at threshold {thr:.3f}")
        st.dataframe(pd.DataFrame(cm, index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"]))

        st.markdown("\n⚠️ **{}**".format(DISCLAIMER_TEXT))

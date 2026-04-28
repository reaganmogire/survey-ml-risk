#!/usr/bin/env python3
# coding: utf-8
"""
Streamlit app: Survey-based chronic Disease Prediction (BRFSS 2011–2015).

Key features
- Loads trained artifacts (joblib) from app/artifacts/
- If artifacts are missing, downloads them from GitHub Releases (model-v1)
- Provides interactive disease prediction for multiple outcomes
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
st.set_page_config(page_title="Chronic Disease Prediction", page_icon="🩺", layout="wide")

st.markdown(
    """
<style>
  /* --- Page background --- */
  .stApp {
    background: linear-gradient(180deg, #F5F5F5 0%, #FFFFFF 50%, #F5F5F5 100%);
  }

  /* --- Sidebar --- */
  section[data-testid="stSidebar"] {
    background: #F0F4F8 !important;
    border-right: 1px solid #D0D8E0;
  }

  /* Make ALL text readable */
  html, body, [class*="css"]  {
    color: #1A1A1A !important;
  }

  /* Headings */
  h1, h2, h3, h4, h5, h6 {
    color: #0F3A66 !important;
  }

  /* Paragraph/help text */
  p, li, span, label, small {
    color: #1A1A1A !important;
  }

  /* Cards */
  .card {
    background: #FFFFFF;
    border: 1px solid #E0E0E0;
    border-radius: 8px;
    padding: 16px 18px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
  }

  /* Buttons */
  .stButton>button {
    background: #2E7D32 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.5rem 1.2rem !important;
  }
  .stButton>button:hover {
    background: #1B5E20 !important;
    color: #FFFFFF !important;
  }

  /* --- Widget labels --- */
  div[data-testid="stWidgetLabel"] label,
  div[data-testid="stWidgetLabel"] p,
  .stRadio label, .stRadio p,
  .stSelectbox label, .stSelectbox p,
  .stNumberInput label, .stNumberInput p {
    color: #1A1A1A !important;
    font-weight: 600 !important;
  }

  /* --- Input boxes --- */
  div[data-baseweb="input"] input,
  div[data-baseweb="base-input"] input,
  div[data-baseweb="textarea"] textarea {
    background: #FAFAFA !important;
    color: #1A1A1A !important;
    border: 1px solid #D0D8E0 !important;
    border-radius: 6px !important;
  }

  /* Selectbox (closed state) */
  div[data-baseweb="select"] > div {
    background: #FAFAFA !important;
    color: #1A1A1A !important;
    border: 1px solid #D0D8E0 !important;
    border-radius: 6px !important;
  }

  /* Selectbox dropdown menu - LIGHT BACKGROUND WITH DARK TEXT */
  ul[role="listbox"] {
    background-color: #FFFFFF !important;
    border: 1px solid #D0D8E0 !important;
  }
  ul[role="listbox"] > li {
    color: #1A1A1A !important;
    background-color: #FFFFFF !important;
  }
  ul[role="listbox"] > li:hover {
    background-color: #E8F5E9 !important;
    color: #1A1A1A !important;
  }
  div[data-baseweb="menu"] {
    background-color: #FFFFFF !important;
  }
  div[data-baseweb="menu"] div {
    color: #1A1A1A !important;
    background-color: #FFFFFF !important;
  }
  div[data-baseweb="menu"] div:hover {
    background-color: #E8F5E9 !important;
    color: #1A1A1A !important;
  }
  [role="option"] {
    color: #1A1A1A !important;
    background-color: #FFFFFF !important;
  }
  [role="option"]:hover {
    background-color: #E8F5E9 !important;
    color: #1A1A1A !important;
  }

  /* Radio labels in sidebar */
  section[data-testid="stSidebar"] .stRadio label,
  section[data-testid="stSidebar"] .stRadio p {
    color: #1A1A1A !important;
    font-weight: 600 !important;
  }

  /* DataFrame container */
  div[data-testid="stDataFrame"] {
    background: #FAFAFA !important;
    border-radius: 8px !important;
    border: 1px solid #E0E0E0 !important;
  }

  {
    background-color: #F5F5F5 !important;
    color: #1A1A1A !important;
  }
  pre {
    background-color: #F5F5F5 !important;
    color: #1A1A1A !important;
  }
  pre code {
    background-color: #F5F5F5 !important;
    color: #1A1A1A !important;
  }

  /* JSON/code output blocks */
  div[data-testid="stDataFrame"],
  div[data-testid="stJson"],
  div[data-testid="stCode"],
  pre,
  code,
  .streamlit-expanderContent {
    background-color: #F5F5F5 !important;
    color: #1A1A1A !important;
  }
  div[data-testid="stJson"] * {
    color: #1A1A1A !important;
  }
  pre * {
    color: #1A1A1A !important;
  }

  /* Alerts */
  div[role="alert"] * {
    color: #1A1A1A !important;
  }
</style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 1) Artifact auto-download (GitHub Release: model-v1)
# ============================================================
REPO_OWNER = "reaganmogire"
REPO_NAME  = "survey-ml-risk"
MODEL_TAG  = os.environ.get("MODEL_TAG", "model-v1")

RELEASE_BASE = f"https://github.com/{REPO_OWNER}/{REPO_NAME}/releases/download/{MODEL_TAG}"

APP_DIR      = Path(__file__).resolve().parent
ARTIFACT_DIR = APP_DIR / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

ARTIFACTS = {
    "disease_models.joblib":     f"{RELEASE_BASE}/disease_models.joblib",
    "optimal_thresholds.joblib": f"{RELEASE_BASE}/optimal_thresholds.joblib",
    "predictor_cols.joblib":     f"{RELEASE_BASE}/predictor_cols.joblib",
}

DISCLAIMER_TEXT = (
    "Disclaimer: This tool does not provide a diagnosis and is not a substitute for "
    "professional medical advice. If you have health concerns, consult a qualified clinician."
)


# ============================================================
# 1b) Human-friendly labels
# ============================================================
DISEASE_LABELS = {
    "heart_attack": "Heart attack (myocardial infarction)",
    "coronary_hd":  "Coronary heart disease",
    "stroke":       "Stroke",
    "kidney":       "Chronic kidney disease",
    "depression":   "Depression",
    "diabetes":     "Diabetes",
}

FEATURE_LABELS = {
    "_STATE":   "State",
    "SEX":      "Sex",
    "_AGEG5YR": "Age group",
    "_EDUCAG":  "Education",
    "_INCOMG":  "Household income (USD, per year)",
    "_MRACE1":  "Race / ethnicity",
    "_HISPANC": "Hispanic ethnicity",
    "SMOKE100": "Ever smoked (100 cigarettes)",
    "SMOKDAY2": "Current smoking frequency",
    "ALCDAY5":  "Alcohol use frequency",
    "DRNKANY5": "Any alcohol use (past 30 days)",
    "EXERANY2": "Any exercise (past 30 days)",
    "FRUIT1":   "Fruit intake frequency",
    "VEGETAB1": "Vegetable intake frequency",
    "HLTHPLN1": "Has health insurance",
    "PERSDOC2": "Has a personal doctor",
    "MEDCOST":  "Cost barrier to care (past 12 months)",
    "CHECKUP1": "Time since last routine checkup",
    "BPHIGH4":  "Ever told you have high blood pressure",
    "BPMEDS":   "Currently taking blood pressure medication",
    "TOLDHI2":  "Ever told you have high cholesterol",
    "CHOLCHK":  "Time since last cholesterol check",
    "ASTHMA3":  "Ever told you have asthma",
    "HAVARTH3": "Ever told you have arthritis",
    "GENHLTH":  "Self-rated general health",
    "PHYSHLTH": "Days physical health was not good (past 30 days)",
    "MENTHLTH": "Days mental health was not good (past 30 days)",
    "POORHLTH": "Days poor health limited usual activities (past 30 days)",
    "DIFFWALK": "Difficulty walking or climbing stairs",
    "DECIDE":   "Difficulty concentrating, remembering, or making decisions",
    "WEIGHT2":  "Weight (pounds)",
    "HEIGHT3":  "Height (feet and inches)",
    "_BMI5":    "BMI (×100; BRFSS-coded)",
}

SEX_MAP = {1: "Male", 2: "Female"}

STATE_MAP = {
    1: "Alabama", 2: "Alaska", 4: "Arizona", 5: "Arkansas", 6: "California",
    8: "Colorado", 9: "Connecticut", 10: "Delaware", 12: "Florida", 13: "Georgia",
    15: "Hawaii", 16: "Idaho", 17: "Illinois", 18: "Indiana", 19: "Iowa",
    20: "Kansas", 21: "Kentucky", 22: "Louisiana", 23: "Maine", 24: "Maryland",
    25: "Massachusetts", 26: "Michigan", 27: "Minnesota", 28: "Mississippi", 29: "Missouri",
    30: "Montana", 31: "Nebraska", 32: "Nevada", 33: "New Hampshire", 34: "New Jersey",
    35: "New Mexico", 36: "New York", 37: "North Carolina", 38: "North Dakota", 39: "Ohio",
    40: "Oklahoma", 41: "Oregon", 42: "Pennsylvania", 44: "Rhode Island", 45: "South Carolina",
    46: "South Dakota", 47: "Tennessee", 48: "Texas", 49: "Utah", 50: "Vermont",
    51: "Virginia", 53: "Washington", 54: "West Virginia", 55: "Wisconsin", 56: "Wyoming",
}

CHECKUP1_MAP = {
    1: "Within past year",
    2: "Within past 2 years",
    3: "Within past 5 years",
    4: "5 or more years ago",
    7: "Never had a checkup",
    9: "Don't know / Refused",
}

SMOKDAY2_MAP = {
    1: "Every day",
    2: "Some days",
    3: "Not at all",
    9: "Don't know / Refused",
}

TOLDHI2_MAP = {
    1: "Yes",
    2: "No",
    7: "Borderline / Not sure",
    9: "Don't know / Refused",
}

EDUCAG_MAP = {
    1: "Did not graduate high school",
    2: "Graduated high school",
    3: "Attended college or technical school",
    4: "Graduated from college or technical school",
    9: "Don't know / Refused",
}

MRACE1_MAP = {
    1: "White only, non-Hispanic",
    2: "Black / African American only, non-Hispanic",
    3: "American Indian or Alaskan Native only, non-Hispanic",
    4: "Asian only, non-Hispanic",
    5: "Native Hawaiian / Other Pacific Islander only, non-Hispanic",
    6: "Other race only, non-Hispanic",
    7: "Multiracial, non-Hispanic",
    8: "Hispanic (any race)",
    9: "Don't know / Refused",
}

HISPANC_MAP = {
    1: "Yes — Hispanic or Latino",
    2: "No — not Hispanic or Latino",
    9: "Don't know / Refused",
}

INCOMG_MAP = {
    1: "Less than $15,000 per year",
    2: "$15,000 to less than $25,000 per year",
    3: "$25,000 to less than $35,000 per year",
    4: "$35,000 to less than $50,000 per year",
    5: "$50,000 or more per year",
    9: "Don't know / Refused",
}

PERSDOC2_MAP = {
    1: "Yes — one personal doctor or health care provider",
    2: "Yes — more than one",
    3: "No",
    7: "Don't know / Refused",
}

BPMEDS_MAP = {
    1: "Yes",
    2: "No",
    4: "Not applicable (no high BP diagnosis)",
    9: "Don't know / Refused",
}

CHOLCHK_MAP = {
    1: "Within the past year (less than 12 months ago)",
    2: "Within the past 2 years (1–2 years ago)",
    3: "Within the past 5 years (2–5 years ago)",
    4: "5 or more years ago",
    8: "Never had cholesterol checked",
    9: "Don't know / Refused",
}

FRUIT_MAP = {
    555: "Never",
    300: "Less than once a month",
    301: "About once a month",
    302: "2–3 times a month",
    201: "About once a week",
    203: "2–4 times a week",
    205: "5–6 times a week",
    101: "Once a day",
    102: "2 or more times a day",
    777: "Don't know / Refused",
}


def pretty_disease(d: str) -> str:
    return DISEASE_LABELS.get(d, d.replace("_", " ").title())


def pretty_feature_name(raw: str) -> str:
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
                    level_i   = int(float(level))
                    level_label = SEX_MAP.get(level_i, str(level_i))
                    return f"{var_label} = {level_label}"
                except Exception:
                    return f"{var_label} = {level}"
            return f"{var_label} = {level}"
        return FEATURE_LABELS.get(base, base)

    s2     = s.replace("Num__", "").replace("Cat__", "")
    s2     = s2.replace("num__", "").replace("cat__", "")
    s2     = s2.replace("__", "")
    s2     = re.sub(r"[^A-Za-z0-9_]", "", s2)
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
        total      = int(r.headers.get("content-length", 0))
        downloaded = 0

        prog = st.progress(0.0)
        msg  = st.empty()

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
    disease_models     = joblib.load(ARTIFACT_DIR / "disease_models.joblib")
    optimal_thresholds = joblib.load(ARTIFACT_DIR / "optimal_thresholds.joblib")
    predictor_cols     = joblib.load(ARTIFACT_DIR / "predictor_cols.joblib")
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
        thr   = float(optimal_thresholds.get(disease, 0.5))

        proba = float(model.predict_proba(row)[0, 1])
        label = "Likely present" if proba >= thr else "Likely absent / lower probability"
        records.append(
            {
                "Condition":                pretty_disease(disease),
                "Predicted probability (0–1)": proba,
                "Threshold":               thr,
                "Prediction category":      label,
                "Uncertainty (0–1)":       uncertainty_from_proba(proba),
                "_disease_code":           disease,
            }
        )

    df = (
        pd.DataFrame(records)
        .sort_values("Predicted probability (0–1)", ascending=False)
        .reset_index(drop=True)
    )
    return df


def rule_based_guidance(user_inputs: Dict[str, Any], results_df: pd.DataFrame) -> str:
    lines = []
    flagged = results_df.loc[
        results_df["Prediction category"] == "Likely present", "Condition"
    ].tolist()

    if flagged:
        lines.append(
            "**Conditions flagged as likely present (model-based):** "
            + ", ".join(flagged) + "."
        )
    else:
        lines.append(
            "**Model-based result:** No conditions flagged as likely present "
            "at the stored thresholds."
        )

    bmi = _safe_float(user_inputs.get("BMI (kg/m²)"))
    if bmi is not None:
        if bmi >= 30:
            lines.append(
                "- BMI suggests obesity. Gradual weight reduction (diet quality + "
                "regular activity) can reduce cardiometabolic risk."
            )
        elif bmi >= 25:
            lines.append("- BMI suggests overweight. Small sustained changes can improve risk.")

    smoke = str(user_inputs.get("Smoking status", "")).lower()
    if "current" in smoke:
        lines.append(
            "- Current smoking increases cardiovascular and overall risk. "
            "Consider evidence-based cessation support."
        )

    alc = _safe_float(user_inputs.get("Alcohol (drinks/week)"))
    if alc is not None and alc >= 14:
        lines.append(
            "- Reported alcohol intake is relatively high. Reducing intake can lower "
            "blood pressure and improve cardiometabolic health."
        )

    phys = str(user_inputs.get("Any exercise in past month?", "")).lower()
    if phys == "no":
        lines.append(
            "- Increasing physical activity (as medically appropriate) supports "
            "cardiometabolic, renal, and mental health."
        )

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
            explainer    = shap.TreeExplainer(clf)
            out[disease] = {"preprocessor": pre, "explainer": explainer}
        except Exception:
            out[disease] = None

    return out


tree_explainers = build_tree_explainers()


def explain_instance_pipeline(
    model_input: Dict[str, Any],
    disease: str,
    top_n: int = 20,
) -> Optional[pd.DataFrame]:
    if not _HAS_SHAP or tree_explainers is None:
        return None

    bundle = tree_explainers.get(disease)
    if bundle is None:
        return None

    pre      = bundle["preprocessor"]
    explainer = bundle["explainer"]

    x_raw = pd.DataFrame([model_input]).reindex(columns=predictor_cols, fill_value=np.nan)

    try:
        X_t           = pre.transform(x_raw)
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
    df["abs"]     = df["shap_value"].abs()
    df = (
        df.sort_values("abs", ascending=False)
        .head(top_n)
        .drop(columns=["abs"])
        .reset_index(drop=True)
    )
    return df[["Feature", "shap_value"]]


def plot_shap_bar(df: pd.DataFrame, title: str) -> plt.Figure:
    d = df.copy().sort_values("shap_value", ascending=True)
    y = d["Feature"].tolist()
    x = d["shap_value"].values

    colors = np.where(x >= 0, "#D9534F", "#2F80ED")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(y, x, color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("SHAP value (impact on predicted probability)")
    ax.grid(axis="x", linestyle="--", alpha=0.25)

    ax.text(0.02, -0.15, "← Decreases predicted probability (SHAP < 0)",
            transform=ax.transAxes, ha="left", va="top", fontsize=11, color="#2F80ED")
    ax.text(0.98, -0.15, "Increases predicted probability (SHAP > 0) →",
            transform=ax.transAxes, ha="right", va="top", fontsize=11, color="#D9534F")

    fig.tight_layout()
    return fig


# ============================================================
# 4) Streamlit UI
# ============================================================
st.sidebar.title("Survey-ML: Disease Prediction")
page = st.sidebar.radio("Navigate", ["Disease prediction", "Model evaluation", "About"])
st.sidebar.markdown("---")
st.sidebar.caption("Research/demonstration only; not yet for clinical use.")


if page == "About":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.title("About this app")
    st.markdown(
        """
This repository provides an interpretable machine-learning framework for chronic disease
prediction using population survey data (BRFSS 2011–2015).

**Key design goals**
- Scalable disease prediction without EHRs, biomarkers, or laboratory data
- Interpretable feature contributions (SHAP, optional)
- Reproducible, publication-oriented outputs (tables/figures)
        """
    )
    st.warning(DISCLAIMER_TEXT, icon="⚠️")
    st.markdown("### Loaded artifact summary")
    st.write(f"Model tag: `{MODEL_TAG}`")
    st.write(f"Artifacts directory: `{ARTIFACT_DIR}`")
    st.write("Predictor columns loaded:", len(predictor_cols))
    st.write("Conditions loaded:")
    for condition in disease_models.keys():
        st.write(f"- {pretty_disease(condition)}")
    st.markdown("</div>", unsafe_allow_html=True)


elif page == "Disease prediction":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.title("Chronic disease prediction (survey-based ML)")
    st.write(
        "Enter inputs to generate predicted probabilities for each condition. "
        "On first run, the app downloads model artifacts from GitHub Releases."
    )
    st.warning(DISCLAIMER_TEXT, icon="⚠️")
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    cols    = [col1, col2, col3]
    col_idx = 0

    model_input: Dict[str, Any] = {}
    user_inputs                  = {}

    def age_to_ageg5yr(a: int) -> int:
        a    = max(18, min(int(a), 99))
        bins = [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
        for i, upper in enumerate(bins, start=1):
            if a < upper:
                return i
        return 13

    for col_name in predictor_cols:
        col           = cols[col_idx % 3]
        feature_label = pretty_feature_name(col_name)

        with col:
            # ── Demographics ──────────────────────────────────────
            if col_name == "_AGEG5YR":
                age = st.number_input("Age (years)", min_value=18, max_value=99, value=35, step=1)
                model_input["_AGEG5YR"] = age_to_ageg5yr(age)
                user_inputs["Age (years)"] = age

            elif col_name == "SEX":
                sex = st.selectbox("Sex", ["Male", "Female"])
                model_input["SEX"] = 1 if sex == "Male" else 2
                user_inputs["Sex"] = sex

            elif col_name == "_STATE":
                state_name = st.selectbox("State", sorted(STATE_MAP.values()))
                state_code = [k for k, v in STATE_MAP.items() if v == state_name][0]
                model_input["_STATE"] = state_code
                user_inputs["State"]  = state_name

            elif col_name == "_EDUCAG":
                edu_label = st.selectbox(
                    "Highest education level completed",
                    list(EDUCAG_MAP.values()),
                    help="BRFSS education category based on the highest grade or year of school completed.",
                )
                edu_code = [k for k, v in EDUCAG_MAP.items() if v == edu_label][0]
                model_input["_EDUCAG"]   = edu_code
                user_inputs["Education"] = edu_label

            elif col_name == "_MRACE1":
                race_label = st.selectbox("Race / ethnicity", list(MRACE1_MAP.values()))
                race_code  = [k for k, v in MRACE1_MAP.items() if v == race_label][0]
                model_input["_MRACE1"]         = race_code
                user_inputs["Race / ethnicity"] = race_label

            elif col_name == "_HISPANC":
                hisp_label = st.selectbox("Hispanic or Latino ethnicity", list(HISPANC_MAP.values()))
                hisp_code  = [k for k, v in HISPANC_MAP.items() if v == hisp_label][0]
                model_input["_HISPANC"]            = hisp_code
                user_inputs["Hispanic ethnicity"]   = hisp_label

            elif col_name == "_INCOMG":
                inc_label = st.selectbox(
                    "Household income (USD, per year)",
                    list(INCOMG_MAP.values()),
                    help="Annual household income bracket (BRFSS 2011–2015)",
                )
                inc_code = [k for k, v in INCOMG_MAP.items() if v == inc_label][0]
                model_input["_INCOMG"]          = inc_code
                user_inputs["Household income"]  = inc_label

            # ── Anthropometrics ───────────────────────────────────
            elif col_name == "_BMI5":
                bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=70.0, value=27.0, step=0.1)
                model_input["_BMI5"]     = int(round(float(bmi) * 100))
                user_inputs["BMI (kg/m²)"] = bmi

            elif col_name == "WEIGHT2":
                weight_lbs = st.number_input(
                    "Weight (pounds)",
                    min_value=50, max_value=700, value=160, step=1,
                    help="Enter your weight in pounds (1 kg ≈ 2.205 lbs). BRFSS records weight in pounds.",
                )
                model_input["WEIGHT2"]       = int(weight_lbs)
                user_inputs["Weight (lbs)"]  = weight_lbs

            elif col_name == "HEIGHT3":
                st.markdown("**Height**")
                h_col1, h_col2 = st.columns(2)
                with h_col1:
                    h_ft = st.number_input("Feet",   min_value=4, max_value=7, value=5, step=1)
                with h_col2:
                    h_in = st.number_input("Inches", min_value=0, max_value=11, value=7, step=1)
                model_input["HEIGHT3"]  = int(h_ft) * 100 + int(h_in)
                user_inputs["Height"]   = f"{int(h_ft)}′{int(h_in)}″"

            # ── General health ────────────────────────────────────
            elif col_name == "GENHLTH":
                genhlth = st.selectbox(
                    "General health (self-rated)",
                    ["Excellent", "Very good", "Good", "Fair", "Poor"],
                )
                model_input["GENHLTH"] = {
                    "Excellent": 1, "Very good": 2, "Good": 3, "Fair": 4, "Poor": 5
                }[genhlth]

            elif col_name == "PHYSHLTH":
                phys_days = st.number_input(
                    "Days physical health was NOT good (past 30 days)",
                    min_value=0, max_value=30, value=0, step=1,
                    help="Enter 0 if none.",
                )
                model_input["PHYSHLTH"] = 88 if int(phys_days) == 0 else int(phys_days)
                user_inputs["Physically unhealthy days (past 30)"] = int(phys_days)

            elif col_name == "MENTHLTH":
                ment_days = st.number_input(
                    "Days mental health was NOT good (past 30 days)",
                    min_value=0, max_value=30, value=0, step=1,
                    help="Enter 0 if none.",
                )
                model_input["MENTHLTH"] = 88 if int(ment_days) == 0 else int(ment_days)
                user_inputs["Mentally unhealthy days (past 30)"] = int(ment_days)

            elif col_name == "POORHLTH":
                poor_days = st.number_input(
                    "Days poor health limited usual activities (past 30 days)",
                    min_value=0, max_value=30, value=0, step=1,
                    help="Enter 0 if none.",
                )
                model_input["POORHLTH"] = 88 if int(poor_days) == 0 else int(poor_days)
                user_inputs["Activity-limiting poor health days (past 30)"] = int(poor_days)

            elif col_name == "DIFFWALK":
                diffwalk = st.selectbox(
                    "Difficulty walking or climbing stairs?",
                    ["No", "Yes"],
                    help="Do you have serious difficulty walking or climbing stairs?",
                )
                model_input["DIFFWALK"] = 1 if diffwalk == "Yes" else 2

            elif col_name == "DECIDE":
                decide = st.selectbox(
                    "Difficulty concentrating, remembering, or making decisions?",
                    ["No", "Yes"],
                    help="Because of a physical, mental, or emotional condition.",
                )
                model_input["DECIDE"] = 1 if decide == "Yes" else 2

            # ── Cardiovascular risk factors ───────────────────────
            elif col_name == "BPHIGH4":
                htn = st.selectbox(
                    "Ever told you have high blood pressure?",
                    ["No", "Yes", "Borderline / Pre-hypertension", "Don't know / Refused"],
                )
                model_input["BPHIGH4"] = {
                    "Yes": 1, "No": 2,
                    "Borderline / Pre-hypertension": 3,
                    "Don't know / Refused": 9,
                }[htn]

            elif col_name == "BPMEDS":
                bp_med_label = st.selectbox(
                    "Currently taking blood pressure medication?",
                    list(BPMEDS_MAP.values()),
                    help="Are you currently taking medicine for high blood pressure?",
                )
                bp_med_code        = [k for k, v in BPMEDS_MAP.items() if v == bp_med_label][0]
                model_input["BPMEDS"] = bp_med_code

            elif col_name == "TOLDHI2":
                cholesterol = st.selectbox(
                    "Ever told you have high blood cholesterol?",
                    list(TOLDHI2_MAP.values()),
                )
                cholesterol_code       = [k for k, v in TOLDHI2_MAP.items() if v == cholesterol][0]
                model_input["TOLDHI2"] = cholesterol_code
                user_inputs["Told high cholesterol"] = cholesterol

            elif col_name == "CHOLCHK":
                chol_label = st.selectbox(
                    "How long since last cholesterol check?",
                    list(CHOLCHK_MAP.values()),
                    help="Blood cholesterol check by a doctor, nurse, or other health professional.",
                )
                chol_code          = [k for k, v in CHOLCHK_MAP.items() if v == chol_label][0]
                model_input["CHOLCHK"] = chol_code

            # ── Health care access ────────────────────────────────
            elif col_name == "HLTHPLN1":
                ins = st.selectbox(
                    "Do you have any kind of health care coverage?",
                    ["Yes", "No", "Don't know / Refused"],
                    help="Includes health insurance, prepaid plans, government plans (Medicare, Medicaid), etc.",
                )
                model_input["HLTHPLN1"] = {"Yes": 1, "No": 2, "Don't know / Refused": 9}[ins]

            elif col_name == "PERSDOC2":
                doc_label = st.selectbox(
                    "Do you have one or more personal doctors?",
                    list(PERSDOC2_MAP.values()),
                    help="Personal doctor or health care provider you see regularly.",
                )
                doc_code           = [k for k, v in PERSDOC2_MAP.items() if v == doc_label][0]
                model_input["PERSDOC2"] = doc_code

            elif col_name == "MEDCOST":
                cost = st.selectbox(
                    "In the past 12 months, was there a time you needed to see a doctor but could not due to cost?",
                    ["No", "Yes", "Don't know / Refused"],
                )
                model_input["MEDCOST"] = {"Yes": 1, "No": 2, "Don't know / Refused": 9}[cost]

            elif col_name == "CHECKUP1":
                checkup      = st.selectbox("How long since your last routine checkup?", list(CHECKUP1_MAP.values()))
                checkup_code = [k for k, v in CHECKUP1_MAP.items() if v == checkup][0]
                model_input["CHECKUP1"] = checkup_code

            # ── Lifestyle ─────────────────────────────────────────
            elif col_name == "SMOKE100":
                smoke = st.selectbox(
                    "Smoking status",
                    ["Never", "Former", "Current"],
                    help="'Never' = fewer than 100 cigarettes lifetime; 'Former' = smoked 100+ but not now; 'Current' = still smokes.",
                )
                model_input["SMOKE100"]       = 2 if smoke == "Never" else 1
                user_inputs["Smoking status"] = smoke

            elif col_name == "SMOKDAY2":
                smokday      = st.selectbox("Current smoking frequency", list(SMOKDAY2_MAP.values()))
                smokday_code = [k for k, v in SMOKDAY2_MAP.items() if v == smokday][0]
                model_input["SMOKDAY2"]                   = smokday_code
                user_inputs["Current smoking frequency"]  = smokday

            elif col_name == "DRNKANY5":
                drnk = st.selectbox(
                    "Any alcohol use in the past 30 days?",
                    ["No", "Yes", "Don't know / Refused"],
                    help="At least one alcoholic drink in the past 30 days.",
                )
                model_input["DRNKANY5"] = {"Yes": 1, "No": 2, "Don't know / Refused": 9}[drnk]

            elif col_name == "ALCDAY5":
                drinks_pw = st.number_input(
                    "Alcohol (drinks/week)", min_value=0.0, max_value=70.0, value=0.0, step=1.0
                )
                if drinks_pw <= 0:
                    model_input["ALCDAY5"] = 888
                else:
                    d = int(round(min(float(drinks_pw), 7.0)))
                    model_input["ALCDAY5"] = 200 + d
                user_inputs["Alcohol (drinks/week)"] = drinks_pw

            elif col_name == "EXERANY2":
                exer = st.selectbox(
                    "Any physical activity or exercise in the past 30 days?",
                    ["No", "Yes", "Don't know / Refused"],
                    help="Other than your regular job.",
                )
                model_input["EXERANY2"]                  = {"Yes": 1, "No": 2, "Don't know / Refused": 9}[exer]
                user_inputs["Any exercise in past month?"] = exer

            elif col_name == "FRUIT1":
                fruit_label = st.selectbox(
                    "How often do you eat fruit?",
                    list(FRUIT_MAP.values()),
                    help="Not counting juice. Count fresh, frozen, canned, or dried fruit.",
                )
                fruit_code         = [k for k, v in FRUIT_MAP.items() if v == fruit_label][0]
                model_input["FRUIT1"] = fruit_code

            elif col_name == "VEGETAB1":
                veg_label = st.selectbox(
                    "How often do you eat vegetables?",
                    list(FRUIT_MAP.values()),
                    help="Not counting juice or potatoes. Count fresh, frozen, canned, or dried vegetables.",
                )
                veg_code             = [k for k, v in FRUIT_MAP.items() if v == veg_label][0]
                model_input["VEGETAB1"] = veg_code

            # ── Chronic conditions ────────────────────────────────
            elif col_name == "ASTHMA3":
                asthma = st.selectbox(
                    "Ever told by a doctor, nurse, or health professional that you have asthma?",
                    ["No", "Yes", "Don't know / Refused"],
                )
                model_input["ASTHMA3"] = {"Yes": 1, "No": 2, "Don't know / Refused": 9}[asthma]

            elif col_name == "HAVARTH3":
                arth = st.selectbox(
                    "Ever told by a doctor, nurse, or health professional that you have some form of arthritis, "
                    "rheumatoid arthritis, gout, lupus, or fibromyalgia?",
                    ["No", "Yes", "Don't know / Refused"],
                )
                model_input["HAVARTH3"] = {"Yes": 1, "No": 2, "Don't know / Refused": 9}[arth]

            else:
                val = st.number_input(feature_label, value=0.0, step=1.0)
                model_input[col_name] = val

        col_idx += 1

    for c in predictor_cols:
        model_input.setdefault(c, np.nan)

    run_btn = st.button("Run prediction")

    if run_btn:
        st.session_state["results_df"]         = predict_all_conditions(model_input)
        st.session_state["model_input_cache"]  = model_input.copy()
        st.session_state["user_inputs_cache"]  = user_inputs.copy()

    if "results_df" in st.session_state:
        results_df        = st.session_state["results_df"]
        model_input_cache = st.session_state["model_input_cache"]
        user_inputs_cache = st.session_state["user_inputs_cache"]

        st.subheader("Predicted probabilities")
        show = results_df.drop(columns=["_disease_code"]).copy()
        show["Predicted probability (0–1)"] = show["Predicted probability (0–1)"].map(lambda x: f"{x:.4f}")
        show["Threshold"]                   = show["Threshold"].map(lambda x: f"{x:.2f}")
        show["Uncertainty (0–1)"]           = show["Uncertainty (0–1)"].map(lambda x: f"{x:.4f}")
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
            contrib = explain_instance_pipeline(model_input_cache, disease_choice, top_n=20)
            if contrib is None or contrib.empty:
                st.info("SHAP explanation not available for this model/configuration.")
            else:
                st.dataframe(
                    contrib.assign(
                        interpretation=lambda d: np.where(
                            d["shap_value"] >= 0,
                            "Increases predicted probability",
                            "Decreases predicted probability",
                        )
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
                st.pyplot(
                    plot_shap_bar(
                        contrib,
                        title=f"Local explanation: {pretty_disease(disease_choice)}",
                    )
                )

        st.subheader("Guidance")
        st.write(rule_based_guidance(user_inputs_cache, results_df))


elif page == "Model evaluation":
    st.title("Model evaluation")

    available = []
    for disease, info in disease_models.items():
        if isinstance(info, dict) and ("y_test" in info and "y_proba" in info):
            available.append(disease)

    if not available:
        st.warning(
            "No stored test predictions found in `disease_models.joblib` "
            "(missing `y_test` and/or `y_proba`). "
            "Use precomputed outputs in `tables/` and `figures/`."
        )
        st.markdown("\n⚠️ **{}**".format(DISCLAIMER_TEXT))
    else:
        disease_choice = st.selectbox(
            "Select condition",
            options=available,
            format_func=lambda d: pretty_disease(d),
        )
        info = disease_models[disease_choice]
        y    = np.asarray(info["y_test"]).astype(int)
        p    = np.asarray(info["y_proba"]).astype(float)

        auroc  = roc_auc_score(y, p)
        ap     = average_precision_score(y, p)
        brier  = brier_score_loss(y, p)

        c1, c2, c3 = st.columns(3)
        c1.metric("AUROC",  f"{auroc:.3f}")
        c2.metric("PR-AUC", f"{ap:.3f}")
        c3.metric("Brier",  f"{brier:.3f}")

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

        thr  = float(optimal_thresholds.get(disease_choice, 0.5))
        yhat = (p >= thr).astype(int)
        cm   = confusion_matrix(y, yhat, labels=[0, 1])
        st.write(f"Confusion matrix at threshold {thr:.3f}")
        st.dataframe(
            pd.DataFrame(cm, index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"])
        )

        st.markdown("\n⚠️ **{}**".format(DISCLAIMER_TEXT))

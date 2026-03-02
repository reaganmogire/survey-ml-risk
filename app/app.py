k#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit app for survey-based chronic disease risk prediction (research/demo).

- Loads pre-trained model artifacts from app/artifacts/
- Produces per-condition predicted risk, threshold-based risk category, and uncertainty proxy
- Provides optional local explanation (SHAP) with intuitive labels and a clear sign convention:
    SHAP > 0  => increases predicted risk
    SHAP < 0  => decreases predicted risk
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt

# Optional SHAP
try:
    import shap  # type: ignore
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False


# =============================================================================
# Page config + bright "medical" theme
# =============================================================================
st.set_page_config(
    page_title="Survey ML Risk (Research Demo)",
    page_icon="🩺",
    layout="wide",
)

st.markdown(
    """
    <style>
      /* Overall background */
      .stApp {
        background: linear-gradient(180deg, #F7FBFF 0%, #FFFFFF 35%, #F5FAFF 100%);
        color: #0B1F35;
      }

      /* Reduce dark mode feel */
      section[data-testid="stSidebar"] {
        background: #EAF5FF;
        border-right: 1px solid rgba(11,31,53,0.08);
      }

      /* Headings */
      h1, h2, h3, h4 {
        color: #0B3B66;
      }

      /* Cards */
      .card {
        background: #FFFFFF;
        border: 1px solid rgba(11,31,53,0.10);
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 6px 20px rgba(11,31,53,0.06);
      }

      /* Buttons */
      .stButton>button {
        background: #2F80ED;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.55rem 1.0rem;
      }
      .stButton>button:hover {
        background: #1F6FDC;
        color: white;
      }

      /* Dataframe container */
      div[data-testid="stDataFrame"] {
        background: #FFFFFF;
        border-radius: 12px;
        border: 1px solid rgba(11,31,53,0.10);
      }

      /* Info / warning blocks */
      .stAlert {
        border-radius: 12px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Constants / Labels
# =============================================================================
REPO_ROOT = Path(__file__).resolve().parents[1]  # app/app.py -> repo root
ARTIFACT_DIR = Path(os.environ.get("ARTIFACT_DIR", REPO_ROOT / "app" / "artifacts"))

ART_DISEASE_MODELS = ARTIFACT_DIR / "disease_models.joblib"
ART_THRESHOLDS = ARTIFACT_DIR / "optimal_thresholds.joblib"
ART_PREDICTORS = ARTIFACT_DIR / "predictor_cols.joblib"

DISEASE_LABELS = {
    "heart_attack": "Heart attack (myocardial infarction)",
    "coronary_hd": "Coronary heart disease",
    "stroke": "Stroke",
    "kidney": "Chronic kidney disease",
    "depression": "Depression",
    "diabetes": "Diabetes",
}

# BRFSS feature labels (intuitive display)
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

# Compact mappings for common categorical fields (for nicer inputs)
SEX_MAP = {1: "Male", 2: "Female"}
YESNO_MAP = {1: "Yes", 2: "No"}

AGEG5YR_MAP = {
    1: "18–24", 2: "25–29", 3: "30–34", 4: "35–39", 5: "40–44",
    6: "45–49", 7: "50–54", 8: "55–59", 9: "60–64", 10: "65–69",
    11: "70–74", 12: "75–79", 13: "80+",
}

EDUCAG_MAP = {
    1: "< High school",
    2: "High school graduate",
    3: "Some college",
    4: "College graduate",
}

INCOMG_MAP = {
    1: "<$15k",
    2: "$15–25k",
    3: "$25–35k",
    4: "$35–50k",
    5: "$50–75k",
    6: "$75k+",
}

GENHLTH_MAP = {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"}


# =============================================================================
# Helpers
# =============================================================================
def friendly_feature_name(raw: str) -> str:
    """
    Convert model feature names (including transformed names from preprocessing)
    into more intuitive labels for display.

    Examples:
      num__MENTHLTH -> Mentally unhealthy days
      cat__SEX_1.0  -> Sex = Male
    """
    s = str(raw)

    # Strip sklearn prefixes
    if s.startswith("num__"):
        base = s[len("num__"):]
        return FEATURE_LABELS.get(base, base)

    if s.startswith("cat__"):
        base = s[len("cat__"):]
        # Often: <VAR>_<LEVEL>
        # Try to split last underscore as level
        if "_" in base:
            var, level = base.rsplit("_", 1)
            var_label = FEATURE_LABELS.get(var, var)
            # Special handling for SEX
            if var == "SEX":
                try:
                    level_i = int(float(level))
                    level_label = SEX_MAP.get(level_i, level)
                    return f"{var_label} = {level_label}"
                except Exception:
                    return f"{var_label} = {level}"
            # Generic
            return f"{var_label} = {level}"
        return FEATURE_LABELS.get(base, base)

    # Fall back to raw mapping when possible
    # Also handle strings like "Num__MENTHLTH" from older pipelines
    s2 = s.replace("Num__", "").replace("Cat__", "").replace("num__", "").replace("cat__", "")
    s2 = s2.replace("__", "")
    return FEATURE_LABELS.get(s2, s2)


def disease_label(code: str) -> str:
    return DISEASE_LABELS.get(code, code.replace("_", " ").title())


def uncertainty_from_proba(p: float) -> float:
    """
    Uncertainty proxy in [0,1], highest at p=0.5, lowest near 0 or 1.
    """
    return float(1.0 - abs(p - 0.5) * 2.0)


@st.cache_resource(show_spinner=False)
def load_artifacts() -> Tuple[Dict[str, Any], Dict[str, float], list]:
    """
    Load model artifacts saved by src/model_ai.py
    """
    if not ART_DISEASE_MODELS.exists():
        raise FileNotFoundError(f"Missing model artifact: {ART_DISEASE_MODELS}")
    if not ART_THRESHOLDS.exists():
        raise FileNotFoundError(f"Missing thresholds artifact: {ART_THRESHOLDS}")
    if not ART_PREDICTORS.exists():
        raise FileNotFoundError(f"Missing predictor columns artifact: {ART_PREDICTORS}")

    disease_models = joblib.load(ART_DISEASE_MODELS)
    thresholds = joblib.load(ART_THRESHOLDS)
    predictor_cols = joblib.load(ART_PREDICTORS)

    # Ensure disease_models dict has expected structure
    if not isinstance(disease_models, dict):
        raise ValueError("disease_models.joblib did not load as a dict")

    return disease_models, thresholds, predictor_cols


def predict_all(input_row: Dict[str, Any],
                disease_models: Dict[str, Any],
                thresholds: Dict[str, float],
                predictor_cols: list) -> pd.DataFrame:
    """
    Predict across all diseases; return a tidy DataFrame for display.
    """
    X = pd.DataFrame([input_row]).reindex(columns=predictor_cols, fill_value=np.nan)

    rows = []
    for dis, info in disease_models.items():
        model = info["model"] if isinstance(info, dict) and "model" in info else info
        p = float(model.predict_proba(X)[0, 1])
        thr = float(thresholds.get(dis, 0.5))
        unc = uncertainty_from_proba(p)
        label = "Higher risk" if p >= thr else "Lower / moderate risk"

        rows.append({
            "Condition": disease_label(dis),
            "Predicted risk (0–1)": p,
            "Threshold": thr,
            "Risk category": label,
            "Uncertainty (0–1)": unc,
            "_disease_code": dis,
        })

    df = pd.DataFrame(rows).sort_values("Predicted risk (0–1)", ascending=False).reset_index(drop=True)
    return df


def shap_local_explanation(input_row: Dict[str, Any],
                           disease: str,
                           disease_models: Dict[str, Any],
                           predictor_cols: list,
                           top_n: int = 15) -> Optional[pd.DataFrame]:
    """
    Compute SHAP contributions for one selected disease model.
    Returns a DataFrame with intuitive feature labels.

    Requires SHAP and a pipeline with:
      - model.named_steps["preprocessor"]
      - model.named_steps["clf"]  (tree-based)
    """
    if not HAS_SHAP:
        return None

    info = disease_models[disease]
    model = info["model"] if isinstance(info, dict) and "model" in info else info

    # Expect sklearn Pipeline
    if not hasattr(model, "named_steps"):
        return None
    if "preprocessor" not in model.named_steps or "clf" not in model.named_steps:
        return None

    preprocessor = model.named_steps["preprocessor"]
    clf = model.named_steps["clf"]

    X = pd.DataFrame([input_row]).reindex(columns=predictor_cols, fill_value=np.nan)
    Xt = preprocessor.transform(X)

    try:
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(Xt)
    except Exception:
        return None

    # Binary classification: shap_values could be list [neg, pos] or array
    if isinstance(shap_values, list):
        # Use positive class
        vals = shap_values[1][0]
    else:
        vals = shap_values[0] if shap_values.ndim > 1 else shap_values

    try:
        feat_names = preprocessor.get_feature_names_out()
    except Exception:
        # fallback: generic names
        feat_names = np.array([f"feature_{i}" for i in range(len(vals))])

    df = pd.DataFrame({
        "feature_raw": feat_names,
        "feature": [friendly_feature_name(f) for f in feat_names],
        "shap_value": vals.astype(float),
    })
    df["abs_shap"] = df["shap_value"].abs()
    df = df.sort_values("abs_shap", ascending=False).head(top_n).drop(columns=["abs_shap"]).reset_index(drop=True)
    return df


def plot_shap_bar(contrib: pd.DataFrame, title: str) -> plt.Figure:
    """
    Horizontal bar plot with sign convention explicitly labeled:
      +SHAP increases risk, -SHAP decreases risk
    """
    df = contrib.copy()

    # Order: smallest at top -> largest at bottom looks nice in barh
    df = df.iloc[::-1].reset_index(drop=True)

    y = df["feature"].tolist()
    x = df["shap_value"].values

    # Color by sign (increase vs decrease)
    colors = np.where(x >= 0, "#D9534F", "#2F80ED")  # red-ish for increase, blue for decrease

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(y, x, color=colors, edgecolor="none")
    ax.axvline(0, color="#0B1F35", linewidth=1)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("SHAP value (impact on predicted risk)", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)

    # Make the sign convention explicit on the plot
    # Place directional annotations near the top
    xlim = ax.get_xlim()
    xr = max(abs(xlim[0]), abs(xlim[1]))
    ax.set_xlim(-xr, xr)

    ax.text(0.98, 1.02, "→ Increase risk (SHAP > 0)", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=11, color="#D9534F")
    ax.text(0.02, 1.02, "Decrease risk (SHAP < 0) ←", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=11, color="#2F80ED")

    ax.grid(axis="x", linestyle="--", alpha=0.25)
    fig.tight_layout()
    return fig


# =============================================================================
# UI
# =============================================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.title("🩺 Survey-based Chronic Disease Risk Prediction (Research Demo)")
st.write(
    "This demo uses survey-derived inputs to generate model-based risk estimates for multiple chronic diseases, "
    "with optional local explanations."
)
st.markdown("</div>", unsafe_allow_html=True)

# Disclaimer (updated as requested)
st.warning(
    "Disclaimer: This tool does not provide a diagnosis and is not a substitute for professional medical advice. "
    "If you have health concerns, consult a qualified clinician.",
    icon="⚠️",
)

# Load artifacts
with st.spinner("Loading model artifacts..."):
    try:
        disease_models, thresholds, predictor_cols = load_artifacts()
    except Exception as e:
        st.error(
            f"Could not load model artifacts.\n\n"
            f"Expected files under: `{ARTIFACT_DIR}`\n"
            f"- disease_models.joblib\n"
            f"- optimal_thresholds.joblib\n"
            f"- predictor_cols.joblib\n\n"
            f"Error: {e}"
        )
        st.stop()

# Sidebar inputs
st.sidebar.header("Input survey fields")

# Minimal “friendly” inputs for key variables.
# For advanced/repro: allow raw-code editing too.
use_advanced = st.sidebar.toggle("Advanced mode (edit raw BRFSS-coded fields)", value=False)

input_row: Dict[str, Any] = {}

def select_code(label: str, mapping: Dict[int, str], default_code: int) -> int:
    options = list(mapping.keys())
    idx = options.index(default_code) if default_code in options else 0
    choice = st.sidebar.selectbox(label, options=options, index=idx, format_func=lambda k: mapping.get(k, str(k)))
    return int(choice)

def yesno(label: str, default_yes: bool = False) -> int:
    code = 1 if default_yes else 2
    return select_code(label, YESNO_MAP, code)

# Core demographics
input_row["SEX"] = select_code("Sex", SEX_MAP, 1)
input_row["_AGEG5YR"] = select_code("Age group", AGEG5YR_MAP, 9)
input_row["_EDUCAG"] = select_code("Education", EDUCAG_MAP, 3)
input_row["_INCOMG"] = select_code("Household income", INCOMG_MAP, 4)
input_row["GENHLTH"] = select_code("Self-rated general health", GENHLTH_MAP, 2)

# Health history / behaviors (simple toggles)
input_row["BPHIGH4"] = select_code("History of high blood pressure", {1: "Yes", 2: "No", 3: "Borderline", 4: "Pregnancy-related"}, 2)
input_row["BPMEDS"] = yesno("On BP medication", default_yes=False)
input_row["TOLDHI2"] = yesno("Told high cholesterol", default_yes=False)
input_row["DIFFWALK"] = yesno("Difficulty walking/climbing stairs", default_yes=False)
input_row["DECIDE"] = yesno("Cognitive difficulty (memory/concentration)", default_yes=False)
input_row["ASTHMA3"] = yesno("History of asthma", default_yes=False)
input_row["HAVARTH3"] = yesno("History of arthritis", default_yes=False)
input_row["EXERANY2"] = yesno("Any exercise (past 30 days)", default_yes=True)

# “Days” fields
input_row["PHYSHLTH"] = st.sidebar.slider("Physically unhealthy days (past 30 days)", 0, 30, 0)
input_row["MENTHLTH"] = st.sidebar.slider("Mentally unhealthy days (past 30 days)", 0, 30, 0)
input_row["POORHLTH"] = st.sidebar.slider("Days poor health limited activities", 0, 30, 0)

# BMI input (store as BRFSS-coded _BMI5 = BMI*100)
bmi = st.sidebar.slider("BMI (kg/m²)", 15.0, 60.0, 26.5, 0.1)
input_row["_BMI5"] = int(round(bmi * 100))

# Optional: allow users to set state + other codes (kept simple)
input_row["_STATE"] = st.sidebar.number_input("State code (BRFSS _STATE)", min_value=1, max_value=99, value=24, step=1)

# Additional fields required by model but not exposed in simple UI:
# set as NaN; model pipeline handles missing via imputation.
for col in predictor_cols:
    input_row.setdefault(col, np.nan)

# If advanced mode, show an editable table for all predictor cols
if use_advanced:
    st.sidebar.markdown("---")
    st.sidebar.caption("Advanced: edit raw BRFSS-coded predictors used by the model.")
    adv_df = pd.DataFrame([input_row], columns=predictor_cols)
    edited = st.sidebar.data_editor(adv_df, use_container_width=True, hide_index=True)
    # Pull edited values back
    for c in predictor_cols:
        v = edited.loc[0, c]
        input_row[c] = v


# =============================================================================
# Predictions
# =============================================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Predicted risks")

pred_df = predict_all(input_row, disease_models, thresholds, predictor_cols)

# Display table with intuitive column names and formatting
display_df = pred_df.drop(columns=["_disease_code"]).copy()
display_df["Predicted risk (0–1)"] = display_df["Predicted risk (0–1)"].map(lambda x: f"{x:.4f}")
display_df["Threshold"] = display_df["Threshold"].map(lambda x: f"{x:.2f}")
display_df["Uncertainty (0–1)"] = display_df["Uncertainty (0–1)"].map(lambda x: f"{x:.4f}")

st.dataframe(display_df, use_container_width=True, hide_index=True)

st.markdown(
    """
    **Field definitions**
    - **Predicted risk (0–1)**: model-estimated probability
    - **Threshold**: stored decision threshold for the condition
    - **Risk category**: label using the stored threshold
    - **Uncertainty (0–1)**: simple proxy; 0 = confident (p near 0 or 1), 1 = most uncertain (p near 0.5)
    """
)
st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# Optional: SHAP local explanations
# =============================================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Local feature contributions (optional)")

if not HAS_SHAP:
    st.info("SHAP is not available in this environment. Install `shap` to enable local explanations.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

disease_options = pred_df["_disease_code"].tolist()
default_disease = "depression" if "depression" in disease_options else disease_options[0]
selected_disease = st.selectbox(
    "Select a condition to explain",
    options=disease_options,
    index=disease_options.index(default_disease),
    format_func=lambda d: disease_label(d),
)

top_n = st.slider("Number of top features to display", 5, 30, 15, 1)

contrib = shap_local_explanation(input_row, selected_disease, disease_models, predictor_cols, top_n=top_n)

if contrib is None or contrib.empty:
    st.info("SHAP explanation not available for this model/configuration.")
else:
    # Table: intuitive labels + explicit sign interpretation
    contrib_table = contrib.copy()
    contrib_table.rename(
        columns={
            "feature": "Feature",
            "shap_value": "SHAP value",
        },
        inplace=True,
    )
    contrib_table["Interpretation"] = np.where(
        contrib_table["SHAP value"] >= 0,
        "Increases predicted risk",
        "Decreases predicted risk",
    )
    contrib_table["SHAP value"] = contrib_table["SHAP value"].astype(float)

    st.markdown("**Interpretation:** Positive SHAP values increase predicted risk; negative values decrease predicted risk.")
    st.dataframe(
        contrib_table[["Feature", "SHAP value", "Interpretation"]]
        .assign(**{"SHAP value": lambda d: d["SHAP value"].map(lambda x: f"{x:.4f}")}),
        use_container_width=True,
        hide_index=True,
    )

    # Plot
    fig = plot_shap_bar(contrib, title=f"Local explanation: {disease_label(selected_disease)}")
    st.pyplot(fig, clear_figure=True)

st.markdown("</div>", unsafe_allow_html=True)

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
    import shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False


# ============================================================
# 1) Artifact auto-download
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
    "Disclaimer: This tool does not provide a diagnosis and is not a substitute for "
    "professional medical advice. If you have health concerns, consult a qualified clinician."
)


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
# 2) Dropdown visibility fix ONLY (no other styling touched)
# ============================================================
st.markdown(
    """
<style>

/* Closed selectbox */
div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    color: #000000 !important;
}

/* Dropdown menu */
ul[role="listbox"] {
    background-color: #ffffff !important;
    color: #000000 !important;
}

/* Dropdown items */
ul[role="listbox"] li {
    background-color: #ffffff !important;
    color: #000000 !important;
}

/* Hover state */
ul[role="listbox"] li:hover {
    background-color: #f2f2f2 !important;
    color: #000000 !important;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# 3) Utility functions
# ============================================================
def uncertainty_from_proba(p: float) -> float:
    p = float(p)
    return float(1.0 - abs(p - 0.5) * 2.0)


def _safe_float(x: Any) -> Optional[float]:
    try:
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

    return pd.DataFrame(records).sort_values("probability", ascending=False)


# ============================================================
# 4) UI
# ============================================================
st.set_page_config(page_title="Survey-ML Risk", layout="wide")

st.sidebar.title("Survey-ML Risk")
page = st.sidebar.radio("Navigate", ["Risk prediction", "Model evaluation", "About"])
st.sidebar.markdown("---")
st.sidebar.caption("Research/demonstration only; not yet for clinical use.")


if page == "About":
    st.title("About this app")
    st.markdown(
        """
This repository provides an interpretable machine-learning framework for estimating chronic disease risk
using population survey data (BRFSS 2011–2015).

⚠️ **{}**
        """.format(DISCLAIMER_TEXT)
    )
    st.markdown("### Project repository")
    st.markdown("[View source code on GitHub](https://github.com/reaganmogire/survey-ml-risk)")


elif page == "Risk prediction":
    st.title("Risk prediction (survey-based ML)")
    st.warning(DISCLAIMER_TEXT)

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age (years)", 18, 99, 35)
        sex = st.selectbox("Sex", ["Male", "Female"])

    with col2:
        bmi = st.number_input("BMI (kg/m²)", 10.0, 70.0, 27.0)

    model_input = {}
    run_btn = st.button("Run prediction")

    if run_btn:
        results_df = predict_all_conditions(model_input)
        st.subheader("Predicted risks")
        st.dataframe(results_df, use_container_width=True)
        st.markdown("\n⚠️ **{}**".format(DISCLAIMER_TEXT))


elif page == "Model evaluation":
    st.title("Model evaluation")
    st.warning(DISCLAIMER_TEXT)

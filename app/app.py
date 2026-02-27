import os
from pathlib import Path

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

# ---------- Optional deps ----------
# SHAP (optional)
try:
    import shap  # noqa: F401
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False


# =============================
# Paths + GitHub Release artifact download
# =============================
REPO_ROOT = Path(__file__).resolve().parents[1]          # repo root (../)
APP_DIR = Path(__file__).resolve().parent                # app/
ARTIFACT_DIR = APP_DIR / "artifacts"                     # app/artifacts/

# GitHub Release tag that contains the model artifacts
MODEL_RELEASE_TAG = os.environ.get("MODEL_RELEASE_TAG", "model-v1")
GITHUB_RELEASE_BASE = (
    f"https://github.com/reaganmogire/survey-ml-risk/releases/download/{MODEL_RELEASE_TAG}"
)

ARTIFACTS = {
    "disease_models.joblib": f"{GITHUB_RELEASE_BASE}/disease_models.joblib",
    "optimal_thresholds.joblib": f"{GITHUB_RELEASE_BASE}/optimal_thresholds.joblib",
    "predictor_cols.joblib": f"{GITHUB_RELEASE_BASE}/predictor_cols.joblib",
}


def _download_with_progress(url: str, dest: Path) -> None:
    """Download a file with a Streamlit progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0

        prog = st.progress(0.0)
        status = st.empty()

        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1 MB chunks
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    prog.progress(min(downloaded / total, 1.0))
                    status.caption(f"Downloaded {downloaded/1e6:.1f} / {total/1e6:.1f} MB")

    tmp.replace(dest)
    status.empty()


def ensure_artifacts() -> None:
    """Ensure required artifacts exist locally; otherwise download from GitHub Release."""
    missing = [name for name in ARTIFACTS if not (ARTIFACT_DIR / name).exists()]
    if not missing:
        return

    st.info(
        "Model artifacts are missing. Downloading from GitHub Releases (first run only)..."
    )
    for name in missing:
        url = ARTIFACTS[name]
        dest = ARTIFACT_DIR / name
        with st.spinner(f"Downloading {name}..."):
            _download_with_progress(url, dest)


# =============================
# Load trained artifacts
# =============================
@st.cache_resource(show_spinner=False)
def load_artifacts():
    ensure_artifacts()
    disease_models = joblib.load(ARTIFACT_DIR / "disease_models.joblib")
    optimal_thresholds = joblib.load(ARTIFACT_DIR / "optimal_thresholds.joblib")
    predictor_cols = joblib.load(ARTIFACT_DIR / "predictor_cols.joblib")
    return disease_models, optimal_thresholds, predictor_cols


disease_models, optimal_thresholds, predictor_cols = load_artifacts()


# =============================
# Helper functions
# =============================
def uncertainty_from_proba(p: float) -> float:
    """
    0 = model very confident (p near 0 or 1)
    1 = maximally uncertain (p = 0.5)
    """
    return float(1.0 - abs(p - 0.5) * 2.0)


def age_to_ageg5yr(age_years: int) -> int:
    """Map age in years to BRFSS _AGEG5YR categories."""
    age = max(18, min(int(age_years), 99))
    bins = [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    for i, upper in enumerate(bins, start=1):
        if age < upper:
            return i
    return 13  # 80+


def bmi_to_bmi5(bmi: float) -> int:
    """BMI (kg/m2) -> BRFSS _BMI5 (integer * 100)."""
    return int(round(float(bmi) * 100))


def drinks_per_week_to_alcday5(drinks_per_week: float) -> int:
    """
    Approximate mapping:
    0 -> 888 (no drinks)
    >0 -> BRFSS 'per week' coding: 201–207 (1–7 drinks/week, capped at 7).
    """
    d = float(drinks_per_week)
    if d <= 0:
        return 888
    d_int = int(round(min(d, 7)))
    return 200 + d_int


def predict_user(
    input_dict,
    disease_models=disease_models,
    thresholds=optimal_thresholds,
    predictor_cols=predictor_cols,
) -> pd.DataFrame:
    """Build 1-row dataframe from input_dict, run all models, return df indexed by disease."""
    row = pd.DataFrame([input_dict]).reindex(columns=predictor_cols, fill_value=np.nan)

    records = []
    for disease, info in disease_models.items():
        model = info["model"]
        t = float(thresholds[disease])

        p = float(model.predict_proba(row)[0, 1])
        unc = uncertainty_from_proba(p)
        label = "High risk" if p >= t else "Low / moderate risk"

        records.append(
            {
                "disease": disease,
                "probability": p,
                "threshold_used": t,
                "risk_classification": label,
                "uncertainty": unc,
            }
        )

    df = pd.DataFrame(records).set_index("disease")
    return df


def generate_offline_guidance(risk_profile: dict, user_friendly_input: dict) -> str:
    """
    Rule-based, educational guidance only.
    This replaces any LLM-based guidance. No external AI calls.
    """
    lines = []

    # Identify higher-risk conditions (based on model classification only)
    high = [k for k, v in risk_profile.items() if v.get("risk_classification") == "High risk"]
    if high:
        lines.append(
            "Based on the model output, the following conditions are flagged as **higher risk**: "
            + ", ".join(high)
            + "."
        )
    else:
        lines.append("Based on the model output, no conditions are flagged as **higher risk**.")

    # Tailor a few tips from user inputs (examples)
    bmi = user_friendly_input.get("BMI")
    if bmi is not None:
        try:
            bmi_val = float(bmi)
            if bmi_val >= 30:
                lines.append(
                    "Your BMI suggests obesity. Gradual weight reduction through diet quality, "
                    "portion control, and regular physical activity can lower cardiometabolic risk."
                )
            elif bmi_val >= 25:
                lines.append(
                    "Your BMI suggests overweight. Small, sustainable changes (daily walking, "
                    "reducing sugary drinks, more vegetables/fiber) can meaningfully improve risk."
                )
        except Exception:
            pass

    smoker = user_friendly_input.get("Smoking status")
    if smoker and "current" in str(smoker).lower():
        lines.append(
            "You reported current smoking. Quitting smoking is among the most effective ways to "
            "reduce long-term cardiovascular and overall health risk."
        )

    alc = user_friendly_input.get("Alcohol (drinks/week)")
    if alc is not None:
        try:
            alc_val = float(alc)
            if alc_val >= 14:
                lines.append(
                    "Your reported alcohol intake is relatively high. Reducing intake can lower "
                    "blood pressure and improve cardiometabolic health."
                )
        except Exception:
            pass

    diffwalk = user_friendly_input.get("Difficulty walking/climbing stairs")
    if diffwalk == "Yes":
        lines.append(
            "You reported difficulty walking or climbing stairs. Consider discussing safe activity "
            "options with a clinician (e.g., tailored exercise, physiotherapy)."
        )

    lines.append(
        "\n⚠️ This information is educational only and **not** a substitute for personalised medical advice "
        "or diagnosis. Please discuss your situation with a qualified healthcare professional."
    )

    return "\n".join(lines)


# =============================
# Optional SHAP explainers
# =============================
@st.cache_resource(show_spinner=False)
def build_shap_explainers():
    if not _HAS_SHAP:
        return None
    explainers = {}
    for disease, info in disease_models.items():
        model = info["model"]
        try:
            # TreeExplainer works well for GBDTs
            explainers[disease] = shap.TreeExplainer(model)
        except Exception:
            explainers[disease] = None
    return explainers


shap_explainers = build_shap_explainers()


def explain_instance(model_input: dict, disease: str) -> pd.DataFrame | None:
    if not _HAS_SHAP or shap_explainers is None:
        return None
    explainer = shap_explainers.get(disease)
    if explainer is None:
        return None

    x = pd.DataFrame([model_input]).reindex(columns=predictor_cols, fill_value=np.nan)
    try:
        sv = explainer.shap_values(x)
        # Binary classifier: shap may return list [class0, class1]
        if isinstance(sv, list) and len(sv) == 2:
            sv = sv[1]
        sv = np.array(sv).reshape(-1)
    except Exception:
        return None

    df = pd.DataFrame(
        {
            "feature": predictor_cols,
            "shap_value": sv,
        }
    ).assign(abs_shap=lambda d: d["shap_value"].abs())
    df = df.sort_values("abs_shap", ascending=False).head(20).drop(columns=["abs_shap"])
    df["display_feature"] = df["feature"]
    return df


# =============================
# Streamlit layout
# =============================
st.set_page_config(page_title="Survey-ML Risk (BRFSS)", layout="wide")

st.sidebar.title("Survey-ML Risk")
page = st.sidebar.radio("Go to", ["Risk prediction", "Model evaluation"])
st.sidebar.markdown("—")
st.sidebar.caption("Research prototype – not for clinical use.")


# =============================
# PAGE 1 – Risk prediction
# =============================
if page == "Risk prediction":
    st.title("Chronic Disease Risk Explorer (Survey-based ML)")

    st.markdown(
        """
Enter health information below. The model estimates risk for several chronic conditions
and reports both **risk** and **uncertainty**.

⚠️ **This tool does not provide a diagnosis and is not a substitute for professional medical advice.**
        """
    )

    # ------- Inputs -------
    # (Keeping your original structure: map human inputs -> BRFSS-coded features used by model)

    col1, col2, col3 = st.columns(3)

    with col1:
        age_years = st.number_input("Age (years)", min_value=18, max_value=99, value=35, step=1)
        sex = st.selectbox("Sex", ["Male", "Female"])
        bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=70.0, value=27.0, step=0.1)

    with col2:
        has_htn = st.selectbox("Ever told you have high blood pressure?", ["No", "Yes"])
        diffwalk = st.selectbox("Difficulty walking/climbing stairs?", ["No", "Yes"])
        genhlth = st.selectbox(
            "General health", ["Excellent", "Very good", "Good", "Fair", "Poor"]
        )

    with col3:
        smoke = st.selectbox("Smoking status", ["Never", "Former", "Current"])
        drinks_pw = st.number_input("Alcohol (drinks/week)", min_value=0.0, max_value=70.0, value=0.0, step=1.0)
        physact = st.selectbox("Any physical activity/exercise in past month?", ["No", "Yes"])

    submitted = st.button("Run risk prediction")

    # ------- Build model input (example mapping; keep/extend to match your training features) -------
    # NOTE: You must ensure these keys match your trained model features (predictor_cols).
    # The previous app relied on the same mapping logic—keep your full mapping here if you already had it.
    model_input = {
        "_AGEG5YR": age_to_ageg5yr(age_years),
        "_BMI5": bmi_to_bmi5(bmi),
        "SEX": 1 if sex == "Male" else 2,
        "BPHIGH4": 1 if has_htn == "Yes" else 2,  # example coding
        "DIFFWALK": 1 if diffwalk == "Yes" else 2,
        "GENHLTH": {"Excellent": 1, "Very good": 2, "Good": 3, "Fair": 4, "Poor": 5}[genhlth],
        "SMOKER3": {"Never": 4, "Former": 3, "Current": 1}[smoke],  # example coding
        "ALCDAY5": drinks_per_week_to_alcday5(drinks_pw),
        "EXERANY2": 1 if physact == "Yes" else 2,
    }

    user_friendly_input = {
        "Age": age_years,
        "Sex": sex,
        "BMI": bmi,
        "Hypertension history": has_htn,
        "Difficulty walking/climbing stairs": diffwalk,
        "General health": genhlth,
        "Smoking status": smoke,
        "Alcohol (drinks/week)": drinks_pw,
        "Physical activity": physact,
    }

    if submitted:
        results_df = predict_user(model_input)

        display_df = (
            results_df.reset_index()
            .rename(columns={"disease": "Condition"})
            .sort_values("probability", ascending=False)
        )

        risk_json = results_df.reset_index().to_dict(orient="records")

        st.markdown("### Model predictions")
        st.dataframe(display_df, use_container_width=True)

        st.markdown(
            """
- **probability** – predicted risk (0–1)  
- **threshold_used** – optimized threshold for high vs low/moderate risk  
- **risk_classification** – high vs low/moderate risk  
- **uncertainty** – 0 = very confident, 1 = very uncertain  
            """
        )

        with st.expander("Raw JSON output (for API use)"):
            st.json(risk_json)

        st.markdown("### Lifestyle guidance (rule-based)")
        st.write(generate_offline_guidance({r["Condition"] if "Condition" in r else r.get("disease"): r for r in []}, user_friendly_input))

        # Better: build a dict keyed by disease
        rp = results_df.reset_index().set_index("disease").to_dict(orient="index")
        st.write(generate_offline_guidance(rp, user_friendly_input))

        # SHAP explanations (optional)
        st.markdown("### Why did the model predict this? (feature contributions)")
        if shap_explainers is None:
            st.info("Install the `shap` package to see feature explanations.")
        else:
            default_disease = results_df["probability"].idxmax()
            disease_choice = st.selectbox(
                "Select condition to explain",
                options=list(results_df.index),
                index=list(results_df.index).index(default_disease),
            )

            df_shap = explain_instance(model_input, disease_choice)
            if df_shap is not None and not df_shap.empty:
                st.write("Top features influencing the prediction:")
                st.dataframe(
                    df_shap[["display_feature", "shap_value"]].style.format(
                        {"shap_value": "{:.3f}"}
                    ),
                    use_container_width=True,
                )

                fig, ax = plt.subplots(figsize=(6, 4))
                df_plot = df_shap.sort_values("shap_value")
                ax.barh(df_plot["display_feature"], df_plot["shap_value"])
                ax.set_xlabel("SHAP value (impact on model output)")
                ax.set_title(f"Local explanation for {disease_choice}")
                st.pyplot(fig)
            else:
                st.info("SHAP explanation not available for this model/instance.")


# =============================
# PAGE 2 – Model evaluation
# =============================
elif page == "Model evaluation":
    st.title("Model evaluation")

    st.markdown(
        """
This section provides evaluation utilities (AUROC, PR-AUC, calibration, confusion matrix)
for precomputed predictions if available in your stored model artifacts.

If your training script stored per-disease test predictions inside `disease_models.joblib`,
this page can visualize them. Otherwise, you can extend your training pipeline to export
a compact `tables/` CSV of metrics and load it here.
        """
    )

    # If your disease_models object contains test labels/preds, try to use them.
    # This depends on how you serialized disease_models in model_ai.py.
    available = []
    for disease, info in disease_models.items():
        if isinstance(info, dict) and ("y_test" in info and "p_test" in info):
            available.append(disease)

    if not available:
        st.warning(
            "No stored test predictions found inside disease_models.joblib "
            "(expected keys: y_test and p_test). "
            "If you want this page to work, update model_ai.py to store them."
        )
    else:
        disease_choice = st.selectbox("Select condition", options=available)
        info = disease_models[disease_choice]
        y = np.asarray(info["y_test"])
        p = np.asarray(info["p_test"])

        # Metrics
        auroc = roc_auc_score(y, p)
        ap = average_precision_score(y, p)
        brier = brier_score_loss(y, p)

        c1, c2, c3 = st.columns(3)
        c1.metric("AUROC", f"{auroc:.3f}")
        c2.metric("PR-AUC", f"{ap:.3f}")
        c3.metric("Brier", f"{brier:.3f}")

        # ROC
        fpr, tpr, _ = roc_curve(y, p)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(fpr, tpr)
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"ROC curve: {disease_choice}")
        st.pyplot(fig)

        # PR
        prec, rec, _ = precision_recall_curve(y, p)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(rec, prec)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision–Recall curve: {disease_choice}")
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

        # Confusion matrix at threshold
        t = float(optimal_thresholds.get(disease_choice, 0.5))
        yhat = (p >= t).astype(int)
        cm = confusion_matrix(y, yhat)
        st.write(f"Confusion matrix at threshold {t:.3f}")
        st.dataframe(pd.DataFrame(cm, index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"]))

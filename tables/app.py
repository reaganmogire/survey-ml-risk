import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt

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
# SHAP
try:
    import shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

# Language model client (OpenAI SDK)
try:
    from openai import OpenAI
    _HAS_OPENAI = True
    _OPENAI_CLIENT = OpenAI()
except Exception:
    _HAS_OPENAI = False
    _OPENAI_CLIENT = None


# =============================
# Load trained artifacts
# =============================
@st.cache_resource
def load_artifacts():
    disease_models = joblib.load("disease_models.joblib")
    optimal_thresholds = joblib.load("optimal_thresholds.joblib")
    predictor_cols = joblib.load("predictor_cols.joblib")
    return disease_models, optimal_thresholds, predictor_cols


disease_models, optimal_thresholds, predictor_cols = load_artifacts()


# =============================
# Helper functions
# =============================
def uncertainty_from_proba(p: float) -> float:
    """
    Original uncertainty measure:
    - 0 = model very confident (p near 0 or 1)
    - 1 = maximally uncertain (p = 0.5)
    Simple linear distance from 0.5.
    """
    return float(1.0 - abs(p - 0.5) * 2.0)


def age_to_ageg5yr(age_years: int) -> int:
    """Map age in years to BRFSS _AGEG5YR categories."""
    age = max(18, min(age_years, 99))
    bins = [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    for i, upper in enumerate(bins, start=1):
        if age < upper:
            return i
    return 13  # 80+


def bmi_to_bmi5(bmi: float) -> int:
    """BMI (kg/m2) -> BRFSS _BMI5 (integer * 100)."""
    return int(round(bmi * 100))


def drinks_per_week_to_alcday5(drinks_per_week: float) -> int:
    """
    Approximate mapping:
    0 -> 888 (no drinks)
    >0 -> BRFSS 'per week' coding: 201–207 (1–7 drinks/week, capped at 7).
    """
    if drinks_per_week <= 0:
        return 888
    d = int(round(min(drinks_per_week, 7)))
    return 200 + d


def predict_user(input_dict,
                 disease_models=disease_models,
                 thresholds=optimal_thresholds,
                 predictor_cols=predictor_cols):
    """
    Build a 1-row dataframe from input_dict, run all disease models,
    and return a DataFrame indexed by disease.
    """
    row = pd.DataFrame([input_dict]).reindex(columns=predictor_cols, fill_value=np.nan)

    records = []
    for disease, info in disease_models.items():
        model = info["model"]
        t = thresholds[disease]

        p = model.predict_proba(row)[0, 1]
        unc = uncertainty_from_proba(p)
        label = "High risk" if p >= t else "Low / moderate risk"

        records.append({
            "disease": disease,
            "probability": float(p),
            "threshold_used": float(t),
            "risk_classification": label,
            "uncertainty": float(unc),
        })

    df = pd.DataFrame(records).set_index("disease")
    return df


def predict_user_json(input_dict):
    """JSON-friendly wrapper (good for LLM prompts or APIs)."""
    df = predict_user(input_dict)
    return df.to_dict(orient="index")


# =============================
# Evaluation helpers
# =============================
def evaluate_models(disease_models, threshold_dict=None, global_threshold=None):
    """
    Compute AUROC, AUPRC, Brier, sens/spec/PPV/NPV per disease.
    """
    rows = []

    for disease, info in disease_models.items():
        y_test = info["y_test"]
        y_proba = info["y_proba"]

        if threshold_dict is not None:
            t = threshold_dict[disease]
        else:
            t = global_threshold if global_threshold is not None else 0.5

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
            "NPV": npv,
        })

    return pd.DataFrame(rows).set_index("disease")


def plot_calibration(disease):
    info = disease_models[disease]
    y_test = info["y_test"]
    y_proba = info["y_proba"]

    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(prob_pred, prob_true, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"Calibration: {disease}")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_roc(disease):
    info = disease_models[disease]
    y_test = info["y_test"]
    y_proba = info["y_proba"]

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC curve: {disease}")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_pr(disease):
    info = disease_models[disease]
    y_test = info["y_test"]
    y_proba = info["y_proba"]

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(recall, precision, label=f"PR (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision–Recall: {disease}")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


# =============================
# Friendly feature names for SHAP
# =============================

# Include both with and without leading underscores for robustness
FRIENDLY_COL_MAP = {
    "_BMI5": "BMI",
    "BMI5": "BMI",
    "_AGEG5YR": "Age group",
    "AGEG5YR": "Age group",
    "_INCOMG": "Household income",
    "INCOMG": "Household income",
    "_GENHLTH": "General health rating",
    "GENHLTH": "General health rating",
    "_BPHIGH4": "Ever told high blood pressure",
    "BPHIGH4": "Ever told high blood pressure",
    "_CHOLCHK": "Had cholesterol test in last 5 years",
    "CHOLCHK": "Had cholesterol test in last 5 years",
    "SMOKE100": "Ever smoked ≥100 cigarettes",
    "ALCDAY5": "Alcohol use frequency",
    "DIFFWALK": "Difficulty walking",
    "TOLDHI2": "Ever told high cholesterol",
}


def pretty_feature_name(raw_name: str) -> str:
    """
    Map raw preprocessor feature names (e.g. 'preprocessor__num___BMI5',
    'num__BMI5', 'cat__SMOKE100_1.0') to friendlier labels.

    Strategy:
    - Strip 'preprocessor__' prefix if present.
    - Handle numeric features with 'num__' or 'num___'.
    - Handle categorical features with 'cat__'.
    - Fallback: substring match against FRIENDLY_COL_MAP keys.
    """
    name = raw_name

    # Strip pipeline prefix if present
    if name.startswith("preprocessor__"):
        name = name.split("preprocessor__", 1)[1]

    # Numeric features
    if "num__" in name:
        after = name.split("num__", 1)[1]  # e.g. '_BMI5' or '_BMI5_something'
        # Remove extra leading underscores
        col = after.lstrip("_").split("_")[0]  # 'BMI5'
        # Try direct mapping
        if col in FRIENDLY_COL_MAP:
            return FRIENDLY_COL_MAP[col]
        if "_" + col in FRIENDLY_COL_MAP:
            return FRIENDLY_COL_MAP["_" + col]
        # Fallback to col itself
        return col

    # Categorical features
    if "cat__" in name:
        after = name.split("cat__", 1)[1]  # e.g. 'SMOKE100_1.0'
        parts = after.split("_")
        col_part = parts[0]  # 'SMOKE100'
        value_part = "_".join(parts[1:]) if len(parts) > 1 else ""

        base = FRIENDLY_COL_MAP.get(col_part, FRIENDLY_COL_MAP.get("_" + col_part, col_part))
        if value_part:
            return f"{base} = {value_part}"
        return base

    # Fallback: try any FRIENDLY_COL_MAP key that appears as substring
    for key, label in FRIENDLY_COL_MAP.items():
        if key in name:
            return label

    # Last resort: return raw name
    return raw_name


# =============================
# SHAP explainers
# =============================
@st.cache_resource
def get_shap_explainers():
    if not _HAS_SHAP:
        return None

    explainers = {}
    for disease, info in disease_models.items():
        model = info["model"]
        preproc = model.named_steps["preprocessor"]
        clf = model.named_steps["clf"]

        explainer = shap.TreeExplainer(clf)
        explainers[disease] = (explainer, preproc)
    return explainers


shap_explainers = get_shap_explainers()


def explain_instance(user_input: dict, disease: str):
    """Return top features (by |SHAP|) for one disease & one input, with pretty labels."""
    if shap_explainers is None:
        return None

    explainer, preproc = shap_explainers[disease]

    row = pd.DataFrame([user_input]).reindex(columns=predictor_cols, fill_value=np.nan)
    X_t = preproc.transform(row)

    shap_vals = explainer.shap_values(X_t)[0]
    feature_names = preproc.get_feature_names_out()

    df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_vals
    })
    df["abs_value"] = df["shap_value"].abs()
    df["display_feature"] = df["feature"].map(pretty_feature_name)
    df_top = df.sort_values("abs_value", ascending=False).head(10)
    return df_top


# =============================
# Offline rule-based guidance
# =============================
def generate_offline_guidance(risk_profile: dict, user_friendly_input: dict) -> str:
    """
    Purely offline, rule-based lifestyle guidance.
    Uses risk_profile + human-friendly inputs (age, BMI, smoking, etc.).
    """
    high_risk = [
        (d, v["probability"])
        for d, v in risk_profile.items()
        if v["risk_classification"].startswith("High")
    ]

    age = user_friendly_input.get("age_years")
    bmi = user_friendly_input.get("BMI")
    sex = user_friendly_input.get("sex")
    drinks = user_friendly_input.get("drinks_per_week")
    smoke = user_friendly_input.get("ever_smoked_100_cigs")
    bp = user_friendly_input.get("ever_high_bp")
    chol = user_friendly_input.get("ever_high_cholesterol")
    genhlth = user_friendly_input.get("general_health")
    diffwalk = user_friendly_input.get("difficulty_walking")

    lines = []

    # Summary of risk profile
    if not high_risk:
        lines.append("The model estimates your risks to be in the low–moderate range "
                     "for the conditions it predicts.")
    else:
        lines.append("The model identified higher estimated risk (this is **not** a diagnosis) for:")
        for d, p in high_risk:
            lines.append(f"- **{d}** (estimated probability ≈ {p:.2f})")

    lines.append("")

    # Age & general cardiovascular risk
    if age is not None and age >= 50:
        lines.append(
            f"You are {age} years old. Cardiometabolic risk generally increases with age, "
            "so routine preventive care (blood pressure, cholesterol, diabetes screening) "
            "is especially important."
        )

    # BMI-based guidance
    if bmi is not None:
        if bmi >= 30:
            lines.append(
                f"Your BMI is about {bmi:.1f}, which is in the obese range. Gradual, "
                "sustained weight loss through a healthier diet and more physical activity "
                "can meaningfully reduce long-term risk of diabetes, heart disease, and "
                "kidney disease."
            )
        elif bmi >= 25:
            lines.append(
                f"Your BMI is about {bmi:.1f}, which is in the overweight range. "
                "Modest weight reduction or preventing further weight gain can lower "
                "your risk for several chronic conditions."
            )

    # Smoking
    if smoke == "Yes":
        lines.append(
            "You reported having smoked at least 100 cigarettes. Stopping smoking is "
            "one of the most powerful steps you can take to improve cardiovascular and "
            "overall health. Discuss structured cessation support (counselling, nicotine "
            "replacement, or medications) with your clinician."
        )

    # Alcohol
    if drinks is not None and drinks > 0:
        if drinks >= 15:
            lines.append(
                f"You report about {drinks:.0f} alcoholic drinks per week. Reducing alcohol "
                "intake can lower blood pressure, improve liver health, and reduce overall risk."
            )
        else:
            lines.append(
                f"You report about {drinks:.0f} alcoholic drinks per week. Keeping intake "
                "moderate or low is generally better for long-term health."
            )

    # Blood pressure / cholesterol
    if bp == "Yes" or chol == "Yes":
        lines.append(
            "Because you've been told you have high blood pressure and/or high cholesterol, "
            "it's important to follow up regularly and adhere to any treatment and lifestyle "
            "plans agreed with your healthcare provider."
        )

    # General health & mobility
    if genhlth in ["Fair", "Poor"]:
        lines.append(
            f"You rated your general health as {genhlth.lower()}. It may be helpful to "
            "review this with a clinician to identify specific areas where changes could "
            "improve your quality of life."
        )

    if diffwalk == "Yes":
        lines.append(
            "You reported difficulty walking or climbing stairs. Discuss with your clinician "
            "whether tailored physical activity, physiotherapy, or assistive devices could "
            "help maintain mobility safely."
        )

    lines.append(
        "\n⚠️ This information is educational only and **not** a substitute for personalised "
        "medical advice or diagnosis. Always discuss your specific situation with a qualified "
        "healthcare professional."
    )

    return "\n".join(lines)


# =============================
# LLM-based lifestyle guidance (robust, with toggle)
# =============================
def generate_recommendation(
    risk_profile: dict,
    model_input: dict,
    user_friendly_input: dict,
    use_online_ai: bool,
) -> str:
    """
    - Always able to return offline rule-based guidance.
    - If use_online_ai is True AND API is available, will try a language model.
      On ANY error (network/quota/etc.), falls back to offline guidance.
    """

    offline_text = generate_offline_guidance(risk_profile, user_friendly_input)

    # If toggle off or API not configured, return offline guidance
    if not use_online_ai or not (_HAS_OPENAI and os.getenv("OPENAI_API_KEY")):
        return offline_text

    client = _OPENAI_CLIENT

    prompt = f"""
You are a preventive-medicine assistant. You DO NOT diagnose disease or
recommend specific medications. You only give general, evidence-based
lifestyle guidance.

Model output (risk_profile, JSON-like dict):
{risk_profile}

Model input (BRFSS-style coded features):
{model_input}

User's own answers in human-readable form:
{user_friendly_input}

Task:
1. Briefly summarise which conditions appear higher risk.
2. Suggest practical lifestyle and behavioural changes (movement, diet,
   smoking, alcohol, sleep, stress, regular check-ups) that could help
   reduce long-term risk.
3. Base your advice on the user's own responses where relevant (e.g. BMI,
   smoking, alcohol, difficulty walking).
4. Include explicit disclaimers that this is not medical advice or a
   diagnosis and they should see a clinician for personalised guidance.

Write 2–4 short paragraphs in clear, non-technical English.
"""

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=400,
        )
        text = completion.choices[0].message.content.strip()
        return text
    except Exception as e:
        # Log to Streamlit server console, but don't crash UI
        print("LLM error in generate_recommendation:", repr(e))
        return offline_text + (
            "\n\n(Note: an external language model could not be reached, so this is "
            "generic rule-based guidance.)"
        )


# =============================
# Streamlit layout
# =============================
st.set_page_config(
    page_title="HealthyMe AI – Risk & Model Evaluation",
    layout="wide"
)

st.sidebar.title("HealthyMe AI")
page = st.sidebar.radio("Go to", ["Risk prediction", "Model evaluation"])
use_online_ai = st.sidebar.checkbox(
    "Use enhanced AI guidance",
    value=False,
    help="If enabled, guidance will be generated using an advanced language model "
         "when available. If disabled, the app will use the built-in rule-based guidance."
)
st.sidebar.markdown("—")
st.sidebar.caption("Prototype – not for clinical use.")


# =============================
# PAGE 1 – Risk prediction
# =============================
if page == "Risk prediction":
    st.title("HealthyMe AI – Disease Risk Explorer")

    st.markdown(
        """
Enter health information below. The model estimates the risk of several
chronic conditions and reports both **risk** and **certainty**.

⚠️ **This tool does not provide a diagnosis and is not a substitute
for professional medical advice.**
        """
    )

    st.subheader("Risk factor inputs")

    with st.form("risk_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age_years = st.slider("Age (years)", min_value=18, max_value=90, value=60)
            sex_label = st.selectbox("Sex", ["Male", "Female"])
            sex = 1 if sex_label == "Male" else 2

            income_label = st.selectbox(
                "Household income (approximate)",
                [
                    "<$15k",
                    "$15–25k",
                    "$25–35k",
                    "$35–50k",
                    "$50–75k",
                    "$75–100k",
                    ">$100k",
                    "Prefer not to say",
                ],
                index=3,
            )
            income_map = {
                "<$15k": 1,
                "$15–25k": 2,
                "$25–35k": 3,
                "$35–50k": 4,
                "$50–75k": 5,
                "$75–100k": 6,
                ">$100k": 7,
                "Prefer not to say": 8,
            }
            incomg = income_map[income_label]

        with col2:
            bmi = st.slider("BMI (kg/m²)", min_value=15.0, max_value=50.0, value=27.0)
            genhlth_label = st.selectbox(
                "General health (overall health rating)",
                ["Excellent", "Very good", "Good", "Fair", "Poor"],
                index=2,
                help="Your overall health as you feel it, including physical and mental wellbeing."
            )
            genhlth_map = {
                "Excellent": 1,
                "Very good": 2,
                "Good": 3,
                "Fair": 4,
                "Poor": 5,
            }
            genhlth = genhlth_map[genhlth_label]

            bp_label = st.selectbox(
                "Ever told you have high blood pressure?",
                ["No", "Yes"],
                index=0,
            )
            # Approx: 1=High BP, 3=No high BP
            bphigh4 = 1 if bp_label == "Yes" else 3

            cholchk_label = st.selectbox(
                "Had cholesterol test in last 5 years?",
                ["Yes", "No"],
                index=0,
            )
            cholchk = 1 if cholchk_label == "Yes" else 2

        with col3:
            smoke_label = st.selectbox(
                "Have you smoked at least 100 cigarettes in your life?",
                ["No", "Yes"],
                index=0,
            )
            smoke100 = 1 if smoke_label == "Yes" else 2

            drinks_week = st.slider(
                "Average alcoholic drinks per week",
                min_value=0.0, max_value=50.0, value=0.0, step=1.0
            )

            diffwalk_label = st.selectbox(
                "Serious difficulty walking or climbing stairs?",
                ["No", "Yes"],
                index=0,
            )
            diffwalk = 1 if diffwalk_label == "Yes" else 2

            toldhi_label = st.selectbox(
                "Ever told by a doctor you have high cholesterol?",
                ["No", "Yes"],
                index=0,
            )
            toldhi2 = 1 if toldhi_label == "Yes" else 2

        submitted = st.form_submit_button("Predict risk")

    # When the form is submitted, run the model and store results in session_state
    if submitted:
        # Map to BRFSS-style codes (model inputs)
        ageg5yr = age_to_ageg5yr(age_years)
        bmi5 = bmi_to_bmi5(bmi)
        alcday5 = drinks_per_week_to_alcday5(drinks_week)

        model_input = {
            "_STATE": 1,  # default state; model will mostly ignore
            "SEX": sex,
            "_AGEG5YR": ageg5yr,
            "_INCOMG": incomg,
            "_BMI5": bmi5,
            "_GENHLTH": genhlth,
            "_BPHIGH4": bphigh4,
            "_CHOLCHK": cholchk,
            "SMOKE100": smoke100,
            "ALCDAY5": alcday5,
            "DIFFWALK": diffwalk,
            "TOLDHI2": toldhi2,
        }

        # Human-readable user inputs
        user_friendly_input = {
            "age_years": age_years,
            "sex": sex_label,
            "BMI": bmi,
            "income": income_label,
            "general_health": genhlth_label,
            "ever_high_bp": bp_label,
            "cholesterol_checked_5y": cholchk_label,
            "ever_smoked_100_cigs": smoke_label,
            "drinks_per_week": drinks_week,
            "difficulty_walking": diffwalk_label,
            "ever_high_cholesterol": toldhi_label,
        }

        results_df = predict_user(model_input)
        display_df = results_df.copy()
        display_df["probability"] = display_df["probability"].map(lambda x: f"{x:.3f}")
        display_df["threshold_used"] = display_df["threshold_used"].map(lambda x: f"{x:.3f}")
        display_df["uncertainty"] = display_df["uncertainty"].map(lambda x: f"{x:.3f}")

        risk_json = predict_user_json(model_input)

        # Store everything in session_state so it persists across reruns
        st.session_state["results_df"] = results_df
        st.session_state["display_df"] = display_df
        st.session_state["model_input"] = model_input
        st.session_state["user_friendly_input"] = user_friendly_input
        st.session_state["risk_json"] = risk_json

    # Render results if we have them in session_state (even if submitted is False this rerun)
    if "results_df" in st.session_state:
        results_df = st.session_state["results_df"]
        display_df = st.session_state["display_df"]
        model_input = st.session_state["model_input"]
        user_friendly_input = st.session_state["user_friendly_input"]
        risk_json = st.session_state["risk_json"]

        st.markdown("### Model predictions")
        st.dataframe(display_df, use_container_width=True)

        st.markdown(
            """
- **probability** – predicted risk (0–1)  
- **threshold_used** – optimized threshold for high vs low/moderate risk  
- **risk_classification** – high vs low/moderate risk  
- **uncertainty** – 0 = model very confident, 1 = very uncertain  
            """
        )

        with st.expander("Raw JSON output (for API / LLM use)"):
            st.json(risk_json)

        st.markdown("### AI lifestyle guidance")
        guidance = generate_recommendation(
            risk_json,
            model_input,
            user_friendly_input,
            use_online_ai=use_online_ai,
        )
        st.write(guidance)

        # SHAP explanations
        st.markdown("### Why did the model predict this? (feature contributions)")
        if shap_explainers is None:
            st.info("Install the `shap` package to see feature explanations.")
        else:
            # default disease = highest probability
            default_disease = results_df["probability"].idxmax()
            disease_choice = st.selectbox(
                "Select disease to explain",
                options=list(results_df.index),
                index=list(results_df.index).index(default_disease),
            )

            df_shap = explain_instance(model_input, disease_choice)
            if df_shap is not None:
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
                ax.set_xlabel("SHAP value (impact on log-odds)")
                ax.set_title(f"Local explanation for {disease_choice}")
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("Unable to compute SHAP explanation for this instance.")
    else:
        st.caption("Set the inputs above and click **Predict risk** to see results.")


# =============================
# PAGE 2 – Model evaluation
# =============================
elif page == "Model evaluation":
    st.title("Model evaluation – HealthyMe AI")

    st.markdown(
        """
This page summarizes how the models perform on held-out BRFSS data
(the internal test sets).
        """
    )

    st.subheader("Summary metrics (optimized thresholds)")

    eval_opt = evaluate_models(disease_models, threshold_dict=optimal_thresholds)
    df_show = eval_opt.copy()
    for col in [
        "prevalence",
        "AUROC",
        "AUPRC",
        "Brier",
        "sensitivity",
        "specificity",
        "PPV",
        "NPV",
    ]:
        df_show[col] = df_show[col].map(lambda x: f"{x:.3f}")

    st.dataframe(df_show, use_container_width=True)

    st.markdown(
        """
**Notes**

- Prevalence is the fraction of positive cases in the test set.  
- AUROC and AUPRC summarize ranking performance.  
- Brier score evaluates probability calibration (lower is better).  
- Sensitivity, specificity, PPV, and NPV are computed at the per-disease
  optimized thresholds (Youden’s J).
        """
    )

    st.subheader("Diagnostic curves")

    disease = st.selectbox(
        "Select disease", options=sorted(disease_models.keys())
    )
    plot_type = st.radio(
        "Plot type",
        ["Calibration", "ROC curve", "Precision–Recall"],
        horizontal=True,
    )

    if plot_type == "Calibration":
        fig = plot_calibration(disease)
    elif plot_type == "ROC curve":
        fig = plot_roc(disease)
    else:
        fig = plot_pr(disease)

    st.pyplot(fig)
    st.caption("These curves are computed on the internal test set.")


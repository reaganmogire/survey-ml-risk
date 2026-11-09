#!/usr/bin/env python3
# coding: utf-8
"""
scripts/make_manuscript_outputs.py

FINAL ORCHESTRATOR: computes the cheap, directly-computable manuscript
outputs itself (Table 1/3, ROC, calibration, example profiles, combined
Table 1 + prevalence), and for every OTHER table/figure/dataset in the
manuscript, CHECKS whether its output file already exists, is non-empty,
has the expected columns, and isn't stale relative to the model
artifact -- WITHOUT rerunning any expensive analysis (no retraining,
no SHAP recomputation, no bootstrap resampling other than what this
script itself does for Table 1).

This script ends by printing a COMPLETE MANUSCRIPT STATUS REPORT: every
Table 1-3, every Table, every S Table, every Fig and S Fig, every
S Dataset (labels follow the CURRENT manuscript numbering) -- what it is, which script generates it, which review
comment / TRIPOD+AI item it satisfies, and whether it's present, stale,
empty, or missing. This is the single source of truth for "is the
supplementary package done."

============================================================
HOW TO READ THE FINAL REPORT
============================================================
  [OK]      file exists, non-empty, has expected columns (if checked),
            not stale relative to disease_models.joblib
  [STALE]   file exists but is OLDER than disease_models.joblib --
            likely computed from a previous model version; verify
            before trusting its numbers
  [EMPTY]   file exists but has 0 data rows
  [MISSING] file not found -- run the listed generating script
  [CODE]    computed directly by THIS script (no separate file to check)

============================================================
SOME FILENAMES BELOW ARE INFERRED, NOT CONFIRMED
============================================================
Every registry entry whose exact output filename I have not personally
seen in a pasted script is marked inline with `# UNCONFIRMED`. If the
status report says MISSING for one of these, it may just mean the
filename guess is wrong, not that the analysis was never run --
check your actual tables/ directory and tell me the real filename so
the registry can be corrected.

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
import re
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

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

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

import sys
THIS_FILE_FOR_IMPORT = Path(__file__).resolve()
_FEATURE_LABELS_DIR = THIS_FILE_FOR_IMPORT.parent
if str(_FEATURE_LABELS_DIR) not in sys.path:
    sys.path.insert(0, str(_FEATURE_LABELS_DIR))

_feature_labels_module_path = _FEATURE_LABELS_DIR / "feature_labels.py"
if not _feature_labels_module_path.exists():
    raise FileNotFoundError(
        f"Expected scripts/feature_labels.py at {_feature_labels_module_path}, "
        f"but it was not found. Save feature_labels.py to the same "
        f"directory as this script ({_FEATURE_LABELS_DIR}) before rerunning."
    )

from feature_labels import pretty_feature_from_explainer as _pretty_feature_orig  # noqa: E402

# BRFSS _MRACE1 coding: 1 White, 2 Black, 3 American Indian/Alaska Native, 4 Asian,
# 5 Native Hawaiian/Pacific Islander, 6 Other race, 7 Multiracial. Older label maps had 3 and 4
# swapped and 5-7 mislabelled, so race labels are overridden here regardless of feature_labels.py.
_RACE_LABELS = {1: "White", 2: "Black", 3: "Native American/Alaska Native", 4: "Asian",
                5: "Native Hawaiian/Pacific Islander", 6: "Other race", 7: "Multiracial"}


def pretty_feature_from_explainer(name):
    s = str(name)
    if "_MRACE1_" in s:
        try:
            return f"Race: {_RACE_LABELS[int(float(s.split('_MRACE1_')[-1]))]}"
        except (ValueError, KeyError):
            pass
    return _pretty_feature_orig(name)

_plot_shap_violin_module_path = _FEATURE_LABELS_DIR / "plot_shap_violin.py"
if not _plot_shap_violin_module_path.exists():
    raise FileNotFoundError(
        f"Expected scripts/plot_shap_violin.py at {_plot_shap_violin_module_path}, "
        f"but it was not found. Save plot_shap_violin.py to the same "
        f"directory as this script ({_FEATURE_LABELS_DIR}) before rerunning."
    )

from plot_shap_violin import plot_combined_violin  # noqa: E402


# ============================================================
# Paths (repo-relative defaults; override via env vars)
# ============================================================
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1] if THIS_FILE.parent.name in {"src", "scripts"} else THIS_FILE.parent

ARTIFACT_DIR = Path(os.environ.get("ARTIFACT_DIR", REPO_ROOT / "app" / "artifacts"))
TABLES_DIR   = Path(os.environ.get("TABLES_DIR",   REPO_ROOT / "tables"))
FIGURES_DIR  = Path(os.environ.get("FIGURES_DIR",  REPO_ROOT / "figures"))

CLEAN_DATA_CSV = Path(os.environ.get(
    "CLEAN_DATA_CSV",
    REPO_ROOT / "data" / "derived" / "BRFSS_2011_2015_clean_model.csv"
))

for d in [TABLES_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_BOOTSTRAP = 200

FIG_GRID_ROWS = 3
FIG_GRID_COLS = 2


# ============================================================
# Shared figure-saving helper
# ============================================================
def _save_both(fig, pdf_path: Path, dpi: int = 300, bbox_inches: Optional[str] = None) -> Tuple[Path, Path]:
    pdf_path = Path(pdf_path)
    png_path = pdf_path.with_suffix(".png")
    save_kwargs = {"dpi": dpi}
    if bbox_inches is not None:
        save_kwargs["bbox_inches"] = bbox_inches
    fig.savefig(pdf_path, **save_kwargs)
    fig.savefig(png_path, **save_kwargs)
    return pdf_path, png_path


def _format_sig_figs(value: float, sig: int = 2) -> str:
    if value is None or (isinstance(value, float) and (np.isnan(value) or value == 0)):
        return "0.0"
    exponent = math.floor(math.log10(abs(value)))
    decimals = max(sig - exponent - 1, 0)
    return f"{value:.{decimals}f}"


DISEASE_NAME_MAP = {
    "heart_attack": "Heart attack",
    "coronary_hd":  "Coronary heart disease",
    "stroke":       "Stroke",
    "kidney":       "Chronic kidney disease",
    "depression":   "Depression",
    "diabetes":     "Diabetes",
}

def pretty_disease_name(d: str) -> str:
    return DISEASE_NAME_MAP.get(d, d.replace("_", " ").title())


# ============================================================
# Calibration metrics
# ============================================================
def compute_calibration_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
    min_bin_count: int = 20,
) -> Tuple[float, float, float, int]:
    y_true  = np.asarray(y_true,  dtype=float)
    y_proba = np.asarray(y_proba, dtype=float)
    n = len(y_true)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.digitize(y_proba, bin_edges[1:-1], right=False)
    raw_counts = np.array([(bin_idx == j).sum() for j in range(n_bins)])

    active_bins = [{"lo": j, "hi": j, "mask": (bin_idx == j)} for j in range(n_bins) if raw_counts[j] > 0]
    n_bins_merged = 0

    def bin_count(b):
        return int(b["mask"].sum())

    changed = True
    while changed and len(active_bins) > 1:
        changed = False
        counts = [bin_count(b) for b in active_bins]
        min_i = int(np.argmin(counts))
        if counts[min_i] >= min_bin_count:
            break
        neighbor_i = min_i - 1 if min_i > 0 else min_i + 1
        merged_mask = active_bins[min_i]["mask"] | active_bins[neighbor_i]["mask"]
        merged_lo = min(active_bins[min_i]["lo"], active_bins[neighbor_i]["lo"])
        merged_hi = max(active_bins[min_i]["hi"], active_bins[neighbor_i]["hi"])
        new_bin = {"lo": merged_lo, "hi": merged_hi, "mask": merged_mask}
        keep_indices = [i for i in range(len(active_bins)) if i not in (min_i, neighbor_i)]
        active_bins = [active_bins[i] for i in keep_indices] + [new_bin]
        active_bins.sort(key=lambda b: b["lo"])
        n_bins_merged += 1
        changed = True

    abs_errors, weights = [], []
    for b in active_bins:
        mask = b["mask"]
        if not np.any(mask):
            continue
        p_bin = y_proba[mask].mean()
        o_bin = y_true[mask].mean()
        abs_errors.append(abs(o_bin - p_bin))
        weights.append(mask.sum() / n)

    if abs_errors:
        abs_errors = np.array(abs_errors)
        weights = np.array(weights)
        ece = float(np.sum(weights * abs_errors))
        mce = float(np.max(abs_errors))
    else:
        ece, mce = float("nan"), float("nan")

    try:
        ir = IsotonicRegression(out_of_bounds="clip")
        order = np.argsort(y_proba)
        p_sorted, y_sorted = y_proba[order], y_true[order]
        y_cal = ir.fit_transform(p_sorted, y_sorted)
        emax = float(np.max(np.abs(y_cal - p_sorted)))
    except Exception:
        emax = float("nan")

    return emax, ece, mce, n_bins_merged


def bootstrap_ci_for_metrics(
    y_true: np.ndarray, y_proba: np.ndarray, n_boot: int, rng: np.random.RandomState
) -> Dict[str, Tuple[float, float]]:
    n = len(y_true)
    auroc_s, auprc_s, brier_s = [], [], []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt, yp = y_true[idx], y_proba[idx]
        if len(np.unique(yt)) < 2:
            continue
        auroc_s.append(roc_auc_score(yt, yp))
        auprc_s.append(average_precision_score(yt, yp))
        brier_s.append(brier_score_loss(yt, yp))

    def ci(vals):
        return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)

    return {"AUROC_CI": ci(auroc_s), "AUPRC_CI": ci(auprc_s), "Brier_CI": ci(brier_s)}


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
        optimal_thresholds = {
            d: float(info.get("best_threshold", 0.5))
            for d, info in disease_models.items()
        }

    pred_path      = ARTIFACT_DIR / "predictor_cols.joblib"
    predictor_cols = joblib.load(pred_path) if pred_path.exists() else None

    return disease_models, optimal_thresholds, predictor_cols


# ============================================================
# CORE 1: Table 1/3 -- model performance (computed here directly)
# ============================================================
def make_table3_model_performance(
    disease_models: Dict[str, Any],
    thresholds: Dict[str, float],
    save_path: Path,
    n_bins: int = 10,
) -> pd.DataFrame:
    rng = np.random.RandomState(RANDOM_STATE)
    rows = []

    for disease, info in disease_models.items():
        if not isinstance(info, dict) or ("y_test" not in info) or ("y_proba" not in info):
            raise ValueError(
                f"disease_models[{disease}] must contain 'y_test' and 'y_proba'."
            )

        y_test  = np.asarray(info["y_test"],  dtype=int)
        y_proba = np.asarray(info["y_proba"], dtype=float)
        thr     = float(thresholds.get(disease, info.get("best_threshold", 0.5)))
        y_pred  = (y_proba >= thr).astype(int)

        cm   = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        ppv  = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        npv  = tn / (tn + fn) if (tn + fn) > 0 else np.nan
        nns  = 1.0 / ppv if ppv and ppv > 0 else np.nan

        emax, ece, mce, n_merged = compute_calibration_metrics(y_test, y_proba, n_bins=n_bins, min_bin_count=20)
        auroc = float(roc_auc_score(y_test, y_proba))
        auprc = float(average_precision_score(y_test, y_proba))
        brier = float(brier_score_loss(y_test, y_proba))

        cis = bootstrap_ci_for_metrics(y_test, y_proba, N_BOOTSTRAP, rng)

        rows.append({
            "Disease":      pretty_disease_name(disease),
            "N_test":       int(len(y_test)),
            "Prevalence_%": round(float(y_test.mean() * 100.0), 3),
            "Threshold":    round(thr, 3),
            "AUROC":        round(auroc, 3),
            "AUROC_CI_low": round(cis["AUROC_CI"][0], 3),
            "AUROC_CI_high":round(cis["AUROC_CI"][1], 3),
            "AUPRC":        round(auprc, 3),
            "AUPRC_CI_low": round(cis["AUPRC_CI"][0], 3),
            "AUPRC_CI_high":round(cis["AUPRC_CI"][1], 3),
            "Brier":        round(brier, 3),
            "Brier_CI_low": round(cis["Brier_CI"][0], 3),
            "Brier_CI_high":round(cis["Brier_CI"][1], 3),
            "Sensitivity":  round(float(sens), 3),
            "Specificity":  round(float(spec), 3),
            "PPV":          round(float(ppv),  3),
            "NPV":          round(float(npv),  3),
            "NNS":          round(float(nns), 2) if not np.isnan(nns) else np.nan,
            "Emax":         round(float(emax), 3),
            "ECE":          round(float(ece),  5),
            "MCE":          round(float(mce),  3),
            "MCE_bins_merged": n_merged,
        })

    df = pd.DataFrame(rows).set_index("Disease")
    df.to_csv(save_path)
    print("=== Table 1/3: recomputed from CURRENT model, with bootstrap CIs ===")
    print(df.to_string())
    return df


# ============================================================
# CORE 2: ROC + Calibration figures
# ============================================================
def plot_roc_curves(disease_models: Dict[str, Any], save_path: Path) -> None:
    diseases = list(disease_models.keys())
    fig, axes = plt.subplots(FIG_GRID_ROWS, FIG_GRID_COLS, figsize=(11, 4.2 * FIG_GRID_ROWS))
    axes = np.array(axes).reshape(-1)
    for ax, disease in zip(axes, diseases):
        y_test  = np.asarray(disease_models[disease]["y_test"],  dtype=int)
        y_proba = np.asarray(disease_models[disease]["y_proba"], dtype=float)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, label=f"AUROC={auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_title(pretty_disease_name(disease))
        ax.set_xlabel("False positive rate (1 - specificity)")
        ax.set_ylabel("True positive rate (sensitivity)")
        ax.grid(True)
        ax.legend(fontsize=8)
    for j in range(len(diseases), len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    _save_both(fig, save_path, dpi=300)
    plt.close(fig)


def plot_calibration_curves(disease_models: Dict[str, Any], save_path: Path, n_bins: int = 10) -> None:
    diseases = list(disease_models.keys())
    fig, axes = plt.subplots(FIG_GRID_ROWS, FIG_GRID_COLS, figsize=(11, 4.2 * FIG_GRID_ROWS))
    axes = np.array(axes).reshape(-1)
    for ax, disease in zip(axes, diseases):
        y_test  = np.asarray(disease_models[disease]["y_test"],  dtype=int)
        y_proba = np.asarray(disease_models[disease]["y_proba"], dtype=float)
        prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=n_bins, strategy="quantile")
        ax.plot(prob_pred, prob_true, marker="o", label="Model calibration")
        ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect")
        ax.set_title(pretty_disease_name(disease))
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.grid(True)
        ax.legend(fontsize=8)
    for j in range(len(diseases), len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    _save_both(fig, save_path, dpi=300)
    plt.close(fig)


# ============================================================
# CORE 3: Example-user profiles (Fig 3)
# ============================================================
FEATURE_LABELS = {
    "GENHLTH": "General health", "PHYSHLTH": "Physical health days",
    "MENTHLTH": "Mental health days", "POORHLTH": "Days poor health limited activities",
    "BPHIGH4": "High blood pressure", "BPMEDS": "Blood pressure medication",
    "TOLDHI2": "Told high cholesterol", "CHOLCHK": "Cholesterol checked recently",
    "SMOKE100": "Ever smoked 100 cigarettes", "SMOKDAY2": "Current smoking frequency",
    "ALCDAY5": "Alcohol drinking days (per day rate)", "DRNKANY5": "Any alcohol use",
    "EXERANY2": "Any exercise in past month", "FRUIT1": "Fruit intake (per day rate)",
    "VEGETAB1": "Vegetable intake (per day rate)", "ASTHMA3": "Asthma",
    "HAVARTH3": "Arthritis", "DIFFWALK": "Difficulty walking/climbing",
    "DECIDE": "Cognitive difficulty (memory/concentration)", "PERSDOC2": "Has personal doctor",
    "HLTHPLN1": "Has health insurance", "MEDCOST": "Could not see doctor (cost)",
    "CHECKUP1": "Time since last checkup", "_BMI5": "BMI", "WEIGHT2": "Weight (lb)",
    "HEIGHT3": "Height (feet and inches code)", "_AGEG5YR": "Age group", "_EDUCAG": "Education category",
    "_INCOMG": "Income category", "_MRACE1": "Race", "_HISPANC": "Hispanic ethnicity",
    "_STATE": "State", "SEX": "Sex",
}

NOMINAL_CATEGORICAL_VARS = {
    "_STATE", "SEX", "_MRACE1", "_HISPANC", "SMOKE100", "SMOKDAY2",
    "DRNKANY5", "EXERANY2", "HLTHPLN1", "PERSDOC2", "MEDCOST",
    "BPHIGH4", "BPMEDS", "TOLDHI2", "ASTHMA3", "HAVARTH3",
    "DIFFWALK", "DECIDE",
}

_BINARY_YES_NO = {1: "Yes", 2: "No"}

CATEGORY_VALUE_LABELS = {
    "SEX":      {1: "Male", 2: "Female"},
    "BPHIGH4":  {1: "Yes", 2: "No", 3: "Borderline/pre-hypertensive", 4: "Yes, during pregnancy only"},
    "BPMEDS":   _BINARY_YES_NO, "TOLDHI2": _BINARY_YES_NO, "ASTHMA3": _BINARY_YES_NO,
    "HAVARTH3": _BINARY_YES_NO, "DIFFWALK": _BINARY_YES_NO, "DECIDE": _BINARY_YES_NO,
    "SMOKE100": _BINARY_YES_NO, "DRNKANY5": _BINARY_YES_NO, "EXERANY2": _BINARY_YES_NO,
    "HLTHPLN1": _BINARY_YES_NO, "MEDCOST": _BINARY_YES_NO,
    "SMOKDAY2": {1: "Every day", 2: "Some days", 3: "Not at all"},
    "PERSDOC2": {1: "Yes, one", 2: "Yes, more than one", 3: "No"},
    "_MRACE1":  {1: "White", 2: "Black", 3: "Native American/Alaska Native", 4: "Asian",
                 5: "Native Hawaiian/Pacific Islander", 6: "Other race", 7: "Multiracial"},
    "_HISPANC": {1: "Hispanic", 2: "Non-Hispanic"},
    "_STATE":   {6: "California", 13: "Georgia", 24: "Maryland"},
}


def _split_categorical_feature(base_name: str) -> Optional[Tuple[str, float]]:
    for var in sorted(NOMINAL_CATEGORICAL_VARS, key=len, reverse=True):
        prefix = var + "_"
        if base_name == var:
            return None
        if base_name.startswith(prefix):
            suffix = base_name[len(prefix):]
            try:
                return (var, float(suffix))
            except ValueError:
                continue
    return None


def pretty_feature_from_explainer(name: str) -> str:
    s = str(name)
    if s.lower().startswith(("num__", "cat__")):
        s = s[5:]
    split = _split_categorical_feature(s)
    if split is not None:
        var, code = split
        var_label = FEATURE_LABELS.get(var, var.lstrip("_").replace("_", " "))
        code_int = int(code) if float(code).is_integer() else code
        value_labels = CATEGORY_VALUE_LABELS.get(var, {})
        value_label = value_labels.get(code_int, f"code {code_int}")
        return f"{var_label}: {value_label}"
    bare = s.lstrip("_")
    if s in FEATURE_LABELS:
        return FEATURE_LABELS[s]
    if bare in FEATURE_LABELS:
        return FEATURE_LABELS[bare]
    if f"_{bare}" in FEATURE_LABELS:
        return FEATURE_LABELS[f"_{bare}"]
    return s.replace("_", " ").strip()


def predict_user_all(input_dict, disease_models, predictor_cols):
    row = (pd.DataFrame([input_dict]).reindex(columns=predictor_cols, fill_value=np.nan)
           if predictor_cols is not None else pd.DataFrame([input_dict]))
    out = []
    for disease, info in disease_models.items():
        model = info["model"]
        p = float(model.predict_proba(row)[0, 1])
        thr = float(info.get("best_threshold", 0.5))
        out.append({
            "disease": disease, "probability": p, "threshold_used": thr,
            "classification": "Above threshold (likely existing condition)" if p >= thr else "Below threshold",
        })
    return pd.DataFrame(out).set_index("disease")


def explain_user_shap(input_dict, disease, disease_models, predictor_cols, top_n=10):
    if not HAS_SHAP or disease not in disease_models:
        return None
    info, model = disease_models[disease], disease_models[disease]["model"]
    row = (pd.DataFrame([input_dict]).reindex(columns=predictor_cols, fill_value=np.nan)
           if predictor_cols is not None else pd.DataFrame([input_dict]))
    if not hasattr(model, "named_steps") or "preprocessor" not in model.named_steps or "clf" not in model.named_steps:
        return None
    pre, clf = model.named_steps["preprocessor"], model.named_steps["clf"]
    X_t = pre.transform(row)
    try:
        feature_names = pre.get_feature_names_out()
    except Exception:
        feature_names = np.array([f"f{i}" for i in range(X_t.shape[1])], dtype=object)
    try:
        import shap as shap_lib
        explainer = shap_lib.TreeExplainer(clf)
        sv = explainer.shap_values(X_t)
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


# NOTE: FRUIT1 / VEGETAB1 / ALCDAY5 are the DECODED "times per day" rates that the
# trained pipeline expects (the decode runs in data_preparation.py, not in the saved
# pipeline). Raw BRFSS codes (e.g. 205, 201) would be read as 205 servings / 201 drinks
# per day. Decode: 1xx -> xx/day (FRUIT1, VEGETAB1) or xx/7 (ALCDAY5); 2xx -> xx/7
# (FRUIT1, VEGETAB1) or xx/30 (ALCDAY5).
# AGE NOTE: _AGEG5YR is a 5-year band. Example 1 = 13 (80+), Example 2 = 7 (50-54, "middle-aged"),
# Example 3 = 3 (30-34, "younger adult").
EXAMPLE_USERS = {
    "Example 1: Older female, multiple CVD risk factors": {
        "_STATE": 13, "SEX": 2, "_AGEG5YR": 13, "_EDUCAG": 2, "_INCOMG": 2, "_MRACE1": 2,
        "_HISPANC": 2, "SMOKE100": 1, "SMOKDAY2": 2, "ALCDAY5": 0.0333, "DRNKANY5": 1,
        "EXERANY2": 2, "FRUIT1": 0.71, "VEGETAB1": 0.71, "HLTHPLN1": 1, "PERSDOC2": 1,
        "MEDCOST": 2, "CHECKUP1": 1, "BPHIGH4": 1, "BPMEDS": 1, "TOLDHI2": 1, "CHOLCHK": 1,
        "ASTHMA3": 2, "HAVARTH3": 1, "GENHLTH": 3, "PHYSHLTH": 10, "MENTHLTH": 5,
        "POORHLTH": 7, "DIFFWALK": 1, "DECIDE": 2, "WEIGHT2": 210, "HEIGHT3": 504, "_BMI5": 3600,
    },
    "Example 2: Middle-aged, favourable health indicators": {
        "_STATE": 24, "SEX": 1, "_AGEG5YR": 7, "_EDUCAG": 3, "_INCOMG": 4, "_MRACE1": 1,
        "_HISPANC": 2, "SMOKE100": 2, "SMOKDAY2": 3, "ALCDAY5": 0.1429, "DRNKANY5": 2,
        "EXERANY2": 1, "FRUIT1": 1.0, "VEGETAB1": 1.0, "HLTHPLN1": 1, "PERSDOC2": 1,
        "MEDCOST": 2, "CHECKUP1": 2, "BPHIGH4": 2, "BPMEDS": 2, "TOLDHI2": 2, "CHOLCHK": 1,
        "ASTHMA3": 2, "HAVARTH3": 2, "GENHLTH": 2, "PHYSHLTH": 0, "MENTHLTH": 0,
        "POORHLTH": 0, "DIFFWALK": 2, "DECIDE": 2, "WEIGHT2": 170, "HEIGHT3": 507, "_BMI5": 2650,
    },
    "Example 3: Younger adult, prominent mental health indicators": {
        "_STATE": 6, "SEX": 2, "_AGEG5YR": 3, "_EDUCAG": 4, "_INCOMG": 5, "_MRACE1": 1,
        "_HISPANC": 1, "SMOKE100": 2, "SMOKDAY2": 3, "ALCDAY5": 0.4286, "DRNKANY5": 1,
        "EXERANY2": 1, "FRUIT1": 1.0, "VEGETAB1": 0.14, "HLTHPLN1": 1, "PERSDOC2": 2,
        "MEDCOST": 1, "CHECKUP1": 3, "BPHIGH4": 2, "BPMEDS": 2, "TOLDHI2": 2, "CHOLCHK": 1,
        "ASTHMA3": 1, "HAVARTH3": 2, "GENHLTH": 3, "PHYSHLTH": 3, "MENTHLTH": 15,
        "POORHLTH": 5, "DIFFWALK": 2, "DECIDE": 1, "WEIGHT2": 150, "HEIGHT3": 505, "_BMI5": 2500,
    },
}


def make_example_inputs_table(tables_dir: Path) -> pd.DataFrame:
    example_labels = list(EXAMPLE_USERS.keys())
    all_vars = list(next(iter(EXAMPLE_USERS.values())).keys())
    rows = []
    for var in all_vars:
        pretty_label = FEATURE_LABELS.get(var, var.lstrip("_").replace("_", " "))
        row = {"Variable": var, "Label": {"_BMI5": "BMI (x100)"}.get(var, pretty_label)}
        for ex_label in example_labels:
            val = EXAMPLE_USERS[ex_label].get(var)
            if var in CATEGORY_VALUE_LABELS and val in CATEGORY_VALUE_LABELS[var]:
                row[ex_label] = f"{val} ({CATEGORY_VALUE_LABELS[var][val]})"
            else:
                row[ex_label] = val
        rows.append(row)
    df = pd.DataFrame(rows)
    out_path = tables_dir / "table_example_inputs.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved -> {out_path}")
    return df


def write_example_outputs(disease_models, predictor_cols, tables_dir, figures_dir):
    pred_rows, shap_rows = [], []
    for label, user in EXAMPLE_USERS.items():
        pred_df = predict_user_all(user, disease_models, predictor_cols)
        for dis, r in pred_df.iterrows():
            pred_rows.append({
                "Example": label, "Disease_internal": dis, "Disease": pretty_disease_name(dis),
                "Predicted_probability": float(r["probability"]), "Threshold_used": float(r["threshold_used"]),
                "Classification": r["classification"],
            })
        dis_explain = pred_df["probability"].idxmax()
        contrib = explain_user_shap(user, dis_explain, disease_models, predictor_cols, top_n=10)
        if contrib is not None and not contrib.empty:
            for rank, (_, row) in enumerate(contrib.iterrows(), start=1):
                shap_rows.append({
                    "Example": label, "Explained_disease": pretty_disease_name(dis_explain), "Rank": rank,
                    "Feature": row["feature"], "Feature_pretty": row["feature_pretty"],
                    "SHAP_value": float(row["shap_value"]),
                })

    tableA, tableB = pd.DataFrame(pred_rows), pd.DataFrame(shap_rows)
    tableA.to_csv(tables_dir / "tableA_example_predictions.csv", index=False)
    tableB.to_csv(tables_dir / "tableB_example_explanations.csv", index=False)
    make_example_inputs_table(tables_dir)

    diseases_order = ["heart_attack", "coronary_hd", "stroke", "kidney", "depression", "diabetes"]
    examples = list(EXAMPLE_USERS.keys())
    n_cols = 2
    n_rows = int(np.ceil(len(examples) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = np.array(axes).reshape(-1)
    for ax, label in zip(axes, examples):
        pred_df = predict_user_all(EXAMPLE_USERS[label], disease_models, predictor_cols)
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
    _save_both(fig, figures_dir / "example_predicted_probabilities.pdf", dpi=300)
    plt.close(fig)

    if HAS_SHAP and not tableB.empty:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
        axes = np.array(axes).reshape(-1)
        for ax, label in zip(axes, examples):
            pred_df_this = predict_user_all(EXAMPLE_USERS[label], disease_models, predictor_cols)
            dis_explain = pred_df_this["probability"].idxmax()
            contrib = explain_user_shap(EXAMPLE_USERS[label], dis_explain, disease_models, predictor_cols, top_n=10)
            if contrib is None or contrib.empty:
                ax.axis("off")
                continue
            dfp = contrib.sort_values("shap_value")
            colors = ["steelblue" if v < 0 else "firebrick" for v in dfp["shap_value"]]
            ax.barh(dfp["feature_pretty"], dfp["shap_value"], color=colors)
            ax.axvline(0, linewidth=0.8, color="black")
            ax.set_title(f"{label}\n{pretty_disease_name(dis_explain)} - SHAP")
            ax.set_xlabel("SHAP value (impact on prediction)")
            ax.tick_params(axis="y", labelsize=8)
        for j in range(len(examples), len(axes)):
            axes[j].axis("off")
        plt.tight_layout()
        _save_both(fig, figures_dir / "example_shap_explanations.pdf", dpi=300)
        plt.close(fig)


# ============================================================
# CORE 4: Combined Table 1 (demographics + disease prevalence)
# ============================================================
def _disease_prevalence_rows(df: pd.DataFrame, year_col: str, years: list) -> List[dict]:
    disease_cols = DISEASE_NAME_MAP
    missing = [c for c in disease_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing disease columns in cleaned dataset: {missing}")
    rows = []
    for dis_code, dis_name in disease_cols.items():
        row = {"Category": dis_name}
        for yr in years:
            sub = df[df[year_col] == yr]
            nonmiss = int(sub[dis_code].notna().sum())
            cases = float(sub[dis_code].sum())
            prev = (cases / nonmiss) * 100 if nonmiss > 0 else np.nan
            row[str(int(yr))] = f"{int(cases):,} / {nonmiss:,} ({prev:.2f}%)" if nonmiss > 0 else "n/a"
        nonmiss_all = int(df[dis_code].notna().sum())
        cases_all = float(df[dis_code].sum())
        prev_all = (cases_all / nonmiss_all) * 100 if nonmiss_all > 0 else np.nan
        row["All years"] = f"{int(cases_all):,} / {nonmiss_all:,} ({prev_all:.2f}%)" if nonmiss_all > 0 else "n/a"
        rows.append(row)
    return rows


def make_table1_wide(df: pd.DataFrame, include_prevalence: bool = False) -> pd.DataFrame:
    year_col = "YEAR" if "YEAR" in df.columns else "IYEAR"
    years = sorted(df[year_col].dropna().unique())
    all_year_labels = [str(int(y)) for y in years] + ["All years"]

    sex_map = {1: "Male", 2: "Female"}
    age_labels = {1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44", 6: "45-49",
                  7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69", 11: "70-74", 12: "75-79", 13: "80+"}
    educ_map = {1: "< High school", 2: "High school graduate", 3: "Some college", 4: "College graduate"}
    income_labels = {1: "<$15k", 2: "$15-25k", 3: "$25-35k", 4: "$35-50k", 5: "$50k or more"}   # _INCOMG has 5 categories; 9 = don't know/refused -> missing
    race_map = {1: "White", 2: "Black", 3: "Native American/Alaska Native", 4: "Asian",
                5: "Native Hawaiian/Pacific Islander", 6: "Other race", 7: "Multiracial"}
    hisp_map = {1: "Hispanic", 2: "Non-Hispanic"}

    variable_sets = [
        ("Sex", "SEX", sex_map), ("Age group", "_AGEG5YR", age_labels),
        ("Education", "_EDUCAG", educ_map), ("Income", "_INCOMG", income_labels),
        ("Race", "_MRACE1", race_map), ("Hispanic ethnicity", "_HISPANC", hisp_map),
    ]

    def cell_value(count, total):
        return "0 (0.0%)" if total == 0 else f"{count:,} ({100*count/total:.1f}%)"

    out_rows = []
    for section, col, label_map in variable_sets:
        if col not in df.columns:
            continue
        for cat in sorted(df[col].dropna().unique()):
            row = {"Section": section, "Category": label_map.get(cat, str(cat))}
            for yr in years:
                sub = df[df[year_col] == yr]
                row[str(int(yr))] = cell_value(int((sub[col] == cat).sum()), sub.shape[0])
            row["All years"] = cell_value(int((df[col] == cat).sum()), df.shape[0])
            out_rows.append(row)
        # explicit missing row (don't know / refused / not asked / not collected)
        miss = {"Section": section, "Category": "Missing / not collected"}
        for yr in years:
            sub = df[df[year_col] == yr]
            miss[str(int(yr))] = cell_value(int(sub[col].isna().sum()), sub.shape[0])
        miss["All years"] = cell_value(int(df[col].isna().sum()), df.shape[0])
        out_rows.append(miss)

    if "BMI" in df.columns:
        def bmi_summary(series):
            series = series.dropna()
            if series.empty:
                return ""
            return (f"{series.mean():.1f} ({series.std():.1f}); {series.median():.1f} "
                    f"[{series.quantile(0.25):.1f}-{series.quantile(0.75):.1f}]")
        row = {"Section": "BMI (continuous)", "Category": "BMI mean (SD); median [Q1-Q3]"}
        for yr in years:
            row[str(int(yr))] = bmi_summary(df.loc[df[year_col] == yr, "BMI"])
        row["All years"] = bmi_summary(df["BMI"])
        out_rows.append(row)

    table = pd.DataFrame(out_rows)[["Section", "Category"] + all_year_labels]

    if include_prevalence:
        prev_rows = _disease_prevalence_rows(df, year_col, years)
        for row in prev_rows:
            row["Section"] = "Disease prevalence"
        prev_df = pd.DataFrame(prev_rows)[["Section", "Category"] + all_year_labels]
        table = pd.concat([table, prev_df], ignore_index=True)

    return table


def make_combined_table1(df: pd.DataFrame, save_path: Path) -> pd.DataFrame:
    year_col = "YEAR" if "YEAR" in df.columns else "IYEAR"
    years = sorted(df[year_col].dropna().unique())
    all_year_labels = [str(int(y)) for y in years] + ["All years"]

    table1 = make_table1_wide(df, include_prevalence=False)
    prev_rows = _disease_prevalence_rows(df, year_col, years)
    for row in prev_rows:
        row["Section"] = "Disease prevalence"
    prev_df = pd.DataFrame(prev_rows)[["Section", "Category"] + all_year_labels]

    combined = pd.concat([table1, prev_df], ignore_index=True)
    combined.to_csv(save_path, index=False)
    print(f"  Saved combined Table 1 (+ disease prevalence) -> {save_path}")
    return combined


# ============================================================
# NEW: GBM vs RF vs LR comparison figure (reads existing CSV)
# ============================================================
def make_baseline_comparison_outputs(tables_dir: Path, figures_dir: Path) -> Optional[pd.DataFrame]:
    src = tables_dir / "baseline_comparison" / "baseline_model_comparison.csv"
    if not src.exists():
        print(f"  [SKIP] Baseline comparison not found at {src}.")
        return None

    df = pd.read_csv(src)
    print(f"  Loaded baseline comparison from {src}")

    diseases = df["disease"].tolist()
    fig, axes = plt.subplots(FIG_GRID_ROWS, FIG_GRID_COLS, figsize=(11, 4.2 * FIG_GRID_ROWS))
    axes = np.array(axes).reshape(-1)
    for ax, (_, row) in zip(axes, df.iterrows()):
        models = ["Logistic\nRegression", "Random\nForest", "Gradient\nBoosting"]
        aurocs = [row["AUROC_lr"], row["AUROC_rf"], row["AUROC_gbm"]]
        colors = ["#888888", "#4C72B0", "#C44E52"]
        bars = ax.bar(models, aurocs, color=colors)
        for bar, val in zip(bars, aurocs):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_ylim(min(aurocs) - 0.05, max(aurocs) + 0.03)
        ax.set_ylabel("AUROC")
        ax.set_title(row["disease_pretty"])
        ax.grid(True, axis="y", alpha=0.3)
    for j in range(len(diseases), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Model comparison: Gradient Boosting vs. Random Forest vs. Logistic Regression", fontsize=13, y=1.02)
    plt.tight_layout()
    fig_path = figures_dir / "fig_model_comparison_gbm_rf_lr.pdf"
    _save_both(fig, fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {fig_path} (+ .png)")

    metrics = [("AUROC", "AUROC_lr", "AUROC_rf", "AUROC_gbm", "higher is better"),
               ("AUPRC", "AUPRC_lr", "AUPRC_rf", "AUPRC_gbm", "higher is better"),
               ("Brier", "Brier_lr", "Brier_rf", "Brier_gbm", "lower is better")]
    fig2, axes2 = plt.subplots(3, 1, figsize=(9, 13))
    x = np.arange(len(diseases))
    width = 0.25
    for ax, (metric_name, lr_col, rf_col, gbm_col, direction) in zip(axes2, metrics):
        bars_lr  = ax.bar(x - width, df[lr_col],  width, label="Logistic Regression", color="#888888")
        bars_rf  = ax.bar(x,         df[rf_col],  width, label="Random Forest",       color="#4C72B0")
        bars_gbm = ax.bar(x + width, df[gbm_col], width, label="Gradient Boosting",   color="#C44E52")
        for bars in (bars_lr, bars_rf, bars_gbm):
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height, _format_sig_figs(height, sig=2),
                        ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(df["disease_pretty"], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(metric_name)
        ax.set_title(f"{metric_name} ({direction})")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(0, float(df[[lr_col, rf_col, gbm_col]].values.max()) * 1.15)
    axes2[0].legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    fig2_path = figures_dir / "fig_model_comparison_all_metrics.pdf"
    _save_both(fig2, fig2_path, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved -> {fig2_path} (+ .png)")


    # --- Formatted summary table for manuscript: AUROC, AUPRC and Brier ---
    # Sign convention for deltas is first-named model minus second-named model.
    # AUROC/AUPRC: positive favours the first-named model.
    # Brier: negative favours the first-named model (lower Brier is better).
    metrics_for_table = ["AUROC", "AUPRC", "Brier"]
    pairs = [("gbm", "lr"), ("gbm", "rf"), ("rf", "lr")]

    for metric in metrics_for_table:
        for a, b in pairs:
            delta_col = f"{metric}_{a}_minus_{b}"
            if delta_col not in df.columns:
                col_a, col_b = f"{metric}_{a}", f"{metric}_{b}"
                if col_a in df.columns and col_b in df.columns:
                    df[delta_col] = df[col_a] - df[col_b]

    summary_cols = ["disease_pretty"]
    for metric in metrics_for_table:
        summary_cols += [f"{metric}_lr", f"{metric}_rf", f"{metric}_gbm"]
        summary_cols += [f"{metric}_{a}_minus_{b}" for a, b in pairs]

    missing = [c for c in summary_cols if c not in df.columns]
    if missing:
        print(f"  [WARNING] These expected columns are not in "
              f"baseline_model_comparison.csv and were skipped: {missing}")
    summary_cols = [c for c in summary_cols if c in df.columns]

    summary_df = df[summary_cols].rename(columns={"disease_pretty": "Disease"})
    summary_df = summary_df.round(4)
    out_path = tables_dir / "model_comparison_summary_formatted.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"  Saved formatted summary ({len(summary_cols) - 1} metric columns) -> {out_path}")
    return df


def plot_global_shap_bar_chart(tables_dir: Path, figures_dir: Path) -> None:
    ranking_path = tables_dir / "global_shap_full_ranking_all_outcomes.csv"
    if not ranking_path.exists():
        print(f"  [SKIP] {ranking_path} not found.")
        return
    ranking = pd.read_csv(ranking_path)
    diseases = list(DISEASE_NAME_MAP.keys())
    fig, axes = plt.subplots(FIG_GRID_ROWS, FIG_GRID_COLS, figsize=(13, 4.5 * FIG_GRID_ROWS))
    axes = np.array(axes).reshape(-1)
    TOP_N_SHAP_PLOT = 15
    for ax, disease in zip(axes, diseases):
        sub = ranking[ranking["disease"] == disease].sort_values("mean_abs_shap", ascending=False).head(TOP_N_SHAP_PLOT)
        sub = sub.iloc[::-1]
        ax.barh(sub["feature_pretty"], sub["mean_abs_shap"], color="#4C72B0")
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(pretty_disease_name(disease))
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(True, axis="x", alpha=0.3)
    for j in range(len(diseases), len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"Global feature importance (top {TOP_N_SHAP_PLOT} by mean |SHAP value|)", fontsize=14, y=1.01)
    plt.tight_layout()
    out_path = figures_dir / "fig_global_shap_bar_all_outcomes.pdf"
    _save_both(fig, out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path} (+ .png)")


def make_global_shap_beeswarm_figures(tables_dir: Path, figures_dir: Path) -> None:
    shap_dir = tables_dir / "global_shap"
    if not shap_dir.exists():
        print(f"  [SKIP] Global SHAP data not found at {shap_dir}.")
        return
    print("  Bar chart (top-15, all outcomes):")
    plot_global_shap_bar_chart(tables_dir, figures_dir)
    print("  Violin-style SHAP summary figure:")
    plot_combined_violin(tables_dir=tables_dir, figures_dir=figures_dir)


# ============================================================
# ============================================================
# MANUSCRIPT STATUS REGISTRY -- the single source of truth for
# "what tables/figures exist in this manuscript, where do they come
# from, what do they show, and is their output file present/fresh."
#
# section        : "Main Table" | "Main Fig" | "S Table" | "S Fig" | "S Dataset"
# item            : display label, e.g. "Table 1", "S4 Table"
# title           : manuscript caption title (short form)
# path            : Path to the expected output file, or None if this
#                    script computes it directly (status = CODE)
# generator       : which script produces `path`
# reviewer        : reviewer comment / TRIPOD+AI item this satisfies
# note            : one-line explanation of what this item shows /
#                    why it's in the manuscript
# required_cols   : optional list of columns to spot-check for (CSV only)
# confirmed       : True if I have personally seen the generating
#                    script's code and confirmed this exact filename;
#                    False if the filename is inferred from your
#                    descriptions and needs your verification
# ============================================================
# ============================================================

# ------------------------------------------------------------
# Labels used in the CURRENT manuscript / Supporting Information. The registry below was
# written under an earlier numbering; this map translates every entry to the labels the
# manuscript now cites, so the status report matches the submitted files.
#   old (section, item)        -> new (section, item)
# ------------------------------------------------------------
_CURRENT_LABELS = {
    ("Main Table", "Table 2"):  ("S Dataset", "S2 Dataset"),    # partition counts
    ("Main Fig", "Fig 2"):      ("S Fig", "S4 Fig"),            # calibration plots
    ("Main Fig", "Fig 3"):      ("Main Fig", "Fig 4"),          # example profiles + SHAP
    ("Main Fig", "Fig 4"):      ("Main Fig", "Fig 5"),          # participant flow
    ("S Table", "S2 Table"):    ("S Dataset", "S10 Dataset"),   # TRIPOD+AI checklist
    ("S Table", "S3 Table"):    ("S Table", "S3 Table"),        # GBM vs RF vs LR
    ("S Table", "S4 Table"):    ("S Table", "S2 Table"),        # restricted-years performance
    ("S Table", "S5 Table"):    ("S Dataset", "S4 Dataset"),    # restricted-years SHAP rank stability
    ("S Table", "S6 Table"):    ("S Table", "S4 Table"),        # PPV/NPV vs prevalence
    ("S Table", "S7 Table"):    ("S Table", "S5 Table"),        # decision-curve summary
    ("S Table", "S8 Table"):    ("S Dataset", "S5 Dataset"),    # top-15 SHAP
    ("S Table", "S9 Table"):    ("S Dataset", "S6 Dataset"),    # full SHAP ranking
    ("S Table", "S10 Table"):   ("S Dataset", "S7 Dataset"),    # SHAP rank, structurally-missing predictors
    ("S Table", "S11 Table"):   ("S Dataset", "S8 Dataset"),    # subgroup performance
    ("S Table", "S12 Table"):   ("S Dataset", "S9 Dataset"),    # largest subgroup gaps
    ("S Table", "S13 Table"):   ("S Dataset", "S1 Dataset"),    # full vs analytic cohort
    ("S Table", "S14 Table"):   ("S Dataset", "S3 Dataset"),    # development vs evaluation
    ("S Table", "S15 Table"):   ("Main Table", "Table 2"),      # comparison with prior studies
    ("S Fig", "S1 Fig"):        ("S Fig", "S5 Fig"),            # web interface screenshot
    ("S Fig", "S2 Fig"):        ("S Fig", "S1 Fig"),            # PPV vs prevalence
    ("S Fig", "S3 Fig"):        ("S Fig", "S2 Fig"),            # decision curves
    ("S Fig", "S4 Fig"):        ("S Fig", "S3 Fig"),            # global SHAP bar chart
    ("S Fig", "S5 Fig"):        ("Main Fig", "Fig 3"),          # global SHAP beeswarm
    ("S Dataset", "S1 Dataset"): ("S Dataset", "S13 Dataset"),  # participant flow
    ("S Dataset", "S2 Dataset"): ("S Dataset", "S14 Dataset"),  # per-outcome flow
    ("S Dataset", "S3 Dataset"): ("S Dataset", "S11 Dataset"),  # predictor missingness
}


def _apply_current_labels(entries: List[dict], T: Path, F: Path) -> List[dict]:
    relabelled = []
    for e in entries:
        key = (e["section"], e["item"])
        e = dict(e)
        e["section"], e["item"] = _CURRENT_LABELS.get(key, key)
        relabelled.append(e)

    # Items that exist in the current manuscript but were missing from the old registry.
    relabelled += [
        {"section": "Main Fig", "item": "Fig 2",
         "title": "GBM vs. logistic regression vs. random forest (AUROC, AUPRC, Brier)",
         "path": F / "fig_model_comparison_all_metrics.pdf",
         "generator": "baseline_model_comparison.py -> make_manuscript_outputs.py",
         "reviewer": "", "note": "Baseline comparison figure.",
         "required_cols": None, "confirmed": False},
        {"section": "S Table", "item": "S6 Table",
         "title": "Characteristics of the three illustrative participant profiles",
         "path": T / "table_example_inputs.csv",
         "generator": "make_manuscript_outputs.py (this script, CORE)",
         "reviewer": "", "note": "Decoded inputs for Fig 4 profiles.",
         "required_cols": None, "confirmed": True},
        {"section": "S Dataset", "item": "S12 Dataset",
         "title": "TRIPOD+AI for Abstracts checklist", "path": None,
         "generator": "manual", "reviewer": "TRIPOD+AI for Abstracts",
         "note": "Update numbers by hand after the re-run.",
         "required_cols": None, "confirmed": True},
        {"section": "S Dataset", "item": "S15 Dataset",
         "title": "Subgroup calibration (slope, intercept, O/E, ECE)",
         "path": T / "fairness" / "subgroup_calibration.csv",
         "generator": "subgroup_calibration.py", "reviewer": "",
         "note": "Calibration within sex, age, race/ethnicity and income subgroups.",
         "required_cols": ["calibration_slope", "calibration_intercept", "obs_exp_ratio", "ECE"],
         "confirmed": True},
    ]

    def _nat(e):
        m = re.search(r"(\d+)", e["item"])
        return (int(m.group(1)) if m else 0, e["item"])

    order = ["Main Table", "Main Fig", "S Table", "S Fig", "S Dataset"]
    relabelled.sort(key=lambda e: (order.index(e["section"]), _nat(e)))
    return relabelled


def _registry() -> List[dict]:
    T = TABLES_DIR
    F = FIGURES_DIR
    entries = [
        # ---------------- MAIN TEXT TABLES ----------------
        {
            "section": "Main Table", "item": "Table 1",
            "title": "Predictive performance across six outcomes",
            "path": T / "table3_model_performance_updated.csv",
            "generator": "make_manuscript_outputs.py (this script, CORE)",
            "reviewer": "",
            "note": "AUROC/AUPRC/Brier with 95% bootstrap CIs, NNS, post-bin-merge MCE. Core result table.",
            "required_cols": ["AUROC", "AUROC_CI_low", "AUROC_CI_high", "NNS", "MCE", "MCE_bins_merged"],
            "confirmed": True,
        },
        {
            "section": "Main Table", "item": "Table 2",
            "title": "Participant and outcome-event counts by analysis stage",
            "path": T / "tripod_item21_partition_counts.csv",
            "generator": "verify_train_test_equivalence.py",  # UNCONFIRMED -- never seen this script
            "reviewer": "TRIPOD+AI Item 21",
            "note": "Exact N/events per outcome per train/val/test partition. Long format: 6 diseases x 3 partitions = 18 rows (this is correct, not a bug).",
            "required_cols": None,
            "confirmed": False,
        },
        # ---------------- MAIN TEXT FIGURES ----------------
        {
            "section": "Main Fig", "item": "Fig 1",
            "title": "ROC curves, six outcomes",
            "path": F / "fig_roc_curves.pdf",
            "generator": "make_manuscript_outputs.py (this script, CORE)",
            "reviewer": "Original submission figure",
            "note": "Discrimination visualization, one panel per disease.",
            "required_cols": None, "confirmed": True,
        },
        {
            "section": "Main Fig", "item": "Fig 2",
            "title": "Calibration plots, six outcomes",
            "path": F / "fig_calibration_curves.pdf",
            "generator": "make_manuscript_outputs.py (this script, CORE)",
            "reviewer": "Original submission figure",
            "note": "Quantile-binned calibration curves, one panel per disease.",
            "required_cols": None, "confirmed": True,
        },
        {
            "section": "Main Fig", "item": "Fig 3",
            "title": "Example profiles + SHAP explanations",
            "path": F / "example_shap_explanations.pdf",
            "generator": "make_manuscript_outputs.py (this script, CORE)",
            "reviewer": "",
            "note": "⚠️ FIGURE IS CURRENT, but the manuscript's Example 1 NARRATIVE TEXT describes the wrong "
                    "explained disease (CHD instead of the corrected CKD) -- fix the Results prose, not this figure.",
            "required_cols": None, "confirmed": True,
        },
        {
            "section": "Main Fig", "item": "Fig 4",
            "title": "Participant flow diagram",
            "path": T / "fig4_flow_summary.csv",
            "generator": "generate_participant_flow.py",
            "reviewer": "TRIPOD+AI Item 20a",
            "note": "⚠️ Must be regenerated -- original diagram had a diabetes arithmetic error "
                    "(Included+Excluded != cohort total). Run generate_participant_flow.py and verify the [OK] cross-check.",
            "required_cols": None, "confirmed": True,
        },
        # ---------------- SUPPLEMENTARY TABLES ----------------
        {
            "section": "S Table", "item": "S1 Table",
            "title": "Descriptive characteristics of BRFSS respondents",
            "path": T / "table1_combined_with_prevalence.csv",
            "generator": "make_manuscript_outputs.py (this script, CORE)",
            "reviewer": "Original submission table",
            "note": "Demographics by year + disease prevalence by year, single combined file.",
            "required_cols": None, "confirmed": True,
        },
        {
            "section": "S Table", "item": "S2 Table",
            "title": "TRIPOD+AI reporting checklist",
            "path": None,  # lives as TRIPOD.docx, not a CSV
            "generator": "manual (TRIPOD.docx)",
            "reviewer": "",
            "note": "27-item checklist. Item 23a currently cites stale 'S13 Table' -- needs fixing to S11 Table.",
            "required_cols": None, "confirmed": True,
        },
        {
            "section": "S Table", "item": "S3 Table",
            "title": "GBM vs. RF vs. LR comparison",
            "path": T / "model_comparison_summary_formatted.csv",
            "generator": "baseline_model_comparison.py -> make_manuscript_outputs.py",  # UNCONFIRMED for baseline_model_comparison.py itself
            "reviewer": "",
            "note": "Justifies gradient boosting over simpler baselines.",
            "required_cols": None, "confirmed": False,
        },
        {
            "section": "S Table", "item": "S4 Table",
            "title": "Restricted-years (2013+2015) performance comparison",
            "path": T / "restricted_years_sensitivity" / "restricted_years_performance_comparison.csv",
            "generator": "restricted_years_sensitivity.py",
            "reviewer": "",
            "note": "AUROC/AUPRC/Brier deltas, full vs. years-with-complete-predictor-administration. Pre-computed deltas confirmed present.",
            "required_cols": ["AUROC_full", "AUROC_restricted", "AUROC_delta"],
            "confirmed": True,
        },
        {
            "section": "S Table", "item": "S5 Table",
            "title": "Restricted-years SHAP rank stability",
            "path": T / "restricted_years_sensitivity" / "restricted_years_shap_rank_stability.csv",
            "generator": "restricted_years_sensitivity.py",
            "reviewer": "",
            "note": "Confirms the 10 structurally-missing predictors aren't artifacts of imputed years. "
                    "Check NaN count: one-hot vars report only their best-ranked dummy level.",
            "required_cols": ["rank_full", "rank_restricted", "rank_delta"],
            "confirmed": True,
        },
        {
            "section": "S Table", "item": "S6 Table",
            "title": "PPV/NPV across assumed prevalence",
            "path": T / "ppv_npv_vs_prevalence.csv",
            "generator": "compute_nns_and_ppv_prevalence.py",
            "reviewer": "",
            "note": "0.5x/1x/2x prevalence scenarios, holding sensitivity/specificity fixed.",
            "required_cols": ["PPV", "NPV", "prevalence_multiplier"],
            "confirmed": True,
        },
        {
            "section": "S Table", "item": "S7 Table",
            "title": "Decision-curve analysis summary",
            "path": T / "decision_curve_summary.csv",
            "generator": "decision_curve_analysis.py",
            "reviewer": "",
            "note": "Threshold-probability range of positive net benefit, per outcome.",
            "required_cols": None, "confirmed": True,
        },
        {
            "section": "S Table", "item": "S8 Table",
            "title": "Top 15 SHAP predictors by outcome",
            "path": T / "global_shap_top15_all_outcomes.csv",
            "generator": "generate_global_shap.py",  # UNCONFIRMED -- never seen this script
            "reviewer": "",
            "note": "Global feature importance, stratified 20,000-obs sample per outcome.",
            "required_cols": None, "confirmed": False,
        },
        {
            "section": "S Table", "item": "S9 Table",
            "title": "Full SHAP ranking, all predictors",
            "path": T / "global_shap_full_ranking_all_outcomes.csv",
            "generator": "generate_global_shap.py",  # UNCONFIRMED
            "reviewer": "",
            "note": "Exhaustive version of S8 Table.",
            "required_cols": None, "confirmed": False,
        },
        {
            "section": "S Table", "item": "S10 Table",
            "title": "SHAP rank for structurally-missing predictors",
            "path": T / "global_shap_flagged_cluster_summary.csv",
            "generator": "generate_global_shap.py",  # UNCONFIRMED
            "reviewer": "",
            "note": "Justifies retaining the 10 high-missingness predictors (they're highly ranked).",
            "required_cols": None, "confirmed": False,
        },
        {
            "section": "S Table", "item": "S11 Table",
            "title": "Subgroup performance (sex/age/race/income)",
            "path": T / "fairness" / "subgroup_fairness_full.csv",
            "generator": "subgroup_fairness_analysis.py",  # UNCONFIRMED -- never seen this script
            "reviewer": "",
            "note": "AUROC/Brier with 95% CIs by subgroup, all 6 outcomes. ALSO now the sole representation "
                    "of subgroup performance (S6-S9 Figs were dropped -- never implemented).",
            "required_cols": None, "confirmed": False,
        },
        {
            "section": "S Table", "item": "S12 Table",
            "title": "Largest subgroup AUROC gaps",
            "path": T / "fairness" / "subgroup_auroc_gaps_summary.csv",
            "generator": "subgroup_fairness_analysis.py",  # UNCONFIRMED
            "reviewer": "",
            "note": "Companion summary to S11 Table -- biggest disparities at a glance.",
            "required_cols": None, "confirmed": False,
        },
        {
            "section": "S Table", "item": "S13 Table",
            "title": "Full loaded sample vs. outcome-specific analytic samples",
            "path": T / "phase1_3_full_vs_analytic_cohort.csv",
            "generator": "revision_analyses.py --phase 1 (phase_1_3_full_vs_analytic_cohort)",
            "reviewer": "",
            "note": "Confirms outcome-specific missing-data exclusions didn't introduce demographic bias.",
            "required_cols": ["sample", "variable", "category", "n", "pct"],
            "confirmed": True,
        },
        {
            "section": "S Table", "item": "S14 Table",
            "title": "Development vs. evaluation partition comparison",
            "path": T / "tripod_item20c_dev_vs_eval_comparison.csv",
            "generator": "verify_train_test_equivalence.py",  # UNCONFIRMED
            "reviewer": "TRIPOD+AI Item 20c",
            "note": "⚠️ Possibly the SAME underlying data as Table 2 (tripod_item21_partition_counts.csv) "
                    "viewed differently -- confirm these are genuinely distinct outputs, not duplicates.",
            "required_cols": None, "confirmed": False,
        },
        {
            "section": "S Table", "item": "S15 Table",
            "title": "Comparison with prior BRFSS-based ML studies",
            "path": None,  # lives inline in manuscript / response letter, not yet a standalone CSV
            "generator": "manual (literature review)",
            "reviewer": "",
            "note": "⚠️ Citations unresolved for 5 of 6 comparator studies; [4]/[4b] CKD relationship unconfirmed.",
            "required_cols": None, "confirmed": False,
        },
        # ---------------- SUPPLEMENTARY FIGURES ----------------
        {
            "section": "S Fig", "item": "S1 Fig",
            "title": "Web-based input interface screenshot",
            "path": None,  # static screenshot image, not a generated CSV/figure
            "generator": "manual screenshot",
            "reviewer": "Original submission figure",
            "note": "Caption updated to remove uncertainty-measures language.",
            "required_cols": None, "confirmed": True,
        },
        {
            "section": "S Fig", "item": "S2 Fig",
            "title": "PPV vs. prevalence curves",
            "path": F / "fig_ppv_vs_prevalence.pdf",
            "generator": "compute_nns_and_ppv_prevalence.py",
            "reviewer": "",
            "note": "3x2 grid, PDF+PNG parity patched.",
            "required_cols": None, "confirmed": True,
        },
        {
            "section": "S Fig", "item": "S3 Fig",
            "title": "Decision curves",
            "path": F / "fig_decision_curves.pdf",
            "generator": "decision_curve_analysis.py",
            "reviewer": "",
            "note": "PDF+PNG parity patched.",
            "required_cols": None, "confirmed": True,
        },
        {
            "section": "S Fig", "item": "S4 Fig",
            "title": "Global SHAP bar chart, top 15, all outcomes",
            "path": F / "fig_global_shap_bar_all_outcomes.pdf",
            "generator": "generate_global_shap.py -> make_manuscript_outputs.py",
            "reviewer": "",
            "note": "Visual companion to S8 Table.",
            "required_cols": None, "confirmed": True,
        },
        {
            "section": "S Fig", "item": "S5 Fig",
            "title": "Global SHAP violin/beeswarm plots",
            "path": F / "fig_global_shap_violin_combined.pdf",
            "generator": "plot_shap_violin.py -> make_manuscript_outputs.py",
            "reviewer": "",
            "note": "Richer visual companion to S9 Table. PDF+PNG parity patched. Last item in S-Fig sequence "
                    "(S6-S9 Figs dropped -- never implemented, superseded by S11/S12 Tables).",
            "required_cols": None, "confirmed": True,
        },
        # ---------------- SUPPLEMENTARY DATASETS ----------------
        {
            "section": "S Dataset", "item": "S1 Dataset",
            "title": "Participant flow counts by year",
            "path": T / "S1_participant_flow_by_year.csv",
            "generator": "generate_participant_flow.py",
            "reviewer": "TRIPOD+AI Item 20a (supports Fig 4)",
            "note": "Recovers pre-exclusion counts that data_preparation.py's pipeline silently discards.",
            "required_cols": ["year", "n_total_loaded", "n_excluded_all_outcomes_missing", "n_analytic_cohort"],
            "confirmed": True,
        },
        {
            "section": "S Dataset", "item": "S2 Dataset",
            "title": "Per-outcome participant flow counts by year",
            "path": T / "S2_participant_flow_by_outcome.csv",
            "generator": "generate_participant_flow.py",
            "reviewer": "TRIPOD+AI Item 20a (supports Fig 4)",
            "note": "Long-format companion to S1 Dataset.",
            "required_cols": ["year", "outcome", "n_included", "n_excluded_missing", "prevalence_pct"],
            "confirmed": True,
        },
        {
            "section": "S Dataset", "item": "S3 Dataset",
            "title": "Missing data counts, structural vs. person-level",
            "path": T / "S3_missing_data_predictors.csv",
            "generator": "generate_s3_missingness_table.py",  # UNCONFIRMED -- never seen this script
            "reviewer": "",
            "note": "Distinguishes 'not administered that year' from genuine person-level nonresponse.",
            "required_cols": None, "confirmed": False,
        },
    ]
    return _apply_current_labels(entries, T, F)


# ============================================================
# STATUS CHECKER
# ============================================================
def _human_size(num_bytes: int) -> str:
    for unit in ["B", "KB", "MB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.0f}{unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f}GB"


def check_item_status(entry: dict, reference_mtime: Optional[float]) -> Tuple[str, Optional[int], str]:
    """
    Returns (status, n_rows_or_None, detail_string).
    status in {"CODE", "OK", "STALE", "EMPTY", "MISSING", "NO-COLS"}
    """
    path = entry["path"]
    if path is None:
        return "CODE", None, "(generated directly by this script, or lives outside tables/figures/ e.g. TRIPOD.docx)"

    if not path.exists():
        return "MISSING", None, f"run {entry['generator']}"

    size_bytes = path.stat().st_size
    if size_bytes == 0:
        return "EMPTY", 0, _human_size(size_bytes)

    n_rows = None
    missing_cols = []
    if path.suffix.lower() == ".csv":
        try:
            df_check = pd.read_csv(path)
            n_rows = len(df_check)
            if n_rows == 0:
                return "EMPTY", 0, _human_size(size_bytes)
            if entry.get("required_cols"):
                missing_cols = [c for c in entry["required_cols"] if c not in df_check.columns]
        except Exception as e:
            return "EMPTY", None, f"unreadable: {e}"

    if missing_cols:
        return "NO-COLS", n_rows, f"missing columns: {missing_cols}"

    if reference_mtime is not None and path.stat().st_mtime < reference_mtime:
        age_days = (reference_mtime - path.stat().st_mtime) / 86400
        return "STALE", n_rows, f"{age_days:.1f} days older than disease_models.joblib"

    detail = f"{_human_size(size_bytes)}"
    return "OK", n_rows, detail


def print_manuscript_status_report(reference_mtime: Optional[float]) -> None:
    print("\n" + "=" * 100)
    print("COMPLETE MANUSCRIPT STATUS REPORT")
    print("=" * 100)
    print("Legend: [OK] present & valid  [STALE] older than model artifact  [EMPTY] 0 rows  "
          "[MISSING] not found  [CODE] computed by this script  [NO-COLS] missing expected columns")
    print("⚠️  = inferred filename, not confirmed against an actual script -- verify independently\n")

    registry = _registry()
    sections_order = ["Main Table", "Main Fig", "S Table", "S Fig", "S Dataset"]
    section_titles = {
        "Main Table": "MAIN TEXT TABLES", "Main Fig": "MAIN TEXT FIGURES",
        "S Table": "SUPPLEMENTARY TABLES", "S Fig": "SUPPLEMENTARY FIGURES",
        "S Dataset": "SUPPLEMENTARY DATASETS",
    }

    counts = {"CODE": 0, "OK": 0, "STALE": 0, "EMPTY": 0, "MISSING": 0, "NO-COLS": 0}

    for section in sections_order:
        items = [e for e in registry if e["section"] == section]
        if not items:
            continue
        print(f"\n--- {section_titles[section]} ---")
        for entry in items:
            status, n_rows, detail = check_item_status(entry, reference_mtime)
            counts[status] = counts.get(status, 0) + 1
            unconfirmed_flag = "" if entry["confirmed"] else "  ⚠️ UNCONFIRMED FILENAME"
            rows_str = f"{n_rows:,} rows" if n_rows is not None else ""
            print(f"  [{status:7s}] {entry['item']:10s} {entry['title']}")
            print(f"            {rows_str:14s} {detail}{unconfirmed_flag}")
            print(f"            Generator: {entry['generator']}")
            print(f"            Satisfies: {entry['reviewer']}")
            print(f"            Shows:     {entry['note']}")

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    total = sum(counts.values())
    for status in ["OK", "CODE", "STALE", "NO-COLS", "EMPTY", "MISSING"]:
        if counts.get(status, 0) > 0:
            print(f"  {status:9s}: {counts[status]:3d} / {total}")

    n_unconfirmed = sum(1 for e in _registry() if not e["confirmed"])
    print(f"\n  {n_unconfirmed} item(s) use an UNCONFIRMED filename (never-seen generating script) -- "
          f"verify these against your actual tables/ directory.")

    missing_items = [e["item"] for e in _registry()
                      if check_item_status(e, reference_mtime)[0] == "MISSING"]
    if missing_items:
        print(f"\n  🔴 ACTION NEEDED -- these items are MISSING and must be generated before submission:")
        for item in missing_items:
            print(f"     - {item}")
    else:
        print(f"\n  🟢 No MISSING items (among entries with a checkable file path).")

    print("=" * 100)


# ============================================================
# Main
# ============================================================
def main() -> None:
    print("== Survey-ML case-finding: manuscript outputs (final orchestrator) ==")
    print("Repo root:", REPO_ROOT)
    print("Artifacts:", ARTIFACT_DIR)
    print("Tables:   ", TABLES_DIR)
    print("Figures:  ", FIGURES_DIR)

    disease_models, optimal_thresholds, predictor_cols = load_artifacts()
    artifact_mtime = (ARTIFACT_DIR / "disease_models.joblib").stat().st_mtime

    print("\n[CORE 1] Table 1/3: model performance, recomputed with bootstrap CIs")
    make_table3_model_performance(
        disease_models=disease_models, thresholds=optimal_thresholds,
        save_path=TABLES_DIR / "table3_model_performance_updated.csv", n_bins=10,
    )

    print("\n[CORE 2] Figures: ROC + calibration")
    plot_roc_curves(disease_models, FIGURES_DIR / "fig_roc_curves.pdf")
    plot_calibration_curves(disease_models, FIGURES_DIR / "fig_calibration_curves.pdf", n_bins=10)

    print("\n[CORE 3] Example users: input table + prediction/SHAP tables + figures")
    write_example_outputs(disease_models, predictor_cols, TABLES_DIR, FIGURES_DIR)

    print("\n[CORE 4] Baseline comparison figures (reads existing CSV, if present)")
    make_baseline_comparison_outputs(TABLES_DIR, FIGURES_DIR)

    print("\n[CORE 5] Global SHAP figures (reads existing CSVs, if present)")
    make_global_shap_beeswarm_figures(TABLES_DIR, FIGURES_DIR)

    print("\n[CORE 6] Combined Table 1 (demographics + disease prevalence)")
    if CLEAN_DATA_CSV.exists():
        df = pd.read_csv(CLEAN_DATA_CSV)
        make_combined_table1(df, TABLES_DIR / "table1_combined_with_prevalence.csv")
    else:
        print(f"  [SKIP] Cleaned dataset not found at {CLEAN_DATA_CSV}.")

    # ---- Final, authoritative status report for EVERYTHING ----
    print_manuscript_status_report(reference_mtime=artifact_mtime)


if __name__ == "__main__":
    main()

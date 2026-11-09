#!/usr/bin/env python
# coding: utf-8
"""
scripts/subgroup_fairness_analysis.py

Computes AUROC and Brier score, with bootstrap confidence intervals,
stratified by sex, age group, race/ethnicity, and income tertile, for
all six outcome models. Addresses a review comment (algorithmic
fairness / subgroup performance).

Uses the FINAL trained models' stored test-set predictions and raw
predictor values (info["X_test"], info["y_test"], info["y_proba"]) --
no retraining needed. Subgroup membership is derived directly from the
raw BRFSS predictor codes in X_test (e.g. SEX, _AGEG5YR, _MRACE1,
_HISPANC, _INCOMG), not from the trained model's one-hot encoded
features.

SUBGROUP DEFINITIONS (documented here so they can be stated plainly
and identically in the Methods text):
  Sex: Male / Female (BRFSS SEX: 1/2)
  Age group (collapsed from 13 BRFSS 5-year bands to 3 clinically
    meaningful bands): 18-44 / 45-64 / 65+
  Race/ethnicity: White non-Hispanic / Black non-Hispanic / Hispanic
    (any race) / Other/Multiracial non-Hispanic. Hispanic ethnicity
    (_HISPANC=1) takes precedence over race category, consistent with
    standard demographic reporting conventions, to avoid double-
    counting respondents across race and ethnicity dimensions.
    Respondents missing race/ethnicity data (predominantly 2011-2012,
    per the structural missingness analysis) are reported as a
    separate "Missing race/ethnicity" category rather than silently
    dropped, since dropping would materially shrink the 2011-2012
    contribution to this specific analysis.
  Income tertile (collapsed from 6 BRFSS ordinal income categories
    into 3 groups, since true continuous income is not available):
    Lower (<$25k) / Middle ($25-50k) / Higher ($50k+)

RELIABILITY FLAGGING: any subgroup cell with fewer than 100
observations or fewer than 10 positive-class events is flagged
(reliable=False) rather than silently reported, since AUROC and Brier
score estimates become unstable below these thresholds. Flagged cells
are still computed and reported (not dropped) for transparency, but
should be interpreted cautiously and are noted as such in the
Results/Supplementary text.

BOOTSTRAP: 200 resamples per subgroup cell for 95% percentile
confidence intervals. This is a moderate number chosen for
computational tractability across the full grid (6 outcomes x 4
stratification variables x multiple subgroups each); increase
N_BOOTSTRAP below if compute budget allows for the final submission
version.

Run (from repo root):
  python scripts/subgroup_fairness_analysis.py

Environment variables (optional):
  ARTIFACT_DIR    (default: <repo>/app/artifacts)
  TABLES_DIR      (default: <repo>/tables)
  FIGURES_DIR     (default: <repo>/figures)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import roc_auc_score, brier_score_loss

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)


# ============================================================
# Path configuration
# ============================================================
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1] if THIS_FILE.parent.name in {"src", "scripts"} else THIS_FILE.parent

ARTIFACT_DIR = Path(os.environ.get("ARTIFACT_DIR", REPO_ROOT / "app" / "artifacts"))
TABLES_DIR   = Path(os.environ.get("TABLES_DIR",   REPO_ROOT / "tables"))
FIGURES_DIR  = Path(os.environ.get("FIGURES_DIR",  REPO_ROOT / "figures"))
FAIRNESS_TABLES_DIR = TABLES_DIR / "fairness"
FAIRNESS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_BOOTSTRAP = 200
MIN_N = 100
MIN_POSITIVES = 10

# Standard grid layout for six-outcome faceted figures, matching
# make_manuscript_outputs.py and decision_curve_analysis.py for
# visual consistency across the paper.
FIG_GRID_ROWS = 3
FIG_GRID_COLS = 2

DISEASE_NAME_MAP = {
    "heart_attack": "Heart attack",
    "coronary_hd":  "Coronary heart disease",
    "stroke":       "Stroke",
    "kidney":       "Chronic kidney disease",
    "depression":   "Depression",
    "diabetes":     "Diabetes",
}


# ============================================================
# Subgroup derivation from raw BRFSS predictor codes
# ============================================================
def derive_sex_group(X: pd.DataFrame) -> pd.Series:
    return X["SEX"].map({1: "Male", 2: "Female"}).fillna("Missing")


def derive_age_group(X: pd.DataFrame) -> pd.Series:
    # BRFSS _AGEG5YR: 1=18-24 ... 13=80+ (14 = don't know/refused -> missing).
    # Collapse into 3 bands; missing age is kept as its own "Missing" subgroup.
    def bucket(v):
        if pd.isna(v):
            return "Missing"
        v = int(v)
        if v <= 5:      # 18-24 through 40-44
            return "18-44"
        elif v <= 9:    # 45-49 through 60-64
            return "45-64"
        else:           # 65-69 through 80+
            return "65+"
    return X["_AGEG5YR"].apply(bucket)


def derive_race_ethnicity_group(X: pd.DataFrame) -> pd.Series:
    # Hispanic ethnicity (_HISPANC=1) takes precedence over race
    # category, consistent with standard demographic reporting.
    def bucket(row):
        hispanc = row["_HISPANC"]
        mrace = row["_MRACE1"]
        if pd.isna(hispanc) and pd.isna(mrace):
            return "Missing race/ethnicity"
        if hispanc == 1:
            return "Hispanic (any race)"
        if mrace == 1:
            return "White non-Hispanic"
        if mrace == 2:
            return "Black non-Hispanic"
        if mrace in (3, 4, 5, 6, 7, 8):
            return "Other/Multiracial non-Hispanic"
        return "Missing race/ethnicity"
    return X.apply(bucket, axis=1)


def derive_income_group(X: pd.DataFrame) -> pd.Series:
    # BRFSS _INCOMG: 1=<$15k, 2=$15-25k, 3=$25-35k, 4=$35-50k, 5=$50k or more
    # (9 = don't know/refused -> missing). Collapse into 3 approximate tertiles.
    # Missing income is kept as its own "Missing" subgroup.
    def bucket(v):
        if pd.isna(v):
            return "Missing"
        v = int(v)
        if v <= 2:      # <$25k
            return "Lower (<$25k)"
        elif v <= 4:    # $25-50k
            return "Middle ($25-50k)"
        else:           # $50k+
            return "Higher ($50k+)"
    return X["_INCOMG"].apply(bucket)


STRATIFICATION_VARS = {
    "sex": derive_sex_group,
    "age_group": derive_age_group,
    "race_ethnicity": derive_race_ethnicity_group,
    "income_tertile": derive_income_group,
}


# ============================================================
# Metric computation with bootstrap CIs
# ============================================================
def bootstrap_auroc_brier(y_true: np.ndarray, y_proba: np.ndarray, n_boot: int, rng: np.random.RandomState):
    n = len(y_true)
    auroc_samples = []
    brier_samples = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt, yp = y_true[idx], y_proba[idx]
        if len(np.unique(yt)) < 2:
            continue  # AUROC undefined for a single-class resample
        auroc_samples.append(roc_auc_score(yt, yp))
        brier_samples.append(brier_score_loss(yt, yp))
    if not auroc_samples:
        return (np.nan, np.nan, np.nan), (np.nan, np.nan, np.nan)
    auroc_ci = (np.percentile(auroc_samples, 2.5), np.percentile(auroc_samples, 97.5))
    brier_ci = (np.percentile(brier_samples, 2.5), np.percentile(brier_samples, 97.5))
    return auroc_ci, brier_ci


def compute_subgroup_metrics(
    disease: str,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    subgroup_labels: pd.Series,
    strat_var_name: str,
    rng: np.random.RandomState,
) -> pd.DataFrame:
    rows = []
    for level in sorted(subgroup_labels.dropna().unique()):
        mask = (subgroup_labels == level).values
        n = int(mask.sum())
        yt = y_true[mask]
        yp = y_proba[mask]
        n_pos = int(yt.sum())

        reliable = (n >= MIN_N) and (n_pos >= MIN_POSITIVES) and (n - n_pos) >= MIN_POSITIVES

        if n_pos > 0 and (n - n_pos) > 0:
            auroc = roc_auc_score(yt, yp)
            brier = brier_score_loss(yt, yp)
            (auroc_lo, auroc_hi), (brier_lo, brier_hi) = bootstrap_auroc_brier(yt, yp, N_BOOTSTRAP, rng)
        else:
            auroc = brier = auroc_lo = auroc_hi = brier_lo = brier_hi = np.nan

        rows.append({
            "disease": disease,
            "disease_pretty": DISEASE_NAME_MAP.get(disease, disease),
            "stratification_variable": strat_var_name,
            "subgroup": level,
            "n": n,
            "n_positive": n_pos,
            "prevalence_pct": round(100 * n_pos / n, 2) if n > 0 else np.nan,
            "AUROC": round(auroc, 4) if not np.isnan(auroc) else np.nan,
            "AUROC_CI_low": round(auroc_lo, 4) if not np.isnan(auroc_lo) else np.nan,
            "AUROC_CI_high": round(auroc_hi, 4) if not np.isnan(auroc_hi) else np.nan,
            "Brier": round(brier, 4) if not np.isnan(brier) else np.nan,
            "Brier_CI_low": round(brier_lo, 4) if not np.isnan(brier_lo) else np.nan,
            "Brier_CI_high": round(brier_hi, 4) if not np.isnan(brier_hi) else np.nan,
            "reliable": reliable,
        })
    return pd.DataFrame(rows)


def main():
    disease_models_path = ARTIFACT_DIR / "disease_models.joblib"
    disease_models = joblib.load(disease_models_path)
    rng = np.random.RandomState(RANDOM_STATE)

    all_results = []
    overall_rows = []

    for disease, info in disease_models.items():
        print(f"\n=== {DISEASE_NAME_MAP.get(disease, disease)} ===")
        X_test = info["X_test"].reset_index(drop=True)
        y_test_raw = info["y_test"]
        if hasattr(y_test_raw, "reset_index"):
            y_test_raw = y_test_raw.reset_index(drop=True)
        y_test = np.asarray(y_test_raw, dtype=int)
        y_proba = np.asarray(info["y_proba"], dtype=float)

        overall_auroc = roc_auc_score(y_test, y_proba)
        overall_brier = brier_score_loss(y_test, y_proba)
        overall_rows.append({
            "disease": disease, "disease_pretty": DISEASE_NAME_MAP.get(disease, disease),
            "overall_AUROC": round(overall_auroc, 4), "overall_Brier": round(overall_brier, 4),
        })
        print(f"  Overall: AUROC={overall_auroc:.4f}  Brier={overall_brier:.4f}")

        for strat_name, derive_fn in STRATIFICATION_VARS.items():
            subgroup_labels = derive_fn(X_test)
            result_df = compute_subgroup_metrics(disease, y_test, y_proba, subgroup_labels, strat_name, rng)
            all_results.append(result_df)

            print(f"  --- {strat_name} ---")
            print(result_df[["subgroup", "n", "n_positive", "AUROC", "AUROC_CI_low",
                              "AUROC_CI_high", "Brier", "reliable"]].to_string(index=False))

    combined = pd.concat(all_results, ignore_index=True)
    overall_df = pd.DataFrame(overall_rows)

    out_path = FAIRNESS_TABLES_DIR / "subgroup_fairness_full.csv"
    combined.to_csv(out_path, index=False)
    print(f"\nSaved full subgroup results -> {out_path}")

    overall_out = FAIRNESS_TABLES_DIR / "overall_performance_reference.csv"
    overall_df.to_csv(overall_out, index=False)
    print(f"Saved overall (unstratified) reference -> {overall_out}")

    # ------------------------------------------------------------
    # Flag largest AUROC gaps per outcome per stratification variable,
    # among RELIABLE subgroups only, for quick identification of
    # potential fairness concerns worth discussing.
    # ------------------------------------------------------------
    print("\n=== Largest AUROC gap (max - min) per outcome x stratification variable (reliable subgroups only) ===")
    gap_rows = []
    # "Missing ..." subgroups (unknown age / income / race) are excluded from the best-vs-worst gap:
    # they are residual categories, not demographic groups, and otherwise show up as the "best" group.
    _gap_pool = combined[combined["reliable"] & ~combined["subgroup"].astype(str).str.startswith("Missing")]
    for (disease, strat_var), grp in _gap_pool.groupby(["disease", "stratification_variable"]):
        if len(grp) < 2 or grp["AUROC"].isna().all():
            continue
        max_row = grp.loc[grp["AUROC"].idxmax()]
        min_row = grp.loc[grp["AUROC"].idxmin()]
        gap = max_row["AUROC"] - min_row["AUROC"]
        gap_rows.append({
            "disease": disease, "disease_pretty": DISEASE_NAME_MAP.get(disease, disease),
            "stratification_variable": strat_var,
            "best_subgroup": max_row["subgroup"], "best_AUROC": max_row["AUROC"],
            "worst_subgroup": min_row["subgroup"], "worst_AUROC": min_row["AUROC"],
            "AUROC_gap": round(gap, 4),
        })
    gap_df = pd.DataFrame(gap_rows).sort_values("AUROC_gap", ascending=False)
    print(gap_df.to_string(index=False))

    gap_out = FAIRNESS_TABLES_DIR / "subgroup_auroc_gaps_summary.csv"
    gap_df.to_csv(gap_out, index=False)
    print(f"\nSaved -> {gap_out}")

    # ------------------------------------------------------------
    # Figure: small-multiple plots, one per stratification variable,
    # AUROC by subgroup with error bars, faceted by disease.
    # ------------------------------------------------------------
    for strat_name in STRATIFICATION_VARS.keys():
        sub = combined[combined["stratification_variable"] == strat_name]
        diseases = list(DISEASE_NAME_MAP.keys())
        fig, axes = plt.subplots(FIG_GRID_ROWS, FIG_GRID_COLS, figsize=(11, 4.5 * FIG_GRID_ROWS))
        axes = axes.flatten()

        for ax, disease in zip(axes, diseases):
            d_sub = sub[sub["disease"] == disease].copy()
            d_sub = d_sub.sort_values("subgroup")
            x = np.arange(len(d_sub))
            colors = ["steelblue" if r else "lightgray" for r in d_sub["reliable"]]

            yerr_low = d_sub["AUROC"] - d_sub["AUROC_CI_low"]
            yerr_high = d_sub["AUROC_CI_high"] - d_sub["AUROC"]
            ax.bar(x, d_sub["AUROC"], color=colors)
            ax.errorbar(x, d_sub["AUROC"], yerr=[yerr_low, yerr_high],
                        fmt="none", ecolor="black", capsize=3)
            ax.axhline(overall_df.loc[overall_df["disease"] == disease, "overall_AUROC"].values[0],
                       linestyle="--", color="red", alpha=0.6, label="Overall AUROC")
            ax.set_xticks(x)
            ax.set_xticklabels(d_sub["subgroup"], rotation=45, ha="right", fontsize=8)
            ax.set_ylim(0.5, 1.0)
            ax.set_ylabel("AUROC")
            ax.set_title(DISEASE_NAME_MAP.get(disease, disease))
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"Subgroup AUROC by {strat_name.replace('_', ' ')} "
                     "(gray bars = unreliable, n<100 or <10 events per class)",
                     fontsize=12)
        plt.tight_layout()
        fig_path = FIGURES_DIR / f"fig_fairness_{strat_name}.pdf"
        fig.savefig(fig_path, dpi=300)
        plt.close(fig)
        print(f"Saved figure -> {fig_path}")

    print(
        "\nDone. Review subgroup_auroc_gaps_summary.csv for any large "
        "gaps worth discussing explicitly, and subgroup_fairness_full.csv "
        "for the complete stratified results table (S[X] Table)."
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# coding: utf-8
"""
scripts/revision_analyses.py

Analyses for the PDIG-D-26-01022 revision (PLOS Digital Health).
Organized by phase, matching the "Analysis Execution Plan" agreed with
the response-to-reviewers letters. Run phases independently and in order:

    python scripts/revision_analyses.py --phase 0
    python scripts/revision_analyses.py --phase 1

[PATCH] Output directories changed: this script previously wrote all
its tables/figures into a SEPARATE tree (revision/tables/,
revision/figures/), isolated from the main app/artifacts/, tables/,
figures/ directories used by every other script in this pipeline
(make_manuscript_outputs.py, decision_curve_analysis.py,
restricted_years_sensitivity.py, etc.). This caused a real bug: S13
Table (phase1_3_full_vs_analytic_cohort.csv) was being generated
correctly, but make_manuscript_outputs.py's manuscript-status registry
looks for it in the shared tables/ directory and reported it as
[MISSING] even though it existed one directory over, in revision/tables/.

TABLES_DIR and FIGURES_DIR now point at the SAME shared directories
every other script in this pipeline uses, by default. The old
isolated revision/ tree is now used ONLY for the Phase 0.1 frozen
model-artifact snapshots (v1_original/<timestamp>/), which genuinely
should stay separate from the live tables/figures outputs -- those are
backup copies of the model state, not manuscript-facing outputs, and
mixing them into tables/ would clutter the shared directory with
non-manuscript files.

If you relied on the old revision/tables/, revision/figures/ layout
for anything else, override via REVISION_TABLES_DIR /
REVISION_FIGURES_DIR env vars (see below) to restore the previous
behavior, or just copy files from the old location if you have
existing outputs there from before this patch.

Each phase writes its outputs to the shared tables/ and figures/
directories (same ones used by every other script), EXCEPT for the
Phase 0.1 frozen snapshot, which stays under revision/v1_original/ as
before since it's a timestamped backup, not a manuscript output.

Portability: same repo-relative path conventions as data_preparation.py
and model_ai.py. Override via environment variables if needed:
    ARTIFACT_DIR, DATA_DERIVED_DIR, BRFSS_RAW_DIR, REVISION_DIR,
    TABLES_DIR, FIGURES_DIR
"""

from __future__ import annotations

import os
import argparse
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import joblib

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)


# ============================================================
# Path configuration (same conventions as existing scripts)
# ============================================================
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1] if THIS_FILE.parent.name in {"src", "scripts"} else THIS_FILE.parent

ARTIFACT_DIR      = Path(os.environ.get("ARTIFACT_DIR",      REPO_ROOT / "app" / "artifacts"))
DATA_DERIVED_DIR  = Path(os.environ.get("DATA_DERIVED_DIR",  REPO_ROOT / "data" / "derived"))
BRFSS_RAW_DIR     = Path(os.environ.get("BRFSS_RAW_DIR",     REPO_ROOT / "data" / "raw"))

# [PATCH] REVISION_DIR is now used ONLY for the Phase 0.1 frozen
# model-artifact snapshots (timestamped backups of disease_models.joblib
# etc.), which genuinely belong in their own isolated tree separate
# from manuscript-facing outputs -- these are safety-net copies of the
# model state at a point in time, not something the manuscript cites.
# [PATCH 2] Guard against doubling "revision" in the path when the
# repo root itself is already named "revision" (as in this project's
# actual layout: .../presubmission/revision/scripts/revision_analyses.py
# -> REPO_ROOT = .../presubmission/revision/). Appending another
# "revision" segment unconditionally produced revision/revision/v1_original,
# which still worked but was confusing and error-prone to navigate.
if REPO_ROOT.name == "revision":
    REVISION_DIR = Path(os.environ.get("REVISION_DIR", REPO_ROOT))
else:
    REVISION_DIR = Path(os.environ.get("REVISION_DIR", REPO_ROOT / "revision"))

FROZEN_DIR = REVISION_DIR / "v1_original"

# [PATCH] TABLES_DIR / FIGURES_DIR now default to the SAME shared
# directories used by make_manuscript_outputs.py, decision_curve_analysis.py,
# restricted_years_sensitivity.py, compute_nns_and_ppv_prevalence.py, and
# generate_participant_flow.py -- i.e. <repo>/tables and <repo>/figures,
# NOT <repo>/revision/tables and <repo>/revision/figures as before.
# This is what make_manuscript_outputs.py's manuscript-status registry
# actually checks, so outputs from this script are now visible to that
# report without any manual file-copying step.
#
# If you need the OLD isolated-directory behavior for some reason,
# set REVISION_TABLES_DIR / REVISION_FIGURES_DIR explicitly; otherwise
# TABLES_DIR / FIGURES_DIR env vars (matching every other script in
# this pipeline) take precedence, falling back to the shared defaults.
TABLES_DIR  = Path(os.environ.get(
    "TABLES_DIR", os.environ.get("REVISION_TABLES_DIR", REPO_ROOT / "tables")
))
FIGURES_DIR = Path(os.environ.get(
    "FIGURES_DIR", os.environ.get("REVISION_FIGURES_DIR", REPO_ROOT / "figures")
))

for d in [REVISION_DIR, FROZEN_DIR, TABLES_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

COMBINED_CSV = Path(os.environ.get(
    "BRFSS_CLEAN_CSV",
    DATA_DERIVED_DIR / "BRFSS_2011_2015_clean_model.csv"
))


# ============================================================
# Shared constants (mirrors data_preparation.py / model_ai.py)
# ============================================================
PREDICTOR_COLS = [
    "_STATE", "SEX", "_AGEG5YR", "_EDUCAG", "_INCOMG", "_MRACE1", "_HISPANC",
    "SMOKE100", "SMOKDAY2", "ALCDAY5", "DRNKANY5", "EXERANY2", "FRUIT1", "VEGETAB1",
    "HLTHPLN1", "PERSDOC2", "MEDCOST", "CHECKUP1",
    "BPHIGH4", "BPMEDS", "TOLDHI2", "CHOLCHK", "ASTHMA3", "HAVARTH3",
    "GENHLTH", "PHYSHLTH", "MENTHLTH", "POORHLTH",
    "DIFFWALK", "DECIDE",
    "WEIGHT2", "HEIGHT3", "_BMI5",
]

TARGET_COLS = ["heart_attack", "coronary_hd", "stroke", "kidney", "depression", "diabetes"]

DISEASE_NAME_MAP = {
    "heart_attack": "Heart attack",
    "coronary_hd":  "Coronary heart disease",
    "stroke":       "Stroke",
    "kidney":       "Chronic kidney disease",
    "depression":   "Depression",
    "diabetes":     "Diabetes",
}

# BRFSS variables that are nominal or ordinal *codes* even though they are
# stored as numeric dtypes in the CSV. Used only for the Phase 0.2
# diagnostic below -- this list does NOT alter any existing trained model.
TRUE_NOMINAL_OR_ORDINAL_CODES = [
    "_STATE", "SEX", "_MRACE1", "_HISPANC",
    "SMOKE100", "SMOKDAY2", "DRNKANY5", "EXERANY2",
    "HLTHPLN1", "PERSDOC2", "MEDCOST", "CHECKUP1",
    "BPHIGH4", "BPMEDS", "TOLDHI2", "CHOLCHK", "ASTHMA3", "HAVARTH3",
    "DIFFWALK", "DECIDE", "GENHLTH",
    "_AGEG5YR", "_EDUCAG", "_INCOMG",   # ordinal groupings, not truly continuous
    "FRUIT1", "VEGETAB1", "ALCDAY5",    # BRFSS mixed day/week/month encodings
]

DEMO_COLS_FOR_COMPARISON = ["SEX", "_AGEG5YR", "_EDUCAG", "_INCOMG"]


# ============================================================
# PHASE 0.1 — Freeze current models & extract flat predictions
# ============================================================
def phase_0_1_freeze_artifacts() -> Path:
    """
    Copy the current joblib artifacts into revision/v1_original/<timestamp>/
    before any retraining happens in later phases. Also extracts a flat
    (disease, y_true, y_proba) predictions table for reuse without needing
    to reload full model pipelines in later phases.

    NOTE: this is the one output of this script that INTENTIONALLY stays
    under the isolated revision/ tree (not TABLES_DIR) -- it's a
    timestamped backup snapshot of model state, not a manuscript-facing
    table, so keeping it separate from tables/ is correct, not a bug.
    """
    print("=== Phase 0.1: Freezing current artifacts ===")
    src_files = [
        ARTIFACT_DIR / "disease_models.joblib",
        ARTIFACT_DIR / "optimal_thresholds.joblib",
        ARTIFACT_DIR / "predictor_cols.joblib",
    ]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_dir = FROZEN_DIR / stamp
    dest_dir.mkdir(parents=True, exist_ok=True)

    for f in src_files:
        if f.exists():
            shutil.copy2(f, dest_dir / f.name)
            print(f"  Copied {f.name} -> {dest_dir}")
        else:
            print(f"  WARNING: expected artifact not found: {f}")

    disease_models_path = ARTIFACT_DIR / "disease_models.joblib"
    if not disease_models_path.exists():
        print("  Skipping flat-predictions export: disease_models.joblib not found.")
        return dest_dir

    disease_models = joblib.load(disease_models_path)
    rows = []
    for disease, info in disease_models.items():
        y_test  = np.asarray(info["y_test"],  dtype=int)
        y_proba = np.asarray(info["y_proba"], dtype=float)
        thr     = float(info.get("best_threshold", 0.5))
        for yt, yp in zip(y_test, y_proba):
            rows.append({
                "disease": disease,
                "y_true": yt,
                "y_proba": yp,
                "threshold": thr,
            })

    preds_df = pd.DataFrame(rows)
    out_path = dest_dir / "frozen_test_predictions.csv"
    preds_df.to_csv(out_path, index=False)
    print(f"  Saved flat predictions table ({len(preds_df):,} rows) -> {out_path}")
    print(f"  Frozen snapshot directory: {dest_dir}")
    return dest_dir


# ============================================================
# PHASE 0.2 — Diagnostic: confirm what the existing pipeline
#             ACTUALLY does with race/ethnicity and other
#             BRFSS-coded predictors, before writing any
#             response-letter claims about it.
# ============================================================
def phase_0_2_pipeline_diagnostic() -> dict:
    """
    Replicates model_ai.py's exact numeric/categorical split logic against
    the real cleaned dataset, and reports:
      (a) which predictors actually land in the numeric (median-imputed +
          standardized) branch vs. the categorical (mode-imputed +
          one-hot encoded) branch,
      (b) whether _MRACE1 / _HISPANC specifically fall into the numeric
          or categorical branch, and what that implies about how missing
          2011-2012 values were actually handled,
      (c) a full list of variables misclassified as numeric relative to
          what they actually represent (nominal/ordinal BRFSS codes).

    This is read-only: it does not retrain or alter any model.

    NOTE: this diagnostic checks model_ai.py's SOURCE-LEVEL column lists
    (PREDICTOR_COLS / dtype inspection of the cleaned CSV), which is a
    static approximation. For ground-truth confirmation of what the
    ACTUALLY FITTED pipeline inside disease_models.joblib does, use
    scripts/inspect_pipeline_encoding.py instead, which inspects the
    live ColumnTransformer object directly -- that is the authoritative
    check if this diagnostic's output ever conflicts with reality (as
    happened during this revision: this diagnostic appeared to show 0
    categorical columns, but inspect_pipeline_encoding.py confirmed the
    actual fitted model correctly uses 18 categorical / 15 numeric
    columns. If you see a mismatch between this function's output and
    inspect_pipeline_encoding.py, TRUST inspect_pipeline_encoding.py --
    it reads the real fitted object, not a static approximation of what
    the pipeline-building code is expected to do.)
    """
    print("=== Phase 0.2: Preprocessing pipeline diagnostic ===")
    print("    (NOTE: this is a static diagnostic against the cleaned CSV's dtypes, "
          "not the live fitted model. If this disagrees with "
          "inspect_pipeline_encoding.py's output, trust inspect_pipeline_encoding.py.)")
    if not COMBINED_CSV.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {COMBINED_CSV}")

    df = pd.read_csv(COMBINED_CSV)

    numeric_features = []
    categorical_features = []
    missing_cols = []

    for col in PREDICTOR_COLS:
        if col not in df.columns:
            missing_cols.append(col)
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_features.append(col)
        else:
            categorical_features.append(col)

    if missing_cols:
        print(f"  WARNING: predictors not found in cleaned dataset: {missing_cols}")

    print(f"\n  Columns classified as NUMERIC by dtype inspection of the cleaned CSV "
          f"(this is NOT necessarily what the fitted pipeline actually does -- "
          f"see inspect_pipeline_encoding.py for ground truth): {len(numeric_features)}")
    for c in numeric_features:
        print(f"    - {c}")

    print(f"\n  Columns classified as CATEGORICAL by dtype inspection: {len(categorical_features)}")
    for c in categorical_features:
        print(f"    - {c}")

    misclassified = [c for c in numeric_features if c in TRUE_NOMINAL_OR_ORDINAL_CODES]
    print(f"\n  *** Variables with NUMERIC dtype in the raw CSV that represent "
          f"nominal or ordinal BRFSS codes: {len(misclassified)} ***")
    print(f"      (This does NOT mean the fitted model treats them as continuous --")
    print(f"       model_ai.py's actual NOMINAL_CATEGORICAL_COLS list routes most of")
    print(f"       these through one-hot encoding regardless of raw CSV dtype. See")
    print(f"       inspect_pipeline_encoding.py to confirm what the fitted model does.)")
    for c in misclassified:
        print(f"    - {c}")

    # --- Race/ethnicity specific check (directly relevant to R1 #3, R3 #3c) ---
    print("\n  --- Race (_MRACE1) specific check ---")
    if "_MRACE1" in df.columns:
        dtype = df["_MRACE1"].dtype
        is_numeric = pd.api.types.is_numeric_dtype(df["_MRACE1"])
        print(f"    _MRACE1 raw CSV dtype: {dtype} | is_numeric_dtype: {is_numeric}")
        print(f"    (Raw dtype being numeric does NOT mean the fitted pipeline median-imputes")
        print(f"     it as continuous -- model_ai.py's NOMINAL_CATEGORICAL_COLS list explicitly")
        print(f"     includes _MRACE1, routing it to one-hot encoding + mode imputation")
        print(f"     regardless of its raw numeric dtype. CONFIRMED via")
        print(f"     inspect_pipeline_encoding.py: _MRACE1 -> CATEGORICAL (one-hot encoded).)")

        if "YEAR" in df.columns:
            missing_by_year = df.groupby("YEAR")["_MRACE1"].apply(lambda s: s.isna().mean() * 100)
            print("    % missing _MRACE1 by year:")
            print(missing_by_year.round(1).to_string())
    else:
        print("    _MRACE1 not found in cleaned dataset columns.")

    print("\n  --- Hispanic ethnicity (_HISPANC) specific check ---")
    if "_HISPANC" in df.columns:
        is_numeric = pd.api.types.is_numeric_dtype(df["_HISPANC"])
        print(f"    _HISPANC raw CSV dtype is_numeric: {is_numeric} "
              f"(CONFIRMED via inspect_pipeline_encoding.py: routed to CATEGORICAL branch "
              f"in the actual fitted pipeline, regardless of raw dtype.)")
    else:
        print("    _HISPANC not found in cleaned dataset columns.")

    # Save full report
    report_path = TABLES_DIR / "phase0_2_pipeline_diagnostic.txt"
    with open(report_path, "w") as f:
        f.write("STATIC diagnostic against cleaned CSV dtypes (NOT the live fitted model --\n")
        f.write("see inspect_pipeline_encoding.py for ground truth on the actual pipeline).\n\n")
        f.write("NUMERIC dtype in raw CSV:\n")
        f.write("\n".join(f"  - {c}" for c in numeric_features) + "\n\n")
        f.write("CATEGORICAL dtype in raw CSV:\n")
        f.write("\n".join(f"  - {c}" for c in categorical_features) + "\n\n")
        f.write("Nominal/ordinal codes with numeric raw dtype (NOT necessarily "
                "treated as continuous by the fitted model -- see inspect_pipeline_encoding.py):\n")
        f.write("\n".join(f"  - {c}" for c in misclassified) + "\n")
    print(f"\n  Full diagnostic report saved -> {report_path}")

    return {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "misclassified_as_numeric": misclassified,
    }


# ============================================================
# PHASE 1.1 — MCE bin-size detail (R1 Comment 5)
#             Run for ALL six outcomes, not just CHD, since the
#             same diagnostic is useful supporting context for
#             every outcome's Table 1 footnote.
# ============================================================
def phase_1_1_mce_bin_detail(n_bins: int = 10) -> pd.DataFrame:
    print("=== Phase 1.1: MCE bin-size diagnostic (all outcomes, RERUN against current/corrected model) ===")
    disease_models = joblib.load(ARTIFACT_DIR / "disease_models.joblib")

    all_rows = []
    for disease, info in disease_models.items():
        y_test  = np.asarray(info["y_test"],  dtype=int)
        y_proba = np.asarray(info["y_proba"], dtype=float)
        n = len(y_test)

        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_rows = []
        for j in range(n_bins):
            if j < n_bins - 1:
                mask = (y_proba >= bins[j]) & (y_proba < bins[j + 1])
            else:
                mask = (y_proba >= bins[j]) & (y_proba <= bins[j + 1])
            if not np.any(mask):
                continue
            p_bin = float(y_proba[mask].mean())
            o_bin = float(y_test[mask].mean())
            err = abs(o_bin - p_bin)
            bin_rows.append({
                "disease": disease,
                "disease_pretty": DISEASE_NAME_MAP.get(disease, disease),
                "bin_low": round(bins[j], 2),
                "bin_high": round(bins[j + 1], 2),
                "n_in_bin": int(mask.sum()),
                "pct_of_test_set": round(100 * mask.sum() / n, 4),
                "mean_predicted": round(p_bin, 4),
                "mean_observed": round(o_bin, 4),
                "abs_calibration_error": round(err, 4),
            })

        bin_df = pd.DataFrame(bin_rows)
        if bin_df.empty:
            print(f"  {disease}: no populated bins found (unexpected).")
            continue

        worst = bin_df.loc[bin_df["abs_calibration_error"].idxmax()]
        sorted_by_err = bin_df.sort_values("abs_calibration_error", ascending=False)
        second_worst = sorted_by_err.iloc[1] if len(sorted_by_err) > 1 else None
        n_sparse_bins = int((bin_df["n_in_bin"] < 10).sum())

        print(
            f"  {DISEASE_NAME_MAP.get(disease, disease):28s} "
            f"MCE={worst['abs_calibration_error']:.4f}  "
            f"bin=[{worst['bin_low']:.2f},{worst['bin_high']:.2f}]  "
            f"n={int(worst['n_in_bin']):>7,}  "
            f"({worst['pct_of_test_set']:.4f}% of N_test={n:,})  "
            f"| bins with n<10: {n_sparse_bins}"
        )
        if second_worst is not None:
            print(
                f"    2nd-worst: err={second_worst['abs_calibration_error']:.4f}  "
                f"bin=[{second_worst['bin_low']:.2f},{second_worst['bin_high']:.2f}]  "
                f"n={int(second_worst['n_in_bin']):,}"
            )
        all_rows.append(bin_df)

    full_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    out_path = TABLES_DIR / "phase1_1_mce_bin_detail_all_outcomes.csv"
    full_df.to_csv(out_path, index=False)
    print(f"\n  Full per-bin detail (all outcomes, all bins) saved -> {out_path}")
    return full_df


# ============================================================
# PHASE 1.3 — Full loaded sample vs. analytic cohort comparison
#             (R3 Comment 3a)
#
# [PATCH] Output now goes to the SHARED TABLES_DIR (<repo>/tables/),
# not the isolated revision/tables/ tree -- this is the file
# make_manuscript_outputs.py's status registry checks for as "S13
# Table". Previously this was silently written to a directory the
# registry never looked at, causing a false [MISSING] status.
# ============================================================
def _resolve_year_file(year: int, raw_dir: Path) -> Path:
    zipped = raw_dir / f"{year}.csv.zip"
    plain  = raw_dir / f"{year}.csv"
    if zipped.exists():
        return zipped
    if plain.exists():
        return plain
    raise FileNotFoundError(
        f"Could not find input file for year {year} in {raw_dir}. "
        f"Expected {zipped} or {plain}."
    )


def _summarize_demographics(df: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    n_total = len(df)
    rows.append({"sample": label, "variable": "N", "category": "(total)", "n": n_total, "pct": 100.0})
    for col in DEMO_COLS_FOR_COMPARISON:
        if col not in df.columns:
            continue
        vc = df[col].value_counts(dropna=False)
        for cat, count in vc.items():
            rows.append({
                "sample": label,
                "variable": col,
                "category": ("Missing" if pd.isna(cat)
                             else (int(cat) if isinstance(cat, (float, np.floating)) and float(cat).is_integer() else cat)),
                "n": int(count),
                "pct": round(100 * count / n_total, 2) if n_total > 0 else np.nan,
            })
    return pd.DataFrame(rows)


def phase_1_3_full_vs_analytic_cohort(raw_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Compares demographics (sex, age group, education, income) between:
      - the full loaded sample (raw BRFSS rows, before any outcome-based
        exclusion),
      - the pooled analytic cohort (data_preparation.py output, after
        excluding respondents missing all six outcomes),
      - each outcome-specific analytic sample (after excluding
        respondents missing that specific outcome, i.e. the mask used
        in model_ai.py for each disease model).

    [PATCH] Saves to TABLES_DIR (shared <repo>/tables/), which now
    defaults to the same directory make_manuscript_outputs.py checks
    for "S13 Table" -- phase1_3_full_vs_analytic_cohort.csv.
    """
    print("=== Phase 1.3: Full vs. analytic cohort demographic comparison ===")
    raw_dir = raw_dir or BRFSS_RAW_DIR
    years = [2011, 2012, 2013, 2014, 2015]

    full_frames = []
    for yr in years:
        infile = _resolve_year_file(yr, raw_dir)
        print(f"  Loading raw file for {yr}: {infile}")
        df_raw = pd.read_csv(infile)
        keep_cols = [c for c in DEMO_COLS_FOR_COMPARISON if c in df_raw.columns]
        df_raw = df_raw[keep_cols].copy()
        df_raw["YEAR"] = yr
        full_frames.append(df_raw)

    full_df = pd.concat(full_frames, ignore_index=True)
    print(f"  Full loaded sample (pre-exclusion), all years combined: n = {len(full_df):,}")

    # The analytic cohort has don't-know / refused codes set to NaN by data_preparation.py.
    # Apply the SAME per-column codes to the raw full sample, otherwise the two samples are
    # coded differently (e.g. age 14, education 9, income 9 would appear as categories in the
    # full sample but as "Missing" in the analytic samples) and the comparison is invalid.
    _DEMO_MISSING_CODES = {"SEX": [9], "_AGEG5YR": [14], "_EDUCAG": [9], "_INCOMG": [9]}
    for _col, _codes in _DEMO_MISSING_CODES.items():
        if _col in full_df.columns:
            full_df.loc[full_df[_col].isin(_codes), _col] = np.nan

    if not COMBINED_CSV.exists():
        raise FileNotFoundError(f"Cleaned/analytic dataset not found: {COMBINED_CSV}")
    analytic_df = pd.read_csv(COMBINED_CSV)
    print(f"  Pooled analytic cohort (data_preparation.py output): n = {len(analytic_df):,}")

    summaries = [
        _summarize_demographics(full_df, "Full loaded sample (pre-exclusion)"),
        _summarize_demographics(analytic_df, "Pooled analytic cohort"),
    ]

    for outcome in TARGET_COLS:
        if outcome not in analytic_df.columns:
            print(f"  WARNING: outcome '{outcome}' not found in analytic dataset, skipping.")
            continue
        sub = analytic_df[analytic_df[outcome].notna()]
        label = f"Analytic sample: {DISEASE_NAME_MAP.get(outcome, outcome)}"
        print(f"  {label}: n = {len(sub):,}")
        summaries.append(_summarize_demographics(sub, label))

    result = pd.concat(summaries, ignore_index=True)
    out_path = TABLES_DIR / "phase1_3_full_vs_analytic_cohort.csv"
    result.to_csv(out_path, index=False)
    print(f"\n  Saved -> {out_path}")

    # Sanity check: confirm every expected sample block is present
    # (1 full + 1 pooled + 6 disease-specific = 8 total). This is the
    # check flagged as worth doing manually before -- now automated.
    expected_labels = {"Full loaded sample (pre-exclusion)", "Pooled analytic cohort"} | {
        f"Analytic sample: {DISEASE_NAME_MAP.get(o, o)}" for o in TARGET_COLS
    }
    actual_labels = set(result["sample"].unique())
    missing_labels = expected_labels - actual_labels
    if missing_labels:
        print(f"\n  [WARNING] Expected {len(expected_labels)} sample blocks, "
              f"missing: {sorted(missing_labels)}")
    else:
        print(f"\n  [OK] All {len(expected_labels)} expected sample blocks present "
              f"(1 full + 1 pooled + {len(TARGET_COLS)} disease-specific).")

    return result


# ============================================================
# Main / CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Revision analyses, run by phase.")
    parser.add_argument(
        "--phase", type=str, default="0",
        help="Phase to run: '0' (setup+diagnostic), '1' (cheap items), or 'all'."
    )
    args = parser.parse_args()

    print(f"[paths] Tables  -> {TABLES_DIR}")
    print(f"[paths] Figures -> {FIGURES_DIR}")
    print(f"[paths] Frozen snapshots (model backups only) -> {FROZEN_DIR}")

    if args.phase in ("0", "all"):
        phase_0_1_freeze_artifacts()
        phase_0_2_pipeline_diagnostic()

    if args.phase in ("1", "all"):
        phase_1_1_mce_bin_detail(n_bins=10)
        try:
            phase_1_3_full_vs_analytic_cohort()
        except FileNotFoundError as e:
            print(f"\n  Phase 1.3 skipped: {e}")
            print("  (Set BRFSS_RAW_DIR to the directory containing your raw .csv/.csv.zip files.)")

    print("\nDone.")


if __name__ == "__main__":
    main()

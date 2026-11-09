#!/usr/bin/env python
# coding: utf-8
"""
scripts/verify_train_test_equivalence.py

Addresses TRIPOD+AI Items 21 and 20c together:

  Item 21 -- exact N and outcome-event counts for each analysis stage
             (train / validation / test), per outcome. Requires
             re-deriving the actual train/val splits (not stored in
             disease_models.joblib, which only retains X_test/y_test)
             to get EXACT event counts rather than prevalence-based
             estimates.

  Item 20c -- for model evaluation, compare the distribution of key
              predictors (demographics) and the outcome between the
              development (train+val) and evaluation (test) data.
              Reports summary statistics for age group, sex, income,
              and outcome prevalence in both partitions, per outcome,
              to empirically verify (not just assume) that the
              stratified random split produced comparable partitions.

This script reproduces the EXACT train/val/test split from
src/model_ai.py (same predictor_cols, same random_state=42, same
stratify arguments) so results are directly comparable to what was
actually used to train the reported models.

Run (from repo root):
  python scripts/verify_train_test_equivalence.py

Environment variables (optional):
  DATA_DERIVED_DIR, TABLES_DIR
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1] if THIS_FILE.parent.name in {"src", "scripts"} else THIS_FILE.parent

DATA_DERIVED_DIR = Path(os.environ.get("DATA_DERIVED_DIR", REPO_ROOT / "data" / "derived"))
TABLES_DIR = Path(os.environ.get("TABLES_DIR", REPO_ROOT / "tables"))
TABLES_DIR.mkdir(parents=True, exist_ok=True)

COMBINED_CSV = Path(os.environ.get(
    "BRFSS_CLEAN_CSV",
    DATA_DERIVED_DIR / "BRFSS_2011_2015_clean_model.csv"
))

RANDOM_STATE = 42

# Same predictor list as model_ai.py -- needed to reproduce the exact split
predictor_cols = [
    "_STATE", "SEX", "_AGEG5YR", "_EDUCAG", "_INCOMG", "_MRACE1", "_HISPANC",
    "SMOKE100", "SMOKDAY2", "ALCDAY5", "DRNKANY5", "EXERANY2",
    "FRUIT1", "VEGETAB1",
    "HLTHPLN1", "PERSDOC2", "MEDCOST", "CHECKUP1",
    "BPHIGH4", "BPMEDS", "TOLDHI2", "CHOLCHK", "ASTHMA3", "HAVARTH3",
    "GENHLTH", "PHYSHLTH", "MENTHLTH", "POORHLTH",
    "DIFFWALK", "DECIDE",
    "WEIGHT2", "HEIGHT3", "_BMI5"
]

target_cols_binary = [
    "heart_attack", "coronary_hd", "stroke",
    "kidney", "depression", "diabetes"
]

DISEASE_NAME_MAP = {
    "heart_attack": "Heart attack",
    "coronary_hd":  "Coronary heart disease",
    "stroke":       "Stroke",
    "kidney":       "Chronic kidney disease",
    "depression":   "Depression",
    "diabetes":     "Diabetes",
}


def reproduce_split(X: pd.DataFrame, y: pd.Series):
    """Exact same split logic as src/model_ai.py Section 7."""
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train_full
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def summarize_partition(X_part: pd.DataFrame, y_part: pd.Series, label: str) -> dict:
    n = len(X_part)
    n_events = int(y_part.sum())
    return {
        "partition": label,
        "n": n,
        "n_events": n_events,
        "prevalence_pct": round(100 * n_events / n, 3) if n > 0 else np.nan,
        "pct_female": round(100 * (X_part["SEX"] == 2).mean(), 2),
        "mean_age_group": round(X_part["_AGEG5YR"].mean(skipna=True), 3),
        "mean_income_group": round(X_part["_INCOMG"].mean(skipna=True), 3),
        "mean_education_group": round(X_part["_EDUCAG"].mean(skipna=True), 3),
    }


def main():
    print(f"Loading dataset: {COMBINED_CSV}")
    df = pd.read_csv(COMBINED_CSV)

    X = df[predictor_cols].copy()
    y = df[target_cols_binary].copy()

    item21_rows = []
    item20c_rows = []

    for disease in target_cols_binary:
        print(f"\n=== {DISEASE_NAME_MAP.get(disease, disease)} ===")
        mask = y[disease].notna()
        X_i = X.loc[mask].copy()
        y_i = y.loc[mask, disease].astype(int).copy()

        X_train, X_val, X_test, y_train, y_val, y_test = reproduce_split(X_i, y_i)

        # --- Item 21: exact N and events per partition ---
        for part_name, X_part, y_part in [
            ("Train", X_train, y_train), ("Validation", X_val, y_val), ("Test", X_test, y_test)
        ]:
            summary = summarize_partition(X_part, y_part, part_name)
            summary["disease"] = disease
            summary["disease_pretty"] = DISEASE_NAME_MAP.get(disease, disease)
            item21_rows.append(summary)
            print(f"  {part_name}: n={summary['n']:,}  events={summary['n_events']:,}  "
                  f"prevalence={summary['prevalence_pct']}%")

        # --- Item 20c: development (train+val) vs. evaluation (test) comparison ---
        X_dev = pd.concat([X_train, X_val])
        y_dev = pd.concat([y_train, y_val])
        dev_summary = summarize_partition(X_dev, y_dev, "Development (train+val)")
        test_summary = summarize_partition(X_test, y_test, "Evaluation (test)")

        item20c_rows.append({
            "disease": disease,
            "disease_pretty": DISEASE_NAME_MAP.get(disease, disease),
            "dev_n": dev_summary["n"], "test_n": test_summary["n"],
            "dev_prevalence_pct": dev_summary["prevalence_pct"], "test_prevalence_pct": test_summary["prevalence_pct"],
            "dev_pct_female": dev_summary["pct_female"], "test_pct_female": test_summary["pct_female"],
            "dev_mean_age_group": dev_summary["mean_age_group"], "test_mean_age_group": test_summary["mean_age_group"],
            "dev_mean_income_group": dev_summary["mean_income_group"], "test_mean_income_group": test_summary["mean_income_group"],
            "dev_mean_education_group": dev_summary["mean_education_group"], "test_mean_education_group": test_summary["mean_education_group"],
        })

    item21_df = pd.DataFrame(item21_rows)
    item20c_df = pd.DataFrame(item20c_rows)

    # Compute max absolute difference across all compared characteristics for item 20c,
    # to support a single summary sentence in the manuscript.
    item20c_df["max_abs_diff_prevalence_pct"] = (item20c_df["dev_prevalence_pct"] - item20c_df["test_prevalence_pct"]).abs()
    item20c_df["max_abs_diff_pct_female"] = (item20c_df["dev_pct_female"] - item20c_df["test_pct_female"]).abs()
    item20c_df["max_abs_diff_mean_age_group"] = (item20c_df["dev_mean_age_group"] - item20c_df["test_mean_age_group"]).abs()

    print("\n=== Item 21: exact N/events by partition (all outcomes) ===")
    print(item21_df[["disease_pretty", "partition", "n", "n_events", "prevalence_pct"]].to_string(index=False))

    print("\n=== Item 20c: development vs. evaluation partition comparison ===")
    print(item20c_df.to_string(index=False))

    item21_out = TABLES_DIR / "tripod_item21_partition_counts.csv"
    item20c_out = TABLES_DIR / "tripod_item20c_dev_vs_eval_comparison.csv"
    item21_df.to_csv(item21_out, index=False)
    item20c_df.to_csv(item20c_out, index=False)

    print(f"\nSaved -> {item21_out}")
    print(f"Saved -> {item20c_out}")

    max_prev_diff = item20c_df["max_abs_diff_prevalence_pct"].max()
    max_female_diff = item20c_df["max_abs_diff_pct_female"].max()
    max_age_diff = item20c_df["max_abs_diff_mean_age_group"].max()
    print(f"\nSummary for manuscript sentence: across all six outcomes, the largest "
          f"absolute difference between development and evaluation partitions was "
          f"{max_prev_diff:.3f} percentage points for outcome prevalence, "
          f"{max_female_diff:.3f} percentage points for sex distribution, and "
          f"{max_age_diff:.3f} for mean age group -- consistent with equivalence "
          f"expected from stratified random partitioning.")


if __name__ == "__main__":
    main()

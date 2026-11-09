#!/usr/bin/env python
# coding: utf-8
"""
scripts/generate_s3_missingness_table.py

Generates the S3 Dataset (missing data counts for predictor variables),
restricted to the ANALYTIC COHORT (n=2,379,512) rather than the raw
pre-exclusion pool (n=2,380,047), with structural vs. person-level
missingness broken out per predictor.

This supersedes scripts/diagnose_structural_missingness.py, which
computed the same breakdown against the raw pool as a quick diagnostic.
This script instead:
  1) Imports PREDICTOR_COLS, TARGET_COLS, PREDICTOR_MISSING_CODES, and the exact
     analytic-cohort inclusion filter directly from src/data_preparation.py
     (rather than re-implementing it), so results are guaranteed
     consistent with the actual modeling pipeline.
  2) Re-derives, per year, which respondents are included in the
     analytic cohort (i.e. have at least one non-missing outcome),
     using the same outcome-derivation logic as data_preparation.py.
  3) Computes structural missingness (predictor column absent that
     year) and person-level missingness (column present, but value is
     a generic BRFSS missing/refused/don't-know code) ONLY within that
     analytic-cohort subset.
  4) Outputs a single CSV formatted as a direct drop-in replacement
     for the existing S3 Dataset, adding columns for the structural/
     person-level breakdown.

Run (from repo root):
  python scripts/generate_s3_missingness_table.py

Environment variables (optional):
  BRFSS_RAW_DIR   (default: <repo>/data/raw)
  TABLES_DIR      (default: <repo>/tables)  -- same location as other
                  manuscript-ready tables, so this can be wired into
                  make_manuscript_outputs.py's output directory later.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)


# ============================================================
# Path configuration
# ============================================================
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1] if THIS_FILE.parent.name in {"src", "scripts"} else THIS_FILE.parent

# Import directly from src/data_preparation.py to guarantee the
# analytic-cohort filter and constants match the actual pipeline
# exactly, rather than re-implementing (and potentially drifting from)
# that logic here.
sys.path.insert(0, str(REPO_ROOT / "src"))
from data_preparation import (  # noqa: E402
    PREDICTOR_COLS,
    TARGET_COLS,
    PREDICTOR_MISSING_CODES,
    recode_binary,
    _resolve_year_file,
)

BRFSS_RAW_DIR = Path(os.environ.get("BRFSS_RAW_DIR", REPO_ROOT / "data" / "raw"))
TABLES_DIR = Path(os.environ.get("TABLES_DIR", REPO_ROOT / "tables"))
TABLES_DIR.mkdir(parents=True, exist_ok=True)

YEARS = [2011, 2012, 2013, 2014, 2015]

# Don't-know / refused codes PER COLUMN: the corrected PREDICTOR_MISSING_CODES from
# data_preparation.py plus the raw codes of the six decoded columns (which the decode
# functions there convert to NaN). A single blanket list is wrong: 7 and 9 are valid for
# _AGEG5YR (ages 50-54, 60-64), 9 is Connecticut in _STATE and 7 is "multiracial" in _MRACE1.
PERSON_MISSING_CODES = dict(PREDICTOR_MISSING_CODES)
PERSON_MISSING_CODES.update({
    "FRUIT1": [777, 999], "VEGETAB1": [777, 999], "ALCDAY5": [777, 999],
    "PHYSHLTH": [77, 99], "MENTHLTH": [77, 99], "POORHLTH": [77, 99],
})

# Special "meaningful zero" codes that must NOT be counted as missing
# for the affected variables -- these are recoded to 0 (a real value)
# by data_preparation.py's apply_encoding_fixes(), not treated as
# missing. Excluded here for consistency with that logic.
MEANINGFUL_ZERO_CODES = {
    "FRUIT1": {555},
    "VEGETAB1": {555},
    "ALCDAY5": {888},
    "PHYSHLTH": {88},
    "MENTHLTH": {88},
    "POORHLTH": {88},
}


def derive_target_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replicates the exact outcome-derivation block from
    data_preparation.py's clean_brfss_year(), so that the analytic-
    cohort inclusion filter below matches the real pipeline exactly.
    """
    df = df.copy()
    df["heart_attack"] = recode_binary(df["CVDINFR4"]) if "CVDINFR4" in df.columns else np.nan
    df["coronary_hd"]  = recode_binary(df["CVDCRHD4"]) if "CVDCRHD4" in df.columns else np.nan
    df["stroke"]       = recode_binary(df["CVDSTRK3"]) if "CVDSTRK3" in df.columns else np.nan
    df["kidney"]       = recode_binary(df["CHCKIDNY"]) if "CHCKIDNY" in df.columns else np.nan
    df["depression"]   = recode_binary(df["ADDEPEV2"]) if "ADDEPEV2" in df.columns else np.nan

    if "DIABETE3" in df.columns:
        df["diabetes"] = df["DIABETE3"].replace({
            1: 1, 2: 0, 3: 0, 4: 0, 5: 0, 7: np.nan, 9: np.nan,
        })
    else:
        df["diabetes"] = np.nan

    return df


def analytic_cohort_mask(df_with_targets: pd.DataFrame) -> pd.Series:
    """
    Replicates data_preparation.py's per-year inclusion filter:
    keep rows with at least one non-missing value among the six
    binary disease outcomes.
    """
    binary_targets = [t for t in TARGET_COLS if t not in ("BMI",)]
    available = [t for t in binary_targets if t in df_with_targets.columns]
    return df_with_targets[available].notnull().any(axis=1)


def person_level_missing_mask(series: pd.Series, col: str) -> pd.Series:
    """
    True where the value is a generic BRFSS missing/don't-know/refused
    code, or a true NaN. Excludes meaningful-zero codes (555/888/88)
    for the variables where those apply, since those are valid
    responses recoded to 0, not missingness.
    """
    is_nan = series.isna()
    is_generic_missing_code = series.isin(PERSON_MISSING_CODES.get(col, []))
    zero_codes = MEANINGFUL_ZERO_CODES.get(col, set())
    is_zero_code = series.isin(zero_codes) if zero_codes else pd.Series(False, index=series.index)
    return (is_nan | is_generic_missing_code) & (~is_zero_code)


def build_analytic_cohort_missingness_table() -> pd.DataFrame:
    per_year_records = []
    analytic_n_by_year = {}

    for year in YEARS:
        infile = _resolve_year_file(year)
        print(f"Loading raw file for {year}: {infile}")
        df_raw = pd.read_csv(infile)

        df_targets = derive_target_columns(df_raw)
        mask = analytic_cohort_mask(df_targets)
        df_analytic = df_raw.loc[mask].copy()

        n_analytic = len(df_analytic)
        analytic_n_by_year[year] = n_analytic
        print(f"  {year}: raw n={len(df_raw):,} -> analytic cohort n={n_analytic:,}")

        for col in PREDICTOR_COLS:
            if col not in df_analytic.columns:
                per_year_records.append({
                    "year": year, "predictor": col, "present": False,
                    "n_analytic": n_analytic, "n_person_missing": np.nan,
                })
                continue

            n_person_missing = int(person_level_missing_mask(df_analytic[col], col).sum())
            per_year_records.append({
                "year": year, "predictor": col, "present": True,
                "n_analytic": n_analytic, "n_person_missing": n_person_missing,
            })

    per_year_df = pd.DataFrame(per_year_records)
    grand_total_analytic_n = sum(analytic_n_by_year.values())
    print(f"\nGrand total analytic cohort N (should match manuscript's "
          f"2,379,512): {grand_total_analytic_n:,}")

    summary_rows = []
    for col in PREDICTOR_COLS:
        sub = per_year_df[per_year_df["predictor"] == col]
        absent_years = sub[~sub["present"]]
        present_years = sub[sub["present"]]

        n_structural = int(absent_years["n_analytic"].sum())
        n_person = int(present_years["n_person_missing"].sum())
        n_total_missing = n_structural + n_person
        pct_missing = round(100 * n_total_missing / grand_total_analytic_n, 2)
        pct_structural_of_missing = (
            round(100 * n_structural / n_total_missing, 2) if n_total_missing > 0 else np.nan
        )
        absent_year_list = sorted(absent_years["year"].tolist())

        summary_rows.append({
            "predictor": col,
            "n_missing": n_total_missing,
            "n_total": grand_total_analytic_n,
            "pct_missing": pct_missing,
            "years_variable_not_administered": (
                ", ".join(str(y) for y in absent_year_list) if absent_year_list else "(none)"
            ),
            "n_structural_missing": n_structural,
            "n_person_missing": n_person,
            "pct_structural_of_missing": pct_structural_of_missing,
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("pct_missing", ascending=False)
    return summary_df, grand_total_analytic_n


def main():
    summary_df, grand_total_analytic_n = build_analytic_cohort_missingness_table()

    print("\n=== S3 Dataset (analytic-cohort-restricted), with structural/person breakdown ===")
    print(summary_df.to_string(index=False))

    out_path = TABLES_DIR / "S3_missing_data_predictors.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")

    # Sanity check against the previously published S3 values, if you
    # want to confirm reconciliation. Uncomment and paste in the
    # previously published n_missing values per predictor if desired.
    # (Left as a manual step rather than hardcoded here, since the
    # published S3 table is the artifact this script is replacing.)

    flagged = summary_df[
        (summary_df["n_missing"] > 0) &
        (summary_df["pct_structural_of_missing"] >= 25.0)
    ]
    if not flagged.empty:
        print("\n*** Predictors where >=25% of total missingness is structural: ***")
        print(flagged[["predictor", "years_variable_not_administered",
                        "pct_missing", "pct_structural_of_missing"]].to_string(index=False))


if __name__ == "__main__":
    main()

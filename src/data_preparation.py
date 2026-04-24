#!/usr/bin/env python
# coding: utf-8
"""
data_preparation.py

Publication pipeline: Prepare a harmonized BRFSS 2011–2015 modeling dataset for
survey-based chronic disease risk prediction.

This script:
  1) Loads BRFSS annual extracts (2011–2015) from data/raw/
     - Expected filenames: 2011.csv.zip (preferred) or 2011.csv (fallback), etc.
  2) Derives binary outcome variables using BRFSS items (same mapping as your notebook)
  3) Selects the predictor columns used for training/inference
  4) Replaces BRFSS missing codes with NaN
  5) Keeps rows with at least one observed target (per-year)
  6) Concatenates across years and saves a pooled clean dataset to data/derived/

Default repo layout:
  <repo>/
    data/raw/2011.csv.zip ... 2015.csv.zip
    data/derived/BRFSS_2011_2015_clean_model.csv

Portability:
  - No absolute paths.
  - Override input/output locations with environment variables:
      BRFSS_RAW_DIR     -> directory containing yearly files
      BRFSS_DERIVED_DIR -> directory to write the pooled clean dataset
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Path configuration (portable, repo-relative)
# ============================================================
THIS_FILE = Path(__file__).resolve()
# If script is in src/, repo root is one level up; otherwise repo root is script dir
REPO_ROOT = THIS_FILE.parents[1] if THIS_FILE.parent.name in {"src", "scripts"} else THIS_FILE.parent

RAW_DIR = Path(os.environ.get("BRFSS_RAW_DIR", REPO_ROOT / "data" / "raw"))
DERIVED_DIR = Path(os.environ.get("BRFSS_DERIVED_DIR", REPO_ROOT / "data" / "derived"))

RAW_DIR.mkdir(parents=True, exist_ok=True)
DERIVED_DIR.mkdir(parents=True, exist_ok=True)

OUTFILE = DERIVED_DIR / "BRFSS_2011_2015_clean_model.csv"


# ============================================================
# Predictors and targets (same as your original script)
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

TARGET_COLS = [
    "heart_attack", "coronary_hd", "stroke",
    "kidney", "depression", "diabetes", "BMI",
]

# BRFSS missing codes used in your original workflow
MISSING_CODES = [7, 9, 77, 99, 777, 999, 7777, 9999]


# ============================================================
# Helpers (same logic as original)
# ============================================================
def recode_binary(series: pd.Series) -> pd.Series:
    """
    Recode BRFSS binary health outcomes:
      1 = Yes -> 1
      2 = No  -> 0
      7/9     -> NaN (missing/refused/don't know)
    """
    return series.replace({1: 1, 2: 0, 7: np.nan, 9: np.nan})


def _resolve_year_file(year: int) -> Path:
    """
    Resolve the preferred input file for a given year.
    Priority:
      1) data/raw/{year}.csv.zip
      2) data/raw/{year}.csv
    """
    zipped = RAW_DIR / f"{year}.csv.zip"
    plain = RAW_DIR / f"{year}.csv"

    if zipped.exists():
        return zipped
    if plain.exists():
        return plain

    raise FileNotFoundError(
        f"Could not find input file for year {year}. Expected one of:\n"
        f"  {zipped}\n"
        f"  {plain}\n"
        "If you store the files elsewhere, set BRFSS_RAW_DIR to that directory."
    )


def clean_brfss_year(year: int) -> pd.DataFrame:
    # ----- Load -----
    infile = _resolve_year_file(year)
    print(f"\n=== Cleaning BRFSS {year} from: {infile}")

    # pandas can read .zip containing a single CSV
    df = pd.read_csv(infile)

    # ----- Derive outcome variables (same logic as your original script) -----
    df["heart_attack"] = recode_binary(df["CVDINFR4"]) if "CVDINFR4" in df.columns else np.nan
    df["coronary_hd"]  = recode_binary(df["CVDCRHD4"]) if "CVDCRHD4" in df.columns else np.nan
    df["stroke"]       = recode_binary(df["CVDSTRK3"]) if "CVDSTRK3" in df.columns else np.nan
    df["kidney"]       = recode_binary(df["CHCKIDNY"]) if "CHCKIDNY" in df.columns else np.nan
    df["depression"]   = recode_binary(df["ADDEPEV2"]) if "ADDEPEV2" in df.columns else np.nan

    if "DIABETE3" in df.columns:
        # Your custom mapping for diabetes
        df["diabetes"] = df["DIABETE3"].replace({
            1: 1,        # Diabetes
            2: 0,        # No diabetes
            3: 0,        # No (borderline/prediabetes treated as 0 here)
            4: 0,
            5: 0,
            7: np.nan,   # Don't know / refused
            9: np.nan,
        })
    else:
        df["diabetes"] = np.nan

    df["BMI"] = (df["_BMI5"] / 100.0) if "_BMI5" in df.columns else np.nan

    # Keep provenance for pooled dataset
    df["YEAR"] = year

    # ----- Build model-ready dataset -----
    predictors_available = [c for c in PREDICTOR_COLS if c in df.columns]
    missing_pred = [c for c in PREDICTOR_COLS if c not in df.columns]
    if missing_pred:
        print(f"  Warning: {len(missing_pred)} predictors missing in {year}: {missing_pred}")

    targets_available = [c for c in TARGET_COLS if c in df.columns]

    # Include YEAR to enable year-stratified sensitivity checks later if needed
    cols_to_keep = ["YEAR"] + predictors_available + targets_available
    df_model = df[cols_to_keep].copy()

    # Replace BRFSS missing codes with NaN
    df_model = df_model.replace(MISSING_CODES, np.nan)

    # Keep rows with at least one non-missing target (per-year filtering)
    if targets_available:
        df_model = df_model[df_model[targets_available].notnull().any(axis=1)]

    print(f"  Cleaned {year} shape: {df_model.shape}")
    return df_model


def main() -> None:
    years = [2011, 2012, 2013, 2014, 2015]
    all_years = []

    for yr in years:
        df_clean = clean_brfss_year(yr)
        all_years.append(df_clean)

    combined = pd.concat(all_years, ignore_index=True)

    combined.to_csv(OUTFILE, index=False)
    print(f"\nCombined dataset saved to: {OUTFILE}")
    print(f"Combined shape: {combined.shape}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# coding: utf-8
"""
data_preparation.py

Publication pipeline: Prepare a harmonized BRFSS 2011-2015 modeling dataset for
survey-based chronic disease risk prediction.

REVISION UPDATE: this version adds three BRFSS-encoding corrections,
applied IN PLACE (same column names, same output filename as before)
so nothing downstream needs to change:

  FIX 1 -- Compound frequency encoding (FRUIT1, VEGETAB1, ALCDAY5):
    These variables use a compound day/week/month (or week/month, for
    ALCDAY5) code (e.g. 107 = 7x/day, 204 = 4x/week, 301 = 1x/month),
    plus special codes 555 ("Never", FRUIT1/VEGETAB1 only) and 888
    ("No drinks past 30 days", ALCDAY5 only) that are NOT ordinary
    missing-data codes. Verified against CDC's official FRUTDA1_ /
    VEGEDA1_ / DROCDY3_ calculated-variable formulas (2011-2015
    documentation).

  FIX 2 -- "88 = None" recode (PHYSHLTH, MENTHLTH, POORHLTH):
    These "number of days" variables use 88 = "None" (i.e. zero days),
    which is NOT an ordinary missing-data code either. Verified against
    CDC codebook reports (2013/2014/2016/2017/2018, identical
    convention throughout). This is the majority response for these
    variables (~55-70% of respondents in this cohort), so leaving 88
    unrecoded was a large-magnitude, high-prevalence data error.

FIX 3 -- Per-column missing codes (corrects a bug in the original generic
  replace). The previous version ran df.replace([7, 9, 77, 99, 777, 999,
  7777, 9999], NaN) over EVERY column. Those codes mean "don't know /
  refused" only for some variables; for others they are real values:
    _AGEG5YR : 7 = ages 50-54 and 9 = ages 60-64 (only 14 is missing).
               This blanked ~20.7% of respondents' age.
    _STATE   : 9 = Connecticut (FIPS).
    _MRACE1  : 7 = multiracial (missing is 77 / 99).
    PHYSHLTH / MENTHLTH / POORHLTH : after the 88="None" decode, real
               values of 7 and 9 days were blanked as well.
    FRUIT1 / VEGETAB1 : decoded rates of 7 or 9 per day were blanked.
  Missing codes are now defined per column in PREDICTOR_MISSING_CODES, and
  columns that are decoded by the functions above are not touched again.

ORDER OF OPERATIONS: the FIX 1 / FIX 2 decodes run first, on raw values,
so that 555 / 888 / 88 are converted to their meaning (0). The decoded
columns are then EXCLUDED from the missing-code step, because decoded
values can legitimately equal 7, 9, 77 or 99 (e.g. FRUIT1 = 107 decodes
to 7.0; MENTHLTH = 7 days). All other predictors get only their own
don't-know / refused codes replaced with NaN.

This script:
  1) Loads BRFSS annual extracts (2011-2015) from data/raw/
     - Expected filenames: 2011.csv.zip (preferred) or 2011.csv (fallback), etc.
  2) Derives binary outcome variables using BRFSS items
  3) Selects the predictor columns used for training/inference
  4) Applies the two encoding fixes above, in place
  5) Replaces each predictor's own don't-know / refused codes with NaN
     (per-column; see PREDICTOR_MISSING_CODES)
  6) Keeps rows with at least one observed target (per-year)
  7) Concatenates across years and saves a pooled clean dataset to data/derived/

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
# Path configuration (portable, repo-relative) -- unchanged
# ============================================================
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1] if THIS_FILE.parent.name in {"src", "scripts"} else THIS_FILE.parent

RAW_DIR = Path(os.environ.get("BRFSS_RAW_DIR", REPO_ROOT / "data" / "raw"))
DERIVED_DIR = Path(os.environ.get("BRFSS_DERIVED_DIR", REPO_ROOT / "data" / "derived"))

RAW_DIR.mkdir(parents=True, exist_ok=True)
DERIVED_DIR.mkdir(parents=True, exist_ok=True)

# Same filename as before -- overwritten in place on rerun.
OUTFILE = DERIVED_DIR / "BRFSS_2011_2015_clean_model.csv"


# ============================================================
# Predictors and targets (unchanged)
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

# Don't-know / refused / not-asked codes, PER PREDICTOR (BRFSS 2011-2015
# codebooks). Never apply one blanket list to all columns: 7, 9, 77 and 99
# are valid values for several predictors (see FIX 3 in the module docstring).
# An empty list means the column has no such codes.
PREDICTOR_MISSING_CODES = {
    "_STATE":   [],                 # all values are valid FIPS codes (9 = Connecticut)
    "SEX":      [9],
    "_AGEG5YR": [14],               # 1-13 are age bands; 7 and 9 are VALID
    "_EDUCAG":  [9],
    "_INCOMG":  [9],
    "_MRACE1":  [77, 99],           # 7 = multiracial is VALID
    "_HISPANC": [9],
    "SMOKE100": [7, 9], "SMOKDAY2": [7, 9], "DRNKANY5": [7, 9], "EXERANY2": [7, 9],
    "HLTHPLN1": [7, 9], "PERSDOC2": [7, 9], "MEDCOST":  [7, 9], "CHECKUP1": [7, 9],
    "BPHIGH4":  [7, 9], "BPMEDS":   [7, 9], "TOLDHI2":  [7, 9], "CHOLCHK":  [7, 9],
    "ASTHMA3":  [7, 9], "HAVARTH3": [7, 9], "GENHLTH":  [7, 9],
    "DIFFWALK": [7, 9], "DECIDE":   [7, 9],
    "WEIGHT2":  [777, 999, 7777, 9999],   # 777/999 lb are implausible and were blanked by the original step; kept blanked so only the 7/9/77/99 over-recoding changes results
    "HEIGHT3":  [777, 999, 7777, 9999],
    "_BMI5":    [9999],
}

# Columns fully handled by their decode functions (these already map their own
# don't-know / refused codes to NaN). Excluded from the missing-code step.
DECODED_COLS = {"FRUIT1", "VEGETAB1", "ALCDAY5", "PHYSHLTH", "MENTHLTH", "POORHLTH"}


# ============================================================
# Helpers
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


# ============================================================
# FIX 1: Compound frequency encoding decode
# (FRUIT1, VEGETAB1, ALCDAY5) -- verified against CDC's FRUTDA1_ /
# VEGEDA1_ / DROCDY3_ calculated-variable formulas.
# ============================================================
def _decode_day_week_month(value: float) -> float:
    """
    FRUIT1 / VEGETAB1: three-tier day/week/month compound code.
    Returns a continuous "times per day" rate. Replaces the raw
    column in place -- same column name, same semantic slot, now
    holding a genuinely continuous, correctly-scaled value.
    """
    if pd.isna(value):
        return np.nan
    v = float(value)

    if 100 < v < 200:
        return v - 100
    if 200 < v < 300:
        return round((v - 200) / 7.0, 2)
    if 300 < v < 400:
        return round((v - 300) / 30.0, 2)
    if v == 555:
        return 0.0
    if v == 300:
        # CDC's own documented edge case (FRUTDA1_/VEGEDA1_ formulas):
        # exactly 300 is not caught by "300 < v < 400" (strict), so
        # CDC special-cases it to a small nonzero rate.
        return 0.02
    if v in (777, 999):
        return np.nan
    return np.nan


def _decode_week_month(value: float) -> float:
    """
    ALCDAY5: two-tier week/month compound code (no per-day tier).
    Returns a continuous "times per day" rate. Verified against CDC's
    DROCDY3_ calculated variable and codebook value ranges
    (101-199 = days/week, 201-299 = days/month).
    """
    if pd.isna(value):
        return np.nan
    v = float(value)

    if 100 < v < 200:
        return round((v - 100) / 7.0, 4)
    if 200 < v < 300:
        return round((v - 200) / 30.0, 4)
    if v == 888:
        return 0.0
    if v in (777, 999):
        return np.nan
    return np.nan


# ============================================================
# FIX 2: "88 = None" recode (PHYSHLTH, MENTHLTH, POORHLTH) --
# verified against CDC codebook reports 2013/2014/2016/2017/2018.
# ============================================================
def _decode_none_88(value: float) -> float:
    """
    PHYSHLTH / MENTHLTH / POORHLTH: valid range 0-30 days.
    88 = "None" -> 0. 77/99 = don't know/refused -> NaN.
    """
    if pd.isna(value):
        return np.nan
    v = float(value)

    if 0 <= v <= 30:
        return v
    if v == 88:
        return 0.0
    if v in (77, 99):
        return np.nan
    return np.nan


# Column -> decode function mapping, applied in place before the
# per-column missing-code step.
_ENCODING_FIXES = {
    "FRUIT1":   _decode_day_week_month,
    "VEGETAB1": _decode_day_week_month,
    "ALCDAY5":  _decode_week_month,
    "PHYSHLTH": _decode_none_88,
    "MENTHLTH": _decode_none_88,
    "POORHLTH": _decode_none_88,
}


def apply_encoding_fixes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies FIX 1 and FIX 2 in place, column by column, BEFORE the
    the per-column missing-code step runs. Prints a brief summary per
    column for auditability in the pipeline log.
    """
    df = df.copy()
    for col, decode_fn in _ENCODING_FIXES.items():
        if col not in df.columns:
            continue
        n_special = df[col].isin([555, 888, 88]).sum()
        df[col] = df[col].apply(decode_fn)
        if n_special > 0:
            print(f"    Encoding fix applied to {col}: "
                  f"{n_special:,} special-code values (555/888/88) recoded.")
    return df


def apply_missing_codes(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Replace each predictor's own don't-know / refused codes with NaN.
    Decoded columns are skipped. Prints how many values were blanked per
    column so over-recoding is visible in the pipeline log.
    """
    df = df.copy()
    uncovered = [c for c in PREDICTOR_COLS
                 if c in df.columns and c not in DECODED_COLS and c not in PREDICTOR_MISSING_CODES]
    if uncovered:
        raise KeyError(f"No missing-code rule defined for predictors: {uncovered}")

    blanked = {}
    for col, codes in PREDICTOR_MISSING_CODES.items():
        if col not in df.columns or not codes:
            continue
        m = df[col].isin(codes)
        if m.any():
            blanked[col] = int(m.sum())
            df.loc[m, col] = np.nan
    if blanked:
        print("    Missing-code replacement (values set to NaN): "
              + ", ".join(f"{k}={v:,}" for k, v in blanked.items()))

    # Guard against the original bug: valid age bands must survive.
    if "_AGEG5YR" in df.columns:
        present = set(df["_AGEG5YR"].dropna().astype(int).unique())
        assert {7, 9} <= present, (
            f"{year}: _AGEG5YR categories 7 and 9 were removed; "
            "check PREDICTOR_MISSING_CODES.")
    return df


def clean_brfss_year(year: int) -> pd.DataFrame:
    # ----- Load -----
    infile = _resolve_year_file(year)
    print(f"\n=== Cleaning BRFSS {year} from: {infile}")

    # pandas can read .zip containing a single CSV
    df = pd.read_csv(infile)

    # ----- Derive outcome variables (same logic as original script) -----
    df["heart_attack"] = recode_binary(df["CVDINFR4"]) if "CVDINFR4" in df.columns else np.nan
    df["coronary_hd"]  = recode_binary(df["CVDCRHD4"]) if "CVDCRHD4" in df.columns else np.nan
    df["stroke"]       = recode_binary(df["CVDSTRK3"]) if "CVDSTRK3" in df.columns else np.nan
    df["kidney"]       = recode_binary(df["CHCKIDNY"]) if "CHCKIDNY" in df.columns else np.nan
    df["depression"]   = recode_binary(df["ADDEPEV2"]) if "ADDEPEV2" in df.columns else np.nan

    if "DIABETE3" in df.columns:
        df["diabetes"] = df["DIABETE3"].replace({
            1: 1,        # Yes
            2: 0,        # Yes, but only during pregnancy (gestational) -> coded as NOT having diabetes, not excluded
            3: 0,        # No
            4: 0,        # No, pre-diabetes or borderline diabetes
            7: np.nan,   # Don't know / not sure
            9: np.nan,   # Refused
        })
    else:
        df["diabetes"] = np.nan

    df["BMI"] = (df["_BMI5"] / 100.0) if "_BMI5" in df.columns else np.nan
    df["YEAR"] = year

    # ----- Build model-ready dataset -----
    predictors_available = [c for c in PREDICTOR_COLS if c in df.columns]
    missing_pred = [c for c in PREDICTOR_COLS if c not in df.columns]
    if missing_pred:
        print(f"  Warning: {len(missing_pred)} predictors missing in {year}: {missing_pred}")

    targets_available = [c for c in TARGET_COLS if c in df.columns]

    cols_to_keep = ["YEAR"] + predictors_available + targets_available
    df_model = df[cols_to_keep].copy()

    # ----- FIX 1 + FIX 2: apply compound-encoding / "88=None" decode
    #       IN PLACE, on raw values, BEFORE the per-column missing-code
    #       step below. See module docstring for rationale. -----
    print(f"  Applying BRFSS encoding fixes (compound frequency + 88=None) for {year}...")
    df_model = apply_encoding_fixes(df_model)

    # ----- Replace each predictor's own don't-know / refused codes with NaN -----
    df_model = apply_missing_codes(df_model, year)

    # Keep rows with at least one non-missing DISEASE outcome (per-year filtering).
    # BMI is a continuous target and is deliberately NOT part of this filter: including
    # it kept respondents with no disease outcome at all, which made the derived file
    # 221 rows larger (2,379,733) than the 2,379,512-respondent analytic cohort.
    binary_targets = [c for c in TARGET_COLS if c != "BMI" and c in df_model.columns]
    if binary_targets:
        df_model = df_model[df_model[binary_targets].notnull().any(axis=1)]

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

#!/usr/bin/env python
# coding: utf-8
"""
generate_flowchart_data.py

Generates participant flow counts for the STROBE flow diagram.
Tracks, per year and overall:
  1) Total BRFSS records loaded
  2) Records excluded: missing ALL six outcomes
  3) Final analytic cohort (at least one outcome non-missing)
  4) Per-outcome: included (non-missing) and excluded (missing) counts
  5) Missing predictor counts for the final analytic cohort

Outputs:
  data/derived/flowchart_overall.csv            -- one row per year + pooled total
  data/derived/flowchart_per_outcome.csv        -- per-outcome inclusion/exclusion per year + pooled
  data/derived/flowchart_missing_predictors.csv -- missing predictor counts in final cohort

Usage:
  python generate_flowchart_data.py

Override data paths with environment variables:
  BRFSS_RAW_DIR     -- directory containing yearly raw BRFSS files
  BRFSS_DERIVED_DIR -- directory for outputs
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Path configuration (mirrors data_preparation.py)
# ============================================================
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = (
    THIS_FILE.parents[1]
    if THIS_FILE.parent.name in {"src", "scripts"}
    else THIS_FILE.parent
)

RAW_DIR     = Path(os.environ.get("BRFSS_RAW_DIR",     REPO_ROOT / "data" / "raw"))
DERIVED_DIR = Path(os.environ.get("BRFSS_DERIVED_DIR", REPO_ROOT / "data" / "derived"))
DERIVED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Variable mappings (same as data_preparation.py)
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

OUTCOME_SOURCE = {
    "heart_attack": "CVDINFR4",
    "coronary_hd":  "CVDCRHD4",
    "stroke":       "CVDSTRK3",
    "kidney":       "CHCKIDNY",
    "depression":   "ADDEPEV2",
    "diabetes":     "DIABETE3",
}

MISSING_CODES = [7, 9, 77, 99, 777, 999, 7777, 9999]


# ============================================================
# Helpers
# ============================================================
def recode_binary(series: pd.Series) -> pd.Series:
    return series.replace({1: 1, 2: 0, 7: np.nan, 9: np.nan}).astype("float64")


def recode_diabetes(series: pd.Series) -> pd.Series:
    return series.replace(
        {1: 1, 2: 0, 3: 0, 4: 0, 5: 0, 7: np.nan, 9: np.nan}
    ).astype("float64")


def resolve_year_file(year: int) -> Path:
    for ext in [".csv.zip", ".csv"]:
        p = RAW_DIR / f"{year}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No input file found for year {year}. "
        f"Expected {RAW_DIR / str(year)}.csv.zip or .csv"
    )


def load_and_recode(year: int) -> pd.DataFrame:
    """
    Load raw BRFSS file, recode outcomes, and return a clean dataframe.
    Outcome columns are built separately and concatenated once to avoid
    the PerformanceWarning from repeated fragmented inserts.
    """
    infile = resolve_year_file(year)
    print(f"Loading {year} from: {infile}")
    df = pd.read_csv(infile, low_memory=False)

    # Build all new columns in a dict, then concat once — avoids fragmentation
    new_cols: dict[str, pd.Series] = {}

    for outcome, src in OUTCOME_SOURCE.items():
        if src in df.columns:
            if outcome == "diabetes":
                new_cols[outcome] = recode_diabetes(df[src])
            else:
                new_cols[outcome] = recode_binary(df[src])
        else:
            new_cols[outcome] = pd.Series(np.nan, index=df.index, dtype="float64")
            print(f"  Warning: {src} not found in {year} — {outcome} set to NaN")

    new_cols["YEAR"] = pd.Series(year, index=df.index, dtype="int32")

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


# ============================================================
# Flow counts
# ============================================================
def compute_flow(df: pd.DataFrame, year: int) -> dict:
    """Compute flow counts for one year's dataframe."""
    outcomes = list(OUTCOME_SOURCE.keys())
    n_total = len(df)

    all_missing_mask = df[outcomes].isnull().all(axis=1)
    n_excluded_all   = int(all_missing_mask.sum())
    n_analytic       = int((~all_missing_mask).sum())

    row: dict = {
        "year":                            str(year),
        "n_total_loaded":                  n_total,
        "n_excluded_all_outcomes_missing": n_excluded_all,
        "n_analytic_cohort":               n_analytic,
    }

    df_analytic = df.loc[~all_missing_mask]
    for outcome in outcomes:
        n_inc  = int(df_analytic[outcome].notnull().sum())
        n_excl = int(df_analytic[outcome].isnull().sum())
        n_pos  = int((df_analytic[outcome] == 1).sum())
        n_neg  = int((df_analytic[outcome] == 0).sum())
        row[f"{outcome}_included"]         = n_inc
        row[f"{outcome}_excluded_missing"] = n_excl
        row[f"{outcome}_positive"]         = n_pos
        row[f"{outcome}_negative"]         = n_neg

    return row


def compute_predictor_missingness(df_analytic: pd.DataFrame) -> pd.DataFrame:
    """Count missing values per predictor in the pooled analytic cohort."""
    available = [c for c in PREDICTOR_COLS if c in df_analytic.columns]
    counts = df_analytic[available].isnull().sum().reset_index()
    counts.columns = ["predictor", "n_missing"]
    counts["n_total"]     = len(df_analytic)
    counts["pct_missing"] = (counts["n_missing"] / counts["n_total"] * 100).round(2)
    return counts.sort_values("pct_missing", ascending=False)


# ============================================================
# Pooled totals helpers
# ============================================================
def pool_flow_rows(flow_rows: list[dict]) -> dict:
    """Sum all numeric fields across years; label year as 'ALL'."""
    pooled: dict = {"year": "ALL"}
    numeric_keys = [k for k in flow_rows[0] if k != "year"]
    for k in numeric_keys:
        pooled[k] = sum(r[k] for r in flow_rows)
    return pooled


def pool_per_outcome(per_outcome_df: pd.DataFrame, outcomes: list[str]) -> list[dict]:
    pooled_rows = []
    for outcome in outcomes:
        sub = per_outcome_df[per_outcome_df["outcome"] == outcome]
        n_inc = sub["n_included"].sum()
        n_pos = sub["n_positive"].sum()
        pooled_rows.append({
            "year":               "ALL",
            "outcome":            outcome,
            "n_analytic_cohort":  sub["n_analytic_cohort"].sum(),
            "n_included":         n_inc,
            "n_excluded_missing": sub["n_excluded_missing"].sum(),
            "n_positive":         n_pos,
            "n_negative":         sub["n_negative"].sum(),
            "prevalence_pct":     round(n_pos / n_inc * 100, 2) if n_inc > 0 else np.nan,
        })
    return pooled_rows


# ============================================================
# Main
# ============================================================
def main() -> None:
    years    = [2011, 2012, 2013, 2014, 2015]
    outcomes = list(OUTCOME_SOURCE.keys())

    flow_rows:      list[dict]        = []
    analytic_frames: list[pd.DataFrame] = []

    for yr in years:
        df = load_and_recode(yr)
        df = df.replace(MISSING_CODES, np.nan)

        flow_rows.append(compute_flow(df, yr))

        analytic_mask = df[outcomes].notnull().any(axis=1)
        analytic_frames.append(df.loc[analytic_mask])

    # ---- Overall flow table ----
    pooled_flow = pool_flow_rows(flow_rows)
    flow_df = pd.DataFrame(flow_rows + [pooled_flow])

    out_flow = DERIVED_DIR / "flowchart_overall.csv"
    flow_df.to_csv(out_flow, index=False)
    print(f"\nOverall flow table saved to: {out_flow}")

    # ---- Per-outcome table (long format) ----
    per_outcome_rows: list[dict] = []
    for row in flow_rows:
        yr = row["year"]
        for outcome in outcomes:
            n_inc = row[f"{outcome}_included"]
            n_pos = row[f"{outcome}_positive"]
            per_outcome_rows.append({
                "year":               yr,
                "outcome":            outcome,
                "n_analytic_cohort":  row["n_analytic_cohort"],
                "n_included":         n_inc,
                "n_excluded_missing": row[f"{outcome}_excluded_missing"],
                "n_positive":         n_pos,
                "n_negative":         row[f"{outcome}_negative"],
                "prevalence_pct":     round(n_pos / n_inc * 100, 2) if n_inc > 0 else np.nan,
            })

    per_outcome_df = pd.DataFrame(per_outcome_rows)
    pooled_outcome = pool_per_outcome(per_outcome_df, outcomes)
    per_outcome_df = pd.concat(
        [per_outcome_df, pd.DataFrame(pooled_outcome)], ignore_index=True
    )

    out_outcome = DERIVED_DIR / "flowchart_per_outcome.csv"
    per_outcome_df.to_csv(out_outcome, index=False)
    print(f"Per-outcome flow table saved to: {out_outcome}")

    # ---- Predictor missingness in pooled analytic cohort ----
    df_analytic_all = pd.concat(analytic_frames, ignore_index=True)
    missing_df = compute_predictor_missingness(df_analytic_all)

    out_missing = DERIVED_DIR / "flowchart_missing_predictors.csv"
    missing_df.to_csv(out_missing, index=False)
    print(f"Predictor missingness table saved to: {out_missing}")

    # ---- Console summary ----
    p = pooled_flow
    print("\n" + "=" * 62)
    print("FLOW DIAGRAM SUMMARY (pooled 2011–2015)")
    print("=" * 62)
    print(f"  Total records loaded:              {p['n_total_loaded']:>12,}")
    print(f"  Excluded (all outcomes missing):   {p['n_excluded_all_outcomes_missing']:>12,}")
    print(f"  Final analytic cohort:             {p['n_analytic_cohort']:>12,}")
    print()
    print(f"  {'Outcome':<22} {'Included':>10} {'Excl.(miss.)':>13} {'Positive':>10} {'Prev.%':>8}")
    print(f"  {'-' * 65}")
    for r in pooled_outcome:
        print(
            f"  {r['outcome']:<22} {r['n_included']:>10,} "
            f"{r['n_excluded_missing']:>13,} {r['n_positive']:>10,} "
            f"{r['prevalence_pct']:>7.2f}%"
        )
    print("=" * 62)


if __name__ == "__main__":
    main()
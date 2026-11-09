#!/usr/bin/env python
# coding: utf-8
"""
scripts/generate_participant_flow.py

Reconstructs the participant-flow numbers behind Fig 4 (the CONSORT-
style flow diagram: BRFSS respondents assessed -> excluded (all
outcomes missing) -> final analytic cohort -> six disease-specific
included/excluded/positive/prevalence boxes), plus the underlying
S1 and S2 Dataset flow-count tables described in the manuscript's
Supporting Information captions.

WHY THIS IS A SEPARATE SCRIPT (not just re-running data_preparation.py):
data_preparation.py's clean_brfss_year() computes the "keep rows with
>=1 non-missing outcome" filter and then IMMEDIATELY drops the
excluded rows before returning -- main() only ever sees and saves the
post-drop, concatenated dataset. That means:
  - the raw per-year N before any exclusion (n_total_loaded), and
  - the count of respondents excluded for having all six outcomes
    missing (n_excluded_all_outcomes_missing)
are never written anywhere; they exist for a moment inside that
function and are then discarded. Fig 5 and the S13/S14 Datasets need
both of those numbers, so this script has to reconstruct them.

HOW IT AVOIDS DRIFT: rather than re-implementing outcome derivation,
predictor selection, or the encoding fixes a second time (which could
silently fall out of sync if data_preparation.py is edited later),
this script imports data_preparation.py directly as a module and
reuses its actual functions and constants:
  dp.recode_binary, dp.PREDICTOR_COLS, dp.TARGET_COLS,
  dp.apply_encoding_fixes, dp.apply_missing_codes, dp._resolve_year_file,
  dp.OUTFILE
It reproduces clean_brfss_year()'s logic step-for-step, but returns
the dataframe BEFORE the final "keep >=1 non-missing outcome" drop, so
both the excluded rows and the retained (analytic-cohort) rows are
visible in the same place.

OUTPUTS:
  tables/S1_participant_flow_by_year.csv
      One row per survey year + a pooled "ALL" row. Wide format:
      n_total_loaded, n_excluded_all_outcomes_missing,
      n_analytic_cohort, then four columns per outcome
      (n_included / n_excluded_missing / n_positive / n_negative).
  tables/S2_participant_flow_by_outcome.csv
      Long format: one row per outcome per year (+ "ALL"), with
      n_analytic_cohort, n_included, n_excluded_missing, n_positive,
      n_negative, prevalence_pct.
  tables/fig4_flow_summary.csv
      A small, Fig-4-ready table: one summary row (total assessed /
      excluded-all-missing / final analytic cohort) followed by one
      row per disease (included / excluded / positive / prevalence%),
      formatted to drop straight into the flow-diagram boxes.

CROSS-CHECK: after reconstructing the pooled analytic cohort, this
script compares its row count against the actual saved
data/derived/BRFSS_2011_2015_clean_model.csv (if present) and prints
an explicit OK/WARNING, so any future change to data_preparation.py
that isn't mirrored here gets caught immediately rather than silently
producing a mismatched flow diagram.

Run (from repo root):
  python scripts/generate_participant_flow.py

Environment variables (optional):
  BRFSS_RAW_DIR       (default: <repo>/data/raw)      -- passed through to data_preparation
  BRFSS_DERIVED_DIR   (default: <repo>/data/derived)   -- passed through to data_preparation
  BRFSS_DATA_PREP_DIR (default: auto-search this script's dir, src/, scripts/, repo root)
                      -- set this explicitly if data_preparation.py
                         lives somewhere else, e.g.:
                         BRFSS_DATA_PREP_DIR=/path/to/src python scripts/generate_participant_flow.py
  TABLES_DIR          (default: <repo>/tables)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ============================================================
# Path configuration + import data_preparation.py as a module
#
# data_preparation.py is not necessarily in the SAME directory as this
# script (e.g. this script lives in scripts/ while data_preparation.py
# lives in src/, or vice versa, or both sit at repo root). Rather than
# assuming same-directory, we search a small set of plausible
# locations relative to the repo root and add whichever one actually
# contains it to sys.path before importing. Set BRFSS_DATA_PREP_DIR
# explicitly to override/skip the search entirely if your layout
# doesn't match any of the candidates below.
# ============================================================
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1] if THIS_FILE.parent.name in {"src", "scripts"} else THIS_FILE.parent

_env_override = os.environ.get("BRFSS_DATA_PREP_DIR")
_candidate_dirs = (
    [Path(_env_override)] if _env_override else [
        THIS_FILE.parent,          # same directory as this script
        REPO_ROOT / "src",
        REPO_ROOT / "scripts",
        REPO_ROOT,                 # repo root itself
    ]
)

_found_dir = None
for _d in _candidate_dirs:
    if (_d / "data_preparation.py").exists():
        _found_dir = _d
        break

if _found_dir is None:
    searched = "\n  ".join(str(d / "data_preparation.py") for d in _candidate_dirs)
    raise ImportError(
        "Could not find data_preparation.py in any of the expected locations:\n"
        f"  {searched}\n"
        "Set the BRFSS_DATA_PREP_DIR environment variable to the directory "
        "containing data_preparation.py, e.g.:\n"
        '  BRFSS_DATA_PREP_DIR=/path/to/dir python scripts/generate_participant_flow.py'
    )

if str(_found_dir) not in sys.path:
    sys.path.insert(0, str(_found_dir))

try:
    import data_preparation as dp  # noqa: E402
except ImportError as e:
    raise ImportError(
        f"Found data_preparation.py at {_found_dir}, but it failed to import "
        f"(see underlying error above). Check that its own dependencies "
        f"(numpy, pandas) are installed in this environment."
    ) from e

print(f"[import] Using data_preparation.py from: {_found_dir / 'data_preparation.py'}")

TABLES_DIR = Path(os.environ.get("TABLES_DIR", REPO_ROOT / "tables"))
TABLES_DIR.mkdir(parents=True, exist_ok=True)

YEARS = [2011, 2012, 2013, 2014, 2015]

OUTCOME_DISPLAY = {
    "heart_attack": "Heart attack",
    "coronary_hd":  "Coronary heart disease",
    "stroke":       "Stroke",
    "kidney":       "Chronic kidney disease",
    "depression":   "Depression",
    "diabetes":     "Diabetes",
}
OUTCOME_COLS = list(OUTCOME_DISPLAY.keys())


# ============================================================
# Step 1: reproduce clean_brfss_year()'s logic, WITHOUT the final
# "keep rows with >=1 non-missing outcome" drop, so both the excluded
# and retained rows remain visible.
# ============================================================
def build_unfiltered_year(year: int) -> Tuple[int, pd.DataFrame]:
    """
    Mirrors data_preparation.clean_brfss_year() exactly (outcome
    derivation, predictor/target selection, encoding fixes, generic
    missing-code replace), but returns the dataframe BEFORE the final
    "keep >=1 non-missing outcome" filter, plus the raw pre-exclusion
    row count for the year.

    Returns
    -------
    (n_total_loaded, df_model_unfiltered)
    """
    infile = dp._resolve_year_file(year)
    df = pd.read_csv(infile)
    n_total_loaded = len(df)

    # ----- Derive outcome variables (identical to clean_brfss_year) -----
    df["heart_attack"] = dp.recode_binary(df["CVDINFR4"]) if "CVDINFR4" in df.columns else np.nan
    df["coronary_hd"]  = dp.recode_binary(df["CVDCRHD4"]) if "CVDCRHD4" in df.columns else np.nan
    df["stroke"]       = dp.recode_binary(df["CVDSTRK3"]) if "CVDSTRK3" in df.columns else np.nan
    df["kidney"]       = dp.recode_binary(df["CHCKIDNY"]) if "CHCKIDNY" in df.columns else np.nan
    df["depression"]   = dp.recode_binary(df["ADDEPEV2"]) if "ADDEPEV2" in df.columns else np.nan

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

    predictors_available = [c for c in dp.PREDICTOR_COLS if c in df.columns]
    targets_available = [c for c in dp.TARGET_COLS if c in df.columns]
    cols_to_keep = ["YEAR"] + predictors_available + targets_available
    df_model = df[cols_to_keep].copy()

    # Same encoding fixes + per-column missing-code step as the real
    # pipeline (the old blanket replace over all columns was a bug). These only ever touch predictor columns (FRUIT1,
    # VEGETAB1, ALCDAY5, PHYSHLTH, MENTHLTH, POORHLTH) plus the
    # already-binary-recoded outcome columns (which have no 7/9/77/99
    # left to replace), so outcome missingness/positivity here will
    # exactly match what data_preparation.py itself produces.
    df_model = dp.apply_encoding_fixes(df_model)
    df_model = dp.apply_missing_codes(df_model, year)   # per-column codes (7/9 are valid for age, state, race)

    return n_total_loaded, df_model


# ============================================================
# Step 2: compute the flow statistics for one year
# ============================================================
def compute_flow_for_year(year: int) -> Tuple[dict, List[dict], pd.DataFrame]:
    """
    Returns
    -------
    year_summary : dict
        One S1-style row for this year (wide format).
    outcome_rows : list[dict]
        One S2-style row per outcome for this year (long format).
    df_analytic : pd.DataFrame
        The analytic-cohort rows for this year (>=1 outcome
        non-missing) -- i.e. exactly what
        data_preparation.clean_brfss_year() returns for this year.
        Returned so the pooled version can be cross-checked against
        the actual saved dataset at the end of main().
    """
    n_total_loaded, df_model = build_unfiltered_year(year)

    outcomes_present = [c for c in OUTCOME_COLS if c in df_model.columns]
    has_any_outcome = df_model[outcomes_present].notna().any(axis=1)
    n_excluded_all_missing = int((~has_any_outcome).sum())
    df_analytic = df_model[has_any_outcome].copy()
    n_analytic = len(df_analytic)

    year_summary = {
        "year": year,
        "n_total_loaded": n_total_loaded,
        "n_excluded_all_outcomes_missing": n_excluded_all_missing,
        "n_analytic_cohort": n_analytic,
    }

    outcome_rows = []
    for oc in outcomes_present:
        col = df_analytic[oc]
        n_included = int(col.notna().sum())
        n_excluded_missing = n_analytic - n_included
        n_positive = int(col.sum(skipna=True))
        n_negative = n_included - n_positive
        prevalence_pct = round(100 * n_positive / n_included, 2) if n_included > 0 else np.nan

        year_summary[f"{oc}_n_included"] = n_included
        year_summary[f"{oc}_n_excluded_missing"] = n_excluded_missing
        year_summary[f"{oc}_n_positive"] = n_positive
        year_summary[f"{oc}_n_negative"] = n_negative

        outcome_rows.append({
            "year": year,
            "outcome": oc,
            "outcome_pretty": OUTCOME_DISPLAY[oc],
            "n_analytic_cohort": n_analytic,
            "n_included": n_included,
            "n_excluded_missing": n_excluded_missing,
            "n_positive": n_positive,
            "n_negative": n_negative,
            "prevalence_pct": prevalence_pct,
        })

    return year_summary, outcome_rows, df_analytic


# ============================================================
# Step 3: pool across years and assemble a Fig-4-ready summary table
# ============================================================
def make_fig4_summary_table(all_summary: dict, pooled_outcome_rows: List[dict]) -> pd.DataFrame:
    """
    Produces a small, copy-paste-ready table matching the flow
    diagram's boxes exactly: one top row for the three overall counts,
    then one row per disease with Included/Excluded/Positive/Prev.%.
    """
    rows = [{
        "Box": "BRFSS respondents assessed (2011-2015)",
        "N_or_Included": all_summary["n_total_loaded"],
        "Excluded": np.nan,
        "Positive": np.nan,
        "Prevalence_%": np.nan,
    }, {
        "Box": "Excluded (all outcomes missing)",
        "N_or_Included": np.nan,
        "Excluded": all_summary["n_excluded_all_outcomes_missing"],
        "Positive": np.nan,
        "Prevalence_%": np.nan,
    }, {
        "Box": "Final analytic cohort",
        "N_or_Included": all_summary["n_analytic_cohort"],
        "Excluded": np.nan,
        "Positive": np.nan,
        "Prevalence_%": np.nan,
    }]
    for row in pooled_outcome_rows:
        rows.append({
            "Box": row["outcome_pretty"],
            "N_or_Included": row["n_included"],
            "Excluded": row["n_excluded_missing"],
            "Positive": row["n_positive"],
            "Prevalence_%": row["prevalence_pct"],
        })
    return pd.DataFrame(rows)


def main() -> None:
    print("== Participant flow reconstruction (Fig 4 / S1 & S2 Datasets) ==")
    print("Repo root:", REPO_ROOT)
    print("Tables dir:", TABLES_DIR)

    per_year_summaries: List[dict] = []
    per_year_outcome_rows: List[dict] = []
    analytic_frames: List[pd.DataFrame] = []

    for yr in YEARS:
        print(f"\n--- {yr} ---")
        year_summary, outcome_rows, df_analytic = compute_flow_for_year(yr)
        print(f"  n_total_loaded:                   {year_summary['n_total_loaded']:>10,}")
        print(f"  n_excluded_all_outcomes_missing:  {year_summary['n_excluded_all_outcomes_missing']:>10,}")
        print(f"  n_analytic_cohort:                {year_summary['n_analytic_cohort']:>10,}")
        per_year_summaries.append(year_summary)
        per_year_outcome_rows.extend(outcome_rows)
        analytic_frames.append(df_analytic)

    # ----- Pool across years -----
    n_total_loaded_all = sum(y["n_total_loaded"] for y in per_year_summaries)
    n_excluded_all = sum(y["n_excluded_all_outcomes_missing"] for y in per_year_summaries)
    n_analytic_all = sum(y["n_analytic_cohort"] for y in per_year_summaries)

    all_summary = {
        "year": "ALL",
        "n_total_loaded": n_total_loaded_all,
        "n_excluded_all_outcomes_missing": n_excluded_all,
        "n_analytic_cohort": n_analytic_all,
    }

    pooled_analytic = pd.concat(analytic_frames, ignore_index=True)

    pooled_outcome_rows: List[dict] = []
    for oc in OUTCOME_COLS:
        if oc not in pooled_analytic.columns:
            continue
        col = pooled_analytic[oc]
        n_included = int(col.notna().sum())
        n_excluded_missing = n_analytic_all - n_included
        n_positive = int(col.sum(skipna=True))
        n_negative = n_included - n_positive
        prevalence_pct = round(100 * n_positive / n_included, 2) if n_included > 0 else np.nan

        all_summary[f"{oc}_n_included"] = n_included
        all_summary[f"{oc}_n_excluded_missing"] = n_excluded_missing
        all_summary[f"{oc}_n_positive"] = n_positive
        all_summary[f"{oc}_n_negative"] = n_negative

        pooled_outcome_rows.append({
            "year": "ALL",
            "outcome": oc,
            "outcome_pretty": OUTCOME_DISPLAY[oc],
            "n_analytic_cohort": n_analytic_all,
            "n_included": n_included,
            "n_excluded_missing": n_excluded_missing,
            "n_positive": n_positive,
            "n_negative": n_negative,
            "prevalence_pct": prevalence_pct,
        })

    per_year_summaries.append(all_summary)
    per_year_outcome_rows.extend(pooled_outcome_rows)

    # ----- Save S1 Dataset (wide, one row per year + ALL) -----
    s1_df = pd.DataFrame(per_year_summaries)
    s1_path = TABLES_DIR / "S1_participant_flow_by_year.csv"
    s1_df.to_csv(s1_path, index=False)
    print(f"\nSaved -> {s1_path}")

    # ----- Save S2 Dataset (long, one row per outcome per year + ALL) -----
    s2_df = pd.DataFrame(per_year_outcome_rows)
    s2_path = TABLES_DIR / "S2_participant_flow_by_outcome.csv"
    s2_df.to_csv(s2_path, index=False)
    print(f"Saved -> {s2_path}")

    # ----- Save Fig-4-ready summary table -----
    fig4_df = make_fig4_summary_table(all_summary, pooled_outcome_rows)
    fig4_path = TABLES_DIR / "fig4_flow_summary.csv"
    fig4_df.to_csv(fig4_path, index=False)
    print(f"Saved -> {fig4_path}")

    # ----- Cross-check against the actual pooled clean dataset -----
    if dp.OUTFILE.exists():
        actual = pd.read_csv(dp.OUTFILE)
        if len(actual) != n_analytic_all:
            print(
                f"\n[WARNING] Reconstructed analytic cohort N ({n_analytic_all:,}) does NOT "
                f"match the actual pooled dataset row count ({len(actual):,}) at {dp.OUTFILE}.\n"
                f"This means data_preparation.py has changed since this script was last "
                f"aligned with it, or the saved dataset is stale. Re-run data_preparation.py, "
                f"then re-run this script, before trusting the numbers above for Fig 4."
            )
        else:
            print(f"\n[OK] Reconstructed analytic cohort N ({n_analytic_all:,}) matches "
                  f"the actual pooled dataset at {dp.OUTFILE}.")
    else:
        print(f"\n[NOTE] Pooled dataset not found at {dp.OUTFILE} -- skipping cross-check. "
              f"Run data_preparation.py first for a full consistency check.")

    # ----- Print a Fig-4-ready summary block for quick manual copy -----
    print("\n" + "=" * 72)
    print("FIG 4 / PARTICIPANT FLOW DIAGRAM -- updated numbers")
    print("=" * 72)
    print(f"BRFSS respondents assessed (2011-2015):  N = {n_total_loaded_all:,}")
    print(f"Excluded (all outcomes missing):         N = {n_excluded_all:,}")
    print(f"Final analytic cohort:                   N = {n_analytic_all:,}")
    print("-" * 72)
    for row in pooled_outcome_rows:
        print(f"{row['outcome_pretty']:<24} "
              f"Included={row['n_included']:>10,}  "
              f"Excluded={row['n_excluded_missing']:>7,}  "
              f"Positive={row['n_positive']:>8,}  "
              f"Prev.%={row['prevalence_pct']:>6.2f}%")
    print("=" * 72)
    print(
        "\nSanity check: for every disease above, Included + Excluded should equal "
        f"the final analytic cohort ({n_analytic_all:,}). If any row does not sum "
        "correctly, something has changed in outcome derivation or filtering -- "
        "investigate before updating the manuscript figure."
    )


if __name__ == "__main__":
    main()

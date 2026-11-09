#!/usr/bin/env python
# coding: utf-8
"""
scripts/refresh_flagged_cluster_summary.py

Lightweight re-filter of the already-computed full SHAP ranking table
(tables/global_shap_full_ranking_all_outcomes.csv, produced by
generate_global_shap.py) against the EXPANDED 10-variable
FLAGGED_CLUSTER definition. Does NOT recompute SHAP values -- this
just re-derives the cluster summary from data that already exists,
since the original run's FLAGGED_CLUSTER only covered 6 of the 10
structurally-missing variables.

Run this instead of re-running generate_global_shap.py in full.

Run (from repo root):
  python scripts/refresh_flagged_cluster_summary.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1] if THIS_FILE.parent.name in {"src", "scripts"} else THIS_FILE.parent
TABLES_DIR = Path(os.environ.get("TABLES_DIR", REPO_ROOT / "tables"))

FLAGGED_CLUSTER = ["BPHIGH4", "BPMEDS", "TOLDHI2", "CHOLCHK", "FRUIT1", "VEGETAB1",
                   "_MRACE1", "_HISPANC", "DIFFWALK", "DECIDE"]


def main():
    full_ranking_path = TABLES_DIR / "global_shap_full_ranking_all_outcomes.csv"
    if not full_ranking_path.exists():
        raise FileNotFoundError(
            f"{full_ranking_path} not found. Run generate_global_shap.py first."
        )

    full_ranking = pd.read_csv(full_ranking_path)

    cluster_rows = full_ranking[full_ranking["underlying_brfss_var"].isin(FLAGGED_CLUSTER)].copy()
    cluster_summary = (
        cluster_rows.sort_values("mean_abs_shap", ascending=False)
        .groupby(["disease", "underlying_brfss_var"], as_index=False)
        .first()[["disease", "disease_pretty", "underlying_brfss_var", "feature",
                   "feature_pretty", "rank", "mean_abs_shap"]]
        .sort_values(["underlying_brfss_var", "disease"])
    )

    out_path = TABLES_DIR / "global_shap_flagged_cluster_summary.csv"
    cluster_summary.to_csv(out_path, index=False)

    print(f"Refreshed cluster summary (10 variables) -> {out_path}")
    print(cluster_summary.to_string(index=False))


if __name__ == "__main__":
    main()

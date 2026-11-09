#!/usr/bin/env python
# coding: utf-8
"""
scripts/generate_global_shap.py

Computes population-level (global) SHAP feature importance for all six
disease models. Addresses a review comment (global
interpretability) and provides the evidence used to finalize the
predictor-retention decision for the structurally-missing predictor
cluster (BPHIGH4, BPMEDS, TOLDHI2, CHOLCHK, FRUIT1, VEGETAB1, plus
_MRACE1, _HISPANC, DIFFWALK, DECIDE).

COMPUTE-ONLY: this script does NOT generate figures. Plotting (bar
charts + beeswarm plots) lives in scripts/make_manuscript_outputs.py,
consistent with that script's role as the single orchestrator for all
manuscript figures/tables. This script's job is solely to run the
(comparatively expensive) TreeExplainer computation once and save its
outputs to CSV, so make_manuscript_outputs.py can be rerun cheaply
(e.g. to tweak plot styling) without re-running SHAP each time.

OUTPUTS, per disease:
  - tables/global_shap_top15_all_outcomes.csv
  - tables/global_shap_full_ranking_all_outcomes.csv
  - tables/global_shap_flagged_cluster_summary.csv
  - tables/global_shap/shap_beeswarm_values_<disease>.csv
  - tables/global_shap/shap_feature_values_<disease>.csv

Labels are generated via scripts/feature_labels.py (shared module).

Run (from repo root):
  python scripts/generate_global_shap.py
  python scripts/generate_global_shap.py --sample-size 50000
  python scripts/generate_global_shap.py --sample-size none

Environment variables (optional):
  ARTIFACT_DIR    (default: <repo>/app/artifacts)
  TABLES_DIR      (default: <repo>/tables)
"""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import joblib
import shap

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1] if THIS_FILE.parent.name in {"src", "scripts"} else THIS_FILE.parent

ARTIFACT_DIR = Path(os.environ.get("ARTIFACT_DIR", REPO_ROOT / "app" / "artifacts"))
TABLES_DIR   = Path(os.environ.get("TABLES_DIR",   REPO_ROOT / "tables"))
SHAP_DIR     = TABLES_DIR / "global_shap"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
SHAP_DIR.mkdir(parents=True, exist_ok=True)

_FEATURE_LABELS_DIR = THIS_FILE.parent
if str(_FEATURE_LABELS_DIR) not in sys.path:
    sys.path.insert(0, str(_FEATURE_LABELS_DIR))

_feature_labels_module_path = _FEATURE_LABELS_DIR / "feature_labels.py"
if not _feature_labels_module_path.exists():
    raise FileNotFoundError(
        f"Expected scripts/feature_labels.py at {_feature_labels_module_path}, "
        f"but it was not found. Save feature_labels.py to "
        f"{_FEATURE_LABELS_DIR} before rerunning."
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

RANDOM_STATE = 42

DISEASE_NAME_MAP = {
    "heart_attack": "Heart attack",
    "coronary_hd":  "Coronary heart disease",
    "stroke":       "Stroke",
    "kidney":       "Chronic kidney disease",
    "depression":   "Depression",
    "diabetes":     "Diabetes",
}

FLAGGED_CLUSTER = ["BPHIGH4", "BPMEDS", "TOLDHI2", "CHOLCHK", "FRUIT1", "VEGETAB1",
                   "_MRACE1", "_HISPANC", "DIFFWALK", "DECIDE"]


def pretty_disease_name(d: str) -> str:
    return DISEASE_NAME_MAP.get(d, d)


def underlying_brfss_var(feature_name: str) -> Optional[str]:
    s = str(feature_name)
    if s.lower().startswith(("num__", "cat__")):
        s = s[5:]
    for col in FLAGGED_CLUSTER:
        if s == col or s.startswith(col + "_"):
            return col
    return None


def load_artifacts():
    disease_models_path = ARTIFACT_DIR / "disease_models.joblib"
    if not disease_models_path.exists():
        raise FileNotFoundError(f"Missing required artifact: {disease_models_path}")
    return joblib.load(disease_models_path)


def compute_global_shap_for_disease(
    disease: str,
    info: Dict[str, Any],
    sample_size: Optional[int],
) -> Dict[str, Any]:
    model = info["model"]
    X_test = info["X_test"]
    y_test = np.asarray(info["y_test"], dtype=int)

    if not hasattr(model, "named_steps") or "preprocessor" not in model.named_steps or "clf" not in model.named_steps:
        raise ValueError(f"Model for {disease} does not have expected pipeline structure.")

    preprocessor = model.named_steps["preprocessor"]
    clf = model.named_steps["clf"]

    if sample_size is not None and sample_size < len(X_test):
        rng = np.random.RandomState(RANDOM_STATE)
        idx_pos = np.where(y_test == 1)[0]
        idx_neg = np.where(y_test == 0)[0]
        prevalence = len(idx_pos) / len(y_test)
        n_pos = max(1, int(round(sample_size * prevalence)))
        n_neg = sample_size - n_pos
        n_pos = min(n_pos, len(idx_pos))
        n_neg = min(n_neg, len(idx_neg))
        sampled_idx = np.concatenate([
            rng.choice(idx_pos, size=n_pos, replace=False),
            rng.choice(idx_neg, size=n_neg, replace=False),
        ])
        rng.shuffle(sampled_idx)
        X_sample = X_test.iloc[sampled_idx].reset_index(drop=True)
        print(f"  [{disease}] Using stratified subsample: n={len(X_sample):,} "
              f"(of {len(X_test):,} test-set rows, prevalence preserved at "
              f"{prevalence:.4f})")
    else:
        X_sample = X_test.reset_index(drop=True)
        print(f"  [{disease}] Using full test set: n={len(X_sample):,}")

    X_t = preprocessor.transform(X_sample)
    if hasattr(X_t, "toarray"):
        X_t = X_t.toarray()

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = np.array([f"f{i}" for i in range(X_t.shape[1])], dtype=object)

    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(X_t)
    if isinstance(sv, list) and len(sv) == 2:
        sv = sv[1]
    sv = np.asarray(sv)

    mean_abs_shap = np.mean(np.abs(sv), axis=0)
    ranking = pd.DataFrame({
        "feature": feature_names,
        "feature_pretty": [pretty_feature_from_explainer(f) for f in feature_names],
        "underlying_brfss_var": [underlying_brfss_var(f) for f in feature_names],
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    ranking["disease"] = disease
    ranking["disease_pretty"] = DISEASE_NAME_MAP.get(disease, disease)

    beeswarm_df = pd.DataFrame(sv, columns=feature_names)
    beeswarm_df.insert(0, "disease", disease)

    feature_values_df = pd.DataFrame(X_t, columns=feature_names)
    feature_values_df.insert(0, "disease", disease)

    return {
        "ranking": ranking,
        "beeswarm": beeswarm_df,
        "feature_values": feature_values_df,
        "feature_names": feature_names,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-size", type=str, default="20000",
        help="Stratified subsample size per outcome for SHAP computation. "
             "Pass 'none' to use the full test set (much slower)."
    )
    args = parser.parse_args()
    sample_size = None if args.sample_size.lower() == "none" else int(args.sample_size)

    disease_models = load_artifacts()

    all_rankings = []
    for disease, info in disease_models.items():
        print(f"\n=== Global SHAP: {disease} ===")
        result = compute_global_shap_for_disease(disease, info, sample_size)
        all_rankings.append(result["ranking"])

        beeswarm_path = SHAP_DIR / f"shap_beeswarm_values_{disease}.csv"
        result["beeswarm"].to_csv(beeswarm_path, index=False)
        print(f"  Saved beeswarm data -> {beeswarm_path}")

        feature_values_path = SHAP_DIR / f"shap_feature_values_{disease}.csv"
        result["feature_values"].to_csv(feature_values_path, index=False)
        print(f"  Saved corresponding feature values -> {feature_values_path}")

        top15 = result["ranking"].head(15)
        print(f"  Top 15 features for {pretty_disease_name(disease)}:")
        print(top15[["rank", "feature_pretty", "mean_abs_shap"]].to_string(index=False))

    combined_ranking = pd.concat(all_rankings, ignore_index=True)
    ranking_out_path = TABLES_DIR / "global_shap_top15_all_outcomes.csv"
    combined_ranking[combined_ranking["rank"] <= 15].to_csv(ranking_out_path, index=False)
    print(f"\nSaved combined top-15 ranking table (all outcomes) -> {ranking_out_path}")

    full_ranking_out_path = TABLES_DIR / "global_shap_full_ranking_all_outcomes.csv"
    combined_ranking.to_csv(full_ranking_out_path, index=False)
    print(f"Saved full ranking table (all outcomes, all features) -> {full_ranking_out_path}")

    cluster_rows = combined_ranking[combined_ranking["underlying_brfss_var"].isin(FLAGGED_CLUSTER)].copy()
    cluster_summary = (
        cluster_rows.sort_values("mean_abs_shap", ascending=False)
        .groupby(["disease", "underlying_brfss_var"], as_index=False)
        .first()[["disease", "disease_pretty", "underlying_brfss_var", "feature",
                   "feature_pretty", "rank", "mean_abs_shap"]]
        .sort_values(["underlying_brfss_var", "disease"])
    )
    cluster_out_path = TABLES_DIR / "global_shap_flagged_cluster_summary.csv"
    cluster_summary.to_csv(cluster_out_path, index=False)
    print("\n=== Flagged structurally-missing predictor cluster: rank by outcome ===")
    print(cluster_summary.to_string(index=False))
    print(f"\nSaved -> {cluster_out_path}")

    print(
        "\nDone. Run scripts/make_manuscript_outputs.py next to generate the "
        "corresponding global SHAP figures (bar chart + beeswarm plots) from "
        "the tables saved above."
    )


if __name__ == "__main__":
    main()

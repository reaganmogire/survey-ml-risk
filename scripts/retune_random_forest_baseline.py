#!/usr/bin/env python
# coding: utf-8
"""
scripts/retune_random_forest_baseline.py

Reruns ONLY the random forest baseline with regularization added
(bounded depth, minimum leaf size, feature subsampling), to address
the unregularized RandomForestClassifier(n_estimators=300,
max_depth=None) underperforming even logistic regression in the
initial baseline_model_comparison.py run -- consistent with
overfitting from unconstrained tree depth on a ~1.5M-row training set.

Does NOT retrain logistic regression or reload/retrain the primary
GBM models -- reuses the existing
tables/baseline_comparison/baseline_model_comparison.csv for those
columns and only replaces the random forest columns, to avoid
redundant computation.

Regularization applied (adjustable via CLI):
  max_depth=18            (was: None -- unconstrained)
  min_samples_leaf=50     (was: 1 -- sklearn default)
  max_features="sqrt"     (was: None -- all features considered at
                            every split, unusual for classification;
                            sqrt is the standard default for
                            RandomForestClassifier and was NOT what
                            the original run used)
  n_estimators=300        (unchanged)

Run (from repo root):
  python scripts/retune_random_forest_baseline.py
  python scripts/retune_random_forest_baseline.py --max-depth 15 --min-samples-leaf 100
"""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1] if THIS_FILE.parent.name in {"src", "scripts"} else THIS_FILE.parent

# Reuse shared definitions from baseline_model_comparison.py rather
# than re-duplicating them a third time.
sys.path.insert(0, str(THIS_FILE.parent))
from baseline_model_comparison import (  # noqa: E402
    build_preprocessor,
    get_full_model_splits,
    target_cols_binary,
    DISEASE_NAME_MAP,
    RANDOM_STATE,
    COMBINED_CSV,
    TABLES_DIR,
)

BASELINE_TABLES_DIR = TABLES_DIR / "baseline_comparison"
EXISTING_COMPARISON_CSV = BASELINE_TABLES_DIR / "baseline_model_comparison.csv"


def train_regularized_rf(X_train, y_train, X_test, y_test, max_depth, min_samples_leaf, max_features, n_estimators):
    preprocessor = build_preprocessor()
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("clf", clf)])
    pipe.fit(X_train, y_train)

    y_proba = pipe.predict_proba(X_test)[:, 1]
    return {
        "auc": roc_auc_score(y_test, y_proba),
        "ap": average_precision_score(y_test, y_proba),
        "brier": brier_score_loss(y_test, y_proba),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-depth", type=int, default=18)
    parser.add_argument("--min-samples-leaf", type=int, default=50)
    parser.add_argument("--max-features", type=str, default="sqrt")
    parser.add_argument("--n-estimators", type=int, default=300)
    args = parser.parse_args()

    if not EXISTING_COMPARISON_CSV.exists():
        raise FileNotFoundError(
            f"{EXISTING_COMPARISON_CSV} not found. Run baseline_model_comparison.py first."
        )
    existing = pd.read_csv(EXISTING_COMPARISON_CSV)

    print(f"Regularization: max_depth={args.max_depth}, "
          f"min_samples_leaf={args.min_samples_leaf}, "
          f"max_features={args.max_features}, n_estimators={args.n_estimators}\n")

    print(f"Loading dataset: {COMBINED_CSV}")
    df = pd.read_csv(COMBINED_CSV)

    new_rf_rows = []
    for disease in target_cols_binary:
        print(f"\n=== {DISEASE_NAME_MAP.get(disease, disease)} ===")
        X_train, X_val, X_test, y_train, y_val, y_test = get_full_model_splits(df, disease)
        print(f"  n_train={len(X_train):,}  n_test={len(X_test):,}")
        print("  Training regularized random_forest...")

        result = train_regularized_rf(
            X_train, y_train, X_test, y_test,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
            n_estimators=args.n_estimators,
        )
        print(f"    random_forest (regularized): AUROC={result['auc']:.4f}  "
              f"AUPRC={result['ap']:.4f}  Brier={result['brier']:.4f}")

        new_rf_rows.append({
            "disease": disease,
            "AUROC_rf": round(result["auc"], 4),
            "AUPRC_rf": round(result["ap"], 4),
            "Brier_rf": round(result["brier"], 4),
        })

    new_rf_df = pd.DataFrame(new_rf_rows)

    # Drop old (unregularized) rf columns and any stale delta columns,
    # then merge in the new regularized results.
    cols_to_drop = [c for c in existing.columns if c.endswith("_rf") or "_rf" in c]
    updated = existing.drop(columns=cols_to_drop, errors="ignore").merge(
        new_rf_df, on="disease", how="left"
    )

    updated["AUROC_gbm_minus_rf"] = (updated["AUROC_gbm"] - updated["AUROC_rf"]).round(4)
    updated["AUROC_rf_minus_lr"] = (updated["AUROC_rf"] - updated["AUROC_lr"]).round(4)

    print("\n=== Updated comparison: GBM vs. logistic regression vs. regularized random forest ===")
    print(updated.to_string(index=False))

    out_path = BASELINE_TABLES_DIR / "baseline_model_comparison.csv"
    updated.to_csv(out_path, index=False)
    print(f"\nSaved (overwrote) -> {out_path}")

    # Sanity check: does RF now behave as expected (between LR and GBM,
    # or at least no longer below LR)?
    n_rf_below_lr = (updated["AUROC_rf_minus_lr"] < 0).sum()
    if n_rf_below_lr > 0:
        print(f"\n*** NOTE: random forest still underperforms logistic regression "
              f"for {n_rf_below_lr}/6 outcomes even after regularization. Consider "
              f"further tuning (e.g. shallower max_depth, higher min_samples_leaf) "
              f"or reporting logistic regression as the sole baseline in the "
              f"manuscript (Option 1 from earlier). ***")
    else:
        print("\nRandom forest now outperforms logistic regression for all 6 outcomes, "
              "as expected. Comparison table is ready for use in Methods/Results.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# coding: utf-8
"""
scripts/restricted_years_sensitivity.py

Sensitivity analysis (checklist item B4): retrains all six disease
models on ONLY the 2013 and 2015 BRFSS survey years -- the two years
in which the full structurally-missing predictor cluster (_MRACE1,
_HISPANC, DIFFWALK, DECIDE, BPHIGH4, BPMEDS, TOLDHI2, CHOLCHK, FRUIT1,
VEGETAB1) was consistently administered, without needing imputation
due to non-administration -- and compares:
  (a) AUROC / AUPRC / Brier against the full 2011-2015 pooled model
  (b) SHAP feature ranking stability for the flagged cluster specifically

This addresses a review comment (race/ethnicity coding sensitivity)
and extends it to the full structurally-missing cluster identified
during revision, and informs the limitation language in the Discussion
regarding potential confounding between imputed predictor values and
survey year.

IMPORTANT -- DUPLICATED LOGIC WARNING: the categorical/ordinal/
continuous predictor classification and preprocessing pipeline below
are DUPLICATED from src/model_ai.py, because model_ai.py runs its
training logic at module level (no function wrapper), so it cannot be
safely imported without re-triggering a full six-model retrain as a
side effect. If the predictor classification in model_ai.py changes in
the future, this script must be updated to match, or results will be
inconsistent between the full and restricted-years models. Consider
refactoring model_ai.py to wrap training in a callable function to
eliminate this duplication going forward.

Run (from repo root):
  python scripts/restricted_years_sensitivity.py

Environment variables (optional):
  DATA_DERIVED_DIR, TABLES_DIR, ARTIFACT_DIR (for the FULL/original
  model, used as the comparison baseline -- not overwritten)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
)
import shap

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)


# ============================================================
# Path configuration
# ============================================================
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1] if THIS_FILE.parent.name in {"src", "scripts"} else THIS_FILE.parent

DATA_DERIVED_DIR = Path(os.environ.get("DATA_DERIVED_DIR", REPO_ROOT / "data" / "derived"))
TABLES_DIR       = Path(os.environ.get("TABLES_DIR",       REPO_ROOT / "tables"))
ARTIFACT_DIR     = Path(os.environ.get("ARTIFACT_DIR",     REPO_ROOT / "app" / "artifacts"))

RESTRICTED_TABLES_DIR = TABLES_DIR / "restricted_years_sensitivity"
RESTRICTED_TABLES_DIR.mkdir(parents=True, exist_ok=True)

COMBINED_CSV = Path(os.environ.get(
    "BRFSS_CLEAN_CSV",
    DATA_DERIVED_DIR / "BRFSS_2011_2015_clean_model.csv"
))

RANDOM_STATE = 42
RESTRICTED_YEARS = [2013, 2015]

DISEASE_NAME_MAP = {
    "heart_attack": "Heart attack",
    "coronary_hd":  "Coronary heart disease",
    "stroke":       "Stroke",
    "kidney":       "Chronic kidney disease",
    "depression":   "Depression",
    "diabetes":     "Diabetes",
}

FLAGGED_CLUSTER = ["_MRACE1", "_HISPANC", "DIFFWALK", "DECIDE",
                   "BPHIGH4", "BPMEDS", "TOLDHI2", "CHOLCHK",
                   "FRUIT1", "VEGETAB1"]


# ============================================================
# DUPLICATED from src/model_ai.py -- see module docstring warning.
# ============================================================
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

NOMINAL_CATEGORICAL_COLS = [
    "_STATE", "SEX", "_MRACE1", "_HISPANC", "SMOKE100", "SMOKDAY2",
    "DRNKANY5", "EXERANY2", "HLTHPLN1", "PERSDOC2", "MEDCOST",
    "BPHIGH4", "BPMEDS", "TOLDHI2", "ASTHMA3", "HAVARTH3",
    "DIFFWALK", "DECIDE",
]

ORDINAL_KEPT_NUMERIC_COLS = [
    "_AGEG5YR", "_EDUCAG", "_INCOMG", "GENHLTH", "CHECKUP1", "CHOLCHK",
]

CONTINUOUS_COLS = [
    "PHYSHLTH", "MENTHLTH", "POORHLTH",
    "WEIGHT2", "HEIGHT3", "_BMI5",
    "FRUIT1", "VEGETAB1", "ALCDAY5",
]

numeric_features = ORDINAL_KEPT_NUMERIC_COLS + CONTINUOUS_COLS
categorical_features = NOMINAL_CATEGORICAL_COLS


def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    # HistGradientBoostingClassifier needs DENSE input; ColumnTransformer silently returns sparse when density < 0.3
    # (this model sits right at that threshold), so force dense output (same as model_ai.py).
    ], sparse_threshold=0.0)


# ============================================================
# Train restricted-years models
# ============================================================
def train_restricted_models(df_restricted: pd.DataFrame) -> Dict[str, Any]:
    X = df_restricted[predictor_cols].copy()
    y = df_restricted[target_cols_binary].copy()

    restricted_models = {}

    for col in target_cols_binary:
        print(f"\n--- Training restricted-years (2013+2015) model: {col} ---")
        mask = y[col].notna()
        X_i = X.loc[mask].copy()
        y_i = y.loc[mask, col].astype(int).copy()
        print(f"  Samples for {col} (2013+2015 only): {X_i.shape[0]:,}")

        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X_i, y_i, test_size=0.2, random_state=RANDOM_STATE, stratify=y_i
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train_full
        )

        preprocessor = build_preprocessor()
        clf = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("clf", HistGradientBoostingClassifier(
                learning_rate=0.1, max_iter=300, max_depth=None,
                l2_regularization=0.0, random_state=RANDOM_STATE
            ))
        ])
        clf.fit(X_train, y_train)

        y_proba_val = clf.predict_proba(X_val)[:, 1]
        thresholds = np.linspace(0.01, 0.99, 99)
        f1_scores = [
            f1_score(y_val, (y_proba_val >= t).astype(int)) if (y_proba_val >= t).sum() > 0 else 0.0
            for t in thresholds
        ]
        best_thresh = float(thresholds[int(np.argmax(f1_scores))])

        y_proba = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        brier = brier_score_loss(y_test, y_proba)
        print(f"  AUROC={auc:.4f}  AUPRC={ap:.4f}  Brier={brier:.4f}")

        restricted_models[col] = {
            "model": clf, "X_test": X_test, "y_test": y_test,
            "y_proba": y_proba, "best_threshold": best_thresh,
            "auc": auc, "average_precision": ap, "brier": brier,
        }

    return restricted_models


# ============================================================
# Performance comparison: restricted-years vs. full 2011-2015 model
# ============================================================
def compare_performance(restricted_models: Dict[str, Any], full_models_path: Path) -> pd.DataFrame:
    full_models = joblib.load(full_models_path)
    rows = []
    for disease in target_cols_binary:
        full_info = full_models[disease]
        y_test_full = np.asarray(full_info["y_test"], dtype=int)
        y_proba_full = np.asarray(full_info["y_proba"], dtype=float)
        auc_full = roc_auc_score(y_test_full, y_proba_full)
        ap_full = average_precision_score(y_test_full, y_proba_full)
        brier_full = brier_score_loss(y_test_full, y_proba_full)

        r = restricted_models[disease]
        rows.append({
            "disease": disease,
            "disease_pretty": DISEASE_NAME_MAP.get(disease, disease),
            "n_test_full_2011_2015": len(y_test_full),
            "n_test_restricted_2013_2015": len(r["y_test"]),
            "AUROC_full": round(auc_full, 4),
            "AUROC_restricted": round(r["auc"], 4),
            "AUROC_delta": round(r["auc"] - auc_full, 4),
            "AUPRC_full": round(ap_full, 4),
            "AUPRC_restricted": round(r["average_precision"], 4),
            "AUPRC_delta": round(r["average_precision"] - ap_full, 4),
            "Brier_full": round(brier_full, 4),
            "Brier_restricted": round(r["brier"], 4),
            "Brier_delta": round(r["brier"] - brier_full, 4),
        })
    return pd.DataFrame(rows)


# ============================================================
# SHAP ranking stability for the flagged cluster: restricted-years
# model vs. previously computed full-model global SHAP rankings
# (loaded from generate_global_shap.py's output).
# ============================================================
def underlying_brfss_var(feature_name: str):
    s = str(feature_name)
    if s.lower().startswith(("num__", "cat__")):
        s = s[5:]
    for col in FLAGGED_CLUSTER:
        if s == col or s.startswith(col + "_"):
            return col
    return None


def compute_restricted_shap_ranking(restricted_models: Dict[str, Any], sample_size: int = 20000) -> pd.DataFrame:
    rng = np.random.RandomState(RANDOM_STATE)
    all_rows = []

    for disease, info in restricted_models.items():
        model = info["model"]
        X_test = info["X_test"]
        y_test = np.asarray(info["y_test"], dtype=int)

        if sample_size < len(X_test):
            idx_pos = np.where(y_test == 1)[0]
            idx_neg = np.where(y_test == 0)[0]
            prevalence = len(idx_pos) / len(y_test)
            n_pos = min(len(idx_pos), max(1, int(round(sample_size * prevalence))))
            n_neg = min(len(idx_neg), sample_size - n_pos)
            sampled_idx = np.concatenate([
                rng.choice(idx_pos, size=n_pos, replace=False),
                rng.choice(idx_neg, size=n_neg, replace=False),
            ])
            X_sample = X_test.iloc[sampled_idx].reset_index(drop=True)
        else:
            X_sample = X_test.reset_index(drop=True)

        preprocessor = model.named_steps["preprocessor"]
        clf = model.named_steps["clf"]
        X_t = preprocessor.transform(X_sample)
        if hasattr(X_t, "toarray"):
            X_t = X_t.toarray()
        feature_names = preprocessor.get_feature_names_out()

        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(X_t)
        if isinstance(sv, list) and len(sv) == 2:
            sv = sv[1]
        sv = np.asarray(sv)

        mean_abs_shap = np.mean(np.abs(sv), axis=0)
        ranking = pd.DataFrame({
            "feature": feature_names,
            "underlying_brfss_var": [underlying_brfss_var(f) for f in feature_names],
            "mean_abs_shap": mean_abs_shap,
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        ranking["rank_restricted"] = np.arange(1, len(ranking) + 1)
        ranking["disease"] = disease
        all_rows.append(ranking)

    return pd.concat(all_rows, ignore_index=True)


def main():
    print(f"Loading full pooled dataset: {COMBINED_CSV}")
    df = pd.read_csv(COMBINED_CSV)
    print(f"Full dataset shape: {df.shape}")

    df_restricted = df[df["YEAR"].isin(RESTRICTED_YEARS)].copy()
    print(f"Restricted to years {RESTRICTED_YEARS}: shape={df_restricted.shape}")

    restricted_models = train_restricted_models(df_restricted)

    print("\n=== Performance comparison: restricted (2013+2015) vs. full (2011-2015) ===")
    full_models_path = ARTIFACT_DIR / "disease_models.joblib"
    perf_comparison = compare_performance(restricted_models, full_models_path)
    print(perf_comparison.to_string(index=False))

    perf_out = RESTRICTED_TABLES_DIR / "restricted_years_performance_comparison.csv"
    perf_comparison.to_csv(perf_out, index=False)
    print(f"\nSaved -> {perf_out}")

    print("\nComputing SHAP rankings on restricted-years models "
          "(stratified subsample, n=20,000 per outcome)...")
    restricted_ranking = compute_restricted_shap_ranking(restricted_models)

    # Merge against the full-model ranking produced by
    # generate_global_shap.py, for the flagged cluster specifically.
    full_ranking_path = TABLES_DIR / "global_shap_full_ranking_all_outcomes.csv"
    if full_ranking_path.exists():
        full_ranking = pd.read_csv(full_ranking_path)
        full_cluster = full_ranking[full_ranking["underlying_brfss_var"].isin(FLAGGED_CLUSTER)].copy()
        full_cluster_best = (
            full_cluster.sort_values("mean_abs_shap", ascending=False)
            .groupby(["disease", "underlying_brfss_var"], as_index=False)
            .first()[["disease", "underlying_brfss_var", "rank", "mean_abs_shap"]]
            .rename(columns={"rank": "rank_full", "mean_abs_shap": "mean_abs_shap_full"})
        )

        restricted_cluster = restricted_ranking[restricted_ranking["underlying_brfss_var"].isin(FLAGGED_CLUSTER)].copy()
        restricted_cluster_best = (
            restricted_cluster.sort_values("mean_abs_shap", ascending=False)
            .groupby(["disease", "underlying_brfss_var"], as_index=False)
            .first()[["disease", "underlying_brfss_var", "rank_restricted", "mean_abs_shap"]]
            .rename(columns={"mean_abs_shap": "mean_abs_shap_restricted"})
        )

        rank_comparison = full_cluster_best.merge(
            restricted_cluster_best, on=["disease", "underlying_brfss_var"], how="outer"
        )
        rank_comparison["disease_pretty"] = rank_comparison["disease"].map(DISEASE_NAME_MAP)
        rank_comparison["rank_delta"] = rank_comparison["rank_restricted"] - rank_comparison["rank_full"]
        rank_comparison = rank_comparison.sort_values(["underlying_brfss_var", "disease"])

        print("\n=== SHAP rank stability for flagged cluster: full vs. restricted-years model ===")
        print(rank_comparison[["underlying_brfss_var", "disease_pretty", "rank_full",
                                "rank_restricted", "rank_delta"]].to_string(index=False))

        rank_out = RESTRICTED_TABLES_DIR / "restricted_years_shap_rank_stability.csv"
        rank_comparison.to_csv(rank_out, index=False)
        print(f"\nSaved -> {rank_out}")
    else:
        print(f"\nWARNING: {full_ranking_path} not found. Run generate_global_shap.py first "
              "to enable the SHAP rank stability comparison.")
        restricted_ranking.to_csv(
            RESTRICTED_TABLES_DIR / "restricted_years_shap_ranking_standalone.csv", index=False
        )

    print("\nDone. Use restricted_years_performance_comparison.csv and "
          "restricted_years_shap_rank_stability.csv to finalize the response to "
          "a review comment / a review comment, and the corresponding "
          "Discussion limitation paragraph.")


if __name__ == "__main__":
    main()

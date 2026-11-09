#!/usr/bin/env python
"""Calibration by subgroup: slope, intercept, O/E ratio, ECE (deciles). Run from repo root.
Subgroups match subgroup_fairness_analysis.py. Output: tables/fairness/subgroup_calibration.csv (S15 Dataset)."""
import os
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.linear_model import LogisticRegression

REPO = Path.cwd()
ART = Path(os.environ.get("ARTIFACT_DIR", REPO / "app" / "artifacts"))
OUT = REPO / "tables" / "fairness"; OUT.mkdir(parents=True, exist_ok=True)
NAMES = {"heart_attack": "Heart attack", "coronary_hd": "Coronary heart disease",
         "stroke": "Stroke", "kidney": "Chronic kidney disease",
         "depression": "Depression", "diabetes": "Diabetes"}

# Subgroup definitions are IDENTICAL to subgroup_fairness_analysis.py (same labels, same
# handling of missing values) so that S15 lines up row-for-row with S8 Dataset.
def _age(v):
    if pd.isna(v):
        return "Missing"
    v = int(v)
    return "18-44" if v <= 5 else ("45-64" if v <= 9 else "65+")


def _income(v):
    if pd.isna(v):
        return "Missing"
    v = int(v)
    return "Lower (<$25k)" if v <= 2 else ("Middle ($25-50k)" if v <= 4 else "Higher ($50k+)")


def _race(row):
    h, m = row["_HISPANC"], row["_MRACE1"]
    if pd.isna(h) and pd.isna(m):
        return "Missing race/ethnicity"
    if h == 1:
        return "Hispanic (any race)"
    if m == 1:
        return "White non-Hispanic"
    if m == 2:
        return "Black non-Hispanic"
    if m in (3, 4, 5, 6, 7, 8):
        return "Other/Multiracial non-Hispanic"
    return "Missing race/ethnicity"


def subgroups(X):
    return {"age_group": X["_AGEG5YR"].apply(_age),
            "income_tertile": X["_INCOMG"].apply(_income),
            "race_ethnicity": X.apply(_race, axis=1),
            "sex": X["SEX"].map({1: "Male", 2: "Female"}).fillna("Missing")}


def calib(y, p, n_bins=10):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e10, max_iter=500).fit(lp, y)
    q = pd.qcut(p, n_bins, duplicates="drop")
    g = pd.DataFrame({"y": y, "p": p, "q": q}).groupby("q", observed=True).agg(
        y=("y", "mean"), p=("p", "mean"), n=("y", "size"))
    ece = float((g["n"] * (g["y"] - g["p"]).abs()).sum() / g["n"].sum())
    return float(lr.coef_[0][0]), float(lr.intercept_[0]), float(y.mean() / p.mean()), ece

models = joblib.load(ART / "disease_models.joblib")
rows = []
for d, info in models.items():
    if "X_test" not in info:
        raise SystemExit(f"{d}: no 'X_test' in disease_models.joblib; send me how "
                         "subgroup_fairness_analysis.py gets test-set demographics.")
    X = info["X_test"].reset_index(drop=True)
    y = np.asarray(info["y_test"], dtype=int); p = np.asarray(info["y_proba"], dtype=float)
    for var, labels in subgroups(X).items():
        for sg in labels.dropna().unique():
            m = (labels == sg).to_numpy()
            n, ev = int(m.sum()), int(y[m].sum())
            reliable = n >= 100 and ev >= 10 and (n - ev) >= 10
            slope = icpt = oe = ece = np.nan
            if reliable:
                slope, icpt, oe, ece = calib(y[m], p[m])
            rows.append(dict(disease_pretty=NAMES.get(d, d), stratification_variable=var,
                             subgroup=str(sg), n=n, n_positive=ev, calibration_slope=slope,
                             calibration_intercept=icpt, obs_exp_ratio=oe, ECE=ece, reliable=reliable))
res = pd.DataFrame(rows).round(4)
res.to_csv(OUT / "subgroup_calibration.csv", index=False)
print(res.to_string(index=False))
print("\nSaved ->", OUT / "subgroup_calibration.csv")
print("Slope near 1, intercept near 0 and O/E near 1 indicate good calibration.")

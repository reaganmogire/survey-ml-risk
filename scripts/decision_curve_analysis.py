# coding: utf-8
"""
scripts/decision_curve_analysis.py

Computes decision-curve (net benefit) analysis for all six outcomes,
following Vickers & Elkin (2006), comparing the model against "treat
all" and "treat none" reference strategies across a range of threshold
probabilities. Addresses Reviewer 3 Comment 5 (clinical utility beyond
discrimination/calibration).

Uses the FINAL trained models' stored test-set predictions
(info["y_test"], info["y_proba"]) -- no retraining needed.

Net benefit formulas (standard decision-curve analysis):
    NB_model(pt)      = (TP/n) - (FP/n) * (pt / (1 - pt))
    NB_treat_all(pt)  = prevalence - (1 - prevalence) * (pt / (1 - pt))
    NB_treat_none(pt) = 0

where pt is the threshold probability (the probability above which a
"positive" classification is made), TP/FP are computed at that
threshold, and n is the test-set size. The threshold odds term
pt/(1-pt) represents the exchange rate between false positives and
true positives implied by a given threshold.

PATCH (this revision): the manuscript figure (fig_decision_curves.pdf)
is now saved as BOTH a .pdf and a .png via the new _save_both() helper
below, matching the convention used throughout
scripts/make_manuscript_outputs.py. Previously this script called
fig.savefig() directly and only ever produced the PDF, which is why
running make_manuscript_outputs.py (which only confirms the PDF/PNG
pair's presence for this figure rather than generating it) silently
never got a PNG for the decision-curve plot.

Run (from repo root):
  python scripts/decision_curve_analysis.py

Environment variables (optional):
  ARTIFACT_DIR    (default: <repo>/app/artifacts)
  TABLES_DIR      (default: <repo>/tables)
  FIGURES_DIR     (default: <repo>/figures)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)


# ============================================================
# Path configuration
# ============================================================
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1] if THIS_FILE.parent.name in {"src", "scripts"} else THIS_FILE.parent

ARTIFACT_DIR = Path(os.environ.get("ARTIFACT_DIR", REPO_ROOT / "app" / "artifacts"))
TABLES_DIR   = Path(os.environ.get("TABLES_DIR",   REPO_ROOT / "tables"))
FIGURES_DIR  = Path(os.environ.get("FIGURES_DIR",  REPO_ROOT / "figures"))
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DISEASE_NAME_MAP = {
    "heart_attack": "Heart attack",
    "coronary_hd":  "Coronary heart disease",
    "stroke":       "Stroke",
    "kidney":       "Chronic kidney disease",
    "depression":   "Depression",
    "diabetes":     "Diabetes",
}

# Threshold probability range for decision curves. Upper bound capped
# at 0.5 (standard practice -- net benefit analysis is most
# informative and most commonly reported over low-to-moderate
# thresholds, and pt/(1-pt) grows explosively as pt -> 1, which is
# rarely a clinically relevant operating range for population
# screening).
PT_RANGE = np.linspace(0.01, 0.50, 100)

# Standard grid layout for six-outcome faceted figures, matching
# make_manuscript_outputs.py for visual consistency across the paper.
FIG_GRID_ROWS = 3
FIG_GRID_COLS = 2


# ============================================================
# NEW: shared figure-saving helper -- saves BOTH a .pdf and a .png
# for every manuscript figure this script generates, mirroring
# _save_both() in scripts/make_manuscript_outputs.py. Kept as a local
# copy (rather than imported) so this script remains fully standalone.
# ============================================================
def _save_both(fig, pdf_path: Path, dpi: int = 300, bbox_inches: Optional[str] = None) -> Tuple[Path, Path]:
    """
    Saves `fig` as both a PDF (at `pdf_path`, exactly as given) and a
    PNG (same path with the extension swapped to .png). Returns both
    paths as a tuple.
    """
    pdf_path = Path(pdf_path)
    png_path = pdf_path.with_suffix(".png")
    save_kwargs = {"dpi": dpi}
    if bbox_inches is not None:
        save_kwargs["bbox_inches"] = bbox_inches
    fig.savefig(pdf_path, **save_kwargs)
    fig.savefig(png_path, **save_kwargs)
    return pdf_path, png_path


def compute_net_benefit_curve(y_true: np.ndarray, y_proba: np.ndarray, pt_range: np.ndarray) -> pd.DataFrame:
    n = len(y_true)
    prevalence = y_true.mean()
    rows = []

    for pt in pt_range:
        y_pred = (y_proba >= pt).astype(int)
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))

        odds = pt / (1 - pt)
        nb_model = (tp / n) - (fp / n) * odds
        nb_treat_all = prevalence - (1 - prevalence) * odds
        nb_treat_none = 0.0

        rows.append({
            "threshold_probability": round(float(pt), 4),
            "net_benefit_model": nb_model,
            "net_benefit_treat_all": nb_treat_all,
            "net_benefit_treat_none": nb_treat_none,
        })

    return pd.DataFrame(rows)


def find_positive_range(curve_df: pd.DataFrame) -> tuple:
    """
    Identifies the threshold-probability range over which the model's
    net benefit exceeds BOTH reference strategies (treat all, treat
    none). Returns (low, high) or (None, None) if the model never
    exceeds both.
    """
    exceeds_both = (
        (curve_df["net_benefit_model"] > curve_df["net_benefit_treat_all"]) &
        (curve_df["net_benefit_model"] > curve_df["net_benefit_treat_none"])
    )
    if not exceeds_both.any():
        return None, None
    thresholds_exceeding = curve_df.loc[exceeds_both, "threshold_probability"]
    return float(thresholds_exceeding.min()), float(thresholds_exceeding.max())


def main():
    disease_models_path = ARTIFACT_DIR / "disease_models.joblib"
    optimal_thresholds_path = ARTIFACT_DIR / "optimal_thresholds.joblib"
    disease_models = joblib.load(disease_models_path)
    optimal_thresholds = (
        joblib.load(optimal_thresholds_path) if optimal_thresholds_path.exists() else {}
    )

    all_curves = []
    summary_rows = []

    fig, axes = plt.subplots(FIG_GRID_ROWS, FIG_GRID_COLS, figsize=(11, 4.2 * FIG_GRID_ROWS))
    axes = axes.flatten()

    for ax, (disease, info) in zip(axes, disease_models.items()):
        y_test = np.asarray(info["y_test"], dtype=int)
        y_proba = np.asarray(info["y_proba"], dtype=float)
        f1_threshold = float(optimal_thresholds.get(disease, info.get("best_threshold", 0.5)))

        print(f"\n=== {DISEASE_NAME_MAP.get(disease, disease)} ===")
        curve_df = compute_net_benefit_curve(y_test, y_proba, PT_RANGE)
        curve_df["disease"] = disease
        curve_df["disease_pretty"] = DISEASE_NAME_MAP.get(disease, disease)
        all_curves.append(curve_df)

        pt_low, pt_high = find_positive_range(curve_df)
        # pt_low/pt_high are fractions (0-1) here, matching f1_threshold's
        # scale -- comparison and percentage conversion for display are
        # kept as two separate, clearly-labeled steps to avoid the
        # unit-mismatch bug from the previous version of this script.
        f1_within_range = bool(
            pt_low is not None and pt_low <= f1_threshold <= pt_high
        )

        range_str = f"{pt_low*100:.1f}%-{pt_high*100:.1f}%" if pt_low is not None else "none (model never exceeds both reference strategies)"
        print(f"  Threshold range where model exceeds both 'treat all' and 'treat none': {range_str}")
        print(f"  F1-optimized threshold used elsewhere in manuscript: {f1_threshold*100:.1f}% "
              f"({'within' if f1_within_range else 'OUTSIDE'} "
              f"the positive-net-benefit range)")

        summary_rows.append({
            "disease": disease,
            "disease_pretty": DISEASE_NAME_MAP.get(disease, disease),
            "prevalence_pct": round(float(y_test.mean() * 100), 2),
            "f1_optimized_threshold_pct": round(f1_threshold * 100, 2),
            "positive_net_benefit_range_low_pct": round(pt_low * 100, 2) if pt_low is not None else np.nan,
            "positive_net_benefit_range_high_pct": round(pt_high * 100, 2) if pt_high is not None else np.nan,
            "f1_threshold_within_positive_range": f1_within_range,
        })

        # Plot
        ax.plot(curve_df["threshold_probability"] * 100, curve_df["net_benefit_model"],
                label="Model", linewidth=2, color="steelblue")
        ax.plot(curve_df["threshold_probability"] * 100, curve_df["net_benefit_treat_all"],
                label="Treat all", linewidth=1.5, linestyle="--", color="firebrick")
        ax.plot(curve_df["threshold_probability"] * 100, curve_df["net_benefit_treat_none"],
                label="Treat none", linewidth=1.5, linestyle=":", color="gray")
        ax.axvline(f1_threshold * 100, color="black", alpha=0.4, linestyle="-.",
                   label=f"F1-optimized threshold ({f1_threshold*100:.1f}%)")
        ax.set_xlabel("Threshold probability (%)")
        ax.set_ylabel("Net benefit")
        ax.set_title(DISEASE_NAME_MAP.get(disease, disease))
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        # Clip y-axis to a sensible range -- net benefit can go
        # strongly negative for treat-all at high thresholds, which
        # would otherwise dominate the plot scale.
        ymax = curve_df["net_benefit_model"].max() * 1.3
        ax.set_ylim(-0.02, max(ymax, 0.05))

    plt.tight_layout()
    fig_path = FIGURES_DIR / "fig_decision_curves.pdf"
    # PATCHED: was `fig.savefig(fig_path, dpi=300)` -- now saves both
    # .pdf and .png via _save_both().
    pdf_saved, png_saved = _save_both(fig, fig_path, dpi=300)
    plt.close(fig)
    print(f"\nSaved figure -> {pdf_saved}")
    print(f"Saved figure -> {png_saved}")

    combined_curves = pd.concat(all_curves, ignore_index=True)
    curves_out = TABLES_DIR / "decision_curve_data.csv"
    combined_curves.to_csv(curves_out, index=False)
    print(f"Saved underlying curve data -> {curves_out}")

    summary_df = pd.DataFrame(summary_rows)
    print("\n=== Summary: threshold range of positive net benefit, by outcome ===")
    print(summary_df.to_string(index=False))

    summary_out = TABLES_DIR / "decision_curve_summary.csv"
    summary_df.to_csv(summary_out, index=False)
    print(f"\nSaved -> {summary_out}")

    print(
        "\nUse decision_curve_summary.csv for the Results sentence on threshold "
        "range of clinical utility, and fig_decision_curves.pdf + "
        "decision_curve_data.csv for the new supplementary figure (S[X] Fig)."
    )


if __name__ == "__main__":
    main()

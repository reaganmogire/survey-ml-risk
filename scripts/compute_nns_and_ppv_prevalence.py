#!/usr/bin/env python
# coding: utf-8
"""
scripts/compute_nns_and_ppv_prevalence.py

Computes two related clinical-utility analyses from the FINAL trained
models' existing test-set predictions (no retraining needed):

  B7 -- Number Needed to Screen (NNS = 1/PPV) at the F1-optimized
        threshold, for all six outcomes. Addresses a review comment.

  B8 -- Predicted PPV and NPV across a range of assumed population
        prevalences (0.5x, 1x, 2x the observed test-set prevalence),
        holding sensitivity and specificity fixed at their test-set
        values at the F1-optimized threshold. Addresses a review comment.

Both use the standard PPV/NPV-from-prevalence formulas:
    PPV(prev) = (Sens * prev) / (Sens * prev + (1 - Spec) * (1 - prev))
    NPV(prev) = (Spec * (1 - prev)) / (Spec * (1 - prev) + (1 - Sens) * prev)
which hold sensitivity and specificity constant (properties of the
classifier at a fixed threshold) while allowing the assumed prevalence
of the deployment population to vary -- this is the standard approach
for illustrating how a screening tool's real-world positive/negative
predictive value would shift in populations with different baseline
risk than the study sample.

PATCH 1 (this revision): the supplementary figure
(fig_ppv_vs_prevalence.pdf) is now saved as BOTH a .pdf and a .png via
the new _save_both() helper below, matching the convention used
throughout scripts/make_manuscript_outputs.py,
scripts/decision_curve_analysis.py, and scripts/plot_shap_violin.py.
Previously this script called fig.savefig() directly and only ever
produced the PDF (this is the figure referenced as S2 Fig in the
reconciled captions document).

PATCH 2 (this revision): the figure grid was changed from 2 rows x 3
columns to 3 rows x 2 columns, to match the standard six-outcome
faceted-figure layout used everywhere else in this manuscript (ROC
curves, calibration curves, decision curves, SHAP figures, model
comparison -- see FIG_GRID_ROWS/FIG_GRID_COLS in
make_manuscript_outputs.py). This was the only figure in the pipeline
still using the 2x3 orientation.

Run (from repo root):
  python scripts/compute_nns_and_ppv_prevalence.py

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

from sklearn.metrics import confusion_matrix

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

# Prevalence multipliers relative to the observed test-set prevalence.
PREVALENCE_MULTIPLIERS = [0.5, 1.0, 2.0]

# Standard grid layout for six-outcome faceted figures, matching
# make_manuscript_outputs.py / decision_curve_analysis.py for visual
# consistency across the paper. 3 rows x 2 columns, tall orientation.
FIG_GRID_ROWS = 3
FIG_GRID_COLS = 2


# ============================================================
# Shared figure-saving helper -- saves BOTH a .pdf and a .png for
# every manuscript figure this script generates, mirroring
# _save_both() in scripts/make_manuscript_outputs.py,
# scripts/decision_curve_analysis.py, and scripts/plot_shap_violin.py.
# Kept as a local copy (rather than imported) so this script remains
# fully standalone.
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


def load_artifacts():
    disease_models_path = ARTIFACT_DIR / "disease_models.joblib"
    if not disease_models_path.exists():
        raise FileNotFoundError(f"Missing required artifact: {disease_models_path}")
    disease_models = joblib.load(disease_models_path)

    opt_path = ARTIFACT_DIR / "optimal_thresholds.joblib"
    if opt_path.exists():
        optimal_thresholds = joblib.load(opt_path)
    else:
        optimal_thresholds = {
            d: float(info.get("best_threshold", 0.5)) for d, info in disease_models.items()
        }
    return disease_models, optimal_thresholds


def sens_spec_at_threshold(y_test: np.ndarray, y_proba: np.ndarray, threshold: float):
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    return sens, spec, ppv, npv


def ppv_from_prevalence(sens: float, spec: float, prevalence: float) -> float:
    numerator = sens * prevalence
    denominator = numerator + (1 - spec) * (1 - prevalence)
    return numerator / denominator if denominator > 0 else np.nan


def npv_from_prevalence(sens: float, spec: float, prevalence: float) -> float:
    numerator = spec * (1 - prevalence)
    denominator = numerator + (1 - sens) * prevalence
    return numerator / denominator if denominator > 0 else np.nan


def main():
    disease_models, optimal_thresholds = load_artifacts()

    # ------------------------------------------------------------
    # B7: NNS at the F1-optimized threshold
    # ------------------------------------------------------------
    nns_rows = []
    sens_spec_by_disease = {}

    for disease, info in disease_models.items():
        y_test = np.asarray(info["y_test"], dtype=int)
        y_proba = np.asarray(info["y_proba"], dtype=float)
        threshold = float(optimal_thresholds.get(disease, info.get("best_threshold", 0.5)))

        sens, spec, ppv, npv = sens_spec_at_threshold(y_test, y_proba, threshold)
        observed_prevalence = float(y_test.mean())
        nns = 1.0 / ppv if ppv and ppv > 0 else np.nan

        sens_spec_by_disease[disease] = {"sens": sens, "spec": spec, "prevalence": observed_prevalence}

        nns_rows.append({
            "disease": disease,
            "disease_pretty": DISEASE_NAME_MAP.get(disease, disease),
            "threshold": round(threshold, 3),
            "prevalence_pct": round(observed_prevalence * 100, 2),
            "sensitivity": round(sens, 4),
            "specificity": round(spec, 4),
            "PPV": round(ppv, 4),
            "NPV": round(npv, 4),
            "NNS": round(nns, 2),
        })

    nns_df = pd.DataFrame(nns_rows).sort_values("NNS")
    print("=== B7: Number Needed to Screen (NNS = 1/PPV), at F1-optimized threshold ===")
    print(nns_df.to_string(index=False))

    nns_out = TABLES_DIR / "nns_by_outcome.csv"
    nns_df.to_csv(nns_out, index=False)
    print(f"\nSaved -> {nns_out}")

    # ------------------------------------------------------------
    # B8: PPV / NPV vs. assumed prevalence (0.5x, 1x, 2x observed)
    # ------------------------------------------------------------
    ppv_prev_rows = []
    for disease, ss in sens_spec_by_disease.items():
        sens, spec, obs_prev = ss["sens"], ss["spec"], ss["prevalence"]
        for mult in PREVALENCE_MULTIPLIERS:
            assumed_prev = min(obs_prev * mult, 0.999)  # guard against prevalence >= 1
            ppv_at_prev = ppv_from_prevalence(sens, spec, assumed_prev)
            npv_at_prev = npv_from_prevalence(sens, spec, assumed_prev)
            ppv_prev_rows.append({
                "disease": disease,
                "disease_pretty": DISEASE_NAME_MAP.get(disease, disease),
                "prevalence_multiplier": mult,
                "assumed_prevalence_pct": round(assumed_prev * 100, 2),
                "PPV": round(ppv_at_prev, 4),
                "NPV": round(npv_at_prev, 4),
            })

    ppv_prev_df = pd.DataFrame(ppv_prev_rows)
    print("\n=== B8: PPV / NPV across assumed prevalence (0.5x / 1x / 2x observed) ===")
    print(ppv_prev_df.to_string(index=False))

    ppv_prev_out = TABLES_DIR / "ppv_npv_vs_prevalence.csv"
    ppv_prev_df.to_csv(ppv_prev_out, index=False)
    print(f"\nSaved -> {ppv_prev_out}")

    # ------------------------------------------------------------
    # B8 (continuous version): PPV vs. prevalence curve, for the
    # supplementary figure -- smooth range from near-zero to a
    # reasonable upper bound per outcome (3x observed prevalence).
    #
    # PATCH 2: grid is now FIG_GRID_ROWS x FIG_GRID_COLS (3 rows x 2
    # columns), matching every other six-outcome faceted figure in
    # this manuscript. Was previously plt.subplots(2, 3, figsize=(15, 8)).
    # ------------------------------------------------------------
    fig, axes = plt.subplots(FIG_GRID_ROWS, FIG_GRID_COLS, figsize=(11, 4.2 * FIG_GRID_ROWS))
    axes = axes.flatten()

    curve_rows = []
    for ax, (disease, ss) in zip(axes, sens_spec_by_disease.items()):
        sens, spec, obs_prev = ss["sens"], ss["spec"], ss["prevalence"]
        prev_range = np.linspace(0.001, min(obs_prev * 3, 0.95), 200)
        ppv_curve = [ppv_from_prevalence(sens, spec, p) for p in prev_range]

        for p, ppv_val in zip(prev_range, ppv_curve):
            curve_rows.append({
                "disease": disease, "prevalence": p, "PPV": ppv_val
            })

        ax.plot(prev_range * 100, ppv_curve, linewidth=2)
        ax.axvline(obs_prev * 100, linestyle="--", color="gray", alpha=0.7,
                   label=f"Observed prevalence ({obs_prev*100:.1f}%)")
        ax.set_xlabel("Assumed population prevalence (%)")
        ax.set_ylabel("Predicted PPV")
        ax.set_title(DISEASE_NAME_MAP.get(disease, disease))
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Turn off any unused axes (harmless no-op here since 6 outcomes
    # fill a 3x2 grid exactly, but kept for robustness if the number
    # of outcomes ever changes).
    for j in range(len(sens_spec_by_disease), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    fig_path = FIGURES_DIR / "fig_ppv_vs_prevalence.pdf"
    # PATCH 1: was `fig.savefig(fig_path, dpi=300)` -- now saves both
    # .pdf and .png via _save_both().
    pdf_saved, png_saved = _save_both(fig, fig_path, dpi=300)
    plt.close(fig)
    print(f"\nSaved figure -> {pdf_saved}")
    print(f"Saved figure -> {png_saved}")

    curve_df = pd.DataFrame(curve_rows)
    curve_out = TABLES_DIR / "ppv_vs_prevalence_curve_data.csv"
    curve_df.to_csv(curve_out, index=False)
    print(f"Saved underlying curve data -> {curve_out}")

    print(
        "\nUse nns_by_outcome.csv for Table 1 NNS column / Results sentence, "
        "\nppv_npv_vs_prevalence.csv for the discrete 0.5x/1x/2x table, and "
        "\nfig_ppv_vs_prevalence.pdf + fig_ppv_vs_prevalence.png + "
        "\nppv_vs_prevalence_curve_data.csv for the new supplementary figure (S2 Fig)."
    )


if __name__ == "__main__":
    main()

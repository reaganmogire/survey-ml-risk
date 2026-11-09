#!/usr/bin/env python
# coding: utf-8
"""
scripts/plot_shap_violin.py

Standalone plotting script for the combined global SHAP violin-style
plot. Uses Gaussian KDE to scale the vertical jitter/thickness
proportionally to point density, producing a violin-shaped silhouette
per feature row rather than a uniform-width scatter band.

Reads ONLY the saved CSVs produced by scripts/generate_global_shap.py
(tables/global_shap/shap_beeswarm_values_<disease>.csv and
shap_feature_values_<disease>.csv) -- does NOT recompute SHAP.

This module is used in TWO ways:
  1. Imported directly by scripts/make_manuscript_outputs.py, which
     calls plot_combined_violin(tables_dir=..., figures_dir=...) as
     part of its full pipeline run.
  2. Run standalone from the command line for fast iteration on plot
     styling (marker size, jitter, feature count, row height, etc.)
     without re-running the rest of that script or recomputing SHAP.

PATCH (this revision): the combined violin figure
(fig_global_shap_violin_combined.pdf) is now saved as BOTH a .pdf and
a .png via the new _save_both() helper below, matching the convention
used throughout scripts/make_manuscript_outputs.py. Previously this
script called fig.savefig() directly and only ever produced the PDF --
this was called out explicitly as a known gap in
make_manuscript_outputs.py's own docstring, since that orchestrator's
_save_both() only covers figures it generates directly, not this
module's save call.

Run standalone (from repo root):
  python scripts/plot_shap_violin.py
  python scripts/plot_shap_violin.py --top-n 8
  python scripts/plot_shap_violin.py --top-n 10 --max-thickness 0.38

Environment variables (optional, used only for standalone runs or
when tables_dir/figures_dir are not passed explicitly by a caller):
  TABLES_DIR      (default: <repo>/tables)
  FIGURES_DIR     (default: <repo>/figures)
"""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.stats import gaussian_kde

# ============================================================
# Path configuration + shared feature-labeling import
# ============================================================
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1] if THIS_FILE.parent.name in {"src", "scripts"} else THIS_FILE.parent

TABLES_DIR  = Path(os.environ.get("TABLES_DIR",  REPO_ROOT / "tables"))
FIGURES_DIR = Path(os.environ.get("FIGURES_DIR", REPO_ROOT / "figures"))
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

_FEATURE_LABELS_DIR = THIS_FILE.parent
if str(_FEATURE_LABELS_DIR) not in sys.path:
    sys.path.insert(0, str(_FEATURE_LABELS_DIR))

_feature_labels_module_path = _FEATURE_LABELS_DIR / "feature_labels.py"
if not _feature_labels_module_path.exists():
    raise FileNotFoundError(
        f"Expected scripts/feature_labels.py at {_feature_labels_module_path}, "
        f"but it was not found. Save feature_labels.py to {_FEATURE_LABELS_DIR} "
        f"before running this script."
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

FIG_GRID_ROWS = 3
FIG_GRID_COLS = 2


def pretty_disease_name(d: str) -> str:
    return DISEASE_NAME_MAP.get(d, d)


# ============================================================
# NEW: shared figure-saving helper -- saves BOTH a .pdf and a .png
# for every manuscript figure this script generates, mirroring
# _save_both() in scripts/make_manuscript_outputs.py. Kept as a local
# copy (rather than imported) so this script remains fully standalone
# when run directly via `python scripts/plot_shap_violin.py`.
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


def plot_combined_violin(
    top_n: int = 10,
    marker_size: float = 3.5,
    alpha: float = 0.50,
    max_thickness: float = 0.38,
    row_height: float = 0.34,
    out_name: str = "fig_global_shap_violin_combined.pdf",
    tables_dir: Optional[Path] = None,
    figures_dir: Optional[Path] = None,
) -> bool:
    """
    Generates the combined 3x2 global SHAP violin-style summary
    figure, one panel per outcome, with a single shared colorbar.

    Can be called two ways:
      - Directly with explicit tables_dir/figures_dir (e.g. from
        make_manuscript_outputs.py), in which case those paths are
        used regardless of this module's own TABLES_DIR/FIGURES_DIR
        defaults.
      - With no arguments (or via the standalone CLI in main()), in
        which case this module's own TABLES_DIR/FIGURES_DIR module-
        level constants are used.

    Parameters
    ----------
    top_n : int
        Number of top features (by mean |SHAP value|) to show per panel.
    marker_size : float
        Matplotlib scatter marker size.
    alpha : float
        Point transparency, 0-1.
    max_thickness : float
        Maximum half-width of the density-weighted jitter envelope.
    row_height : float
        Vertical inches allocated per feature row; controls how
        compact/tall each panel is.
    out_name : str
        Output filename (saved into figures_dir). The .png twin is
        derived automatically from this name via _save_both().
    tables_dir : Path, optional
        Directory containing the global_shap/ subfolder with the
        required input CSVs. Defaults to this module's TABLES_DIR.
    figures_dir : Path, optional
        Directory to save the output figure into. Defaults to this
        module's FIGURES_DIR.

    Returns
    -------
    bool
        True if the figure was generated and saved successfully (both
        .pdf and .png), False if required input files were missing (in
        which case a message is printed and no figure is produced).
    """
    tables_dir = tables_dir if tables_dir is not None else TABLES_DIR
    figures_dir = figures_dir if figures_dir is not None else FIGURES_DIR
    shap_dir = tables_dir / "global_shap"

    diseases = list(DISEASE_NAME_MAP.keys())
    missing = []
    for disease in diseases:
        if not (shap_dir / f"shap_beeswarm_values_{disease}.csv").exists():
            missing.append(f"shap_beeswarm_values_{disease}.csv")
        if not (shap_dir / f"shap_feature_values_{disease}.csv").exists():
            missing.append(f"shap_feature_values_{disease}.csv")

    if missing:
        print(f"[SKIP] Missing required files: {missing}. Run scripts/generate_global_shap.py first.")
        return False

    n_cols = FIG_GRID_COLS
    n_rows = FIG_GRID_ROWS

    panel_height = row_height * top_n + 1.0  # +1.0 for margins/title
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, panel_height * n_rows))
    axes = axes.flatten()

    cmap = plt.colormaps.get_cmap("coolwarm")

    for ax, disease in zip(axes, diseases):
        shap_df = pd.read_csv(shap_dir / f"shap_beeswarm_values_{disease}.csv")
        values_df = pd.read_csv(shap_dir / f"shap_feature_values_{disease}.csv")

        feature_cols = [c for c in shap_df.columns if c != "disease"]
        sv = shap_df[feature_cols].values
        fv = values_df[feature_cols].values

        mean_abs = np.mean(np.abs(sv), axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:top_n]
        pretty_labels = [pretty_feature_from_explainer(feature_cols[i]) for i in top_idx]

        n_points = sv.shape[0]

        for row_i, feat_idx in enumerate(top_idx):
            y_base = len(top_idx) - row_i
            shap_vals = sv[:, feat_idx]
            feat_vals = fv[:, feat_idx]

            # Normalize feature values to [0, 1] for colormap
            fmin, fmax = np.nanmin(feat_vals), np.nanmax(feat_vals)
            if fmax > fmin:
                norm_vals = (feat_vals - fmin) / (fmax - fmin)
            else:
                norm_vals = np.full_like(feat_vals, 0.5)

            # Calculate local density using Gaussian KDE, used to scale
            # the vertical jitter envelope -- this is what produces the
            # tapered violin-shaped silhouette (wide where SHAP values
            # cluster, narrow in sparse regions) rather than a uniform
            # rectangular scatter band.
            if np.all(shap_vals == shap_vals[0]):
                densities = np.ones_like(shap_vals)
            else:
                kde = gaussian_kde(shap_vals)
                densities = kde(shap_vals)
                densities = densities / np.max(densities)  # scale to [0, 1]

            rng = np.random.RandomState(RANDOM_STATE + row_i)
            jitter_envelope = densities * max_thickness
            jitter = rng.uniform(-jitter_envelope, jitter_envelope)

            colors = cmap(norm_vals)
            ax.scatter(
                shap_vals,
                np.full(n_points, y_base) + jitter,
                c=colors,
                s=marker_size,
                alpha=alpha,
                edgecolors="none",
                rasterized=True,   # points as a raster layer: keeps the PDF small (was ~19 MB), axes/text stay vector
            )

        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_yticks(range(1, len(top_idx) + 1))
        ax.set_yticklabels(list(reversed(pretty_labels)), fontsize=8)
        ax.set_xlabel("SHAP value (impact on predicted probability)", fontsize=8)
        ax.set_title(pretty_disease_name(disease), fontsize=11, fontweight="bold")
        ax.tick_params(axis="x", labelsize=7)
        ax.set_ylim(0.3, len(top_idx) + 0.7)

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])

    fig.subplots_adjust(right=0.90, hspace=0.45, wspace=0.55, top=0.94)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])
    cbar.set_label("Feature value (per-feature scaled)", fontsize=9)

    fig.suptitle(
        "Global SHAP summary distributions for six chronic disease case-finding models",
        fontsize=12, fontweight="bold"
    )

    out_path = figures_dir / out_name
    # PATCHED: was `fig.savefig(out_path, dpi=300, bbox_inches="tight")`
    # -- now saves both .pdf and .png via _save_both(). Note: out_name
    # is expected to end in .pdf; _save_both() derives the .png sibling
    # automatically regardless of what extension out_name carries, so
    # passing a custom --out-name via the CLI still produces a matching
    # PNG twin.
    pdf_saved, png_saved = _save_both(fig, out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {pdf_saved}")
    print(f"Saved -> {png_saved}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Plot SHAP density-scaled violin beeswarm plots.")
    parser.add_argument("--top-n", type=int, default=10,
                        help="Number of top features to show per panel (default: 10).")
    parser.add_argument("--marker-size", type=float, default=3.5,
                        help="Scatter point size (default: 3.5).")
    parser.add_argument("--alpha", type=float, default=0.50,
                        help="Point transparency, 0-1 (default: 0.50).")
    parser.add_argument("--max-thickness", type=float, default=0.38,
                        help="Maximum half-width envelope for violin density (default: 0.38).")
    parser.add_argument("--row-height", type=float, default=0.34,
                        help="Vertical inches per feature row (default: 0.34).")
    parser.add_argument("--out-name", type=str, default="fig_global_shap_violin_combined.pdf",
                        help="Output filename (default: fig_global_shap_violin_combined.pdf).")
    args = parser.parse_args()

    plot_combined_violin(
        top_n=args.top_n,
        marker_size=args.marker_size,
        alpha=args.alpha,
        max_thickness=args.max_thickness,
        row_height=args.row_height,
        out_name=args.out_name,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# coding: utf-8
"""
generate_flow_diagram_pdf_matplotlib.py

Generates the STROBE participant flow diagram using matplotlib only.
No cairo, no brew, no system libraries required.

Requirements (all standard in conda/miniconda):
    pip install matplotlib Pillow

Outputs (in data/derived/):
    flow_diagram.pdf          -- vector PDF, print-ready
    flow_diagram_600dpi.tiff  -- 600 dpi TIFF for journal submission
    flow_diagram.png          -- 300 dpi PNG preview

Usage:
    python generate_flow_diagram_pdf_matplotlib.py
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

# ── Output directory ──────────────────────────────────────────
THIS_FILE   = Path(__file__).resolve()
REPO_ROOT   = (
    THIS_FILE.parents[1]
    if THIS_FILE.parent.name in {"src", "scripts"}
    else THIS_FILE.parent
)
DERIVED_DIR = Path(os.environ.get("BRFSS_DERIVED_DIR", REPO_ROOT / "data" / "derived"))
DERIVED_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────
def box(ax, x, y, w, h, fc, ec, lw=1.2, radius=0.012):
    """Draw a rounded rectangle."""
    fancy = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw,
        transform=ax.transData, zorder=2,
    )
    ax.add_patch(fancy)


def arrow(ax, x1, y1, x2, y2, color="#555555"):
    """Draw a downward arrow."""
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>", color=color,
            lw=1.2, mutation_scale=10,
        ),
        zorder=3,
    )


def hline(ax, x1, x2, y, color="#555555"):
    ax.plot([x1, x2], [y, y], color=color, lw=1.2, zorder=3)


def vline(ax, x, y1, y2, color="#555555"):
    ax.plot([x, x], [y1, y2], color=color, lw=1.2, zorder=3)


def txt(ax, x, y, s, ha="center", va="center", fs=7, fw="normal",
        color="#2c3e50", style="normal"):
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs, fontweight=fw,
            color=color, fontstyle=style, zorder=4)


def outcome_box(ax, x, y, w, h, title, rows, ec="#5d6d7e"):
    """
    Draw one outcome box.
    rows = list of (label, value, color) tuples
    """
    box(ax, x, y, w, h, fc="#fdfefe", ec=ec, lw=1.0)
    # Title
    txt(ax, x + w / 2, y + h - 0.022, title, fs=7, fw="bold")
    # Divider
    ax.plot([x + 0.005, x + w - 0.005],
            [y + h - 0.036, y + h - 0.036],
            color="#d5d8dc", lw=0.7, zorder=3)
    # Rows
    row_h = (h - 0.044) / len(rows)
    for i, (label, val, col) in enumerate(rows):
        ry = y + h - 0.044 - (i + 0.35) * row_h
        txt(ax, x + 0.01, ry, label, ha="left", fs=6.2, color=col)
        txt(ax, x + w - 0.008, ry, val, ha="right", fs=6.2,
            fw="bold", color=col)


# ── Main ──────────────────────────────────────────────────────
def main():
    fig, ax = plt.subplots(figsize=(13, 10.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ── Title
    txt(ax, 0.5, 0.975, "Participant Flow — BRFSS 2011–2015",
        fs=9, fw="bold", color="#2c3e50")

    # ── BOX 1: Total records ──────────────────────────────────
    b1x, b1y, b1w, b1h = 0.30, 0.875, 0.40, 0.085
    box(ax, b1x, b1y, b1w, b1h, fc="#d6eaf8", ec="#2980b9", lw=1.5)
    txt(ax, 0.50, b1y + b1h - 0.022, "BRFSS respondents assessed",
        fs=7.5, fw="bold", color="#1a5276")
    txt(ax, 0.50, b1y + b1h - 0.042, "Annual surveys, 2011–2015",
        fs=7, color="#1a5276")
    txt(ax, 0.50, b1y + 0.018, "N = 2,380,047",
        fs=8.5, fw="bold", color="#2980b9")

    # stem down from box1 centre to junction
    b1_cx = b1x + b1w / 2
    junc_y = b1y - 0.045
    vline(ax, b1_cx, b1y, junc_y)

    # ── Exclusion branch ─────────────────────────────────────
    excl_x = 0.715
    hline(ax, b1_cx, excl_x, junc_y)
    arrow(ax, excl_x, junc_y, excl_x, junc_y - 0.001)

    ex_y = junc_y - 0.075
    box(ax, excl_x - 0.005, ex_y, 0.275, 0.065,
        fc="#fadbd8", ec="#c0392b", lw=1.4)
    txt(ax, excl_x + 0.132, ex_y + 0.045,
        "Excluded: all six outcomes missing",
        fs=7, color="#922b21")
    txt(ax, excl_x + 0.132, ex_y + 0.018,
        "n = 535", fs=7.5, fw="bold", color="#c0392b")

    # ── Arrow to Box 2 ────────────────────────────────────────
    b2_top = junc_y - 0.105
    arrow(ax, b1_cx, junc_y, b1_cx, b2_top + 0.001)

    # ── BOX 2: Analytic cohort ────────────────────────────────
    b2x, b2y, b2w, b2h = 0.295, b2_top - 0.088, 0.41, 0.088
    box(ax, b2x, b2y, b2w, b2h, fc="#d5f5e3", ec="#1e8449", lw=1.5)
    txt(ax, 0.50, b2y + b2h - 0.022, "Final analytic cohort",
        fs=7.5, fw="bold", color="#145a32")
    txt(ax, 0.50, b2y + b2h - 0.042, "\u2265 1 outcome non-missing",
        fs=7, color="#145a32")
    txt(ax, 0.50, b2y + 0.018, "N = 2,379,512",
        fs=8.5, fw="bold", color="#1e8449")

    # ── Bridge note ───────────────────────────────────────────
    bridge_y = b2y - 0.038
    arrow(ax, b1_cx, b2y, b1_cx, bridge_y + 0.001)
    box(ax, 0.09, bridge_y - 0.028, 0.82, 0.028,
        fc="#f8f9fa", ec="#aab7b8", lw=0.8)
    txt(ax, 0.50, bridge_y - 0.014,
        "Six disease-specific models, each fitted on respondents with "
        "non-missing outcome data for that condition",
        fs=6.5, color="#555555", style="italic")

    # ── Bus to outcome boxes ──────────────────────────────────
    bus_y  = bridge_y - 0.052
    lc_cx  = 0.235   # left column centre
    rc_cx  = 0.765   # right column centre

    vline(ax, b1_cx, bridge_y - 0.028, bus_y)
    hline(ax, lc_cx, rc_cx, bus_y)
    arrow(ax, lc_cx, bus_y, lc_cx, bus_y - 0.001)
    arrow(ax, rc_cx, bus_y, rc_cx, bus_y - 0.001)

    # ── Outcome box geometry ──────────────────────────────────
    ob_w  = 0.445
    ob_h  = 0.135
    ob_gap = 0.018
    lc_x  = lc_cx - ob_w / 2
    rc_x  = rc_cx - ob_w / 2

    # Row y positions (top edge of each box)
    row1_y = bus_y - ob_h - 0.012
    row2_y = row1_y - ob_h - ob_gap
    row3_y = row2_y - ob_h - ob_gap

    # Within-column connectors
    for cx in [lc_cx, rc_cx]:
        arrow(ax, cx, row1_y, cx, row1_y - 0.001)          # already handled by bus
        vline(ax, cx, row1_y, row1_y - ob_gap - 0.001)
        arrow(ax, cx, row1_y - ob_gap, cx, row2_y + ob_h - 0.001)
        vline(ax, cx, row2_y, row2_y - ob_gap - 0.001)
        arrow(ax, cx, row2_y - ob_gap, cx, row3_y + ob_h - 0.001)

    C_INC  = "#2c3e50"
    C_EXCL = "#c0392b"
    C_POS  = "#1e8449"
    C_NEG  = "#555555"

    # ── LEFT COLUMN ───────────────────────────────────────────
    outcome_box(ax, lc_x, row1_y, ob_w, ob_h,
                "Myocardial Infarction", [
                    ("Included (outcome non-missing):", "2,368,389", C_INC),
                    ("Excluded (missing / refused / DK):", "11,123",  C_EXCL),
                    ("Positive cases (MI reported):",   "141,437",   C_POS),
                    ("Negative cases:",                 "2,226,952", C_NEG),
                    ("Prevalence:",                     "5.97%",     "#7f8c8d"),
                ])

    outcome_box(ax, lc_x, row2_y, ob_w, ob_h,
                "Stroke", [
                    ("Included (outcome non-missing):", "2,373,243", C_INC),
                    ("Excluded (missing / refused / DK):", "6,269",  C_EXCL),
                    ("Positive cases (stroke reported):", "97,621",  C_POS),
                    ("Negative cases:",                 "2,275,622", C_NEG),
                    ("Prevalence:",                     "4.11%",     "#7f8c8d"),
                ])

    outcome_box(ax, lc_x, row3_y, ob_w, ob_h,
                "Depression", [
                    ("Included (outcome non-missing):", "2,367,914", C_INC),
                    ("Excluded (missing / refused / DK):", "11,598", C_EXCL),
                    ("Positive cases (depression reported):", "446,544", C_POS),
                    ("Negative cases:",                 "1,921,370", C_NEG),
                    ("Prevalence:",                     "18.86%",    "#7f8c8d"),
                ])

    # ── RIGHT COLUMN ──────────────────────────────────────────
    outcome_box(ax, rc_x, row1_y, ob_w, ob_h,
                "Coronary Heart Disease", [
                    ("Included (outcome non-missing):", "2,359,299", C_INC),
                    ("Excluded (missing / refused / DK):", "20,213", C_EXCL),
                    ("Positive cases (CHD reported):",  "142,475",   C_POS),
                    ("Negative cases:",                 "2,216,824", C_NEG),
                    ("Prevalence:",                     "6.04%",     "#7f8c8d"),
                ])

    outcome_box(ax, rc_x, row2_y, ob_w, ob_h,
                "Chronic Kidney Disease", [
                    ("Included (outcome non-missing):", "2,370,766", C_INC),
                    ("Excluded (missing / refused / DK):", "8,746",  C_EXCL),
                    ("Positive cases (CKD reported):",  "79,285",    C_POS),
                    ("Negative cases:",                 "2,291,481", C_NEG),
                    ("Prevalence:",                     "3.34%",     "#7f8c8d"),
                ])

    outcome_box(ax, rc_x, row3_y, ob_w, ob_h,
                "Diabetes", [
                    ("Included (outcome non-missing):", "2,374,814", C_INC),
                    ("Excluded (missing / refused / DK):", "4,698",  C_EXCL),
                    ("Positive cases (diabetes reported):", "302,943", C_POS),
                    ("Negative cases:",                 "2,071,871", C_NEG),
                    ("Prevalence:",                     "12.76%",    "#7f8c8d"),
                ])

    plt.tight_layout(pad=0.3)

    # ── Save outputs ──────────────────────────────────────────
    pdf_path  = DERIVED_DIR / "flow_diagram.pdf"
    tiff_path = DERIVED_DIR / "flow_diagram_600dpi.tiff"
    png_path  = DERIVED_DIR / "flow_diagram.png"

    fig.savefig(str(pdf_path),  format="pdf",  bbox_inches="tight",
                facecolor="white")
    print(f"PDF  saved to: {pdf_path}")

    fig.savefig(str(tiff_path), format="tiff", dpi=600,
                bbox_inches="tight", facecolor="white",
                pil_kwargs={"compression": "lzw"})
    print(f"TIFF saved to: {tiff_path}")

    fig.savefig(str(png_path),  format="png",  dpi=300,
                bbox_inches="tight", facecolor="white")
    print(f"PNG  saved to: {png_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()

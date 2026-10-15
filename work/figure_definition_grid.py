#!/usr/bin/env python3
"""Figure 1: estimated risk difference by outcome-construction rule.

Reads the definition grid produced by harmonized_definition_variants.py and draws
one row per rule, with the two databases side by side, so that the only thing
varying across a row is the outcome definition.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(os.environ.get("SEDATION_PROJECT_ROOT", Path(__file__).resolve().parents[1])).expanduser().resolve()
OUT = ROOT / "outputs" / "active_comparator_sedation"

EICU = "#0066AC"
MIMIC = "#D25100"

# displayed top to bottom, matching the order of Table 4
ROWS = [
    ("def_k1_any_hour", "One abnormal hour (k = 1)", "Duration of abnormality required"),
    ("def_k2_adjacent_primary", "Two adjacent hours (k = 2) †", "Duration of abnormality required"),
    ("def_k3_adjacent", "Three consecutive hours (k = 3)", "Duration of abnormality required"),
    ("def_hypot_map_only_k2", "MAP <65 mmHg only", "Hypotension component"),
    ("def_hypot_sbp_only_k2", "SBP <90 mmHg only", "Hypotension component"),
    ("def_source_any_k2", "Any available source", "Blood-pressure source"),
    ("def_source_nibp_priority_k2", "Non-invasive priority", "Blood-pressure source"),
    ("def_source_concordant_k2", "Concordant sources", "Blood-pressure source"),
]


def values(table, label, database):
    row = table.query("analysis == @label and database == @database").iloc[0]
    return row.risk_difference_percent, row.RD_CI95_low, row.RD_CI95_high


def main():
    table = pd.read_csv(OUT / "table22_definition_grid.csv")

    # build row positions with a gap between groups, then flip so the first row
    # of ROWS sits at the top of the panel
    raw, previous = [], None
    y = 0.0
    for _, _, group in ROWS:
        if previous is not None and group != previous:
            y += 0.95
        raw.append(y)
        y += 1.0
        previous = group
    span = max(raw)
    positions = [span - value for value in raw]
    group_top = {}
    for (_, _, group), pos in zip(ROWS, positions):
        group_top[group] = max(group_top.get(group, pos), pos)

    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    offsets = {"eICU-CRD": 0.18, "MIMIC-IV": -0.18}
    colours = {"eICU-CRD": EICU, "MIMIC-IV": MIMIC}

    for (label, _, _), y0 in zip(ROWS, positions):
        for database in ("eICU-CRD", "MIMIC-IV"):
            point, low, high = values(table, label, database)
            ax.errorbar(point, y0 + offsets[database], xerr=[[point - low], [high - point]],
                        fmt="o", color=colours[database], markersize=6,
                        elinewidth=1.6, capsize=3, capthick=1.2, zorder=3)

    ax.axvline(0, color="#777777", linestyle="--", linewidth=1.0, zorder=1)

    for text, y0, (label, _, _) in zip([r[1] for r in ROWS], positions, ROWS):
        ax.text(-9.6, y0, text, ha="left", va="center", fontsize=9.5,
                fontweight="bold" if label == "def_k2_adjacent_primary" else "normal")
    for group, top in group_top.items():
        ax.text(-9.6, top + 0.60, group, ha="left", va="center",
                fontsize=9.5, fontstyle="italic", color="#333333")

    ax.set_xlim(-9.6, 14.8)
    ax.set_ylim(min(positions) - 0.85, max(positions) + 1.65)
    ax.set_yticks([])
    ax.set_xlabel("Risk difference for repeated haemodynamic abnormality, percentage points", fontsize=10)
    ax.set_xticks(range(-8, 15, 4))
    ax.tick_params(axis="x", labelsize=9)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    ax.legend(handles=[
        Line2D([], [], marker="o", linestyle="none", color=EICU, label="eICU-CRD", markersize=6),
        Line2D([], [], marker="o", linestyle="none", color=MIMIC, label="MIMIC-IV", markersize=6),
    ], loc="lower right", bbox_to_anchor=(1.0, 1.005), ncol=2, frameon=False, fontsize=9.5)

    # No title inside the artwork: the journal requires figure captions to live in
    # the manuscript text file, not in the image.
    ax.set_title("")
    fig.text(0.012, 0.015,
             "Risk differences with 95% bootstrap intervals. "
             "† pre-specified rule (1,000 replicates); other rules, 300.",
             fontsize=8, color="#555555")

    fig.tight_layout(rect=(0, 0.045, 1, 1))
    png = OUT / "Fig1_definition_dependence.png"
    tif = OUT / "Fig1_definition_dependence.tif"
    fig.savefig(png, dpi=600)
    fig.savefig(tif, dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    print(f"wrote {png.name} ({png.stat().st_size // 1024} KB) and {tif.name} "
          f"({tif.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()

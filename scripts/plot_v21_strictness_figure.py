"""Create the exact-data publication figure for PRI-Audit v21."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "pri_audit_v21"
FIG = ROOT / "figures" / "paper_v21"

BLUE = "#0F4D92"
BLUE_LIGHT = "#77A9D5"
GRAY = "#6D747C"
GRAY_LIGHT = "#CFD6DC"
WARM = "#B64342"
WARM_LIGHT = "#F6CFCB"
GREEN = "#377E6F"
TEXT = "#1F2933"


def style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=8, colors=TEXT)
    axis.xaxis.label.set_color(TEXT)
    axis.yaxis.label.set_color(TEXT)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(OUT / "v21_loose_vs_strict_summary.csv")
    mismatch = pd.read_csv(OUT / "v21_mismatched_reference_sensitivity.csv")
    neighborhood = pd.read_csv(OUT / "v21_nested_neighborhood_strictness.csv")

    plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.titleweight": "bold"})
    figure = plt.figure(figsize=(11.2, 4.1), constrained_layout=True)
    grid = figure.add_gridspec(1, 3, width_ratios=[1.35, 0.95, 1.05])

    # A: Primary strict comparison uses the pre-specified matched expression reference.
    axis = figure.add_subplot(grid[0, 0])
    strict = summary[(summary.arm.str.startswith("strict_")) & (summary.arm != "strict_expression")].copy()
    labels, y_values = [], []
    for dataset in ["GSE151177", "GSE173706", "GSE206391", "GSE47751"]:
        for arm in ["strict_pca", "strict_nmf"]:
            item = strict[(strict.dataset == dataset) & (strict.arm == arm)]
            if item.empty:
                continue
            labels.append(f"{dataset.replace('GSE', '')}  {'PCA' if arm.endswith('pca') else 'NMF'}")
            y_values.append(item.iloc[0])
    y = np.arange(len(y_values))[::-1]
    for position, item in zip(y, y_values):
        delta = item.incremental_auroc_vs_strict_expression
        lo = item.increment_bootstrap_ci95_low
        hi = item.increment_bootstrap_ci95_high
        axis.errorbar(delta, position, xerr=[[delta - lo], [hi - delta]], fmt="o", color=BLUE, ecolor=BLUE_LIGHT, elinewidth=1.8, capsize=3, markersize=5, zorder=3)
    axis.axvline(0, color=GRAY, linewidth=1)
    axis.axvline(0.05, color=WARM, linewidth=1, linestyle="--")
    axis.text(0.052, len(y_values) - 1.25, "gate +0.05", color=WARM, fontsize=7, va="center")
    axis.set_yticks(y, labels)
    axis.set_xlim(-0.12, 0.18)
    axis.set_xlabel("Incremental AUROC vs matched strict expression (95% CI)")
    axis.set_title("A  Strict, matched primary comparison", loc="left", fontsize=10)
    axis.grid(axis="x", color="#E5E7EB", linewidth=.7)
    style_axis(axis)

    # B: one intentionally mismatched control illustrates manufactured gain.
    axis = figure.add_subplot(grid[0, 1])
    gse206 = mismatch[(mismatch.dataset == "GSE206391") & (mismatch.candidate_arm == "strict_pca")].iloc[0]
    primary = summary[(summary.dataset == "GSE206391") & (summary.arm == "strict_pca")].iloc[0]
    bar_labels = ["Matched\nexpression", "Strict\nPCA", "16-HVG\nexpression"]
    bar_values = [
        summary[(summary.dataset == "GSE206391") & (summary.arm == "strict_expression")].iloc[0].auroc,
        primary.auroc,
        gse206.mismatched_reference_auroc,
    ]
    colors = [GRAY_LIGHT, BLUE, WARM_LIGHT]
    edge = [GRAY, BLUE, WARM]
    axis.bar(range(3), bar_values, color=colors, edgecolor=edge, linewidth=1.1, width=.68)
    for index, value in enumerate(bar_values):
        axis.text(index, value + .007, f"{value:.3f}", ha="center", va="bottom", fontsize=8, color=TEXT)
    axis.annotate("true matched increment\n+0.002", xy=(1, bar_values[1]), xytext=(.4, .955), arrowprops={"arrowstyle": "-", "color": BLUE, "lw": 1}, color=BLUE, ha="center", fontsize=7)
    axis.annotate("apparent unfair increment\n+0.082", xy=(1.55, .87), color=WARM, ha="center", fontsize=7)
    axis.set_ylim(.78, 1.005)
    axis.set_xticks(range(3), bar_labels)
    axis.set_ylabel("Held-out patient AUROC")
    axis.set_title("B  Why matched references matter", loc="left", fontsize=10)
    axis.grid(axis="y", color="#E5E7EB", linewidth=.7)
    style_axis(axis)

    # C: fully nested cell-neighbourhood sensitivity at fixed recipes.
    axis = figure.add_subplot(grid[0, 2])
    neighborhood = neighborhood.sort_values("clusters")
    x = np.arange(len(neighborhood))
    width = .34
    axis.bar(x - width / 2, neighborhood.global_auroc, width, color=GRAY_LIGHT, edgecolor=GRAY, linewidth=1, label="Global dictionary")
    axis.bar(x + width / 2, neighborhood.fully_nested_strict_auroc, width, color=BLUE_LIGHT, edgecolor=BLUE, linewidth=1, label="Fully nested")
    for position, item in zip(x, neighborhood.itertuples(index=False)):
        axis.text(position, max(item.global_auroc, item.fully_nested_strict_auroc) + .012, f"gap {item.global_minus_fully_nested_optimism_gap:+.3f}", ha="center", va="bottom", fontsize=7, color=TEXT)
    axis.set_xticks(x, [f"K={value}" for value in neighborhood.clusters])
    axis.set_ylim(.78, 1.04)
    axis.set_ylabel("LOO sample AUROC")
    axis.set_title("C  Fully nested neighbourhood sensitivity", loc="left", fontsize=10)
    axis.grid(axis="y", color="#E5E7EB", linewidth=.7)
    legend = axis.legend(frameon=False, fontsize=7, loc="lower left")
    for handle in legend.legend_handles:
        handle.set_linewidth(1)
    style_axis(axis)

    figure.text(.01, -.01, "All intervals are independent-unit bootstrap intervals. Panel B's 16-HVG comparison is an explicitly non-primary, mismatched-reference sensitivity.", fontsize=7, color=TEXT)
    png = FIG / "fig11_v21_strictness_and_reference_audit.png"
    pdf = FIG / "fig11_v21_strictness_and_reference_audit.pdf"
    figure.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(png)


if __name__ == "__main__":
    main()

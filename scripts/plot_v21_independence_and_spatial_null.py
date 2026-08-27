"""Plot two pre-specified falsification diagnostics from exact v21 outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "pri_audit_v21"
FIG = ROOT / "figures" / "paper_v21"
BLUE, BLUE_LIGHT = "#0F4D92", "#77A9D5"
WARM, WARM_LIGHT = "#B64342", "#F6CFCB"
GRAY, GRAY_LIGHT, TEXT = "#6D747C", "#CFD6DC", "#1F2933"


def style(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=8, colors=TEXT)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    strict = pd.read_csv(OUT / "v21_loose_vs_strict_summary.csv")
    random_split = pd.read_csv(OUT / "v21_measurement_random_split_diagnostic.csv")
    with (OUT / "gse206391_v21_matched_spatial_coordinate_shuffle.json").open() as handle:
        coordinate = json.load(handle)
    arrays = np.load(OUT / "gse206391_v21_matched_spatial_coordinate_shuffle_distributions.npz")

    plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.titleweight": "bold"})
    figure = plt.figure(figsize=(10.8, 4.0), constrained_layout=True)
    grid = figure.add_gridspec(1, 2, width_ratios=[1.0, 1.3])

    # A: the row-level score is intentionally labelled as a different, invalid
    # estimand.  The source overlap is shown instead of disguising it as CV.
    axis = figure.add_subplot(grid[0, 0])
    cohorts = [("GSE151177", "strict_expression", "sample LOO"), ("GSE206391", "strict_expression", "patient LODO")]
    positions = np.arange(len(cohorts))
    primary, permissive, lo, hi = [], [], [], []
    for accession, arm, _ in cohorts:
        primary.append(strict[(strict.dataset == accession) & (strict.arm == arm)].iloc[0].auroc)
        values = random_split[random_split.dataset == accession].random_measurement_split_auroc.to_numpy()
        permissive.append(np.median(values)); lo.append(np.quantile(values, .025)); hi.append(np.quantile(values, .975))
    width = .33
    axis.bar(positions - width / 2, primary, width, color=BLUE_LIGHT, edgecolor=BLUE, linewidth=1, label="PRI-Audit: source-blocked")
    axis.bar(positions + width / 2, permissive, width, color=WARM_LIGHT, edgecolor=WARM, linewidth=1, label="Random cell/spot split*")
    axis.errorbar(positions + width / 2, permissive, yerr=[np.asarray(permissive) - np.asarray(lo), np.asarray(hi) - np.asarray(permissive)], fmt="none", color=WARM, capsize=2, lw=1.1)
    for position, score in zip(positions + width / 2, permissive):
        axis.text(position, score - .035, f"{score:.3f}", ha="center", va="top", color=WARM, fontsize=8)
    axis.set_xticks(positions, ["GSE151177\ncell", "GSE206391\nspot"])
    axis.set_ylim(.84, 1.012)
    axis.set_ylabel("AUROC")
    axis.set_title("A  Random measurements are not independent patients", loc="left", fontsize=10)
    axis.grid(axis="y", color="#E5E7EB", lw=.7)
    axis.legend(frameon=False, fontsize=7, loc="lower left")
    style(axis)
    axis.text(.5, -.31, "*Global preprocessing; 100% of test cells/spots come from source samples seen in training.\nIt estimates within-source interpolation, not patient generalisation.", transform=axis.transAxes, ha="center", va="top", fontsize=7, color=TEXT)

    # B: exact complete coordinate-shuffle distributions, with one plot per
    # dimension-matched representation.  A lower observed AUROC is not framed
    # as proof of absent spatial biology, only absent patient-level advantage.
    axis = figure.add_subplot(grid[0, 1])
    null_specs = [
        ("null_spatial", "spatial_graph_1000", "Spatial graph (1,000)", BLUE),
        ("null_combined", "expression_plus_spatial_1500", "Expression + spatial (1,500)", WARM),
    ]
    bins = np.linspace(.82, .92, 23)
    for key, result_key, label, color in null_specs:
        values = arrays[key]
        axis.hist(values, bins=bins, density=True, histtype="stepfilled", alpha=.22, color=color, label=f"{label}: coordinate null")
        observed = coordinate["observed"][result_key]
        pvalue = coordinate["null"][result_key]["p_ge_observed"]
        axis.axvline(observed, color=color, lw=2)
        axis.text(observed + (.0015 if color == BLUE else -.0015), 21 if color == BLUE else 17, f"observed {observed:.3f}\np={pvalue:.3f}", color=color, fontsize=7, ha="left" if color == BLUE else "right")
    axis.set_xlabel("Held-out patient AUROC after coordinate permutation")
    axis.set_ylabel("Density across 500 shuffles")
    axis.set_title("B  Coordinate-shuffle null: no spatial increment evidence", loc="left", fontsize=10)
    axis.legend(frameon=False, fontsize=7, loc="upper left")
    axis.grid(axis="y", color="#E5E7EB", lw=.7)
    style(axis)
    figure.text(.01, -.02, "All source-blocked scores use one held-out prediction per independent sample/patient. Coordinate shuffles are within sample; counts, labels, folds and fold-specific HVGs remain fixed.", fontsize=7, color=TEXT)

    png, pdf = FIG / "fig12_v21_independence_and_spatial_null.png", FIG / "fig12_v21_independence_and_spatial_null.pdf"
    figure.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(png)


if __name__ == "__main__":
    main()

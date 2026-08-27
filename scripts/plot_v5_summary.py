"""Create a paper-ready, fully data-derived v5 treatment summary figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "ccmac_v3"
FIGURES = ROOT / "figures" / "paper"
FIGURES.mkdir(parents=True, exist_ok=True)


def main() -> None:
    g850 = json.loads((RESULTS / "GSE85034_v5_metrics.json").read_text())
    g226 = json.loads((RESULTS / "GSE226244_v5_group_metrics.json").read_text())
    trajectory = g850["pooled_trajectory_vs_lesional_baseline"]
    labels = ["LS", "W1", "W2", "W4", "W16"]
    means = [0.0] + [row["mean"] for row in trajectory]
    lows = [0.0] + [row["ci_low"] for row in trajectory]
    highs = [0.0] + [row["ci_high"] for row in trajectory]
    errors = np.array([np.array(means) - np.array(lows), np.array(highs) - np.array(means)])
    primary = g850["week16_minus_lesional_baseline"]["by_treatment"]
    bar_labels = ["IL-17A\nGSE226244", "ADA\nGSE85034", "MTX\nGSE85034"]
    bar_means = [g226["on12_minus_baseline"]] + [row["mean"] for row in primary]
    bar_lows = [g226["bootstrap_ci"][0]] + [row["ci_low"] for row in primary]
    bar_highs = [g226["bootstrap_ci"][1]] + [row["ci_high"] for row in primary]
    bar_errors = np.array([np.array(bar_means) - np.array(bar_lows), np.array(bar_highs) - np.array(bar_means)])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].axhline(0, color="0.55", lw=1)
    axes[0].errorbar(range(len(labels)), means, yerr=errors, marker="o", color="#1967a3", capsize=4)
    axes[0].set_xticks(range(len(labels)), labels)
    axes[0].set_ylabel("Locked score change from LS")
    axes[0].set_title("GSE85034 longitudinal trajectory")
    axes[0].text(3, means[3] - .1, "95% CI excludes 0", ha="center", fontsize=9)
    axes[1].axhline(0, color="0.55", lw=1)
    axes[1].bar(range(3), bar_means, yerr=bar_errors, color=["#b24c63", "#1967a3", "#5e8c61"], capsize=4)
    axes[1].set_xticks(range(3), bar_labels)
    axes[1].set_ylabel("Post-treatment minus baseline score")
    axes[1].set_title("Independent treatment contexts")
    axes[1].text(0, bar_means[0] - .08, "sample-level", ha="center", fontsize=8)
    axes[1].text(1.5, min(bar_means[1:]) - .08, "patient-level", ha="center", fontsize=8)
    fig.savefig(FIGURES / "fig5_treatment_direction.png", dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    main()

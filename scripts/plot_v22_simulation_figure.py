"""Render the exact-data simulation figure for the PRI-Audit revision."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "pri_audit_v22"
FIG = ROOT / "figures" / "paper_v22"

BLUE = "#0F4D92"
BLUE_LIGHT = "#77A9D5"
GREEN = "#377E6F"
GREEN_LIGHT = "#AADCA9"
WARM = "#B64342"
WARM_LIGHT = "#F6CFCB"
GRAY = "#6D747C"
GRAY_LIGHT = "#CFD6DC"
TEXT = "#1F2933"


def binomial_interval(successes: int, trials: int) -> tuple[float, float]:
    """Exact Clopper-Pearson interval for a Monte-Carlo event rate."""
    low = 0.0 if successes == 0 else float(beta.ppf(0.025, successes, trials - successes + 1))
    high = 1.0 if successes == trials else float(beta.ppf(0.975, successes + 1, trials - successes))
    return low, high


def style(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=8, colors=TEXT)
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.7)


def proportion_panel(axis, raw: pd.DataFrame, scenario: str, title: str, y_label: str, color: str, reference: float | None = None) -> None:
    frame = raw[raw.scenario.eq(scenario)]
    counts = sorted(frame.n_patients.unique())
    rates, lo, hi = [], [], []
    for n in counts:
        subset = frame[frame.n_patients.eq(n)]
        passed = int(subset.strict_gate_pass.sum())
        lower, upper = binomial_interval(passed, len(subset))
        rates.append(passed / len(subset))
        lo.append(lower)
        hi.append(upper)
    x = np.arange(len(counts))
    axis.bar(x, rates, width=0.62, color=color, edgecolor=TEXT, linewidth=0.8)
    axis.errorbar(x, rates, yerr=[np.array(rates) - np.array(lo), np.array(hi) - np.array(rates)], fmt="none", color=TEXT, capsize=2, lw=1)
    for position, rate in zip(x, rates):
        axis.text(position, rate + .035, f"{rate:.2f}", ha="center", va="bottom", fontsize=8)
    if reference is not None:
        axis.axhline(reference, color=WARM, linewidth=1.2, linestyle="--")
        axis.text(len(counts) - .45, reference + .02, "0.05 gate", color=WARM, fontsize=7, ha="right")
    axis.set_xticks(x, [f"n={value}" for value in counts])
    axis.set_ylim(0, 1.12)
    axis.set_ylabel(y_label)
    axis.set_title(title, loc="left", fontsize=10, fontweight="bold")
    style(axis)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(OUT / "v22_simulation_replicates.csv")
    plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.titleweight": "bold"})
    figure = plt.figure(figsize=(10.7, 6.5), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, wspace=.24, hspace=.35)

    # A and B: the primary simulation quantities—empirical Type I error and
    # power of the complete strict evidence gate, not raw classifier AUROC.
    proportion_panel(
        figure.add_subplot(grid[0, 0]), raw, "null_increment",
        "A  Strict audit is calibrated when space adds no information",
        "Gate pass rate (empirical false-positive rate)", BLUE_LIGHT, reference=0.05,
    )
    proportion_panel(
        figure.add_subplot(grid[0, 1]), raw, "spatial_increment",
        "B  Strict audit detects a zero-mean coordinate signal",
        "Gate pass rate (empirical power)", GREEN_LIGHT,
    )

    # C: explicitly label the target mismatch.  A random cell split looks
    # excellent only because all held-out cells have already-seen sources.
    axis = figure.add_subplot(grid[1, 0])
    frame = raw[raw.scenario.eq("source_overlap")]
    counts = sorted(frame.n_patients.unique())
    x = np.arange(len(counts))
    strict = [frame[frame.n_patients.eq(n)].patient_loo_expression_auroc.median() for n in counts]
    random = [frame[frame.n_patients.eq(n)].random_measurement_split_auroc.median() for n in counts]
    width = .34
    axis.bar(x - width / 2, strict, width, color=BLUE_LIGHT, edgecolor=BLUE, linewidth=.9, label="Patient LOO")
    axis.bar(x + width / 2, random, width, color=WARM_LIGHT, edgecolor=WARM, linewidth=.9, label="Random cell split")
    for position, value in zip(x - width / 2, strict):
        axis.text(position, value + .025, f"{value:.2f}", ha="center", va="bottom", fontsize=7)
    for position, value in zip(x + width / 2, random):
        axis.text(position, value - .045, f"{value:.2f}", ha="center", va="top", fontsize=7, color=WARM)
    axis.set_xticks(x, [f"n={value}" for value in counts])
    axis.set_ylim(0.35, 1.08)
    axis.set_ylabel("Median AUROC")
    axis.set_title("C  Random cells interpolate already-seen sources", loc="left", fontsize=10, fontweight="bold")
    axis.legend(frameon=False, fontsize=7, loc="upper left")
    axis.text(.5, -.25, "All random-split test cells have a source sample represented in training (100% overlap).", transform=axis.transAxes, ha="center", fontsize=7, color=TEXT)
    style(axis)

    # D: candidate and matched expression have the same information here.  An
    # apparent increment only appears after deliberately weakening the control.
    axis = figure.add_subplot(grid[1, 1])
    frame = raw[raw.scenario.eq("mismatched_reference")]
    counts = sorted(frame.n_patients.unique())
    x = np.arange(len(counts))
    strict = [frame[frame.n_patients.eq(n)].strict_increment.mean() for n in counts]
    weak = [frame[frame.n_patients.eq(n)].weak_reference_increment.mean() for n in counts]
    axis.axhline(0, color=GRAY, linewidth=1)
    axis.axhline(.05, color=WARM, linewidth=1, linestyle="--")
    axis.plot(x, strict, marker="o", color=BLUE, lw=1.8, label="Matched expression reference")
    axis.plot(x, weak, marker="o", color=WARM, lw=1.8, label="Weakened expression reference")
    for position, value in zip(x, weak):
        axis.text(position, value + .025, f"+{value:.2f}", ha="center", fontsize=7, color=WARM)
    axis.set_xticks(x, [f"n={value}" for value in counts])
    axis.set_ylim(-.08, .68)
    axis.set_ylabel("Mean apparent ΔAUROC")
    axis.set_title("D  A weak comparator manufactures a gain", loc="left", fontsize=10, fontweight="bold")
    axis.legend(frameon=False, fontsize=7, loc="upper left")
    style(axis)

    figure.text(.01, -.02, "Each replicate contains paired patient samples and nested cells/spots. The strict gate requires ΔAUROC ≥ 0.05 and a patient-bootstrap lower 95% bound > 0. Error bars in A–B are exact 95% binomial intervals over Monte-Carlo replicates.", fontsize=7, color=TEXT)
    png = FIG / "fig6_v22_known_truth_simulation.png"
    pdf = FIG / "fig6_v22_known_truth_simulation.pdf"
    figure.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(png)


if __name__ == "__main__":
    main()

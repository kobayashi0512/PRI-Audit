"""Generate paper figures directly from frozen result files."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "ccmac_v3"
OUT = ROOT / "figures" / "paper"
OUT.mkdir(parents=True, exist_ok=True)


def development_failure() -> None:
    grid = pd.read_csv(RESULTS / "development_loco_grid.csv")
    summary = grid[["gamma", "lambda", "mean_auc"]].drop_duplicates()
    pivot = summary.pivot(index="gamma", columns="lambda", values="mean_auc").sort_index()
    stability = pd.read_csv(RESULTS / "development_coefficient_stability.csv")
    mat = stability.pivot(index="axis", columns="holdout", values="coefficient")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", ax=axes[0], cbar_kws={"label": "mean LOCO AUROC"})
    axes[0].set_title("Development selection: anchor penalty did not win")
    axes[0].set_xlabel("Ridge $\\lambda$")
    axes[0].set_ylabel("Anchor penalty $\\gamma$")
    sns.heatmap(mat, annot=True, fmt=".2f", center=0, cmap="coolwarm", ax=axes[1], cbar_kws={"label": "coefficient"})
    axes[1].set_title("LOCO coefficient stability")
    axes[1].set_xlabel("Held-out cohort")
    axes[1].set_ylabel("Frozen mechanism axis")
    fig.savefig(OUT / "fig3_development_algorithm_boundary.png", dpi=220)


def diagnostic_boundary() -> None:
    prediction = pd.read_csv(RESULTS / "GSE223799_locked_predictions.csv")
    names = ["ccmac_locked_ridge", "v0_nnls", "nos2_minus_ccl27"]
    labels = ["Locked\nmechanism ridge", "v0 NNLS", "NOS2−CCL27"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7), constrained_layout=True, sharex=True)
    for ax, name, label in zip(axes, names, labels):
        plot = prediction.copy()
        plot["disease"] = np.where(plot.label == 1, "PSO", "AD")
        sns.stripplot(data=plot, x="disease", y=name, hue="scale", dodge=True, size=6, ax=ax, palette={"Macro":"#1967a3", "Micro":"#b24c63"})
        ax.set_title(label)
        ax.set_xlabel("")
        ax.set_ylabel("score")
        ax.legend_.set_title("")
    fig.suptitle("GSE223799: all fixed scores perfectly separate PSO from AD", y=1.03)
    fig.savefig(OUT / "fig4_diagnostic_transport_boundary.png", dpi=220, bbox_inches="tight")


def normalization_controls() -> None:
    frame = pd.read_csv(RESULTS / "GSE85034_final_fixed_score_controls.csv")
    score_names = ["locked_ridge", "v0_nnls", "nos2_minus_ccl27"]
    labels = ["Locked mechanism ridge", "v0 NNLS", "NOS2−CCL27"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7), constrained_layout=True)
    for ax, score, label in zip(axes, score_names, labels):
        wide = frame.pivot(index=["subject", "treatment"], columns="time", values=score).dropna(subset=["NL", "LS", "WK16"])
        d_ls = np.abs(wide.LS - wide.NL)
        d_w16 = np.abs(wide.WK16 - wide.NL)
        for before, after in zip(d_ls, d_w16):
            ax.plot([0, 1], [before, after], color="0.75", lw=1)
        ax.scatter(np.zeros(len(d_ls)), d_ls, color="#b24c63", zorder=2, label="LS")
        ax.scatter(np.ones(len(d_w16)), d_w16, color="#1967a3", zorder=2, label="WK16")
        ax.set_xticks([0, 1], ["LS", "WK16"])
        ax.set_title(label)
        ax.set_ylabel("Absolute distance to own NL")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("GSE85034: all fixed scores move toward each patient's NL state", y=1.03)
    fig.savefig(OUT / "fig6_nonlesional_normalization_controls.png", dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    development_failure()
    diagnostic_boundary()
    normalization_controls()

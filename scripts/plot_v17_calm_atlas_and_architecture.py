"""Redesign the paper opening and method diagrams with a calm visual system.

Outputs
-------
1. A low-saturation, data-derived study landscape for all 11 datasets.
2. The complete PRI-Audit information-flow architecture.
3. A module-level input/function/output map.

The study landscape uses the same frozen public-data displays and portrait
exports as v16, but replaces the dense red heatmap wall with compact cohort
summaries, density contours, and paired tissue thumbnails.  Architecture
figures are conceptual vector diagrams and contain no invented measurements.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "figures/paper_v17/main"
ED = ROOT / "figures/paper_v17/extended_data"
CACHE = ROOT / "results/paper_v17"
OUT.mkdir(parents=True, exist_ok=True)
ED.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "scripts"))
from plot_v16_dataset_portraits import (  # noqa: E402
    PANEL,
    PROGRAMS,
    load_bulk,
    load_single_cell_maps,
    ordered_bulk_panel,
    perturbation_portrait,
    spatial_maps,
)


SEED = 20260812
RNG = np.random.default_rng(SEED)

# Calm, low-saturation semantic palette.
PAPER = "#FCFDFB"
WHITE = "#FFFFFF"
INK = "#26343C"
MUTED = "#66767E"
LINE = "#D8E0DF"
SOFT = "#F3F6F4"
BLUE = "#5F8193"
BLUE_LIGHT = "#DCE8EC"
SAGE = "#7E9D89"
SAGE_LIGHT = "#E2ECE5"
GOLD = "#B99A64"
GOLD_LIGHT = "#EFE7D6"
LAVENDER = "#817C9E"
LAVENDER_LIGHT = "#E7E4EF"
SLATE = "#56656D"
NL = "#759C91"
LS = "#8E788A"

PROGRAM_COLORS = {
    "Epidermal": GOLD,
    "IL-17 / innate": BLUE,
    "Interferon": LAVENDER,
}

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 6.5,
        "axes.titlesize": 7.5,
        "axes.labelsize": 6.3,
        "xtick.labelsize": 5.7,
        "ytick.labelsize": 5.7,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": LINE,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def save(fig: plt.Figure, name: str, root: Path = OUT) -> None:
    for extension in ("png", "pdf", "svg"):
        fig.savefig(root / f"{name}.{extension}", dpi=450, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)


def rounded_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    *,
    face: str = WHITE,
    edge: str = LINE,
    radius: float = 0.02,
    linewidth: float = 0.8,
    transform=None,
) -> FancyBboxPatch:
    if transform is None:
        transform = ax.transAxes
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        transform=transform,
        clip_on=False,
    )
    ax.add_patch(patch)
    return patch


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = SLATE,
    width: float = 0.9,
    mutation: float = 8,
    style: str = "-|>",
    connectionstyle: str = "arc3",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            transform=ax.transAxes,
            arrowstyle=style,
            mutation_scale=mutation,
            color=color,
            linewidth=width,
            connectionstyle=connectionstyle,
            clip_on=False,
        )
    )


def title_block(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.text(0.055, 0.965, title, ha="left", va="top", fontsize=12.0, fontweight="bold", color=INK)
    fig.text(0.055, 0.927, subtitle, ha="left", va="top", fontsize=7.0, color=MUTED)


def clean_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(length=0)


def program_summaries(bulk: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort, indices in metadata.groupby("cohort").groups.items():
        patient = bulk.iloc[list(indices)][PANEL].to_numpy(float) / bulk.shape[1]
        for program, genes in PROGRAMS.items():
            positions = [PANEL.index(gene) for gene in genes]
            scores = patient[:, positions].mean(axis=1)
            rows.append(
                {
                    "cohort": cohort,
                    "program": program,
                    "n": len(scores),
                    "median": float(np.median(scores)),
                    "q25": float(np.quantile(scores, 0.25)),
                    "q75": float(np.quantile(scores, 0.75)),
                    "mean": float(np.mean(scores)),
                }
            )
    return pd.DataFrame(rows)


def add_bulk_landscape(ax: plt.Axes, summary: pd.DataFrame) -> None:
    ax.set_facecolor(WHITE)
    cohorts = [
        "GSE117239", "GSE136757", "GSE153007", "GSE182740",
        "GSE51440", "GSE57225", "GSE85034",
    ]
    y = np.arange(len(cohorts))[::-1]
    offsets = {"Epidermal": 0.18, "IL-17 / innate": 0.0, "Interferon": -0.18}
    for program in PROGRAMS:
        q = summary[summary.program == program].set_index("cohort").loc[cohorts]
        yy = y + offsets[program]
        ax.hlines(yy, q.q25, q.q75, color=PROGRAM_COLORS[program], lw=2.2, alpha=0.75)
        ax.scatter(q["median"], yy, s=17, color=PROGRAM_COLORS[program], edgecolor=WHITE,
                   linewidth=0.5, zorder=3, label=program)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    for yi, cohort in zip(y, cohorts):
        ax.text(
            0.012,
            yi,
            cohort,
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=7.2,
            color=MUTED,
            zorder=8,
        )
    ax.axvline(0, color=LINE, lw=0.8)
    ax.set_xlim(-0.08, 0.58)
    ax.set_xlabel("patient-level LS−NL program shift (median and IQR)")
    ax.set_title("Seven paired bulk cohorts · 191 patient pairs", loc="left", pad=8,
                 fontsize=8.1, fontweight="bold")
    ax.grid(axis="x", color="#EDF1F0", lw=0.6)
    ax.set_axisbelow(True)
    clean_axis(ax)
    ax.legend(frameon=False, loc="lower right", ncol=3, columnspacing=0.8,
              handletextpad=0.25, fontsize=5.2)


def density_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    dataset: str,
    condition_order: list[str],
    colors: list[str],
    subtitle: str,
) -> None:
    display = frame
    if len(display) > 7000:
        display = display.iloc[np.sort(RNG.choice(len(display), 7000, replace=False))]
    for condition, color in zip(condition_order, colors):
        q = display[display.condition == condition]
        ax.scatter(q.UMAP1, q.UMAP2, s=0.65, alpha=0.20, color=color,
                   linewidths=0, rasterized=True)
    xlim = np.quantile(display.UMAP1, [0.005, 0.995])
    ylim = np.quantile(display.UMAP2, [0.005, 0.995])
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xticks([]); ax.set_yticks([])
    clean_axis(ax)
    ax.set_title(dataset, loc="left", fontsize=7.0, fontweight="bold", pad=4)
    ax.text(0.0, 0.015, subtitle, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=5.0, color=MUTED)
    handles = [
        mpl.lines.Line2D([0], [0], marker="o", color="none", markerfacecolor=color,
                         markeredgecolor="none", markersize=3.2, label=label)
        for label, color in zip(condition_order, colors)
    ]
    ax.legend(handles=handles, frameon=False, fontsize=4.5, loc="upper left",
              borderpad=0.1, handletextpad=0.2, labelspacing=0.15)


def add_spatial_pair(ax: plt.Axes, maps: list[dict]) -> None:
    all_values = np.concatenate([item["score"] for item in maps])
    vmin, vmax = np.quantile(all_values, [0.03, 0.97])
    cmap = mpl.colors.LinearSegmentedColormap.from_list("spatial_calm", ["#EEF2F0", "#7F7897"])
    for item, offset in zip(maps, [-0.52, 0.52]):
        x = item["col"].astype(float); y = -item["row"].astype(float)
        x = (x - np.median(x)) / max(np.ptp(x), 1) * 0.75 + offset
        y = (y - np.median(y)) / max(np.ptp(y), 1) * 0.70
        ax.scatter(x, y, c=item["score"], cmap=cmap, vmin=vmin, vmax=vmax,
                   s=0.8, linewidths=0, rasterized=True)
        label = "NL" if "NON" in item["condition"].upper() else "LS"
        ax.text(offset, -0.48, label, ha="center", va="top", fontsize=5.4,
                fontweight="bold", color=NL if label == "NL" else LS)
    ax.set_xlim(-1.0, 1.0); ax.set_ylim(-0.60, 0.55)
    ax.set_xticks([]); ax.set_yticks([])
    clean_axis(ax)
    ax.set_title("GSE206391", loc="left", fontsize=7.0,
                 fontweight="bold", pad=4)
    ax.text(0.0, 0.015, "paired tissue · observed spots", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=4.8, color=MUTED)


def add_perturbation(ax: plt.Axes, donor: pd.DataFrame) -> None:
    gene_to_program = {gene: program for program, genes in PROGRAMS.items() for gene in genes}
    q = donor.assign(program=donor.gene.map(gene_to_program))
    q = q.groupby(["condition", "program"], as_index=False).rank_shift.mean()
    conditions = ["IL-17", "TNF", "combination", "synergy residual"]
    x = np.arange(len(conditions))
    for program in PROGRAMS:
        values = q[q.program == program].set_index("condition").loc[conditions].rank_shift
        ax.plot(x, values, "o-", color=PROGRAM_COLORS[program], lw=1.1, ms=2.8, label=program)
    ax.axhline(0, color=LINE, lw=0.8)
    ax.set_xticks(x, ["IL-17", "TNF", "combo", "synergy"], rotation=24, ha="right")
    ax.set_ylabel("mean paired rank shift")
    ax.set_title("GSE24767", loc="left", fontsize=7.0,
                 fontweight="bold", pad=4)
    ax.text(0.0, 0.015, "keratinocyte cytokine response", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=4.8, color=MUTED)
    ax.grid(axis="y", color="#EDF1F0", lw=0.6)
    clean_axis(ax)
    ax.legend(frameon=False, fontsize=4.5, loc="upper left", handletextpad=0.2)


def plot_study_landscape() -> dict:
    bulk, metadata = load_bulk()
    source, target = load_single_cell_maps()
    maps = spatial_maps()
    _, donor = perturbation_portrait()
    summary = program_summaries(bulk, metadata)

    fig = plt.figure(figsize=(7.2, 6.0), facecolor=PAPER)
    title_block(
        fig,
        "Public-data landscape for a patient-level psoriasis audit",
        "Seven paired bulk cohorts plus single-cell, spatial and perturbation views",
    )
    outer = fig.add_gridspec(
        2, 2, left=0.115, right=0.970, bottom=0.080, top=0.845,
        width_ratios=[1.30, 1.0], height_ratios=[1.08, 0.92], wspace=0.20, hspace=0.25,
    )
    bulk_ax = fig.add_subplot(outer[:, 0])
    rounded_box(bulk_ax, (-0.055, -0.075), 1.11, 1.13, face=WHITE, edge=LINE, radius=0.018)
    add_bulk_landscape(bulk_ax, summary)

    cell_grid = outer[0, 1].subgridspec(1, 2, wspace=0.14)
    ax = fig.add_subplot(cell_grid[0, 0])
    rounded_box(ax, (-0.045, -0.075), 1.09, 1.13, face=WHITE, edge=LINE, radius=0.018)
    density_panel(ax, source, "GSE151177", ["Control", "Psoriasis"], [SAGE, LAVENDER],
                  "19,868 cells · development")
    ax = fig.add_subplot(cell_grid[0, 1])
    rounded_box(ax, (-0.045, -0.075), 1.09, 1.13, face=WHITE, edge=LINE, radius=0.018)
    density_panel(
        ax, target, "GSE173706", ["Normal skin", "Peripheral normal", "Psoriatic plaque"],
        [SAGE, BLUE, LAVENDER], "14,493 cells · external",
    )

    lower = outer[1, 1].subgridspec(1, 2, wspace=0.28)
    ax = fig.add_subplot(lower[0, 0])
    rounded_box(ax, (-0.055, -0.080), 1.11, 1.15, face=WHITE, edge=LINE, radius=0.018)
    add_spatial_pair(ax, maps)
    ax = fig.add_subplot(lower[0, 1])
    rounded_box(ax, (-0.055, -0.080), 1.11, 1.15, face=WHITE, edge=LINE, radius=0.018)
    add_perturbation(ax, donor)

    fig.text(0.055, 0.025, "Bulk: 7 cohorts  |  Single-cell: 2 cohorts  |  Spatial: 1 cohort  |  Perturbation: 1 cohort",
             ha="left", fontsize=5.4, color=MUTED)
    save(fig, "fig1_calm_public_data_landscape")
    summary.to_csv(CACHE / "calm_atlas_bulk_program_summary.csv", index=False)
    return {
        "bulk_pairs": int(len(metadata)),
        "source_display_cells": int(len(source)),
        "target_display_cells": int(len(target)),
        "spatial_sections": [item["sample"] for item in maps],
        "perturbation_donors": sorted(donor.donor.unique().tolist()),
    }


def patient_program_values(bulk: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort, indices in metadata.groupby("cohort").groups.items():
        values = bulk.iloc[list(indices)][PANEL].to_numpy(float) / bulk.shape[1]
        subjects = metadata.iloc[list(indices)].subject.astype(str).to_numpy()
        for patient_index, subject in enumerate(subjects):
            for program, genes in PROGRAMS.items():
                positions = [PANEL.index(gene) for gene in genes]
                rows.append({
                    "cohort": cohort,
                    "subject": subject,
                    "program": program,
                    "score": float(values[patient_index, positions].mean()),
                })
    return pd.DataFrame(rows)


def plot_bulk_extended_calm(bulk: pd.DataFrame, metadata: pd.DataFrame) -> None:
    patient = patient_program_values(bulk, metadata)
    cohorts = [
        "GSE117239", "GSE136757", "GSE153007", "GSE182740",
        "GSE51440", "GSE57225", "GSE85034",
    ]
    fig = plt.figure(figsize=(7.2, 7.4), facecolor=PAPER)
    title_block(
        fig,
        "Patient-level lesion-program distributions across seven bulk cohorts",
        "Each dot is one paired patient; thick lines denote cohort medians",
    )
    grid = fig.add_gridspec(4, 2, left=0.09, right=0.97, bottom=0.07, top=0.87,
                            wspace=0.28, hspace=0.45)
    for index, cohort in enumerate(cohorts):
        ax = fig.add_subplot(grid[index // 2, index % 2])
        q = patient[patient.cohort == cohort]
        for program_index, program in enumerate(PROGRAMS):
            values = q[q.program == program].score.to_numpy()
            jitter = RNG.normal(0, 0.045, len(values))
            ax.scatter(values, np.full(len(values), program_index) + jitter, s=8,
                       color=PROGRAM_COLORS[program], alpha=0.34, linewidths=0, rasterized=True)
            median = float(np.median(values))
            q25, q75 = np.quantile(values, [0.25, 0.75])
            ax.hlines(program_index, q25, q75, color=PROGRAM_COLORS[program], lw=2.2)
            ax.scatter(median, program_index, s=22, color=PROGRAM_COLORS[program],
                       edgecolor=WHITE, linewidth=0.6, zorder=3)
        ax.axvline(0, color=LINE, lw=0.8)
        ax.set_yticks(range(3), list(PROGRAMS))
        ax.set_xlim(-0.55, 0.78)
        ax.set_title(f"{cohort}  ·  n={q.subject.nunique()} pairs", loc="left",
                     fontsize=7.1, fontweight="bold")
        ax.grid(axis="x", color="#EDF1F0", lw=0.6)
        clean_axis(ax)
        if index // 2 == 3:
            ax.set_xlabel("paired LS−NL program shift")
    note = fig.add_subplot(grid[3, 1])
    note.axis("off")
    rounded_box(note, (0.03, 0.12), 0.94, 0.72, face=SOFT, edge=LINE)
    note.text(0.08, 0.72, "Display rule", transform=note.transAxes, fontsize=7.0,
              fontweight="bold", color=INK)
    note.text(
        0.08, 0.60,
        "The same fixed 18 genes define three six-gene programs in every cohort.\n"
        "All paired patients are displayed. No cohort or patient is selected by effect size.\n"
        "Program plots are descriptive and remain outside model fitting and claim selection.",
        transform=note.transAxes, fontsize=5.5, color=MUTED, va="top", linespacing=1.45,
    )
    save(fig, "figED13_bulk_patient_program_distributions", ED)
    patient.to_csv(CACHE / "bulk_patient_program_values.csv", index=False)


def plot_perturbation_extended_calm(donor: pd.DataFrame) -> None:
    gene_to_program = {gene: program for program, genes in PROGRAMS.items() for gene in genes}
    q = donor.assign(program=donor.gene.map(gene_to_program))
    q = q.groupby(["donor", "condition", "program"], as_index=False).rank_shift.mean()
    conditions = ["IL-17", "TNF", "combination", "synergy residual"]
    labels = ["IL-17", "TNF", "combo", "synergy"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), facecolor=PAPER)
    title_block(
        fig,
        "GSE24767 donor-resolved cytokine responses",
        "Three complete keratinocyte donors · fixed epidermal, IL-17/innate and interferon programs",
    )
    for panel_index, donor_id in enumerate(sorted(q.donor.unique())):
        ax = axes.flat[panel_index]
        d = q[q.donor == donor_id]
        for program in PROGRAMS:
            values = d[d.program == program].set_index("condition").loc[conditions].rank_shift
            ax.plot(range(4), values, "o-", color=PROGRAM_COLORS[program], lw=1.2, ms=3.2,
                    label=program)
        ax.axhline(0, color=LINE, lw=0.8)
        ax.set_xticks(range(4), labels, rotation=20, ha="right")
        ax.set_title(f"Donor {donor_id}", loc="left", fontsize=7.3, fontweight="bold")
        ax.set_ylabel("paired rank shift")
        ax.grid(axis="y", color="#EDF1F0", lw=0.6)
        clean_axis(ax)
    ax = axes.flat[3]
    for program in PROGRAMS:
        d = q[q.program == program].groupby("condition").rank_shift.agg(["mean", "std"]).loc[conditions]
        ax.errorbar(range(4), d["mean"], yerr=d["std"], fmt="o-", color=PROGRAM_COLORS[program],
                    lw=1.2, ms=3.2, capsize=2, label=program)
    ax.axhline(0, color=LINE, lw=0.8)
    ax.set_xticks(range(4), labels, rotation=20, ha="right")
    ax.set_title("Across-donor mean ± SD", loc="left", fontsize=7.3, fontweight="bold")
    ax.set_ylabel("paired rank shift")
    ax.grid(axis="y", color="#EDF1F0", lw=0.6)
    clean_axis(ax)
    ax.legend(frameon=False, fontsize=5.1, loc="upper left")
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.10, top=0.84, wspace=0.32, hspace=0.40)
    save(fig, "figED14_calm_cytokine_donor_responses", ED)


def stage_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    number: str,
    title: str,
    lines: list[str],
    color: str,
    light: str,
) -> None:
    rounded_box(ax, (x, y), width, height, face=WHITE, edge=LINE, radius=0.016)
    ax.add_patch(Rectangle((x, y), 0.016, height, transform=ax.transAxes,
                           facecolor=color, edgecolor="none", clip_on=False))
    ax.text(x + 0.030, y + height - 0.052, number, transform=ax.transAxes,
            ha="left", va="center", fontsize=5.1, fontweight="bold", color=color)
    ax.text(x + 0.030, y + height - 0.100, title, transform=ax.transAxes,
            ha="left", va="center", fontsize=5.5, fontweight="bold", color=INK,
            linespacing=1.05)
    for index, line in enumerate(lines):
        ax.text(x + 0.030, y + height - 0.158 - index * 0.052, line, transform=ax.transAxes,
                ha="left", va="center", fontsize=5.0, color=MUTED)
    ax.add_patch(Rectangle((x + width - 0.065, y + height - 0.070), 0.038, 0.038,
                           transform=ax.transAxes, facecolor=light, edgecolor=color,
                           linewidth=0.6, clip_on=False))


def plot_overall_architecture() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.85), facecolor=PAPER)
    ax.axis("off")
    title_block(
        fig,
        "PRI-Audit algorithm and model architecture",
        "One information boundary, two matched prediction paths and a falsification-gated claim",
    )

    def box(x, y, width, height, title, lines, color, fill=WHITE, align="left"):
        rounded_box(ax, (x, y), width, height, face=fill, edge=color, radius=0.016)
        tx = x + 0.020 if align == "left" else x + width / 2
        ha = "left" if align == "left" else "center"
        title_size = 5.2 if width < 0.16 else 6.2
        ax.text(tx, y + height - 0.043, title, transform=ax.transAxes, ha=ha,
                va="center", fontsize=title_size, fontweight="bold", color=INK)
        for index, line in enumerate(lines):
            ax.text(tx, y + height - 0.088 - index * 0.034, line, transform=ax.transAxes,
                    ha=ha, va="center", fontsize=4.7, color=MUTED)

    ax.text(0.040, 0.842, "TRAINING GROUPS ONLY", transform=ax.transAxes,
            fontsize=5.3, fontweight="bold", color=BLUE)
    box(0.040, 0.625, 0.145, 0.165, "Nested data",
        ["patient / donor", "sample / biopsy", "cells / spots"], BLUE)
    box(0.225, 0.625, 0.145, 0.165, "M1 · Group split",
        ["leave one group out", "lock fold registry"], BLUE, SOFT)
    box(0.410, 0.625, 0.245, 0.165, "M2–M3 · Fold-local pipeline",
        ["HVGs · scaling · representation", "candidate + matched reference"], LAVENDER)
    box(0.695, 0.625, 0.145, 0.165, "Fit + freeze",
        ["classifier", "training objects"], BLUE, SOFT)
    arrow(ax, (0.185, 0.705), (0.222, 0.705), color=SLATE)
    arrow(ax, (0.370, 0.705), (0.407, 0.705), color=BLUE)
    arrow(ax, (0.655, 0.705), (0.692, 0.705), color=BLUE)

    ax.text(0.225, 0.542, "HELD-OUT GROUP", transform=ax.transAxes,
            fontsize=5.3, fontweight="bold", color=SAGE)
    box(0.225, 0.365, 0.145, 0.135, "Held-out unit",
        ["patient / donor", "no adaptive fitting"], SAGE)
    box(0.410, 0.345, 0.245, 0.175, "Training-fitted prediction",
        ["candidate probability", "matched-expression probability", "same groups · folds · gene budget"], SAGE)
    box(0.695, 0.365, 0.145, 0.135, "M4 · Paired audit",
        ["ΔAUROC + group CI", "calibration / Brier"], SAGE, SOFT, align="center")
    box(0.860, 0.365, 0.115, 0.135, "M5 · Nulls",
        ["labels · cells", "coordinates"], GOLD, WHITE, align="center")
    arrow(ax, (0.297, 0.625), (0.297, 0.503), color=SAGE)
    arrow(ax, (0.370, 0.432), (0.407, 0.432), color=SAGE)
    arrow(ax, (0.767, 0.625), (0.585, 0.522), color=BLUE, connectionstyle="arc3,rad=0.16")
    arrow(ax, (0.655, 0.432), (0.692, 0.432), color=SAGE)

    rounded_box(ax, (0.225, 0.105), 0.750, 0.155, face=WHITE, edge=LINE, radius=0.016)
    ax.text(0.250, 0.222, "M6 · EVIDENCE GATE", transform=ax.transAxes,
            fontsize=5.5, fontweight="bold", color=INK)
    gates = [("patient isolated", SAGE), ("outer-fit strict", BLUE),
             ("Δ ≥ 0.05 + stable CI", LAVENDER), ("null test passed", GOLD)]
    for index, (label, color) in enumerate(gates):
        x = 0.250 + index * 0.154
        rounded_box(ax, (x, 0.145), 0.135, 0.045, face=SOFT, edge=color, radius=0.008)
        ax.text(x + 0.0675, 0.1675, label, transform=ax.transAxes, ha="center", va="center",
                fontsize=4.4, fontweight="bold", color=color)
    rounded_box(ax, (0.858, 0.140), 0.095, 0.055, face=SOFT, edge=SAGE, radius=0.008)
    ax.text(0.9055, 0.1675, "claim\nwording", transform=ax.transAxes, ha="center",
            va="center", fontsize=4.3, fontweight="bold", color=SAGE, linespacing=1.0)
    arrow(ax, (0.767, 0.365), (0.767, 0.263), color=SAGE)
    arrow(ax, (0.918, 0.365), (0.918, 0.263), color=GOLD)

    ax.text(0.040, 0.045,
            "Invariant: a held-out patient never informs feature selection, representation fitting or model fitting.",
            transform=ax.transAxes, fontsize=5.2, color=MUTED)
    save(fig, "fig2_pri_audit_overall_architecture")


def module_card(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    number: str,
    name: str,
    input_text: str,
    function_text: str,
    output_text: str,
    color: str,
    light: str,
) -> None:
    x, y = xy
    rounded_box(ax, (x, y), width, height, face=WHITE, edge=LINE, radius=0.016)
    ax.add_patch(Rectangle((x, y + height - 0.050), width, 0.050, transform=ax.transAxes,
                           facecolor=light, edgecolor="none", clip_on=False))
    ax.text(x + 0.020, y + height - 0.026, number, transform=ax.transAxes,
            ha="left", va="center", fontsize=5.0, fontweight="bold", color=color)
    ax.text(x + 0.052, y + height - 0.026, name, transform=ax.transAxes,
            ha="left", va="center", fontsize=6.5, fontweight="bold", color=INK)
    labels = [("INPUT", input_text, 0.156), ("FUNCTION", function_text, 0.096), ("OUTPUT", output_text, 0.036)]
    for label, value, offset in labels:
        ax.text(x + 0.020, y + offset, label, transform=ax.transAxes, ha="left", va="center",
                fontsize=4.1, fontweight="bold", color=color)
        wrapped = "\n".join(textwrap.wrap(value, width=25))
        ax.text(x + 0.102, y + offset, wrapped, transform=ax.transAxes, ha="left", va="center",
                fontsize=4.0, color=MUTED, linespacing=1.06)


def plot_module_map() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.65), facecolor=PAPER)
    ax.axis("off")
    title_block(
        fig,
        "PRI-Audit modules: inputs, functions and auditable outputs",
        "Each module has one job and one artifact that can be inspected or falsified",
    )
    cards = [
        ("M1", "Unit Guard", "sample metadata + group IDs", "locks patient/donor folds", "eligible groups + fold registry", BLUE, BLUE_LIGHT),
        ("M2", "Fold-local Builder", "training-group matrices", "fits HVGs, scaling and basis", "training-fitted transform objects", BLUE, BLUE_LIGHT),
        ("M3", "Resolution Encoders", "fold-local transformed data", "builds expression, cell and graph views", "one vector per biological sample", LAVENDER, LAVENDER_LIGHT),
        ("M4", "Matched Comparator", "candidate + expression vectors", "matches groups, folds and gene budget", "paired out-of-fold probabilities", SAGE, SAGE_LIGHT),
        ("M5", "Falsification Engine", "fixed pipeline + null operation", "breaks labels, cell ownership or coordinates", "complete null distributions", GOLD, GOLD_LIGHT),
        ("M6", "Evidence Gate", "paired predictions + nulls", "combines ΔAUROC, CI and calibration", "supported / bounded / refuted claim", SAGE, SAGE_LIGHT),
    ]
    positions = [(0.040, 0.600), (0.360, 0.600), (0.680, 0.600),
                 (0.680, 0.285), (0.360, 0.285), (0.040, 0.285)]
    for card, position in zip(cards, positions):
        module_card(ax, position, 0.280, 0.235, *card)

    # Horizontal primary information flow.
    arrow(ax, (0.320, 0.717), (0.357, 0.717), color=SLATE)
    arrow(ax, (0.640, 0.717), (0.677, 0.717), color=SLATE)
    arrow(ax, (0.820, 0.600), (0.820, 0.523), color=SLATE)
    arrow(ax, (0.680, 0.402), (0.643, 0.402), color=SLATE)
    arrow(ax, (0.360, 0.402), (0.323, 0.402), color=SLATE)
    ax.text(0.840, 0.548, "sample vectors", transform=ax.transAxes,
            ha="left", fontsize=4.4, color=MUTED)

    # Resolution encoder detail rail.
    rounded_box(ax, (0.045, 0.085), 0.890, 0.105, face=SOFT, edge=LINE, radius=0.014)
    ax.text(0.065, 0.158, "M3 ENCODER LIBRARY", transform=ax.transAxes, fontsize=4.8,
            fontweight="bold", color=INK)
    encoder_items = [
        ("Pseudobulk", "log-CPM expression", BLUE),
        ("Low-rank", "PCA / NMF", BLUE),
        ("Cell neighborhoods", "equal-mass histograms", LAVENDER),
        ("Spatial graph", "agreement + contrast", SAGE),
        ("Perturbation transport", "IL-17 / TNF priors", GOLD),
    ]
    for i, (name, detail, color) in enumerate(encoder_items):
        x = 0.065 + i * 0.174
        ax.text(x, 0.125, name, transform=ax.transAxes, fontsize=4.7, fontweight="bold", color=color)
        ax.text(x, 0.098, detail, transform=ax.transAxes, fontsize=4.1, color=MUTED)
    ax.text(0.045, 0.035, "Auditable artifacts: fold registry · fitted transforms · prediction rows · bootstrap arrays · null distributions · claim registry",
            transform=ax.transAxes, fontsize=5.1, color=MUTED)
    save(fig, "fig3_pri_audit_module_functions")


def write_brief_and_manifest(atlas_metadata: dict) -> None:
    brief = {
        "figure_goal": "Replace the visually aggressive opening and make the method architecture explicit.",
        "paper_claim": "PRI-Audit separates biological signal from reproducible patient-level incremental value under strict group isolation.",
        "figure_type": "mixed-panel plus system architecture and module workflow",
        "mode": "mixed",
        "panels": [
            {"name": "Study landscape", "content": "11 observed public datasets with restrained data summaries"},
            {"name": "Overall architecture", "content": "training and held-out lanes through evidence gate"},
            {"name": "Module functions", "content": "six modules with explicit input, function and output"},
        ],
        "must_keep_labels": [
            "Patient / donor", "Fold-local preprocessing", "Matched expression", "ΔAUROC",
            "Falsification", "Evidence gate",
        ],
        "data": {
            "bulk_summary": "results/paper_v17/calm_atlas_bulk_program_summary.csv",
            "single_cell": "results/paper_v15 fixed-source UMAP exports",
            "spatial": "GSE206391 observed coordinates",
            "perturbation": "results/paper_v16/gse24767_donor_portrait_values.csv",
        },
        "style_constraints": [
            "white background", "low saturation", "large whitespace", "no red heatmap wall",
            "one claim per figure", "short English labels", "vector-friendly",
        ],
        "output_formats": ["png", "pdf", "svg"],
        "verification_checklist": [
            "All 11 dataset identifiers are visible in Figure 1.",
            "Held-out data never flows into fitting in Figure 2.",
            "Every module in Figure 3 has input, function and output.",
            "Labels remain readable at manuscript width.",
        ],
    }
    (CACHE / "v17_figure_brief.json").write_text(json.dumps(brief, indent=2), encoding="utf-8")
    manifest = {
        "seed": SEED,
        "atlas_metadata": atlas_metadata,
        "outputs": [
            "fig1_calm_public_data_landscape",
            "fig2_pri_audit_overall_architecture",
            "fig3_pri_audit_module_functions",
            "figED13_bulk_patient_program_distributions",
            "figED14_calm_cytokine_donor_responses",
        ],
        "formats": ["png", "pdf", "svg"],
        "palette": {
            "blue": BLUE, "sage": SAGE, "gold": GOLD, "lavender": LAVENDER,
            "ink": INK, "paper": PAPER,
        },
    }
    (CACHE / "v17_figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    bulk, metadata = load_bulk()
    _, donor = perturbation_portrait()
    atlas_metadata = plot_study_landscape()
    plot_overall_architecture()
    plot_module_map()
    plot_bulk_extended_calm(bulk, metadata)
    plot_perturbation_extended_calm(donor)
    write_brief_and_manifest(atlas_metadata)
    print((CACHE / "v17_figure_manifest.json").read_text())


if __name__ == "__main__":
    main()

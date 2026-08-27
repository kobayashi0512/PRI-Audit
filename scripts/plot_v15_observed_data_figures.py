"""Build data-derived figures for the expanded PRI-Audit manuscript.

This script adds the display items missing from v14: actual single-cell
landscapes from GSE151177 and GSE173706, a non-cherry-picked multi-patient
GSE206391 spatial gallery, and a compact robustness summary.  All panels are
computed from the frozen public-data matrices or their recorded strict-model
outputs.  The UMAP is fitted only on the GSE151177 source PCA coordinates; the
GSE173706 cells are transformed into that fixed map.
"""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_curve


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
RES = ROOT / "results/singlecell_v10"
V12 = ROOT / "results/paper_v12"
MAIN = ROOT / "figures/paper_v15/main"
ED = ROOT / "figures/paper_v15/extended_data"
CACHE = ROOT / "results/paper_v15"
for directory in (MAIN, ED, CACHE):
    directory.mkdir(parents=True, exist_ok=True)

SEED = 20260811
RNG = np.random.default_rng(SEED)
CHARCOAL = "#353535"
BLUE = "#0072B2"
SKY = "#56B4E9"
GREEN = "#009E73"
ORANGE = "#D55E00"
PURPLE = "#CC79A7"
GOLD = "#E69F00"
GRAY = "#BDBDBD"
LIGHT = "#F2F2F2"
CONDITION_COLORS = {
    "Control": SKY,
    "Psoriasis": ORANGE,
    "Normal skin": GRAY,
    "Peripheral normal": BLUE,
    "Psoriatic plaque": ORANGE,
}
CLUSTER_COLORS = plt.get_cmap("tab20")(np.linspace(0, 1, 16))
MARKERS = ("S100A7", "S100A7A", "KRT16", "IL36G", "CXCL8", "SERPINB4")

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 6.4,
        "axes.titlesize": 7.2,
        "axes.labelsize": 6.4,
        "xtick.labelsize": 5.6,
        "ytick.labelsize": 5.6,
        "legend.fontsize": 5.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def panel(ax, label: str) -> None:
    ax.text(-0.10, 1.06, label, transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")


def save(fig, root: Path, name: str) -> None:
    for extension in ("pdf", "svg", "png"):
        fig.savefig(root / f"{name}.{extension}", dpi=450, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def clean_embedding(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("fixed-source UMAP 1")
    ax.set_ylabel("fixed-source UMAP 2")
    for spine in ax.spines.values():
        spine.set_visible(False)


def source_cells(model: dict) -> pd.DataFrame:
    packed = np.load(RES / "gse151177_patient_balanced_neighborhood_v1.npz", allow_pickle=True)
    coordinates = packed["pca"].astype(np.float32)
    samples = packed["samples"].astype(str)
    offsets = packed["offsets"].astype(int)
    sample_by_cell = np.concatenate(
        [np.repeat(sample, offsets[i + 1] - offsets[i]) for i, sample in enumerate(samples)]
    )
    metadata = pd.read_csv(RES / "gse151177_sample_metadata.csv").set_index("sample")
    conditions = metadata.loc[sample_by_cell, "condition"].map(
        {"control": "Control", "psoriasis": "Psoriasis"}
    ).to_numpy()
    neighborhoods = model["kmeans"].predict(coordinates).astype(int)
    return pd.DataFrame(
        {
            "sample": sample_by_cell,
            "condition": conditions,
            "neighborhood": neighborhoods,
            **{f"PC{i + 1}": coordinates[:, i] for i in range(coordinates.shape[1])},
        }
    )


def target_cells(model: dict, max_cells_per_sample: int = 450) -> pd.DataFrame:
    cache = CACHE / "gse173706_fixed_source_cell_projection.csv.gz"
    if cache.exists():
        return pd.read_csv(cache)

    gene_idx = np.asarray(model["gene_idx"], dtype=int)
    position = {int(row): i for i, row in enumerate(gene_idx)}
    mitochondrial = {
        "ENSG00000198888", "ENSG00000198763", "ENSG00000198804", "ENSG00000198712",
        "ENSG00000228263", "ENSG00000198899", "ENSG00000198938", "ENSG00000198840",
        "ENSG00000198727", "ENSG00000198886", "ENSG00000198786", "ENSG00000198695",
        "ENSG00000198700", "ENSG00000210049", "ENSG00000211459",
    }
    rows = []
    for path in sorted((RAW / "GSE173706_samples").glob("*.csv.gz")):
        with gzip.open(path, "rt", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            n_cells = len(header) - 1
            totals = np.zeros(n_cells, dtype=np.float64)
            detected = np.zeros(n_cells, dtype=np.int32)
            mito = np.zeros(n_cells, dtype=np.float64)
            selected = np.zeros((len(gene_idx), n_cells), dtype=np.float32)
            for row_index, row in enumerate(reader):
                if not row:
                    continue
                gene = row[0].upper()
                values = np.asarray(row[1:], dtype=np.float32)
                totals += values
                detected += values > 0
                if gene in mitochondrial or gene.startswith("MT-"):
                    mito += values
                selected_position = position.get(row_index)
                if selected_position is not None:
                    selected[selected_position] = values
        keep = np.flatnonzero(
            (totals >= 500) & (detected >= 300) & (mito / np.maximum(totals, 1) <= 0.2)
        )
        if len(keep) > max_cells_per_sample:
            keep = RNG.choice(keep, max_cells_per_sample, replace=False)
        normalized = np.log1p(
            (selected[:, keep] / np.maximum(selected[:, keep].sum(axis=0, keepdims=True), 1) * 1e4).T
        )
        pca = model["pca"].transform(model["scaler"].transform(normalized)).astype(np.float32)
        neighborhoods = model["kmeans"].predict(pca).astype(int)
        sample = path.name.replace(".csv.gz", "")
        label = sample.split("_", 1)[1]
        if label.startswith("NS-"):
            condition = "Normal skin"
        elif label.startswith("PN-"):
            condition = "Peripheral normal"
        else:
            condition = "Psoriatic plaque"
        donor = label.split("-", 1)[1]
        for cell_index in range(len(pca)):
            rows.append(
                {
                    "sample": sample,
                    "donor": donor,
                    "condition": condition,
                    "neighborhood": int(neighborhoods[cell_index]),
                    **{f"PC{i + 1}": float(pca[cell_index, i]) for i in range(pca.shape[1])},
                }
            )
    output = pd.DataFrame(rows)
    output.to_csv(cache, index=False, compression="gzip")
    return output


def fixed_umap(source: pd.DataFrame, target: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    source_cache = CACHE / "gse151177_fixed_source_umap.csv.gz"
    target_cache = CACHE / "gse173706_fixed_source_umap.csv.gz"
    if source_cache.exists() and target_cache.exists():
        return pd.read_csv(source_cache), pd.read_csv(target_cache), {"cache_reused": True}

    pc_columns = [f"PC{i}" for i in range(1, 16)]
    mapper = umap.UMAP(
        n_neighbors=30,
        min_dist=0.25,
        metric="euclidean",
        random_state=SEED,
        transform_seed=SEED,
        low_memory=True,
    )
    source_xy = mapper.fit_transform(source[pc_columns].to_numpy(np.float32))
    target_xy = mapper.transform(target[pc_columns].to_numpy(np.float32))
    source = source[["sample", "condition", "neighborhood"]].copy()
    target = target[["sample", "donor", "condition", "neighborhood"]].copy()
    source[["UMAP1", "UMAP2"]] = source_xy
    target[["UMAP1", "UMAP2"]] = target_xy
    source.to_csv(source_cache, index=False, compression="gzip")
    target.to_csv(target_cache, index=False, compression="gzip")
    metadata = {
        "cache_reused": False,
        "fit_dataset": "GSE151177",
        "transformed_dataset": "GSE173706",
        "n_neighbors": 30,
        "min_dist": 0.25,
        "seed": SEED,
        "source_cells": int(len(source)),
        "target_display_cells": int(len(target)),
    }
    return source, target, metadata


def scatter_condition(ax, frame: pd.DataFrame, title: str, order: list[str]) -> None:
    for condition in order:
        subset = frame[frame.condition == condition]
        ax.scatter(
            subset.UMAP1,
            subset.UMAP2,
            s=1.1,
            alpha=0.48,
            linewidths=0,
            rasterized=True,
            color=CONDITION_COLORS[condition],
            label=condition,
        )
    ax.set_title(title, loc="left")
    clean_embedding(ax)
    ax.legend(frameon=False, markerscale=4, loc="best")


def scatter_neighborhood(ax, frame: pd.DataFrame, title: str) -> None:
    ax.scatter(
        frame.UMAP1,
        frame.UMAP2,
        s=1.0,
        alpha=0.48,
        linewidths=0,
        rasterized=True,
        c=CLUSTER_COLORS[frame.neighborhood.astype(int).to_numpy()],
    )
    ax.set_title(title, loc="left")
    clean_embedding(ax)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=CLUSTER_COLORS[i],
               markeredgecolor="none", markersize=3, label=str(i + 1))
        for i in range(16)
    ]
    ax.legend(handles=handles, title="Neighborhood", ncol=4, frameon=False,
              loc="upper right", columnspacing=0.5, handletextpad=0.15, borderaxespad=0.1)


def spatial_data() -> tuple[csr_matrix, pd.DataFrame, np.ndarray, np.ndarray]:
    h5_path = RAW / "GSE206391_Preprocessed_data.h5"
    with h5py.File(h5_path, "r") as handle:
        indptr = handle["/X/indptr"][...]
        matrix = csr_matrix(
            (handle["/X/data"][...], handle["/X/indices"][...], indptr),
            shape=(len(indptr) - 1, 16685),
        )
        obs = {key: handle[f"/obs/{key}"][...] for key in (
            "patient", "biopsy_type", "sample", "array_row", "array_col", "in_tissue"
        )}
        decode = lambda values: np.array(
            [value.decode() if isinstance(value, bytes) else str(value) for value in values]
        )
        categories = {
            key: decode(handle[f"/obs/__categories/{key}"][...])
            for key in ("patient", "biopsy_type", "sample")
        }
        mitochondrial = handle["/raw/var/mt"][...].astype(bool)
        genes = decode(handle["/var/gene_name"][...])
    categorical = lambda key: categories[key][obs[key].astype(int)]
    metadata = pd.DataFrame(
        {
            "patient": categorical("patient"),
            "condition": categorical("biopsy_type"),
            "sample": categorical("sample"),
            "row": obs["array_row"],
            "col": obs["array_col"],
            "in_tissue": obs["in_tissue"],
        }
    )
    total = np.asarray(matrix.sum(axis=1)).ravel()
    detected = np.asarray((matrix > 0).sum(axis=1)).ravel()
    mito_fraction = np.asarray(matrix[:, mitochondrial].sum(axis=1)).ravel() / np.maximum(total, 1)
    metadata["keep"] = (
        (total >= 500) & (detected >= 300) & (mito_fraction <= 0.2)
        & metadata.in_tissue.astype(bool).to_numpy()
    )
    return matrix, metadata, genes, total


def spatial_selection() -> pd.DataFrame:
    predictions = pd.read_csv(RES / "gse206391_strict_spatial_graph_predictions.csv")
    means = predictions.groupby(["patient", "true"])["expression_plus_spatial_probability"].mean().unstack()
    means["paired_delta"] = means[1] - means[0]
    targets = {"low": 0.10, "median": 0.50, "high": 0.90}
    chosen = []
    used = set()
    for label, quantile in targets.items():
        target = float(means.paired_delta.quantile(quantile))
        candidates = (means.paired_delta - target).abs().sort_values()
        patient = next(str(value) for value in candidates.index if str(value) not in used)
        used.add(patient)
        subset = predictions[predictions.patient.astype(str) == patient]
        subset = (
            subset.sort_values("filtered_spots", ascending=False)
            .groupby("condition", sort=True)
            .head(1)
            .sort_values("true")
            .copy()
        )
        subset["selection_stratum"] = label
        subset["paired_delta"] = float(means.loc[int(patient) if means.index.dtype.kind in "iu" else patient, "paired_delta"])
        chosen.append(subset)
    return pd.concat(chosen, ignore_index=True)


def make_spatial_maps() -> tuple[list[dict], dict]:
    selection = spatial_selection()
    matrix, metadata, genes, total = spatial_data()
    marker_idx = [int(np.flatnonzero(genes == marker)[0]) for marker in MARKERS if np.any(genes == marker)]
    maps = []
    for _, selected in selection.iterrows():
        indices = metadata.index[(metadata["sample"] == selected["sample"]) & metadata["keep"]].to_numpy()
        counts = matrix[indices][:, marker_idx].toarray().astype(np.float32)
        normalized = np.log1p(counts / np.maximum(total[indices, None], 1) * 1e4)
        maps.append(
            {
                "sample": str(selected["sample"]),
                "patient": str(selected["patient"]),
                "condition": str(selected["condition"]),
                "stratum": str(selected["selection_stratum"]),
                "paired_delta": float(selected["paired_delta"]),
                "row": metadata.loc[indices, "row"].to_numpy(),
                "col": metadata.loc[indices, "col"].to_numpy(),
                "score": normalized.mean(axis=1),
            }
        )
    low = min(float(np.quantile(item["score"], 0.01)) for item in maps)
    high = max(float(np.quantile(item["score"], 0.99)) for item in maps)
    metadata_record = {
        "selection_rule": "patients nearest the 10th, 50th and 90th percentiles of strict paired expression-plus-spatial LS-minus-NL probability; largest retained-spot section per condition",
        "marker_panel_declared_in_code": list(MARKERS),
        "selected": [
            {key: item[key] for key in ("sample", "patient", "condition", "stratum", "paired_delta")}
            | {"retained_spots": int(len(item["score"]))}
            for item in maps
        ],
        "vmin": low,
        "vmax": high,
    }
    return maps, metadata_record


def plot_spatial_gallery(maps: list[dict], output_root: Path, name: str, *, panel_label: str | None = None) -> None:
    figure = plt.figure(figsize=(7.2, 4.1))
    grid = figure.add_gridspec(2, 3, wspace=0.08, hspace=0.22)
    scatter = None
    values = np.concatenate([item["score"] for item in maps])
    vmin, vmax = float(np.quantile(values, 0.01)), float(np.quantile(values, 0.99))
    for index, item in enumerate(maps):
        row = 0 if item["condition"] == "NON LESIONAL" else 1
        column = ["low", "median", "high"].index(item["stratum"])
        ax = figure.add_subplot(grid[row, column])
        scatter = ax.scatter(
            item["col"], -item["row"], c=item["score"], s=2.0, cmap="magma",
            vmin=vmin, vmax=vmax, linewidths=0, rasterized=True,
        )
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        condition = "NL" if item["condition"] == "NON LESIONAL" else "LS"
        ax.set_title(
            f"{item['stratum'].title()} effect · patient {item['patient']}\n{condition} · {item['sample']} · n={len(item['score']):,}",
            fontsize=6.3,
        )
        for spine in ax.spines.values():
            spine.set_visible(False)
        if panel_label and index == 0:
            panel(ax, panel_label)
    colorbar = figure.colorbar(scatter, ax=figure.axes, shrink=0.72, pad=0.015)
    colorbar.set_label("fixed inflammatory module score")
    figure.suptitle(
        "Observed GSE206391 tissue heterogeneity under a predefined patient-selection rule",
        fontsize=8.2,
        fontweight="bold",
        y=0.99,
    )
    save(figure, output_root, name)


def plot_cell_extended(source: pd.DataFrame, target: pd.DataFrame) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(7.2, 5.0))
    scatter_condition(axes[0, 0], source, "GSE151177 · observed condition", ["Control", "Psoriasis"])
    panel(axes[0, 0], "a")
    scatter_neighborhood(axes[0, 1], source, "GSE151177 · frozen neighborhoods")
    panel(axes[0, 1], "b")
    source_counts = (
        source.groupby(["sample", "condition", "neighborhood"], as_index=False)
        .size()
        .rename(columns={"size": "cells"})
    )
    source_counts["fraction"] = source_counts["cells"] / source_counts.groupby(
        ["sample", "condition"]
    )["cells"].transform("sum")
    ax = axes[0, 2]
    for condition, color in [("Control", SKY), ("Psoriasis", ORANGE)]:
        subset = source_counts[source_counts.condition == condition]
        means = subset.groupby("neighborhood").fraction.mean().reindex(range(16), fill_value=0)
        ax.plot(np.arange(1, 17), means, "o-", color=color, ms=2.2, lw=0.9, label=condition)
    ax.set_xticks([1, 4, 8, 12, 16])
    ax.set_xlabel("Frozen neighborhood")
    ax.set_ylabel("Mean per-sample cell fraction")
    ax.set_title("Equal-patient source composition", loc="left")
    ax.legend(frameon=False)
    panel(ax, "c")

    scatter_condition(
        axes[1, 0], target, "GSE173706 · fixed-map condition",
        ["Normal skin", "Peripheral normal", "Psoriatic plaque"],
    )
    panel(axes[1, 0], "d")
    scatter_neighborhood(axes[1, 1], target, "GSE173706 · source-defined neighborhoods")
    panel(axes[1, 1], "e")
    arrays = np.load(RES / "gse173706_neighborhood_transfer_arrays.npz", allow_pickle=True)
    histograms = arrays["histograms"].astype(float)
    labels = arrays["labels"].astype(int)
    donors = arrays["donors"].astype(str)
    differences = []
    for donor in np.unique(donors):
        indices = np.flatnonzero(donors == donor)
        differences.append(histograms[indices[labels[indices] == 1][0]] - histograms[indices[labels[indices] == 0][0]])
    differences = np.vstack(differences)
    ax = axes[1, 2]
    for row in differences:
        ax.plot(np.arange(1, 17), row, color=GRAY, lw=0.45, alpha=0.65)
    mean = differences.mean(axis=0)
    ax.plot(np.arange(1, 17), mean, "o-", color=ORANGE, lw=1.1, ms=2.3, label="donor mean")
    ax.axhline(0, color=CHARCOAL, lw=0.6)
    ax.set_xticks([1, 4, 8, 12, 16])
    ax.set_xlabel("Frozen neighborhood")
    ax.set_ylabel("PP − PN cell fraction")
    ax.set_title("Paired external composition shifts", loc="left")
    ax.legend(frameon=False)
    panel(ax, "f")
    figure.subplots_adjust(left=0.07, right=0.98, bottom=0.09, top=0.96, wspace=0.42, hspace=0.38)
    save(figure, ED, "figED11_observed_single_cell_landscapes")


def plot_main_atlas(source: pd.DataFrame, target: pd.DataFrame, maps: list[dict]) -> None:
    figure = plt.figure(figsize=(7.2, 7.0))
    outer = figure.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 1.45], wspace=0.50, hspace=0.46)
    axes = [figure.add_subplot(outer[0, 0:2]), figure.add_subplot(outer[0, 2:4]),
            figure.add_subplot(outer[1, 0:2]), figure.add_subplot(outer[1, 2:4])]
    scatter_condition(axes[0], source, "GSE151177 source cells · condition", ["Control", "Psoriasis"])
    panel(axes[0], "a")
    scatter_neighborhood(axes[1], source, "GSE151177 source cells · frozen neighborhoods")
    panel(axes[1], "b")
    scatter_condition(
        axes[2], target, "GSE173706 cells transformed into the fixed source map",
        ["Normal skin", "Peripheral normal", "Psoriatic plaque"],
    )
    panel(axes[2], "c")
    scatter_neighborhood(axes[3], target, "GSE173706 cells · source-defined neighborhoods")
    panel(axes[3], "d")

    spatial_grid = outer[2, :].subgridspec(2, 3, wspace=0.05, hspace=0.18)
    all_values = np.concatenate([item["score"] for item in maps])
    vmin, vmax = float(np.quantile(all_values, 0.01)), float(np.quantile(all_values, 0.99))
    scatter = None
    for index, item in enumerate(maps):
        row = 0 if item["condition"] == "NON LESIONAL" else 1
        column = ["low", "median", "high"].index(item["stratum"])
        ax = figure.add_subplot(spatial_grid[row, column])
        scatter = ax.scatter(item["col"], -item["row"], c=item["score"], s=1.6, cmap="magma",
                             vmin=vmin, vmax=vmax, linewidths=0, rasterized=True)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        condition = "NL" if item["condition"] == "NON LESIONAL" else "LS"
        ax.set_title(f"{item['stratum'].title()} · patient {item['patient']} · {condition}", fontsize=5.8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        if index == 0:
            panel(ax, "e")
    colorbar_axis = figure.add_axes([0.935, 0.055, 0.012, 0.255])
    colorbar = figure.colorbar(scatter, cax=colorbar_axis)
    colorbar.set_label("fixed inflammatory module score")
    figure.subplots_adjust(left=0.07, right=0.91, bottom=0.05, top=0.97)
    save(figure, MAIN, "fig7_observed_data_atlas")


def plot_robustness_main() -> None:
    figure = plt.figure(figsize=(7.2, 7.0))
    grid = figure.add_gridspec(3, 2, height_ratios=[0.88, 1.05, 1.0], wspace=0.46, hspace=0.52)
    ax = figure.add_subplot(grid[0, 0])
    colors = [BLUE, ORANGE, GREEN]
    for dataset, color in zip(("gse151177", "gse173706", "gse206391"), colors):
        frame = pd.read_csv(RES / f"{dataset}_strict_benchmark_predictions.csv")
        fpr, tpr, _ = roc_curve(frame.true, frame.pseudobulk_probability)
        ax.plot(fpr, tpr, color=color, lw=1.2, label=dataset.upper())
    ax.plot([0, 1], [0, 1], "--", color=GRAY, lw=0.7)
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_title("Strict expression ROC", loc="left")
    ax.legend(frameon=False)
    panel(ax, "a")

    ax = figure.add_subplot(grid[1, :])
    gains = pd.read_csv(V12 / "incremental_gain_bootstrap.csv").iloc[::-1].reset_index(drop=True)
    gains["label"] = gains.dataset + " · " + gains.candidate.str.replace("strict ", "", regex=False)
    y = np.arange(len(gains))
    ax.errorbar(
        gains.delta_auroc, y,
        xerr=[gains.delta_auroc - gains.delta_ci95_low, gains.delta_ci95_high - gains.delta_auroc],
        fmt="none", ecolor=CHARCOAL, lw=0.7, capsize=2,
    )
    ax.scatter(gains.delta_auroc, y, color=BLUE, s=16, zorder=3)
    ax.axvline(0, color=CHARCOAL, lw=0.7)
    ax.axvline(0.05, color=GRAY, lw=0.7, ls="--")
    ax.set_yticks(y, gains.label)
    ax.set_xlabel("Patient/donor-bootstrap ΔAUROC")
    ax.set_title("All paired increments remain below the practical-value gate", loc="left")
    panel(ax, "b")

    sensitivity = pd.read_csv(RES / "gse206391_strict_spatial_graph_sensitivity_matrix.csv")
    ax = figure.add_subplot(grid[2, 0])
    heat = sensitivity.groupby(["hvg", "neighbors"])["auroc"].mean().unstack()
    image = ax.imshow(heat.to_numpy(), aspect="auto", cmap="viridis", vmin=sensitivity.auroc.min(), vmax=sensitivity.auroc.max())
    ax.set_xticks(range(len(heat.columns)), heat.columns)
    ax.set_yticks(range(len(heat.index)), heat.index)
    ax.set_xlabel("Spatial neighbors")
    ax.set_ylabel("HVG count")
    ax.set_title("Mean strict spatial AUROC", loc="left")
    panel(ax, "c")
    figure.colorbar(image, ax=ax, shrink=0.70)

    ax = figure.add_subplot(grid[2, 1])
    primary = sensitivity[(sensitivity.hvg == 500) & (sensitivity.neighbors == 6)]
    heat = primary.pivot_table(index="C", columns="seed", values="auroc", aggfunc="mean")
    image = ax.imshow(heat.to_numpy(), aspect="auto", cmap="viridis", vmin=sensitivity.auroc.min(), vmax=sensitivity.auroc.max())
    ax.set_xticks(range(len(heat.columns)), heat.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(heat.index)), heat.index)
    ax.set_xlabel("Random seed")
    ax.set_ylabel("Regularization C")
    ax.set_title("Primary-grid C × seed", loc="left")
    panel(ax, "d")

    ax = figure.add_subplot(grid[0, 1])
    null_specs = [
        ("Cell", np.load(RES / "cell_cross_patient_shuffle_null_distribution.npz")["null_aurocs"],
         float(np.load(RES / "cell_cross_patient_shuffle_null_distribution.npz")["observed"][0]), ORANGE),
        ("Ext. label", np.load(RES / "gse173706_neighborhood_transfer_null_distributions.npz")["permutation_aurocs"],
         float(np.load(RES / "gse173706_neighborhood_transfer_null_distributions.npz")["observed"][0]), BLUE),
        ("Coordinate", np.load(RES / "gse206391_strict_spatial_coordinate_shuffle_distributions.npz")["null_spatial"],
         float(np.load(RES / "gse206391_strict_spatial_coordinate_shuffle_distributions.npz")["observed_spatial"][0]), GREEN),
    ]
    positions = np.arange(len(null_specs))
    boxes = ax.boxplot([item[1] for item in null_specs], positions=positions, widths=0.55, patch_artist=True, showfliers=False)
    for box, (_, _, _, color) in zip(boxes["boxes"], null_specs):
        box.set_facecolor(color)
        box.set_alpha(0.35)
    for pos, (_, _, observed, color) in zip(positions, null_specs):
        ax.scatter(pos, observed, marker="D", s=25, color=color, edgecolor=CHARCOAL, linewidth=0.4, zorder=4)
    ax.set_xticks(positions, [item[0] for item in null_specs], rotation=25, ha="right")
    ax.set_ylabel("Null AUROC; diamond = observed")
    ax.set_title("Resolution-specific nulls", loc="left")
    panel(ax, "e")
    figure.subplots_adjust(left=0.10, right=0.98, bottom=0.08, top=0.97)
    save(figure, MAIN, "fig8_robustness_multiverse")


def main() -> None:
    model = joblib.load(RES / "patient_balanced_neighborhood_v1_models.joblib")
    source = source_cells(model)
    target = target_cells(model)
    source_umap, target_umap, umap_metadata = fixed_umap(source, target)
    maps, spatial_metadata = make_spatial_maps()
    plot_cell_extended(source_umap, target_umap)
    plot_spatial_gallery(maps, ED, "figED12_predefined_spatial_gallery")
    plot_main_atlas(source_umap, target_umap, maps)
    plot_robustness_main()
    manifest = {
        "seed": SEED,
        "umap": umap_metadata,
        "spatial": spatial_metadata,
        "outputs": {
            "main": ["fig7_observed_data_atlas", "fig8_robustness_multiverse"],
            "extended_data": ["figED11_observed_single_cell_landscapes", "figED12_predefined_spatial_gallery"],
        },
    }
    (CACHE / "observed_data_figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

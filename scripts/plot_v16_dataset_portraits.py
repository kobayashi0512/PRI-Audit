"""Create a warm, data-first visual atlas for all 11 public datasets.

The figure set is descriptive rather than inferential.  Every mark is derived
from a frozen public-data matrix already used by the manuscript: paired bulk
LS-NL rank changes, fixed-map single-cell coordinates, observed spatial spots,
or donor-paired GSE24767 cytokine responses.  No display panel contributes to
feature selection, model tuning, or performance estimation.
"""

from __future__ import annotations

import csv
import gzip
import json
import re
import sys
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
from scipy.sparse import csr_matrix
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
RESULTS = ROOT / "results"
MAIN = ROOT / "figures/paper_v16/main"
ED = ROOT / "figures/paper_v16/extended_data"
CACHE = ROOT / "results/paper_v16"
for directory in (MAIN, ED, CACHE):
    directory.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "scripts"))
from common import collapse_duplicate_genes, parse_geo_series_matrix  # noqa: E402
from run_isrc_v2 import load_platform_map  # noqa: E402


SEED = 20260811
RNG = np.random.default_rng(SEED)
INK = "#2B2530"
MUTED = "#716772"
PAPER = "#FFFDFC"
CARD = "#FFF9F5"
GRID = "#E9DED8"
NL = "#2A9D8F"
LS = "#D1495B"
CONTROL = "#52B6A8"
PSORIASIS = "#D95D6A"
PN = "#5B8DB8"
PP = "#D95D6A"
IMMUNE = "#7E57A2"
KERATIN = "#E9A03B"
PERTURB = "#B85C8A"
SPATIAL = "#874C62"

COHORT_ACCENTS = {
    "GSE117239": "#A44A5B",
    "GSE136757": "#C66B55",
    "GSE153007": "#8F5AA6",
    "GSE182740": "#D28E3D",
    "GSE51440": "#A83F74",
    "GSE57225": "#7D6AA5",
    "GSE85034": "#B5533C",
}

PROGRAMS = {
    "Epidermal": ["SERPINB3", "SERPINB4", "KRT16", "KRT6A", "PI3", "S100A9"],
    "IL-17 / innate": ["IL36G", "CCL20", "IL17C", "KYNU", "GBP5", "TNFAIP3"],
    "Interferon": ["CXCL9", "CXCL10", "IFIT1", "MX1", "OASL", "STAT1"],
}
PANEL = [gene for genes in PROGRAMS.values() for gene in genes]
SPATIAL_MARKERS = ("S100A7", "S100A7A", "KRT16", "IL36G", "SERPINB4", "PI3")

HEATMAP = LinearSegmentedColormap.from_list(
    "lesion_change", ["#275D6E", "#8FC4BD", "#FFF8EE", "#EFAC78", "#A52D48"]
)
CYTOKINE = LinearSegmentedColormap.from_list(
    "cytokine", ["#3A6777", "#A7D0C8", "#FFF9F0", "#E8A26F", "#A93B63"]
)

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 6.3,
        "axes.titlesize": 7.1,
        "axes.labelsize": 6.2,
        "xtick.labelsize": 5.5,
        "ytick.labelsize": 5.5,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": GRID,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def save(fig: plt.Figure, root: Path, name: str) -> None:
    for extension in ("pdf", "svg", "png"):
        fig.savefig(
            root / f"{name}.{extension}", dpi=450, bbox_inches="tight", facecolor=PAPER
        )
    plt.close(fig)


def card_style(ax: plt.Axes, accent: str, modality: str) -> None:
    ax.set_facecolor(CARD)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#E5D8D2")
        spine.set_linewidth(0.65)
    ax.add_patch(
        Rectangle((0, 0.985), 1, 0.025, transform=ax.transAxes, color=accent, clip_on=False)
    )
    ax.text(
        0.985, 0.965, modality.upper(), transform=ax.transAxes, ha="right", va="top",
        fontsize=4.7, fontweight="bold", color=accent,
    )


def load_bulk() -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix = pd.read_parquet(RESULTS / "trigger_v9/v9_expanded_ls_minus_nl_rank.parquet")
    metadata = pd.read_csv(RESULTS / "trigger_v9/v9_expanded_metadata.csv")
    missing = sorted(set(PANEL) - set(matrix.columns))
    if missing:
        raise RuntimeError(f"fixed portrait genes missing from shared bulk matrix: {missing}")
    return matrix, metadata


def ordered_bulk_panel(
    matrix: pd.DataFrame, metadata: pd.DataFrame, cohort: str
) -> tuple[np.ndarray, pd.DataFrame]:
    indices = np.asarray(list(metadata.groupby("cohort").groups[cohort]), dtype=int)
    values = matrix.iloc[indices][PANEL].to_numpy(float) / matrix.shape[1]
    order = np.argsort(values.mean(axis=1))
    return values[order].T, metadata.iloc[indices[order]].reset_index(drop=True)


def program_boundaries(ax: plt.Axes, *, labels: bool) -> None:
    for boundary in (5.5, 11.5):
        ax.axhline(boundary, color=PAPER, lw=1.0)
    if labels:
        for y, label, color in [
            (2.5, "Epidermal", KERATIN),
            (8.5, "IL-17 / innate", LS),
            (14.5, "Interferon", IMMUNE),
        ]:
            ax.text(-0.025, y, label, transform=ax.get_yaxis_transform(), ha="right", va="center",
                    fontsize=5.2, fontweight="bold", color=color)


def bulk_card(
    ax: plt.Axes, matrix: pd.DataFrame, metadata: pd.DataFrame, cohort: str,
    *, show_gene_labels: bool = False,
) -> mpl.image.AxesImage:
    values, ordered = ordered_bulk_panel(matrix, metadata, cohort)
    image = ax.imshow(values, cmap=HEATMAP, norm=TwoSlopeNorm(vmin=-0.60, vcenter=0, vmax=0.60),
                      aspect="auto", interpolation="nearest", rasterized=True)
    ax.set_title(f"{cohort}  ·  {values.shape[1]} paired patients", loc="left", pad=5,
                 fontsize=6.6, fontweight="bold")
    ax.set_xticks([])
    if show_gene_labels:
        ax.set_yticks(np.arange(len(PANEL)), PANEL)
        gene_colors = {
            gene: color
            for color, genes in [(KERATIN, PROGRAMS["Epidermal"]), (LS, PROGRAMS["IL-17 / innate"]),
                                 (IMMUNE, PROGRAMS["Interferon"])]
            for gene in genes
        }
        for label in ax.get_yticklabels():
            label.set_color(gene_colors[label.get_text()])
    else:
        ax.set_yticks([2.5, 8.5, 14.5], ["epidermal", "IL-17", "IFN"])
    ax.tick_params(axis="y", length=0, pad=2)
    program_boundaries(ax, labels=False)
    card_style(ax, COHORT_ACCENTS[cohort], "paired bulk")
    ax.text(0.015, 0.02, "NL  ←  within-patient rank shift  →  LS", transform=ax.transAxes,
            fontsize=4.5, color=MUTED, va="bottom")
    return image


def load_single_cell_maps() -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(RESULTS / "paper_v15/gse151177_fixed_source_umap.csv.gz")
    target = pd.read_csv(RESULTS / "paper_v15/gse173706_fixed_source_umap.csv.gz")
    return source, target


def downsample(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    if len(frame) <= n:
        return frame
    return frame.iloc[np.sort(RNG.choice(len(frame), n, replace=False))]


def single_cell_card(
    ax: plt.Axes, frame: pd.DataFrame, dataset: str, order: list[str], colors: dict[str, str]
) -> None:
    display = downsample(frame, 9000)
    for condition in order:
        subset = display[display.condition == condition]
        ax.scatter(subset.UMAP1, subset.UMAP2, s=0.55, alpha=0.50, color=colors[condition],
                   linewidths=0, rasterized=True, label=condition)
    ax.set_title(f"{dataset}  ·  observed cells", loc="left", pad=5,
                 fontsize=6.6, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("each point = observed cell", fontsize=4.7, labelpad=1)
    ax.legend(frameon=False, loc="lower left", fontsize=4.4, markerscale=3.8,
              borderpad=0.1, handletextpad=0.2, labelspacing=0.25)
    card_style(ax, IMMUNE, "single cell")


def spatial_maps() -> list[dict]:
    predictions = pd.read_csv(RESULTS / "singlecell_v10/gse206391_strict_spatial_graph_predictions.csv")
    means = predictions.groupby(["patient", "true"])["expression_plus_spatial_probability"].mean().unstack()
    means["paired_delta"] = means[1] - means[0]
    target = float(means.paired_delta.median())
    patient = (means.paired_delta - target).abs().idxmin()
    selected = predictions[predictions.patient.astype(str) == str(patient)]
    selected = selected.sort_values("filtered_spots", ascending=False).groupby("condition").head(1)

    h5_path = RAW / "GSE206391_Preprocessed_data.h5"
    with h5py.File(h5_path, "r") as handle:
        indptr = handle["/X/indptr"][...]
        matrix = csr_matrix(
            (handle["/X/data"][...], handle["/X/indices"][...], indptr),
            shape=(len(indptr) - 1, 16685),
        )
        decode = lambda values: np.array([
            value.decode() if isinstance(value, bytes) else str(value) for value in values
        ])
        categories = {
            key: decode(handle[f"/obs/__categories/{key}"][...])
            for key in ("patient", "biopsy_type", "sample")
        }
        obs = {key: handle[f"/obs/{key}"][...] for key in (
            "patient", "biopsy_type", "sample", "array_row", "array_col", "in_tissue"
        )}
        genes = decode(handle["/var/gene_name"][...])
        mitochondrial = handle["/raw/var/mt"][...].astype(bool)
    categorical = lambda key: categories[key][obs[key].astype(int)]
    metadata = pd.DataFrame({
        "patient": categorical("patient"), "condition": categorical("biopsy_type"),
        "sample": categorical("sample"), "row": obs["array_row"], "col": obs["array_col"],
        "in_tissue": obs["in_tissue"],
    })
    total = np.asarray(matrix.sum(axis=1)).ravel()
    detected = np.asarray((matrix > 0).sum(axis=1)).ravel()
    mito_fraction = np.asarray(matrix[:, mitochondrial].sum(axis=1)).ravel() / np.maximum(total, 1)
    keep = ((total >= 500) & (detected >= 300) & (mito_fraction <= 0.20)
            & metadata.in_tissue.astype(bool).to_numpy())
    marker_idx = [int(np.flatnonzero(genes == gene)[0]) for gene in SPATIAL_MARKERS if np.any(genes == gene)]
    maps = []
    for row in selected.itertuples(index=False):
        indices = metadata.index[(metadata["sample"] == row.sample) & keep].to_numpy()
        counts = matrix[indices][:, marker_idx].toarray().astype(np.float32)
        score = np.log1p(counts / np.maximum(total[indices, None], 1) * 1e4).mean(axis=1)
        maps.append({
            "sample": str(row.sample), "patient": str(row.patient), "condition": str(row.condition),
            "row": metadata.loc[indices, "row"].to_numpy(),
            "col": metadata.loc[indices, "col"].to_numpy(), "score": score,
        })
    return sorted(maps, key=lambda item: item["condition"])


def spatial_card(ax: plt.Axes, maps: list[dict]) -> None:
    ax.set_title(f"GSE206391  ·  observed LS/NL tissue", loc="left", pad=5,
                 fontsize=6.6, fontweight="bold")
    values = np.concatenate([item["score"] for item in maps])
    vmin, vmax = np.quantile(values, [0.01, 0.99])
    offsets = [-0.56, 0.56]
    for item, offset in zip(maps, offsets):
        x = item["col"].astype(float); y = -item["row"].astype(float)
        x = (x - np.median(x)) / max(np.ptp(x), 1) * 0.84 + offset
        y = (y - np.median(y)) / max(np.ptp(y), 1) * 0.84
        ax.scatter(x, y, c=item["score"], cmap="magma", vmin=vmin, vmax=vmax,
                   s=0.75, linewidths=0, rasterized=True)
        label = "NL" if "NON" in item["condition"].upper() else "LS"
        ax.text(offset, -0.52, label, ha="center", va="top", fontsize=5.4,
                fontweight="bold", color=NL if label == "NL" else LS)
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-0.68, 0.58)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("observed spots · fixed inflammatory module", fontsize=4.7, labelpad=1)
    card_style(ax, SPATIAL, "spatial")


def geo_chars(path: Path) -> dict[str, list[str]]:
    fields: dict[str, list[str]] = {}
    with gzip.open(path, "rt", errors="replace") as handle:
        for line in handle:
            if line.startswith("!Sample_characteristics_ch1"):
                values = [value.strip('"') for value in line.rstrip("\n").split("\t")[1:]]
                key = values[0].split(":", 1)[0].lower()
                fields[key] = [value.split(":", 1)[-1].strip() for value in values]
    return fields


def perturbation_portrait() -> tuple[pd.DataFrame, pd.DataFrame]:
    common = pd.read_csv(RESULTS / "trigger_v8/v8_common_genes.csv").gene.astype(str).tolist()
    dictionary = pd.read_parquet(RESULTS / "trigger_v8/v8_perturbation_dictionary.parquet")
    dictionary.index = common
    mean = dictionary.loc[PANEL].copy()
    mean.columns = ["IL-17", "TNF", "synergy residual"]

    metadata = pd.read_csv(RESULTS / "trigger_v8/v8_keratinocyte_perturbation_metadata.csv")
    pivot = metadata.pivot_table(index="donor", columns="treatment", values="accession", aggfunc="first")
    needed = ["none", "IL17", "10 ng/ml TNFα", "IL17 + 10 ng/ml TNFα"]
    pivot = pivot.dropna(subset=needed)
    expression, _ = parse_geo_series_matrix(RAW / "GSE24767_series_matrix.txt.gz")
    expression.index = expression.index.astype(str).map(load_platform_map(RAW / "GPL6947.annot.gz"))
    expression = collapse_duplicate_genes(expression)
    expression = expression.loc[expression.index.notna() & ~expression.index.duplicated()]
    expression = expression.loc[common]

    def ranks(accession: str) -> pd.Series:
        values = rankdata(expression[accession].to_numpy(float), method="average") / len(expression)
        return pd.Series(values, index=expression.index)

    rows = []
    for donor, row in pivot.iterrows():
        base = ranks(row["none"])
        il17 = ranks(row["IL17"]) - base
        tnf = ranks(row["10 ng/ml TNFα"]) - base
        combo = ranks(row["IL17 + 10 ng/ml TNFα"]) - base
        directions = {"IL-17": il17, "TNF": tnf, "combination": combo,
                      "synergy residual": combo - il17 - tnf}
        for condition, vector in directions.items():
            for gene in PANEL:
                rows.append({"donor": str(donor), "condition": condition, "gene": gene,
                             "rank_shift": float(vector.loc[gene])})
    donor = pd.DataFrame(rows)
    return mean, donor


def perturbation_card(ax: plt.Axes, mean: pd.DataFrame) -> mpl.image.AxesImage:
    values = mean.to_numpy(float)
    vmax = max(0.15, float(np.quantile(np.abs(values), 0.98)))
    image = ax.imshow(values, cmap=CYTOKINE, norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax),
                      aspect="auto", interpolation="nearest")
    ax.set_title("GSE24767  ·  keratinocyte cytokine response", loc="left", pad=5,
                 fontsize=6.6, fontweight="bold")
    ax.set_xticks(range(3), ["IL-17", "TNF", "synergy"], rotation=25, ha="right")
    ax.set_yticks([2.5, 8.5, 14.5], ["epidermal", "IL-17", "IFN"])
    ax.tick_params(length=0, pad=2)
    program_boundaries(ax, labels=False)
    card_style(ax, PERTURB, "perturbation")
    return image


def plot_main_atlas(
    bulk: pd.DataFrame, metadata: pd.DataFrame, source: pd.DataFrame, target: pd.DataFrame,
    maps: list[dict], perturbation_mean: pd.DataFrame,
) -> None:
    figure = plt.figure(figsize=(7.2, 7.65), facecolor=PAPER)
    grid = figure.add_gridspec(3, 4, left=0.055, right=0.975, bottom=0.08, top=0.895,
                              wspace=0.25, hspace=0.35)
    cohorts = list(COHORT_ACCENTS)
    images = []
    for index, cohort in enumerate(cohorts):
        row, column = divmod(index, 4)
        images.append(bulk_card(figure.add_subplot(grid[row, column]), bulk, metadata, cohort))
    perturbation_card(figure.add_subplot(grid[1, 3]), perturbation_mean)
    single_cell_card(
        figure.add_subplot(grid[2, 0]), source, "GSE151177", ["Control", "Psoriasis"],
        {"Control": CONTROL, "Psoriasis": PSORIASIS},
    )
    single_cell_card(
        figure.add_subplot(grid[2, 1]), target, "GSE173706",
        ["Normal skin", "Peripheral normal", "Psoriatic plaque"],
        {"Normal skin": CONTROL, "Peripheral normal": PN, "Psoriatic plaque": PP},
    )
    spatial_card(figure.add_subplot(grid[2, 2:4]), maps)
    figure.suptitle("Eleven public datasets, eleven visible biological data layers", x=0.055, y=0.975,
                    ha="left", fontsize=12.5, fontweight="bold", color=INK)
    figure.text(
        0.055, 0.932,
        "Observed patient expression, cells, tissue coordinates and cytokine responses — not schematic placeholders",
        ha="left", fontsize=7.1, color=MUTED,
    )
    colorbar_axis = figure.add_axes([0.27, 0.026, 0.46, 0.011])
    colorbar = figure.colorbar(images[0], cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label("paired LS−NL fractional rank shift", labelpad=2)
    colorbar.set_ticks([-0.6, 0, 0.6], labels=["NL higher", "no shift", "LS higher"])
    save(figure, MAIN, "fig1_eleven_dataset_visual_atlas")


def plot_bulk_extended(bulk: pd.DataFrame, metadata: pd.DataFrame) -> None:
    figure = plt.figure(figsize=(7.2, 8.4), facecolor=PAPER)
    grid = figure.add_gridspec(4, 2, left=0.16, right=0.975, bottom=0.075, top=0.915,
                              wspace=0.25, hspace=0.35)
    images = []
    for index, cohort in enumerate(COHORT_ACCENTS):
        ax = figure.add_subplot(grid[index // 2, index % 2])
        image = bulk_card(ax, bulk, metadata, cohort, show_gene_labels=True)
        images.append(image)
        ax.set_xlabel("paired patients ordered by fixed-panel mean", fontsize=5.3)
    legend = figure.add_subplot(grid[3, 1])
    legend.axis("off")
    legend.text(0.02, 0.88, "How to read these portraits", fontsize=8.0, fontweight="bold", color=INK)
    legend.text(
        0.02, 0.76,
        "Each column is one paired patient.\nEach row is one gene fixed before plotting.\n"
        "Color is the within-patient LS−NL shift in\ngenome-wide expression rank on a shared scale.\n\n"
        "Patient order is descriptive and cohort-specific.\nNo portrait entered model selection or inference.",
        fontsize=6.4, color=MUTED, va="top", linespacing=1.45,
    )
    colorbar_axis = figure.add_axes([0.61, 0.083, 0.29, 0.012])
    colorbar = figure.colorbar(images[0], cax=colorbar_axis, orientation="horizontal")
    colorbar.set_ticks([-0.6, 0, 0.6], labels=["NL higher", "0", "LS higher"])
    colorbar.set_label("fractional rank shift", labelpad=2)
    figure.suptitle("Seven independent cohorts reveal recurrent and patient-variable lesion programs",
                    x=0.055, y=0.982, ha="left", fontsize=11.0, fontweight="bold")
    figure.text(0.055, 0.948, "Real paired-patient matrices · 191 biological pairs · one fixed 18-gene panel",
                ha="left", fontsize=6.8, color=MUTED)
    save(figure, ED, "figED13_bulk_patient_portraits")


def plot_perturbation_extended(mean: pd.DataFrame, donor: pd.DataFrame) -> None:
    donors = sorted(donor.donor.unique())
    conditions = ["IL-17", "TNF", "combination", "synergy residual"]
    column_order = [(d, c) for d in donors for c in conditions]
    wide = donor.pivot_table(index="gene", columns=["donor", "condition"], values="rank_shift")
    wide = wide.reindex(index=PANEL, columns=pd.MultiIndex.from_tuples(column_order))
    values = wide.to_numpy(float)
    vmax = max(0.25, float(np.nanquantile(np.abs(values), 0.985)))

    figure = plt.figure(figsize=(7.2, 6.4), facecolor=PAPER)
    grid = figure.add_gridspec(2, 3, height_ratios=[1.45, 1.0], left=0.12, right=0.97,
                              bottom=0.09, top=0.875, wspace=0.42, hspace=0.45)
    ax = figure.add_subplot(grid[0, :])
    image = ax.imshow(values, cmap=CYTOKINE, norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax),
                      aspect="auto", interpolation="nearest")
    ax.set_yticks(np.arange(len(PANEL)), PANEL)
    ax.set_xticks(np.arange(len(column_order)), [c.replace("synergy residual", "synergy") for _, c in column_order],
                  rotation=42, ha="right")
    for boundary in np.arange(1, len(donors)) * len(conditions) - 0.5:
        ax.axvline(boundary, color=PAPER, lw=2.0)
    for i, d in enumerate(donors):
        ax.text(i * len(conditions) + 1.5, -1.8, f"donor {d}", ha="center", va="bottom",
                fontsize=6.1, fontweight="bold", color=PERTURB)
    program_boundaries(ax, labels=True)
    ax.set_title("a  Donor-resolved response matrix", loc="left", fontsize=7.5, fontweight="bold")
    figure.colorbar(image, ax=ax, shrink=0.62, pad=0.015, label="paired rank shift")

    program_score = donor.assign(
        program=donor.gene.map({gene: program for program, genes in PROGRAMS.items() for gene in genes})
    ).groupby(["donor", "condition", "program"], as_index=False).rank_shift.mean()
    for column, program in enumerate(PROGRAMS):
        ax = figure.add_subplot(grid[1, column])
        sub = program_score[program_score.program == program]
        for donor_id, group in sub.groupby("donor"):
            group = group.set_index("condition").reindex(conditions)
            ax.plot(np.arange(4), group.rank_shift, "o-", lw=1.0, ms=3.0, alpha=0.80,
                    label=f"donor {donor_id}")
        ax.axhline(0, color=GRID, lw=0.8)
        ax.set_xticks(np.arange(4), ["IL-17", "TNF", "combo", "synergy"], rotation=28, ha="right")
        ax.set_title(f"{chr(98 + column)}  {program} program", loc="left", fontsize=7.0,
                     fontweight="bold", color=[KERATIN, LS, IMMUNE][column])
        ax.set_ylabel("mean paired rank shift" if column == 0 else "")
        if column == 2:
            ax.legend(frameon=False, fontsize=5.2, loc="best")
        ax.spines[["top", "right"]].set_visible(False)
    figure.suptitle("GSE24767 exposes donor-specific IL-17/TNF response structure",
                    x=0.055, y=0.975, ha="left", fontsize=11.0, fontweight="bold")
    figure.text(0.055, 0.938, "Observed keratinocyte perturbations · three complete donors · fixed genes shared with bulk cohorts",
                ha="left", fontsize=6.8, color=MUTED)
    save(figure, ED, "figED14_cytokine_perturbation_portrait")


def export_data(bulk: pd.DataFrame, metadata: pd.DataFrame, mean: pd.DataFrame, donor: pd.DataFrame,
                source: pd.DataFrame, target: pd.DataFrame, maps: list[dict]) -> None:
    bulk_long = []
    for cohort in COHORT_ACCENTS:
        values, ordered = ordered_bulk_panel(bulk, metadata, cohort)
        for patient_index, subject in enumerate(ordered.subject.astype(str)):
            for gene_index, gene in enumerate(PANEL):
                bulk_long.append({"cohort": cohort, "subject": subject, "gene": gene,
                                  "program": next(k for k, v in PROGRAMS.items() if gene in v),
                                  "fractional_rank_shift": float(values[gene_index, patient_index]),
                                  "display_order": patient_index + 1})
    pd.DataFrame(bulk_long).to_csv(CACHE / "bulk_patient_portrait_values.csv", index=False)
    mean.rename_axis("gene").reset_index().to_csv(CACHE / "gse24767_mean_portrait_values.csv", index=False)
    donor.to_csv(CACHE / "gse24767_donor_portrait_values.csv", index=False)
    manifest = {
        "seed": SEED,
        "descriptive_only": True,
        "fixed_panel": PROGRAMS,
        "bulk_pairs": metadata.groupby("cohort").size().astype(int).to_dict(),
        "single_cell_display": {"GSE151177": int(len(source)), "GSE173706": int(len(target))},
        "spatial_sections": [
            {"sample": m["sample"], "patient": m["patient"], "condition": m["condition"],
             "retained_spots": int(len(m["score"]))} for m in maps
        ],
        "perturbation_donors": sorted(donor.donor.unique().tolist()),
        "outputs": {
            "main": "fig1_eleven_dataset_visual_atlas",
            "extended_data": ["figED13_bulk_patient_portraits", "figED14_cytokine_perturbation_portrait"],
        },
    }
    (CACHE / "dataset_portrait_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    bulk, metadata = load_bulk()
    source, target = load_single_cell_maps()
    maps = spatial_maps()
    mean, donor = perturbation_portrait()
    plot_main_atlas(bulk, metadata, source, target, maps, mean)
    plot_bulk_extended(bulk, metadata)
    plot_perturbation_extended(mean, donor)
    export_data(bulk, metadata, mean, donor, source, target, maps)
    print((CACHE / "dataset_portrait_manifest.json").read_text())


if __name__ == "__main__":
    main()

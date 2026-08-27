"""Render supplementary publication figures from frozen v12 result files."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import auc, roc_curve


ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results/singlecell_v10"
PAPER = ROOT / "results/paper_v12"
OUT = ROOT / "figures/paper_v12/supplement"
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#0F4D92"
BLUE2 = "#3775BA"
GREEN = "#78B87A"
RED = "#B64342"
GRAY = "#BDBDBD"
GOLD = "#D9A441"
DARK = "#3F3F3F"
COLORS = [RED, BLUE2, GREEN]
np.random.seed(20260810)
sns.set_theme(style="whitegrid", font_scale=0.92)
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def js(path):
    return json.loads(Path(path).read_text())


def panel(ax, label):
    ax.text(
        -0.1,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
    )


def save(fig, name):
    for extension in ("png", "pdf", "svg"):
        fig.savefig(
            OUT / f"{name}.{extension}",
            dpi=320,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


# Figure S1: strict outer-fold ROC curves.
datasets = ["gse151177", "gse173706", "gse206391"]
labels = ["GSE151177", "GSE173706", "GSE206391"]
methods = [
    ("pseudobulk_probability", "Expression", RED),
    ("pca_probability", "PCA", BLUE2),
    ("nmf_probability", "NMF", GREEN),
]
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
for index, (stem, label) in enumerate(zip(datasets, labels)):
    frame = pd.read_csv(RES / f"{stem}_strict_benchmark_predictions.csv")
    for column, method, color in methods:
        fpr, tpr, _ = roc_curve(frame.true, frame[column])
        axes[index].plot(
            fpr,
            tpr,
            color=color,
            lw=2,
            label=f"{method} ({auc(fpr, tpr):.3f})",
        )
    axes[index].plot([0, 1], [0, 1], "--", color=GRAY)
    axes[index].set_title(label)
    axes[index].set_xlabel("False-positive rate")
    axes[index].set_ylabel("True-positive rate" if index == 0 else "")
    axes[index].legend(frameon=False, fontsize=8, loc="lower right")
    panel(axes[index], chr(ord("A") + index))
fig.suptitle("Strict outer-fold ROC curves", fontsize=15, fontweight="bold", y=1.03)
fig.tight_layout()
save(fig, "figS1_strict_roc")


# Figure S2: paired incremental-gain uncertainty.
gain = pd.read_csv(PAPER / "incremental_gain_bootstrap.csv")
gain["label"] = gain.dataset + " · " + gain.candidate
gain = gain.iloc[::-1].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(8.2, 5.7))
ypos = np.arange(len(gain))
color = [GREEN if value > 0 else (RED if value < 0 else GRAY) for value in gain.delta_auroc]
ax.errorbar(
    gain.delta_auroc,
    ypos,
    xerr=[
        gain.delta_auroc - gain.delta_ci95_low,
        gain.delta_ci95_high - gain.delta_auroc,
    ],
    fmt="none",
    ecolor=DARK,
    capsize=4,
)
ax.scatter(gain.delta_auroc, ypos, c=color, s=55, zorder=3)
ax.axvline(0, color=DARK, lw=1)
ax.axvline(0.05, color=GRAY, ls="--", lw=1.5)
ax.set_yticks(ypos, gain.label)
ax.set_xlabel("Paired bootstrap ΔAUROC versus matched expression")
ax.set_title("Uncertainty of incremental value")
ax.text(0.052, len(gain) - 0.55, "+0.05 threshold", color=DARK, fontsize=9)
fig.tight_layout()
save(fig, "figS2_incremental_gain_forest")


# Figure S3: strict score distributions and calibration ranges.
fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.7))
for index, (stem, label) in enumerate(zip(datasets, labels)):
    frame = pd.read_csv(RES / f"{stem}_strict_benchmark_predictions.csv")
    outcome = np.where(frame.true == 1, "Disease", "Reference")
    long = pd.DataFrame(
        {
            "outcome": np.tile(outcome, 3),
            "method": np.repeat(["Expression", "PCA", "NMF"], len(frame)),
            "probability": np.concatenate(
                [
                    frame.pseudobulk_probability,
                    frame.pca_probability,
                    frame.nmf_probability,
                ]
            ),
        }
    )
    sns.boxplot(
        data=long,
        x="method",
        y="probability",
        hue="outcome",
        palette={"Reference": GRAY, "Disease": RED},
        width=0.7,
        ax=axes[index],
    )
    axes[index].set_title(label)
    axes[index].set_xlabel("")
    axes[index].set_ylabel("Outer-fold probability" if index == 0 else "")
    if index != 2:
        axes[index].get_legend().remove()
    else:
        axes[index].legend(frameon=False, fontsize=8, loc="lower right")
    panel(axes[index], chr(ord("A") + index))
fig.suptitle(
    "Strict prediction distributions across modalities",
    fontsize=15,
    fontweight="bold",
    y=1.03,
)
fig.tight_layout()
save(fig, "figS3_prediction_distributions")


# Figure S4: neighborhood sensitivity and optimism gaps.
ablation = pd.read_csv(RES / "patient_balanced_neighborhood_ablation.csv")
strict = pd.DataFrame(js(RES / "strict_neighborhood_lodo_results.json"))
global_balanced = ablation[ablation.patient_balanced].sort_values("clusters")
global_unbalanced = ablation[~ablation.patient_balanced].sort_values("clusters")
merged = global_balanced.merge(strict, on="clusters")
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
axes[0].plot(
    global_balanced.clusters,
    global_balanced.loo_auroc,
    "o-",
    color=BLUE,
    label="Equal sample mass",
)
axes[0].plot(
    global_unbalanced.clusters,
    global_unbalanced.loo_auroc,
    "o-",
    color=GRAY,
    label="Cell-count weighted",
)
axes[0].set_title("Patient weighting")
axes[0].set_ylabel("LOO AUROC")
axes[0].legend(frameon=False, fontsize=8)
axes[1].plot(
    merged.clusters,
    merged.loo_auroc,
    "o-",
    color=RED,
    label="Global fit",
)
axes[1].plot(
    merged.clusters,
    merged.strict_loo_auroc,
    "o-",
    color=BLUE,
    label="Outer-fold refit",
)
axes[1].set_title("Unsupervised fit boundary")
axes[1].legend(frameon=False, fontsize=8)
gap = merged.loo_auroc - merged.strict_loo_auroc
axes[2].bar(merged.clusters.astype(str), gap, color=RED, edgecolor=DARK)
axes[2].set_title("Optimism gap")
axes[2].set_ylabel("Global − strict AUROC")
for index, ax in enumerate(axes):
    ax.set_xlabel("Neighborhoods")
    ax.set_xticks([8, 16, 32] if index < 2 else np.arange(3))
    panel(ax, chr(ord("A") + index))
fig.suptitle(
    "Cell-neighborhood sensitivity analysis",
    fontsize=15,
    fontweight="bold",
    y=1.03,
)
fig.tight_layout()
save(fig, "figS4_neighborhood_sensitivity")


# Figure S5: transport diagnostics.
v0 = pd.read_csv(RES / "triad_trigger_transport_v0_lodo.csv")
v0["condition"] = np.where(v0.true == 1, "Psoriasis", "Control")
flux = v0.melt(
    id_vars=["condition"],
    value_vars=["flux_il17", "flux_tnf", "flux_synergy"],
    var_name="trigger",
    value_name="flux",
)
flux["trigger"] = flux.trigger.str.replace("flux_", "", regex=False).str.upper()
summary = [
    js(RES / "strict_multiresolution_benchmarks.json")[0]["methods"]["pseudobulk"][
        "auroc"
    ],
    js(RES / "triad_trigger_transport_v0_summary.json")["loo_auroc"],
    js(RES / "triad_trigger_transport_v1_summary.json")["loo_auroc"],
    js(RES / "triad_fused_gw_v2_summary.json")["loo_auroc"],
]
fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5))
sns.boxplot(
    data=v0,
    x="condition",
    y="cost",
    hue="condition",
    palette={"Control": GRAY, "Psoriasis": RED},
    legend=False,
    ax=axes[0],
)
sns.stripplot(data=v0, x="condition", y="cost", color=DARK, size=4, ax=axes[0])
axes[0].set_title("Trigger-only transport cost")
axes[0].set_xlabel("")
sns.boxplot(
    data=flux,
    x="trigger",
    y="flux",
    hue="condition",
    palette={"Control": GRAY, "Psoriasis": RED},
    ax=axes[1],
)
axes[1].axhline(0, color=DARK, lw=1)
axes[1].set_title("Trigger flux distributions")
axes[1].set_xlabel("")
axes[1].legend(frameon=False, fontsize=8)
axes[2].bar(
    ["Expression", "Trigger OT", "State+trigger", "Fused GW"],
    summary,
    color=[RED, BLUE2, BLUE, GOLD],
    edgecolor=DARK,
)
axes[2].axhline(0.5, color=GRAY, ls="--")
axes[2].tick_params(axis="x", rotation=25)
axes[2].set_ylim(0, 1.03)
axes[2].set_ylabel("LOO AUROC")
axes[2].set_title("Complexity trajectory")
for index, ax in enumerate(axes):
    panel(ax, chr(ord("A") + index))
fig.suptitle("Transport-model diagnostics", fontsize=15, fontweight="bold", y=1.03)
fig.tight_layout()
save(fig, "figS5_transport_diagnostics")


# Figure S6: strict spatial graph diagnostics.
spatial = pd.read_csv(RES / "gse206391_strict_spatial_graph_predictions.csv")
spatial_summary = js(RES / "gse206391_strict_spatial_graph_benchmark.json")["methods"]
spatial_long = spatial.melt(
    id_vars=["patient", "condition", "true"],
    value_vars=[
        "expression_500_probability",
        "spatial_graph_probability",
        "expression_plus_spatial_probability",
    ],
    var_name="method",
    value_name="probability",
)
spatial_long["method"] = spatial_long.method.map(
    {
        "expression_500_probability": "Expression",
        "spatial_graph_probability": "Spatial graph",
        "expression_plus_spatial_probability": "Expression + graph",
    }
)
spatial_long["outcome"] = np.where(spatial_long.true == 1, "Lesional", "Non-lesional")
patient_mean = (
    spatial_long.groupby(["patient", "method", "outcome"]).probability.mean().unstack()
)
patient_mean["paired_difference"] = patient_mean["Lesional"] - patient_mean["Non-lesional"]
patient_mean = patient_mean.reset_index()
method_order = ["Expression", "Spatial graph", "Expression + graph"]
fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5))
sns.boxplot(
    data=spatial_long,
    x="method",
    y="probability",
    hue="outcome",
    palette={"Non-lesional": GRAY, "Lesional": RED},
    order=method_order,
    ax=axes[0],
)
axes[0].tick_params(axis="x", rotation=20)
axes[0].set_title("Outer-fold predictions")
axes[0].set_xlabel("")
axes[0].legend(frameon=False, fontsize=8)
names = ["expression_500", "spatial_graph", "expression_plus_spatial"]
values = [spatial_summary[name]["auroc"] for name in names]
ci = np.array([spatial_summary[name]["group_bootstrap_ci95"] for name in names])
axes[1].bar(["Expression", "Spatial graph", "Expression + graph"], values, color=COLORS, edgecolor=DARK)
axes[1].errorbar(
    np.arange(3),
    values,
    yerr=[np.asarray(values) - ci[:, 0], ci[:, 1] - np.asarray(values)],
    fmt="none",
    color=DARK,
    capsize=4,
)
axes[1].tick_params(axis="x", rotation=20)
axes[1].set_ylim(0.65, 1.02)
axes[1].set_ylabel("Patient LODO AUROC")
axes[1].set_title("Matched-gene comparison")
sns.boxplot(
    data=patient_mean,
    x="method",
    y="paired_difference",
    order=method_order,
    color="#AADCA9",
    ax=axes[2],
)
sns.stripplot(
    data=patient_mean,
    x="method",
    y="paired_difference",
    order=method_order,
    color=DARK,
    size=4,
    ax=axes[2],
)
axes[2].axhline(0, color=DARK, lw=1)
axes[2].tick_params(axis="x", rotation=20)
axes[2].set_ylabel("Lesional − non-lesional score")
axes[2].set_xlabel("")
axes[2].set_title("Within-patient separation")
for index, ax in enumerate(axes):
    panel(ax, chr(ord("A") + index))
fig.suptitle("Strict spatial-graph diagnostics", fontsize=15, fontweight="bold", y=1.03)
fig.tight_layout()
save(fig, "figS6_spatial_graph_diagnostics")


# Figure S7: bulk cross-cohort stability.
bulk = pd.read_csv(ROOT / "results/trigger_v9/v9_tuned_pooled_results.csv")
cohort_size = (
    pd.read_csv(ROOT / "results/trigger_v9/v9_expanded_metadata.csv")
    .groupby("cohort")
    .size()
    .rename("pairs")
)
bulk = bulk.join(cohort_size, on="cohort")
fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
sns.barplot(data=bulk, x="cohort", y="held_gene_cosine", color=BLUE2, edgecolor=DARK, ax=axes[0])
axes[0].tick_params(axis="x", rotation=55)
axes[0].set_xlabel("")
axes[0].set_ylabel("Held-cohort gene cosine")
axes[0].set_title("Transfer by cohort")
sns.barplot(data=bulk, x="cohort", y="pairs", color=GRAY, edgecolor=DARK, ax=axes[1])
axes[1].tick_params(axis="x", rotation=55)
axes[1].set_xlabel("")
axes[1].set_ylabel("LS/NL pairs")
axes[1].set_title("Cohort contribution")
axes[2].scatter(bulk.inner, bulk.held_gene_cosine, s=55, color=BLUE)
for _, row in bulk.iterrows():
    axes[2].annotate(row.cohort.replace("GSE", ""), (row.inner, row.held_gene_cosine), xytext=(3, 3), textcoords="offset points", fontsize=7)
axes[2].set_xlabel("Inner-CV score")
axes[2].set_ylabel("Held-cohort gene cosine")
axes[2].set_title("Selection versus transfer")
for index, ax in enumerate(axes):
    panel(ax, chr(ord("A") + index))
fig.suptitle("Seven-cohort bulk robustness", fontsize=15, fontweight="bold", y=1.03)
fig.tight_layout()
save(fig, "figS7_bulk_robustness")

print(f"wrote 7 supplementary figures to {OUT}")

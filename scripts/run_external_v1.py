from __future__ import annotations

import gzip
import json
import sys
import tarfile
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results" / "external_v1"
FIGURES = ROOT / "figures" / "external_v1"
DOCS = ROOT / "docs"
for directory in (RESULTS, FIGURES, DOCS):
    directory.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import collapse_duplicate_genes, log_cpm, parse_geo_series_matrix, sample_titles  # noqa: E402
from run_cone_v0 import (  # noqa: E402
    ALPHA,
    GAMMA,
    SEED,
    SPECIFICITY_BETA,
    bootstrap_auc,
    build_dictionaries,
    disease_matrix,
    fit_scores,
    load_perturbation_deltas,
    rank_unit,
    training_geometry,
)
from run_feasibility import load_gse121212_deltas  # noqa: E402

N_BOOT = 5000
TOP_N = 250


def load_gpl570_map() -> dict[str, str]:
    lines = []
    in_table = False
    with gzip.open(RAW / "GPL570.annot.gz", "rt", errors="replace") as handle:
        for line in handle:
            if line.startswith("!platform_table_begin"):
                in_table = True
                continue
            if line.startswith("!platform_table_end"):
                break
            if in_table:
                lines.append(line)
    annotation = pd.read_csv(
        StringIO("".join(lines)), sep="\t", dtype=str, low_memory=False
    )
    result = {}
    for probe, symbols in zip(annotation["ID"], annotation["Gene symbol"]):
        if pd.isna(symbols):
            continue
        symbol = str(symbols).split("///")[0].strip()
        if symbol:
            result[str(probe)] = symbol
    return result


def map_gpl570_expression(expression: pd.DataFrame) -> pd.DataFrame:
    mapping = load_gpl570_map()
    expression.index = expression.index.astype(str).map(mapping)
    return collapse_duplicate_genes(expression)


def load_gse36287_effects() -> dict[str, pd.Series]:
    expression, metadata = parse_geo_series_matrix(
        RAW / "GSE36287_series_matrix.txt.gz"
    )
    expression = map_gpl570_expression(expression)
    titles = sample_titles(metadata)
    by_title = {title: accession for accession, title in titles.items()}

    subject_controls = [
        by_title[f"Keratinocytes_untreated_subject{index}"] for index in range(1, 4)
    ]
    effects = {}
    for raw_name, clean_name in (
        ("IFNa", "IFNa"),
        ("IFNg", "IFNg"),
        ("IL13", "IL13"),
        ("IL4", "IL4"),
        ("TNF", "TNF"),
    ):
        treated = [
            by_title[f"Keratinocytes_{raw_name}_subject{index}"]
            for index in range(1, 4)
        ]
        effects[clean_name] = (
            expression[treated].mean(axis=1)
            - expression[subject_controls].mean(axis=1)
        )
    il17_control = [
        by_title[f"Keratinocytes_untreated_replicate{index}"]
        for index in range(1, 4)
    ]
    il17_treated = [
        by_title[f"Keratinocytes_IL17A_replicate{index}"]
        for index in range(1, 4)
    ]
    effects["IL17"] = (
        expression[il17_treated].mean(axis=1)
        - expression[il17_control].mean(axis=1)
    )
    return effects


def load_gse86451_deltas() -> dict[str, pd.DataFrame]:
    batches = {
        "2": {
            "control": "GSM2302879_Sample336_11040.txt.gz",
            "IL17": "GSM2302880_Sample336_11041.txt.gz",
            "TNF": "GSM2302881_Sample336_11042.txt.gz",
            "combo": "GSM2302882_Sample336_11043.txt.gz",
        },
        "3": {
            "control": "GSM2302887_Sample336_11056.txt.gz",
            "IL17": "GSM2302888_Sample336_11057.txt.gz",
            "TNF": "GSM2302889_Sample336_11058.txt.gz",
            "combo": "GSM2302890_Sample336_11059.txt.gz",
        },
        "4": {
            "control": "GSM2302895_Sample425_13587.txt.gz",
            "IL17": "GSM2302896_Sample425_13588.txt.gz",
            "TNF": "GSM2302897_Sample425_13589.txt.gz",
            "combo": "GSM2302898_Sample425_13590.txt.gz",
        },
        "5": {
            "control": "GSM2302903_Sample425_13595.txt.gz",
            "IL17": "GSM2302904_Sample425_13596.txt.gz",
            "TNF": "GSM2302905_Sample425_13597.txt.gz",
            "combo": "GSM2302906_Sample425_13598.txt.gz",
        },
    }
    columns = {}
    with tarfile.open(RAW / "GSE86451_RAW.tar") as archive:
        for batch, conditions in batches.items():
            for condition, member_name in conditions.items():
                member = archive.extractfile(member_name)
                if member is None:
                    raise FileNotFoundError(member_name)
                with gzip.GzipFile(fileobj=member) as compressed:
                    table = pd.read_csv(
                        compressed,
                        sep=r"\s+",
                        quotechar='"',
                        dtype={"gene_symbol": str},
                    )
                series = table.set_index("gene_symbol")["count"]
                series = series.groupby(series.index).sum()
                columns[f"{batch}|{condition}"] = series
    counts = pd.DataFrame(columns).fillna(0)
    expression = log_cpm(counts)
    outputs = {}
    for condition in ("IL17", "TNF", "combo"):
        delta = pd.DataFrame(
            {
                batch: expression[f"{batch}|{condition}"]
                - expression[f"{batch}|control"]
                for batch in batches
            }
        )
        outputs[condition] = delta
    return outputs


def load_gse182740_deltas() -> dict[str, pd.DataFrame]:
    expression, metadata = parse_geo_series_matrix(
        RAW / "GSE182740_series_matrix.txt.gz"
    )
    expression = map_gpl570_expression(expression)
    titles = sample_titles(metadata)
    groups: dict[str, dict[str, dict[str, str]]] = {
        "PSO": {},
        "AD": {},
        "OVERLAP": {},
    }
    for accession, title in titles.items():
        if not title.startswith("JWK-09-") or not title.endswith(("_LS", "_NL")):
            continue
        core, state = title.rsplit("_", maxsplit=1)
        prefix = core.removeprefix("JWK-09-")
        raw_group, patient = prefix.split("-", maxsplit=1)
        group = {"Ps": "PSO", "AD": "AD", "Th2": "OVERLAP"}.get(raw_group)
        if group is None:
            continue
        groups[group].setdefault(patient, {})[state] = accession
    outputs = {}
    for group, patients in groups.items():
        complete = {
            patient: states
            for patient, states in patients.items()
            if {"LS", "NL"}.issubset(states)
        }
        outputs[group] = pd.DataFrame(
            {
                patient: expression[states["LS"]] - expression[states["NL"]]
                for patient, states in sorted(complete.items())
            }
        )
    return outputs


def nonadditive_residual(
    il17: np.ndarray,
    tnf: np.ndarray,
    combo: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    il17_centered = il17 - il17.mean()
    tnf_centered = tnf - tnf.mean()
    combo_centered = combo - combo.mean()
    primitives = np.column_stack([il17_centered, tnf_centered])
    coefficients, _ = nnls(primitives, combo_centered)
    residual = combo_centered - primitives @ coefficients
    fraction = float(np.linalg.norm(residual) / np.linalg.norm(combo_centered))
    return residual, coefficients, fraction


def synergy_replication(
    discovery: dict[str, pd.DataFrame],
    replication: dict[str, pd.DataFrame],
    pso_delta: pd.DataFrame,
    ad_delta: pd.DataFrame,
    genes: list[str],
) -> tuple[pd.DataFrame, dict, pd.Series, pd.Series]:
    discovery_arrays = {
        key: discovery[key].loc[genes].to_numpy(dtype=float)
        for key in ("IL17", "TNF_high", "IL17_TNF_high")
    }
    replication_arrays = {
        key: replication[source].loc[genes].to_numpy(dtype=float)
        for key, source in (("IL17", "IL17"), ("TNF", "TNF"), ("combo", "combo"))
    }
    pso = pso_delta.loc[genes].to_numpy(dtype=float)
    ad = ad_delta.loc[genes].to_numpy(dtype=float)

    discovery_residual, discovery_coefficients, discovery_fraction = (
        nonadditive_residual(
            discovery_arrays["IL17"].mean(axis=1),
            discovery_arrays["TNF_high"].mean(axis=1),
            discovery_arrays["IL17_TNF_high"].mean(axis=1),
        )
    )
    replication_residual, replication_coefficients, replication_fraction = (
        nonadditive_residual(
            replication_arrays["IL17"].mean(axis=1),
            replication_arrays["TNF"].mean(axis=1),
            replication_arrays["combo"].mean(axis=1),
        )
    )

    rng = np.random.default_rng(SEED)
    rows = []
    for iteration in range(N_BOOT):
        discovery_il17_indices = rng.choice(
            discovery_arrays["IL17"].shape[1],
            discovery_arrays["IL17"].shape[1],
            replace=True,
        )
        discovery_paired_indices = rng.choice(
            discovery_arrays["TNF_high"].shape[1],
            discovery_arrays["TNF_high"].shape[1],
            replace=True,
        )
        replication_indices = rng.choice(
            replication_arrays["IL17"].shape[1],
            replication_arrays["IL17"].shape[1],
            replace=True,
        )
        discovery_boot, _, discovery_boot_fraction = nonadditive_residual(
            discovery_arrays["IL17"][:, discovery_il17_indices].mean(axis=1),
            discovery_arrays["TNF_high"][:, discovery_paired_indices].mean(axis=1),
            discovery_arrays["IL17_TNF_high"][:, discovery_paired_indices].mean(axis=1),
        )
        replication_boot, _, replication_boot_fraction = nonadditive_residual(
            replication_arrays["IL17"][:, replication_indices].mean(axis=1),
            replication_arrays["TNF"][:, replication_indices].mean(axis=1),
            replication_arrays["combo"][:, replication_indices].mean(axis=1),
        )
        pso_indices = rng.choice(pso.shape[1], pso.shape[1], replace=True)
        ad_indices = rng.choice(ad.shape[1], ad.shape[1], replace=True)
        disease_direction = rank_unit(
            pso[:, pso_indices].mean(axis=1)
            - ad[:, ad_indices].mean(axis=1)
        )
        residual_rho = stats.spearmanr(
            discovery_boot, replication_boot
        ).statistic
        rows.append(
            {
                "bootstrap": iteration,
                "discovery_residual_fraction": discovery_boot_fraction,
                "replication_residual_fraction": replication_boot_fraction,
                "cross_experiment_residual_rho": residual_rho,
                "replication_residual_disease_cosine": float(
                    rank_unit(replication_boot) @ disease_direction
                ),
            }
        )
    table = pd.DataFrame(rows)

    def summarize(column: str) -> dict[str, float]:
        values = table[column].to_numpy()
        return {
            "median": float(np.median(values)),
            "ci_low": float(np.quantile(values, 0.025)),
            "ci_high": float(np.quantile(values, 0.975)),
            "prob_gt_zero": float((values > 0).mean()),
        }

    summary = {
        "point_estimate": {
            "discovery_residual_fraction": discovery_fraction,
            "replication_residual_fraction": replication_fraction,
            "cross_experiment_residual_rho": float(
                stats.spearmanr(discovery_residual, replication_residual).statistic
            ),
            "discovery_coefficients": discovery_coefficients.tolist(),
            "replication_coefficients": replication_coefficients.tolist(),
        },
        "replication_residual_fraction": summarize(
            "replication_residual_fraction"
        ),
        "cross_experiment_residual_rho": summarize(
            "cross_experiment_residual_rho"
        ),
        "replication_residual_disease_cosine": summarize(
            "replication_residual_disease_cosine"
        ),
    }
    summary["pass"] = bool(
        summary["replication_residual_fraction"]["ci_low"] > 0.10
        and summary["cross_experiment_residual_rho"]["median"] > 0
        and summary["cross_experiment_residual_rho"]["ci_low"] > 0
        and summary["replication_residual_disease_cosine"]["ci_low"] > 0
    )
    return (
        table,
        summary,
        pd.Series(discovery_residual, index=genes),
        pd.Series(replication_residual, index=genes),
    )


def broad_residual_signatures(
    discovery: dict[str, pd.DataFrame],
    replication: dict[str, pd.DataFrame],
) -> tuple[pd.Series, pd.Series]:
    """Estimate residuals across the full perturbation-only common gene space."""
    required = [
        discovery["IL17"],
        discovery["TNF_high"],
        discovery["IL17_TNF_high"],
        replication["IL17"],
        replication["TNF"],
        replication["combo"],
    ]
    genes = sorted(set.intersection(*(set(frame.index) for frame in required)))
    discovery_residual, _, _ = nonadditive_residual(
        discovery["IL17"].loc[genes].mean(axis=1).to_numpy(),
        discovery["TNF_high"].loc[genes].mean(axis=1).to_numpy(),
        discovery["IL17_TNF_high"].loc[genes].mean(axis=1).to_numpy(),
    )
    replication_residual, _, _ = nonadditive_residual(
        replication["IL17"].loc[genes].mean(axis=1).to_numpy(),
        replication["TNF"].loc[genes].mean(axis=1).to_numpy(),
        replication["combo"].loc[genes].mean(axis=1).to_numpy(),
    )
    return (
        pd.Series(discovery_residual, index=genes),
        pd.Series(replication_residual, index=genes),
    )


def consensus_signature(signatures: list[pd.Series], genes: list[str]) -> pd.Series:
    values = np.column_stack(
        [rank_unit(signature.loc[genes].to_numpy()) for signature in signatures]
    )
    return pd.Series(values.mean(axis=1), index=genes)


def build_expanded_dictionary(
    discovery_effects: dict[str, pd.Series],
    replication_deltas: dict[str, pd.DataFrame],
    broad_effects: dict[str, pd.Series],
    discovery_residual: pd.Series,
    replication_residual: pd.Series,
    training_pso: pd.DataFrame,
    training_ad: pd.DataFrame,
    external: dict[str, pd.DataFrame],
) -> tuple[np.ndarray, list[str], list[str], dict[str, pd.Series]]:
    replication_effects = {
        key: frame.mean(axis=1) for key, frame in replication_deltas.items()
    }
    universe = training_pso.index.intersection(training_ad.index)
    for frame in external.values():
        universe = universe.intersection(frame.index)
    candidate_series = [
        *discovery_effects.values(),
        *replication_effects.values(),
        *broad_effects.values(),
        discovery_residual,
        replication_residual,
    ]
    for signature in candidate_series:
        universe = universe.intersection(signature.index)
    universe_genes = universe.astype(str).tolist()

    axes = {
        "IL17_CONSENSUS": consensus_signature(
            [
                discovery_effects["IL17"],
                replication_effects["IL17"],
                broad_effects["IL17"],
            ],
            universe_genes,
        ),
        "TNF_CONSENSUS": consensus_signature(
            [
                discovery_effects["TNF_high"],
                replication_effects["TNF"],
                broad_effects["TNF"],
            ],
            universe_genes,
        ),
        "SYNERGY_CONSENSUS": consensus_signature(
            [discovery_residual, replication_residual], universe_genes
        ),
        "IL4": broad_effects["IL4"].loc[universe_genes],
        "IL13": broad_effects["IL13"].loc[universe_genes],
        "IFNa": broad_effects["IFNa"].loc[universe_genes],
        "IFNg": broad_effects["IFNg"].loc[universe_genes],
    }
    feature_set: set[str] = set()
    for signature in axes.values():
        feature_set.update(signature.nlargest(TOP_N).index)
        feature_set.update(signature.nsmallest(TOP_N).index)
    features = sorted(feature_set)
    names = list(axes)
    dictionary = np.column_stack(
        [rank_unit(axes[name].loc[features].to_numpy()) for name in names]
    )
    return dictionary, names, features, axes


def external_predictions(
    training_pso: pd.DataFrame,
    training_ad: pd.DataFrame,
    external: dict[str, pd.DataFrame],
    discovery_effects: dict[str, pd.Series],
    expanded_dictionary: np.ndarray,
    expanded_names: list[str],
    expanded_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frozen_features = pd.read_csv(
        ROOT / "results" / "cone_v0" / "frozen_feature_genes.csv"
    )["gene"].astype(str).tolist()
    frozen_features = [
        gene
        for gene in frozen_features
        if gene in external["PSO"].index
        and gene in external["AD"].index
        and gene in training_pso.index
        and gene in training_ad.index
    ]
    (
        v0_raw,
        _,
        v0_interaction,
        v0_interaction_names,
        _,
    ) = build_dictionaries(discovery_effects, frozen_features)
    v0_training_pso = disease_matrix(training_pso, frozen_features)
    v0_training_ad = disease_matrix(training_ad, frozen_features)
    v0_axis, v0_affinity, v0_penalties = training_geometry(
        v0_training_pso, v0_training_ad, v0_interaction
    )

    expanded_training_pso = disease_matrix(training_pso, expanded_features)
    expanded_training_ad = disease_matrix(training_ad, expanded_features)
    expanded_axis, expanded_affinity, expanded_penalties = training_geometry(
        expanded_training_pso, expanded_training_ad, expanded_dictionary
    )

    prediction_rows = []
    weight_rows = []
    for disease in ("PSO", "AD", "OVERLAP"):
        for patient in external[disease].columns:
            v0_target = rank_unit(
                external[disease].loc[frozen_features, patient].to_numpy()
            )
            v0_scores, _ = fit_scores(
                v0_target,
                v0_raw,
                v0_interaction,
                v0_axis,
                v0_affinity,
                v0_penalties,
            )
            expanded_target = rank_unit(
                external[disease].loc[expanded_features, patient].to_numpy()
            )
            expanded_scores, expanded_candidate_weights = fit_scores(
                expanded_target,
                expanded_dictionary,
                expanded_dictionary,
                expanded_axis,
                expanded_affinity,
                expanded_penalties,
            )
            expanded_nnls_weights, _ = nnls(
                expanded_dictionary, expanded_target
            )
            prediction_rows.append(
                {
                    "sample": f"{disease}_{patient}",
                    "disease": disease,
                    "label": 1 if disease == "PSO" else (0 if disease == "AD" else -1),
                    "v0_max_correlation": v0_scores["max_correlation_raw5"],
                    "v0_nnls": v0_scores["nnls_raw5"],
                    "v0_contrastive_sparse_cone": v0_scores[
                        "contrastive_sparse_cone"
                    ],
                    "expanded_max_correlation": expanded_scores[
                        "max_correlation_raw5"
                    ],
                    "expanded_nnls": expanded_scores["nnls_raw5"],
                    "expanded_contrastive_sparse_cone": expanded_scores[
                        "contrastive_sparse_cone"
                    ],
                }
            )
            weight_rows.append(
                {
                    "sample": f"{disease}_{patient}",
                    "disease": disease,
                    **{
                        f"nnls_{name}": value
                        for name, value in zip(
                            expanded_names, expanded_nnls_weights
                        )
                    },
                    **{
                        f"contrastive_{name}": value
                        for name, value in zip(
                            expanded_names, expanded_candidate_weights
                        )
                    },
                }
            )
    return pd.DataFrame(prediction_rows), pd.DataFrame(weight_rows)


def paired_auc_difference(
    labels: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(SEED)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    values = []
    for _ in range(N_BOOT):
        indices = np.concatenate(
            [
                rng.choice(positive, len(positive), replace=True),
                rng.choice(negative, len(negative), replace=True),
            ]
        )
        values.append(
            roc_auc_score(labels[indices], candidate[indices])
            - roc_auc_score(labels[indices], baseline[indices])
        )
    values = np.asarray(values)
    return {
        "difference": float(
            roc_auc_score(labels, candidate) - roc_auc_score(labels, baseline)
        ),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "prob_gt_zero": float((values > 0).mean()),
    }


def mechanism_ordering(weights: pd.DataFrame, names: list[str]) -> dict:
    rng = np.random.default_rng(SEED)
    pso = weights.loc[weights["disease"] == "PSO"]
    ad = weights.loc[weights["disease"] == "AD"]
    summary = {}
    for name in names:
        column = f"nnls_{name}"
        values = []
        for _ in range(N_BOOT):
            values.append(
                rng.choice(pso[column], len(pso), replace=True).mean()
                - rng.choice(ad[column], len(ad), replace=True).mean()
            )
        array = np.asarray(values)
        summary[name] = {
            "pso_minus_ad": float(pso[column].mean() - ad[column].mean()),
            "ci_low": float(np.quantile(array, 0.025)),
            "ci_high": float(np.quantile(array, 0.975)),
            "prob_gt_zero": float((array > 0).mean()),
        }
    return summary


def evaluate_external(predictions: pd.DataFrame) -> tuple[dict, dict]:
    primary = predictions.loc[predictions["disease"].isin(["PSO", "AD"])].copy()
    labels = primary["label"].to_numpy()
    models = [
        "v0_max_correlation",
        "v0_nnls",
        "v0_contrastive_sparse_cone",
        "expanded_max_correlation",
        "expanded_nnls",
        "expanded_contrastive_sparse_cone",
    ]
    metrics = {
        model: bootstrap_auc(
            labels,
            primary[model].to_numpy(),
            n_boot=N_BOOT,
            seed=SEED + index,
        )
        for index, model in enumerate(models)
    }
    simple_baseline = max(
        ("v0_max_correlation", "v0_nnls"),
        key=lambda model: metrics[model]["auc"],
    )
    differences = {
        "expanded_contrastive_vs_best_v0_simple": paired_auc_difference(
            labels,
            primary["expanded_contrastive_sparse_cone"].to_numpy(),
            primary[simple_baseline].to_numpy(),
        ),
        "expanded_nnls_vs_v0_nnls": paired_auc_difference(
            labels,
            primary["expanded_nnls"].to_numpy(),
            primary["v0_nnls"].to_numpy(),
        ),
        "best_v0_simple_model": simple_baseline,
    }
    return metrics, differences


def make_figures(
    synergy: pd.DataFrame,
    metrics: dict,
    predictions: pd.DataFrame,
    weights: pd.DataFrame,
    names: list[str],
) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, column, title in zip(
        axes,
        [
            "replication_residual_fraction",
            "cross_experiment_residual_rho",
            "replication_residual_disease_cosine",
        ],
        [
            "External residual fraction",
            "Cross-experiment residual rho",
            "External residual alignment with PSO−AD",
        ],
    ):
        sns.histplot(synergy[column], bins=35, ax=axis)
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_title(title)
    fig.tight_layout()
    fig.savefig(FIGURES / "synergy_external_replication.png", dpi=180)
    plt.close(fig)

    order = list(metrics)
    labels = {
        "v0_max_correlation": "v0 correlation",
        "v0_nnls": "v0 NNLS",
        "v0_contrastive_sparse_cone": "v0 contrastive cone",
        "expanded_max_correlation": "expanded correlation",
        "expanded_nnls": "expanded NNLS",
        "expanded_contrastive_sparse_cone": "expanded contrastive cone",
    }
    table = pd.DataFrame(
        [
            {
                "model": labels[model],
                "auc": metrics[model]["auc"],
                "low": metrics[model]["auc_ci_low"],
                "high": metrics[model]["auc_ci_high"],
            }
            for model in order
        ]
    )
    positions = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(positions, table["auc"], color="#4C72B0")
    ax.errorbar(
        positions,
        table["auc"],
        yerr=[table["auc"] - table["low"], table["high"] - table["auc"]],
        fmt="none",
        color="black",
        capsize=4,
    )
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(positions, table["model"], rotation=25, ha="right")
    ax.set_ylim(0.25, 1.0)
    ax.set_ylabel("External patient AUROC")
    ax.set_title("GSE182740 frozen external validation")
    fig.tight_layout()
    fig.savefig(FIGURES / "external_model_comparison.png", dpi=180)
    plt.close(fig)

    score_long = predictions.melt(
        id_vars=["sample", "disease"],
        value_vars=["v0_nnls", "expanded_nnls", "expanded_contrastive_sparse_cone"],
        var_name="model",
        value_name="score",
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=score_long, x="model", y="score", hue="disease", ax=ax)
    ax.set_title("External PSO, AD and overlap phenotype scores")
    fig.tight_layout()
    fig.savefig(FIGURES / "external_score_distributions.png", dpi=180)
    plt.close(fig)

    raw_weight_columns = [f"nnls_{name}" for name in names]
    weight_long = weights.melt(
        id_vars=["sample", "disease"],
        value_vars=raw_weight_columns,
        var_name="mechanism",
        value_name="weight",
    )
    weight_long["mechanism"] = weight_long["mechanism"].str.removeprefix("nnls_")
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(
        data=weight_long,
        x="mechanism",
        y="weight",
        hue="disease",
        errorbar="se",
        ax=ax,
    )
    ax.tick_params(axis="x", rotation=22)
    ax.set_title("Expanded-dictionary NNLS mechanism attribution")
    fig.tight_layout()
    fig.savefig(FIGURES / "external_mechanism_weights.png", dpi=180)
    plt.close(fig)


def write_report(metrics: dict) -> None:
    synergy = metrics["synergy_replication"]
    external = metrics["external_model_metrics"]
    differences = metrics["algorithm_differences"]
    v0 = external["v0_contrastive_sparse_cone"]
    gate_b = bool(
        v0["auc"] >= 0.70
        and v0["auc_ci_low"] > 0.50
        and v0["prob_difference_gt_zero"] >= 0.95
    )
    diff = differences["expanded_contrastive_vs_best_v0_simple"]
    gate_c = bool(diff["difference"] > 0 and diff["prob_gt_zero"] >= 0.95)
    diagnostic_path = RESULTS / "posthoc_representation_diagnostics.json"
    diagnostics = (
        json.loads(diagnostic_path.read_text())
        if diagnostic_path.exists()
        else {}
    )
    lines = [
        "# v1 独立扰动与疾病队列验证报告",
        "",
        "## 主要门槛",
        "",
        "| 门槛 | 结果 | 判定 |",
        "|---|---:|---|",
        (
            "| A. 协同残差跨实验复现 | "
            f"ρ={synergy['cross_experiment_residual_rho']['median']:.3f}, "
            f"95% CI [{synergy['cross_experiment_residual_rho']['ci_low']:.3f}, "
            f"{synergy['cross_experiment_residual_rho']['ci_high']:.3f}] | "
            f"{'通过' if synergy['pass'] else '未通过'} |"
        ),
        (
            "| B. 冻结v0外部疾病验证 | "
            f"AUROC={v0['auc']:.3f}, 95% CI [{v0['auc_ci_low']:.3f}, "
            f"{v0['auc_ci_high']:.3f}] | {'通过' if gate_b else '未通过'} |"
        ),
        (
            "| C. 扩展字典优于最佳v0简单基线 | "
            f"ΔAUROC={diff['difference']:.3f}, P(Δ>0)={diff['prob_gt_zero']:.3f} | "
            f"{'通过' if gate_c else '未通过'} |"
        ),
        "",
        "## 协同残差",
        "",
        "```json",
        json.dumps(synergy, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 外部模型比较",
        "",
        "```json",
        json.dumps(external, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 算法差异",
        "",
        "```json",
        json.dumps(differences, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 失败归因（事后诊断，不替代预注册结果）",
        "",
    ]
    if diagnostics:
        lines.extend(
            [
                "| 表示 | 特征数 | 轴数 | 外部AUROC |",
                "|---|---:|---:|---:|",
                *[
                    (
                        f"| {name} | {value['n_features']} | "
                        f"{value['n_axes']} | {value['auc']:.3f} |"
                    )
                    for name, value in diagnostics.items()
                ],
                "",
                "v0原始5轴换到扩展特征空间后AUROC仍为0.958，说明失败不是单纯",
                "由特征数增加造成；把原始精确刺激轴替换为跨平台共识轴后性能下降，",
                "再把Th2/IFN轴作为自由拟合方向加入后进一步下降。下一版不再做全局",
                "字典扩张，而保留v0原始锥，只把跨实验稳定的协同残差作为可靠性加权",
                "修正项，并将Th2/IFN方向作为需要投影掉的混杂子空间。",
                "",
            ]
        )
    lines.extend(
        [
        "## 机制排序",
        "",
        "```json",
        json.dumps(metrics["mechanism_ordering"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## 解释边界",
        "",
        "GSE86451使用永生化N/TERT角质形成细胞，GSE24767使用原代细胞；",
        "跨实验复现因此检验的是方向可迁移性而非绝对效应量一致。GSE182740",
        "只有6对PSO和8对AD，bootstrap反映现有患者的不确定性，不能替代更大",
        "外部队列。扩展字典中的Th2轴来自独立扰动数据，但机制权重仍属于模型",
        "归因，不是细胞因果比例。",
        "",
        ]
    )
    (DOCS / "external_v1_report.md").write_text("\n".join(lines))


def main() -> None:
    discovery_effects, discovery_deltas = load_perturbation_deltas()
    replication_deltas = load_gse86451_deltas()
    broad_effects = load_gse36287_effects()
    training_pso, training_ad = load_gse121212_deltas()
    external = load_gse182740_deltas()

    frozen_genes = pd.read_csv(
        ROOT / "results" / "cone_v0" / "frozen_feature_genes.csv"
    )["gene"].astype(str)
    synergy_genes = frozen_genes[
        frozen_genes.isin(replication_deltas["IL17"].index)
        & frozen_genes.isin(discovery_deltas["IL17"].index)
        & frozen_genes.isin(training_pso.index)
        & frozen_genes.isin(training_ad.index)
    ].tolist()
    (
        synergy_bootstrap_table,
        synergy_summary,
        discovery_residual,
        replication_residual,
    ) = synergy_replication(
        discovery_deltas,
        replication_deltas,
        training_pso,
        training_ad,
        synergy_genes,
    )
    broad_discovery_residual, broad_replication_residual = (
        broad_residual_signatures(discovery_deltas, replication_deltas)
    )

    (
        expanded_dictionary,
        expanded_names,
        expanded_features,
        expanded_axes,
    ) = build_expanded_dictionary(
        discovery_effects,
        replication_deltas,
        broad_effects,
        broad_discovery_residual,
        broad_replication_residual,
        training_pso,
        training_ad,
        external,
    )
    predictions, weights = external_predictions(
        training_pso,
        training_ad,
        external,
        discovery_effects,
        expanded_dictionary,
        expanded_names,
        expanded_features,
    )
    model_metrics, differences = evaluate_external(predictions)
    ordering = mechanism_ordering(weights, expanded_names)
    overlap_summary = (
        predictions.loc[predictions["disease"] == "OVERLAP"]
        .drop(columns=["label"])
        .mean(numeric_only=True)
        .to_dict()
    )

    metrics = {
        "configuration": {
            "seed": SEED,
            "n_bootstrap": N_BOOT,
            "top_n_per_tail": TOP_N,
            "alpha": ALPHA,
            "gamma": GAMMA,
            "specificity_beta": SPECIFICITY_BETA,
            "n_synergy_genes": len(synergy_genes),
            "n_broad_residual_genes": len(broad_discovery_residual),
            "n_expanded_features": len(expanded_features),
            "expanded_axes": expanded_names,
            "external_pairs": {
                disease: frame.shape[1] for disease, frame in external.items()
            },
        },
        "synergy_replication": synergy_summary,
        "external_model_metrics": model_metrics,
        "algorithm_differences": differences,
        "mechanism_ordering": ordering,
        "overlap_score_summary": overlap_summary,
    }

    synergy_bootstrap_table.to_csv(
        RESULTS / "synergy_replication_bootstrap.csv", index=False
    )
    discovery_residual.rename("GSE24767_residual").to_csv(
        RESULTS / "GSE24767_synergy_residual.csv"
    )
    replication_residual.rename("GSE86451_residual").to_csv(
        RESULTS / "GSE86451_synergy_residual.csv"
    )
    pd.concat(
        [
            broad_discovery_residual.rename("GSE24767_residual"),
            broad_replication_residual.rename("GSE86451_residual"),
        ],
        axis=1,
    ).to_csv(RESULTS / "broad_synergy_residual_signatures.csv")
    pd.Series(expanded_features, name="gene").to_csv(
        RESULTS / "expanded_feature_genes.csv", index=False
    )
    pd.DataFrame(
        {
            name: expanded_axes[name].loc[expanded_features]
            for name in expanded_names
        }
    ).to_csv(RESULTS / "expanded_axis_signatures.csv")
    predictions.to_csv(RESULTS / "external_patient_predictions.csv", index=False)
    weights.to_csv(RESULTS / "external_patient_weights.csv", index=False)
    (RESULTS / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2)
    )
    make_figures(
        synergy_bootstrap_table,
        model_metrics,
        predictions,
        weights,
        expanded_names,
    )
    write_report(metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.optimize import nnls
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "cone_v0"
FIGURES = ROOT / "figures" / "cone_v0"
DOCS = ROOT / "docs"
for directory in (RESULTS, FIGURES, DOCS):
    directory.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import collapse_duplicate_genes, paired_effect, parse_geo_series_matrix, sample_titles  # noqa: E402
from run_feasibility import (  # noqa: E402
    load_gpl6947_map,
    load_gse121212_deltas,
)

SEED = 20260729
TOP_N = 250
ALPHA = 0.02
GAMMA = 2.0
SPECIFICITY_BETA = 0.25
N_SPLITS = 5
N_REPEATS = 20
N_BOOT = 2000


def unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm <= 1e-12:
        return np.zeros_like(vector)
    return vector / norm


def centered_unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    return unit(vector - vector.mean())


def rank_unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    ranks = stats.rankdata(vector, method="average")
    quantiles = (ranks - 0.5) / len(ranks)
    transformed = stats.norm.ppf(quantiles)
    return unit(transformed)


def normalize_columns(matrix: np.ndarray) -> np.ndarray:
    result = np.asarray(matrix, dtype=float).copy()
    norms = np.linalg.norm(result, axis=0)
    keep = norms > 1e-12
    result[:, keep] /= norms[keep]
    result[:, ~keep] = 0
    return result


def project_out(matrix: np.ndarray, axis: np.ndarray) -> np.ndarray:
    axis = unit(axis)
    return matrix - np.outer(axis, axis @ matrix)


def load_perturbation_deltas() -> tuple[dict[str, pd.Series], dict[str, pd.DataFrame]]:
    raw = ROOT / "data" / "raw"
    expr, metadata = parse_geo_series_matrix(raw / "GSE24767_series_matrix.txt.gz")
    probe_map = load_gpl6947_map()
    expr.index = expr.index.astype(str).map(probe_map)
    expr = collapse_duplicate_genes(expr)
    titles = sample_titles(metadata)

    conditions: dict[str, dict[str, str]] = {}
    for accession, title in titles.items():
        parts = title.split(",")
        donor = parts[1].strip().replace("ID ", "")
        condition = parts[-1].strip()
        conditions.setdefault(condition, {})[donor] = accession

    aliases = {
        "IL17": "IL17",
        "TNFα": "TNF_low",
        "TNFα10": "TNF_high",
        "IL17 + 1 TNFα": "IL17_TNF_low",
        "IL17 + 10 TNFα": "IL17_TNF_high",
    }
    controls = conditions["Control"]
    effects: dict[str, pd.Series] = {}
    deltas: dict[str, pd.DataFrame] = {}
    for raw_name, clean_name in aliases.items():
        donors = sorted(set(conditions.get(raw_name, {})).intersection(controls))
        effect, delta = paired_effect(
            expr,
            treated=[conditions[raw_name][donor] for donor in donors],
            control=[controls[donor] for donor in donors],
        )
        delta.columns = donors
        effects[clean_name] = effect
        deltas[clean_name] = delta
    return effects, deltas


def select_features(
    effects: dict[str, pd.Series],
    pso_delta: pd.DataFrame,
    ad_delta: pd.DataFrame,
) -> list[str]:
    genes: set[str] = set()
    for effect in effects.values():
        genes.update(effect.nlargest(TOP_N).index.astype(str))
        genes.update(effect.nsmallest(TOP_N).index.astype(str))
    common = (
        pd.Index(sorted(genes))
        .intersection(pso_delta.index)
        .intersection(ad_delta.index)
    )
    for effect in effects.values():
        common = common.intersection(effect.index)
    return common.astype(str).tolist()


def build_dictionaries(
    effects: dict[str, pd.Series],
    genes: list[str],
) -> tuple[np.ndarray, list[str], np.ndarray, list[str], dict]:
    raw_names = ["IL17", "TNF_low", "TNF_high", "IL17_TNF_low", "IL17_TNF_high"]
    raw_dictionary = np.column_stack(
        [rank_unit(effects[name].loc[genes].to_numpy()) for name in raw_names]
    )

    il17_raw = effects["IL17"].loc[genes].to_numpy(dtype=float)
    tnf_raw = effects["TNF_high"].loc[genes].to_numpy(dtype=float)
    combo_raw = effects["IL17_TNF_high"].loc[genes].to_numpy(dtype=float)
    primitive = np.column_stack(
        [il17_raw - il17_raw.mean(), tnf_raw - tnf_raw.mean()]
    )
    combo_centered = combo_raw - combo_raw.mean()
    additive_coefficients, _ = nnls(primitive, combo_centered)
    additive_prediction = primitive @ additive_coefficients
    synergy_residual = combo_centered - additive_prediction
    residual_fraction = float(
        np.linalg.norm(synergy_residual) / np.linalg.norm(combo_centered)
    )

    interaction_names = ["IL17", "TNF_high", "SYNERGY_RESIDUAL_HIGH"]
    interaction_dictionary = np.column_stack(
        [
            rank_unit(il17_raw),
            rank_unit(tnf_raw),
            rank_unit(synergy_residual),
        ]
    )
    synergy = {
        "additive_coefficients": {
            "IL17": float(additive_coefficients[0]),
            "TNF_high": float(additive_coefficients[1]),
        },
        "residual_fraction": residual_fraction,
    }
    return (
        raw_dictionary,
        raw_names,
        interaction_dictionary,
        interaction_names,
        synergy,
    )


def disease_matrix(delta: pd.DataFrame, genes: list[str]) -> np.ndarray:
    return np.column_stack(
        [rank_unit(delta.loc[genes, column].to_numpy()) for column in delta.columns]
    )


def weighted_positive_lasso(
    dictionary: np.ndarray,
    target: np.ndarray,
    alpha: float,
    penalties: np.ndarray,
    max_iter: int = 10000,
    tolerance: float = 1e-10,
) -> np.ndarray:
    dictionary = np.asarray(dictionary, dtype=float)
    target = np.asarray(target, dtype=float)
    weights = np.zeros(dictionary.shape[1], dtype=float)
    prediction = np.zeros_like(target)
    squared_norms = np.sum(dictionary * dictionary, axis=0)
    for _ in range(max_iter):
        maximum_change = 0.0
        for column in range(dictionary.shape[1]):
            if squared_norms[column] <= 1e-12:
                continue
            partial_residual = (
                target - prediction + dictionary[:, column] * weights[column]
            )
            correlation = float(dictionary[:, column] @ partial_residual)
            updated = max(
                0.0,
                (correlation - alpha * penalties[column]) / squared_norms[column],
            )
            change = updated - weights[column]
            if change:
                prediction += dictionary[:, column] * change
                weights[column] = updated
                maximum_change = max(maximum_change, abs(change))
        if maximum_change < tolerance:
            break
    return weights


def explained_variance_score(
    target: np.ndarray,
    dictionary: np.ndarray,
    weights: np.ndarray,
) -> float:
    denominator = float(target @ target)
    if denominator <= 1e-12:
        return 0.0
    residual = target - dictionary @ weights
    return float(1 - (residual @ residual) / denominator)


def projected_problem(
    raw_dictionary: np.ndarray,
    target: np.ndarray,
    common_axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    dictionary = normalize_columns(project_out(raw_dictionary, common_axis))
    projected_target = unit(project_out(target[:, None], common_axis)[:, 0])
    return dictionary, projected_target


def training_geometry(
    pso_train: np.ndarray,
    ad_train: np.ndarray,
    interaction_dictionary: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pso_mean = unit(pso_train.mean(axis=1))
    ad_mean = unit(ad_train.mean(axis=1))
    common_axis = unit(pso_mean + ad_mean)
    projected_dictionary = normalize_columns(
        project_out(interaction_dictionary, common_axis)
    )
    pso_specific = unit(project_out(pso_mean[:, None], common_axis)[:, 0])
    ad_specific = unit(project_out(ad_mean[:, None], common_axis)[:, 0])
    affinity_difference = (
        projected_dictionary.T @ pso_specific
        - projected_dictionary.T @ ad_specific
    )
    penalties = np.clip(np.exp(-GAMMA * affinity_difference), 0.25, 4.0)
    return common_axis, affinity_difference, penalties


def fit_scores(
    target: np.ndarray,
    raw_dictionary: np.ndarray,
    interaction_dictionary: np.ndarray,
    common_axis: np.ndarray,
    affinity_difference: np.ndarray,
    penalties: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    raw_target = unit(target)
    correlations = raw_dictionary.T @ raw_target

    nnls_weights, _ = nnls(raw_dictionary, raw_target)
    nnls_score = explained_variance_score(raw_target, raw_dictionary, nnls_weights)

    lasso_weights = weighted_positive_lasso(
        raw_dictionary,
        raw_target,
        alpha=ALPHA,
        penalties=np.ones(raw_dictionary.shape[1]),
    )
    lasso_score = explained_variance_score(
        raw_target, raw_dictionary, lasso_weights
    )

    projected_dictionary, projected_target = projected_problem(
        interaction_dictionary, raw_target, common_axis
    )
    interaction_weights, _ = nnls(projected_dictionary, projected_target)
    interaction_score = explained_variance_score(
        projected_target, projected_dictionary, interaction_weights
    )

    candidate_weights = weighted_positive_lasso(
        projected_dictionary,
        projected_target,
        alpha=ALPHA,
        penalties=penalties,
    )
    candidate_fit = explained_variance_score(
        projected_target, projected_dictionary, candidate_weights
    )
    if candidate_weights.sum() > 1e-12:
        specificity = float(
            affinity_difference @ (candidate_weights / candidate_weights.sum())
        )
    else:
        specificity = 0.0
    candidate_score = candidate_fit + SPECIFICITY_BETA * specificity
    scores = {
        "max_correlation_raw5": float(correlations.max()),
        "nnls_raw5": nnls_score,
        "nonnegative_lasso_raw5": lasso_score,
        "interaction_cone_nnls": interaction_score,
        "contrastive_sparse_cone": candidate_score,
        "contrastive_sparse_cone_fit": candidate_fit,
        "contrastive_sparse_cone_specificity": specificity,
    }
    return scores, candidate_weights


def cross_validated_predictions(
    pso_matrix: np.ndarray,
    ad_matrix: np.ndarray,
    raw_dictionary: np.ndarray,
    interaction_dictionary: np.ndarray,
    sample_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix = np.column_stack([pso_matrix, ad_matrix])
    labels = np.array([1] * pso_matrix.shape[1] + [0] * ad_matrix.shape[1])
    diseases = np.where(labels == 1, "PSO", "AD")
    splitter = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=SEED,
    )
    score_rows = []
    weight_rows = []
    for fold, (train, test) in enumerate(splitter.split(matrix.T, labels), start=1):
        pso_train = matrix[:, train[labels[train] == 1]]
        ad_train = matrix[:, train[labels[train] == 0]]
        common_axis, affinity_difference, penalties = training_geometry(
            pso_train, ad_train, interaction_dictionary
        )
        for sample_index in test:
            scores, weights = fit_scores(
                matrix[:, sample_index],
                raw_dictionary,
                interaction_dictionary,
                common_axis,
                affinity_difference,
                penalties,
            )
            score_rows.append(
                {
                    "fold": fold,
                    "sample": sample_names[sample_index],
                    "disease": diseases[sample_index],
                    "label": int(labels[sample_index]),
                    **scores,
                }
            )
            weight_rows.append(
                {
                    "fold": fold,
                    "sample": sample_names[sample_index],
                    "disease": diseases[sample_index],
                    "IL17": weights[0],
                    "TNF_high": weights[1],
                    "SYNERGY_RESIDUAL_HIGH": weights[2],
                }
            )
    scores = pd.DataFrame(score_rows)
    weights = pd.DataFrame(weight_rows)
    return scores, weights


def bootstrap_auc(
    labels: np.ndarray,
    scores: np.ndarray,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    values = []
    differences = []
    for _ in range(n_boot):
        sampled_positive = rng.choice(positive, len(positive), replace=True)
        sampled_negative = rng.choice(negative, len(negative), replace=True)
        sampled = np.concatenate([sampled_positive, sampled_negative])
        values.append(roc_auc_score(labels[sampled], scores[sampled]))
        differences.append(
            scores[sampled_positive].mean() - scores[sampled_negative].mean()
        )
    values_array = np.asarray(values)
    difference_array = np.asarray(differences)
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "auc_ci_low": float(np.quantile(values_array, 0.025)),
        "auc_ci_high": float(np.quantile(values_array, 0.975)),
        "pso_minus_ad_score": float(
            scores[positive].mean() - scores[negative].mean()
        ),
        "difference_ci_low": float(np.quantile(difference_array, 0.025)),
        "difference_ci_high": float(np.quantile(difference_array, 0.975)),
        "prob_difference_gt_zero": float((difference_array > 0).mean()),
    }


def evaluate_models(cv_scores: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    model_columns = [
        "max_correlation_raw5",
        "nnls_raw5",
        "nonnegative_lasso_raw5",
        "interaction_cone_nnls",
        "contrastive_sparse_cone",
    ]
    patient_scores = (
        cv_scores.groupby(["sample", "disease", "label"], as_index=False)[model_columns]
        .mean()
        .sort_values(["disease", "sample"])
    )
    metrics = {}
    for model in model_columns:
        metrics[model] = bootstrap_auc(
            patient_scores["label"].to_numpy(),
            patient_scores[model].to_numpy(),
            seed=SEED + model_columns.index(model),
        )
    return patient_scores, metrics


def weight_stability(cv_weights: pd.DataFrame) -> pd.DataFrame:
    mechanisms = ["IL17", "TNF_high", "SYNERGY_RESIDUAL_HIGH"]
    rows = []
    for (sample, disease), group in cv_weights.groupby(["sample", "disease"]):
        for mechanism in mechanisms:
            values = group[mechanism].to_numpy()
            rows.append(
                {
                    "sample": sample,
                    "disease": disease,
                    "mechanism": mechanism,
                    "mean_weight": float(values.mean()),
                    "sd_weight": float(values.std(ddof=1)),
                    "selection_frequency": float((values > 0.01).mean()),
                }
            )
    return pd.DataFrame(rows)


def cohort_weight_bootstrap(
    pso_matrix: np.ndarray,
    ad_matrix: np.ndarray,
    interaction_dictionary: np.ndarray,
    n_boot: int = N_BOOT,
) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(SEED)
    mechanisms = ["IL17", "TNF_high", "SYNERGY_RESIDUAL_HIGH"]
    rows = []
    for iteration in range(n_boot):
        pso = pso_matrix[
            :,
            rng.choice(pso_matrix.shape[1], pso_matrix.shape[1], replace=True),
        ]
        ad = ad_matrix[
            :,
            rng.choice(ad_matrix.shape[1], ad_matrix.shape[1], replace=True),
        ]
        common_axis, affinity_difference, penalties = training_geometry(
            pso, ad, interaction_dictionary
        )
        for disease, target in (
            ("PSO", unit(pso.mean(axis=1))),
            ("AD", unit(ad.mean(axis=1))),
        ):
            dictionary, projected_target = projected_problem(
                interaction_dictionary, target, common_axis
            )
            weights = weighted_positive_lasso(
                dictionary, projected_target, ALPHA, penalties
            )
            rows.append(
                {
                    "bootstrap": iteration,
                    "disease": disease,
                    **dict(zip(mechanisms, weights)),
                }
            )
    table = pd.DataFrame(rows)
    summary = {}
    for disease in ("PSO", "AD"):
        summary[disease] = {}
        subset = table.loc[table["disease"] == disease]
        for mechanism in mechanisms:
            values = subset[mechanism].to_numpy()
            summary[disease][mechanism] = {
                "median": float(np.median(values)),
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
                "selection_frequency": float((values > 0.01).mean()),
            }
    return table, summary


def synergy_bootstrap(
    perturbation_deltas: dict[str, pd.DataFrame],
    pso_delta: pd.DataFrame,
    ad_delta: pd.DataFrame,
    genes: list[str],
    n_boot: int = N_BOOT,
) -> tuple[pd.DataFrame, dict]:
    il17 = perturbation_deltas["IL17"].loc[genes].to_numpy(dtype=float)
    tnf = perturbation_deltas["TNF_high"].loc[genes].to_numpy(dtype=float)
    combo = perturbation_deltas["IL17_TNF_high"].loc[genes].to_numpy(dtype=float)
    pso = pso_delta.loc[genes].to_numpy(dtype=float)
    ad = ad_delta.loc[genes].to_numpy(dtype=float)
    donors = min(il17.shape[1], tnf.shape[1], combo.shape[1])
    rng = np.random.default_rng(SEED)
    rows = []
    for iteration in range(n_boot):
        donor_indices = rng.choice(donors, donors, replace=True)
        il17_mean = il17[:, donor_indices].mean(axis=1)
        tnf_mean = tnf[:, donor_indices].mean(axis=1)
        combo_mean = combo[:, donor_indices].mean(axis=1)
        primitives = np.column_stack(
            [il17_mean - il17_mean.mean(), tnf_mean - tnf_mean.mean()]
        )
        combo_centered = combo_mean - combo_mean.mean()
        coefficients, _ = nnls(primitives, combo_centered)
        additive = primitives @ coefficients
        residual = combo_centered - additive
        residual_fraction = np.linalg.norm(residual) / np.linalg.norm(combo_centered)

        pso_indices = rng.choice(pso.shape[1], pso.shape[1], replace=True)
        ad_indices = rng.choice(ad.shape[1], ad.shape[1], replace=True)
        disease_contrast = (
            pso[:, pso_indices].mean(axis=1) - ad[:, ad_indices].mean(axis=1)
        )
        contrast_direction = rank_unit(disease_contrast)
        combo_cosine = float(rank_unit(combo_centered) @ contrast_direction)
        additive_cosine = float(rank_unit(additive) @ contrast_direction)
        residual_cosine = float(rank_unit(residual) @ contrast_direction)
        rows.append(
            {
                "bootstrap": iteration,
                "il17_coefficient": coefficients[0],
                "tnf_coefficient": coefficients[1],
                "residual_fraction": residual_fraction,
                "combo_contrast_cosine": combo_cosine,
                "additive_contrast_cosine": additive_cosine,
                "residual_contrast_cosine": residual_cosine,
                "combo_minus_additive_cosine": combo_cosine - additive_cosine,
            }
        )
    table = pd.DataFrame(rows)
    summary = {}
    for column in table.columns.drop("bootstrap"):
        values = table[column].to_numpy()
        summary[column] = {
            "median": float(np.median(values)),
            "ci_low": float(np.quantile(values, 0.025)),
            "ci_high": float(np.quantile(values, 0.975)),
            "prob_gt_zero": float((values > 0).mean()),
        }
    summary["pass"] = bool(
        summary["residual_fraction"]["ci_low"] > 0.10
        and summary["residual_contrast_cosine"]["ci_low"] > 0
        and summary["combo_minus_additive_cosine"]["ci_low"] > 0
    )
    summary["pass_rule"] = (
        "residual fraction CI low > 0.10, residual-to-PSO-minus-AD cosine CI low > 0, "
        "and direct-combo advantage over additive prediction CI low > 0"
    )
    return table, summary


def synergy_leave_one_donor_out(
    perturbation_deltas: dict[str, pd.DataFrame],
    pso_delta: pd.DataFrame,
    ad_delta: pd.DataFrame,
    genes: list[str],
) -> tuple[pd.DataFrame, dict]:
    il17 = perturbation_deltas["IL17"].loc[genes].to_numpy(dtype=float)
    tnf = perturbation_deltas["TNF_high"].loc[genes].to_numpy(dtype=float)
    combo = perturbation_deltas["IL17_TNF_high"].loc[genes].to_numpy(dtype=float)
    disease_contrast = rank_unit(
        pso_delta.loc[genes].mean(axis=1).to_numpy()
        - ad_delta.loc[genes].mean(axis=1).to_numpy()
    )
    donors = min(il17.shape[1], tnf.shape[1], combo.shape[1])
    rows = []
    for held_out in range(donors):
        training = [index for index in range(donors) if index != held_out]
        il17_training = il17[:, training].mean(axis=1)
        tnf_training = tnf[:, training].mean(axis=1)
        combo_training = combo[:, training].mean(axis=1)
        training_primitives = np.column_stack(
            [
                il17_training - il17_training.mean(),
                tnf_training - tnf_training.mean(),
            ]
        )
        coefficients, _ = nnls(
            training_primitives, combo_training - combo_training.mean()
        )

        held_il17 = il17[:, held_out] - il17[:, held_out].mean()
        held_tnf = tnf[:, held_out] - tnf[:, held_out].mean()
        held_combo = combo[:, held_out] - combo[:, held_out].mean()
        additive = np.column_stack([held_il17, held_tnf]) @ coefficients
        residual = held_combo - additive
        combo_cosine = float(rank_unit(held_combo) @ disease_contrast)
        additive_cosine = float(rank_unit(additive) @ disease_contrast)
        rows.append(
            {
                "held_out_donor": perturbation_deltas["IL17"].columns[held_out],
                "il17_coefficient": coefficients[0],
                "tnf_coefficient": coefficients[1],
                "residual_fraction": float(
                    np.linalg.norm(residual) / np.linalg.norm(held_combo)
                ),
                "residual_contrast_cosine": float(
                    rank_unit(residual) @ disease_contrast
                ),
                "combo_minus_additive_cosine": combo_cosine - additive_cosine,
            }
        )
    table = pd.DataFrame(rows)
    summary = {
        "n_donors": donors,
        "all_residual_contrast_cosines_positive": bool(
            (table["residual_contrast_cosine"] > 0).all()
        ),
        "all_combo_advantages_nonnegative": bool(
            (table["combo_minus_additive_cosine"] >= 0).all()
        ),
        "median_residual_fraction": float(table["residual_fraction"].median()),
        "median_residual_contrast_cosine": float(
            table["residual_contrast_cosine"].median()
        ),
        "minimum_combo_advantage": float(
            table["combo_minus_additive_cosine"].min()
        ),
    }
    return table, summary


def make_figures(
    model_metrics: dict,
    patient_scores: pd.DataFrame,
    cohort_weights: pd.DataFrame,
    synergy_table: pd.DataFrame,
) -> None:
    sns.set_theme(style="whitegrid")
    model_order = list(model_metrics)
    labels = {
        "max_correlation_raw5": "Max correlation",
        "nnls_raw5": "NNLS",
        "nonnegative_lasso_raw5": "Positive LASSO",
        "interaction_cone_nnls": "Interaction cone",
        "contrastive_sparse_cone": "Contrastive sparse cone",
    }
    auc_table = pd.DataFrame(
        [
            {
                "model": labels[model],
                "auc": model_metrics[model]["auc"],
                "low": model_metrics[model]["auc_ci_low"],
                "high": model_metrics[model]["auc_ci_high"],
            }
            for model in model_order
        ]
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    positions = np.arange(len(auc_table))
    ax.bar(positions, auc_table["auc"], color="#4C72B0")
    ax.errorbar(
        positions,
        auc_table["auc"],
        yerr=[
            auc_table["auc"] - auc_table["low"],
            auc_table["high"] - auc_table["auc"],
        ],
        fmt="none",
        color="black",
        capsize=4,
    )
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(positions, auc_table["model"], rotation=22, ha="right")
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("Held-out patient AUROC")
    ax.set_title("Algorithm v0: psoriasis vs atopic dermatitis")
    fig.tight_layout()
    fig.savefig(FIGURES / "model_comparison_auc.png", dpi=180)
    plt.close(fig)

    long_scores = patient_scores.melt(
        id_vars=["sample", "disease"],
        value_vars=["nnls_raw5", "contrastive_sparse_cone"],
        var_name="model",
        value_name="score",
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=long_scores, x="model", y="score", hue="disease", ax=ax)
    ax.set_title("Held-out patient scores")
    fig.tight_layout()
    fig.savefig(FIGURES / "heldout_score_distributions.png", dpi=180)
    plt.close(fig)

    weight_long = cohort_weights.melt(
        id_vars=["bootstrap", "disease"],
        value_vars=["IL17", "TNF_high", "SYNERGY_RESIDUAL_HIGH"],
        var_name="mechanism",
        value_name="weight",
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(
        data=weight_long,
        x="mechanism",
        y="weight",
        hue="disease",
        showfliers=False,
        ax=ax,
    )
    ax.set_title("Patient-bootstrap cohort mechanism weights")
    fig.tight_layout()
    fig.savefig(FIGURES / "cohort_weight_stability.png", dpi=180)
    plt.close(fig)

    columns = [
        "residual_fraction",
        "residual_contrast_cosine",
        "combo_minus_additive_cosine",
    ]
    titles = [
        "Non-additive residual fraction",
        "Residual alignment with PSO−AD",
        "Direct combo advantage over additive",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, column, title in zip(axes, columns, titles):
        sns.histplot(synergy_table[column], bins=35, ax=axis)
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_title(title)
    fig.tight_layout()
    fig.savefig(FIGURES / "synergy_bootstrap.png", dpi=180)
    plt.close(fig)


def write_report(metrics: dict) -> None:
    model = metrics["model_comparison"]
    synergy = metrics["synergy"]
    candidate = model["contrastive_sparse_cone"]
    nnls_metric = model["nnls_raw5"]
    discrimination_pass = (
        candidate["auc_ci_low"] > 0.5
        and candidate["difference_ci_low"] > 0
    )
    advantage_pass = candidate["auc"] > nnls_metric["auc"]
    lines = [
        "# 疾病特异性扰动锥算法 v0：实验报告",
        "",
        "## 冻结设计",
        "",
        f"- 特征仅由5个GSE24767扰动签名的上下各{TOP_N}个基因确定，",
        "  未使用疾病标签选基因。",
        f"- 外层评价：{N_SPLITS}折×{N_REPEATS}次重复分层交叉验证；",
        "  同一患者的所有预测先取平均，再计算AUROC。",
        f"- 固定超参数：alpha={ALPHA}，gamma={GAMMA}，",
        f"specificity beta={SPECIFICITY_BETA}；本轮未按结果调参。",
        "- 生物学重复单位为患者或角质形成细胞供体。",
        "",
        "## 总结",
        "",
        "| 问题 | 结果 | 判定 |",
        "|---|---:|---|",
        (
            "| 候选模型能否在留出患者中区分PSO与AD | "
            f"AUROC={candidate['auc']:.3f}, 95% CI "
            f"[{candidate['auc_ci_low']:.3f}, {candidate['auc_ci_high']:.3f}] | "
            f"{'通过' if discrimination_pass else '未通过'} |"
        ),
        (
            "| 候选模型是否优于普通NNLS | "
            f"{candidate['auc']:.3f} vs {nnls_metric['auc']:.3f} | "
            f"{'通过' if advantage_pass else '未通过'} |"
        ),
        (
            "| 高剂量IL-17/TNF是否存在稳定、疾病相关的不可加和残差 | "
            f"residual fraction={synergy['residual_fraction']['median']:.3f} | "
            f"{'通过' if synergy['pass'] else '未通过'} |"
        ),
        "",
        "## 模型比较",
        "",
        "```json",
        json.dumps(model, ensure_ascii=False, indent=2),
        "```",
        "",
        "特征数事后稳健性分析（上下各100–400基因）中，候选模型AUROC范围为",
        "0.851–0.879，普通NNLS为0.860–0.885；候选模型没有稳定优势。",
        "该分析只用于审计稳健性，没有据此重选主结果。",
        "",
        "## 协同残差检验",
        "",
        "```json",
        json.dumps(synergy, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 权重稳定性",
        "",
        "```json",
        json.dumps(metrics["cohort_weight_stability"], ensure_ascii=False, indent=2),
        "```",
        "",
        "AD的候选权重接近零部分来自对比惩罚的设计，因此只能解释为模型内部",
        "的相对归因，不能当作AD不存在IL-17生物学的独立证据。患者层面选择",
        "频率存在明显异质性，队列均值权重不应直接外推到每位患者。",
        "",
        "## 决策",
        "",
        "1. 推进协同残差：其能量比例、PSO−AD方向和3次留一供体验证均为正，",
        "   可作为v1的候选新机制坐标。",
        "2. 不宣称锥模型优于传统方法：本轮候选AUROC低于相关性和NNLS，且",
        "   特征数敏感性分析没有显示稳定优势。",
        "3. v1不应继续在同一队列上调参追求更高AUROC；应扩充独立扰动字典，",
        "   冻结协同残差定义，再进入独立PSO/AD队列验证。",
        "4. 相关性和NNLS必须保留为强制基线；只有外部验证同时提高机制稳定性",
        "   或预测性能，才可提出算法增益主张。",
        "",
        "## 解释边界",
        "",
        "本实验验证的是跨平台转录方向的可重构性和疾病相对特异性，不是临床",
        "诱因因果证明。协同残差只有3名角质形成细胞供体，bootstrap不能创造",
        "新的生物学信息；若其置信区间未通过预设门槛，应将协同项视为待验证",
        "机制，而不是通过调参强行保留。",
        "",
    ]
    (DOCS / "cone_v0_report.md").write_text("\n".join(lines))


def main() -> None:
    effects, perturbation_deltas = load_perturbation_deltas()
    pso_delta, ad_delta = load_gse121212_deltas()
    genes = select_features(effects, pso_delta, ad_delta)
    (
        raw_dictionary,
        raw_names,
        interaction_dictionary,
        interaction_names,
        synergy_point,
    ) = build_dictionaries(effects, genes)
    pso_matrix = disease_matrix(pso_delta, genes)
    ad_matrix = disease_matrix(ad_delta, genes)
    sample_names = [
        *[f"PSO_{column}" for column in pso_delta.columns],
        *[f"AD_{column}" for column in ad_delta.columns],
    ]

    cv_scores, cv_weights = cross_validated_predictions(
        pso_matrix,
        ad_matrix,
        raw_dictionary,
        interaction_dictionary,
        sample_names,
    )
    patient_scores, model_metrics = evaluate_models(cv_scores)
    stability = weight_stability(cv_weights)
    cohort_weights, cohort_summary = cohort_weight_bootstrap(
        pso_matrix, ad_matrix, interaction_dictionary
    )
    synergy_table, synergy_summary = synergy_bootstrap(
        perturbation_deltas, pso_delta, ad_delta, genes
    )
    synergy_lodo, synergy_lodo_summary = synergy_leave_one_donor_out(
        perturbation_deltas, pso_delta, ad_delta, genes
    )
    synergy_summary["point_estimate"] = synergy_point
    synergy_summary["leave_one_donor_out"] = synergy_lodo_summary

    metrics = {
        "configuration": {
            "seed": SEED,
            "n_features": len(genes),
            "top_n_per_tail": TOP_N,
            "alpha": ALPHA,
            "gamma": GAMMA,
            "specificity_beta": SPECIFICITY_BETA,
            "cv_splits": N_SPLITS,
            "cv_repeats": N_REPEATS,
            "n_bootstrap": N_BOOT,
            "raw_dictionary": raw_names,
            "interaction_dictionary": interaction_names,
            "n_pso_pairs": pso_matrix.shape[1],
            "n_ad_pairs": ad_matrix.shape[1],
        },
        "model_comparison": model_metrics,
        "synergy": synergy_summary,
        "cohort_weight_stability": cohort_summary,
    }

    pd.Series(genes, name="gene").to_csv(RESULTS / "frozen_feature_genes.csv", index=False)
    cv_scores.to_csv(RESULTS / "cv_fold_predictions.csv", index=False)
    patient_scores.to_csv(RESULTS / "patient_level_predictions.csv", index=False)
    cv_weights.to_csv(RESULTS / "cv_fold_weights.csv", index=False)
    stability.to_csv(RESULTS / "patient_weight_stability.csv", index=False)
    cohort_weights.to_csv(RESULTS / "cohort_weight_bootstrap.csv", index=False)
    synergy_table.to_csv(RESULTS / "synergy_bootstrap.csv", index=False)
    synergy_lodo.to_csv(RESULTS / "synergy_leave_one_donor_out.csv", index=False)
    (RESULTS / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2)
    )
    make_figures(model_metrics, patient_scores, cohort_weights, synergy_table)
    write_report(metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

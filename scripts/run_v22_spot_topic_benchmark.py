"""Fold-local spot-topic benchmark for the PRI-Audit revision.

This is a deliberately modest high-resolution comparator rather than a large
neural network: with only 12 independent spatial patients, a high-capacity GNN
would make an untestable claim.  In every held-patient fold, MiniBatchNMF is
fitted only to training-patient spots and transformed on held-patient spots.
Mean topic activations create an eight-dimensional patient/sample-level
composition representation.  The benchmark then compares it against an
equally eight-dimensional fold-local expression PCA reference and compares
the 16-dimensional expression+topic fusion against a 16-dimensional expression
PCA reference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import MiniBatchNMF, PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "pri_audit_v22"
SEED = 20260824
METHODS = ("expression_pca_8", "spot_topic_8", "expression_pca_16", "expression_plus_topic_16")

sys.path.insert(0, str(ROOT / "scripts"))
import run_pri_audit_v21 as v21  # noqa: E402
import run_strict_spatial_graph_benchmark as legacy  # noqa: E402


def patient_subset() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    metadata = pd.read_csv(v21.OUT / "gse206391_sample_metadata.csv")
    counts = np.load(v21.OUT / "gse206391_sample_gene_counts.npz")["counts"].astype(float)
    eligible = metadata[metadata.condition.isin(["NON LESIONAL", "LESIONAL"])]
    paired = eligible.groupby("patient").condition.nunique()
    patients = paired[paired == 2].index
    keep = (metadata.patient.isin(patients) & metadata.condition.isin(["NON LESIONAL", "LESIONAL"])).to_numpy()
    metadata = metadata.loc[keep].reset_index(drop=True)
    counts = counts[keep]
    log_cpm = np.log1p(counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0) * 1e6)
    labels = (metadata.condition == "LESIONAL").astype(int).to_numpy()
    groups = metadata.patient.astype(str).to_numpy()
    return metadata, log_cpm, labels, groups


def normalised_spots(matrix, total, indices, genes, max_spots, seed) -> np.ndarray:
    """Create a fixed random spot subsample without crossing fold boundaries."""
    if len(indices) > max_spots:
        rng = np.random.default_rng(seed)
        indices = rng.choice(indices, size=max_spots, replace=False)
    values = matrix[indices][:, genes].toarray().astype(np.float32)
    return np.log1p(values / np.maximum(total[indices, None], 1.0) * 1e4)


def fit_predict(train: np.ndarray, labels: np.ndarray, test: np.ndarray) -> np.ndarray:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, solver="liblinear", class_weight="balanced", max_iter=2000, random_state=0),
    )
    model.fit(train, labels)
    return model.predict_proba(test)[:, 1]


def build_folds(matrix, spot_meta, total, sample_meta, log_cpm, groups, max_spots: int, n_topics: int) -> list[dict[str, object]]:
    spot_groups = legacy.prepare_spot_groups(spot_meta, sample_meta)
    folds: list[dict[str, object]] = []
    for fold_no, held in enumerate(np.unique(groups)):
        test = groups == held
        train = ~test
        genes = np.argsort(log_cpm[train].var(axis=0))[-500:]

        # The expression comparators use the same fold-local HVG set and have
        # exactly 8 or 16 dimensions to match the topic and fused candidates.
        scaler = StandardScaler()
        scaled_train = scaler.fit_transform(log_cpm[train][:, genes])
        scaled_test = scaler.transform(log_cpm[test][:, genes])
        pca8 = PCA(n_components=8, random_state=SEED)
        pca16 = PCA(n_components=16, random_state=SEED)
        expression8_train, expression8_test = pca8.fit_transform(scaled_train), pca8.transform(scaled_test)
        expression16_train, expression16_test = pca16.fit_transform(scaled_train), pca16.transform(scaled_test)

        sampled_spots: dict[str, np.ndarray] = {}
        for sample_number, sample in enumerate(sample_meta["sample"].astype(str)):
            indices, _ = spot_groups[sample]
            sampled_spots[sample] = normalised_spots(
                matrix, total, indices, genes, max_spots,
                seed=SEED + 100_000 * fold_no + sample_number,
            )
        train_spots = np.vstack([sampled_spots[str(sample)] for sample in sample_meta.loc[train, "sample"]])
        topic_model = MiniBatchNMF(
            n_components=n_topics,
            init="nndsvda",
            random_state=SEED,
            batch_size=min(1024, train_spots.shape[0]),
            max_iter=600,
            tol=1e-4,
        )
        topic_model.fit(train_spots)
        topic_summary = np.vstack([
            topic_model.transform(sampled_spots[str(sample)]).mean(axis=0)
            for sample in sample_meta["sample"]
        ])
        folds.append({
            "held_patient": str(held),
            "train_ix": np.flatnonzero(train),
            "test_ix": np.flatnonzero(test),
            "representations": {
                "expression_pca_8": (expression8_train, expression8_test),
                "spot_topic_8": (topic_summary[train], topic_summary[test]),
                "expression_pca_16": (expression16_train, expression16_test),
                "expression_plus_topic_16": (
                    np.hstack([expression8_train, topic_summary[train]]),
                    np.hstack([expression8_test, topic_summary[test]]),
                ),
            },
        })
    return folds


def predictions(folds: list[dict[str, object]], labels: np.ndarray, method: str) -> np.ndarray:
    probability = np.full(labels.size, np.nan)
    for fold in folds:
        train, test = fold["train_ix"], fold["test_ix"]
        x_train, x_test = fold["representations"][method]
        probability[test] = fit_predict(x_train, labels[train], x_test)
    if np.isnan(probability).any():
        raise RuntimeError(f"Missing predictions for {method}")
    return probability


def bootstrap(y, probabilities, groups, rng, n_bootstrap):
    unique = np.unique(groups)
    by_group = {group: np.flatnonzero(groups == group) for group in unique}
    scores = {method: [] for method in METHODS}
    deltas = {"spot_topic_8_vs_expression_pca_8": [], "expression_plus_topic_16_vs_expression_pca_16": []}
    for _ in range(n_bootstrap):
        selected = rng.choice(unique, size=unique.size, replace=True)
        ix = np.concatenate([by_group[group] for group in selected])
        current = {method: roc_auc_score(y[ix], value[ix]) for method, value in probabilities.items()}
        for method, score in current.items():
            scores[method].append(score)
        deltas["spot_topic_8_vs_expression_pca_8"].append(current["spot_topic_8"] - current["expression_pca_8"])
        deltas["expression_plus_topic_16_vs_expression_pca_16"].append(
            current["expression_plus_topic_16"] - current["expression_pca_16"]
        )
    return {key: np.asarray(value) for key, value in scores.items()}, {key: np.asarray(value) for key, value in deltas.items()}


def paired_label_null(folds, labels, groups, rng, n_permutations):
    results = {method: [] for method in METHODS}
    for _ in range(n_permutations):
        shuffled = labels.copy()
        for patient in np.unique(groups):
            if rng.random() < 0.5:
                ix = groups == patient
                shuffled[ix] = 1 - shuffled[ix]
        for method in METHODS:
            results[method].append(float(roc_auc_score(shuffled, predictions(folds, shuffled, method))))
    return {key: np.asarray(value) for key, value in results.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fold-local GSE206391 spot-topic benchmark.")
    parser.add_argument("--max-spots-per-sample", type=int, default=300)
    parser.add_argument("--topics", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--tag", default="primary", help="Stable suffix for a primary or predeclared sensitivity run.")
    args = parser.parse_args()
    if args.topics != 8:
        raise ValueError("The primary, predeclared comparison uses exactly eight topics.")
    if args.max_spots_per_sample < 100:
        raise ValueError("At least 100 spots per sample are required for the primary topic summary.")

    OUT.mkdir(parents=True, exist_ok=True)
    legacy.OUT = v21.OUT
    matrix, spot_meta, total = legacy.load_spots()
    sample_meta, log_cpm, labels, groups = patient_subset()
    folds = build_folds(
        matrix, spot_meta, total, sample_meta, log_cpm, groups,
        max_spots=args.max_spots_per_sample, n_topics=args.topics,
    )
    observed = {method: predictions(folds, labels, method) for method in METHODS}
    rng = np.random.default_rng(SEED)
    boot_scores, boot_deltas = bootstrap(labels, observed, groups, rng, args.bootstrap)
    label_null = paired_label_null(folds, labels, groups, rng, args.permutations)

    comparisons = {
        "spot_topic_8": ("expression_pca_8", "spot_topic_8_vs_expression_pca_8"),
        "expression_plus_topic_16": ("expression_pca_16", "expression_plus_topic_16_vs_expression_pca_16"),
    }
    result: dict[str, object] = {
        "dataset": "GSE206391",
        "n_samples": int(labels.size),
        "n_patients": int(np.unique(groups).size),
        "outer_split": "paired-patient leave-one-patient-out",
        "hvg_selection": "top 500 sample-pseudobulk HVGs fitted in each outer-patient fold",
        "topic_model": "MiniBatchNMF fitted only on training-patient spots; transformed held-patient spots",
        "n_topics": args.topics,
        "max_spots_per_sample": args.max_spots_per_sample,
        "matched_dimension_contract": "topic 8 vs expression PCA 8; expression+topic 16 vs expression PCA 16",
        "methods": {},
    }
    for method, probability in observed.items():
        entry: dict[str, object] = {
            "auroc": float(roc_auc_score(labels, probability)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, probability >= 0.5)),
            "patient_bootstrap_ci95": [float(value) for value in np.quantile(boot_scores[method], [0.025, 0.975])],
            "label_permutation_p_ge_observed": float(
                (1 + np.sum(label_null[method] >= roc_auc_score(labels, probability))) / (args.permutations + 1)
            ),
        }
        if method in comparisons:
            reference, key = comparisons[method]
            delta = boot_deltas[key]
            entry.update({
                "matched_reference": reference,
                "incremental_auroc": float(roc_auc_score(labels, probability) - roc_auc_score(labels, observed[reference])),
                "increment_patient_bootstrap_ci95": [float(value) for value in np.quantile(delta, [0.025, 0.975])],
                "increment_bootstrap_probability_ge_0_05": float(np.mean(delta >= 0.05)),
            })
        result["methods"][method] = entry

    prediction_table = sample_meta.copy()
    prediction_table["true"] = labels
    prediction_table["outer_patient"] = groups
    for method, probability in observed.items():
        prediction_table[f"{method}_probability"] = probability
    suffix = "" if args.tag == "primary" else f"_{args.tag}"
    prediction_table.to_csv(OUT / f"gse206391_v22_spot_topic_predictions{suffix}.csv", index=False)
    np.savez_compressed(
        OUT / f"gse206391_v22_spot_topic_distributions{suffix}.npz",
        **{f"bootstrap_{key}": value for key, value in boot_scores.items()},
        **{f"bootstrap_{key}": value for key, value in boot_deltas.items()},
        **{f"label_permutation_{key}": value for key, value in label_null.items()},
        seed=np.array([SEED]),
    )
    (OUT / f"gse206391_v22_spot_topic_benchmark{suffix}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Coordinate-shuffle null for the fair-dimension v21 spatial graph benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_pri_audit_v21 as v21  # noqa: E402
import run_v21_matched_spatial_graph as graph  # noqa: E402
import run_strict_spatial_graph_benchmark as legacy  # noqa: E402


OUT = v21.OUT
SEED = 20260823


def neighbours_for(coordinates: np.ndarray, k: int = 6) -> np.ndarray:
    if len(coordinates) < 2:
        return np.zeros((len(coordinates), 0), dtype=int)
    return NearestNeighbors(n_neighbors=min(k + 1, len(coordinates))).fit(coordinates).kneighbors(coordinates, return_distance=False)[:, 1:]


def spatial_from_values(values: np.ndarray, neighbours: np.ndarray) -> np.ndarray:
    if values.shape[0] < 2 or neighbours.shape[1] == 0:
        return np.zeros(values.shape[1] * 2, dtype=float)
    neighbour_mean = values[neighbours].mean(axis=1)
    centered = values - values.mean(axis=0, keepdims=True)
    neighbour_centered = centered[neighbours].mean(axis=1)
    variance = np.mean(centered * centered, axis=0)
    moran = np.divide(np.mean(centered * neighbour_centered, axis=0), variance, out=np.zeros_like(variance), where=variance > 1e-10)
    contrast = np.mean(np.abs(values - neighbour_mean), axis=0)
    return np.hstack([moran, contrast]).astype(float)


def build_cache(matrix, spot_meta, total, sample_meta, log_cpm, groups):
    spot_groups = legacy.prepare_spot_groups(spot_meta, sample_meta)
    cache = []
    for held in np.unique(groups):
        test = groups == held
        train = ~test
        top = np.argsort(log_cpm[train].var(axis=0))[-500:]
        sample_values = []
        for sample in sample_meta["sample"]:
            indices, neighbours = spot_groups[str(sample)]
            values = matrix[indices][:, top].toarray().astype(np.float32)
            values = np.log1p(values / np.maximum(total[indices, None], 1) * 1e4)
            coordinates = spot_meta.loc[indices, ["row", "col"]].to_numpy()
            sample_values.append((values, coordinates, neighbours))
        cache.append({
            "train_ix": np.flatnonzero(train), "test_ix": np.flatnonzero(test),
            "expression_1000": log_cpm[:, np.argsort(log_cpm[train].var(axis=0))[-1000:]],
            "expression_1500": log_cpm[:, np.argsort(log_cpm[train].var(axis=0))[-1500:]],
            "sample_values": sample_values,
        })
    return cache


def predict(cache, labels, neighbour_sets):
    spatial_probability = np.full(labels.size, np.nan)
    combined_probability = np.full(labels.size, np.nan)
    for fold, current_neighbours in zip(cache, neighbour_sets):
        spatial = np.vstack([spatial_from_values(values, neighbours) for (values, _, _), neighbours in zip(fold["sample_values"], current_neighbours)])
        # Spatial contains two 500-gene summaries.  Pair only the expression
        # plus spatial model with 500 expression features, yielding 1,500 total.
        combined = np.hstack([fold["expression_1500"][:, -500:], spatial])
        train, test = fold["train_ix"], fold["test_ix"]
        spatial_probability[test] = graph.fit_predict(spatial[train], labels[train], spatial[test])
        combined_probability[test] = graph.fit_predict(combined[train], labels[train], combined[test])
    if np.isnan(spatial_probability).any() or np.isnan(combined_probability).any():
        raise RuntimeError("Missing coordinate-shuffle prediction")
    return spatial_probability, combined_probability


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--permutations", type=int, default=500)
    args = parser.parse_args()
    legacy.OUT = OUT
    matrix, spot_meta, total = legacy.load_spots()
    sample_meta, log_cpm, labels, groups = graph.patient_subset()
    cache = build_cache(matrix, spot_meta, total, sample_meta, log_cpm, groups)
    observed_neighbours = [[entry[2] for entry in fold["sample_values"]] for fold in cache]
    observed_spatial, observed_combined = predict(cache, labels, observed_neighbours)
    expression_1000 = np.loadtxt(OUT / "gse206391_v21_matched_spatial_graph_predictions.csv", delimiter=",", skiprows=1, usecols=[-4])
    expression_1500 = np.loadtxt(OUT / "gse206391_v21_matched_spatial_graph_predictions.csv", delimiter=",", skiprows=1, usecols=[-2])
    observed = {"spatial_graph_1000": float(roc_auc_score(labels, observed_spatial)), "expression_plus_spatial_1500": float(roc_auc_score(labels, observed_combined))}
    rng = np.random.default_rng(SEED)
    null_spatial, null_combined = [], []
    for _ in range(args.permutations):
        shuffled = []
        for fold in cache:
            current = []
            for _, coordinates, _ in fold["sample_values"]:
                current.append(neighbours_for(coordinates[rng.permutation(len(coordinates))]))
            shuffled.append(current)
        probability_spatial, probability_combined = predict(cache, labels, shuffled)
        null_spatial.append(roc_auc_score(labels, probability_spatial))
        null_combined.append(roc_auc_score(labels, probability_combined))
    null_spatial, null_combined = np.asarray(null_spatial), np.asarray(null_combined)
    np.savez_compressed(
        OUT / "gse206391_v21_matched_spatial_coordinate_shuffle_distributions.npz",
        observed_spatial=np.array([observed["spatial_graph_1000"]]), observed_combined=np.array([observed["expression_plus_spatial_1500"]]),
        null_spatial=null_spatial, null_combined=null_combined,
        null_spatial_increment_vs_expression_1000=null_spatial - roc_auc_score(labels, expression_1000),
        null_combined_increment_vs_expression_1500=null_combined - roc_auc_score(labels, expression_1500), seed=np.array([SEED]),
    )
    result = {
        "dataset": "GSE206391", "n_coordinate_shuffles": int(args.permutations),
        "shuffle_unit": "coordinates independently permuted within each sample; counts, labels, folds and fold-specific HVGs fixed",
        "observed": observed,
        "null": {
            "spatial_graph_1000": {"mean": float(null_spatial.mean()), "ci95": [float(value) for value in np.quantile(null_spatial, [.025, .975])], "p_ge_observed": float((1 + np.sum(null_spatial >= observed["spatial_graph_1000"])) / (null_spatial.size + 1))},
            "expression_plus_spatial_1500": {"mean": float(null_combined.mean()), "ci95": [float(value) for value in np.quantile(null_combined, [.025, .975])], "p_ge_observed": float((1 + np.sum(null_combined >= observed["expression_plus_spatial_1500"])) / (null_combined.size + 1))},
        },
    }
    (OUT / "gse206391_v21_matched_spatial_coordinate_shuffle.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

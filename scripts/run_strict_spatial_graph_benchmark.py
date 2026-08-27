"""Strict patient-LODO benchmark of explicit spatial graph statistics.

The method uses per-gene Moran-like neighbor agreement and local contrast, not
an average of smoothed expression. HVG selection and every model fit occur
inside the held-out-patient loop.
"""
from pathlib import Path
import json

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/singlecell_v10"
H5 = ROOT / "data/raw/GSE206391_Preprocessed_data.h5"
RNG = np.random.default_rng(20260810)
METHODS = ("expression_500", "spatial_graph", "expression_plus_spatial")


def decode(values):
    return np.array(
        [value.decode() if isinstance(value, bytes) else str(value) for value in values]
    )


def load_spots():
    with h5py.File(H5, "r") as handle:
        indptr = handle["/X/indptr"][...]
        matrix = csr_matrix(
            (
                handle["/X/data"][...],
                handle["/X/indices"][...],
                indptr,
            ),
            shape=(len(indptr) - 1, 16685),
        )
        obs = {
            key: handle[f"/obs/{key}"][...]
            for key in (
                "patient",
                "biopsy_type",
                "sample",
                "array_row",
                "array_col",
                "in_tissue",
            )
        }
        categories = {
            key: decode(handle[f"/obs/__categories/{key}"][...])
            for key in ("patient", "biopsy_type", "sample")
        }
        mitochondrial = handle["/raw/var/mt"][...].astype(bool)

    def categorical(key):
        return categories[key][obs[key].astype(int)]

    meta = pd.DataFrame(
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
    mito_fraction = (
        np.asarray(matrix[:, mitochondrial].sum(axis=1)).ravel()
        / np.maximum(total, 1)
    )
    keep = (
        (total >= 500)
        & (detected >= 300)
        & (mito_fraction <= 0.2)
        & meta.in_tissue.astype(bool).to_numpy()
    )
    meta["keep"] = keep
    return matrix, meta, total


def patient_subset():
    sample_meta = pd.read_csv(OUT / "gse206391_sample_metadata.csv")
    counts = np.load(OUT / "gse206391_sample_gene_counts.npz")["counts"].astype(float)
    eligible = sample_meta[
        sample_meta.condition.isin(["NON LESIONAL", "LESIONAL"])
    ]
    paired = eligible.groupby("patient").condition.nunique()
    paired = paired[paired == 2].index
    mask = sample_meta.patient.isin(paired).to_numpy()
    sample_meta = sample_meta.loc[mask].reset_index(drop=True)
    counts = counts[mask]
    log_cpm = np.log1p(
        counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0) * 1e6
    )
    y = (sample_meta.condition == "LESIONAL").astype(int).to_numpy()
    groups = sample_meta.patient.astype(str).to_numpy()
    return sample_meta, log_cpm, y, groups


def prepare_spot_groups(spot_meta, sample_meta):
    groups = {}
    for sample in sample_meta["sample"]:
        idx = spot_meta.index[
            (spot_meta["sample"] == sample) & spot_meta["keep"]
        ].to_numpy()
        coordinates = spot_meta.loc[idx, ["row", "col"]].to_numpy()
        if len(idx) > 1:
            k = min(6, len(idx) - 1)
            neighbors = (
                NearestNeighbors(n_neighbors=k + 1)
                .fit(coordinates)
                .kneighbors(coordinates, return_distance=False)[:, 1:]
            )
        else:
            neighbors = np.zeros((len(idx), 0), dtype=int)
        groups[str(sample)] = (idx, neighbors)
    return groups


def graph_features(matrix, total, indices, neighbors, genes):
    values = matrix[indices][:, genes].toarray().astype(np.float32)
    values = np.log1p(values / np.maximum(total[indices, None], 1.0) * 1e4)
    if len(indices) < 2 or neighbors.shape[1] == 0:
        zeros = np.zeros(len(genes), dtype=float)
        return zeros, zeros
    neighbor_mean = values[neighbors].mean(axis=1)
    centered = values - values.mean(axis=0, keepdims=True)
    neighbor_centered = centered[neighbors].mean(axis=1)
    variance = np.mean(centered * centered, axis=0)
    moran = np.divide(
        np.mean(centered * neighbor_centered, axis=0),
        variance,
        out=np.zeros_like(variance),
        where=variance > 1e-10,
    )
    contrast = np.mean(np.abs(values - neighbor_mean), axis=0)
    return moran.astype(float), contrast.astype(float)


def fit_predict(z_train, y_train, z_test):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1,
            solver="liblinear",
            class_weight="balanced",
            max_iter=2000,
            random_state=0,
        ),
    )
    model.fit(z_train, y_train)
    return model.predict_proba(z_test)[:, 1]


def build_folds(matrix, spot_meta, total, sample_meta, log_cpm, groups):
    spot_groups = prepare_spot_groups(spot_meta, sample_meta)
    folds = []
    for held_group in np.unique(groups):
        test = groups == held_group
        train = ~test
        top = np.argsort(log_cpm[train].var(axis=0))[-500:]
        moran_rows = []
        contrast_rows = []
        for sample in sample_meta["sample"]:
            indices, neighbors = spot_groups[str(sample)]
            moran, contrast = graph_features(matrix, total, indices, neighbors, top)
            moran_rows.append(moran)
            contrast_rows.append(contrast)
        expression = log_cpm[:, top]
        spatial = np.hstack([np.vstack(moran_rows), np.vstack(contrast_rows)])
        combined = np.hstack([expression, spatial])
        folds.append(
            {
                "held_group": held_group,
                "train_idx": np.flatnonzero(train),
                "test_idx": np.flatnonzero(test),
                "representations": {
                    "expression_500": (expression[train], expression[test]),
                    "spatial_graph": (spatial[train], spatial[test]),
                    "expression_plus_spatial": (combined[train], combined[test]),
                },
            }
        )
    return folds


def predictions_for(folds, labels, method):
    probabilities = np.full(len(labels), np.nan)
    for fold in folds:
        train = fold["train_idx"]
        test = fold["test_idx"]
        z_train, z_test = fold["representations"][method]
        probabilities[test] = fit_predict(z_train, labels[train], z_test)
    return probabilities


def bootstrap(y, predictions, groups, n_boot=5000):
    unique_groups = np.unique(groups)
    by_group = {group: np.flatnonzero(groups == group) for group in unique_groups}
    scores = {method: [] for method in predictions}
    deltas = {method: [] for method in predictions if method != "expression_500"}
    for _ in range(n_boot):
        selected = RNG.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([by_group[group] for group in selected])
        if np.unique(y[idx]).size < 2:
            continue
        current = {
            method: roc_auc_score(y[idx], probability[idx])
            for method, probability in predictions.items()
        }
        for method, score in current.items():
            scores[method].append(score)
        for method in deltas:
            deltas[method].append(current[method] - current["expression_500"])
    return scores, deltas


def paired_label_permutations(folds, y, groups, n_perm=500):
    unique_groups = np.unique(groups)
    values = {method: [] for method in METHODS}
    for _ in range(n_perm):
        shuffled = y.copy()
        for group in unique_groups:
            if RNG.random() < 0.5:
                shuffled[groups == group] = 1 - shuffled[groups == group]
        for method in METHODS:
            probability = predictions_for(folds, shuffled, method)
            values[method].append(float(roc_auc_score(shuffled, probability)))
    return values


def main():
    matrix, spot_meta, total = load_spots()
    sample_meta, log_cpm, y, groups = patient_subset()
    folds = build_folds(matrix, spot_meta, total, sample_meta, log_cpm, groups)
    predictions = {
        method: predictions_for(folds, y, method) for method in METHODS
    }
    scores, deltas = bootstrap(y, predictions, groups)
    permutation = paired_label_permutations(folds, y, groups)
    result = {
        "dataset": "GSE206391",
        "n_samples": int(len(y)),
        "n_patients": int(np.unique(groups).size),
        "n_hvg_per_outer_fold": 500,
        "neighbors": 6,
        "n_paired_label_permutations": 500,
        "spatial_statistics": ["Moran-like neighbor agreement", "local contrast"],
        "methods": {},
    }
    for method, probability in predictions.items():
        entry = {
            "auroc": float(roc_auc_score(y, probability)),
            "balanced_accuracy": float(
                balanced_accuracy_score(y, probability >= 0.5)
            ),
            "group_bootstrap_ci95": [
                float(value) for value in np.quantile(scores[method], [0.025, 0.975])
            ],
            "permutation_mean": float(np.mean(permutation[method])),
            "permutation_p_ge_observed": float(
                (1 + np.sum(np.asarray(permutation[method]) >= roc_auc_score(y, probability)))
                / (len(permutation[method]) + 1)
            ),
        }
        if method != "expression_500":
            values = np.asarray(deltas[method])
            entry["delta_vs_expression"] = float(
                entry["auroc"] - result["methods"]["expression_500"]["auroc"]
            )
            entry["delta_bootstrap_ci95"] = [
                float(value) for value in np.quantile(values, [0.025, 0.975])
            ]
            entry["bootstrap_probability_delta_ge_0.05"] = float(
                np.mean(values >= 0.05)
            )
        result["methods"][method] = entry

    output = sample_meta.copy()
    output["true"] = y
    output["outer_group"] = groups
    for method, probability in predictions.items():
        output[f"{method}_probability"] = probability
    output.to_csv(OUT / "gse206391_strict_spatial_graph_predictions.csv", index=False)
    np.savez_compressed(
        OUT / "gse206391_strict_spatial_graph_resampling_distributions.npz",
        **{f"bootstrap_{method}": np.asarray(scores[method]) for method in METHODS},
        **{f"delta_bootstrap_{method}": np.asarray(deltas[method]) for method in deltas},
        **{f"label_permutation_{method}": np.asarray(permutation[method]) for method in METHODS},
        seed=np.array([20260810]),
    )
    (OUT / "gse206391_strict_spatial_graph_benchmark.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

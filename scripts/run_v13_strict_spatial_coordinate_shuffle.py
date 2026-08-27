"""Coordinate-shuffle null for the strict outer spatial-graph benchmark.

Coordinates are permuted within each sample, while counts, labels, patient
folds, fold-specific HVGs and the classifier are held fixed.  Thus the null
tests the spatial graph term itself rather than a global expression signal.
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/singlecell_v10"
sys.path.insert(0, str(ROOT / "scripts"))
import run_strict_spatial_graph_benchmark as base  # noqa: E402


def neighbors_for(coords, k=6):
    if len(coords) <= 1:
        return np.zeros((len(coords), 0), dtype=int)
    return NearestNeighbors(n_neighbors=min(k + 1, len(coords))).fit(coords).kneighbors(
        coords, return_distance=False
    )[:, 1:]


def fit_predict(z_train, y_train, z_test, seed=0):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1, solver="liblinear", class_weight="balanced", max_iter=2000, random_state=seed),
    )
    model.fit(z_train, y_train)
    return model.predict_proba(z_test)[:, 1]


def prepare_cache(matrix, spot_meta, total, sample_meta, log_cpm, groups):
    spot_groups = base.prepare_spot_groups(spot_meta, sample_meta)
    cache = []
    for held_group in np.unique(groups):
        train = groups != held_group
        top = np.argsort(log_cpm[train].var(axis=0))[-500:]
        samples = []
        for sample in sample_meta["sample"]:
            indices, neighbors = spot_groups[str(sample)]
            values = matrix[indices][:, top].toarray().astype(np.float32)
            values = np.log1p(values / np.maximum(total[indices, None], 1.0) * 1e4)
            coords = spot_meta.loc[indices, ["row", "col"]].to_numpy()
            samples.append((indices, values, coords, neighbors))
        expression = log_cpm[:, top]
        cache.append({
            "held_group": held_group,
            "train_idx": np.flatnonzero(train),
            "test_idx": np.flatnonzero(~train),
            "top": top,
            "expression": expression,
            "samples": samples,
        })
    return cache


def spatial_from_values(values, neighbors):
    if values.shape[0] < 2 or neighbors.shape[1] == 0:
        return np.zeros(values.shape[1] * 2, dtype=float)
    neighbor_mean = values[neighbors].mean(axis=1)
    centered = values - values.mean(axis=0, keepdims=True)
    neighbor_centered = centered[neighbors].mean(axis=1)
    variance = np.mean(centered * centered, axis=0)
    moran = np.divide(np.mean(centered * neighbor_centered, axis=0), variance,
                      out=np.zeros_like(variance), where=variance > 1e-10)
    contrast = np.mean(np.abs(values - neighbor_mean), axis=0)
    return np.hstack([moran, contrast]).astype(float)


def predictions(cache, y, coord_neighbors, seed=0):
    pred_spatial = np.full(len(y), np.nan)
    pred_combo = np.full(len(y), np.nan)
    for fold, shuffled in zip(cache, coord_neighbors):
        train, test = fold["train_idx"], fold["test_idx"]
        spatial = np.vstack([spatial_from_values(item[1], nb) for item, nb in zip(fold["samples"], shuffled)])
        expression = fold["expression"]
        combo = np.hstack([expression, spatial])
        pred_spatial[test] = fit_predict(spatial[train], y[train], spatial[test], seed)
        pred_combo[test] = fit_predict(combo[train], y[train], combo[test], seed)
    return pred_spatial, pred_combo


def main(n_perm=100, seed=20260810):
    matrix, spot_meta, total = base.load_spots()
    sample_meta, log_cpm, y, groups = base.patient_subset()
    cache = prepare_cache(matrix, spot_meta, total, sample_meta, log_cpm, groups)
    # Observed graph uses the same fold cache and exact neighbors.
    observed_neighbours = [[item[3] for item in fold["samples"]] for fold in cache]
    observed_spatial, observed_combo = predictions(cache, y, observed_neighbours, seed)
    observed = {
        "spatial_graph": float(roc_auc_score(y, observed_spatial)),
        "expression_plus_spatial": float(roc_auc_score(y, observed_combo)),
    }
    rng = np.random.default_rng(seed)
    null_spatial, null_combo = [], []
    # Build a fresh coordinate assignment per sample and reuse it across folds.
    # The cache has one copy per fold, so use its coordinates to maintain the
    # original spot order exactly.
    for _ in range(n_perm):
        shuffled_folds = []
        for fold in cache:
            fold_neighbors = []
            for _, _, coords, _ in fold["samples"]:
                permuted_coords = coords[rng.permutation(len(coords))]
                fold_neighbors.append(neighbors_for(permuted_coords, 6))
            shuffled_folds.append(fold_neighbors)
        p_spatial, p_combo = predictions(cache, y, shuffled_folds, seed)
        null_spatial.append(roc_auc_score(y, p_spatial))
        null_combo.append(roc_auc_score(y, p_combo))
    null_spatial, null_combo = np.asarray(null_spatial), np.asarray(null_combo)
    np.savez_compressed(
        OUT / "gse206391_strict_spatial_coordinate_shuffle_distributions.npz",
        observed_spatial=np.array([observed["spatial_graph"]]),
        observed_combo=np.array([observed["expression_plus_spatial"]]),
        null_spatial=null_spatial, null_combo=null_combo, seed=np.array([seed]),
    )
    result = {
        "dataset": "GSE206391", "n_permutations": int(n_perm),
        "shuffle_unit": "coordinates within sample; expression and labels fixed",
        "observed": observed,
        "null": {
            "spatial_graph": {"mean": float(null_spatial.mean()), "ci95": [float(np.quantile(null_spatial,.025)), float(np.quantile(null_spatial,.975))], "p_ge_observed": float((1 + np.sum(null_spatial >= observed["spatial_graph"])) / (len(null_spatial)+1))},
            "expression_plus_spatial": {"mean": float(null_combo.mean()), "ci95": [float(np.quantile(null_combo,.025)), float(np.quantile(null_combo,.975))], "p_ge_observed": float((1 + np.sum(null_combo >= observed["expression_plus_spatial"])) / (len(null_combo)+1))},
        },
    }
    (OUT / "gse206391_strict_spatial_coordinate_shuffle.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

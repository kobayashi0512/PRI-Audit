"""Fully nested cell-neighbourhood audit for GSE151177.

Unlike the historical partial strictness check, this implementation refits
sample-level HVG selection, cell scaling, PCA, and MiniBatchKMeans *inside every
held-out sample fold*.  The global arm is preserved only as an explicitly
optimistic diagnostic contrast.  No disease labels enter any representation
fitting step.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.sparse import csr_matrix
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_pri_audit_v21 as v21  # noqa: E402


ROOT = v21.ROOT
RAW = ROOT / "data" / "raw" / "GSE151177_samples"
OUT = v21.OUT
SEED = 20260821


@dataclass
class CellSample:
    name: str
    counts: csr_matrix  # cells x genes, after raw cell QC and deterministic cap
    library_size: np.ndarray


def load_cells(sample_name: str, genes_ref: np.ndarray, rng: np.random.Generator, max_cells: int) -> CellSample:
    matrix_path = RAW / f"{sample_name}_matrix.mtx.gz"
    with gzip.open(matrix_path, "rb") as handle:
        raw = mmread(handle).tocsr().astype(np.float32)
    with gzip.open(RAW / f"{sample_name}_features.tsv.gz", "rt") as handle:
        genes = np.array([line.rstrip("\n").split("\t")[0] for line in handle])
    if not np.array_equal(genes, genes_ref):
        raise ValueError(f"Raw gene order mismatch in {sample_name}")
    total = np.asarray(raw.sum(axis=0)).ravel()
    detected = np.asarray((raw > 0).sum(axis=0)).ravel()
    mitochondrial = np.char.startswith(np.char.upper(genes.astype(str)), "MT-")
    mito_fraction = np.asarray(raw[mitochondrial].sum(axis=0)).ravel() / np.maximum(total, 1)
    keep = np.flatnonzero((total >= 500) & (detected >= 300) & (mito_fraction <= 0.20))
    if keep.size > max_cells:
        keep = rng.choice(keep, size=max_cells, replace=False)
    return CellSample(name=sample_name, counts=raw[:, keep].T.tocsr(), library_size=total[keep].astype(float))


def cell_matrix(sample: CellSample, feature_ix: np.ndarray) -> np.ndarray:
    values = sample.counts[:, feature_ix].toarray().astype(np.float64, copy=False)
    return np.log1p(values / np.maximum(sample.library_size[:, None], 1.0) * 1e4)


def choose_hvg(sample_counts: np.ndarray, sample_ix: np.ndarray, n_hvg: int) -> np.ndarray:
    x = v21.log_cpm(sample_counts[sample_ix])
    return np.argsort(x.var(axis=0))[-min(n_hvg, x.shape[1]):]


def fit_dictionary(samples: list[CellSample], feature_ix: np.ndarray, clusters: int, n_components: int) -> tuple[StandardScaler, IncrementalPCA, MiniBatchKMeans]:
    """Fit all unsupervised operations in bounded memory with three passes."""
    scaler = StandardScaler()
    for sample in samples:
        scaler.partial_fit(cell_matrix(sample, feature_ix))
    pca = IncrementalPCA(n_components=n_components, batch_size=max(256, min(1500, sum(s.counts.shape[0] for s in samples))))
    for sample in samples:
        pca.partial_fit(scaler.transform(cell_matrix(sample, feature_ix)))
    kmeans = MiniBatchKMeans(n_clusters=clusters, random_state=0, batch_size=2048, n_init=10)
    for sample in samples:
        z = pca.transform(scaler.transform(cell_matrix(sample, feature_ix)))
        kmeans.partial_fit(z)
    return scaler, pca, kmeans


def histogram(sample: CellSample, feature_ix: np.ndarray, scaler: StandardScaler, pca: IncrementalPCA, kmeans: MiniBatchKMeans) -> tuple[np.ndarray, np.ndarray]:
    z = pca.transform(scaler.transform(cell_matrix(sample, feature_ix)))
    labels = kmeans.predict(z)
    return np.bincount(labels, minlength=kmeans.n_clusters) / max(labels.size, 1), labels


def loo_probabilities(histograms: np.ndarray, labels: np.ndarray) -> np.ndarray:
    probabilities = np.full(labels.size, np.nan)
    for held in range(labels.size):
        train = np.arange(labels.size) != held
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1, solver="liblinear", class_weight="balanced", max_iter=2000, random_state=0),
        )
        model.fit(histograms[train], labels[train])
        probabilities[held] = model.predict_proba(histograms[held : held + 1])[:, 1][0]
    return probabilities


def fold_probabilities(folds: list[dict[str, object]], labels: np.ndarray) -> np.ndarray:
    probabilities = np.full(labels.size, np.nan)
    for fold in folds:
        train_ix = fold["train_ix"]
        test_ix = fold["test_ix"]
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1, solver="liblinear", class_weight="balanced", max_iter=2000, random_state=0),
        )
        model.fit(fold["hist_train"], labels[train_ix])
        probabilities[test_ix] = model.predict_proba(fold["hist_test"])[:, 1]
    if np.isnan(probabilities).any():
        raise RuntimeError("Missing strict-fold probability")
    return probabilities


def bootstrap_gap(labels: np.ndarray, global_probability: np.ndarray, strict_probability: np.ndarray, rng: np.random.Generator, n_boot: int) -> np.ndarray:
    values = []
    for _ in range(n_boot):
        ix = rng.integers(0, labels.size, size=labels.size)
        if np.unique(labels[ix]).size == 2:
            values.append(roc_auc_score(labels[ix], global_probability[ix]) - roc_auc_score(labels[ix], strict_probability[ix]))
    return np.asarray(values)


def global_cell_shuffle(cluster_labels: list[np.ndarray], labels: np.ndarray, clusters: int, rng: np.random.Generator, n_shuffle: int) -> np.ndarray:
    observed_sizes = [len(item) for item in cluster_labels]
    pooled = np.concatenate(cluster_labels)
    null = []
    for _ in range(n_shuffle):
        shuffled = rng.permutation(pooled)
        hists, start = [], 0
        for size in observed_sizes:
            current = shuffled[start : start + size]
            hists.append(np.bincount(current, minlength=clusters) / max(size, 1))
            start += size
        null.append(roc_auc_score(labels, loo_probabilities(np.vstack(hists), labels)))
    return np.asarray(null)


def run_cluster_count(samples: list[CellSample], labels: np.ndarray, sample_counts: np.ndarray, clusters: int, n_hvg: int, n_boot: int, n_shuffle: int, rng: np.random.Generator) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    all_ix = np.arange(len(samples))
    # Intentionally global diagnostic arm: only this arm can touch all samples.
    global_hvg = choose_hvg(sample_counts, all_ix, n_hvg)
    global_scaler, global_pca, global_kmeans = fit_dictionary(samples, global_hvg, clusters, n_components=15)
    global_hists, global_clusters = [], []
    for sample in samples:
        hist, assignments = histogram(sample, global_hvg, global_scaler, global_pca, global_kmeans)
        global_hists.append(hist)
        global_clusters.append(assignments)
    global_hists_array = np.vstack(global_hists)
    global_probability = loo_probabilities(global_hists_array, labels)

    # Fully nested strict arm: all four transformations are refit without the
    # held-out sample.  Cluster identities are fold-specific, which is fine
    # because each held-out prediction is scored by its own fold-specific model.
    folds: list[dict[str, object]] = []
    for held in all_ix:
        train_ix = all_ix != held
        feature_ix = choose_hvg(sample_counts, all_ix[train_ix], n_hvg)
        scaler, pca, kmeans = fit_dictionary([samples[index] for index in all_ix[train_ix]], feature_ix, clusters, n_components=15)
        hist_train = np.vstack([histogram(samples[index], feature_ix, scaler, pca, kmeans)[0] for index in all_ix[train_ix]])
        hist_test, _ = histogram(samples[held], feature_ix, scaler, pca, kmeans)
        folds.append({"train_ix": all_ix[train_ix], "test_ix": np.array([held]), "hist_train": hist_train, "hist_test": hist_test[None, :]})
    strict_probability = fold_probabilities(folds, labels)
    global_auc = float(roc_auc_score(labels, global_probability))
    strict_auc = float(roc_auc_score(labels, strict_probability))
    gap = bootstrap_gap(labels, global_probability, strict_probability, rng, n_boot)
    cell_null = global_cell_shuffle(global_clusters, labels, clusters, rng, n_shuffle)

    record = {
        "dataset": "GSE151177", "clusters": clusters, "cells_per_sample_cap": 1500, "hvg_per_fold": n_hvg,
        "global_auroc": global_auc, "fully_nested_strict_auroc": strict_auc,
        "global_minus_fully_nested_optimism_gap": global_auc - strict_auc,
        "sample_bootstrap_ci95_low": float(np.quantile(gap, .025)), "sample_bootstrap_ci95_high": float(np.quantile(gap, .975)),
        "sample_bootstrap_probability_gap_ge_0_05": float(np.mean(gap >= .05)), "n_bootstrap": int(gap.size),
        "global_cell_shuffle_null_mean": float(cell_null.mean()), "global_cell_shuffle_null_ci95_low": float(np.quantile(cell_null, .025)),
        "global_cell_shuffle_null_ci95_high": float(np.quantile(cell_null, .975)),
        "global_cell_shuffle_p_ge_observed": float((1 + np.sum(cell_null >= global_auc)) / (cell_null.size + 1)), "n_cell_shuffle": int(cell_null.size),
    }
    arrays = {
        "labels": labels, "global_probability": global_probability, "fully_nested_strict_probability": strict_probability,
        "bootstrap_global_minus_strict_gap": gap, "global_cell_shuffle_auroc": cell_null,
        "global_histograms": global_hists_array,
    }
    return record, arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--hvg", type=int, default=1000)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--cell-shuffles", type=int, default=500)
    parser.add_argument("--max-cells", type=int, default=1500)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    data = np.load(OUT / "gse151177_sample_gene_counts.npz")
    counts, genes = data["counts"].astype(float), data["genes"]
    metadata = pd.read_csv(OUT / "gse151177_sample_metadata.csv")
    if not metadata.condition.isin(["control", "psoriasis"]).all():
        raise ValueError("Unexpected GSE151177 condition in v21 sample metadata")
    labels = (metadata.condition == "psoriasis").astype(int).to_numpy()
    rng = np.random.default_rng(SEED)
    samples = [load_cells(sample, genes, rng, args.max_cells) for sample in metadata["sample"]]
    if len(samples) != len(metadata):
        raise RuntimeError("Raw cell/sample reconstruction mismatch")
    records = []
    for clusters in args.clusters:
        record, arrays = run_cluster_count(samples, labels, counts, clusters, args.hvg, args.bootstrap, args.cell_shuffles, rng)
        records.append(record)
        np.savez_compressed(OUT / f"gse151177_v21_nested_neighborhood_k{clusters}.npz", samples=metadata["sample"].to_numpy(), **arrays)
    frame = pd.DataFrame(records)
    frame.to_csv(OUT / "v21_nested_neighborhood_strictness.csv", index=False)
    (OUT / "v21_nested_neighborhood_strictness.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()

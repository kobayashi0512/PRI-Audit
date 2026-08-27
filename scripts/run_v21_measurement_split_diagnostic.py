"""Demonstrate the estimand error caused by cell/spot-level random splitting.

This is intentionally *not* an additional candidate PRI-Audit model.  In
single-cell and spatial datasets, many measurements arise from the same biopsy
or patient.  A random row split therefore places measurements from every test
sample in the training data.  It estimates within-sample measurement
interpolation, not prediction for a new patient.

The script fixes this deliberately permissive recipe up front: global HVG
selection and global PCA, followed by 50 stratified random measurement splits.
The output records the overlap explicitly so the resulting AUROCs cannot be
mistaken for patient-level generalisation.  The legitimate comparator remains
the patient/sample-blocked strict PRI-Audit results in ``v21_loose_vs_strict``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.sparse import csr_matrix, vstack
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_pri_audit_v21 as v21  # noqa: E402
import run_strict_spatial_graph_benchmark as spatial_legacy  # noqa: E402


OUT = v21.OUT
RAW_151 = v21.RAW / "GSE151177_samples"
SEED = 20260823


def log_normalise(values: np.ndarray, library: np.ndarray) -> np.ndarray:
    return np.log1p(values / np.maximum(library[:, None], 1.0) * 1e4)


def load_gse151177(max_cells_per_sample: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Return globally normalised cell rows from retained source samples."""
    sample_meta = pd.read_csv(OUT / "gse151177_sample_metadata.csv")
    counts_ref = np.load(OUT / "gse151177_sample_gene_counts.npz")
    genes_ref = counts_ref["genes"].astype(str)
    selected = sample_meta[sample_meta.condition.isin(["control", "psoriasis"])].copy()
    pieces, labels, samples = [], [], []
    for row in selected.itertuples(index=False):
        path = RAW_151 / f"{row.sample}_matrix.mtx.gz"
        with gzip.open(path, "rb") as handle:
            raw = mmread(handle).tocsr().astype(np.float32)
        with gzip.open(RAW_151 / f"{row.sample}_features.tsv.gz", "rt") as handle:
            genes = np.asarray([line.rstrip("\n").split("\t")[0] for line in handle], dtype=str)
        if not np.array_equal(genes, genes_ref):
            raise ValueError(f"Gene ordering mismatch: {row.sample}")
        library = np.asarray(raw.sum(axis=0)).ravel()
        detected = np.asarray((raw > 0).sum(axis=0)).ravel()
        mitochondrial = np.char.startswith(np.char.upper(genes), "MT-")
        mito_fraction = np.asarray(raw[mitochondrial].sum(axis=0)).ravel() / np.maximum(library, 1.0)
        keep = np.flatnonzero((library >= 500) & (detected >= 300) & (mito_fraction <= 0.20))
        if keep.size > max_cells_per_sample:
            keep = np.sort(rng.choice(keep, size=max_cells_per_sample, replace=False))
        pieces.append(raw[:, keep].T.tocsr())
        labels.append(np.full(keep.size, int(row.condition == "psoriasis"), dtype=np.int8))
        samples.append(np.full(keep.size, row.sample, dtype=object))
    # Keep the full 33k-gene cell matrix sparse until the 1,000 globally chosen
    # features are extracted.  This makes the intentionally loose diagnostic
    # runnable on a normal workstation without changing its recipe.
    matrix = vstack(pieces, format="csr")
    library = np.asarray(matrix.sum(axis=1)).ravel()
    feature_score = np.asarray(matrix.multiply(matrix).mean(axis=0)).ravel() - np.square(np.asarray(matrix.mean(axis=0)).ravel())
    hvg = np.argsort(feature_score)[-1000:]
    x = log_normalise(matrix[:, hvg].toarray().astype(np.float32), library).astype(np.float32)
    return x, np.concatenate(labels), np.concatenate(samples), {
        "dataset": "GSE151177", "measurement_unit": "single cell", "n_source_samples": int(selected.shape[0]),
        "n_measurements": int(x.shape[0]), "max_measurements_per_sample": max_cells_per_sample,
        "n_global_hvgs": int(hvg.size),
    }


def load_gse206391(max_spots_per_sample: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Return globally normalised spot rows from paired LS/NL source biopsies."""
    spatial_legacy.OUT = OUT
    matrix, spot_meta, total = spatial_legacy.load_spots()
    sample_meta = pd.read_csv(OUT / "gse206391_sample_metadata.csv")
    eligible = sample_meta[sample_meta.condition.isin(["NON LESIONAL", "LESIONAL"])]
    paired = eligible.groupby("patient").condition.nunique()
    paired_patients = paired[paired == 2].index
    selected = eligible[eligible.patient.isin(paired_patients)].copy()
    indices, labels, samples = [], [], []
    for row in selected.itertuples(index=False):
        keep = spot_meta.index[(spot_meta["sample"] == row.sample) & spot_meta.keep].to_numpy()
        if keep.size > max_spots_per_sample:
            keep = np.sort(rng.choice(keep, size=max_spots_per_sample, replace=False))
        indices.append(keep)
        labels.append(np.full(keep.size, int(row.condition == "LESIONAL"), dtype=np.int8))
        samples.append(np.full(keep.size, row.sample, dtype=object))
    rows = np.concatenate(indices)
    view = matrix[rows]
    library = total[rows]
    feature_score = np.asarray(view.multiply(view).mean(axis=0)).ravel() - np.square(np.asarray(view.mean(axis=0)).ravel())
    hvg = np.argsort(feature_score)[-1000:]
    x = log_normalise(view[:, hvg].toarray().astype(np.float32), library).astype(np.float32)
    return x, np.concatenate(labels), np.concatenate(samples), {
        "dataset": "GSE206391", "measurement_unit": "spatial spot", "n_source_samples": int(selected.shape[0]),
        "n_source_patients": int(selected.patient.nunique()), "n_measurements": int(x.shape[0]),
        "max_measurements_per_sample": max_spots_per_sample, "n_global_hvgs": int(hvg.size),
    }


def run_random_measurement_split(x: np.ndarray, labels: np.ndarray, sample_ids: np.ndarray, info: dict[str, object], repeats: int) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Fit the predeclared global PCA and repeat invalid random row splits."""
    n_components = min(50, x.shape[0] - 1, x.shape[1])
    embedding = PCA(n_components=n_components, svd_solver="randomized", random_state=SEED).fit_transform(StandardScaler().fit_transform(x))
    splitter = StratifiedShuffleSplit(n_splits=repeats, test_size=.20, random_state=SEED)
    rows, probabilities, truths, test_samples = [], [], [], []
    for repeat, (train, test) in enumerate(splitter.split(embedding, labels)):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1, solver="liblinear", class_weight="balanced", max_iter=2000, random_state=SEED + repeat),
        )
        model.fit(embedding[train], labels[train])
        probability = model.predict_proba(embedding[test])[:, 1]
        train_samples, held_samples = set(sample_ids[train]), set(sample_ids[test])
        rows.append({
            "dataset": info["dataset"], "measurement_unit": info["measurement_unit"], "repeat": repeat,
            "random_measurement_split_auroc": float(roc_auc_score(labels[test], probability)),
            "n_train_measurements": int(train.size), "n_test_measurements": int(test.size),
            "n_train_source_samples": len(train_samples), "n_test_source_samples": len(held_samples),
            "test_source_sample_overlap_with_train": len(train_samples & held_samples) / max(len(held_samples), 1),
            "test_measurements_from_source_samples_seen_in_train": float(np.mean(np.isin(sample_ids[test], list(train_samples)))),
            "preprocessing_scope": "all measurements (global HVG, normalisation and PCA)",
            "estimand": "within-source-measurement interpolation; NOT new-patient prediction",
        })
        probabilities.append(probability)
        truths.append(labels[test])
        test_samples.append(sample_ids[test])
    return pd.DataFrame(rows), {
        "probabilities": np.concatenate(probabilities), "truth": np.concatenate(truths),
        "source_sample": np.concatenate(test_samples), "embedding_components": np.array([n_components]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--cells-per-sample", type=int, default=1500)
    parser.add_argument("--spots-per-sample", type=int, default=600)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    all_rows, distributions, manifest = [], {}, []
    for loader, cap in [(load_gse151177, args.cells_per_sample), (load_gse206391, args.spots_per_sample)]:
        x, labels, sample_ids, info = loader(cap, rng)
        result, arrays = run_random_measurement_split(x, labels, sample_ids, info, args.repeats)
        all_rows.append(result)
        distributions[info["dataset"]] = arrays
        manifest.append(info)
    table = pd.concat(all_rows, ignore_index=True)
    table.to_csv(OUT / "v21_measurement_random_split_diagnostic.csv", index=False)
    np.savez_compressed(
        OUT / "v21_measurement_random_split_distributions.npz",
        **{f"{dataset}_{key}": value for dataset, values in distributions.items() for key, value in values.items()},
        seed=np.array([SEED]),
    )
    summary = {
        "analysis": "v21 random cell/spot split diagnostic", "seed": SEED, "repeats": args.repeats,
        "recipe": "global HVG selection + global normalisation/PCA + stratified random measurement split",
        "warning": "This intentionally violates source-sample and patient independence. It is a diagnostic of an invalid estimand, not a primary performance estimate.",
        "datasets": manifest,
        "summary": table.groupby(["dataset", "measurement_unit"], as_index=False).agg(
            median_auroc=("random_measurement_split_auroc", "median"),
            auroc_q025=("random_measurement_split_auroc", lambda v: float(np.quantile(v, .025))),
            auroc_q975=("random_measurement_split_auroc", lambda v: float(np.quantile(v, .975))),
            source_sample_overlap=("test_source_sample_overlap_with_train", "mean"),
            measurement_overlap=("test_measurements_from_source_samples_seen_in_train", "mean"),
        ).to_dict(orient="records"),
    }
    (OUT / "v21_measurement_random_split_diagnostic.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

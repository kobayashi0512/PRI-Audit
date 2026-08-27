"""PRI-Audit v21: raw reconstruction, strict/loose contrast, and external bulk test.

This is an outcome-agnostic, fixed-recipe rerun.  It creates a new result tree
and never overwrites the historical ``singlecell_v10`` outputs used to develop
the manuscript.  The script deliberately reports the globally fitted low-rank
arm as a diagnostic (leakage) contrast, not as a competing primary model.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import platform
import re
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.io import mmread
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF, PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
LEGACY = ROOT / "results" / "singlecell_v10"
OUT = ROOT / "results" / "pri_audit_v21"
SEED = 20260819


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, content: object) -> None:
    path.write_text(json.dumps(content, indent=2, sort_keys=True), encoding="utf-8")


def decode(values: np.ndarray) -> np.ndarray:
    return np.array([value.decode() if isinstance(value, bytes) else str(value) for value in values])


def load_10x_sample(path: Path, source: Path) -> tuple[str, np.ndarray, np.ndarray, int]:
    """Return gene order, QC-filtered sample sum, and retained cell count."""
    stem = path.name.replace("_matrix.mtx.gz", "")
    with gzip.open(path, "rb") as handle:
        matrix = mmread(handle).tocsr().astype(np.float32)
    with gzip.open(source / f"{stem}_features.tsv.gz", "rt") as handle:
        genes = np.array([line.rstrip("\n").split("\t")[0] for line in handle])
    total = np.asarray(matrix.sum(axis=0)).ravel()
    detected = np.asarray((matrix > 0).sum(axis=0)).ravel()
    mito = np.char.startswith(np.char.upper(genes.astype(str)), "MT-")
    mito_fraction = np.asarray(matrix[mito].sum(axis=0)).ravel() / np.maximum(total, 1)
    keep = (total >= 500) & (detected >= 300) & (mito_fraction <= 0.20)
    return stem, genes, np.asarray(matrix[:, keep].sum(axis=1)).ravel(), int(keep.sum())


def rebuild_gse151177(out: Path) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    source = RAW / "GSE151177_samples"
    qc = pd.read_csv(LEGACY / "gse151177_sample_qc.csv").set_index("sample")
    matrices, rows, genes_ref = [], [], None
    for matrix_path in sorted(source.glob("*_matrix.mtx.gz")):
        stem, genes, sample_sum, retained = load_10x_sample(matrix_path, source)
        q = qc.loc[stem]
        if float(q.median_mito_fraction) > 0.20 or float(q.median_detected_genes) < 300:
            continue
        if genes_ref is None:
            genes_ref = genes
        elif not np.array_equal(genes_ref, genes):
            raise ValueError(f"GSE151177 gene ordering differs in {stem}")
        rows.append({
            "sample": stem,
            "condition": "control" if "Control" in stem else "psoriasis",
            "filtered_cells": retained,
        })
        matrices.append(sample_sum)
    counts = np.vstack(matrices).astype(np.float64)
    meta = pd.DataFrame(rows)
    np.savez_compressed(out / "gse151177_sample_gene_counts.npz", counts=counts, genes=genes_ref)
    meta.to_csv(out / "gse151177_sample_metadata.csv", index=False)
    return counts, meta, genes_ref


MITO_ENSEMBL = {
    "ENSG00000198888", "ENSG00000198763", "ENSG00000198804", "ENSG00000198712",
    "ENSG00000228263", "ENSG00000198899", "ENSG00000198938", "ENSG00000198840", "ENSG00000198727",
    "ENSG00000198886", "ENSG00000198786", "ENSG00000198695", "ENSG00000198700",
    "ENSG00000210049", "ENSG00000211459",
}


def rebuild_gse173706(out: Path) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    source = RAW / "GSE173706_samples"
    qc = pd.read_csv(LEGACY / "gse173706_sample_qc.csv").set_index("sample")
    matrices, rows, genes_ref = [], [], None
    for sample_path in sorted(source.glob("*.csv.gz")):
        sample = sample_path.name.replace(".csv.gz", "")
        # These dense CSVs can contain tens of thousands of cells.  The first
        # implementation retained every gene-by-cell row in memory while
        # discovering QC, which is needlessly quadratic in the number of
        # entries and can be killed by a normal workstation memory limit.
        # Stream twice instead: pass 1 finds retained cells; pass 2 aggregates
        # each gene immediately.  This is slower but faithful and bounded.
        with gzip.open(sample_path, "rt", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            n_cells = len(header) - 1
            total = np.zeros(n_cells)
            detected = np.zeros(n_cells, dtype=np.int32)
            mito = np.zeros(n_cells)
            for row in reader:
                if not row:
                    continue
                gene = row[0].upper()
                values = np.asarray(row[1:], dtype=np.float64)
                total += values
                detected += values > 0
                if gene in MITO_ENSEMBL or gene.startswith("MT-"):
                    mito += values
        keep = (total >= 500) & (detected >= 300) & (mito / np.maximum(total, 1) <= 0.20)
        genes_list, sums_list = [], []
        with gzip.open(sample_path, "rt", newline="") as handle:
            reader = csv.reader(handle)
            next(reader)
            for row in reader:
                if not row:
                    continue
                genes_list.append(row[0].upper())
                values = np.asarray(row[1:], dtype=np.float64)
                sums_list.append(float(values[keep].sum()))
        genes = np.asarray(genes_list)
        sample_sum = np.asarray(sums_list)
        if genes_ref is None:
            genes_ref = genes
        elif not np.array_equal(genes_ref, genes):
            raise ValueError(f"GSE173706 gene ordering differs in {sample}")
        label = sample.split("_", 1)[1]
        condition = (
            "normal_skin" if label.startswith("NS-") else
            "peripheral_normal" if label.startswith("PN-") else "psoriatic"
        )
        donor = re.sub(r"^(NS|PN|PP)-", "", label)
        rows.append({
            "sample": sample,
            "condition": condition,
            "donor": donor,
            "filtered_cells": int(qc.loc[sample, "filtered_cells"]),
        })
        matrices.append(sample_sum)
    counts = np.vstack(matrices).astype(np.float64)
    meta = pd.DataFrame(rows)
    np.savez_compressed(out / "gse173706_sample_gene_counts.npz", counts=counts, genes=genes_ref)
    meta.to_csv(out / "gse173706_sample_metadata.csv", index=False)
    return counts, meta, genes_ref


def rebuild_gse206391(out: Path) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    source = RAW / "GSE206391_Preprocessed_data.h5"
    with h5py.File(source, "r") as handle:
        data = handle["/X/data"][...]
        indices = handle["/X/indices"][...]
        indptr = handle["/X/indptr"][...]
        matrix = csr_matrix((data, indices, indptr), shape=(len(indptr) - 1, 16685))
        obs = {key: handle[f"/obs/{key}"][...] for key in ["patient", "biopsy_type", "sample", "specimen", "array_row", "array_col", "in_tissue"]}
        categories = {key: decode(handle[f"/obs/__categories/{key}"][...]) for key in ["patient", "biopsy_type", "sample", "specimen"]}
        genes = decode(handle["/raw/var/gene_id"][...])
        mitochondrial = handle["/raw/var/mt"][...].astype(bool)
    def categorical(key: str) -> np.ndarray:
        return categories[key][obs[key].astype(int)]
    meta = pd.DataFrame({
        "patient": categorical("patient"),
        "condition": categorical("biopsy_type"),
        "sample": categorical("sample"),
        "specimen": categorical("specimen"),
        "row": obs["array_row"],
        "col": obs["array_col"],
        "in_tissue": obs["in_tissue"],
    })
    total = np.asarray(matrix.sum(axis=1)).ravel()
    detected = np.asarray((matrix > 0).sum(axis=1)).ravel()
    mitochondrial_fraction = np.asarray(matrix[:, mitochondrial].sum(axis=1)).ravel() / np.maximum(total, 1)
    meta["keep"] = (
        (total >= 500) & (detected >= 300) & (mitochondrial_fraction <= 0.20) & meta.in_tissue.astype(bool).to_numpy()
    )
    rows, matrices = [], []
    for sample, group in meta.groupby("sample", sort=True):
        ix = group.index[group.keep].to_numpy()
        rows.append({
            "sample": sample, "patient": group.patient.iloc[0], "condition": group.condition.iloc[0],
            "raw_spots": int(len(group)), "filtered_spots": int(len(ix)),
        })
        matrices.append(np.asarray(matrix[ix].sum(axis=0)).ravel())
    counts = np.vstack(matrices).astype(np.float64)
    sample_meta = pd.DataFrame(rows)
    np.savez_compressed(out / "gse206391_sample_gene_counts.npz", counts=counts, genes=genes)
    sample_meta.to_csv(out / "gse206391_sample_metadata.csv", index=False)
    return counts, sample_meta, genes


def inputs_manifest() -> dict[str, object]:
    files = [
        RAW / "GSE206391_Preprocessed_data.h5",
        RAW / "GSE47751_series_matrix_fresh.txt.gz",
        LEGACY / "gse151177_sample_qc.csv",
        LEGACY / "gse173706_sample_qc.csv",
    ]
    for directory in [RAW / "GSE151177_samples", RAW / "GSE173706_samples"]:
        files.extend(sorted(directory.glob("*.gz")))
    return {
        "analysis_version": "pri_audit_v21",
        "seed": SEED,
        "inputs": [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size} for path in files],
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__, "h5py": h5py.__version__,
        },
    }


def log_cpm(counts: np.ndarray) -> np.ndarray:
    return np.log1p(counts / np.maximum(counts.sum(axis=1, keepdims=True), 1) * 1e6)


def eligible_core(stem: str, counts: np.ndarray, metadata: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, bool]:
    if stem == "gse151177":
        keep = metadata.condition.isin(["control", "psoriasis"]).to_numpy()
        y = (metadata.loc[keep, "condition"] == "psoriasis").astype(int).to_numpy()
        groups = metadata.loc[keep, "sample"].astype(str).to_numpy()
        paired = False
    elif stem == "gse173706":
        subset = metadata[metadata.condition.isin(["peripheral_normal", "psoriatic"])]
        paired_donors = subset.groupby("donor").condition.nunique()
        donors = paired_donors[paired_donors == 2].index
        keep = (metadata.donor.isin(donors) & metadata.condition.isin(["peripheral_normal", "psoriatic"])).to_numpy()
        y = (metadata.loc[keep, "condition"] == "psoriatic").astype(int).to_numpy()
        groups = metadata.loc[keep, "donor"].astype(str).to_numpy()
        paired = True
    elif stem == "gse206391":
        subset = metadata[metadata.condition.isin(["NON LESIONAL", "LESIONAL"])]
        paired_patients = subset.groupby("patient").condition.nunique()
        patients = paired_patients[paired_patients == 2].index
        keep = (metadata.patient.isin(patients) & metadata.condition.isin(["NON LESIONAL", "LESIONAL"])).to_numpy()
        y = (metadata.loc[keep, "condition"] == "LESIONAL").astype(int).to_numpy()
        groups = metadata.loc[keep, "patient"].astype(str).to_numpy()
        paired = True
    else:
        raise ValueError(stem)
    return log_cpm(counts[keep]), y, groups, metadata.loc[keep].reset_index(drop=True), paired


def fit_strict_representations(train: np.ndarray, test: np.ndarray, allow_nmf: bool) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    top = np.argsort(train.var(axis=0))[-min(2000, train.shape[1]):]
    train_hvg, test_hvg = train[:, top], test[:, top]
    output: dict[str, tuple[np.ndarray, np.ndarray]] = {"expression": (train_hvg, test_hvg)}
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_hvg)
    scaled_test = scaler.transform(test_hvg)
    n_components = min(8, scaled_train.shape[0] - 1, scaled_train.shape[1])
    pca = PCA(n_components=n_components, random_state=0)
    output["pca"] = (pca.fit_transform(scaled_train), pca.transform(scaled_test))
    if allow_nmf:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            nmf = NMF(n_components=n_components, init="nndsvda", random_state=0, max_iter=2000)
            output["nmf"] = (nmf.fit_transform(train_hvg), nmf.transform(test_hvg))
    return output


def fit_loose_representations(x: np.ndarray, allow_nmf: bool) -> dict[str, np.ndarray]:
    top = np.argsort(x.var(axis=0))[-min(2000, x.shape[1]):]
    hvg = x[:, top]
    output: dict[str, np.ndarray] = {"expression": hvg}
    scaler = StandardScaler()
    scaled = scaler.fit_transform(hvg)
    n_components = min(8, scaled.shape[0] - 1, scaled.shape[1])
    output["pca"] = PCA(n_components=n_components, random_state=0).fit_transform(scaled)
    if allow_nmf:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            output["nmf"] = NMF(n_components=n_components, init="nndsvda", random_state=0, max_iter=2000).fit_transform(hvg)
    return output


def classify_outer(folds: list[dict[str, object]], labels: np.ndarray, method: str) -> np.ndarray:
    probability = np.full(labels.size, np.nan)
    for fold in folds:
        train_ix = fold["train_ix"]
        test_ix = fold["test_ix"]
        y_train = labels[train_ix]
        if np.unique(y_train).size < 2:
            raise RuntimeError("A held-out group leaves only one training class")
        train, test = fold["representations"][method]
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1, solver="liblinear", class_weight="balanced", max_iter=2000, random_state=0))
        model.fit(train, y_train)
        probability[test_ix] = model.predict_proba(test)[:, 1]
    if np.isnan(probability).any():
        raise RuntimeError(f"Missing outer predictions for {method}")
    return probability


def group_bootstrap(y: np.ndarray, probabilities: np.ndarray, groups: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[list[float], np.ndarray]:
    units = np.unique(groups)
    indices = {unit: np.flatnonzero(groups == unit) for unit in units}
    values = []
    for _ in range(n_boot):
        chosen = rng.choice(units, size=units.size, replace=True)
        ix = np.concatenate([indices[unit] for unit in chosen])
        if np.unique(y[ix]).size == 2:
            values.append(roc_auc_score(y[ix], probabilities[ix]))
    array = np.asarray(values)
    return [float(x) for x in np.quantile(array, [0.025, 0.975])], array


def bootstrap_delta(y: np.ndarray, candidate: np.ndarray, reference: np.ndarray, groups: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[list[float], np.ndarray]:
    units = np.unique(groups)
    indices = {unit: np.flatnonzero(groups == unit) for unit in units}
    values = []
    for _ in range(n_boot):
        chosen = rng.choice(units, size=units.size, replace=True)
        ix = np.concatenate([indices[unit] for unit in chosen])
        if np.unique(y[ix]).size == 2:
            values.append(roc_auc_score(y[ix], candidate[ix]) - roc_auc_score(y[ix], reference[ix]))
    array = np.asarray(values)
    return [float(x) for x in np.quantile(array, [0.025, 0.975])], array


def permuted_labels(labels: np.ndarray, groups: np.ndarray, paired: bool, rng: np.random.Generator) -> np.ndarray:
    if not paired:
        return rng.permutation(labels)
    shuffled = labels.copy()
    for group in np.unique(groups):
        if rng.random() < 0.5:
            ix = groups == group
            shuffled[ix] = 1 - shuffled[ix]
    return shuffled


def evaluate_strict_loose(
    name: str, x: np.ndarray, y: np.ndarray, groups: np.ndarray, metadata: pd.DataFrame,
    paired: bool, allow_nmf: bool, n_permutations: int, n_bootstrap: int, out: Path,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Run fixed strict and globally-preprocessed outer-group classifiers."""
    methods = ["expression", "pca"] + (["nmf"] if allow_nmf else [])
    folds = []
    for group in np.unique(groups):
        test_ix = np.flatnonzero(groups == group)
        train_ix = np.flatnonzero(groups != group)
        reps = fit_strict_representations(x[train_ix], x[test_ix], allow_nmf)
        folds.append({"group": str(group), "train_ix": train_ix, "test_ix": test_ix, "representations": reps})
    strict_probability = {method: classify_outer(folds, y, method) for method in methods}

    global_reps = fit_loose_representations(x, allow_nmf)
    loose_folds = [
        {"group": str(group), "train_ix": np.flatnonzero(groups != group), "test_ix": np.flatnonzero(groups == group),
         "representations": {method: (global_reps[method][groups != group], global_reps[method][groups == group]) for method in methods}}
        for group in np.unique(groups)
    ]
    loose_probability = {method: classify_outer(loose_folds, y, method) for method in methods}

    rng = np.random.default_rng(SEED)
    summary_rows: list[dict[str, object]] = []
    all_probabilities = {f"strict_{method}": values for method, values in strict_probability.items()} | {f"loose_{method}": values for method, values in loose_probability.items()}
    permutation_scores = {method: [] for method in all_probabilities}
    for _ in range(n_permutations):
        shuffled = permuted_labels(y, groups, paired, rng)
        for method in methods:
            permutation_scores[f"strict_{method}"].append(float(roc_auc_score(shuffled, classify_outer(folds, shuffled, method))))
            permutation_scores[f"loose_{method}"].append(float(roc_auc_score(shuffled, classify_outer(loose_folds, shuffled, method))))

    bootstrap_values: dict[str, np.ndarray] = {}
    for arm, probability in all_probabilities.items():
        ci, values = group_bootstrap(y, probability, groups, rng, n_bootstrap)
        bootstrap_values[arm] = values
        observed = float(roc_auc_score(y, probability))
        null = np.asarray(permutation_scores[arm])
        summary_rows.append({
            "dataset": name, "arm": arm, "representation_fit_scope": "outer_group" if arm.startswith("strict_") else "global_all_eligible_samples",
            "formal_role": "primary" if arm.startswith("strict_") else "diagnostic_leakage_contrast",
            "auroc": observed, "balanced_accuracy": float(balanced_accuracy_score(y, probability >= 0.5)),
            "group_bootstrap_ci95_low": ci[0], "group_bootstrap_ci95_high": ci[1], "n_bootstrap": int(values.size),
            "permutation_mean": float(null.mean()), "permutation_ci95_low": float(np.quantile(null, .025)),
            "permutation_ci95_high": float(np.quantile(null, .975)),
            "permutation_p_ge_observed": float((1 + np.sum(null >= observed)) / (n_permutations + 1)), "n_permutations": n_permutations,
        })
    for method in methods:
        for arm, probability, ref_name in [
            (f"strict_{method}", strict_probability[method], "strict_expression"),
            (f"loose_{method}", loose_probability[method], "strict_expression"),
        ]:
            ci, values = bootstrap_delta(y, probability, strict_probability["expression"], groups, rng, n_bootstrap)
            for row in summary_rows:
                if row["arm"] == arm:
                    row.update({
                        "reference_arm": ref_name,
                        "incremental_auroc_vs_strict_expression": float(roc_auc_score(y, probability) - roc_auc_score(y, strict_probability["expression"])),
                        "increment_bootstrap_ci95_low": ci[0], "increment_bootstrap_ci95_high": ci[1],
                        "increment_bootstrap_probability_ge_0_05": float(np.mean(values >= .05)),
                    })
    predictions = metadata.copy()
    predictions["true"] = y
    predictions["outer_group"] = groups
    for arm, probability in all_probabilities.items():
        predictions[f"{arm}_probability"] = probability
    predictions.to_csv(out / f"{name}_strict_loose_predictions.csv", index=False)
    np.savez_compressed(out / f"{name}_strict_loose_nulls.npz", **{key: np.asarray(value) for key, value in permutation_scores.items()}, **{f"bootstrap_{key}": value for key, value in bootstrap_values.items()})
    return {"dataset": name, "n_samples": int(y.size), "n_groups": int(np.unique(groups).size), "paired": paired, "methods": summary_rows}, predictions


def geo_header(path: Path) -> dict[str, list[str]]:
    header: dict[str, list[str]] = {}
    with gzip.open(path, "rt", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if not row:
                continue
            if row[0].rstrip("\r\n") == "!series_matrix_table_begin":
                break
            if row[0].startswith("!Sample_"):
                header[row[0][1:]] = row[1:]
    return header


def table_skiprows(path: Path) -> int:
    with gzip.open(path, "rt") as handle:
        for line_number, line in enumerate(handle):
            if line.rstrip("\n") == "!series_matrix_table_begin":
                return line_number + 1
    raise ValueError(f"Missing series matrix table in {path}")


def rebuild_gse47751(out: Path) -> tuple[np.ndarray, pd.DataFrame]:
    path = RAW / "GSE47751_series_matrix_fresh.txt.gz"
    header = geo_header(path)
    accession = header["Sample_geo_accession"]
    titles = header["Sample_title"]
    info = pd.DataFrame({"geo_accession": accession, "title": titles})
    parsed = info.title.str.extract(r"^(?P<patient>[A-Z]+\d+)_T\d+_(?P<state>PN|PP)_day(?P<day>\d+)_")
    info = pd.concat([info, parsed], axis=1)
    baseline = info[(info.day == "0") & info.state.isin(["PN", "PP"])].copy()
    paired = baseline.groupby("patient").state.nunique()
    baseline = baseline[baseline.patient.isin(paired[paired == 2].index)].copy().reset_index(drop=True)
    matrix = pd.read_csv(path, compression="gzip", sep="\t", skiprows=table_skiprows(path), index_col=0)
    # The GEO matrix terminator is parsed as a final all-missing row by pandas.
    # Drop only rows with a non-finite selected value; retain every real probe.
    matrix = matrix.apply(pd.to_numeric, errors="coerce")
    n_rows_before = int(matrix.shape[0])
    matrix = matrix.loc[np.isfinite(matrix.to_numpy(dtype=float)).all(axis=1)].copy()
    write_json(out / "gse47751_input_qc.json", {
        "source": str(path.relative_to(ROOT)), "matrix_rows_before_nonfinite_filter": n_rows_before,
        "matrix_rows_after_nonfinite_filter": int(matrix.shape[0]),
        "rows_removed_for_nonfinite_values": n_rows_before - int(matrix.shape[0]),
    })
    missing = set(baseline.geo_accession) - set(matrix.columns)
    if missing:
        raise ValueError(f"GSE47751 selected samples missing from matrix: {sorted(missing)}")
    x = matrix.loc[:, baseline.geo_accession].to_numpy(dtype=float).T
    baseline["condition"] = baseline.state.map({"PN": "non_lesional", "PP": "lesional"})
    baseline.to_csv(out / "gse47751_baseline_metadata.csv", index=False)
    np.savez_compressed(out / "gse47751_baseline_expression.npz", expression=x, probe_ids=matrix.index.to_numpy())
    return x, baseline


def rebuild_equivalence(rebuilt: dict[str, tuple[np.ndarray, pd.DataFrame]], out: Path) -> pd.DataFrame:
    rows = []
    for stem, (counts, metadata) in rebuilt.items():
        legacy_counts = np.load(LEGACY / f"{stem}_sample_gene_counts.npz")["counts"].astype(float)
        legacy_meta = pd.read_csv(LEGACY / f"{stem}_sample_metadata.csv")
        # Compare the canonical CSV serialization, rather than the temporary
        # in-memory H5 category dtype.  This distinguishes a real metadata
        # change from harmless string-versus-integer representation.
        rebuilt_meta = pd.read_csv(out / f"{stem}_sample_metadata.csv")
        numeric_equal = counts.shape == legacy_counts.shape and np.array_equal(counts, legacy_counts)
        rows.append({
            "dataset": stem.upper(), "rebuild_samples": counts.shape[0], "rebuild_features": counts.shape[1],
            "legacy_samples": legacy_counts.shape[0], "legacy_features": legacy_counts.shape[1],
            "sample_metadata_exact_match": rebuilt_meta.equals(legacy_meta), "count_matrix_exact_match": bool(numeric_equal),
            "maximum_absolute_count_difference": float(np.max(np.abs(counts - legacy_counts))) if counts.shape == legacy_counts.shape else np.nan,
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "v21_rebuild_equivalence.csv", index=False)
    return frame


def candidate_audit(out: Path, selected_n: int) -> pd.DataFrame:
    rows = [
        {"accession": "GSE47751", "independent_of_PRI_Audit_development": True, "phenotype": "paired baseline PN/PP", "platform": "bulk microarray", "local_integrity": "fresh official GEO matrix passes gzip -t", "eligible_for": "external strict bulk replication", "selection": "selected", "n_selected_samples": selected_n},
        {"accession": "GSE261704", "independent_of_PRI_Audit_development": True, "phenotype": "psoriasis lesions versus atopic dermatitis / prurigo nodularis; no psoriasis NL pairing", "platform": "NanoString 770-gene panel", "local_integrity": "series matrix available", "eligible_for": "differential-diagnosis context only", "selection": "excluded", "n_selected_samples": 0},
        {"accession": "GSE223799", "independent_of_PRI_Audit_development": False, "phenotype": "lesional psoriasis versus eczema", "platform": "bulk RNA-seq", "local_integrity": "processed table available", "eligible_for": "technical diagnostic transport only", "selection": "excluded", "n_selected_samples": 0},
        {"accession": "GSE36287", "independent_of_PRI_Audit_development": True, "phenotype": "cytokine-stimulated keratinocytes", "platform": "bulk microarray", "local_integrity": "series matrix available", "eligible_for": "mechanistic context only; not clinical lesion/normal", "selection": "excluded", "n_selected_samples": 0},
        {"accession": "GSE6601", "independent_of_PRI_Audit_development": True, "phenotype": "psoriasis versus atopic-dermatitis lesional epidermis", "platform": "two-colour microarray", "local_integrity": "series matrix available", "eligible_for": "differential-diagnosis context only", "selection": "excluded", "n_selected_samples": 0},
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "v21_external_cohort_audit.csv", index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--rebuild-only", action="store_true", help="reconstruct raw inputs and write provenance without fitting models")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "reproducibility_manifest.json", inputs_manifest())

    rebuilt = {
        "gse151177": rebuild_gse151177(OUT)[:2],
        "gse173706": rebuild_gse173706(OUT)[:2],
        "gse206391": rebuild_gse206391(OUT)[:2],
    }
    equivalence = rebuild_equivalence(rebuilt, OUT)
    gse47751_x, gse47751_meta = rebuild_gse47751(OUT)
    audit = candidate_audit(OUT, len(gse47751_meta))
    if args.rebuild_only:
        write_json(OUT / "v21_run_summary.json", {
            "specification": "docs/pri_audit_v21_experiment_spec.md",
            "status": "raw_reconstruction_complete; evaluation_not_run",
            "rebuild_equivalence": equivalence.to_dict(orient="records"),
            "external_selection": audit.to_dict(orient="records"),
        })
        print(equivalence.to_string(index=False))
        return
    evaluations = []
    for stem, (counts, metadata) in rebuilt.items():
        x, y, groups, eligible_meta, paired = eligible_core(stem, counts, metadata)
        result, _ = evaluate_strict_loose(stem.upper(), x, y, groups, eligible_meta, paired, True, args.permutations, args.bootstrap, OUT)
        evaluations.extend(result["methods"])

    y = (gse47751_meta.condition == "lesional").astype(int).to_numpy()
    groups = gse47751_meta.patient.to_numpy()
    external_result, _ = evaluate_strict_loose("GSE47751", gse47751_x, y, groups, gse47751_meta, True, False, args.permutations, args.bootstrap, OUT)
    evaluations.extend(external_result["methods"])

    summary = pd.DataFrame(evaluations)
    summary.to_csv(OUT / "v21_loose_vs_strict_summary.csv", index=False)
    write_json(OUT / "v21_run_summary.json", {
        "specification": "docs/pri_audit_v21_experiment_spec.md",
        "rebuild_equivalence": equivalence.to_dict(orient="records"),
        "external_selection": audit.to_dict(orient="records"),
        "n_evaluation_rows": int(len(summary)),
        "classifier": "L2 logistic regression (C=1, class_weight=balanced, solver=liblinear, max_iter=2000)",
        "permutations": args.permutations, "bootstrap": args.bootstrap,
    })
    print(summary[["dataset", "arm", "auroc", "incremental_auroc_vs_strict_expression"]].to_string(index=False))


if __name__ == "__main__":
    main()

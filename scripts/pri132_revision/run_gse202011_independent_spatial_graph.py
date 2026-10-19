"""Independent spatial-graph check in GSE202011 using recovered Visium coordinates.

The 10x H5 matrices retain the exact Visium spot barcodes but not their spatial
folder.  We recover array coordinates by joining those barcodes to the public
Space Ranger v1.3.1 Visium V1 coordinate reference (4,992 spots).  This is a
coordinate reconstruction, not an image-registration analysis.  Every barcode
used below must join exactly once; the script fails otherwise.

The selected five biologic-naive PsO pairs and their deterministic ST14 R1
choice are the frozen GSE202011 boundary cohort used in v21.  The analysis
implements the same six-neighbour Moran-like agreement and local-contrast
features as the primary GSE206391 graph branch, with feature selection and
classification fit inside leave-one-patient-out folds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "pri_1_32_gse202011"
H5_DIR = RAW / "h5"
OUT = ROOT / "results" / "pri_1_32_revision"
SEED = 20261017
SAMPLES = [
    ("ST13", "GSM6086615", "LESIONAL"), ("ST13", "GSM6086616", "NON_LESIONAL"),
    ("ST14", "GSM6086617", "LESIONAL"), ("ST14", "GSM6086619", "NON_LESIONAL"),
    ("ST17", "GSM6086625", "LESIONAL"), ("ST17", "GSM6086626", "NON_LESIONAL"),
    ("ST20", "GSM6086630", "LESIONAL"), ("ST20", "GSM6086631", "NON_LESIONAL"),
    ("ST21", "GSM6086632", "LESIONAL"), ("ST21", "GSM6086633", "NON_LESIONAL"),
]


def decode(values: np.ndarray) -> np.ndarray:
    return np.asarray([x.decode() if isinstance(x, bytes) else str(x) for x in values], dtype=str)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_for(gsm: str) -> Path:
    matches = list(H5_DIR.glob(f"{gsm}_*_filtered_feature_bc_matrix.h5"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one H5 matrix for {gsm}, found {matches}")
    return matches[0]


def load_coordinate_map() -> pd.DataFrame:
    coords = pd.read_csv(ROOT / "scripts" / "pri132_revision" / "visium_v1_barcode_row_col.csv", dtype={"barcode": str})
    if list(coords.columns) != ["barcode", "col", "row"] or len(coords) != 4992 or coords.barcode.duplicated().any():
        raise ValueError("Unexpected Visium barcode-coordinate reference")
    return coords.set_index("barcode")


def load_sample(path: Path, coordinate_map: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    with h5py.File(path, "r") as handle:
        matrix_group = handle["matrix"]
        shape = tuple(int(x) for x in matrix_group["shape"][...])
        # H5 is genes-by-spots CSC.  Work in the conventional spots-by-genes orientation.
        matrix = csc_matrix((matrix_group["data"][...], matrix_group["indices"][...], matrix_group["indptr"][...]), shape=shape).T.tocsr()
        genes = decode(matrix_group["features"]["id"][...])
        symbols = decode(matrix_group["features"]["name"][...])
        barcodes = np.char.replace(decode(matrix_group["barcodes"][...]), "-1", "")
    total = np.asarray(matrix.sum(axis=1)).ravel()
    detected = np.asarray((matrix > 0).sum(axis=1)).ravel()
    mitochondrial = np.char.startswith(np.char.upper(symbols), "MT-")
    mitochondrial_fraction = np.asarray(matrix[:, mitochondrial].sum(axis=1)).ravel() / np.maximum(total, 1)
    keep = (total >= 500) & (detected >= 300) & (mitochondrial_fraction <= 0.20)
    selected_barcodes = barcodes[keep]
    if len(np.unique(selected_barcodes)) != len(selected_barcodes):
        raise ValueError(f"Repeated filtered barcodes in {path.name}")
    joined = coordinate_map.reindex(selected_barcodes)
    if joined.isna().any().any():
        raise ValueError(f"Coordinate join failed for {int(joined.isna().any(axis=1).sum())} barcodes in {path.name}")
    return matrix[keep].toarray().astype(np.float32), genes, joined[["row", "col"]].to_numpy(float), total[keep], int(matrix.shape[0]), int(keep.sum())


def rebuild() -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, pd.DataFrame]:
    coordinate_map = load_coordinate_map()
    spot_counts, spot_coordinates, sample_counts, rows, genes_ref = [], [], [], [], None
    for patient, gsm, condition in SAMPLES:
        path = file_for(gsm)
        counts, genes, coords, totals, raw_spots, qc_spots = load_sample(path, coordinate_map)
        if genes_ref is None:
            genes_ref = genes
        elif not np.array_equal(genes_ref, genes):
            raise ValueError(f"Gene order differs in {path.name}")
        spot_counts.append(counts)
        spot_coordinates.append(coords)
        sample_counts.append(counts.sum(axis=0))
        rows.append({"patient": patient, "gsm": gsm, "condition": condition, "file": path.name, "raw_filtered_matrix_spots": raw_spots, "qc_retained_spots": qc_spots, "coordinate_join_rate": 1.0, "input_sha256": sha256(path)})
    metadata = pd.DataFrame(rows)
    if not metadata.groupby("patient").condition.nunique().eq(2).all():
        raise RuntimeError("Selected independent cohort is not paired")
    return spot_counts, spot_coordinates, np.vstack(sample_counts).astype(float), metadata


def log_cpm(counts: np.ndarray) -> np.ndarray:
    return np.log1p(counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0) * 1e6)


def neighbors(coords: np.ndarray) -> np.ndarray:
    k = min(6, len(coords) - 1)
    return NearestNeighbors(n_neighbors=k + 1).fit(coords).kneighbors(coords, return_distance=False)[:, 1:]


def graph_features(counts: np.ndarray, totals: np.ndarray, near: np.ndarray, genes: np.ndarray) -> np.ndarray:
    values = np.log1p(counts[:, genes] / np.maximum(totals[:, None], 1.0) * 1e4)
    local = values[near].mean(axis=1)
    centered = values - values.mean(axis=0, keepdims=True)
    centered_local = centered[near].mean(axis=1)
    variance = np.mean(centered * centered, axis=0)
    agreement = np.divide(np.mean(centered * centered_local, axis=0), variance, out=np.zeros_like(variance), where=variance > 1e-10)
    contrast = np.mean(np.abs(values - local), axis=0)
    return np.concatenate([agreement, contrast])


def predict(train: np.ndarray, labels: np.ndarray, test: np.ndarray) -> np.ndarray:
    learner = make_pipeline(StandardScaler(), LogisticRegression(C=1, solver="liblinear", class_weight="balanced", max_iter=2000, random_state=0))
    learner.fit(train, labels)
    return learner.predict_proba(test)[:, 1]


def run_predictions(spot_counts: list[np.ndarray], spot_coordinates: list[np.ndarray], sample_counts: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict[str, np.ndarray]:
    expression = log_cpm(sample_counts)
    output = {"expression_500": np.full(len(y), np.nan), "expression_1500": np.full(len(y), np.nan), "spatial_graph": np.full(len(y), np.nan), "expression_plus_spatial": np.full(len(y), np.nan)}
    totals = [count.sum(axis=1) for count in spot_counts]
    nearest = [neighbors(coords) for coords in spot_coordinates]
    for held in np.unique(groups):
        test = groups == held
        train = ~test
        hvg_500 = np.argsort(expression[train].var(axis=0))[-500:]
        hvg_1500 = np.argsort(expression[train].var(axis=0))[-1500:]
        spatial = np.vstack([graph_features(count, total, near, hvg_500) for count, total, near in zip(spot_counts, totals, nearest)])
        for name, train_rep, test_rep in (
            ("expression_500", expression[train][:, hvg_500], expression[test][:, hvg_500]),
            ("expression_1500", expression[train][:, hvg_1500], expression[test][:, hvg_1500]),
            ("spatial_graph", spatial[train], spatial[test]),
            ("expression_plus_spatial", np.hstack([expression[train][:, hvg_500], spatial[train]]), np.hstack([expression[test][:, hvg_500], spatial[test]])),
        ):
            output[name][test] = predict(train_rep, y[train], test_rep)
    if any(np.isnan(values).any() for values in output.values()):
        raise RuntimeError("At least one held-out prediction is missing")
    return output


def grouped_bootstrap(y: np.ndarray, predictions: dict[str, np.ndarray], groups: np.ndarray, draws: int, rng: np.random.Generator) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    patients = np.unique(groups)
    index = {p: np.flatnonzero(groups == p) for p in patients}
    scores = {name: [] for name in predictions}
    references = {"spatial_graph": "expression_500", "expression_plus_spatial": "expression_1500"}
    deltas = {name: [] for name in references}
    for _ in range(draws):
        picked = rng.choice(patients, size=len(patients), replace=True)
        ix = np.concatenate([index[p] for p in picked])
        current = {name: roc_auc_score(y[ix], value[ix]) for name, value in predictions.items()}
        for name, value in current.items(): scores[name].append(value)
        for name, reference in references.items(): deltas[name].append(current[name] - current[reference])
    return {k: np.asarray(v) for k, v in scores.items()}, {k: np.asarray(v) for k, v in deltas.items()}


def coordinate_null(spot_counts: list[np.ndarray], spot_coordinates: list[np.ndarray], sample_counts: np.ndarray, y: np.ndarray, groups: np.ndarray, draws: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    values = {"spatial_graph": [], "expression_plus_spatial": []}
    for draw in range(draws):
        shuffled = [coords[rng.permutation(len(coords))] for coords in spot_coordinates]
        predictions = run_predictions(spot_counts, shuffled, sample_counts, y, groups)
        for name in values: values[name].append(roc_auc_score(y, predictions[name]))
        if (draw + 1) % 25 == 0: print(f"coordinate shuffle {draw + 1}/{draws}", flush=True)
    return {name: np.asarray(value) for name, value in values.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coordinate-shuffles", type=int, default=500)
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    spots, coords, aggregate_counts, metadata = rebuild()
    y = (metadata.condition == "LESIONAL").astype(int).to_numpy()
    groups = metadata.patient.to_numpy(str)
    observed = run_predictions(spots, coords, aggregate_counts, y, groups)
    bootstrap_scores, bootstrap_deltas = grouped_bootstrap(y, observed, groups, args.bootstrap, np.random.default_rng(SEED))
    null = coordinate_null(spots, coords, aggregate_counts, y, groups, args.coordinate_shuffles, np.random.default_rng(SEED + 1))
    references = {"spatial_graph": "expression_500", "expression_plus_spatial": "expression_1500"}
    report = {"dataset": "GSE202011", "role": "independent spatial-graph boundary replication", "n_patients": int(len(np.unique(groups))), "n_samples": int(len(y)), "coordinate_reconstruction": {"source": "Space Ranger v1.3.1 Visium V1 barcode-to-array-coordinate reference", "reference_rows": 4992, "joined_barcodes": int(sum(item.shape[0] for item in spots)), "join_rate": 1.0, "coordinate_type": "array-row and array-column; image registration was not used"}, "feature_construction": {"outer_split": "leave-one-patient-out", "spatial_graph_reference": "500 training-fold pseudobulk HVGs", "combined_reference": "1,500 training-fold pseudobulk HVGs", "spatial_features": "Moran-like six-neighbour agreement and local contrast per 500 HVGs", "learner": "class-balanced logistic regression (C=1, liblinear)"}, "methods": {}, "coordinate_shuffles": args.coordinate_shuffles, "bootstrap_draws": args.bootstrap, "seed": SEED}
    for name, values in observed.items():
        entry = {"auroc": float(roc_auc_score(y, values)), "group_bootstrap_ci95": [float(x) for x in np.quantile(bootstrap_scores[name], [.025, .975])]}
        if name in references:
            delta = bootstrap_deltas[name]
            null_values = null[name]
            entry.update({"matched_reference": references[name], "delta_vs_expression": float(roc_auc_score(y, values) - roc_auc_score(y, observed[references[name]])), "delta_bootstrap_ci95": [float(x) for x in np.quantile(delta, [.025, .975])], "coordinate_null_mean_auroc": float(null_values.mean()), "coordinate_null_ci95": [float(x) for x in np.quantile(null_values, [.025, .975])], "coordinate_shuffle_p_observed_ge_null": float((1 + np.sum(null_values >= entry["auroc"])) / (len(null_values) + 1))})
        report["methods"][name] = entry
    metadata.assign(true=y, **{f"{name}_probability": value for name, value in observed.items()}).to_csv(OUT / "gse202011_independent_spatial_graph_predictions.csv", index=False)
    np.savez_compressed(OUT / "gse202011_independent_spatial_graph_distributions.npz", **{f"bootstrap_{k}": v for k, v in bootstrap_scores.items()}, **{f"bootstrap_delta_{k}": v for k, v in bootstrap_deltas.items()}, **{f"coordinate_null_{k}": v for k, v in null.items()})
    (OUT / "gse202011_independent_spatial_graph.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

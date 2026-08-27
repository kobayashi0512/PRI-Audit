"""Develop and lock the pre-registered CC-MAC v3 model.

This script deliberately has no external-data loader.  It may only read the
three development disease cohorts; a distinct post-lock script will implement
the pre-registered external feasibility gate and blind evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

# The project environment has the analysis stack; the bundled runtime supplies
# openpyxl used exclusively to read the immutable public DerM22 reference.
try:
    import openpyxl  # noqa: F401
except ModuleNotFoundError:
    sys.path.append(
        "/Users/tongyue/.cache/codex-runtimes/codex-primary-runtime/"
        "dependencies/python/lib/python3.12/site-packages"
    )

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
REFERENCE = ROOT / "data" / "reference"
RESULTS = ROOT / "results" / "ccmac_v3"
RESULTS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    collapse_duplicate_genes,
    log_cpm,
    parse_geo_series_matrix,
    patient_id_from_title,
    sample_titles,
)
from run_cone_v0 import build_dictionaries, load_perturbation_deltas, rank_unit  # noqa: E402
from run_external_v1 import load_gse36287_effects, load_gse86451_deltas, map_gpl570_expression  # noqa: E402
from run_feasibility import map_count_index  # noqa: E402
from run_isrc_v2 import load_platform_map, reliability_weighted_residual  # noqa: E402

GAMMAS = [1, 2, 4, 8, 16, 32]
LAMBDAS = [0.0, 0.01, 0.1, 1.0]
AXIS_NAMES = [
    "IL17",
    "TNF_low",
    "TNF_high",
    "IL17_TNF_low",
    "IL17_TNF_high",
    "reliable_synergy_residual",
]
NUISANCE_NAMES = ["IL4", "IL13", "IFNa", "IFNg"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def zrank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    ranks = rankdata(values, method="average")
    return (ranks - ranks.mean()) / max(ranks.std(ddof=0), 1e-12)


def minmax_rank(values: np.ndarray) -> np.ndarray:
    ranks = rankdata(np.asarray(values, dtype=float), method="average")
    return (ranks - ranks.min()) / max(ranks.max() - ranks.min(), 1.0)


def unit_columns(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float).copy()
    norms = np.linalg.norm(matrix, axis=0)
    matrix[:, norms > 0] /= norms[norms > 0]
    return matrix


def collapse_derm22() -> pd.DataFrame:
    """Map GPL570 probes in DerM22 to symbols and average duplicate probes."""
    derm = pd.read_excel(
        REFERENCE / "DerM22_reference.xlsx", sheet_name="signatureMatrix_Derm22"
    )
    mapping = load_platform_map(RAW / "GPL570.annot.gz")
    derm["gene"] = derm["NAME"].astype(str).map(mapping)
    derm = derm.dropna(subset=["gene"]).drop(columns="NAME")
    return derm.groupby("gene", sort=False).mean()


def load_gse121212_lesions() -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = pd.read_csv(RAW / "GSE121212_readcount.txt.gz", sep="\t", index_col=0)
    counts.index = map_count_index(counts.index)
    counts = collapse_duplicate_genes(counts)
    keep = (counts.div(counts.sum(axis=0), axis=1) * 1e6 >= 1).sum(axis=1) >= 10
    expression = log_cpm(counts.loc[keep])
    _, metadata = parse_geo_series_matrix(RAW / "GSE121212_series_matrix.txt.gz")
    rows = []
    for title in sample_titles(metadata).values():
        if title not in expression.columns:
            continue
        if title.endswith("_lesional"):
            disease = title.split("_", maxsplit=1)[0]
        elif title.endswith("_chronic_lesion"):
            disease = title.split("_", maxsplit=1)[0]
        else:
            continue
        if disease not in {"PSO", "AD"}:
            continue
        rows.append({"sample": title, "label": int(disease == "PSO"), "patient": patient_id_from_title(title)})
    meta = pd.DataFrame(rows).drop_duplicates("sample")
    return expression.loc[:, meta["sample"]], meta.assign(cohort="GSE121212")


def load_gse182740_lesions() -> tuple[pd.DataFrame, pd.DataFrame]:
    expression, metadata = parse_geo_series_matrix(RAW / "GSE182740_series_matrix.txt.gz")
    expression = map_gpl570_expression(expression)
    rows = []
    for accession, title in sample_titles(metadata).items():
        if not title.startswith("JWK-09-") or not title.endswith("_LS"):
            continue
        core = title.removeprefix("JWK-09-").removesuffix("_LS")
        group, patient = core.split("-", maxsplit=1)
        if group not in {"Ps", "AD"}:
            continue
        rows.append({"sample": accession, "label": int(group == "Ps"), "patient": patient})
    meta = pd.DataFrame(rows).drop_duplicates("sample")
    return expression.loc[:, meta["sample"]], meta.assign(cohort="GSE182740")


def load_gse57225_lesions() -> tuple[pd.DataFrame, pd.DataFrame]:
    expression, metadata = parse_geo_series_matrix(RAW / "GSE57225_series_matrix.txt.gz")
    mapping = load_platform_map(RAW / "GPL14550_full.txt")
    expression.index = expression.index.astype(str).map(mapping)
    expression = collapse_duplicate_genes(expression)
    rows = []
    for accession, title in sample_titles(metadata).items():
        condition = pd.Series([title]).str.extract(r"_(pso|ecz|ACD)(?:\s|\(|$)", expand=False).iloc[0]
        if pd.isna(condition):
            continue
        patient = title.split("_", maxsplit=1)[0]
        rows.append({"sample": accession, "label": int(condition == "pso"), "patient": patient})
    meta = pd.DataFrame(rows).drop_duplicates("sample")
    return expression.loc[:, meta["sample"]], meta.assign(cohort="GSE57225")


def mechanism_dictionary() -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    effects, discovery = load_perturbation_deltas()
    replication = load_gse86451_deltas()
    broad = load_gse36287_effects()
    frozen = pd.read_csv(ROOT / "results" / "cone_v0" / "frozen_feature_genes.csv")["gene"].astype(str)
    genes = [
        gene
        for gene in frozen
        if all(gene in effect.index for effect in effects.values())
        and all(gene in frame.index for frame in replication.values())
        and all(gene in effect.index for effect in broad.values())
    ]
    baseline, _, _, _, _ = build_dictionaries(effects, genes)
    residual, _ = reliability_weighted_residual(discovery, replication, genes)
    residual = residual - baseline @ np.linalg.lstsq(baseline, residual, rcond=None)[0]
    mechanism = unit_columns(np.column_stack([baseline, residual]))
    nuisance = unit_columns(
        np.column_stack([rank_unit(broad[name].loc[genes].to_numpy()) for name in NUISANCE_NAMES])
    )
    return genes, mechanism, nuisance, residual


def score_samples(
    expression: pd.DataFrame,
    mechanism_genes: list[str],
    mechanism: np.ndarray,
    nuisance: np.ndarray,
    derm: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    missing = set(mechanism_genes).difference(expression.index)
    if missing:
        raise ValueError(f"Development expression missing {len(missing)} locked mechanism genes")
    derm_genes = derm.index.intersection(expression.index)
    if len(derm_genes) < 100:
        raise ValueError(f"Only {len(derm_genes)} DerM22 genes available")
    x = expression.loc[mechanism_genes].to_numpy(dtype=float)
    # Each multiplication yields one feature vector per sample; transpose to
    # preserve the invariant used throughout this script: rows are samples.
    mech_scores = np.vstack([zrank(x[:, j]) @ mechanism for j in range(x.shape[1])])
    nuisance_scores = np.vstack([zrank(x[:, j]) @ nuisance for j in range(x.shape[1])])
    derm_matrix = derm.loc[derm_genes].to_numpy(dtype=float)
    derm_matrix = np.column_stack([minmax_rank(derm_matrix[:, j]) for j in range(derm_matrix.shape[1])])
    compositions = []
    for sample in expression.columns:
        weights, _ = nnls(derm_matrix, minmax_rank(expression.loc[derm_genes, sample].to_numpy()))
        compositions.append(weights / max(weights.sum(), 1e-12))
    return mech_scores, nuisance_scores, np.asarray(compositions)


def anchor_projection(anchor: np.ndarray) -> np.ndarray:
    centered = anchor - anchor.mean(axis=0, keepdims=True)
    q, _ = np.linalg.qr(centered, mode="reduced")
    keep = np.linalg.norm(q, axis=0) > 1e-10
    q = q[:, keep]
    return q @ q.T


def fit_anchor_ridge(x: np.ndarray, y: np.ndarray, anchor: np.ndarray, gamma: float, lam: float) -> dict:
    mean = x.mean(axis=0)
    scale = np.where(x.std(axis=0, ddof=0) > 1e-12, x.std(axis=0, ddof=0), 1.0)
    z = (x - mean) / scale
    projection = anchor_projection(anchor)
    weight = np.eye(len(y)) + (gamma - 1.0) * projection
    theta = np.linalg.solve(z.T @ weight @ z + lam * np.eye(z.shape[1]), z.T @ weight @ y)
    return {"mean": mean, "scale": scale, "theta": theta}


def predict(model: dict, x: np.ndarray) -> np.ndarray:
    return ((x - model["mean"]) / model["scale"]) @ model["theta"]


def build_development_table() -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray, pd.DataFrame]:
    mechanism_genes, mechanism, nuisance, _ = mechanism_dictionary()
    derm = collapse_derm22()
    datasets = [load_gse121212_lesions(), load_gse182740_lesions(), load_gse57225_lesions()]
    shared = set(mechanism_genes)
    for expression, _ in datasets:
        shared &= set(expression.index.astype(str))
    genes = [gene for gene in mechanism_genes if gene in shared]
    if len(genes) < 500:
        raise RuntimeError(f"Only {len(genes)} shared mechanism genes; pre-registered model cannot be developed")
    # Reconstruct each fixed dictionary on the final common gene list.
    effects, discovery = load_perturbation_deltas()
    replication = load_gse86451_deltas()
    broad = load_gse36287_effects()
    baseline, _, _, _, _ = build_dictionaries(effects, genes)
    residual, _ = reliability_weighted_residual(discovery, replication, genes)
    residual = residual - baseline @ np.linalg.lstsq(baseline, residual, rcond=None)[0]
    mechanism = unit_columns(np.column_stack([baseline, residual]))
    nuisance = unit_columns(np.column_stack([rank_unit(broad[name].loc[genes].to_numpy()) for name in NUISANCE_NAMES]))
    rows, features, anchors = [], [], []
    for expression, meta in datasets:
        mech, nuis, comp = score_samples(expression, genes, mechanism, nuisance, derm)
        features.append(mech)
        anchors.append(np.column_stack([comp, nuis]))
        rows.append(meta.reset_index(drop=True))
    table = pd.concat(rows, ignore_index=True)
    return table, genes, np.vstack(features), np.vstack(anchors), derm


def cohort_anchor(table: pd.DataFrame, continuous: np.ndarray, categories: list[str]) -> np.ndarray:
    cohort = pd.Categorical(table["cohort"], categories=categories)
    onehot = pd.get_dummies(cohort, dtype=float).to_numpy()
    return np.column_stack([onehot, continuous])


def loco_grid(table: pd.DataFrame, x: np.ndarray, continuous: np.ndarray) -> tuple[pd.DataFrame, tuple[float, float]]:
    cohorts = sorted(table["cohort"].unique())
    y = table["label"].to_numpy(dtype=int)
    rows = []
    for gamma in GAMMAS:
        for lam in LAMBDAS:
            aucs = []
            for holdout in cohorts:
                train = table["cohort"].to_numpy() != holdout
                test = ~train
                model = fit_anchor_ridge(x[train], y[train], cohort_anchor(table.loc[train], continuous[train], cohorts), gamma, lam)
                auc = roc_auc_score(y[test], predict(model, x[test]))
                aucs.append(auc)
                rows.append({"gamma": gamma, "lambda": lam, "holdout": holdout, "auc": auc})
    grid = pd.DataFrame(rows)
    summary = grid.groupby(["gamma", "lambda"], as_index=False).agg(mean_auc=("auc", "mean"), sd_auc=("auc", "std"))
    selected = summary.sort_values(["mean_auc", "gamma", "lambda"], ascending=[False, True, False]).iloc[0]
    return grid.merge(summary, on=["gamma", "lambda"]), (float(selected.gamma), float(selected["lambda"]))


def develop() -> None:
    table, genes, x, continuous, derm = build_development_table()
    grid, (gamma, lam) = loco_grid(table, x, continuous)
    cohorts = sorted(table["cohort"].unique())
    y = table["label"].to_numpy(dtype=int)
    model = fit_anchor_ridge(x, y, cohort_anchor(table, continuous, cohorts), gamma, lam)
    predictions = table.copy()
    predictions[AXIS_NAMES] = x
    predictions["ccmac_score"] = predict(model, x)
    stability = []
    for holdout in cohorts:
        train = table["cohort"].to_numpy() != holdout
        fold = fit_anchor_ridge(x[train], y[train], cohort_anchor(table.loc[train], continuous[train], cohorts), gamma, lam)
        for name, value in zip(AXIS_NAMES, fold["theta"]):
            stability.append({"holdout": holdout, "axis": name, "coefficient": float(value), "sign": int(np.sign(value))})
    artifact = RESULTS / "locked_model.npz"
    np.savez_compressed(
        artifact,
        mechanism_genes=np.asarray(genes),
        mechanism_dictionary=mechanism_dictionary_for_lock(genes),
        derm_genes=np.asarray(derm.index.astype(str)),
        derm_matrix=derm.to_numpy(dtype=float),
        gamma=np.asarray([gamma]), lambda_=np.asarray([lam]),
        mean=model["mean"], scale=model["scale"], theta=model["theta"],
    )
    manifest = {
        "algorithm": "Cell-Composition Anchored Mechanistic Cone (CC-MAC) v3",
        "pre_registration_sha256": sha256(ROOT / "docs" / "v3_preregistered_plan.md"),
        "derm22_sha256": sha256(REFERENCE / "DerM22_reference.xlsx"),
        "script_sha256": sha256(Path(__file__)),
        "selected_gamma": gamma, "selected_lambda": lam,
        "n_development_samples": int(len(table)), "n_mechanism_genes": int(len(genes)),
        "axis_names": AXIS_NAMES, "nuisance_axes": NUISANCE_NAMES,
        "external_expression_read": False,
        "artifact_sha256": sha256(artifact),
    }
    grid.to_csv(RESULTS / "development_loco_grid.csv", index=False)
    predictions.to_csv(RESULTS / "development_predictions.csv", index=False)
    pd.DataFrame(stability).to_csv(RESULTS / "development_coefficient_stability.csv", index=False)
    (RESULTS / "locked_model_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def mechanism_dictionary_for_lock(genes: list[str]) -> np.ndarray:
    effects, discovery = load_perturbation_deltas()
    replication = load_gse86451_deltas()
    baseline, _, _, _, _ = build_dictionaries(effects, genes)
    residual, _ = reliability_weighted_residual(discovery, replication, genes)
    residual = residual - baseline @ np.linalg.lstsq(baseline, residual, rcond=None)[0]
    return unit_columns(np.column_stack([baseline, residual]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["develop"], default="develop")
    parser.parse_args()
    develop()

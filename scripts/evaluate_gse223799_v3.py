"""Locked blind evaluation of the GSE223799 macro/micro RNA-seq cohort."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results" / "ccmac_v3"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import collapse_duplicate_genes  # noqa: E402

SEED = 20260730
N_BOOT = 10000


def zrank(values: np.ndarray) -> np.ndarray:
    ranks = rankdata(np.asarray(values, dtype=float), method="average")
    return (ranks - ranks.mean()) / max(ranks.std(ddof=0), 1e-12)


def unit_columns(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float).copy()
    norms = np.linalg.norm(matrix, axis=0)
    matrix[:, norms > 0] /= norms[norms > 0]
    return matrix


def series_titles(path: Path) -> dict[str, str]:
    metadata: dict[str, list[str]] = {}
    with gzip.open(path, "rt", errors="replace") as handle:
        for line in handle:
            if not line.startswith("!Sample_"):
                continue
            key, *values = line.rstrip("\n").split("\t")
            metadata[key] = [v.strip('"') for v in values]
    return dict(zip(metadata["!Sample_geo_accession"], metadata["!Sample_title"]))


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    table = pd.read_csv(RAW / "GSE223799_Fischer_F_et_al.tsv.gz", sep="\t")
    expression = table.set_index("gene_name").drop(columns="gene_id")
    expression.index = expression.index.astype(str)
    expression = collapse_duplicate_genes(expression)
    # logCPM is monotone within each sample and stabilizes zero-heavy RNA-seq.
    expression = np.log2((expression + 0.5).div(expression.sum(axis=0) + 0.5, axis=1) * 1e6)
    titles = series_titles(RAW / "GSE223799_series_matrix.txt.gz")
    rows = []
    for title in titles.values():
        pieces = [piece.strip() for piece in title.split(",")]
        disease, scale, sample = pieces
        rows.append({"sample": sample, "label": int(disease == "Psoriasis"), "scale": scale.split()[0]})
    # Convert to a plain list so the expression-column Index name cannot
    # overwrite the metadata index name during .loc alignment.
    meta = pd.DataFrame(rows).set_index("sample").loc[list(expression.columns)].reset_index()
    meta = meta.rename(columns={meta.columns[0]: "sample"})
    return expression, meta


def v0_nnls_score(target: np.ndarray, dictionary: np.ndarray) -> float:
    weights, _ = nnls(dictionary, target)
    residual = target - dictionary @ weights
    return float(1 - np.sum(residual**2) / max(np.sum((target - target.mean()) ** 2), 1e-12))


def bootstrap(y: np.ndarray, scores: dict[str, np.ndarray]) -> dict[str, dict]:
    rng = np.random.default_rng(SEED)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    values = {name: np.empty(N_BOOT) for name in scores}
    for i in range(N_BOOT):
        index = np.concatenate([
            rng.choice(positive, len(positive), replace=True),
            rng.choice(negative, len(negative), replace=True),
        ])
        for name, score in scores.items():
            values[name][i] = roc_auc_score(y[index], score[index])
    result = {}
    for name, arr in values.items():
        result[name] = {"ci_low": float(np.quantile(arr, 0.025)), "ci_high": float(np.quantile(arr, 0.975))}
    for baseline in ("v0_nnls", "nos2_minus_ccl27"):
        delta = values["ccmac_locked_ridge"] - values[baseline]
        result[f"delta_ccmac_minus_{baseline}"] = {
            "mean": float(delta.mean()), "ci_low": float(np.quantile(delta, .025)),
            "ci_high": float(np.quantile(delta, .975)), "prob_gt_zero": float((delta > 0).mean()),
        }
    return result


def main() -> None:
    archive = np.load(RESULTS / "locked_model.npz", allow_pickle=True)
    genes = archive["mechanism_genes"].astype(str).tolist()
    dictionary = archive["mechanism_dictionary"]
    mean, scale, theta = archive["mean"], archive["scale"], archive["theta"]
    expression, meta = load_data()
    available = [gene for gene in genes if gene in expression.index]
    positions = [genes.index(gene) for gene in available]
    mechanism = unit_columns(dictionary[positions])
    baseline = unit_columns(mechanism[:, :5])
    rows = []
    for sample in meta["sample"]:
        target = zrank(expression.loc[available, sample].to_numpy())
        feature = target @ mechanism
        ccmac = float(((feature - mean) / scale) @ theta)
        full_rank = pd.Series(zrank(expression[sample].to_numpy()), index=expression.index)
        rows.append({
            "sample": sample,
            "ccmac_locked_ridge": ccmac,
            "v0_nnls": v0_nnls_score(target, baseline),
            "nos2_minus_ccl27": float(full_rank["NOS2"] - full_rank["CCL27"]),
            **{f"feature_{i+1}": float(value) for i, value in enumerate(feature)},
        })
    pred = meta.merge(pd.DataFrame(rows), on="sample", validate="one_to_one")
    metrics = {}
    for scale_name, group in pred.groupby("scale", sort=True):
        y = group["label"].to_numpy()
        scores = {name: group[name].to_numpy() for name in ("ccmac_locked_ridge", "v0_nnls", "nos2_minus_ccl27")}
        metrics[scale_name] = {
            "n_pso": int(y.sum()), "n_ad": int((y == 0).sum()),
            "point_auc": {name: float(roc_auc_score(y, score)) for name, score in scores.items()},
            "bootstrap": bootstrap(y, scores),
        }
    pred.to_csv(RESULTS / "GSE223799_locked_predictions.csv", index=False)
    (RESULTS / "GSE223799_locked_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

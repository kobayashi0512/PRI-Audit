"""Pre-registered v4 primary external evaluation on GSE153007."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RAW, RESULTS = ROOT / "data" / "raw", ROOT / "results" / "ccmac_v3"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import collapse_duplicate_genes, parse_geo_series_matrix, sample_titles  # noqa: E402
from run_isrc_v2 import load_platform_map  # noqa: E402

N_BOOT, SEED = 10000, 20260730


def zrank(values: np.ndarray) -> np.ndarray:
    ranks = rankdata(np.asarray(values, dtype=float), method="average")
    return (ranks - ranks.mean()) / max(ranks.std(ddof=0), 1e-12)


def unit_columns(matrix: np.ndarray) -> np.ndarray:
    result = np.asarray(matrix, dtype=float).copy()
    norms = np.linalg.norm(result, axis=0)
    result[:, norms > 0] /= norms[norms > 0]
    return result


def nnls_fit(target: np.ndarray, dictionary: np.ndarray) -> float:
    weights, _ = nnls(dictionary, target)
    return float(1 - np.sum((target - dictionary @ weights) ** 2) / max(np.sum((target - target.mean()) ** 2), 1e-12))


def bootstrap(y: np.ndarray, values: dict[str, np.ndarray]) -> dict:
    rng = np.random.default_rng(SEED)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    auc = {name: np.empty(N_BOOT) for name in values}
    for i in range(N_BOOT):
        index = np.r_[rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)]
        for name, score in values.items():
            auc[name][i] = roc_auc_score(y[index], score[index])
    report = {name: {"ci_low": float(np.quantile(a, .025)), "ci_high": float(np.quantile(a, .975))} for name, a in auc.items()}
    for baseline in ("v0_nnls", "nos2_minus_ccl27"):
        delta = auc["locked_mechanism_ridge"] - auc[baseline]
        report[f"delta_ridge_minus_{baseline}"] = {"mean": float(delta.mean()), "ci_low": float(np.quantile(delta,.025)), "ci_high": float(np.quantile(delta,.975)), "prob_gt_zero": float((delta > 0).mean())}
    return report


def main() -> None:
    archive = np.load(RESULTS / "locked_model.npz", allow_pickle=True)
    genes = archive["mechanism_genes"].astype(str).tolist()
    dictionary = archive["mechanism_dictionary"]
    mean, scale, theta = archive["mean"], archive["scale"], archive["theta"]
    expression, metadata = parse_geo_series_matrix(RAW / "GSE153007_series_matrix.txt.gz")
    mapping = load_platform_map(RAW / "GPL6480.annot.gz")
    expression.index = expression.index.astype(str).map(mapping)
    expression = collapse_duplicate_genes(expression)
    rows = []
    for accession, title in sample_titles(metadata).items():
        if "_Psoriasis_Psoriasis_LESION" in title:
            label = 1
        elif "_Atopic Dermatitis_Atopic Dermatitis_LESION" in title:
            label = 0
        else:
            continue
        # Frozen v4 rule: leading field identifies a patient/biological unit.
        patient = title.split("_", maxsplit=1)[0]
        rows.append({"accession": accession, "title": title, "patient": patient, "label": label})
    meta = pd.DataFrame(rows)
    # Mean technical arrays within the frozen patient/disease unit before scoring.
    groups = meta.groupby(["patient", "label"], sort=True)["accession"].agg(list).reset_index()
    aggregate = pd.DataFrame({f"{r.patient}|{r.label}": expression[r.accession].mean(axis=1) for r in groups.itertuples()})
    meta = groups.assign(sample=[f"{r.patient}|{r.label}" for r in groups.itertuples()])
    available = [gene for gene in genes if gene in aggregate.index]
    coverage = len(available) / len(genes)
    diagnostic_present = {"NOS2", "CCL27"}.issubset(aggregate.index)
    if coverage < .70 or not diagnostic_present:
        raise RuntimeError(
            f"Pre-registered external gate failed: coverage={coverage:.3f}, "
            f"NOS2/CCL27 present={diagnostic_present}"
        )
    positions = [genes.index(gene) for gene in available]
    mechanism = unit_columns(dictionary[positions])
    baseline = unit_columns(mechanism[:, :5])
    result = []
    for sample in meta["sample"]:
        target = zrank(aggregate.loc[available, sample].to_numpy())
        features = target @ mechanism
        all_rank = pd.Series(zrank(aggregate[sample].to_numpy()), index=aggregate.index)
        result.append({"sample": sample, "locked_mechanism_ridge": float(((features - mean) / scale) @ theta), "v0_nnls": nnls_fit(target, baseline), "nos2_minus_ccl27": float(all_rank.NOS2 - all_rank.CCL27), **{f"feature_{i+1}": float(v) for i,v in enumerate(features)}})
    pred = meta.merge(pd.DataFrame(result), on="sample", validate="one_to_one")
    y = pred.label.to_numpy()
    scores = {name: pred[name].to_numpy() for name in ("locked_mechanism_ridge", "v0_nnls", "nos2_minus_ccl27")}
    report = {"n_pso": int(y.sum()), "n_ad": int((y==0).sum()), "n_patient_units": int(len(y)), "mechanism_coverage": coverage, "point_auc": {name: float(roc_auc_score(y, score)) for name, score in scores.items()}, "bootstrap": bootstrap(y, scores)}
    pred.to_csv(RESULTS / "GSE153007_v4_predictions.csv", index=False)
    (RESULTS / "GSE153007_v4_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

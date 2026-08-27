"""Locked patient-level v5 treatment-direction replication in GSE85034."""

from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
RAW, RESULTS = ROOT / "data" / "raw", ROOT / "results" / "ccmac_v3"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import collapse_duplicate_genes, parse_geo_series_matrix, sample_titles  # noqa: E402
from run_isrc_v2 import load_platform_map  # noqa: E402

SEED, N_BOOT = 20260730, 10000


def zrank(x: np.ndarray) -> np.ndarray:
    r = rankdata(np.asarray(x, dtype=float), method="average")
    return (r - r.mean()) / max(r.std(ddof=0), 1e-12)


def unit_columns(x: np.ndarray) -> np.ndarray:
    out = np.asarray(x, dtype=float).copy()
    norm = np.linalg.norm(out, axis=0)
    out[:, norm > 0] /= norm[norm > 0]
    return out


def characteristic_rows(path: Path) -> list[list[str]]:
    out = []
    with gzip.open(path, "rt", errors="replace") as handle:
        for line in handle:
            if line.startswith("!Sample_characteristics_ch1"):
                out.append([item.strip('"') for item in line.rstrip("\n").split("\t")[1:]])
    return out


def paired_bootstrap(delta: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    values = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(N_BOOT)])
    return {"mean": float(delta.mean()), "ci_low": float(np.quantile(values,.025)), "ci_high": float(np.quantile(values,.975)), "prob_lt_zero": float((values < 0).mean())}


def main() -> None:
    archive = np.load(RESULTS / "locked_model.npz", allow_pickle=True)
    genes = archive["mechanism_genes"].astype(str).tolist()
    locked_dictionary = archive["mechanism_dictionary"]
    mean, scale, theta = archive["mean"], archive["scale"], archive["theta"]
    path = RAW / "GSE85034_series_matrix.txt.gz"
    expression, metadata = parse_geo_series_matrix(path)
    expression.index = expression.index.astype(str).map(load_platform_map(RAW / "GPL10558.annot.gz"))
    expression = collapse_duplicate_genes(expression)
    accessions = list(sample_titles(metadata))
    titles = list(sample_titles(metadata).values())
    chars = characteristic_rows(path)
    treatment = next(row for row in chars if row and row[0].startswith("treatment:"))
    pasi = next(row for row in chars if row and row[0].startswith("pasi:"))
    rows = []
    for accession, title, drug, pasi_value in zip(accessions, titles, treatment, pasi):
        match = re.fullmatch(r"Subject (\d+) (NL|LS|WK ?\d+)", title)
        if not match:
            continue
        subject, time = match.groups()
        time = time.replace("WK ", "WK")
        rows.append({"accession": accession, "subject": int(subject), "time": time, "treatment": drug.split(": ",1)[1], "pasi": pd.to_numeric(pasi_value.split(": ",1)[1], errors="coerce")})
    meta = pd.DataFrame(rows)
    available = [gene for gene in genes if gene in expression.index]
    coverage = len(available) / len(genes)
    positions = [genes.index(gene) for gene in available]
    dictionary = unit_columns(locked_dictionary[positions])
    if coverage < .70 or (dictionary[:, :5] != 0).sum(axis=0).min() < 30:
        raise RuntimeError(f"Pre-registered gate failed: coverage={coverage:.3f}")
    feature_rows = [
        zrank(expression.loc[available, row.accession].to_numpy()) @ dictionary
        for row in meta.itertuples()
    ]
    features = np.asarray(feature_rows)
    meta[[f"feature_{i + 1}" for i in range(features.shape[1])]] = features
    meta["locked_mechanism_ridge"] = [
        float(((feature - mean) / scale) @ theta) for feature in features
    ]
    scores = meta.pivot(index=["subject", "treatment"], columns="time", values="locked_mechanism_ridge")
    results = {"analysis_unit": "patient with explicit subject/time metadata", "mechanism_coverage": coverage}
    primary_rows = []
    for drug in ("ADA", "MTX"):
        subset = scores.xs(drug, level="treatment") if drug in scores.index.get_level_values("treatment") else pd.DataFrame()
        complete = subset.dropna(subset=["LS", "WK16"])
        delta = (complete["WK16"] - complete["LS"]).to_numpy()
        primary_rows.append({"treatment": drug, "n_complete": int(len(delta)), **paired_bootstrap(delta), "wilcoxon_two_sided_p": float(wilcoxon(delta, alternative="two-sided").pvalue) if len(delta) else np.nan, "n_decrease": int((delta < 0).sum())})
    pooled = scores.dropna(subset=["LS", "WK16"])
    pooled_delta = (pooled["WK16"] - pooled["LS"]).to_numpy()
    results["week16_minus_lesional_baseline"] = {"by_treatment": primary_rows, "pooled": {"n_complete": int(len(pooled_delta)), **paired_bootstrap(pooled_delta), "wilcoxon_two_sided_p": float(wilcoxon(pooled_delta, alternative="two-sided").pvalue), "n_decrease": int((pooled_delta < 0).sum())}}
    trajectory = []
    for time in ("WK1", "WK2", "WK4", "WK16"):
        complete = scores.dropna(subset=["LS", time])
        delta = (complete[time] - complete["LS"]).to_numpy()
        trajectory.append({"time": time, "n_complete": int(len(delta)), **paired_bootstrap(delta), "n_decrease": int((delta < 0).sum())})
    results["pooled_trajectory_vs_lesional_baseline"] = trajectory
    # Pre-specified score is the only inferential endpoint.  This is an
    # explicitly post-hoc decomposition used to identify whether one frozen
    # axis alone accounts for the observed longitudinal direction.
    feature_columns = [f"feature_{i + 1}" for i in range(features.shape[1])]
    decomposition = []
    for feature in feature_columns:
        pivot = meta.pivot(index=["subject", "treatment"], columns="time", values=feature)
        complete = pivot.dropna(subset=["LS", "WK16"])
        delta = (complete["WK16"] - complete["LS"]).to_numpy()
        decomposition.append({"feature": feature, "n_complete": int(len(delta)), **paired_bootstrap(delta), "n_decrease": int((delta < 0).sum())})
    results["posthoc_frozen_axis_decomposition_week16"] = decomposition
    meta.to_csv(RESULTS / "GSE85034_v5_patient_scores.csv", index=False)
    pd.DataFrame(trajectory).to_csv(RESULTS / "GSE85034_v5_trajectory.csv", index=False)
    (RESULTS / "GSE85034_v5_metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

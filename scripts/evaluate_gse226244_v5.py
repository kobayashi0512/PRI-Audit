"""Locked v5 group-level treatment-direction evaluation for GSE226244."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, rankdata

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


def all_characteristics(path: Path) -> list[list[str]]:
    rows = []
    with gzip.open(path, "rt", errors="replace") as handle:
        for line in handle:
            if line.startswith("!Sample_characteristics_ch1"):
                rows.append([v.strip('"') for v in line.rstrip("\n").split("\t")[1:]])
    return rows


def main() -> None:
    archive = np.load(RESULTS / "locked_model.npz", allow_pickle=True)
    genes = archive["mechanism_genes"].astype(str).tolist()
    dictionary = archive["mechanism_dictionary"]
    mean, scale, theta = archive["mean"], archive["scale"], archive["theta"]
    path = RAW / "GSE226244_series_matrix.txt.gz"
    expression, metadata = parse_geo_series_matrix(path)
    mapping = load_platform_map(RAW / "GPL570.annot.gz")
    expression.index = expression.index.astype(str).map(mapping)
    expression = collapse_duplicate_genes(expression)
    accessions = list(sample_titles(metadata))
    characteristics = all_characteristics(path)
    timeline = next(row for row in characteristics if row and row[0].startswith("treatment_timeline:"))
    disease = next(row for row in characteristics if row and row[0].startswith("disease state:"))
    lesion = next(row for row in characteristics if row and row[0].startswith("lesional (ls)"))
    sample_meta = pd.DataFrame({
        "accession": accessions,
        "timeline": [x.split(": ", 1)[1] for x in timeline],
        "disease": [x.split(": ", 1)[1] for x in disease],
        "lesion": [x.split(": ", 1)[1] for x in lesion],
    })
    sample_meta = sample_meta.query("disease == 'Psoriasis' and lesion == 'LS'").copy()
    sample_meta = sample_meta.loc[sample_meta.timeline.isin(["Baseline", "on12"])].copy()
    available = [gene for gene in genes if gene in expression.index]
    coverage = len(available) / len(genes)
    axis_counts = (unit_columns(dictionary[[genes.index(g) for g in available], :5]) != 0).sum(axis=0)
    if coverage < .70 or (axis_counts < 30).any():
        raise RuntimeError(f"Gate failed: coverage={coverage:.3f}, axis_counts={axis_counts.tolist()}")
    dictionary = unit_columns(dictionary[[genes.index(g) for g in available]])
    scores = []
    for accession in sample_meta.accession:
        feature = zrank(expression.loc[available, accession].to_numpy()) @ dictionary
        scores.append(float(((feature - mean) / scale) @ theta))
    sample_meta["locked_mechanism_ridge"] = scores
    baseline = sample_meta.loc[sample_meta.timeline == "Baseline", "locked_mechanism_ridge"].to_numpy()
    post = sample_meta.loc[sample_meta.timeline == "on12", "locked_mechanism_ridge"].to_numpy()
    rng = np.random.default_rng(SEED)
    delta = np.empty(N_BOOT)
    for i in range(N_BOOT):
        delta[i] = rng.choice(post, len(post), replace=True).mean() - rng.choice(baseline, len(baseline), replace=True).mean()
    report = {
        "analysis_unit": "sample (patient pairing unavailable in GEO metadata)",
        "n_baseline": int(len(baseline)), "n_on12": int(len(post)),
        "mechanism_coverage": coverage,
        "baseline_mean": float(baseline.mean()), "on12_mean": float(post.mean()),
        "on12_minus_baseline": float(post.mean() - baseline.mean()),
        "bootstrap_ci": [float(np.quantile(delta,.025)), float(np.quantile(delta,.975))],
        "bootstrap_prob_lt_zero": float((delta < 0).mean()),
        "mann_whitney_two_sided_p": float(mannwhitneyu(post, baseline, alternative="two-sided").pvalue),
        "interpretation": "group-level direction only; not paired and not a clinical-response model",
    }
    sample_meta.to_csv(RESULTS / "GSE226244_v5_group_scores.csv", index=False)
    (RESULTS / "GSE226244_v5_group_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

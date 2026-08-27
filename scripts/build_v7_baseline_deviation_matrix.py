"""Build v7 inputs: within-patient baseline LS-minus-NL gene-rank deviations."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "results" / "prismr_v7"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import collapse_duplicate_genes, parse_geo_series_matrix  # noqa: E402
from run_isrc_v2 import load_platform_map  # noqa: E402


SOURCES = {
    "GSE117239": (RAW / "GSE117239_series_matrix.txt.gz", RAW / "GPL570.annot.gz"),
    "GSE136757": (RAW / "GSE136757_series_matrix.txt.gz", RAW / "GPL570.annot.gz"),
    "GSE85034": (RAW / "GSE85034_series_matrix.txt.gz", RAW / "GPL10558.annot.gz"),
}


def load_expression(cohort: str) -> pd.DataFrame:
    path, annotation = SOURCES[cohort]
    expression, _ = parse_geo_series_matrix(path)
    expression.index = expression.index.astype(str).map(load_platform_map(annotation))
    expression = collapse_duplicate_genes(expression)
    expression = expression.loc[expression.index.notna()].copy()
    expression = expression.loc[~expression.index.duplicated()].copy()
    return expression


def within_sample_rank(values: np.ndarray) -> np.ndarray:
    return rankdata(values.astype(float), method="average") / len(values)


def main() -> None:
    registry = pd.read_csv(OUT / "v7_response_patient_registry.csv")
    expressions = {cohort: load_expression(cohort) for cohort in SOURCES}
    common = sorted(set.intersection(*(set(frame.index) for frame in expressions.values())))
    if len(common) < 5000:
        raise RuntimeError(f"Cross-platform overlap gate failed: {len(common)} common genes")
    rows = []
    kept = []
    for row in registry.itertuples(index=False):
        expr = expressions[row.cohort].loc[common]
        if row.ls_accession not in expr or row.nl_accession not in expr:
            continue
        ls = within_sample_rank(expr[row.ls_accession].to_numpy())
        nl = within_sample_rank(expr[row.nl_accession].to_numpy())
        rows.append((ls - nl).astype(np.float32))
        kept.append(row)
    meta = pd.DataFrame(kept)
    matrix = pd.DataFrame(np.vstack(rows), columns=common)
    if len(meta) != len(registry):
        missing = registry.loc[~registry.ls_accession.isin(meta.ls_accession)]
        raise RuntimeError(f"Missing baseline expression for {len(missing)} eligible patients")
    matrix.to_parquet(OUT / "v7_baseline_ls_minus_nl_rank.parquet", index=False)
    meta.to_csv(OUT / "v7_baseline_ls_minus_nl_metadata.csv", index=False)
    pd.DataFrame({"gene": common}).to_csv(OUT / "v7_common_genes.csv", index=False)
    summary = meta.groupby(["cohort", "drug", "label_type"], as_index=False).size().rename(columns={"size": "n_patients"})
    summary.to_csv(OUT / "v7_input_matrix_summary.csv", index=False)
    print(f"Built {matrix.shape[0]} x {matrix.shape[1]} baseline LS-NL rank-deviation matrix")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

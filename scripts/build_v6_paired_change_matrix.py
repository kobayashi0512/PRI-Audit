"""Create a gene-harmonized paired longitudinal change matrix for v6.

No outcome labels are used.  Each expression vector is converted to within-sample
gene ranks before paired differences, avoiding cross-platform absolute-scale
comparisons.  Feature selection is deliberately deferred to nested LODO training.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "results" / "motifr_v6"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import collapse_duplicate_genes, parse_geo_series_matrix  # noqa: E402
from run_isrc_v2 import load_platform_map  # noqa: E402


CONFIG = {
    "GSE117239": {"matrix": "GSE117239_series_matrix.txt.gz", "platform": "GPL570.annot.gz", "meta": "GSE117239_metadata_audit.csv", "time": "time", "drug": "drug", "tissue": "tissue", "baseline": "Baseline", "early": "Week1", "late": "Week12"},
    "GSE51440": {"matrix": "GSE51440_series_matrix.txt.gz", "platform": "GPL13158.annot.gz", "meta": "GSE51440_metadata_audit.csv", "time": "visit", "drug": "drug", "tissue": "skin", "baseline": "Baseline", "early": "week 1", "late": "week 12"},
    "GSE136757": {"matrix": "GSE136757_series_matrix.txt.gz", "platform": "GPL570.annot.gz", "meta": "GSE136757_metadata_audit.csv", "time": "time", "drug": "treatment", "tissue": "tissue", "baseline": "Day0", "early": "Day14", "late": "Day28"},
}


def read_expression(name: str, cfg: dict) -> pd.DataFrame:
    expression, _ = parse_geo_series_matrix(RAW / cfg["matrix"])
    expression.index = expression.index.astype(str).map(load_platform_map(RAW / cfg["platform"]))
    expression = collapse_duplicate_genes(expression)
    audit = pd.read_csv(OUT / cfg["meta"])
    valid = audit.loc[audit[cfg["tissue"]].eq("LS"), "accession"].tolist()
    expression = expression.loc[:, [x for x in valid if x in expression.columns]]
    # Percentile ranks make each platform/sample comparable while retaining the
    # within-subject directional change that MOTIF-R learns from.
    return expression.rank(axis=0, pct=True).astype("float32")


def paired_rows(name: str, cfg: dict, expression: pd.DataFrame, common: list[str]) -> tuple[list[dict], list[np.ndarray]]:
    audit = pd.read_csv(OUT / cfg["meta"])
    audit = audit.loc[(audit[cfg["tissue"]] == "LS") & audit.accession.isin(expression.columns)].copy()
    rows, vectors = [], []
    for drug, group in audit.groupby(cfg["drug"]):
        pivot = group.pivot_table(index="subject", columns=cfg["time"], values="accession", aggfunc="first")
        for label, start, end in [("early", cfg["baseline"], cfg["early"]), ("late", cfg["baseline"], cfg["late"])]:
            if start not in pivot or end not in pivot:
                continue
            for subject, pair in pivot.dropna(subset=[start, end]).iterrows():
                change = (expression.loc[common, pair[end]] - expression.loc[common, pair[start]]).to_numpy(dtype="float32")
                rows.append({"cohort": name, "drug": drug, "subject": str(subject), "phase": label,
                             "start": start, "end": end, "start_accession": pair[start], "end_accession": pair[end]})
                vectors.append(change)
    return rows, vectors


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tables = {name: read_expression(name, cfg) for name, cfg in CONFIG.items()}
    common = sorted(set.intersection(*(set(table.index) for table in tables.values())))
    if len(common) < 5000:
        raise RuntimeError(f"Only {len(common)} common genes; platform harmonization gate failed")
    all_rows, all_vectors = [], []
    for name, cfg in CONFIG.items():
        rows, vectors = paired_rows(name, cfg, tables[name], common)
        all_rows.extend(rows)
        all_vectors.extend(vectors)
    metadata = pd.DataFrame(all_rows)
    matrix = pd.DataFrame(np.vstack(all_vectors), columns=common)
    metadata.to_csv(OUT / "v6_paired_change_metadata.csv", index=False)
    matrix.to_parquet(OUT / "v6_paired_change_rank_matrix.parquet", index=False)
    pd.DataFrame({"gene": common}).to_csv(OUT / "v6_common_genes.csv", index=False)
    summary = metadata.groupby(["cohort", "drug", "phase"]).size().rename("n_pairs").reset_index()
    summary.to_csv(OUT / "v6_paired_change_summary.csv", index=False)
    print(f"common_genes={len(common)}; paired_changes={len(metadata)}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

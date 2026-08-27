"""Expand public LS/NL trigger validation with additional audited GEO cohorts."""

from __future__ import annotations

import gzip
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "results" / "trigger_v9"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import collapse_duplicate_genes, parse_geo_series_matrix, sample_titles  # noqa: E402
from run_isrc_v2 import load_platform_map  # noqa: E402

SOURCES = {
    "GSE153007": (RAW / "GSE153007_series_matrix.txt.gz", RAW / "GPL6480.annot.gz"),
    "GSE182740": (RAW / "GSE182740_series_matrix.txt.gz", RAW / "GPL570.annot.gz"),
    "GSE57225": (RAW / "GSE57225_series_matrix.txt.gz", RAW / "GPL14550_full.txt"),
    "GSE117239": (RAW / "GSE117239_series_matrix.txt.gz", RAW / "GPL570.annot.gz"),
    "GSE136757": (RAW / "GSE136757_series_matrix.txt.gz", RAW / "GPL570.annot.gz"),
    "GSE51440": (RAW / "GSE51440_series_matrix.txt.gz", RAW / "GPL13158.annot.gz"),
    "GSE85034": (RAW / "GSE85034_series_matrix.txt.gz", RAW / "GPL10558.annot.gz"),
}


def rows_from_audit(cohort, path, filters, state_key, subject_key):
    frame = pd.read_csv(path)
    for key, value in filters.items():
        frame = frame.loc[frame[key].eq(value)]
    ls = frame.loc[frame[state_key].eq("LS")].rename(columns={"accession": "ls_accession", subject_key: "subject"})
    nl = frame.loc[frame[state_key].eq("NL")].rename(columns={"accession": "nl_accession", subject_key: "subject"})
    out = ls[["subject", "ls_accession"]].merge(nl[["subject", "nl_accession"]], on="subject", how="inner")
    return out.assign(cohort=cohort)[["cohort", "subject", "ls_accession", "nl_accession"]]


def geo_chars(path):
    rows = {}
    with gzip.open(path, "rt", errors="replace") as handle:
        for line in handle:
            if line.startswith("!Sample_characteristics_ch1"):
                values = [x.strip('"') for x in line.rstrip("\n").split("\t")[1:]]
                rows[values[0].split(":", 1)[0].lower()] = [x.split(":", 1)[-1].strip() for x in values]
    return rows


def gse85034_pairs():
    path = RAW / "GSE85034_series_matrix.txt.gz"
    _, metadata = parse_geo_series_matrix(path)
    titles = sample_titles(metadata); chars = geo_chars(path)
    trt, pasi = chars["treatment"], chars["pasi"]
    frame = []
    for acc, title, treatment, _ in zip(titles, titles.values(), trt, pasi):
        match = re.fullmatch(r"Subject (\d+) (NL|LS|WK ?\d+)", title)
        if match:
            subject, state = match.groups(); frame.append({"subject": subject, "state": state.replace("WK ", "WK"), "accession": acc})
    frame = pd.DataFrame(frame)
    ls = frame.loc[frame.state.eq("LS")].rename(columns={"accession": "ls_accession"})
    nl = frame.loc[frame.state.eq("NL")].rename(columns={"accession": "nl_accession"})
    out = ls[["subject", "ls_accession"]].merge(nl[["subject", "nl_accession"]], on="subject", how="inner")
    return out.assign(cohort="GSE85034")[["cohort", "subject", "ls_accession", "nl_accession"]]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = [pd.read_csv(ROOT / "results" / "trigger_v8" / "v8_paired_psoriasis_metadata.csv")[["cohort", "subject", "accession"]].rename(columns={"accession": "ls_accession"})]
    # Rebuild the original three cohorts from the audited metadata so the
    # expanded matrix has one consistent row definition.
    pairs = [
        rows_from_audit("GSE117239", ROOT / "results" / "motifr_v6" / "GSE117239_metadata_audit.csv", {"time": "Baseline"}, "tissue", "subject"),
        rows_from_audit("GSE136757", ROOT / "results" / "motifr_v6" / "GSE136757_metadata_audit.csv", {"time": "Day0"}, "tissue", "subject"),
        rows_from_audit("GSE51440", ROOT / "results" / "motifr_v6" / "GSE51440_metadata_audit.csv", {"visit": "Baseline"}, "skin", "subject"),
        gse85034_pairs(),
    ]
    old = pd.read_csv(ROOT / "results" / "trigger_v8" / "v8_paired_psoriasis_metadata.csv")
    # Existing metadata stores one row per state; reconstruct explicit pairs.
    old_pairs = []
    for (cohort, subject), sub in old.groupby(["cohort", "subject"]):
        if set(sub.state) >= {"LS", "NL"}:
            old_pairs.append({"cohort": cohort, "subject": subject, "ls_accession": sub.loc[sub.state.eq("LS"), "accession"].iloc[0], "nl_accession": sub.loc[sub.state.eq("NL"), "accession"].iloc[0]})
    pairs.append(pd.DataFrame(old_pairs))
    meta = pd.concat(pairs, ignore_index=True).drop_duplicates(["cohort", "subject"])
    dictionary_genes = pd.read_csv(ROOT / "results" / "trigger_v8" / "v8_common_genes.csv").gene.tolist()
    expressions = {}
    for cohort, (path, platform) in SOURCES.items():
        x, _ = parse_geo_series_matrix(path); x.index = x.index.astype(str).map(load_platform_map(platform)); x = collapse_duplicate_genes(x); expressions[cohort] = x.loc[x.index.notna() & ~x.index.duplicated()]
    genes = sorted(set(dictionary_genes).intersection(*(set(x.index) for x in expressions.values())))
    if len(genes) < 5000: raise RuntimeError(f"expanded overlap gate failed: {len(genes)} genes")
    matrix, kept = [], []
    for row in meta.itertuples(index=False):
        x = expressions[row.cohort].loc[genes]
        if row.ls_accession not in x or row.nl_accession not in x: continue
        matrix.append((rankdata(x[row.ls_accession].to_numpy()) - rankdata(x[row.nl_accession].to_numpy())).astype(np.float32)); kept.append(row)
    matrix = pd.DataFrame(np.vstack(matrix), columns=genes); kept = pd.DataFrame(kept)
    matrix.to_parquet(OUT / "v9_expanded_ls_minus_nl_rank.parquet", index=False); kept.to_csv(OUT / "v9_expanded_metadata.csv", index=False); pd.DataFrame({"gene": genes}).to_csv(OUT / "v9_common_genes.csv", index=False)
    print(f"matrix={matrix.shape}"); print(kept.groupby("cohort").size().to_string())


if __name__ == "__main__": main()

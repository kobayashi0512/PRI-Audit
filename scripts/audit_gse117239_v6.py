"""Audit explicit longitudinal subject, drug, time and response metadata in GSE117239."""

from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "results" / "motifr_v6"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import parse_geo_series_matrix, sample_titles  # noqa: E402


def characteristic_rows(path: Path) -> list[list[str]]:
    rows = []
    with gzip.open(path, "rt", errors="replace") as handle:
        for line in handle:
            if line.startswith("!Sample_characteristics_ch1"):
                rows.append([x.strip('"') for x in line.rstrip("\n").split("\t")[1:]])
    return rows


def find_field(rows: list[list[str]], key: str) -> list[str]:
    for row in rows:
        if row and row[0].lower().startswith(key.lower() + ":"):
            return [item.split(":", 1)[1].strip() for item in row]
    raise KeyError(key)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = RAW / "GSE117239_series_matrix.txt.gz"
    _, metadata = parse_geo_series_matrix(path)
    titles = sample_titles(metadata)
    chars = characteristic_rows(path)
    subject = find_field(chars, "subjectid")
    response = find_field(chars, "pasi75 response at week 12")
    rows = []
    for accession, title, sid, response_label in zip(titles.keys(), titles.values(), subject, response):
        # e.g. 2015-00002.Psoriasis.LS.Week12.Ustekinumab90mg
        fields = title.split(".")
        if len(fields) != 5:
            continue
        _, disease, tissue, time, drug = fields
        rows.append({"accession": accession, "title": title, "subject": sid, "disease": disease,
                     "tissue": tissue, "time": time, "drug": drug, "pasi75_w12": response_label})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "GSE117239_metadata_audit.csv", index=False)
    ls = frame.loc[frame.tissue.eq("LS")].copy()
    counts = ls.groupby(["drug", "time"], dropna=False).subject.nunique().unstack(fill_value=0).sort_index()
    counts.to_csv(OUT / "GSE117239_ls_subject_counts.csv")
    pair_rows = []
    for drug, sub in ls.groupby("drug"):
        pivot = sub.pivot_table(index="subject", columns="time", values="accession", aggfunc="first")
        for start, end in [("Baseline", "Week1"), ("Baseline", "Week12"), ("Week1", "Week12")]:
            pair_rows.append({"drug": drug, "contrast": f"{end}-{start}", "n_complete": int(pivot.dropna(subset=[start, end]).shape[0])})
    pd.DataFrame(pair_rows).to_csv(OUT / "GSE117239_pairing_audit.csv", index=False)
    response_counts = (ls.loc[ls.time.eq("Baseline")].groupby(["drug", "pasi75_w12"]).subject.nunique()
                       .rename("n_subjects").reset_index())
    response_counts.to_csv(OUT / "GSE117239_baseline_response_counts.csv", index=False)
    report = {"n_samples_total": int(len(frame)), "n_lesional_samples": int(len(ls)),
              "drugs": sorted(ls.drug.unique().tolist()), "timepoints": sorted(ls.time.unique().tolist()),
              "has_explicit_subject": bool(frame.subject.notna().all()), "has_pasi75_label": bool((ls.pasi75_w12 != "NA").any())}
    (OUT / "GSE117239_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

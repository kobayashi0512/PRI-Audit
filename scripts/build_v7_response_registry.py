"""Create an auditable patient-level eligibility registry for v7 PRISM-R."""

from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "results" / "prismr_v7"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import parse_geo_series_matrix, sample_titles  # noqa: E402


def gse117239() -> pd.DataFrame:
    frame = pd.read_csv(ROOT / "results" / "motifr_v6" / "GSE117239_metadata_audit.csv")
    ls = frame.query("tissue == 'LS' and time == 'Baseline'").copy()
    nl = frame.query("tissue == 'NL' and time == 'Baseline'")[["subject", "drug", "accession"]].rename(columns={"accession": "nl_accession"})
    out = ls.merge(nl, on=["subject", "drug"], how="inner")
    out = out.rename(columns={"accession": "ls_accession", "pasi75_w12": "response_label"})
    out["cohort"] = "GSE117239"
    out["label_type"] = "pasi75_binary"
    out["response_value"] = out.response_label.map({"NR": 0.0, "R": 1.0})
    return out[["cohort", "drug", "subject", "ls_accession", "nl_accession", "label_type", "response_value", "response_label"]]


def gse136757() -> pd.DataFrame:
    frame = pd.read_csv(ROOT / "results" / "motifr_v6" / "GSE136757_metadata_audit.csv")
    ls0 = frame.query("tissue == 'LS' and time == 'Day0'")[["subject", "treatment", "accession", "pasi_total"]].rename(columns={"treatment": "drug", "accession": "ls_accession", "pasi_total": "pasi0"})
    nl0 = frame.query("tissue == 'NL' and time == 'Day0'")[["subject", "treatment", "accession"]].rename(columns={"treatment": "drug", "accession": "nl_accession"})
    late = frame.query("tissue == 'LS' and time == 'Day28'")[["subject", "treatment", "pasi_total"]].rename(columns={"treatment": "drug", "pasi_total": "pasi28"})
    out = ls0.merge(nl0, on=["subject", "drug"], how="inner").merge(late, on=["subject", "drug"], how="inner")
    out = out.loc[out.pasi0.gt(0)].copy()
    out["cohort"], out["label_type"] = "GSE136757", "fractional_pasi_reduction"
    out["response_value"] = 1.0 - out.pasi28 / out.pasi0
    out["response_label"] = out.response_value.round(6).astype(str)
    return out[["cohort", "drug", "subject", "ls_accession", "nl_accession", "label_type", "response_value", "response_label"]]


def characteristic_rows(path: Path) -> list[list[str]]:
    rows = []
    with gzip.open(path, "rt", errors="replace") as handle:
        for line in handle:
            if line.startswith("!Sample_characteristics_ch1"):
                rows.append([x.strip('"') for x in line.rstrip("\n").split("\t")[1:]])
    return rows


def gse85034() -> pd.DataFrame:
    path = RAW / "GSE85034_series_matrix.txt.gz"
    _, metadata = parse_geo_series_matrix(path)
    titles = sample_titles(metadata)
    chars = characteristic_rows(path)
    treatment = next(row for row in chars if row and row[0].startswith("treatment:"))
    pasi = next(row for row in chars if row and row[0].startswith("pasi:"))
    rows = []
    for accession, title, trt, score in zip(titles, titles.values(), treatment, pasi):
        match = re.fullmatch(r"Subject (\d+) (NL|LS|WK ?\d+)", title)
        if match:
            subject, time = match.groups()
            rows.append({"subject": subject, "time": time.replace("WK ", "WK"), "drug": trt.split(": ", 1)[1],
                         "accession": accession, "pasi": pd.to_numeric(score.split(": ", 1)[1], errors="coerce")})
    frame = pd.DataFrame(rows)
    ls = frame.query("time == 'LS'")[["subject", "drug", "accession", "pasi"]].rename(columns={"accession": "ls_accession", "pasi": "pasi0"})
    nl = frame.query("time == 'NL'")[["subject", "drug", "accession"]].rename(columns={"accession": "nl_accession"})
    wk16 = frame.query("time == 'WK16'")[["subject", "drug", "pasi"]].rename(columns={"pasi": "pasi16"})
    out = ls.merge(nl, on=["subject", "drug"], how="inner").merge(wk16, on=["subject", "drug"], how="inner")
    out = out.loc[out.pasi0.gt(0)].copy()
    out["cohort"], out["label_type"] = "GSE85034", "fractional_pasi_reduction"
    out["response_value"] = 1.0 - out.pasi16 / out.pasi0
    out["response_label"] = out.response_value.round(6).astype(str)
    return out[["cohort", "drug", "subject", "ls_accession", "nl_accession", "label_type", "response_value", "response_label"]]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    registry = pd.concat([gse117239(), gse85034(), gse136757()], ignore_index=True)
    registry["context"] = registry.cohort + "::" + registry.drug
    registry = registry.sort_values(["cohort", "drug", "subject"]).reset_index(drop=True)
    registry.to_csv(OUT / "v7_response_patient_registry.csv", index=False)
    summary = registry.groupby(["cohort", "drug", "label_type"], as_index=False).agg(
        n_patients=("subject", "nunique"),
        n_rows=("subject", "size"),
        response_min=("response_value", "min"), response_max=("response_value", "max"),
    )
    summary.to_csv(OUT / "v7_response_eligibility_summary.csv", index=False)
    report = {"n_patients": int(len(registry)), "n_active_contexts": int((registry.drug != "Placebo").groupby(registry.context).first().sum()),
              "contexts": registry.groupby("context").size().to_dict(), "label_types": registry.label_type.value_counts().to_dict()}
    (OUT / "v7_response_registry_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

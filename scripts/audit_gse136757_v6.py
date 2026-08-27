"""Audit GSE136757 longitudinal brepocitinib/placebo metadata."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "results" / "motifr_v6"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import parse_geo_series_matrix, sample_titles  # noqa: E402


def fields(path: Path) -> dict[str, list[str]]:
    result = {}
    with gzip.open(path, "rt", errors="replace") as handle:
        for line in handle:
            if line.startswith("!Sample_characteristics_ch1"):
                values = [item.strip('"') for item in line.rstrip("\n").split("\t")[1:]]
                key = values[0].split(":", 1)[0].strip().lower()
                result[key] = [v.split(":", 1)[1].strip() for v in values]
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = RAW / "GSE136757_series_matrix.txt.gz"
    _, metadata = parse_geo_series_matrix(path)
    titles, meta = sample_titles(metadata), fields(path)
    frame = pd.DataFrame({"accession": list(titles), "title": list(titles.values()), "subject": meta["subjid"],
                          "tissue": meta["tissue"], "time": meta["time"], "treatment": meta["treatment"],
                          "pasi_total": pd.to_numeric(meta["pasi_total"], errors="coerce")})
    frame["tissue"] = frame.tissue.map({"lesional skin": "LS", "non-lesional skin": "NL"})
    frame.to_csv(OUT / "GSE136757_metadata_audit.csv", index=False)
    ls = frame.query("tissue == 'LS'")
    counts = ls.groupby(["treatment", "time"]).subject.nunique().rename("n_subjects").reset_index()
    counts.to_csv(OUT / "GSE136757_ls_subject_counts.csv", index=False)
    pairs = []
    for drug, sub in ls.groupby("treatment"):
        pivot = sub.pivot_table(index="subject", columns="time", values="accession", aggfunc="first")
        for late in ("Day14", "Day28"):
            if late in pivot:
                pairs.append({"treatment": drug, "contrast": f"{late}-Day0", "n_complete": int(pivot.dropna(subset=["Day0", late]).shape[0])})
    pd.DataFrame(pairs).to_csv(OUT / "GSE136757_pairing_audit.csv", index=False)
    report = {"n_samples": int(len(frame)), "n_lesional": int(len(ls)), "n_nonlesional": int((frame.tissue == "NL").sum()),
              "treatments": sorted(frame.treatment.unique().tolist()), "has_explicit_subject": True,
              "clinical_response_metadata": "sample-level PASI total is deposited and can support within-subject Day0-to-Day28 change"}
    (OUT / "GSE136757_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

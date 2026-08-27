"""Audit the longitudinal guselkumab cohort GSE51440 from deposited metadata."""

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
    output = {}
    with gzip.open(path, "rt", errors="replace") as handle:
        for line in handle:
            if not line.startswith("!Sample_characteristics_ch1"):
                continue
            values = [item.strip('"') for item in line.rstrip("\n").split("\t")[1:]]
            key = values[0].split(":", 1)[0].strip().lower()
            output[key] = [value.split(":", 1)[1].strip() for value in values]
    return output


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = RAW / "GSE51440_series_matrix.txt.gz"
    _, metadata = parse_geo_series_matrix(path)
    titles = sample_titles(metadata)
    meta = fields(path)
    frame = pd.DataFrame({"accession": list(titles), "title": list(titles.values()),
                          "subject": meta["subject"], "visit": meta["visit"], "skin": meta["skin"]})
    frame["drug"] = "Guselkumab"
    frame.to_csv(OUT / "GSE51440_metadata_audit.csv", index=False)
    ls = frame.loc[frame.skin.eq("LS")]
    counts = ls.groupby("visit").subject.nunique().rename("n_subjects").reset_index()
    counts.to_csv(OUT / "GSE51440_ls_subject_counts.csv", index=False)
    pivot = ls.pivot_table(index="subject", columns="visit", values="accession", aggfunc="first")
    pairs = []
    for first, second in [("Baseline", "week 1"), ("Baseline", "week 12"), ("week 1", "week 12")]:
        pairs.append({"contrast": f"{second}-{first}", "n_complete": int(pivot.dropna(subset=[first, second]).shape[0])})
    pd.DataFrame(pairs).to_csv(OUT / "GSE51440_pairing_audit.csv", index=False)
    report = {"drug": "Guselkumab", "n_samples": int(len(frame)), "n_lesional": int(len(ls)),
              "n_nonlesional": int((frame.skin == "NL").sum()), "has_explicit_subject": True,
              "visits": counts.to_dict(orient="records"), "clinical_response_metadata": "not deposited in series matrix"}
    (OUT / "GSE51440_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

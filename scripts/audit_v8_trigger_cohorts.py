"""Metadata-only audit for public, paired psoriasis trigger-recovery cohorts."""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "results" / "trigger_v8"


def geo_metadata(path: Path):
    accessions, titles, characteristics = [], [], {}
    with gzip.open(path, "rt", errors="replace") as handle:
        for line in handle:
            values = [v.strip('"') for v in line.rstrip("\n").split("\t")[1:]]
            if line.startswith("!Sample_geo_accession"):
                accessions = values
            elif line.startswith("!Sample_title"):
                titles = values
            elif line.startswith("!Sample_characteristics_ch1"):
                key = values[0].split(":", 1)[0].strip().lower()
                characteristics[key] = [v.split(":", 1)[-1].strip() for v in values]
    return accessions, titles, characteristics


def audit_153007():
    acc, title, c = geo_metadata(RAW / "GSE153007_series_matrix.txt.gz")
    d = pd.DataFrame({"accession": acc, "title": title, "subject": c["patient_id"], "diagnosis": c["diagnosis"], "lesion": c["lesion"]})
    d = d.query("diagnosis == 'Psoriasis'").copy()
    d["state"] = d.lesion.map({"TRUE": "LS", "FALSE": "NL"})
    return d[["accession", "subject", "state"]].assign(cohort="GSE153007")


def audit_182740():
    acc, title, c = geo_metadata(RAW / "GSE182740_series_matrix.txt.gz")
    d = pd.DataFrame({"accession": acc, "title": title, "disease": c["disease"], "state": c["lesional (ls) vs. nonlesional (nl) vs. normal"]})
    d = d.query("disease == 'Psoriasis'").copy()
    d["subject"] = d.title.str.extract(r"Ps-(\d+)_")[0]
    d["state"] = d.state.str.upper()
    return d[["accession", "subject", "state"]].dropna().assign(cohort="GSE182740")


def audit_57225():
    acc, title, c = geo_metadata(RAW / "GSE57225_series_matrix.txt.gz")
    d = pd.DataFrame({"accession": acc, "title": title, "subject": c["individual"], "disease": c["disease state"]})
    # The same individual can contribute pso and non-involved skin; eczema/ACD
    # samples remain out of scope for the primary psoriasis trigger contrast.
    d["state"] = d.disease.map({"psoriasis": "LS", "control (non-involved)": "NL"})
    return d.dropna(subset=["state"])[["accession", "subject", "state"]].assign(cohort="GSE57225")


def audit_24767():
    acc, title, c = geo_metadata(RAW / "GSE24767_series_matrix.txt.gz")
    d = pd.DataFrame({"accession": acc, "title": title, "donor": c["id"], "treatment": c["treatment"]})
    return d.assign(cohort="GSE24767")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paired = pd.concat([audit_153007(), audit_182740(), audit_57225()], ignore_index=True)
    paired.to_csv(OUT / "v8_paired_psoriasis_metadata.csv", index=False)
    pairs = []
    for cohort, sub in paired.groupby("cohort"):
        pivot = sub.pivot_table(index="subject", columns="state", values="accession", aggfunc="first")
        pairs.append({"cohort": cohort, "n_psoriasis_samples": len(sub), "n_subjects": sub.subject.nunique(),
                      "n_explicit_ls_nl_pairs": int(pivot.dropna(subset=["LS", "NL"]).shape[0])})
    perturb = audit_24767()
    perturb.to_csv(OUT / "v8_keratinocyte_perturbation_metadata.csv", index=False)
    summary = pd.DataFrame(pairs)
    summary.to_csv(OUT / "v8_trigger_cohort_eligibility.csv", index=False)
    report = {"paired_cohorts": summary.to_dict(orient="records"),
              "perturbation_conditions": perturb.treatment.value_counts().to_dict(),
              "excluded": {"GSE47751": "gzip trailing-garbage integrity failure; not used"}}
    (OUT / "v8_trigger_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(summary.to_string(index=False))
    print("\nPerturbations:\n" + perturb.treatment.value_counts().to_string())


if __name__ == "__main__":
    main()

"""Register the frozen GSE202011 external result in the v21 candidate ledger."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "pri_audit_v21"


def main() -> None:
    report_path = OUT / "gse202011_v21_external_strict.json"
    if not report_path.exists():
        raise FileNotFoundError("Run run_v21_gse202011_external.py before registering this candidate")
    report = json.loads(report_path.read_text())
    path = OUT / "v21_external_cohort_audit.csv"
    audit = pd.read_csv(path)
    audit = audit[audit.accession != "GSE202011"]
    row = pd.DataFrame([{
        "accession": "GSE202011", "independent_of_PRI_Audit_development": True,
        "phenotype": "paired biologic-naive PSO lesional/non-lesional skin",
        "platform": "10x Visium H5 matrices; coordinates absent from the retrieved archive",
        "local_integrity": f"official GSE202011_RAW.tar SHA-256 {report['raw_archive']['sha256']}; 10 selected H5 files hashed",
        "eligible_for": "strict patient-blocked expression/PCA high-resolution boundary replication; not spatial graph",
        "selection": "selected before expression inspection", "n_selected_samples": 10,
    }])
    audit = pd.concat([audit, row], ignore_index=True)
    audit.to_csv(path, index=False)
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()

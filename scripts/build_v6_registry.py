"""Assemble the v6 treatment-cohort registry from completed audit outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "motifr_v6"


def load(name: str) -> dict:
    return json.loads((OUT / name).read_text())


def main() -> None:
    g117 = load("GSE117239_audit.json")
    g514 = load("GSE51440_audit.json")
    g136 = load("GSE136757_audit.json")
    rows = [
        {"accession": "GSE117239", "contexts": "; ".join(g117["drugs"]), "n_samples": g117["n_samples_total"],
         "n_lesional": g117["n_lesional_samples"], "times": "; ".join(g117["timepoints"]), "pairing": "explicit", "clinical_label": "PASI75 week12", "status": "eligible core"},
        {"accession": "GSE51440", "contexts": g514["drug"], "n_samples": g514["n_samples"],
         "n_lesional": g514["n_lesional"], "times": "; ".join(x["visit"] for x in g514["visits"]), "pairing": "explicit", "clinical_label": "not in matrix", "status": "eligible small external"},
        {"accession": "GSE136757", "contexts": "; ".join(g136["treatments"]), "n_samples": g136["n_samples"],
         "n_lesional": g136["n_lesional"], "times": "Day0; Day14; Day28", "pairing": "explicit", "clinical_label": "not primary", "status": "eligible incl. placebo"},
        {"accession": "GSE137218", "contexts": "secukinumab", "n_samples": "pending", "n_lesional": "pending", "times": "Day0; 4; 14; 42; 84", "pairing": "literature-reported; raw file re-acquisition", "clinical_label": "pending audit", "status": "blocked technical retrieval"},
    ]
    pd.DataFrame(rows).to_csv(OUT / "v6_treatment_registry.csv", index=False)


if __name__ == "__main__":
    main()

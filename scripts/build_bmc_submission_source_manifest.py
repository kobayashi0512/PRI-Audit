"""Hash-bind the BMC-formatted PRI-Audit submission package to its sources."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "paper_v21"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> None:
    source_files = [
        ROOT / "manuscript" / "PRI_Audit_psoriasis_manuscript_v8_submission_ready.md",
        ROOT / "scripts" / "build_v21_submission_manuscript_docx.py",
        ROOT / "scripts" / "build_bmc_submission_assets.py",
        OUT / "v8_submission_source_manifest.json",
        OUT / "v22_submission_closure_source_manifest.json",
        OUT / "v9_submission_source_manifest.json",
    ]
    output_files = [
        ROOT / "manuscript" / "PRI_Audit_BMC_Bioinformatics_submission_ready.docx",
        ROOT / "figures" / "paper_v19_fig1_dataset_atlas.png",
        ROOT / "figures" / "paper_v20_fig2_overall_framework.png",
        ROOT / "figures" / "paper_v20_fig3_algorithm_architecture.png",
        ROOT / "figures" / "paper_v21" / "fig11_v21_strictness_and_reference_audit.png",
        ROOT / "figures" / "paper_v21" / "fig12_v21_independence_and_spatial_null.png",
        ROOT / "figures" / "paper_v22" / "fig6_v22_known_truth_simulation.png",
        OUT / "table1_cohort_roles.csv",
        OUT / "table2_strict_matched_comparisons.csv",
        OUT / "table3_audit_diagnostics.csv",
        OUT / "tableS1_v22_known_truth_simulation.csv",
        OUT / "tableS2_v22_spot_topic_sensitivity.csv",
        ROOT / "submission_package" / "BMC_Bioinformatics_submission_assets" / "README.md",
    ]
    payload = {
        "analysis": "PRI-Audit BMC Bioinformatics submission assembly",
        "journal_profile": {
            "target": "BMC Bioinformatics",
            "article_type": "Research Article",
            "format_features": ["double line spacing", "page numbering", "continuous line numbering", "editable Word tables"],
        },
        "source_files": [record(path) for path in source_files],
        "output_files": [record(path) for path in output_files],
        "release_note": (
            "The v1.0.0 GitHub tag is the frozen executable analysis package. "
            "This BMC assembly adds journal-formatted manuscript and upload assets "
            "without changing the underlying v21/v22 numerical outputs."
        ),
        "author_owned_before_submission": [
            "author list, affiliations and corresponding-author details",
            "funding, acknowledgements, CRediT roles and competing-interest statement",
            "institutionally confirmed ethics wording",
            "approved software licence and archival DOI for the frozen v1.0.0 release",
        ],
    }
    rows = []
    for item in payload["source_files"]:
        rows.append({"role": "assembly source", **item})
    for item in payload["output_files"]:
        rows.append({"role": "submission output", **item})
    (OUT / "bmc_submission_source_manifest.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    pd.DataFrame(rows).to_csv(OUT / "bmc_submission_source_manifest.csv", index=False)
    print(f"Wrote BMC submission source manifest with {len(rows)} hash-bound files")


if __name__ == "__main__":
    main()

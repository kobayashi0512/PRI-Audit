"""Bind the complete v9 manuscript assembly to audited source artifacts.

The v8 manifest covers Figures 1–5 and Tables 1–3. The v22 closure manifest
covers Figure 6 and the new simulation/topic table rows. This small release
manifest connects both records to the final v9 Word document without claiming
that a private style reference is scientific input.
"""

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
    source = ROOT / "manuscript" / "PRI_Audit_psoriasis_manuscript_v8_submission_ready.md"
    docx = ROOT / "manuscript" / "PRI_Audit_psoriasis_manuscript_v9_submission_ready.docx"
    builder = ROOT / "scripts" / "build_v21_submission_manuscript_docx.py"
    base_manifest = OUT / "v8_submission_source_manifest.json"
    closure_manifest = OUT / "v22_submission_closure_source_manifest.json"
    outputs = [
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
        docx,
    ]
    sources = [source, builder, base_manifest, closure_manifest]
    payload = {
        "analysis": "PRI-Audit v9 final submission-assembly source manifest",
        "source_files": [record(path) for path in sources],
        "output_files": [record(path) for path in outputs],
        "assembly_note": (
            "The v8 manifest hash-binds Figures 1–5 and Tables 1–3; the v22 "
            "closure manifest binds Figure 6, new Table 2 rows and both "
            "supplementary tables. The v9 DOCX is generated from the listed "
            "Markdown source and builder. A private visual reference is not a "
            "scientific source and is optional in the public builder."
        ),
    }
    rows = []
    for item in payload["source_files"]:
        rows.append({
            "role": "assembly source",
            "path": item["path"],
            "bytes": item["bytes"],
            "sha256": item["sha256"],
        })
    for item in payload["output_files"]:
        rows.append({
            "role": "submission output",
            "path": item["path"],
            "bytes": item["bytes"],
            "sha256": item["sha256"],
        })
    (OUT / "v9_submission_source_manifest.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    pd.DataFrame(rows).to_csv(OUT / "v9_submission_source_manifest.csv", index=False)
    print(f"Wrote v9 source manifest with {len(rows)} hash-bound files")


if __name__ == "__main__":
    main()

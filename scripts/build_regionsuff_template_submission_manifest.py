"""Hash-bind the user-template PRI-Audit submission draft to its evidence.

This companion manifest records the content source, assembly program, result
manifests, figures, tables and final DOCX.  The template reference itself is
user-owned and intentionally omitted from the public release; only its
basename and SHA-256 digest are recorded so the local rendering choice remains
auditable without redistributing third-party manuscript material.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "paper_v21"
TEMPLATE = Path("/Users/tongyue/Documents/ppf/第四论文/RegionSuff_3.2.docx")


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
    builder = ROOT / "scripts" / "build_v21_submission_manuscript_docx.py"
    manuscript = ROOT / "manuscript" / "PRI_Audit_psoriasis_RegionSuff_template_submission.docx"
    base_manifest = OUT / "v8_submission_source_manifest.json"
    closure_manifest = OUT / "v22_submission_closure_source_manifest.json"
    v9_manifest = OUT / "v9_submission_source_manifest.json"
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
        manuscript,
    ]
    sources = [source, builder, base_manifest, closure_manifest, v9_manifest]
    payload = {
        "analysis": "PRI-Audit template-profile submission assembly",
        "template_reference": {
            "basename": TEMPLATE.name,
            "sha256": sha256(TEMPLATE),
            "included_in_public_release": False,
            "reason": "User-owned visual reference; not scientific input or manuscript content.",
        },
        "source_files": [record(path) for path in sources],
        "output_files": [record(path) for path in outputs],
        "assembly_note": (
            "The v8 and v22 source manifests bind all result figures and tables. "
            "The v9 manifest records the profile-independent Word assembly. This "
            "manifest binds the same evidence to the template-profile author draft."
        ),
    }
    rows = []
    for item in payload["source_files"]:
        rows.append({"role": "assembly source", **item})
    for item in payload["output_files"]:
        rows.append({"role": "submission output", **item})
    json_path = OUT / "regionsuff_template_submission_source_manifest.json"
    csv_path = OUT / "regionsuff_template_submission_source_manifest.csv"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Wrote template-submission source manifest with {len(rows)} hash-bound files")


if __name__ == "__main__":
    main()

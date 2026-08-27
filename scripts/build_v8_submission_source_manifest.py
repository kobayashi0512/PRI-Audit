"""Create a hash-bound source manifest for the rendered v8 submission draft.

This is intentionally separate from the v21 experimental manifest.  The latter
proves the result-generation chain; this file additionally records how the
validated quantitative outputs and non-quantitative data/framework visuals are
assembled into the Word manuscript.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "paper_v21"
V21 = ROOT / "results" / "pri_audit_v21"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def record(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest(path),
        "bytes": path.stat().st_size,
    }


def item(identifier: str, role: str, source_files: list[Path], output_files: list[Path], note: str) -> dict[str, object]:
    return {
        "identifier": identifier,
        "role": role,
        "source_files": [record(path) for path in source_files],
        "output_files": [record(path) for path in output_files],
        "note": note,
    }


def main() -> None:
    builder = ROOT / "scripts" / "build_v21_submission_manuscript_docx.py"
    deck = ROOT / "ppt_figures_v1" / "PRI_Audit_data_integrated_figures_v2_audited.pptx"
    entries = [
        item(
            "Figure 1",
            "descriptive data-derived atlas; no v21 performance estimate",
            [deck, V21 / "v21_all11_input_registry.csv"],
            [ROOT / "figures" / "paper_v19_fig1_dataset_atlas.png"],
            "The visual is retained to show the 11 public-data inputs at their native scales. Its table-bound cohort labels and roles are cross-audited against the v21 registry.",
        ),
        item(
            "Figure 2",
            "PRI-Audit framework diagram",
            [deck, ROOT / "docs" / "pri_audit_v21_experiment_spec.md"],
            [ROOT / "figures" / "paper_v20_fig2_overall_framework.png"],
            "Conceptual workflow only; implementation details and claim boundary are frozen in the v21 specification.",
        ),
        item(
            "Figure 3",
            "PRI-Audit module architecture",
            [deck, builder],
            [ROOT / "figures" / "paper_v20_fig3_algorithm_architecture.png"],
            "Conceptual architecture; its module descriptions are audited against the v8 Methods source.",
        ),
        item(
            "Figure 4",
            "strictness, matched-reference, and neighbourhood sensitivity audit",
            [
                ROOT / "scripts" / "plot_v21_strictness_figure.py",
                V21 / "v21_loose_vs_strict_summary.csv",
                V21 / "v21_mismatched_reference_sensitivity.csv",
                V21 / "v21_nested_neighborhood_strictness.csv",
            ],
            [ROOT / "figures" / "paper_v21" / "fig11_v21_strictness_and_reference_audit.png"],
            "Quantitative v21 primary/diagnostic display.",
        ),
        item(
            "Figure 5",
            "random measurement-split and coordinate-null diagnostic",
            [
                ROOT / "scripts" / "plot_v21_independence_and_spatial_null.py",
                V21 / "v21_measurement_random_split_diagnostic.csv",
                V21 / "gse206391_v21_matched_spatial_coordinate_shuffle.json",
                V21 / "gse206391_v21_matched_spatial_coordinate_shuffle_distributions.npz",
            ],
            [ROOT / "figures" / "paper_v21" / "fig12_v21_independence_and_spatial_null.png"],
            "Quantitative v21 counterfactual display.",
        ),
        item(
            "Table 1",
            "cohort roles and independent units",
            [ROOT / "scripts" / "build_v21_manuscript_tables.py", V21 / "v21_all11_input_registry.csv", V21 / "v21_all11_provenance_manifest.json"],
            [OUT / "table1_cohort_roles.csv"],
            "Generated from the frozen all-11 input registry.",
        ),
        item(
            "Table 2",
            "strict matched representation comparisons",
            [ROOT / "scripts" / "build_v21_manuscript_tables.py", V21 / "v21_loose_vs_strict_summary.csv", V21 / "gse206391_v21_matched_spatial_graph.json", V21 / "gse202011_v21_external_strict.json"],
            [OUT / "table2_strict_matched_comparisons.csv"],
            "Generated from strict held-out estimates; saturated cohorts are labelled as such.",
        ),
        item(
            "Table 3",
            "reproducibility and counterfactual diagnostics",
            [ROOT / "scripts" / "build_v21_manuscript_tables.py", V21 / "v21_rebuild_equivalence.csv", V21 / "v21_measurement_random_split_diagnostic.csv", V21 / "v21_nested_neighborhood_strictness.csv", V21 / "v21_mismatched_reference_sensitivity.csv", V21 / "gse206391_v21_matched_spatial_coordinate_shuffle.json", V21 / "gse202011_v21_external_strict.json"],
            [OUT / "table3_audit_diagnostics.csv"],
            "Generated from reproducibility checks and pre-specified diagnostic contrasts; none are alternative primary models.",
        ),
        item(
            "v8 DOCX",
            "rendered submission manuscript",
            [builder, ROOT / "manuscript" / "PRI_Audit_psoriasis_manuscript_v8_submission_ready.md", ROOT / "tmp" / "pri_audit_v8_docx" / "artifact.md", OUT / "table1_cohort_roles.csv", OUT / "table2_strict_matched_comparisons.csv", OUT / "table3_audit_diagnostics.csv"],
            [ROOT / "manuscript" / "PRI_Audit_psoriasis_manuscript_v8_submission_ready.docx"],
            "The document uses the retained reference only for visual language; all scientific content comes from the frozen v21 package.",
        ),
    ]
    rows: list[dict[str, object]] = []
    for entry in entries:
        for source in entry["source_files"]:
            rows.append({
                "identifier": entry["identifier"],
                "role": entry["role"],
                "source_path": source["path"],
                "source_sha256": source["sha256"],
                "output_paths": "; ".join(output["path"] for output in entry["output_files"]),
                "note": entry["note"],
            })
    payload = {"analysis": "PRI-Audit v8 submission-assembly source manifest", "entries": entries}
    (OUT / "v8_submission_source_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUT / "v8_submission_source_manifest.csv", index=False)
    print(f"Wrote {len(entries)} submission components and {len(rows)} source links")


if __name__ == "__main__":
    main()

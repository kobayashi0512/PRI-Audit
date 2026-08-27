"""Hash-map every v21 paper figure/table to code and exact result inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "pri_audit_v21"
FIG = ROOT / "figures" / "paper_v21"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def record(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    return {"path": str(path.relative_to(ROOT)), "sha256": digest(path), "bytes": path.stat().st_size}


def main() -> None:
    entries = [
        {
            "kind": "Figure", "identifier": "Figure 11", "caption_role": "Strictness, matched-reference and neighbourhood sensitivity audit",
            "generator": ROOT / "scripts" / "plot_v21_strictness_figure.py",
            "source_files": [OUT / "v21_loose_vs_strict_summary.csv", OUT / "v21_mismatched_reference_sensitivity.csv", OUT / "v21_nested_neighborhood_strictness.csv"],
            "outputs": [FIG / "fig11_v21_strictness_and_reference_audit.png", FIG / "fig11_v21_strictness_and_reference_audit.pdf"],
        },
        {
            "kind": "Figure", "identifier": "Figure 12", "caption_role": "Invalid random measurement split and spatial coordinate-null diagnostics",
            "generator": ROOT / "scripts" / "plot_v21_independence_and_spatial_null.py",
            "source_files": [OUT / "v21_loose_vs_strict_summary.csv", OUT / "v21_measurement_random_split_diagnostic.csv", OUT / "gse206391_v21_matched_spatial_coordinate_shuffle.json", OUT / "gse206391_v21_matched_spatial_coordinate_shuffle_distributions.npz"],
            "outputs": [FIG / "fig12_v21_independence_and_spatial_null.png", FIG / "fig12_v21_independence_and_spatial_null.pdf"],
        },
        {
            "kind": "Table", "identifier": "Table 1", "caption_role": "Cohort roles, independent units and raw-input traceability",
            "generator": ROOT / "scripts" / "build_v21_manuscript_tables.py",
            "source_files": [OUT / "v21_all11_input_registry.csv", OUT / "v21_all11_provenance_manifest.json"],
            "outputs": [ROOT / "results" / "paper_v21" / "table1_cohort_roles.csv", ROOT / "results" / "paper_v21" / "table1_cohort_roles.md"],
        },
        {
            "kind": "Table", "identifier": "Table 2", "caption_role": "Strict matched comparisons and spatial incremental tests",
            "generator": ROOT / "scripts" / "build_v21_manuscript_tables.py",
            "source_files": [OUT / "v21_loose_vs_strict_summary.csv", OUT / "gse206391_v21_matched_spatial_graph.json", OUT / "gse206391_v21_matched_spatial_coordinate_shuffle.json", OUT / "gse202011_v21_external_strict.json"],
            "outputs": [ROOT / "results" / "paper_v21" / "table2_strict_matched_comparisons.csv", ROOT / "results" / "paper_v21" / "table2_strict_matched_comparisons.md"],
        },
        {
            "kind": "Table", "identifier": "Table 3", "caption_role": "Raw reconstruction and counterfactual audit diagnostics",
            "generator": ROOT / "scripts" / "build_v21_manuscript_tables.py",
            "source_files": [
                OUT / "v21_rebuild_equivalence.csv",
                OUT / "v21_measurement_random_split_diagnostic.csv",
                OUT / "v21_nested_neighborhood_strictness.csv",
                OUT / "v21_mismatched_reference_sensitivity.csv",
                OUT / "gse206391_v21_matched_spatial_coordinate_shuffle.json",
                OUT / "gse202011_v21_external_strict.json",
            ],
            "outputs": [ROOT / "results" / "paper_v21" / "table3_audit_diagnostics.csv", ROOT / "results" / "paper_v21" / "table3_audit_diagnostics.md"],
        },
    ]
    json_entries = []
    rows = []
    for entry in entries:
        packed = {key: value for key, value in entry.items() if key not in {"generator", "source_files", "outputs"}}
        packed["generator"] = record(entry["generator"])
        packed["source_files"] = [record(path) for path in entry["source_files"]]
        packed["outputs"] = [record(path) for path in entry["outputs"]]
        json_entries.append(packed)
        for source in packed["source_files"]:
            rows.append({"kind": entry["kind"], "identifier": entry["identifier"], "caption_role": entry["caption_role"], "generator": packed["generator"]["path"], "source_path": source["path"], "source_sha256": source["sha256"], "source_bytes": source["bytes"], "output_paths": "; ".join(value["path"] for value in packed["outputs"])})
    pd.DataFrame(rows).to_csv(OUT / "v21_figure_table_source_manifest.csv", index=False)
    (OUT / "v21_figure_table_source_manifest.json").write_text(json.dumps({"analysis": "PRI-Audit v21 figure/table source manifest", "entries": json_entries}, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} figure/table source links")


if __name__ == "__main__":
    main()

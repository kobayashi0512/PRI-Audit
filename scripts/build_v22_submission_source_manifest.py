"""Hash-bind v22 simulations and topic comparison to the submission package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V21 = ROOT / "results" / "pri_audit_v21"
V22 = ROOT / "results" / "pri_audit_v22"
PAPER = ROOT / "results" / "paper_v21"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def record(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    return {"path": str(path.relative_to(ROOT)), "sha256": digest(path), "bytes": path.stat().st_size}


def add_entry(entries, identifier, role, sources, outputs, note):
    entries.append({
        "identifier": identifier,
        "role": role,
        "source_files": [record(path) for path in sources],
        "output_files": [record(path) for path in outputs],
        "note": note,
    })


def main() -> None:
    entries: list[dict[str, object]] = []
    add_entry(
        entries,
        "Figure 6",
        "known-truth hierarchy simulation",
        [
            ROOT / "scripts" / "run_v22_pri_audit_simulations.py",
            ROOT / "scripts" / "plot_v22_simulation_figure.py",
            ROOT / "docs" / "v22_submission_closure_protocol.md",
            V22 / "v22_simulation_replicates.csv",
            V22 / "v22_simulation_summary.csv",
            V22 / "v22_simulation_specification.json",
        ],
        [ROOT / "figures" / "paper_v22" / "fig6_v22_known_truth_simulation.png"],
        "Exact local rendering of the fixed-scenario simulation; no simulated quantity is hand-drawn.",
    )
    add_entry(
        entries,
        "Table 2 (v22 rows)",
        "fold-local spot-topic comparison",
        [
            ROOT / "scripts" / "run_v22_spot_topic_benchmark.py",
            V22 / "gse206391_v22_spot_topic_benchmark.json",
            V22 / "gse206391_v22_spot_topic_predictions.csv",
            V22 / "gse206391_v22_spot_topic_distributions.npz",
        ],
        [PAPER / "table2_strict_matched_comparisons.csv"],
        "The topic representation is a fixed low-capacity comparator fitted only on training-patient spots in each fold.",
    )
    add_entry(
        entries,
        "Supplementary Table S1",
        "simulation numeric summary",
        [ROOT / "scripts" / "build_v22_submission_tables.py", V22 / "v22_simulation_summary.csv"],
        [PAPER / "tableS1_v22_known_truth_simulation.csv"],
        "Machine-readable summary of all fixed simulated scenarios and independent-patient counts.",
    )
    add_entry(
        entries,
        "Supplementary Table S2",
        "spot-subsampling sensitivity",
        [
            ROOT / "scripts" / "build_v22_submission_tables.py",
            V22 / "gse206391_v22_spot_topic_benchmark_spots150.json",
            V22 / "gse206391_v22_spot_topic_benchmark.json",
            V22 / "gse206391_v22_spot_topic_benchmark_spots450.json",
        ],
        [PAPER / "tableS2_v22_spot_topic_sensitivity.csv"],
        "The 150- and 450-spot analyses are predeclared robustness checks; the 300-spot result remains primary.",
    )
    rows = []
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
    payload = {"analysis": "PRI-Audit v22 submission-closure source manifest", "entries": entries}
    (PAPER / "v22_submission_closure_source_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(PAPER / "v22_submission_closure_source_manifest.csv", index=False)
    print(f"Wrote {len(entries)} v22 components and {len(rows)} source links")


if __name__ == "__main__":
    main()

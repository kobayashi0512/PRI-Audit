"""Assemble individually uploadable BMC Bioinformatics submission assets.

The main manuscript retains editable tables and embedded figures. This helper
copies the six individual figure PNGs and two machine-readable supplementary
tables into a clean upload directory and writes SHA-256 inventory records.
It deliberately excludes the 3.2 GB GEO cache; public accessions, retrieval
instructions and input hashes remain in the repository and manuscript.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "manuscript" / "PRI_Audit_BMC_Bioinformatics_submission_ready.docx"
ASSETS = [
    ("Figure_1_dataset_landscape.png", ROOT / "figures" / "paper_v19_fig1_dataset_atlas.png"),
    ("Figure_2_PRI_Audit_framework.png", ROOT / "figures" / "paper_v20_fig2_overall_framework.png"),
    ("Figure_3_algorithmic_architecture.png", ROOT / "figures" / "paper_v20_fig3_algorithm_architecture.png"),
    ("Figure_4_strictness_reference_audit.png", ROOT / "figures" / "paper_v21" / "fig11_v21_strictness_and_reference_audit.png"),
    ("Figure_5_independence_spatial_null.png", ROOT / "figures" / "paper_v21" / "fig12_v21_independence_and_spatial_null.png"),
    ("Figure_6_known_truth_validation.png", ROOT / "figures" / "paper_v22" / "fig6_v22_known_truth_simulation.png"),
    ("Additional_file_1_known_truth_simulation.csv", ROOT / "results" / "paper_v21" / "tableS1_v22_known_truth_simulation.csv"),
    ("Additional_file_2_spot_topic_sensitivity.csv", ROOT / "results" / "paper_v21" / "tableS2_v22_spot_topic_sensitivity.csv"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_asset(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True, help="empty BMC upload-directory")
    args = parser.parse_args()
    destination = args.destination.expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Destination is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    copy_asset(MANUSCRIPT, destination / "PRI_Audit_BMC_Bioinformatics_submission_ready.docx")
    rows = [("PRI_Audit_BMC_Bioinformatics_submission_ready.docx", "Main manuscript with editable tables")]
    for name, source in ASSETS:
        copy_asset(source, destination / name)
        kind = "Individual figure" if name.startswith("Figure_") else "Machine-readable additional file"
        rows.append((name, kind))
    copy_asset(ROOT / "submission_package" / "COVER_LETTER_BMC_BIOINFORMATICS.md", destination / "COVER_LETTER_BMC_BIOINFORMATICS.md")
    rows.append(("COVER_LETTER_BMC_BIOINFORMATICS.md", "Cover-letter source; replace author approval and competing-interest placeholders"))

    inventory = [
        "# BMC Bioinformatics upload inventory",
        "",
        "This directory contains individually uploadable manuscript assets. It excludes raw GEO archives; accession identifiers, retrieval instructions and SHA-256 provenance are in the public repository.",
        "",
        "| File | Role | Bytes | SHA-256 |",
        "|---|---:|---:|---|",
    ]
    for name, role in rows:
        path = destination / name
        inventory.append(f"| `{name}` | {role} | {path.stat().st_size} | `{sha256(path)}` |")
    inventory.extend([
        "",
        "Before submission, authors must replace author/affiliation, funding, conflict, acknowledgement, ethics and archival-DOI placeholders; they must also choose a software licence.",
    ])
    (destination / "README.md").write_text("\n".join(inventory) + "\n", encoding="utf-8")
    print(f"Built {len(rows)} BMC upload assets in {destination}")


if __name__ == "__main__":
    main()

"""Build a clean, SHA-256-attested PRI-Audit public-release staging bundle.

The project workspace contains historical experiments, public GEO downloads,
local environments and private style references. A release should contain only
the submission-grade PRI-Audit source, generated evidence and editable figures.
This script copies that curated set to an empty, user-selected directory and
writes a machine-readable release manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

FILES = [
    "README.md",
    "CHANGELOG.md",
    "LICENSE_SELECTION_REQUIRED.md",
    "requirements.txt",
    ".gitignore",
    "CITATION.cff.template",
    "docs/pri_audit_v21_experiment_spec.md",
    "docs/v21_external_candidate_selection.md",
    "docs/v21_result_interpretation_and_claim_update.md",
    "docs/v22_submission_closure_protocol.md",
    "manuscript/PRI_Audit_psoriasis_manuscript_v8_submission_ready.md",
    "manuscript/PRI_Audit_psoriasis_manuscript_v9_submission_ready.docx",
    "manuscript/PRI_Audit_psoriasis_RegionSuff_template_submission.docx",
    "ppt_figures_v1/PRI_Audit_data_integrated_figures_v2_audited.pptx",
    "figures/paper_v19_fig1_dataset_atlas.png",
    "figures/paper_v20_fig2_overall_framework.png",
    "figures/paper_v20_fig3_algorithm_architecture.png",
]
DIRECTORIES = [
    "scripts",
    "reproducibility",
    "results/pri_audit_v21",
    "results/pri_audit_v22",
    "results/pri_audit_release",
    "results/paper_v21",
    "figures/paper_v21",
    "figures/paper_v22",
    "submission_package",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True, help="empty release-staging directory")
    args = parser.parse_args()
    destination = args.destination.expanduser().resolve()
    if destination == ROOT:
        raise ValueError("Release destination must not be the project root")
    if destination.exists():
        if any(destination.iterdir()):
            raise FileExistsError(f"Release destination is not empty: {destination}")
    else:
        destination.mkdir(parents=True)

    copied: list[dict[str, object]] = []
    for relative in FILES:
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        target = destination / relative
        copy_file(source, target)
        copied.append({"path": relative, "bytes": source.stat().st_size, "sha256": sha256(source)})
    for relative in DIRECTORIES:
        source = ROOT / relative
        if not source.is_dir():
            raise FileNotFoundError(source)
        target = destination / relative
        shutil.copytree(
            source,
            target,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".DS_Store"),
        )
        for item in sorted(source.rglob("*")):
            target_item = target / item.relative_to(source)
            if item.is_file() and target_item.is_file():
                copied.append({
                    "path": str(item.relative_to(ROOT)),
                    "bytes": item.stat().st_size,
                    "sha256": sha256(item),
                })

    manifest = {
        "release_name": "PRI-Audit public-release staging bundle",
        "source_workspace": str(ROOT),
        "n_files": len(copied),
        "files": copied,
        "excluded": [
            "data/ public GEO download cache",
            "local environments and temporary render folders",
            "private Word style references",
            "superseded manuscript drafts and historical development outputs",
        ],
    }
    (destination / "release_artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Built {len(copied)} files in {destination}")


if __name__ == "__main__":
    main()

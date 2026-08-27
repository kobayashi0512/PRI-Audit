"""One-command reproducibility runner for the v22 submission-closure analyses."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "pri_audit_v22"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=400)
    parser.add_argument("--bootstrap", type=int, default=250)
    parser.add_argument("--population-patients", type=int, default=240)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    commands = [
        ["scripts/run_v22_pri_audit_simulations.py", "--replicates", str(args.replicates), "--bootstrap", str(args.bootstrap), "--population-patients", str(args.population_patients), "--workers", str(args.workers)],
        ["scripts/run_v22_spot_topic_benchmark.py", "--bootstrap", "5000", "--permutations", "500"],
        ["scripts/run_v22_spot_topic_benchmark.py", "--max-spots-per-sample", "150", "--bootstrap", "2000", "--permutations", "300", "--tag", "spots150"],
        ["scripts/run_v22_spot_topic_benchmark.py", "--max-spots-per-sample", "450", "--bootstrap", "2000", "--permutations", "300", "--tag", "spots450"],
        ["scripts/plot_v22_simulation_figure.py"],
        ["scripts/build_v22_submission_tables.py"],
        ["scripts/build_v22_submission_source_manifest.py"],
        ["scripts/build_v9_submission_source_manifest.py"],
    ]
    log = {"started_utc": datetime.now(timezone.utc).isoformat(), "python": sys.executable, "commands": []}
    for command in commands:
        completed = subprocess.run([sys.executable, *command], cwd=ROOT, text=True, capture_output=True)
        log["commands"].append({"command": [sys.executable, *command], "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr})
        if completed.returncode:
            (OUT / "v22_submission_reproduction_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
            raise SystemExit(completed.returncode)
    outputs = [
        OUT / "v22_simulation_replicates.csv", OUT / "v22_simulation_summary.csv", OUT / "v22_simulation_specification.json",
        OUT / "gse206391_v22_spot_topic_benchmark.json", ROOT / "figures" / "paper_v22" / "fig6_v22_known_truth_simulation.png",
        ROOT / "results" / "paper_v21" / "table2_strict_matched_comparisons.csv",
        ROOT / "results" / "paper_v21" / "tableS1_v22_known_truth_simulation.csv",
        ROOT / "results" / "paper_v21" / "tableS2_v22_spot_topic_sensitivity.csv",
        ROOT / "results" / "paper_v21" / "v9_submission_source_manifest.json",
        ROOT / "manuscript" / "PRI_Audit_psoriasis_manuscript_v9_submission_ready.docx",
    ]
    log["finished_utc"] = datetime.now(timezone.utc).isoformat()
    log["output_sha256"] = {str(path.relative_to(ROOT)): sha256(path) for path in outputs}
    (OUT / "v22_submission_reproduction_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print("v22 submission-closure reproduction complete")


if __name__ == "__main__":
    main()

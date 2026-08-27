"""One-command, non-destructive reproduction of the frozen PRI-Audit v21 pack.

All commands write only into ``results/pri_audit_v21`` or
``figures/paper_v21``.  Historical development outputs are treated as read-only
equivalence references and are never overwritten.  The script writes a compact
machine-readable execution log so a clean rerun can be audited by a reviewer.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "pri_audit_v21"


def command(script: str, *args: str) -> list[str]:
    return [sys.executable, str(ROOT / "scripts" / script), *args]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    # Do not offer a reduced-resampling mode here: a quick run sharing this
    # output directory could silently overwrite a submission-grade result.
    permutation, bootstrap = "500", "5000"
    commands = [
        command("run_pri_audit_v21_pipeline.py", "--permutations", permutation, "--bootstrap", "2000"),
        command("run_v21_mismatched_reference.py", "--bootstrap", bootstrap),
        command("run_v21_nested_neighborhood.py", "--clusters", "8", "16", "32", "--bootstrap", bootstrap, "--cell-shuffles", permutation),
        command("run_v21_matched_spatial_graph.py", "--permutations", permutation, "--bootstrap", bootstrap),
        command("run_v21_matched_spatial_coordinate_shuffle.py", "--permutations", permutation),
        command("run_v21_measurement_split_diagnostic.py", "--repeats", "50"),
        command("run_v21_gse202011_external.py", "--permutations", permutation, "--bootstrap", bootstrap),
        command("update_v21_external_audit.py"),
        command("build_v21_all11_provenance.py"),
        command("build_v21_manuscript_tables.py"),
        command("plot_v21_strictness_figure.py"),
        command("plot_v21_independence_and_spatial_null.py"),
        command("build_v21_figure_table_manifest.py"),
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    log = {"started_utc": datetime.now(timezone.utc).isoformat(), "mode": "submission", "commands": []}
    for current in commands:
        started = datetime.now(timezone.utc).isoformat()
        result = subprocess.run(current, cwd=ROOT, text=True, capture_output=True)
        log["commands"].append({"command": current, "started_utc": started, "returncode": result.returncode, "stdout_tail": result.stdout[-4000:], "stderr_tail": result.stderr[-4000:]})
        (OUT / "v21_submission_reproduction_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"Reproduction stopped at: {' '.join(current)}")
    audited_outputs = [
        OUT / "reproducibility_manifest.json",
        OUT / "v21_rebuild_equivalence.csv",
        OUT / "v21_loose_vs_strict_summary.csv",
        OUT / "v21_mismatched_reference_sensitivity.csv",
        OUT / "v21_nested_neighborhood_strictness.csv",
        OUT / "gse206391_v21_matched_spatial_graph.json",
        OUT / "gse206391_v21_matched_spatial_coordinate_shuffle.json",
        OUT / "v21_measurement_random_split_diagnostic.csv",
        OUT / "gse202011_v21_external_strict.json",
        OUT / "v21_all11_provenance_manifest.json",
        OUT / "v21_figure_table_source_manifest.json",
        ROOT / "figures" / "paper_v21" / "fig11_v21_strictness_and_reference_audit.png",
        ROOT / "figures" / "paper_v21" / "fig12_v21_independence_and_spatial_null.png",
        ROOT / "results" / "paper_v21" / "table1_cohort_roles.csv",
        ROOT / "results" / "paper_v21" / "table2_strict_matched_comparisons.csv",
        ROOT / "results" / "paper_v21" / "table3_audit_diagnostics.csv",
    ]
    missing = [str(path.relative_to(ROOT)) for path in audited_outputs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Expected v21 outputs were not created: {missing}")
    log["finished_utc"] = datetime.now(timezone.utc).isoformat()
    log["output_sha256"] = {
        str(path.relative_to(ROOT)): sha256(path) for path in audited_outputs
    }
    (OUT / "v21_submission_reproduction_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print("PRI-Audit v21 reproduction completed")


if __name__ == "__main__":
    main()

"""Run and attest the complete frozen PRI-Audit submission workflow.

This small umbrella runner deliberately delegates scientific work to the two
versioned, independently auditable runners.  It does not add a quick mode or
change any resampling setting.  Its only output is a release-level execution
ledger that binds the v21 real-data and v22 closure logs by SHA-256.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "pri_audit_release"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(script: str, *args: str) -> dict[str, object]:
    command = [sys.executable, str(ROOT / "scripts" / script), *args]
    started = datetime.now(timezone.utc).isoformat()
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    return {
        "command": command,
        "started_utc": started,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    log_path = OUT / "complete_reproduction_log.json"
    log: dict[str, object] = {
        "analysis": "PRI-Audit complete submission reproduction",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.executable,
        "commands": [],
    }
    for script, args in [
        ("run_v21_submission_reproduction.py", []),
        (
            "run_v22_submission_reproduction.py",
            ["--replicates", "400", "--bootstrap", "250", "--population-patients", "240", "--workers", "4"],
        ),
        ("run_pri_audit_v24_gse151177_corrected.py", []),
    ]:
        result = run(script, *args)
        log["commands"].append(result)
        log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
        if result["returncode"] != 0:
            raise RuntimeError(f"Complete reproduction stopped at: {script}")

    component_logs = [
        ROOT / "results" / "pri_audit_v21" / "v21_submission_reproduction_log.json",
        ROOT / "results" / "pri_audit_v22" / "v22_submission_reproduction_log.json",
        ROOT / "results" / "pri_audit_v24_gse151177" / "reproducibility_manifest.json",
    ]
    missing = [str(path.relative_to(ROOT)) for path in component_logs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Component reproduction logs missing: {missing}")
    log["finished_utc"] = datetime.now(timezone.utc).isoformat()
    log["component_log_sha256"] = {
        str(path.relative_to(ROOT)): sha256(path) for path in component_logs
    }
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print("PRI-Audit complete submission reproduction completed")


if __name__ == "__main__":
    main()

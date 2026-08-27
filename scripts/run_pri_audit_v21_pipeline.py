"""Execute the fully isolated PRI-Audit v21 pipeline with one command."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def run(command: list[str]) -> None:
    print("\n[PRI-Audit v21]", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--skip-rebuild", action="store_true")
    args = parser.parse_args()
    python = sys.executable
    if not args.skip_rebuild:
        run([python, str(SCRIPTS / "run_pri_audit_v21.py"), "--rebuild-only"])
    for dataset in ("gse151177", "gse173706", "gse206391", "gse47751"):
        run([python, str(SCRIPTS / "run_v21_evaluate_dataset.py"), "--dataset", dataset, "--permutations", str(args.permutations), "--bootstrap", str(args.bootstrap)])
    run([python, str(SCRIPTS / "finalize_pri_audit_v21.py")])


if __name__ == "__main__":
    main()

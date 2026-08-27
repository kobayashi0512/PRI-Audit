"""Resumable v13 experiment queue with an auditable status file."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/paper_v13"
OUT.mkdir(parents=True, exist_ok=True)
STATUS = OUT / "experiment_batch_status.json"
PY = sys.executable

TASKS = [
    ("optimism_gap_patient_bootstrap", "scripts/run_v13_neighborhood_optimism_bootstrap.py"),
    ("cell_label_coordinate_null_distributions", "scripts/run_cell_cross_patient_shuffle.py"),
    ("strict_spatial_graph_resampling_distributions", "scripts/run_strict_spatial_graph_benchmark.py"),
    ("external_transfer_null_distributions", "scripts/check_neighborhood_transfer_stability.py"),
    ("external_transfer_calibration", "scripts/run_v13_external_calibration.py"),
    ("strict_spatial_coordinate_shuffle", "scripts/run_v13_strict_spatial_coordinate_shuffle.py"),
    ("representative_LS_NL_spatial_map", "scripts/run_v13_representative_spatial_map.py"),
    ("HVG_neighbor_regularization_seed_sensitivity", "scripts/run_v13_sensitivity_matrix.py"),
    ("publication_plots", "scripts/plot_v13_six_experiments.py"),
]


def load():
    if STATUS.exists():
        return json.loads(STATUS.read_text())
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    return {"batch": "psoriasis_v13_six_experiments", "started_at": now, "updated_at": now, "tasks": {name: {"status": "pending", "script": script} for name, script in TASKS}}


def save(state):
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    STATUS.write_text(json.dumps(state, indent=2), encoding="utf-8")


def main():
    state = load(); save(state)
    for name, script in TASKS:
        record = state["tasks"][name]
        if record.get("status") == "complete":
            continue
        record.update({"status": "running", "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        save(state)
        log = OUT / f"{name}.log"
        with log.open("a", encoding="utf-8") as fh:
            fh.write(f"\n=== {time.strftime('%Y-%m-%dT%H:%M:%S%z')} ===\n")
            proc = subprocess.run([PY, str(ROOT / script)], cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=os.environ.copy())
        if proc.returncode == 0:
            record.update({"status": "complete", "returncode": 0, "log": str(log), "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        else:
            record.update({"status": "failed", "returncode": proc.returncode, "log": str(log), "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
            save(state)
            raise SystemExit(proc.returncode)
        save(state)
    state["status"] = "complete" if all(x["status"] == "complete" for x in state["tasks"].values()) else "incomplete"
    save(state)
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()

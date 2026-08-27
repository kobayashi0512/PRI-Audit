"""Combine isolated v21 evaluation results only after all datasets complete."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_pri_audit_v21 as v21  # noqa: E402


def main() -> None:
    datasets = ["gse151177", "gse173706", "gse206391", "gse47751"]
    result_paths = [v21.OUT / f"v21_{dataset}_strict_loose_result.json" for dataset in datasets]
    missing = [path.name for path in result_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Cannot finalize v21; missing isolated dataset results: {missing}")
    methods = []
    for path in result_paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        methods.extend(result["methods"])
    table = pd.DataFrame(methods)
    table.to_csv(v21.OUT / "v21_loose_vs_strict_summary.csv", index=False)
    equivalence = pd.read_csv(v21.OUT / "v21_rebuild_equivalence.csv")
    audit = pd.read_csv(v21.OUT / "v21_external_cohort_audit.csv")
    null_files = sorted(path.name for path in v21.OUT.glob("*_strict_loose_nulls.npz"))
    v21.write_json(v21.OUT / "v21_run_summary.json", {
        "specification": "docs/pri_audit_v21_experiment_spec.md",
        "status": "complete",
        "classifier": "L2 logistic regression (C=1, class_weight=balanced, solver=liblinear, max_iter=2000)",
        "rebuild_equivalence": equivalence.to_dict(orient="records"),
        "external_selection": audit.to_dict(orient="records"),
        "evaluation_result_files": [path.name for path in result_paths],
        "full_null_distribution_files": null_files,
        "n_evaluation_rows": int(len(table)),
    })
    print(table[["dataset", "arm", "auroc", "incremental_auroc_vs_strict_expression"]].to_string(index=False))


if __name__ == "__main__":
    main()

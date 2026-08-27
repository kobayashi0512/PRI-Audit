"""Run one v21 strict/loose evaluation in an isolated Python process.

Isolation deliberately releases native optimiser memory after every dataset.
It is part of the reproducibility design, not parallel hyperparameter search.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_pri_audit_v21 as v21  # noqa: E402


def load_core(stem: str):
    counts = np.load(v21.OUT / f"{stem}_sample_gene_counts.npz")["counts"]
    metadata = pd.read_csv(v21.OUT / f"{stem}_sample_metadata.csv")
    return v21.eligible_core(stem, counts, metadata)


def load_external():
    expression = np.load(v21.OUT / "gse47751_baseline_expression.npz")["expression"]
    metadata = pd.read_csv(v21.OUT / "gse47751_baseline_metadata.csv")
    labels = (metadata.condition == "lesional").astype(int).to_numpy()
    return expression, labels, metadata.patient.to_numpy(), metadata, True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["gse151177", "gse173706", "gse206391", "gse47751"])
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    if args.dataset == "gse47751":
        x, y, groups, metadata, paired = load_external()
        name, allow_nmf = "GSE47751", False
    else:
        x, y, groups, metadata, paired = load_core(args.dataset)
        name, allow_nmf = args.dataset.upper(), True
    print(f"Starting {name}: {len(y)} samples, {len(np.unique(groups))} independent groups", flush=True)
    result, _ = v21.evaluate_strict_loose(name, x, y, groups, metadata, paired, allow_nmf, args.permutations, args.bootstrap, v21.OUT)
    path = v21.OUT / f"v21_{args.dataset}_strict_loose_result.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Completed {name}; wrote {path.name}", flush=True)


if __name__ == "__main__":
    main()

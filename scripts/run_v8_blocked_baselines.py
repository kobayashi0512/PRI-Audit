"""Fixed blocked-gene generalization baselines for the C-TRIP task."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "trigger_v8"


def cosine(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom else np.nan


def block(gene):
    return int(hashlib.sha256(gene.encode()).hexdigest()[:8], 16) % 5


def main():
    x = pd.read_parquet(OUT / "v8_clinical_ls_minus_nl_rank.parquet").to_numpy(dtype=float)
    b = pd.read_parquet(OUT / "v8_perturbation_dictionary.parquet").to_numpy(dtype=float)[:, :2]
    genes = pd.read_csv(OUT / "v8_common_genes.csv").gene.tolist()
    meta = pd.read_csv(OUT / "v8_clinical_ls_minus_nl_metadata.csv")
    blocks = np.array([block(g) for g in genes])
    rows = []
    for i, patient in meta.iterrows():
        for held in range(5):
            train, test = blocks != held, blocks == held
            # Ordinary NNLS is the strongest fixed per-patient projection
            # baseline on observed genes; evaluation uses entirely unseen genes.
            a_nnls, _ = nnls(b[train], x[i, train])
            pred_nnls = b[test] @ a_nnls
            ridge = np.linalg.solve(b[train].T @ b[train] + .1 * np.eye(2), b[train].T @ x[i, train])
            pred_ridge = b[test] @ ridge
            for name, pred in [("nnls", pred_nnls), ("ridge", pred_ridge), ("il17_only", b[test, 0] * max(0, (b[train, 0] @ x[i, train]) / (b[train, 0] @ b[train, 0]))),
                               ("tnf_only", b[test, 1] * max(0, (b[train, 1] @ x[i, train]) / (b[train, 1] @ b[train, 1])))] :
                rows.append({"cohort": patient.cohort, "subject": patient.subject, "gene_block": held, "method": name,
                             "held_gene_cosine": cosine(x[i, test], pred)})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "v8_blocked_gene_baselines.csv", index=False)
    summary = frame.groupby(["cohort", "method"], as_index=False).held_gene_cosine.mean()
    summary.to_csv(OUT / "v8_blocked_gene_baseline_summary.csv", index=False)
    print(summary.pivot(index="cohort", columns="method", values="held_gene_cosine").round(3).to_string())


if __name__ == "__main__":
    main()

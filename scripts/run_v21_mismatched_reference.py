"""Show why an unmatched low-dimensional expression control is not a baseline.

The analysis is deliberately labelled as a didactic sensitivity.  It compares
the fixed strict PCA/NMF candidates with a 16-gene expression classifier, which
has a materially different feature budget from the 2,000-HVG expression
reference.  It can illuminate an apparent gain created by an unfair comparator,
but it is never part of the PRI-Audit primary gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_pri_audit_v21 as v21  # noqa: E402


def mismatched_predictions(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Outer-group prediction with 16 HVGs fit only in the training groups."""
    probability = np.full(y.size, np.nan)
    for group in np.unique(groups):
        test = groups == group
        train = ~test
        top = np.argsort(x[train].var(axis=0))[-min(16, x.shape[1]):]
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1, solver="liblinear", class_weight="balanced", max_iter=2000, random_state=0),
        )
        model.fit(x[train][:, top], y[train])
        probability[test] = model.predict_proba(x[test][:, top])[:, 1]
    if np.isnan(probability).any():
        raise RuntimeError("Missing mismatched-control predictions")
    return probability


def core_input(stem: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.load(v21.OUT / f"{stem}_sample_gene_counts.npz")["counts"]
    metadata = pd.read_csv(v21.OUT / f"{stem}_sample_metadata.csv")
    x, y, groups, _, _ = v21.eligible_core(stem, counts, metadata)
    return x, y, groups


def external_input() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(v21.OUT / "gse47751_baseline_expression.npz")
    metadata = pd.read_csv(v21.OUT / "gse47751_baseline_metadata.csv")
    return data["expression"], (metadata.condition == "lesional").astype(int).to_numpy(), metadata.patient.to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()
    rng = np.random.default_rng(v21.SEED + 1)
    records = []
    inputs = [(stem.upper(), *core_input(stem)) for stem in ("gse151177", "gse173706", "gse206391")]
    inputs.append(("GSE47751", *external_input()))
    for dataset, x, y, groups in inputs:
        mismatch = mismatched_predictions(x, y, groups)
        candidate_path = v21.OUT / f"{dataset}_strict_loose_predictions.csv"
        predictions = pd.read_csv(candidate_path)
        candidates = ["strict_pca_probability"]
        if "strict_nmf_probability" in predictions:
            candidates.append("strict_nmf_probability")
        for candidate_name in candidates:
            candidate = predictions[candidate_name].to_numpy(dtype=float)
            ci, values = v21.bootstrap_delta(y, candidate, mismatch, groups, rng, args.bootstrap)
            records.append({
                "dataset": dataset,
                "candidate_arm": candidate_name.removesuffix("_probability"),
                "mismatched_reference": "strict_expression_16_HVG",
                "formal_role": "didactic_mismatched_reference_sensitivity_only",
                "candidate_auroc": float(roc_auc_score(y, candidate)),
                "mismatched_reference_auroc": float(roc_auc_score(y, mismatch)),
                "apparent_increment_vs_mismatched_reference": float(roc_auc_score(y, candidate) - roc_auc_score(y, mismatch)),
                "paired_group_bootstrap_ci95_low": ci[0],
                "paired_group_bootstrap_ci95_high": ci[1],
                "paired_group_bootstrap_probability_increment_ge_0_05": float(np.mean(values >= .05)),
                "n_bootstrap": int(values.size),
            })
        save = pd.DataFrame({"true": y, "outer_group": groups, "strict_expression_16_hvg_probability": mismatch})
        save.to_csv(v21.OUT / f"{dataset}_mismatched_reference_predictions.csv", index=False)
    frame = pd.DataFrame(records)
    frame.to_csv(v21.OUT / "v21_mismatched_reference_sensitivity.csv", index=False)
    (v21.OUT / "v21_mismatched_reference_sensitivity.json").write_text(json.dumps(frame.to_dict(orient="records"), indent=2), encoding="utf-8")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()

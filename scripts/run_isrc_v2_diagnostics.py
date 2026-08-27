from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "isrc_v2"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_cone_v0 import explained_variance_score, normalize_columns, rank_unit, unit  # noqa: E402
from run_isrc_v2 import (  # noqa: E402
    load_gse57225_triplets,
    load_locked_model,
    project_nuisance,
)


def main() -> None:
    genes, _, model, _ = load_locked_model()
    deltas, _ = load_gse57225_triplets(genes)
    baseline = model["baseline_dictionary"]
    q = model["nuisance_basis"]
    residual = model["novel_residual"]
    dictionaries = {
        "v0_raw5": baseline,
        "v0_plus_residual": normalize_columns(
            np.column_stack([baseline, residual])
        ),
        "nuisance_projected_v0": normalize_columns(
            project_nuisance(baseline, q)
        ),
        "full_isrc": model["candidate_dictionary"],
    }
    rows = []
    for disease, label in (("PSO", 1), ("ECZEMA", 0)):
        for patient in deltas[disease].columns:
            target = rank_unit(
                deltas[disease].loc[genes, patient].to_numpy()
            )
            projected_target = unit(project_nuisance(target, q))
            row = {"patient": patient, "disease": disease, "label": label}
            for name, dictionary in dictionaries.items():
                scoring_target = (
                    projected_target
                    if name in {"nuisance_projected_v0", "full_isrc"}
                    else target
                )
                weights, _ = nnls(dictionary, scoring_target)
                row[name] = explained_variance_score(
                    scoring_target, dictionary, weights
                )
            rows.append(row)
    predictions = pd.DataFrame(rows)
    metrics = {}
    for name in dictionaries:
        paired = predictions.pivot(
            index="patient", columns="disease", values=name
        )
        metrics[name] = {
            "auc": float(
                roc_auc_score(predictions["label"], predictions[name])
            ),
            "pso_gt_eczema": int((paired["PSO"] > paired["ECZEMA"]).sum()),
            "n_patients": len(paired),
        }
    reliability = pd.read_csv(RESULTS / "gene_reliability.csv")
    stable = reliability.loc[reliability["sign_agreement"].eq(1)].copy()
    stable["absolute_weighted_consensus"] = stable[
        "weighted_consensus"
    ].abs()
    stable.nlargest(100, "absolute_weighted_consensus").to_csv(
        RESULTS / "top_stable_residual_genes.csv", index=False
    )
    predictions.to_csv(
        RESULTS / "posthoc_ablation_predictions.csv", index=False
    )
    (RESULTS / "posthoc_ablation_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2)
    )
    final_path = RESULTS / "final_metrics.json"
    if final_path.exists():
        final_metrics = json.loads(final_path.read_text())
        final_metrics["posthoc_ablation"] = metrics
        final_path.write_text(
            json.dumps(final_metrics, ensure_ascii=False, indent=2)
        )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

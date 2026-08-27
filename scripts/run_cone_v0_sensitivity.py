from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_cone_v0 as cone  # noqa: E402
from run_feasibility import load_gse121212_deltas  # noqa: E402


def main() -> None:
    effects, _ = cone.load_perturbation_deltas()
    pso_delta, ad_delta = load_gse121212_deltas()
    rows = []
    for top_n in (100, 150, 200, 250, 300, 400):
        cone.TOP_N = top_n
        genes = cone.select_features(effects, pso_delta, ad_delta)
        raw, _, interaction, _, synergy = cone.build_dictionaries(effects, genes)
        pso = cone.disease_matrix(pso_delta, genes)
        ad = cone.disease_matrix(ad_delta, genes)
        sample_names = [
            *[f"PSO_{column}" for column in pso_delta.columns],
            *[f"AD_{column}" for column in ad_delta.columns],
        ]
        fold_scores, _ = cone.cross_validated_predictions(
            pso, ad, raw, interaction, sample_names
        )
        patient_scores = (
            fold_scores.groupby(["sample", "label"], as_index=False)[
                [
                    "max_correlation_raw5",
                    "nnls_raw5",
                    "contrastive_sparse_cone",
                ]
            ]
            .mean()
        )
        labels = patient_scores["label"].to_numpy()
        rows.append(
            {
                "top_n_per_tail": top_n,
                "n_features": len(genes),
                "max_correlation_auc": roc_auc_score(
                    labels, patient_scores["max_correlation_raw5"]
                ),
                "nnls_auc": roc_auc_score(labels, patient_scores["nnls_raw5"]),
                "contrastive_sparse_cone_auc": roc_auc_score(
                    labels, patient_scores["contrastive_sparse_cone"]
                ),
                "synergy_residual_fraction": synergy["residual_fraction"],
            }
        )
    output = pd.DataFrame(rows)
    output.to_csv(
        ROOT / "results" / "cone_v0" / "feature_count_sensitivity.csv",
        index=False,
    )
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()

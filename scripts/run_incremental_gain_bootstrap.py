"""Paired group-bootstrap uncertainty for AUROC increments over expression."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results/singlecell_v10"
OUT = ROOT / "results/paper_v12"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20260810)


def paired_bootstrap(y, baseline, candidate, groups, n_boot=10000):
    unique_groups = np.unique(groups)
    by_group = {group: np.flatnonzero(groups == group) for group in unique_groups}
    values = []
    for _ in range(n_boot):
        sampled = RNG.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([by_group[group] for group in sampled])
        if np.unique(y[idx]).size < 2:
            continue
        values.append(
            roc_auc_score(y[idx], candidate[idx])
            - roc_auc_score(y[idx], baseline[idx])
        )
    values = np.asarray(values)
    observed = float(roc_auc_score(y, candidate) - roc_auc_score(y, baseline))
    return {
        "delta_auroc": observed,
        "bootstrap_ci95": [
            float(value) for value in np.quantile(values, [0.025, 0.975])
        ],
        "bootstrap_probability_delta_gt_0": float(np.mean(values > 0)),
        "bootstrap_probability_delta_ge_0.05": float(np.mean(values >= 0.05)),
        "n_bootstrap": int(len(values)),
    }


def add_strict_low_rank(rows, details):
    for stem in ("gse151177", "gse173706", "gse206391"):
        frame = pd.read_csv(RES / f"{stem}_strict_benchmark_predictions.csv")
        y = frame.true.to_numpy()
        groups = frame.outer_group.astype(str).to_numpy()
        baseline = frame.pseudobulk_probability.to_numpy()
        for method, column in (
            ("strict PCA", "pca_probability"),
            ("strict NMF", "nmf_probability"),
        ):
            result = paired_bootstrap(
                y, baseline, frame[column].to_numpy(), groups
            )
            dataset = stem.upper()
            details.append({"dataset": dataset, "method": method, **result})
            rows.append(
                {
                    "dataset": dataset,
                    "candidate": method,
                    "reference": "strict pseudobulk",
                    "delta_auroc": result["delta_auroc"],
                    "delta_ci95_low": result["bootstrap_ci95"][0],
                    "delta_ci95_high": result["bootstrap_ci95"][1],
                    "probability_delta_ge_0.05": result[
                        "bootstrap_probability_delta_ge_0.05"
                    ],
                }
            )


def add_external_transfer(rows, details):
    transfer = np.load(
        RES / "gse173706_neighborhood_transfer_arrays.npz", allow_pickle=True
    )
    baseline = pd.read_csv(RES / "gse173706_strict_benchmark_predictions.csv")
    baseline = baseline.set_index("sample").loc[transfer["samples"].astype(str)]
    result = paired_bootstrap(
        transfer["labels"],
        baseline.pseudobulk_probability.to_numpy(),
        transfer["probabilities"],
        transfer["donors"].astype(str),
    )
    details.append(
        {"dataset": "GSE173706", "method": "transferred neighborhood", **result}
    )
    rows.append(
        {
            "dataset": "GSE173706",
            "candidate": "transferred neighborhood",
            "reference": "strict pseudobulk",
            "delta_auroc": result["delta_auroc"],
            "delta_ci95_low": result["bootstrap_ci95"][0],
            "delta_ci95_high": result["bootstrap_ci95"][1],
            "probability_delta_ge_0.05": result[
                "bootstrap_probability_delta_ge_0.05"
            ],
        }
    )


def add_spatial_graph(rows, details):
    frame = pd.read_csv(RES / "gse206391_strict_spatial_graph_predictions.csv")
    y = frame.true.to_numpy()
    groups = frame.outer_group.astype(str).to_numpy()
    baseline = frame.expression_500_probability.to_numpy()
    for method, column in (
        ("spatial graph", "spatial_graph_probability"),
        ("expression + spatial graph", "expression_plus_spatial_probability"),
    ):
        result = paired_bootstrap(y, baseline, frame[column].to_numpy(), groups)
        details.append({"dataset": "GSE206391", "method": method, **result})
        rows.append(
            {
                "dataset": "GSE206391",
                "candidate": method,
                "reference": "matched 500-gene expression",
                "delta_auroc": result["delta_auroc"],
                "delta_ci95_low": result["bootstrap_ci95"][0],
                "delta_ci95_high": result["bootstrap_ci95"][1],
                "probability_delta_ge_0.05": result[
                    "bootstrap_probability_delta_ge_0.05"
                ],
            }
        )


def main():
    rows = []
    details = []
    add_strict_low_rank(rows, details)
    add_external_transfer(rows, details)
    add_spatial_graph(rows, details)
    pd.DataFrame(rows).to_csv(OUT / "incremental_gain_bootstrap.csv", index=False)
    (OUT / "incremental_gain_bootstrap.json").write_text(
        json.dumps(details, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()

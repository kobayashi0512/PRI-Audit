from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "external_v1"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_cone_v0 import (  # noqa: E402
    SEED,
    bootstrap_auc,
    build_dictionaries,
    explained_variance_score,
    load_perturbation_deltas,
    rank_unit,
)
from run_external_v1 import (  # noqa: E402
    N_BOOT,
    broad_residual_signatures,
    build_expanded_dictionary,
    load_gse182740_deltas,
    load_gse36287_effects,
    load_gse86451_deltas,
)
from run_feasibility import load_gse121212_deltas  # noqa: E402


def axis_dictionary(
    axes: dict[str, pd.Series], names: list[str], genes: list[str]
) -> np.ndarray:
    return np.column_stack(
        [rank_unit(axes[name].loc[genes].to_numpy()) for name in names]
    )


def score_dictionary(
    dictionary: np.ndarray,
    genes: list[str],
    external: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for disease in ("PSO", "AD"):
        for patient in external[disease].columns:
            target = rank_unit(external[disease].loc[genes, patient].to_numpy())
            weights, _ = nnls(dictionary, target)
            rows.append(
                {
                    "sample": f"{disease}_{patient}",
                    "disease": disease,
                    "label": int(disease == "PSO"),
                    "score": explained_variance_score(
                        target, dictionary, weights
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    discovery_effects, discovery_deltas = load_perturbation_deltas()
    replication_deltas = load_gse86451_deltas()
    broad_effects = load_gse36287_effects()
    training_pso, training_ad = load_gse121212_deltas()
    external = load_gse182740_deltas()
    discovery_residual, replication_residual = broad_residual_signatures(
        discovery_deltas, replication_deltas
    )
    _, expanded_names, expanded_features, expanded_axes = (
        build_expanded_dictionary(
            discovery_effects,
            replication_deltas,
            broad_effects,
            discovery_residual,
            replication_residual,
            training_pso,
            training_ad,
            external,
        )
    )

    frozen = pd.read_csv(
        ROOT / "results" / "cone_v0" / "frozen_feature_genes.csv"
    )["gene"].astype(str)
    frozen_features = [
        gene
        for gene in frozen
        if gene in external["PSO"].index
        and gene in external["AD"].index
        and gene in training_pso.index
        and gene in training_ad.index
    ]
    common_frozen_features = [
        gene
        for gene in frozen_features
        if all(gene in axis.index for axis in expanded_axes.values())
    ]
    v0_frozen, _, _, _, _ = build_dictionaries(
        discovery_effects, frozen_features
    )
    v0_common_frozen, _, _, _, _ = build_dictionaries(
        discovery_effects, common_frozen_features
    )
    v0_expanded, _, _, _, _ = build_dictionaries(
        discovery_effects, expanded_features
    )
    core_names = ["IL17_CONSENSUS", "TNF_CONSENSUS", "SYNERGY_CONSENSUS"]
    configurations = {
        "v0_raw5_frozen_features": (
            v0_frozen,
            frozen_features,
        ),
        "v0_raw5_expanded_features": (
            v0_expanded,
            expanded_features,
        ),
        "v0_raw5_common_frozen_features": (
            v0_common_frozen,
            common_frozen_features,
        ),
        "consensus_core3_frozen_features": (
            axis_dictionary(expanded_axes, core_names, common_frozen_features),
            common_frozen_features,
        ),
        "consensus_core3_expanded_features": (
            axis_dictionary(expanded_axes, core_names, expanded_features),
            expanded_features,
        ),
        "expanded7_frozen_features": (
            axis_dictionary(
                expanded_axes, expanded_names, common_frozen_features
            ),
            common_frozen_features,
        ),
        "expanded7_expanded_features": (
            axis_dictionary(expanded_axes, expanded_names, expanded_features),
            expanded_features,
        ),
    }

    prediction_tables = []
    metrics = {}
    for index, (name, (dictionary, genes)) in enumerate(configurations.items()):
        predictions = score_dictionary(dictionary, genes, external)
        predictions.insert(0, "configuration", name)
        prediction_tables.append(predictions)
        metric = bootstrap_auc(
            predictions["label"].to_numpy(),
            predictions["score"].to_numpy(),
            n_boot=N_BOOT,
            seed=SEED + 100 + index,
        )
        metric["n_features"] = len(genes)
        metric["n_axes"] = dictionary.shape[1]
        metric["condition_number"] = float(np.linalg.cond(dictionary))
        metrics[name] = metric

    pd.concat(prediction_tables, ignore_index=True).to_csv(
        RESULTS / "posthoc_representation_diagnostics.csv", index=False
    )
    (RESULTS / "posthoc_representation_diagnostics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2)
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

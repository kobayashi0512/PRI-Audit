"""Build frozen main tables, claim registry and figure source map for v14."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results/singlecell_v10"
V12 = ROOT / "results/paper_v12"
OUT = ROOT / "results/paper_v14"
OUT.mkdir(parents=True, exist_ok=True)


def js(path):
    return json.loads(Path(path).read_text())


def main():
    datasets = pd.read_csv(V12 / "table1_datasets.csv").rename(columns={"patients_or_pairs": "independent_units_after_filtering"})
    datasets["independent_unit"] = np.where(datasets.modality.eq("bulk"), "paired patient", np.where(datasets.dataset.eq("GSE173706"), "donor", np.where(datasets.dataset.eq("GSE24767"), "perturbation experiment", "patient/sample")))
    datasets = datasets[["dataset", "modality", "design", "independent_unit", "independent_units_after_filtering", "cells_or_spots", "role"]]
    datasets.to_csv(OUT / "main_table1_public_datasets.csv", index=False)

    gains = pd.read_csv(V12 / "incremental_gain_bootstrap.csv")
    strict = {row["dataset"]: row for row in js(RES / "strict_multiresolution_benchmarks.json")}
    spatial = js(RES / "gse206391_strict_spatial_graph_benchmark.json")["methods"]
    candidate_auc = {
        ("GSE151177", "strict PCA"): strict["GSE151177"]["methods"]["pca"]["auroc"],
        ("GSE151177", "strict NMF"): strict["GSE151177"]["methods"]["nmf"]["auroc"],
        ("GSE173706", "strict PCA"): strict["GSE173706"]["methods"]["pca"]["auroc"],
        ("GSE173706", "strict NMF"): strict["GSE173706"]["methods"]["nmf"]["auroc"],
        ("GSE206391", "strict PCA"): strict["GSE206391"]["methods"]["pca"]["auroc"],
        ("GSE206391", "strict NMF"): strict["GSE206391"]["methods"]["nmf"]["auroc"],
        ("GSE173706", "transferred neighborhood"): js(RES / "gse173706_neighborhood_transfer_summary.json")["transfer_auroc"],
        ("GSE206391", "spatial graph"): spatial["spatial_graph"]["auroc"],
        ("GSE206391", "expression + spatial graph"): spatial["expression_plus_spatial"]["auroc"],
    }
    reference_auc = {
        "GSE151177": strict["GSE151177"]["methods"]["pseudobulk"]["auroc"],
        "GSE173706": strict["GSE173706"]["methods"]["pseudobulk"]["auroc"],
        "GSE206391": strict["GSE206391"]["methods"]["pseudobulk"]["auroc"],
    }
    rows = []
    for _, row in gains.iterrows():
        key = (row.dataset, row.candidate)
        ref_auc = spatial["expression_500"]["auroc"] if row.reference == "matched 500-gene expression" else reference_auc[row.dataset]
        external = row.dataset == "GSE173706" and row.candidate == "transferred neighborhood"
        falsification = "paired label permutation" if row.dataset == "GSE173706" else ("strict coordinate shuffle" if "spatial" in row.candidate else "paired/sample label permutation")
        rows.append({
            "dataset": row.dataset,
            "candidate_representation": row.candidate,
            "matched_reference": row.reference,
            "candidate_auroc": candidate_auc[key],
            "reference_auroc": ref_auc,
            "delta_auroc": row.delta_auroc,
            "delta_ci95_low": row.delta_ci95_low,
            "delta_ci95_high": row.delta_ci95_high,
            "probability_delta_ge_0_05": row["probability_delta_ge_0.05"],
            "strict_outer_fit": True,
            "external_validation": external,
            "falsification": falsification,
            "calibration": "reported; unstable" if external else "not applicable",
            "claim_status": "no incremental value demonstrated",
        })
    pd.DataFrame(rows).to_csv(OUT / "main_table2_patient_level_incremental_value.csv", index=False)

    claims = pd.DataFrame([
        ["C1", "Patient-level expression contains reproducible psoriasis disease signal across resolutions.", "supported", "contains reproducible disease-state signal", "identifies causal triggers", "Fig. 2"],
        ["C2", "Whole-dataset unsupervised fitting can inflate neighborhood AUROC relative to outer-fold refitting.", "supported", "produces measurable optimism", "always causes leakage in every dataset", "Fig. 3"],
        ["C3", "A fixed cell-neighborhood representation transfers ranking to GSE173706.", "supported with calibration limitation", "transfers ranking", "is calibrated or clinically deployable", "Fig. 4"],
        ["C4", "Transferred neighborhood representation improves over strict expression.", "refuted", "does not add patient-level AUROC", "outperforms expression", "Fig. 4"],
        ["C5", "Strict spatial graph statistics encode lesional state.", "supported", "capture disease-associated structure", "recover spatial causes", "Fig. 5"],
        ["C6", "Strict spatial graph statistics add coordinate-specific predictive value beyond matched expression.", "not demonstrated", "add no reproducible increment", "provide superior spatial prediction", "Fig. 5"],
        ["C7", "PRI-Audit distinguishes disease signal from representation-specific incremental value.", "supported as framework contribution", "audits incremental value under patient isolation", "proves universal inferiority of high-resolution methods", "Fig. 6"],
    ], columns=["claim_id", "frozen_statement", "evidence_status", "allowed_wording", "prohibited_wording", "primary_display"])
    claims.to_csv(OUT / "claim_registry.csv", index=False)

    sources = pd.DataFrame([
        ["Fig1", "a-e", "Study hierarchy, dataset roles, outer-fit boundary and falsification operators", "main_table1_public_datasets.csv; table2_methods_and_leakage.csv"],
        ["Fig2", "a", "Seven-cohort bulk transfer", "results/trigger_v9/v9_tuned_pooled_results.csv"],
        ["Fig2", "b", "Strict expression AUROC and bootstrap CI", "strict_multiresolution_benchmarks.json"],
        ["Fig2", "c-d", "Paired incremental AUROC", "incremental_gain_bootstrap.csv"],
        ["Fig3", "b-c", "Global-strict neighborhood optimism", "gse151177_neighborhood_optimism_bootstrap.csv"],
        ["Fig3", "d", "Cross-patient cell shuffle", "cell_cross_patient_shuffle_null_distribution.npz"],
        ["Fig4", "b-c", "External transfer ranking and paired donors", "gse173706_neighborhood_transfer_arrays.npz"],
        ["Fig4", "d", "Reliability and calibration", "gse173706_neighborhood_transfer_calibration.json"],
        ["Fig4", "e-f", "External paired gain and label null", "incremental_gain_bootstrap.csv; gse173706_neighborhood_transfer_null_distributions.npz"],
        ["Fig5", "b", "Predefined representative LS/NL maps", "gse206391_representative_spatial_selection.json; representative map PNG"],
        ["Fig5", "c-e", "Strict spatial matched comparison", "gse206391_strict_spatial_graph_benchmark.json; predictions.csv"],
        ["Fig5", "f", "Strict coordinate null", "gse206391_strict_spatial_coordinate_shuffle_distributions.npz"],
        ["Fig6", "a", "All paired incremental gains", "incremental_gain_bootstrap.csv"],
        ["Fig6", "b", "Resolution-specific falsification", "cell, label and coordinate NPZ distributions"],
        ["Fig6", "c-e", "Evidence matrix and claim gate", "claim_registry.csv"],
    ], columns=["figure", "panel", "content", "source"])
    sources.to_csv(OUT / "figure_source_data_map.csv", index=False)
    print(json.dumps({"datasets": len(datasets), "incremental_rows": len(rows), "claims": len(claims), "source_map_rows": len(sources)}, indent=2))


if __name__ == "__main__":
    main()

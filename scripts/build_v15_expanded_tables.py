"""Assemble the expanded v15 main and supplementary tables from frozen results."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results/singlecell_v10"
V12 = ROOT / "results/paper_v12"
V14 = ROOT / "results/paper_v14"
OUT = ROOT / "results/paper_v15"
OUT.mkdir(parents=True, exist_ok=True)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def interval(low: float, high: float) -> str:
    return f"{low:.3f} to {high:.3f}"


def main() -> None:
    shutil.copy2(V14 / "main_table1_public_datasets.csv", OUT / "main_table1_public_datasets.csv")
    shutil.copy2(V14 / "main_table2_patient_level_incremental_value.csv", OUT / "main_table2_patient_level_incremental_value.csv")

    strict = {row["dataset"]: row for row in read_json(RES / "strict_multiresolution_benchmarks.json")}
    cell = read_json(RES / "cell_cross_patient_shuffle.json")
    transfer = read_json(RES / "gse173706_neighborhood_transfer_stability.json")
    spatial = read_json(RES / "gse206391_strict_spatial_graph_benchmark.json")
    coordinates = read_json(RES / "gse206391_strict_spatial_coordinate_shuffle.json")

    falsification = []
    for dataset in ("GSE151177", "GSE173706"):
        method = strict[dataset]["methods"]["pseudobulk"]
        falsification.append(
            {
                "dataset": dataset,
                "information_source_broken": "sample/donor labels",
                "tested_model": "strict expression",
                "observed_auroc": method["auroc"],
                "null_mean_auroc": method["permutation_mean"],
                "finite_permutation_p": method["permutation_p_ge_observed"],
                "interpretation": "disease signal removed",
            }
        )
    falsification.extend(
        [
            {
                "dataset": "GSE151177",
                "information_source_broken": "cross-patient cell assignment",
                "tested_model": "global neighborhood",
                "observed_auroc": cell["observed_auroc"],
                "null_mean_auroc": cell["shuffle_mean"],
                "finite_permutation_p": cell["p_ge_observed"],
                "interpretation": "patient-associated cellular structure removed",
            },
            {
                "dataset": "GSE173706",
                "information_source_broken": "paired donor labels",
                "tested_model": "frozen neighborhood transfer",
                "observed_auroc": transfer["observed_auroc"],
                "null_mean_auroc": transfer["permutation_mean"],
                "finite_permutation_p": transfer["permutation_p_ge_observed"],
                "interpretation": "external ranking signal removed",
            },
            {
                "dataset": "GSE206391",
                "information_source_broken": "paired patient labels",
                "tested_model": "strict spatial graph",
                "observed_auroc": spatial["methods"]["spatial_graph"]["auroc"],
                "null_mean_auroc": spatial["methods"]["spatial_graph"]["permutation_mean"],
                "finite_permutation_p": spatial["methods"]["spatial_graph"]["permutation_p_ge_observed"],
                "interpretation": "spatial disease association removed",
            },
            {
                "dataset": "GSE206391",
                "information_source_broken": "within-sample coordinates",
                "tested_model": "strict spatial graph",
                "observed_auroc": coordinates["observed"]["spatial_graph"],
                "null_mean_auroc": coordinates["null"]["spatial_graph"]["mean"],
                "finite_permutation_p": coordinates["null"]["spatial_graph"]["p_ge_observed"],
                "interpretation": "coordinate-specific value not detected",
            },
            {
                "dataset": "GSE206391",
                "information_source_broken": "within-sample coordinates",
                "tested_model": "expression + spatial graph",
                "observed_auroc": coordinates["observed"]["expression_plus_spatial"],
                "null_mean_auroc": coordinates["null"]["expression_plus_spatial"]["mean"],
                "finite_permutation_p": coordinates["null"]["expression_plus_spatial"]["p_ge_observed"],
                "interpretation": "coordinate-specific value not detected",
            },
        ]
    )
    table3 = pd.DataFrame(falsification)
    table3.to_csv(OUT / "main_table3_resolution_specific_falsification.csv", index=False)

    optimism = read_json(RES / "gse151177_neighborhood_optimism_bootstrap.json")["results"]
    calibration = read_json(RES / "gse173706_neighborhood_transfer_calibration.json")
    sensitivity = read_json(RES / "gse206391_sensitivity_matrix_summary.json")
    diagnostics = []
    for row in optimism:
        diagnostics.append(
            {
                "domain": "unsupervised-fit boundary",
                "estimand": f"global − strict AUROC, K={row['clusters']}",
                "estimate": row["optimism_gap_global_minus_strict"],
                "uncertainty_or_range": interval(row["patient_bootstrap_ci95_low"], row["patient_bootstrap_ci95_high"]),
                "analysis_unit": "sample/patient bootstrap",
                "interpretation": "global fitting is optimistic",
            }
        )
    observed = calibration["observed"]
    ci = calibration["bootstrap_ci95"]
    for label, key, ci_key in (
        ("calibration intercept", "calibration_intercept", "intercept"),
        ("calibration slope", "calibration_slope", "slope"),
        ("Brier score", "brier_score", "brier"),
    ):
        diagnostics.append(
            {
                "domain": "external calibration",
                "estimand": label,
                "estimate": observed[key],
                "uncertainty_or_range": interval(ci[ci_key][0], ci[ci_key][1]),
                "analysis_unit": "paired donor bootstrap",
                "interpretation": "unstable" if key != "brier_score" else "poor probability accuracy",
            }
        )
    diagnostics.append(
        {
            "domain": "spatial sensitivity",
            "estimand": "strict AUROC across 144 conditions",
            "estimate": np.nan,
            "uncertainty_or_range": interval(sensitivity["worst_auroc"], sensitivity["best_auroc"]),
            "analysis_unit": "strict patient LODO",
            "interpretation": "range does not establish matched increment",
        }
    )
    table4 = pd.DataFrame(diagnostics)
    table4.to_csv(OUT / "main_table4_diagnostics_and_sensitivity.csv", index=False)

    qc_rows = []
    for dataset, filename, raw_col, kept_col, independent in (
        ("GSE151177", "gse151177_sample_qc.csv", "cells", None, 18),
        ("GSE173706", "gse173706_sample_qc.csv", "raw_cells", "filtered_cells", 11),
        ("GSE206391", "gse206391_sample_metadata.csv", "raw_spots", "filtered_spots", 12),
    ):
        frame = pd.read_csv(RES / filename)
        if dataset == "GSE151177":
            kept = pd.read_csv(RES / "gse151177_sample_metadata.csv")["filtered_cells"].sum()
            raw = frame[raw_col].sum()
            samples = len(pd.read_csv(RES / "gse151177_sample_metadata.csv"))
        else:
            raw = frame[raw_col].sum()
            kept = frame[kept_col].sum()
            samples = len(frame)
        qc_rows.append(
            {
                "dataset": dataset,
                "independent_units": independent,
                "analyzed_samples_or_sections": samples,
                "raw_cells_or_spots": int(raw),
                "retained_cells_or_spots": int(kept),
                "overall_retention": float(kept / raw),
            }
        )
    pd.DataFrame(qc_rows).to_csv(OUT / "supp_table1_high_resolution_qc.csv", index=False)
    shutil.copy2(V12 / "tableS2_strict_resampling.csv", OUT / "supp_table2_strict_resampling.csv")
    shutil.copy2(OUT / "main_table2_patient_level_incremental_value.csv", OUT / "supp_table3_full_incremental_evidence.csv")
    shutil.copy2(OUT / "main_table3_resolution_specific_falsification.csv", OUT / "supp_table4_falsification.csv")

    transfer_table = pd.DataFrame(
        [
            {"metric": "external AUROC", "estimate": transfer["observed_auroc"], "ci_low": transfer["bootstrap_ci95"][0], "ci_high": transfer["bootstrap_ci95"][1], "unit": "donor"},
            {"metric": "calibration intercept", "estimate": observed["calibration_intercept"], "ci_low": ci["intercept"][0], "ci_high": ci["intercept"][1], "unit": "donor"},
            {"metric": "calibration slope", "estimate": observed["calibration_slope"], "ci_low": ci["slope"][0], "ci_high": ci["slope"][1], "unit": "donor"},
            {"metric": "Brier score", "estimate": observed["brier_score"], "ci_low": ci["brier"][0], "ci_high": ci["brier"][1], "unit": "donor"},
        ]
    )
    transfer_table.to_csv(OUT / "supp_table5_external_transfer_calibration.csv", index=False)

    sensitivity_frame = pd.read_csv(RES / "gse206391_strict_spatial_graph_sensitivity_matrix.csv")
    sensitivity_summary = (
        sensitivity_frame.groupby(["hvg", "neighbors"])["auroc"]
        .agg(["min", "mean", "max"]).reset_index()
    )
    sensitivity_summary.to_csv(OUT / "supp_table6_spatial_sensitivity_summary.csv", index=False)
    shutil.copy2(V14 / "claim_registry.csv", OUT / "supp_table7_claim_registry.csv")

    manifest = {
        "main_tables": 4,
        "supplementary_tables": 7,
        "files": sorted(path.name for path in OUT.glob("*table*.csv")),
    }
    (OUT / "expanded_table_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

"""Fail-fast consistency audit for the v12 manuscript package."""
from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures/paper_v12"
RES = ROOT / "results/paper_v12"
SC = ROOT / "results/singlecell_v10"

MAIN_FIGURES = [
    "fig1_study_design",
    "fig2_data_qc",
    "fig3_performance_ceiling",
    "fig4_leakage_audit",
    "fig5_external_transfer",
    "fig6_complexity_failure_chain",
    "fig7_falsification_decision",
]
SUPPLEMENT_FIGURES = [
    "figS1_strict_roc",
    "figS2_incremental_gain_forest",
    "figS3_prediction_distributions",
    "figS4_neighborhood_sensitivity",
    "figS5_transport_diagnostics",
    "figS6_spatial_graph_diagnostics",
    "figS7_bulk_robustness",
]
MAIN_TABLES = [f"table{index}_{name}.csv" for index, name in enumerate(
    ("datasets", "methods_and_leakage", "main_results", "falsification"), start=1
)]
SUPPLEMENT_TABLES = [
    "tableS1_incremental_gain.csv",
    "tableS2_strict_resampling.csv",
    "tableS3_spatial_graph.csv",
    "tableS4_trigger_flux.csv",
]


def load(name):
    return json.loads((SC / name).read_text())


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    missing = []
    for stem in MAIN_FIGURES:
        for extension in ("png", "pdf", "svg"):
            path = FIG / f"{stem}.{extension}"
            if not path.exists() or path.stat().st_size == 0:
                missing.append(str(path))
    for stem in SUPPLEMENT_FIGURES:
        for extension in ("png", "pdf", "svg"):
            path = FIG / "supplement" / f"{stem}.{extension}"
            if not path.exists() or path.stat().st_size == 0:
                missing.append(str(path))
    for filename in MAIN_TABLES + SUPPLEMENT_TABLES:
        path = RES / filename
        if not path.exists() or path.stat().st_size == 0:
            missing.append(str(path))
    require(not missing, f"Missing or empty outputs: {missing}")

    datasets = pd.read_csv(RES / "table1_datasets.csv")
    require(len(datasets) == 11, f"Expected 11 datasets, found {len(datasets)}")
    require(datasets.dataset.nunique() == 11, "Dataset identifiers are not unique")

    main_results = pd.read_csv(RES / "table3_main_results.csv")
    require(
        set(main_results.metric) == {"AUROC"},
        f"Mixed metrics remain in main results: {set(main_results.metric)}",
    )

    strict = load("strict_multiresolution_benchmarks.json")
    require(len(strict) == 3, "Strict benchmark must contain three datasets")
    require(
        all(item["n_permutations"] == 500 for item in strict),
        "Strict benchmark does not use 500 permutations everywhere",
    )

    transfer = load("gse173706_neighborhood_transfer_summary.json")
    require(transfer["paired_donors"] == 11, "Transfer donor count is not 11")
    require(
        transfer["paired_transfer_samples"] == 22,
        "Transfer sample count is not 22",
    )

    spatial = load("gse206391_strict_spatial_graph_benchmark.json")
    require(spatial["n_patients"] == 12, "Spatial patient count is not 12")
    require(spatial["n_samples"] == 48, "Spatial paired sample count is not 48")
    require(
        spatial["n_paired_label_permutations"] == 500,
        "Spatial graph benchmark does not use 500 permutations",
    )

    gain = pd.read_csv(RES / "tableS1_incremental_gain.csv")
    require(len(gain) == 9, f"Expected 9 paired gain comparisons, found {len(gain)}")
    require(
        gain.delta_auroc.max() < 0.05,
        "At least one observed candidate gain reaches the +0.05 threshold",
    )

    report = {
        "status": "pass",
        "datasets": int(len(datasets)),
        "main_figures": len(MAIN_FIGURES),
        "supplementary_figures": len(SUPPLEMENT_FIGURES),
        "figure_files": 3 * (len(MAIN_FIGURES) + len(SUPPLEMENT_FIGURES)),
        "main_tables": len(MAIN_TABLES),
        "supplementary_tables": len(SUPPLEMENT_TABLES),
        "strict_label_permutations_per_dataset": 500,
        "paired_gain_comparisons": int(len(gain)),
        "largest_observed_gain": float(gain.delta_auroc.max()),
    }
    (RES / "package_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

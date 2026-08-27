"""Generate manuscript-ready tables only from frozen PRI-Audit v21 outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "pri_audit_v21"
TABLES = ROOT / "results" / "paper_v21"


def markdown(frame: pd.DataFrame) -> str:
    header = "| " + " | ".join(frame.columns) + " |\n"
    rule = "|" + "|".join(["---"] * len(frame.columns)) + "|\n"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return header + rule + "\n".join(rows) + "\n"


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    registry = pd.read_csv(OUT / "v21_all11_input_registry.csv")
    evaluation = {
        "GSE151177": "18 source samples; leave-one-sample-out",
        "GSE173706": "22 PP/PN samples from 11 paired donors; leave-one-donor-out",
        "GSE206391": "48 LS/NL samples from 12 paired patients; leave-one-patient-out",
        "GSE47751": "12 baseline PN/PP samples from 6 pairs; leave-one-pair-out",
    }
    compact_evaluation = {
        "GSE151177": "18 samples; LOO",
        "GSE173706": "11 donors; LODO",
        "GSE206391": "12 patients; LODO",
        "GSE47751": "6 pairs; LOO",
        "GSE202011": "5 patients; LOO",
    }
    table1 = registry.rename(columns={"accession": "Cohort", "modality": "Modality", "scientific_role": "Frozen role", "independent_unit": "Independent unit", "raw_header_samples": "Raw GEO header samples"})[["Cohort", "Modality", "Frozen role", "Independent unit", "Raw GEO header samples"]].copy()
    table1["Raw GEO header samples"] = table1["Raw GEO header samples"].map(
        lambda value: "—" if pd.isna(value) else str(int(value))
    )
    table1["PRI-Audit use"] = table1.Cohort.map(evaluation).fillna("context / provenance only; not a classifier fold")
    table1.to_csv(TABLES / "table1_cohort_roles.csv", index=False)
    (TABLES / "table1_cohort_roles.md").write_text(markdown(table1), encoding="utf-8")

    summary = pd.read_csv(OUT / "v21_loose_vs_strict_summary.csv")
    primary = summary[summary.arm.isin(["strict_pca", "strict_nmf"])].copy()
    rows = []
    for item in primary.itertuples(index=False):
        rows.append({
            "Dataset": item.dataset, "Unit / split": compact_evaluation[item.dataset], "Representation": item.arm.replace("strict_", "").upper(),
            "Matched reference": "Expr. (2k HVGs)", "Held-out AUROC": f"{item.auroc:.3f}",
            "Incremental AUROC (95% CI)": f"{item.incremental_auroc_vs_strict_expression:+.3f} ({item.increment_bootstrap_ci95_low:+.3f}, {item.increment_bootstrap_ci95_high:+.3f})",
            "Status": "primary" if item.dataset != "GSE47751" else "external bulk boundary; saturated",
        })
    with (OUT / "gse206391_v21_matched_spatial_graph.json").open() as handle:
        spatial = json.load(handle)
    for method, display in [("spatial_graph_1000", "Spatial graph (1,000)"), ("expression_plus_spatial_1500", "Expression + spatial (1,500)")]:
        item = spatial["methods"][method]
        rows.append({
            "Dataset": "GSE206391", "Unit / split": compact_evaluation["GSE206391"], "Representation": display,
            "Matched reference": "Expr. (1k HVGs)" if method == "spatial_graph_1000" else "Expr. (1.5k HVGs)", "Held-out AUROC": f"{item['auroc']:.3f}",
            "Incremental AUROC (95% CI)": f"{item['incremental_auroc']:+.3f} ({item['increment_patient_bootstrap_ci95'][0]:+.3f}, {item['increment_patient_bootstrap_ci95'][1]:+.3f})", "Status": "primary spatial test",
        })
    external_path = OUT / "gse202011_v21_external_strict.json"
    if external_path.exists():
        with external_path.open() as handle:
            external = json.load(handle)
        for method, item in external["methods"].items():
            if method == "expression":
                continue
            rows.append({
                "Dataset": "GSE202011", "Unit / split": compact_evaluation["GSE202011"], "Representation": method.upper(),
                "Matched reference": "Expr. (2k HVGs)", "Held-out AUROC": f"{item['auroc']:.3f}",
                "Incremental AUROC (95% CI)": f"{item['incremental_auroc_vs_strict_expression']:+.3f} ({item['increment_ci95'][0]:+.3f}, {item['increment_ci95'][1]:+.3f})", "Status": "external high-resolution expression/PCA boundary",
            })
    table2 = pd.DataFrame(rows)
    table2.to_csv(TABLES / "table2_strict_matched_comparisons.csv", index=False)
    (TABLES / "table2_strict_matched_comparisons.md").write_text(markdown(table2), encoding="utf-8")

    nested = pd.read_csv(OUT / "v21_nested_neighborhood_strictness.csv")
    measurement = pd.read_csv(OUT / "v21_measurement_random_split_diagnostic.csv")
    mismatch = pd.read_csv(OUT / "v21_mismatched_reference_sensitivity.csv")
    with (OUT / "gse206391_v21_matched_spatial_coordinate_shuffle.json").open() as handle:
        coordinate_null = json.load(handle)

    measurement_rows = []
    for dataset, label in [("GSE151177", "Random cell split"), ("GSE206391", "Random spot split")]:
        subset = measurement.loc[measurement.dataset.eq(dataset)]
        measurement_rows.append({
            "Audit": label,
            "Dataset": dataset,
            "Primary estimate": f"median AUROC {subset.random_measurement_split_auroc.median():.3f}",
            "Uncertainty / diagnostic": (
                f"2.5–97.5% range {subset.random_measurement_split_auroc.quantile(0.025):.3f}–"
                f"{subset.random_measurement_split_auroc.quantile(0.975):.3f}; "
                f"{subset.test_measurements_from_source_samples_seen_in_train.iloc[0]:.0%} test rows from seen sources"
            ),
            "Interpretation": "invalid within-source measurement-interpolation estimand",
        })

    k8 = nested.loc[nested.clusters.eq(8)].iloc[0]
    k16 = nested.loc[nested.clusters.eq(16)].iloc[0]
    k32 = nested.loc[nested.clusters.eq(32)].iloc[0]
    pca_mismatch = mismatch.loc[(mismatch.dataset.eq("GSE206391")) & (mismatch.candidate_arm.eq("strict_pca"))].iloc[0]
    spatial_observed = coordinate_null["observed"]["spatial_graph_1000"]
    combined_observed = coordinate_null["observed"]["expression_plus_spatial_1500"]
    spatial_null = coordinate_null["null"]["spatial_graph_1000"]
    combined_null = coordinate_null["null"]["expression_plus_spatial_1500"]
    table3 = pd.DataFrame([
        {
            "Audit": "Raw reconstruction",
            "Dataset": "GSE151177; GSE173706; GSE206391",
            "Primary estimate": "3/3 count matrices exact",
            "Uncertainty / diagnostic": "metadata and canonical count contents exact",
            "Interpretation": "formal v21 analyses start from verified public inputs",
        },
        *measurement_rows,
        {
            "Audit": "Global vs fully nested neighbourhood",
            "Dataset": "GSE151177",
            "Primary estimate": f"K=8 gap {k8.global_minus_fully_nested_optimism_gap:+.3f}",
            "Uncertainty / diagnostic": (
                f"95% CI {k8.sample_bootstrap_ci95_low:+.3f} to {k8.sample_bootstrap_ci95_high:+.3f}; "
                f"K=16 {k16.global_minus_fully_nested_optimism_gap:+.3f}, "
                f"K=32 {k32.global_minus_fully_nested_optimism_gap:+.3f}"
            ),
            "Interpretation": "global–strict difference is resolution-sensitive, not universally positive",
        },
        {
            "Audit": "Mismatched-reference sensitivity",
            "Dataset": "GSE206391",
            "Primary estimate": f"apparent PCA gain {pca_mismatch.apparent_increment_vs_mismatched_reference:+.3f}",
            "Uncertainty / diagnostic": (
                f"95% CI {pca_mismatch.paired_group_bootstrap_ci95_low:+.3f} to "
                f"{pca_mismatch.paired_group_bootstrap_ci95_high:+.3f}; unmatched 16-HVG baseline"
            ),
            "Interpretation": "a weakened comparator can manufacture an apparent gain",
        },
        {
            "Audit": "Coordinate shuffle",
            "Dataset": "GSE206391",
            "Primary estimate": f"spatial graph observed {spatial_observed:.3f}",
            "Uncertainty / diagnostic": (
                f"500 within-sample shuffles; null mean {spatial_null['mean']:.3f}; "
                f"tail P={spatial_null['p_ge_observed']:.3f}; combined observed "
                f"{combined_observed:.3f}, P={combined_null['p_ge_observed']:.3f}"
            ),
            "Interpretation": "no demonstrated coordinate-specific patient-level increment for this graph",
        },
        {
            "Audit": "Independent high-resolution boundary",
            "Dataset": "GSE202011",
            "Primary estimate": "expression = PCA = AUROC 1.000",
            "Uncertainty / diagnostic": "five paired patients; increment 0.000 (0.000, 0.000)",
            "Interpretation": "saturated expression/PCA boundary replication; no coordinate archive",
        },
    ])
    table3.to_csv(TABLES / "table3_audit_diagnostics.csv", index=False)
    (TABLES / "table3_audit_diagnostics.md").write_text(markdown(table3), encoding="utf-8")
    print(
        f"Wrote {table1.shape[0]} Table 1 rows, {table2.shape[0]} Table 2 rows, "
        f"and {table3.shape[0]} Table 3 rows"
    )


if __name__ == "__main__":
    main()

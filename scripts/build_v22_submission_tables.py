"""Build v22 manuscript and supplement tables from frozen v21/v22 outputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "paper_v21"
V22 = ROOT / "results" / "pri_audit_v22"
sys.path.insert(0, str(ROOT / "scripts"))
import build_v21_manuscript_tables as v21_tables  # noqa: E402


def markdown(frame: pd.DataFrame) -> str:
    header = "| " + " | ".join(frame.columns) + " |\n"
    rule = "|" + "|".join(["---"] * len(frame.columns)) + "|\n"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return header + rule + "\n".join(rows) + "\n"


def topic_rows() -> list[dict[str, str]]:
    with (V22 / "gse206391_v22_spot_topic_benchmark.json").open() as handle:
        result = json.load(handle)
    mapping = [
        ("spot_topic_8", "Fold-local spot topics (8)", "Expr. PCA (8)"),
        ("expression_plus_topic_16", "Expression + spot topics (16)", "Expr. PCA (16)"),
    ]
    rows = []
    for key, representation, reference in mapping:
        item = result["methods"][key]
        low, high = item["increment_patient_bootstrap_ci95"]
        rows.append({
            "Dataset": "GSE206391",
            "Unit / split": "12 patients; LODO",
            "Representation": representation,
            "Matched reference": reference,
            "Held-out AUROC": f"{item['auroc']:.3f}",
            "Incremental AUROC (95% CI)": f"{item['incremental_auroc']:+.3f} ({low:+.3f}, {high:+.3f})",
            "Status": "v22 fold-local high-resolution comparator",
        })
    return rows


def main() -> None:
    v21_tables.main()
    table2_path = TABLES / "table2_strict_matched_comparisons.csv"
    table2 = pd.read_csv(table2_path)
    table2 = pd.concat([table2, pd.DataFrame(topic_rows())], ignore_index=True)
    table2.to_csv(table2_path, index=False)
    (TABLES / "table2_strict_matched_comparisons.md").write_text(markdown(table2), encoding="utf-8")

    simulation = pd.read_csv(V22 / "v22_simulation_summary.csv")
    display = simulation.rename(columns={
        "scenario": "Scenario", "n_patients": "Independent patients", "replicates": "Monte-Carlo replicates",
        "strict_mean_increment": "Mean strict ΔAUROC", "strict_mean_population_increment": "Mean population ΔAUROC",
        "strict_gate_rate": "Strict gate rate", "median_patient_loo_expression_auroc": "Median patient-LOO AUROC",
        "median_random_measurement_split_auroc": "Median random-measurement AUROC",
        "weak_reference_mean_increment": "Mean weak-reference ΔAUROC", "weak_reference_gate_rate": "Weak-reference gate rate",
    })
    display.to_csv(TABLES / "tableS1_v22_known_truth_simulation.csv", index=False)
    (TABLES / "tableS1_v22_known_truth_simulation.md").write_text(markdown(display), encoding="utf-8")
    sensitivity_rows = []
    for spots, suffix in [(150, "_spots150"), (300, ""), (450, "_spots450")]:
        with (V22 / f"gse206391_v22_spot_topic_benchmark{suffix}.json").open() as handle:
            result = json.load(handle)
        for key, label in [("spot_topic_8", "Spot topics (8)"), ("expression_plus_topic_16", "Expression + spot topics (16)")]:
            item = result["methods"][key]
            low, high = item["increment_patient_bootstrap_ci95"]
            sensitivity_rows.append({
                "Spots sampled per sample": spots,
                "Representation": label,
                "Held-out AUROC": f"{item['auroc']:.3f}",
                "Matched ΔAUROC (95% CI)": f"{item['incremental_auroc']:+.3f} ({low:+.3f}, {high:+.3f})",
            })
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(TABLES / "tableS2_v22_spot_topic_sensitivity.csv", index=False)
    (TABLES / "tableS2_v22_spot_topic_sensitivity.md").write_text(markdown(sensitivity), encoding="utf-8")
    print(f"Wrote updated Table 2 ({len(table2)} rows), simulation Supplementary Table S1 ({len(display)} rows), and topic sensitivity Table S2 ({len(sensitivity)} rows).")


if __name__ == "__main__":
    main()

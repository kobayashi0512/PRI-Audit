"""Build supplementary result tables for the v12 manuscript package."""
from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results/singlecell_v10"
OUT = ROOT / "results/paper_v12"


def read_json(name):
    return json.loads((RES / name).read_text())


# Table S1: direct paired uncertainty of algorithmic increments.
gain = pd.read_csv(OUT / "incremental_gain_bootstrap.csv")
gain.to_csv(OUT / "tableS1_incremental_gain.csv", index=False)


# Table S2: strict outer-fold scores and refitted permutation tests.
strict_rows = []
for dataset in read_json("strict_multiresolution_benchmarks.json"):
    for method, result in dataset["methods"].items():
        strict_rows.append(
            {
                "dataset": dataset["dataset"],
                "method": method,
                "n_samples": dataset["n_samples"],
                "n_groups": dataset["n_groups"],
                "auroc": result["auroc"],
                "bootstrap_ci95_low": result["group_bootstrap_ci95"][0],
                "bootstrap_ci95_high": result["group_bootstrap_ci95"][1],
                "permutation_mean": result["permutation_mean"],
                "permutation_p": result["permutation_p_ge_observed"],
                "n_permutations": dataset["n_permutations"],
            }
        )
strict = pd.DataFrame(strict_rows)
strict.to_csv(OUT / "tableS2_strict_resampling.csv", index=False)


# Table S3: matched-gene spatial graph comparison.
spatial = read_json("gse206391_strict_spatial_graph_benchmark.json")
spatial_rows = []
for method, result in spatial["methods"].items():
    spatial_rows.append(
        {
            "method": method,
            "auroc": result["auroc"],
            "balanced_accuracy": result["balanced_accuracy"],
            "bootstrap_ci95_low": result["group_bootstrap_ci95"][0],
            "bootstrap_ci95_high": result["group_bootstrap_ci95"][1],
            "permutation_mean": result["permutation_mean"],
            "permutation_p": result["permutation_p_ge_observed"],
            "delta_vs_expression": result.get("delta_vs_expression", 0.0),
            "delta_ci95_low": result.get("delta_bootstrap_ci95", [0.0, 0.0])[0],
            "delta_ci95_high": result.get("delta_bootstrap_ci95", [0.0, 0.0])[1],
            "probability_delta_ge_0.05": result.get(
                "bootstrap_probability_delta_ge_0.05", 0.0
            ),
        }
    )
pd.DataFrame(spatial_rows).to_csv(OUT / "tableS3_spatial_graph.csv", index=False)


# Table S4: trigger-flux effect estimates and uncertainty.
flux = pd.DataFrame(read_json("triad_flux_effects.json"))
flux["bootstrap_ci95_low"] = flux.bootstrap_ci95.str[0]
flux["bootstrap_ci95_high"] = flux.bootstrap_ci95.str[1]
flux[
    [
        "axis",
        "effect",
        "bootstrap_ci95_low",
        "bootstrap_ci95_high",
        "permutation_p_abs",
    ]
].to_csv(OUT / "tableS4_trigger_flux.csv", index=False)


sections = [
    ("Table S1. Paired incremental-gain bootstrap", "tableS1_incremental_gain.csv"),
    ("Table S2. Strict outer-fold resampling", "tableS2_strict_resampling.csv"),
    ("Table S3. Matched spatial-graph comparison", "tableS3_spatial_graph.csv"),
    ("Table S4. Trigger-flux estimates", "tableS4_trigger_flux.csv"),
]
markdown = ["# v12 补充表", ""]
for title, filename in sections:
    markdown.extend(
        [f"## {title}", "", pd.read_csv(OUT / filename).to_markdown(index=False), ""]
    )
(OUT / "supplementary_tables.md").write_text("\n".join(markdown), encoding="utf-8")
print(f"wrote {len(sections)} supplementary tables to {OUT}")

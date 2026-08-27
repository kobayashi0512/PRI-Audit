"""Build the remaining figures, tables and provenance files from frozen outputs.

This script never fits or re-selects a model.  It only summarizes artifacts that
were locked before the external and treatment data were analyzed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
V3 = RESULTS / "ccmac_v3"
FIG = ROOT / "figures" / "paper"
FIG.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def study_design() -> None:
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6)
    ax.axis("off")
    stages = [
        ("Mechanism discovery", "GSE24767, GSE86451, GSE36287\nkeratinocyte perturbations", 0.25, "#dcecf7"),
        ("Development / lock", "GSE121212, GSE182740, GSE57225\n3-cohort LOCO; no external read", 3.4, "#e6f2df"),
        ("External gate", "GSE223799 included\nGSE261704: 8% coverage\nGSE153007: non-lesional", 6.8, "#fff0d8"),
        ("Treatment direction", "GSE226244: group-level IL-17A\nGSE85034: paired ADA / MTX", 10.1, "#f2e3f2"),
    ]
    for title, body, x, color in stages:
        ax.text(x + 1.25, 3.45, title, ha="center", va="center", fontsize=12, weight="bold")
        ax.text(x + 1.25, 2.1, body, ha="center", va="center", fontsize=9.5,
                bbox={"boxstyle": "round,pad=0.65", "facecolor": color, "edgecolor": "#555555"})
    for x in [2.85, 6.25, 9.55]:
        ax.annotate("", xy=(x + 0.45, 2.1), xytext=(x - 0.15, 2.1),
                    arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "#555555"})
    ax.text(6.5, 5.25, "Pre-specified separation of discovery, model locking, blind transport, and treatment-state analyses",
            ha="center", fontsize=14, weight="bold")
    ax.text(6.5, 0.55, "Final claims: treatment-associated direction and recovery tracking; no superiority claim over simple diagnostic scores.",
            ha="center", fontsize=10.5, color="#7a1f1f")
    fig.savefig(FIG / "fig1_study_design_and_boundaries.png", dpi=220, bbox_inches="tight")


def mechanism_boundary() -> None:
    left = pd.read_csv(RESULTS / "external_v1" / "GSE24767_synergy_residual.csv", index_col=0).iloc[:, 0]
    right = pd.read_csv(RESULTS / "external_v1" / "GSE86451_synergy_residual.csv", index_col=0).iloc[:, 0]
    joined = pd.concat([left.rename("discovery"), right.rename("replication")], axis=1).dropna()
    boot = pd.read_csv(RESULTS / "external_v1" / "synergy_replication_bootstrap.csv")
    rho = joined.corr(method="spearman").iloc[0, 1]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), constrained_layout=True)
    sns.regplot(data=joined, x="discovery", y="replication", scatter_kws={"s": 9, "alpha": 0.22},
                line_kws={"color": "#b24c63"}, ax=axes[0])
    axes[0].axhline(0, color="0.75", lw=0.8)
    axes[0].axvline(0, color="0.75", lw=0.8)
    axes[0].set_title(f"Synergy-residual effects: weak cross-experiment concordance\nSpearman $\\rho$ = {rho:.2f}")
    axes[0].set_xlabel("GSE24767 residual effect")
    axes[0].set_ylabel("GSE86451 residual effect")
    sns.histplot(boot.cross_experiment_residual_rho, bins=28, color="#7655a4", ax=axes[1])
    axes[1].axvline(boot.cross_experiment_residual_rho.median(), color="#b24c63", lw=2,
                    label=f"median = {boot.cross_experiment_residual_rho.median():.2f}")
    axes[1].set_title("Bootstrap concordance of residual signature")
    axes[1].set_xlabel("cross-experiment Spearman $\\rho$")
    axes[1].set_ylabel("bootstrap replicates")
    axes[1].legend(frameon=False)
    fig.suptitle("The residual was retained as a pre-specified axis but not interpreted as a stable synergy program", y=1.03)
    fig.savefig(FIG / "fig2_mechanism_residual_boundary.png", dpi=220, bbox_inches="tight")


def write_tables() -> None:
    grid = pd.read_csv(V3 / "development_loco_grid.csv")
    best = grid.loc[grid.mean_auc.idxmax()]
    external = json.loads((V3 / "GSE223799_locked_metrics.json").read_text())
    g226 = json.loads((V3 / "GSE226244_v5_group_metrics.json").read_text())
    g850 = json.loads((V3 / "GSE85034_v5_metrics.json").read_text())
    controls = json.loads((V3 / "GSE85034_final_fixed_score_controls.json").read_text())

    model_rows = [{
        "analysis": "Development selection (3-cohort LOCO)",
        "unit": "independent cohort holdout",
        "score_or_model": "locked mechanism ridge",
        "n": 120,
        "estimate": f"mean AUROC {best.mean_auc:.3f}",
        "interval_or_uncertainty": f"SD across held-out cohorts {best.sd_auc:.3f}",
        "boundary": "gamma=1, lambda=1; anchor penalty did not improve selection",
    }]
    for scale in ["Macro", "Micro"]:
        d = external[scale]
        model_rows.append({
            "analysis": f"GSE223799 blind transport ({scale})",
            "unit": "5 PSO and 5 AD samples",
            "score_or_model": "mechanism ridge / v0 NNLS / NOS2-CCL27",
            "n": 10,
            "estimate": "AUROC 1.000 / 1.000 / 1.000",
            "interval_or_uncertainty": "all bootstrap CIs 1.000-1.000",
            "boundary": "diagnostic comparison saturated; no complex-score gain",
        })
    model_rows.extend([
        {"analysis": "GSE261704 gate", "unit": "NanoString sample", "score_or_model": "locked mechanism ridge",
         "n": "not analyzed", "estimate": "8.0% feature coverage", "interval_or_uncertainty": "NOS2/CCL27 unavailable",
         "boundary": "excluded under pre-specified platform gate"},
        {"analysis": "GSE153007 audit", "unit": "skin sample", "score_or_model": "external transport",
         "n": "not analyzed", "estimate": "all deposited samples non-lesional", "interval_or_uncertainty": "not a lesion comparison",
         "boundary": "excluded before model evaluation"},
    ])
    pd.DataFrame(model_rows).to_csv(RESULTS / "paper_table_2_model_and_transport_results.csv", index=False)

    treatment_rows = [{
        "dataset": "GSE226244", "contrast": "on-treatment week 12 minus baseline", "unit": "sample (unpaired)",
        "n": "33 baseline; 18 on-treatment", "mean_delta": g226["on12_minus_baseline"],
        "ci_95": f"[{g226['bootstrap_ci'][0]:.3f}, {g226['bootstrap_ci'][1]:.3f}]",
        "direction_probability": g226["bootstrap_prob_lt_zero"], "two_sided_p": g226["mann_whitney_two_sided_p"],
        "interpretation_boundary": "group-level direction only; not patient-paired or clinical response prediction"}]
    for d in g850["week16_minus_lesional_baseline"]["by_treatment"] + [g850["week16_minus_lesional_baseline"]["pooled"]]:
        treatment_rows.append({
            "dataset": "GSE85034", "contrast": f"week 16 minus lesional baseline ({d.get('treatment', 'pooled')})",
            "unit": "paired patient", "n": d["n_complete"], "mean_delta": d["mean"],
            "ci_95": f"[{d['ci_low']:.3f}, {d['ci_high']:.3f}]", "direction_probability": d["prob_lt_zero"],
            "two_sided_p": d["wilcoxon_two_sided_p"],
            "interpretation_boundary": "treatment-associated transcriptomic direction; not causal efficacy estimation"})
    pd.DataFrame(treatment_rows).to_csv(RESULTS / "paper_table_3_treatment_direction_results.csv", index=False)

    control_rows = []
    for score, value in controls["scores"].items():
        control_rows.append({"fixed_score": score, **value["week16_distance_to_own_nl_change"],
                             "interpretation_boundary": "post-hoc descriptive control; raw scales are not comparable"})
    pd.DataFrame(control_rows).to_csv(RESULTS / "paper_supplementary_table_nonlesional_controls.csv", index=False)


def write_manifest() -> None:
    paths = [
        ROOT / "docs" / "final_analysis_boundaries.md",
        ROOT / "docs" / "v3_preregistered_plan.md",
        ROOT / "docs" / "v5_treatment_direction_preregistered_plan.md",
        ROOT / "results" / "paper_dataset_registry.csv",
        V3 / "locked_model.npz", V3 / "locked_model_manifest.json",
        V3 / "development_loco_grid.csv", V3 / "GSE223799_locked_metrics.json",
        V3 / "GSE226244_v5_group_metrics.json", V3 / "GSE85034_v5_metrics.json",
        V3 / "GSE85034_final_fixed_score_controls.json",
        ROOT / "scripts" / "run_ccmac_v3.py", ROOT / "scripts" / "evaluate_gse223799_v3.py",
        ROOT / "scripts" / "evaluate_gse226244_v5.py", ROOT / "scripts" / "evaluate_gse85034_v5.py",
        ROOT / "scripts" / "final_gse85034_controls.py", ROOT / "scripts" / "plot_paper_figures.py",
        ROOT / "scripts" / "plot_v5_summary.py", ROOT / "scripts" / "build_manuscript_docx.py", Path(__file__),
    ]
    manifest = {"scope": "final paper artifacts; no model refitting", "files": []}
    for path in paths:
        manifest["files"].append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path)})
    (RESULTS / "paper_reproducibility_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    study_design()
    mechanism_boundary()
    write_tables()
    write_manifest()

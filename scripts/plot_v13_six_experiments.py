"""Publication-style plots for the six v13 completion analyses."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/singlecell_v10"
FIG = ROOT / "figures/paper_v13"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":8, "axes.spines.top":False, "axes.spines.right":False})


def save(fig, stem):
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.png", dpi=450, bbox_inches="tight")
    plt.close(fig)


def main():
    # A: patient-bootstrap optimism gaps.
    d = pd.read_csv(OUT / "gse151177_neighborhood_optimism_bootstrap.csv")
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    for j, row in d.iterrows():
        ax.plot([row.patient_bootstrap_ci95_low, row.patient_bootstrap_ci95_high], [j, j], color="#0072B2", lw=2)
        ax.scatter(row.optimism_gap_global_minus_strict, j, color="#D55E00", s=28, zorder=3)
    ax.axvline(0, color="0.4", ls="--", lw=.8); ax.set_yticks(range(len(d)), [f"K={int(x)}" for x in d.clusters]); ax.set_xlabel("global − strict AUROC"); ax.set_title("Patient bootstrap optimism gap")
    save(fig, "figED_v13_optimism_gap_bootstrap")

    # B: complete null distributions (cell, label and coordinate invalidations).
    series = []
    paths = [
        (OUT / "cell_cross_patient_shuffle_null_distribution.npz", "cell shuffle", "observed"),
        (OUT / "spatial_coordinate_shuffle_null_distribution.npz", "legacy coordinate shuffle", "observed"),
        (OUT / "gse206391_strict_spatial_graph_resampling_distributions.npz", "strict label shuffle", "observed_label"),
        (OUT / "gse206391_strict_spatial_coordinate_shuffle_distributions.npz", "strict spatial coordinate shuffle", "observed_spatial"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(10.2, 2.8), sharey=False)
    for ax, (path, label, observed_key) in zip(axes, paths):
        z = np.load(path, allow_pickle=True)
        if label == "strict label shuffle":
            null = z["label_permutation_spatial_graph"]
            obs = float(json.loads((OUT / "gse206391_strict_spatial_graph_benchmark.json").read_text())["methods"]["spatial_graph"]["auroc"])
        else:
            null = z["null_aurocs"] if "null_aurocs" in z else z["null_spatial"]
            obs = float(z[observed_key][0])
        ax.hist(null, bins=25, color="#56B4E9", alpha=.85, edgecolor="white"); ax.axvline(obs, color="#D55E00", lw=1.5); ax.set_title(label, fontsize=8); ax.set_xlabel("AUROC")
        ax.text(.03, .95, f"n={len(null)}", transform=ax.transAxes, va="top", fontsize=7)
    axes[0].set_ylabel("null count"); fig.suptitle("Complete permutation / coordinate null distributions", y=1.03); save(fig, "figED_v13_complete_null_distributions")

    # C: external calibration reliability curve plus distribution-free summary.
    cal = json.loads((OUT / "gse173706_neighborhood_transfer_calibration.json").read_text())
    rows = pd.read_csv(OUT / "gse173706_neighborhood_transfer_calibration_rows.csv")
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.8, 2.8), gridspec_kw={"width_ratios":[1.35, 1]})
    ax.plot([0,1], [0,1], ls="--", color="0.4", lw=.8, label="ideal")
    ax.plot(cal["observed"]["reliability_predicted"], cal["observed"]["reliability_observed"], "o-", color="#0072B2", label="external transfer")
    ax.set(xlim=(0,1), ylim=(0,1), xlabel="mean predicted probability", ylabel="observed frequency", title="Reliability curve"); ax.legend(frameon=False, fontsize=7)
    bx.bar([0,1,2], [cal["observed"]["calibration_intercept"], cal["observed"]["calibration_slope"], cal["observed"]["brier_score"]], color=["#E69F00", "#009E73", "#CC79A7"])
    bx.axhline(0, color="0.5", lw=.8); bx.set_xticks([0,1,2], ["intercept", "slope", "Brier"], rotation=30, ha="right"); bx.set_title("Calibration summary")
    save(fig, "figED_v13_external_calibration")

    # D: full sensitivity matrices.
    s = pd.read_csv(OUT / "gse206391_strict_spatial_graph_sensitivity_matrix.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    mat = s[s.C == 1].groupby(["hvg", "neighbors"]).auroc.mean().unstack().sort_index()
    im = axes[0].imshow(mat.to_numpy(), aspect="auto", cmap="viridis", vmin=s.auroc.min(), vmax=s.auroc.max()); axes[0].set_xticks(range(len(mat.columns)), mat.columns); axes[0].set_yticks(range(len(mat.index)), mat.index); axes[0].set_xlabel("neighbors"); axes[0].set_ylabel("HVGs"); axes[0].set_title("HVG × neighbors (C=1)")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]): axes[0].text(j, i, f"{mat.iloc[i,j]:.2f}", ha="center", va="center", color="white", fontsize=6)
    c = s[(s.hvg == 500) & (s.neighbors == 6)].pivot(index="C", columns="seed", values="auroc").sort_index(); axes[1].imshow(c.to_numpy(), aspect="auto", cmap="viridis", vmin=s.auroc.min(), vmax=s.auroc.max()); axes[1].set_xticks(range(len(c.columns)), c.columns); axes[1].set_yticks(range(len(c.index)), c.index); axes[1].set_xlabel("seed"); axes[1].set_ylabel("C"); axes[1].set_title("C × seed (HVG=500, k=6)")
    fig.colorbar(im, ax=axes, shrink=.75, label="AUROC"); save(fig, "figED_v13_sensitivity_matrices")
    print(json.dumps({"figure_dir": str(FIG), "figures": sorted(x.name for x in FIG.glob("figED_v13_*"))}, indent=2))


if __name__ == "__main__":
    main()

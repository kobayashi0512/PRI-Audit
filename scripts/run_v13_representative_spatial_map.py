"""Predefined, non-cherry-picked LS/NL spatial map selection.

Selection rule: compute each paired patient's expression+spatial probability
difference (LS minus NL), then select the pair closest to the across-patient
median.  The map uses a fixed inflammatory marker panel declared in code.
"""
from pathlib import Path
import json
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/singlecell_v10"
H5 = ROOT / "data/raw/GSE206391_Preprocessed_data.h5"
MARKERS = ("S100A7", "S100A7A", "KRT16", "IL36G", "CXCL8", "SERPINB4")


def decode(values):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in values])


def load():
    with h5py.File(H5, "r") as f:
        X = csr_matrix((f["/X/data"][...], f["/X/indices"][...], f["/X/indptr"][...]), shape=(len(f["/obs/_index"]), 16685))
        obs = {k: f["/obs/" + k][...] for k in ("patient", "biopsy_type", "sample", "array_row", "array_col", "in_tissue")}
        cats = {k: decode(f["/obs/__categories/" + k][...]) for k in ("patient", "biopsy_type", "sample")}
        mt = f["/raw/var/mt"][...].astype(bool)
        genes = decode(f["/var/gene_name"][...])
    cat = lambda k: cats[k][obs[k].astype(int)]
    meta = pd.DataFrame({"patient": cat("patient"), "condition": cat("biopsy_type"), "sample": cat("sample"), "row": obs["array_row"], "col": obs["array_col"], "in_tissue": obs["in_tissue"]})
    total = np.asarray(X.sum(axis=1)).ravel()
    detected = np.asarray((X > 0).sum(axis=1)).ravel()
    mito = np.asarray(X[:, mt].sum(axis=1)).ravel() / np.maximum(total, 1)
    keep = (total >= 500) & (detected >= 300) & (mito <= .2) & meta.in_tissue.astype(bool).to_numpy()
    meta["keep"] = keep
    return X, meta, genes, total


def main():
    pred = pd.read_csv(OUT / "gse206391_strict_spatial_graph_predictions.csv")
    # Strictly paired LS/NL patient differences; no visual inspection enters selection.
    pred["delta"] = pred["expression_plus_spatial_probability"] * (pred["true"] * 2 - 1)
    paired = pred.groupby("outer_group").delta.sum().rename("paired_delta").to_frame()
    # Convert signed sum to LS minus NL probability and choose closest to median.
    median_delta = float(paired.paired_delta.median())
    selected_patient = str((paired.paired_delta - median_delta).abs().sort_values().index[0])
    selected = pred[pred.outer_group.astype(str) == selected_patient].copy()
    # If a patient has technical replicate sections, use the section with the
    # largest pre-existing filtered-spot count per condition (never the map
    # score) so the display is a single LS/NL pair.
    selected = selected.sort_values("filtered_spots", ascending=False).groupby("condition", sort=True).head(1).sort_values("true")
    X, meta, genes, total = load()
    marker_idx = [int(np.flatnonzero(genes == g)[0]) for g in MARKERS if np.any(genes == g)]
    available = [MARKERS[i] for i, g in enumerate(MARKERS) if np.any(genes == g)]
    maps = []
    for _, row in selected.iterrows():
        ix = meta.index[(meta["sample"] == row["sample"]) & meta["keep"]].to_numpy()
        counts = X[ix][:, marker_idx].toarray().astype(float)
        norm = np.log1p(counts / np.maximum(total[ix, None], 1) * 1e4)
        score = norm.mean(axis=1)
        maps.append({"sample": row["sample"], "condition": row["condition"], "row": meta.loc[ix, "row"].to_numpy(), "col": meta.loc[ix, "col"].to_numpy(), "score": score})
    vmin = min(float(np.quantile(m["score"], .01)) for m in maps); vmax = max(float(np.quantile(m["score"], .99)) for m in maps)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    fig, axes = plt.subplots(1, len(maps), figsize=(7.2, 3.0), constrained_layout=True)
    if len(maps) == 1: axes = [axes]
    for ax, item in zip(axes, maps):
        sc = ax.scatter(item["col"], -item["row"], c=item["score"], s=2, cmap="magma", vmin=vmin, vmax=vmax, linewidths=0, rasterized=True)
        ax.set_title(f"{item['condition']}\n{item['sample']}", fontsize=8); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    cbar = fig.colorbar(sc, ax=axes, shrink=.75, pad=.02); cbar.set_label("predefined inflammatory module score")
    fig.savefig(OUT / "gse206391_representative_LS_NL_spatial_map.pdf")
    fig.savefig(OUT / "gse206391_representative_LS_NL_spatial_map.svg")
    fig.savefig(OUT / "gse206391_representative_LS_NL_spatial_map.png", dpi=450)
    plt.close(fig)
    result = {"dataset": "GSE206391", "selection_rule": "paired expression+spatial probability difference closest to across-patient median", "median_paired_delta": median_delta, "selected_patient": selected_patient, "selected_samples": [str(x["sample"]) for x in maps], "marker_panel_declared_before_selection": list(MARKERS), "marker_panel_available": available, "n_spots": [int(len(x["score"])) for x in maps], "figure": "gse206391_representative_LS_NL_spatial_map.pdf"}
    (OUT / "gse206391_representative_spatial_selection.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

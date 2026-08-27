"""Small, leakage-safe pilot for currently available GSE151177 10x samples.

This is a feasibility run, not a confirmatory patient-level result: the current
files are all psoriasis samples and therefore cannot test LS/NL classification.
"""
from pathlib import Path
import gzip
import json
import numpy as np
import pandas as pd
from scipy.io import mmread
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "raw" / "GSE151177_samples"
OUT = ROOT / "results" / "singlecell_v10"
OUT.mkdir(parents=True, exist_ok=True)

def load_sample(mtx):
    stem = mtx.name.replace("_matrix.mtx.gz", "")
    with gzip.open(mtx, "rb") as fh:
        mat = mmread(fh).tocsr().astype(np.float32)  # genes x cells
    with gzip.open(SRC / f"{stem}_features.tsv.gz", "rt") as fh:
        genes = [line.rstrip("\n").split("\t")[0] for line in fh]
    totals = np.asarray(mat.sum(axis=0)).ravel()
    detected = np.asarray((mat > 0).sum(axis=0)).ravel()
    mito = np.array([g.upper().startswith("MT-") for g in genes])
    mt_frac = np.asarray(mat[mito].sum(axis=0)).ravel() / np.maximum(totals, 1)
    # Feasibility filter; thresholds are deliberately recorded and not tuned.
    keep = (detected >= 300) & (mt_frac <= 0.20) & (totals >= 500)
    return stem, mat[:, keep], genes, totals[keep], detected[keep], mt_frac[keep], int(keep.sum())

def main():
    rows, sample_mats = [], []
    qc_path = OUT / "gse151177_sample_qc.csv"
    qc = pd.read_csv(qc_path) if qc_path.exists() else pd.DataFrame()
    for mtx in sorted(SRC.glob("*_matrix.mtx.gz")):
        stem, mat, genes, totals, detected, mt_frac, n_keep = load_sample(mtx)
        # Exclude failed samples by pre-registered cell-level QC rules. A
        # sample must retain cells and have reasonable post-filter summaries.
        raw = qc[qc["sample"] == stem]
        raw_bad = (not raw.empty and (float(raw.iloc[0]["median_mito_fraction"]) > 0.20 or
                                      float(raw.iloc[0]["median_detected_genes"]) < 300))
        if n_keep == 0 or raw_bad:
            continue
        sample_mats.append((stem, mat, totals, detected, mt_frac))
        rows.append({"sample": stem, "raw_cells": int(mat.shape[1]),
                     "filtered_cells": n_keep,
                     "median_counts": float(np.median(totals)),
                     "median_detected_genes": float(np.median(detected)),
                     "median_mito_fraction": float(np.median(mt_frac))})
    if not sample_mats:
        raise RuntimeError("No usable samples")
    # Pseudo-bulk sample x gene log-CPM; keeps patient/sample as analysis unit.
    bulk = []
    names = []
    for stem, mat, *_ in sample_mats:
        x = np.asarray(mat.sum(axis=1)).ravel()
        bulk.append(np.log1p(x / max(x.sum(), 1) * 1e6))
        names.append(stem)
    X = np.vstack(bulk)
    var = X.var(axis=0)
    top = np.argsort(var)[-min(2000, X.shape[1]):]
    Z = StandardScaler().fit_transform(X[:, top])
    n_pca = min(3, Z.shape[0], Z.shape[1])
    pca = PCA(n_components=n_pca, random_state=0).fit_transform(Z)
    n_nmf = min(3, X.shape[0], X.shape[1])
    nmf = NMF(n_components=n_nmf, init="nndsvda", random_state=0, max_iter=500).fit_transform(X[:, top] - X[:, top].min() + 1e-6)
    pd.DataFrame(rows).to_csv(OUT / "gse151177_pilot_cell_qc.csv", index=False)
    pd.DataFrame(pca, index=names, columns=[f"PC{i+1}" for i in range(n_pca)]).to_csv(OUT / "gse151177_pilot_pca.csv")
    pd.DataFrame(nmf, index=names, columns=[f"NMF{i+1}" for i in range(n_nmf)]).to_csv(OUT / "gse151177_pilot_nmf.csv")
    all_names = [p.name.replace("_matrix.mtx.gz", "") for p in sorted(SRC.glob("*_matrix.mtx.gz"))]
    summary = {"samples_used": names, "samples_excluded": sorted(set(all_names) - set(names)),
               "genes_total": len(genes), "genes_used": int(len(top)),
               "total_filtered_cells": int(sum(r["filtered_cells"] for r in rows)),
               "note": "Pilot only; no LS/NL labels and no confirmatory performance claim."}
    (OUT / "gse151177_pilot_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()

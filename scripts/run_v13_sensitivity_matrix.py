"""Predefined strict spatial-graph sensitivity matrix.

Crosses HVG count, neighbor count, regularization and three classifier seeds.
The seed is retained in the matrix even though the liblinear solver is nearly
deterministic; this makes seed invariance an explicit, auditable result.
"""
from pathlib import Path
import itertools
import json
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/singlecell_v10"
sys.path.insert(0, str(ROOT / "scripts"))
import run_strict_spatial_graph_benchmark as base  # noqa: E402


def graph(values, neighbors):
    if len(values) < 2 or neighbors.shape[1] == 0:
        return np.zeros(values.shape[1] * 2)
    nm = values[neighbors].mean(axis=1); c = values - values.mean(0, keepdims=True); nc = c[neighbors].mean(axis=1); v = np.mean(c*c, 0)
    moran = np.divide(np.mean(c*nc, 0), v, out=np.zeros_like(v), where=v > 1e-10)
    contrast = np.mean(np.abs(values-nm), 0)
    return np.hstack([moran, contrast])


def main():
    matrix, spot_meta, total = base.load_spots()
    sample_meta, log_cpm, y, groups = base.patient_subset()
    sample_groups = base.prepare_spot_groups(spot_meta, sample_meta)
    hvg_grid, k_grid, c_grid, seed_grid = (250, 500, 1000, 2000), (4, 6, 8, 12), (.1, 1., 10.), (0, 1, 2)
    rows = []
    cache = {}
    for hvg, k in itertools.product(hvg_grid, k_grid):
        fold_cache = []
        for held in np.unique(groups):
            train = groups != held
            top = np.argsort(log_cpm[train].var(0))[-hvg:]
            spatial_rows = []
            for sample in sample_meta["sample"]:
                ix, _ = sample_groups[str(sample)]
                vals = np.log1p(matrix[ix][:, top].toarray().astype(float) / np.maximum(total[ix, None], 1) * 1e4)
                coords = spot_meta.loc[ix, ["row", "col"]].to_numpy()
                nb = NearestNeighbors(n_neighbors=min(k + 1, max(len(ix), 1))).fit(coords).kneighbors(coords, return_distance=False)[:, 1:] if len(ix) > 1 else np.zeros((len(ix), 0), int)
                spatial_rows.append(graph(vals, nb))
            fold_cache.append((np.flatnonzero(train), np.flatnonzero(~train), np.vstack(spatial_rows)))
        cache[(hvg, k)] = fold_cache
        for C, seed in itertools.product(c_grid, seed_grid):
            probs = np.full(len(y), np.nan)
            for train, test, features in fold_cache:
                model = make_pipeline(StandardScaler(), LogisticRegression(C=C, solver="liblinear", class_weight="balanced", max_iter=2000, random_state=seed))
                model.fit(features[train], y[train]); probs[test] = model.predict_proba(features[test])[:, 1]
            rows.append({"method": "strict_spatial_graph", "hvg": hvg, "neighbors": k, "C": C, "seed": seed, "auroc": float(roc_auc_score(y, probs))})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "gse206391_strict_spatial_graph_sensitivity_matrix.csv", index=False)
    pivot = out[out.C == 1].groupby(["hvg", "neighbors"]).auroc.mean().unstack()
    pivot.to_csv(OUT / "gse206391_sensitivity_hvg_neighbors_C1.csv")
    seed_pivot = out[(out.hvg == 500) & (out.neighbors == 6)].pivot(index="C", columns="seed", values="auroc")
    seed_pivot.to_csv(OUT / "gse206391_sensitivity_C_seed_hvg500_k6.csv")
    result = {"grid": {"hvg": list(hvg_grid), "neighbors": list(k_grid), "C": list(c_grid), "seed": list(seed_grid)}, "n_rows": int(len(out)), "best_auroc": float(out.auroc.max()), "worst_auroc": float(out.auroc.min()), "output": "gse206391_strict_spatial_graph_sensitivity_matrix.csv"}
    (OUT / "gse206391_sensitivity_matrix_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

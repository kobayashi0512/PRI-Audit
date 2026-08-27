"""Patient-level optimism gap: global versus strict neighborhood AUROC.

The global representation is intentionally fitted once on all cells.  The
strict representation refits the cell clusters inside every held-out sample
fold.  Patient/sample bootstrap is performed on the paired prediction rows,
so the gap CI cannot be obtained by subtracting independent intervals.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/singlecell_v10"
SEED = 20260810


def hist(km, z, start, stop):
    return np.bincount(km.predict(z[start:stop]), minlength=km.n_clusters) / max(stop - start, 1)


def main(n_boot=5000):
    d = np.load(OUT / "gse151177_patient_balanced_neighborhood_v1.npz", allow_pickle=True)
    z, offsets, samples = d["pca"], d["offsets"], d["samples"]
    meta = pd.read_csv(OUT / "gse151177_sample_metadata.csv").set_index("sample").loc[samples]
    y = (meta.condition == "psoriasis").astype(int).to_numpy()
    rows = []
    rng = np.random.default_rng(SEED)
    for k in (8, 16, 32):
        # Global fit: this is the intentionally optimistic comparator.
        km_global = MiniBatchKMeans(n_clusters=k, random_state=0, batch_size=2048, n_init=10).fit(z)
        h_global = np.vstack([hist(km_global, z, offsets[i], offsets[i + 1]) for i in range(len(y))])
        global_p = np.full(len(y), np.nan)
        # Strict patient/sample LODO representation and prediction.
        strict_p = np.full(len(y), np.nan)
        for i in range(len(y)):
            train = np.arange(len(y)) != i
            train_cells = np.vstack([z[offsets[j]:offsets[j + 1]] for j in np.flatnonzero(train)])
            km = MiniBatchKMeans(n_clusters=k, random_state=0, batch_size=2048, n_init=10).fit(train_cells)
            h_train = np.vstack([hist(km, z, offsets[j], offsets[j + 1]) for j in np.flatnonzero(train)])
            h_test = hist(km, z, offsets[i], offsets[i + 1])[None, :]
            model = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000, random_state=0))
            model.fit(h_train, y[train])
            strict_p[i] = model.predict_proba(h_test)[:, 1][0]
        model_global = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000, random_state=0))
        # For the global comparator the classifier is still evaluated LOO; only
        # the cell representation is leaked.  This matches the legacy ablation.
        for i in range(len(y)):
            train = np.arange(len(y)) != i
            model_global.fit(h_global[train], y[train])
            global_p[i] = model_global.predict_proba(h_global[i:i + 1])[:, 1][0]
        global_auc = float(roc_auc_score(y, global_p))
        strict_auc = float(roc_auc_score(y, strict_p))
        gaps = []
        for _ in range(n_boot):
            ix = rng.integers(0, len(y), size=len(y))
            if np.unique(y[ix]).size < 2:
                continue
            gaps.append(roc_auc_score(y[ix], global_p[ix]) - roc_auc_score(y[ix], strict_p[ix]))
        gap = np.asarray(gaps, dtype=float)
        rows.append({
            "clusters": k,
            "global_auroc": global_auc,
            "strict_auroc": strict_auc,
            "optimism_gap_global_minus_strict": global_auc - strict_auc,
            "patient_bootstrap_n": int(gap.size),
            "patient_bootstrap_ci95_low": float(np.quantile(gap, .025)),
            "patient_bootstrap_ci95_high": float(np.quantile(gap, .975)),
            "patient_bootstrap_probability_gap_ge_0_05": float(np.mean(gap >= .05)),
        })
        np.savez_compressed(
            OUT / f"gse151177_neighborhood_optimism_k{k}_predictions.npz",
            samples=samples, labels=y, global_probabilities=global_p,
            strict_probabilities=strict_p, bootstrap_gaps=gap,
        )
    result = {"dataset": "GSE151177", "resampling_unit": "sample/patient", "seed": SEED, "n_boot": n_boot, "results": rows}
    (OUT / "gse151177_neighborhood_optimism_bootstrap.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUT / "gse151177_neighborhood_optimism_bootstrap.csv", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

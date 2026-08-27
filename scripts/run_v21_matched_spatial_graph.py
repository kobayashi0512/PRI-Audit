"""Fair-dimension strict spatial-graph audit on raw GSE206391 spots.

The historical spatial benchmark compared a 1,000-feature graph summary and a
1,500-feature expression-plus-graph representation with a 500-feature
expression reference.  v21 corrects that mismatch.  Spatial-only is evaluated
against a strict 1,000-HVG expression control; expression-plus-spatial is
evaluated against a strict 1,500-HVG expression control.  All HVG selection,
graph construction and model fitting are outer-patient-fold specific.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_pri_audit_v21 as v21  # noqa: E402
import run_strict_spatial_graph_benchmark as legacy  # noqa: E402


OUT = v21.OUT
SEED = 20260822
METHODS = ("expression_1000", "spatial_graph_1000", "expression_1500", "expression_plus_spatial_1500")


def patient_subset() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    metadata = pd.read_csv(OUT / "gse206391_sample_metadata.csv")
    counts = np.load(OUT / "gse206391_sample_gene_counts.npz")["counts"].astype(float)
    eligible = metadata[metadata.condition.isin(["NON LESIONAL", "LESIONAL"])]
    paired = eligible.groupby("patient").condition.nunique()
    patients = paired[paired == 2].index
    keep = (metadata.patient.isin(patients) & metadata.condition.isin(["NON LESIONAL", "LESIONAL"])).to_numpy()
    metadata = metadata.loc[keep].reset_index(drop=True)
    counts = counts[keep]
    log_cpm = np.log1p(counts / np.maximum(counts.sum(axis=1, keepdims=True), 1) * 1e6)
    labels = (metadata.condition == "LESIONAL").astype(int).to_numpy()
    groups = metadata.patient.astype(str).to_numpy()
    return metadata, log_cpm, labels, groups


def fit_predict(train: np.ndarray, labels: np.ndarray, test: np.ndarray) -> np.ndarray:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1, solver="liblinear", class_weight="balanced", max_iter=2000, random_state=0),
    )
    model.fit(train, labels)
    return model.predict_proba(test)[:, 1]


def build_folds(matrix, spot_meta, total, sample_meta, log_cpm, groups):
    spot_groups = legacy.prepare_spot_groups(spot_meta, sample_meta)
    folds = []
    for held in np.unique(groups):
        test = groups == held
        train = ~test
        ranking = np.argsort(log_cpm[train].var(axis=0))
        top_500 = ranking[-500:]
        top_1000 = ranking[-1000:]
        top_1500 = ranking[-1500:]
        moran, contrast = [], []
        for sample in sample_meta["sample"]:
            indices, neighbors = spot_groups[str(sample)]
            current_moran, current_contrast = legacy.graph_features(matrix, total, indices, neighbors, top_500)
            moran.append(current_moran)
            contrast.append(current_contrast)
        spatial = np.hstack([np.vstack(moran), np.vstack(contrast)])
        expression_1000 = log_cpm[:, top_1000]
        expression_1500 = log_cpm[:, top_1500]
        combined = np.hstack([log_cpm[:, top_500], spatial])
        folds.append({
            "held_patient": held,
            "train_ix": np.flatnonzero(train), "test_ix": np.flatnonzero(test),
            "representations": {
                "expression_1000": (expression_1000[train], expression_1000[test]),
                "spatial_graph_1000": (spatial[train], spatial[test]),
                "expression_1500": (expression_1500[train], expression_1500[test]),
                "expression_plus_spatial_1500": (combined[train], combined[test]),
            },
        })
    return folds


def predictions(folds, labels, method):
    probability = np.full(labels.size, np.nan)
    for fold in folds:
        train, test = fold["train_ix"], fold["test_ix"]
        x_train, x_test = fold["representations"][method]
        probability[test] = fit_predict(x_train, labels[train], x_test)
    if np.isnan(probability).any():
        raise RuntimeError(f"Missing spatial prediction: {method}")
    return probability


def resample(y, probabilities, groups, rng, n_boot):
    unique = np.unique(groups)
    by_group = {group: np.flatnonzero(groups == group) for group in unique}
    scores = {method: [] for method in METHODS}
    deltas = {"spatial_graph_1000_vs_expression_1000": [], "expression_plus_spatial_1500_vs_expression_1500": []}
    for _ in range(n_boot):
        selected = rng.choice(unique, size=unique.size, replace=True)
        ix = np.concatenate([by_group[group] for group in selected])
        if np.unique(y[ix]).size != 2:
            continue
        current = {method: roc_auc_score(y[ix], probability[ix]) for method, probability in probabilities.items()}
        for method, value in current.items():
            scores[method].append(value)
        deltas["spatial_graph_1000_vs_expression_1000"].append(current["spatial_graph_1000"] - current["expression_1000"])
        deltas["expression_plus_spatial_1500_vs_expression_1500"].append(current["expression_plus_spatial_1500"] - current["expression_1500"])
    return {key: np.asarray(value) for key, value in scores.items()}, {key: np.asarray(value) for key, value in deltas.items()}


def paired_permutations(folds, y, groups, rng, n_perm):
    values = {method: [] for method in METHODS}
    for _ in range(n_perm):
        shuffled = y.copy()
        for group in np.unique(groups):
            if rng.random() < .5:
                ix = groups == group
                shuffled[ix] = 1 - shuffled[ix]
        for method in METHODS:
            values[method].append(float(roc_auc_score(shuffled, predictions(folds, shuffled, method))))
    return {key: np.asarray(value) for key, value in values.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()
    legacy.OUT = OUT
    matrix, spot_meta, total = legacy.load_spots()
    sample_meta, log_cpm, y, groups = patient_subset()
    folds = build_folds(matrix, spot_meta, total, sample_meta, log_cpm, groups)
    observed = {method: predictions(folds, y, method) for method in METHODS}
    rng = np.random.default_rng(SEED)
    boot_scores, boot_deltas = resample(y, observed, groups, rng, args.bootstrap)
    null = paired_permutations(folds, y, groups, rng, args.permutations)
    result = {
        "dataset": "GSE206391", "n_samples": int(y.size), "n_patients": int(np.unique(groups).size),
        "outer_split": "paired-patient leave-one-patient-out", "hvg_selection": "within outer patient fold",
        "spatial_graph": "Moran-like neighbour agreement plus local contrast for 500 HVGs (1,000 features)",
        "matched_references": {"spatial_graph_1000": "expression_1000", "expression_plus_spatial_1500": "expression_1500"},
        "methods": {},
    }
    comparisons = {
        "spatial_graph_1000": "spatial_graph_1000_vs_expression_1000",
        "expression_plus_spatial_1500": "expression_plus_spatial_1500_vs_expression_1500",
    }
    for method, probability in observed.items():
        entry = {
            "auroc": float(roc_auc_score(y, probability)),
            "balanced_accuracy": float(balanced_accuracy_score(y, probability >= .5)),
            "patient_bootstrap_ci95": [float(value) for value in np.quantile(boot_scores[method], [.025, .975])],
            "label_permutation_mean": float(null[method].mean()),
            "label_permutation_ci95": [float(value) for value in np.quantile(null[method], [.025, .975])],
            "label_permutation_p_ge_observed": float((1 + np.sum(null[method] >= roc_auc_score(y, probability))) / (null[method].size + 1)),
        }
        if method in comparisons:
            delta = boot_deltas[comparisons[method]]
            reference = result["matched_references"][method]
            entry.update({
                "matched_reference": reference,
                "incremental_auroc": float(roc_auc_score(y, probability) - roc_auc_score(y, observed[reference])),
                "increment_patient_bootstrap_ci95": [float(value) for value in np.quantile(delta, [.025, .975])],
                "increment_bootstrap_probability_ge_0_05": float(np.mean(delta >= .05)),
            })
        result["methods"][method] = entry
    rows = sample_meta.copy()
    rows["true"] = y
    rows["outer_patient"] = groups
    for method, probability in observed.items():
        rows[f"{method}_probability"] = probability
    rows.to_csv(OUT / "gse206391_v21_matched_spatial_graph_predictions.csv", index=False)
    np.savez_compressed(
        OUT / "gse206391_v21_matched_spatial_graph_distributions.npz",
        **{f"bootstrap_{method}": value for method, value in boot_scores.items()},
        **{f"bootstrap_{name}": value for name, value in boot_deltas.items()},
        **{f"label_permutation_{method}": value for method, value in null.items()},
        seed=np.array([SEED]),
    )
    (OUT / "gse206391_v21_matched_spatial_graph.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

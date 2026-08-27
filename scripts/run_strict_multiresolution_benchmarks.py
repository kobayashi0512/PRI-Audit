"""Leakage-safe patient-level expression and low-rank benchmarks.

Every unsupervised operation (variance filtering, scaling, PCA, and NMF) is fit
inside the outer sample/patient fold. Label permutations respect the paired
design for GSE173706 and GSE206391.
"""
from pathlib import Path
import argparse
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/singlecell_v10"
RNG = np.random.default_rng(20260810)
METHODS = ("pseudobulk", "pca", "nmf")


def load_dataset(stem):
    data = np.load(OUT / f"{stem}_sample_gene_counts.npz")
    counts = data["counts"].astype(np.float64)
    meta = pd.read_csv(OUT / f"{stem}_sample_metadata.csv")

    if stem == "gse151177":
        keep = meta.condition.isin(["control", "psoriasis"]).to_numpy()
        y = (meta.loc[keep, "condition"] == "psoriasis").astype(int).to_numpy()
        groups = meta.loc[keep, "sample"].astype(str).to_numpy()
        paired = False
    elif stem == "gse173706":
        eligible = meta[meta.condition.isin(["peripheral_normal", "psoriatic"])]
        paired_donors = eligible.groupby("donor").condition.nunique()
        paired_donors = paired_donors[paired_donors == 2].index
        keep = meta.donor.isin(paired_donors) & meta.condition.isin(
            ["peripheral_normal", "psoriatic"]
        )
        keep = keep.to_numpy()
        y = (meta.loc[keep, "condition"] == "psoriatic").astype(int).to_numpy()
        groups = meta.loc[keep, "donor"].astype(str).to_numpy()
        paired = True
    elif stem == "gse206391":
        eligible = meta[meta.condition.isin(["NON LESIONAL", "LESIONAL"])]
        paired_patients = eligible.groupby("patient").condition.nunique()
        paired_patients = paired_patients[paired_patients == 2].index
        keep = meta.patient.isin(paired_patients) & meta.condition.isin(
            ["NON LESIONAL", "LESIONAL"]
        )
        keep = keep.to_numpy()
        y = (meta.loc[keep, "condition"] == "LESIONAL").astype(int).to_numpy()
        groups = meta.loc[keep, "patient"].astype(str).to_numpy()
        paired = True
    else:
        raise ValueError(stem)

    counts = counts[keep]
    meta = meta.loc[keep].reset_index(drop=True)
    log_cpm = np.log1p(
        counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0) * 1e6
    )
    return log_cpm, y, groups, meta, paired


def fold_representations(x_train, x_test, n_hvg=2000, n_components=8):
    top = np.argsort(x_train.var(axis=0))[-min(n_hvg, x_train.shape[1]) :]
    train = x_train[:, top]
    test = x_test[:, top]

    output = {"pseudobulk": (train, test)}

    gene_scaler = StandardScaler()
    train_scaled = gene_scaler.fit_transform(train)
    test_scaled = gene_scaler.transform(test)
    n_comp = min(n_components, train_scaled.shape[0] - 1, train_scaled.shape[1])
    pca = PCA(n_components=n_comp, random_state=0)
    output["pca"] = (pca.fit_transform(train_scaled), pca.transform(test_scaled))

    # log-CPM is nonnegative, so no global offset or test-derived scaling is needed.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        nmf = NMF(
            n_components=n_comp,
            init="nndsvda",
            random_state=0,
            max_iter=2000,
        )
        output["nmf"] = (nmf.fit_transform(train), nmf.transform(test))
    return output


def prepare_folds(x, y, groups):
    folds = []
    for held_group in np.unique(groups):
        test = groups == held_group
        train = ~test
        reps = fold_representations(x[train], x[test])
        folds.append(
            {
                "held_group": held_group,
                "train_idx": np.flatnonzero(train),
                "test_idx": np.flatnonzero(test),
                "representations": reps,
            }
        )
    return folds


def predict_with_labels(folds, labels, method):
    probabilities = np.full(len(labels), np.nan, dtype=float)
    for fold in folds:
        train_idx = fold["train_idx"]
        test_idx = fold["test_idx"]
        y_train = labels[train_idx]
        if np.unique(y_train).size < 2:
            continue
        z_train, z_test = fold["representations"][method]
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1,
                solver="liblinear",
                class_weight="balanced",
                max_iter=2000,
                random_state=0,
            ),
        )
        model.fit(z_train, y_train)
        probabilities[test_idx] = model.predict_proba(z_test)[:, 1]
    return probabilities


def group_bootstrap_ci(y, probability, groups, n_boot=2000):
    unique_groups = np.unique(groups)
    values = []
    by_group = {group: np.flatnonzero(groups == group) for group in unique_groups}
    for _ in range(n_boot):
        selected = RNG.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([by_group[group] for group in selected])
        if np.unique(y[idx]).size == 2:
            values.append(roc_auc_score(y[idx], probability[idx]))
    return [float(v) for v in np.quantile(values, [0.025, 0.975])]


def permuted_labels(y, groups, paired):
    if not paired:
        return RNG.permutation(y)
    shuffled = y.copy()
    for group in np.unique(groups):
        if RNG.random() < 0.5:
            idx = groups == group
            shuffled[idx] = 1 - shuffled[idx]
    return shuffled


def analyze(stem, n_perm=500):
    x, y, groups, meta, paired = load_dataset(stem)
    folds = prepare_folds(x, y, groups)
    result = {
        "dataset": stem.upper(),
        "n_samples": int(len(y)),
        "n_groups": int(np.unique(groups).size),
        "paired_label_permutation": bool(paired),
        "n_permutations": int(n_perm),
        "methods": {},
    }
    predictions = meta.copy()
    predictions["true"] = y
    predictions["outer_group"] = groups

    observed = {}
    for method in METHODS:
        probability = predict_with_labels(folds, y, method)
        if np.isnan(probability).any():
            raise RuntimeError(f"Missing {method} predictions for {stem}")
        auroc = float(roc_auc_score(y, probability))
        observed[method] = (probability, auroc)
        predictions[f"{method}_probability"] = probability
        result["methods"][method] = {
            "auroc": auroc,
            "balanced_accuracy": float(
                balanced_accuracy_score(y, probability >= 0.5)
            ),
            "group_bootstrap_ci95": group_bootstrap_ci(y, probability, groups),
        }

    null = {method: [] for method in METHODS}
    for _ in range(n_perm):
        shuffled = permuted_labels(y, groups, paired)
        for method in METHODS:
            probability = predict_with_labels(folds, shuffled, method)
            null[method].append(float(roc_auc_score(shuffled, probability)))

    for method in METHODS:
        values = np.asarray(null[method])
        auroc = observed[method][1]
        result["methods"][method]["permutation_mean"] = float(values.mean())
        result["methods"][method]["permutation_ci95"] = [
            float(v) for v in np.quantile(values, [0.025, 0.975])
        ]
        result["methods"][method]["permutation_p_ge_observed"] = float(
            (1 + np.sum(values >= auroc)) / (n_perm + 1)
        )

    predictions.to_csv(OUT / f"{stem}_strict_benchmark_predictions.csv", index=False)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--permutations", type=int, default=500)
    args = parser.parse_args()
    output = [
        analyze(stem, n_perm=args.permutations)
        for stem in ("gse151177", "gse173706", "gse206391")
    ]
    path = OUT / "strict_multiresolution_benchmarks.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

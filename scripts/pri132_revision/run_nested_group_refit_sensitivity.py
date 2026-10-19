"""Full-pipeline grouped-resampling sensitivity for the core GSE206391 comparisons.

This is intentionally separate from the manuscript's original *conditional*
group bootstrap, which resampled saved outer-fold predictions.  Here every
bootstrap draw selects paired test patients with replacement and, for each
selected patient, re-fits HVG selection, scaling, PCA/NMF and the classifier
on a bootstrap sample of the remaining patients.  It therefore propagates
training-pipeline instability as well as test-group sampling variation.

It is a sensitivity analysis, not a replacement for a larger independent
cohort: every prediction remains leave-one-patient-out with respect to the
patient it scores.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cohorts import gse206391  # noqa: E402

OUT = ROOT / "results" / "pri_1_30_revision"
SEED = 20261017


def fit_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, method: str) -> np.ndarray:
    """Recreate the registered fold-local reference/candidate branches."""
    top = np.argsort(train_x.var(axis=0))[-min(2000, train_x.shape[1]):]
    x_train = train_x[:, top]
    x_test = test_x[:, top]
    if method == "expression":
        train_rep, test_rep = x_train, x_test
    elif method == "pca":
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(x_train)
        test_scaled = scaler.transform(x_test)
        n_components = min(8, train_scaled.shape[0] - 1, train_scaled.shape[1])
        model = PCA(n_components=n_components, random_state=0)
        train_rep, test_rep = model.fit_transform(train_scaled), model.transform(test_scaled)
    elif method == "nmf":
        n_components = min(8, x_train.shape[0] - 1, x_train.shape[1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model = NMF(n_components=n_components, init="nndsvda", random_state=0, max_iter=2000)
            train_rep, test_rep = model.fit_transform(x_train), model.transform(x_test)
    else:
        raise ValueError(method)
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1, solver="liblinear", class_weight="balanced", max_iter=2000, random_state=0),
    )
    classifier.fit(train_rep, train_y)
    return classifier.predict_proba(test_rep)[:, 1]


def paired_conditional_delta(y: np.ndarray, groups: np.ndarray, candidate: np.ndarray, reference: np.ndarray, draws: int, rng: np.random.Generator) -> np.ndarray:
    """The published saved-prediction grouped bootstrap, reconstructed for comparison."""
    patients = np.unique(groups)
    index = {patient: np.flatnonzero(groups == patient) for patient in patients}
    delta = []
    for _ in range(draws):
        chosen = rng.choice(patients, size=len(patients), replace=True)
        ix = np.concatenate([index[p] for p in chosen])
        delta.append(roc_auc_score(y[ix], candidate[ix]) - roc_auc_score(y[ix], reference[ix]))
    return np.asarray(delta)


def nested_refit_delta(x: np.ndarray, y: np.ndarray, groups: np.ndarray, method: str, draws: int, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap held-out patients and refit all learned stages for every draw."""
    patients = np.unique(groups)
    by_patient = {patient: np.flatnonzero(groups == patient) for patient in patients}
    values = []
    for draw in range(draws):
        selected_test = rng.choice(patients, size=len(patients), replace=True)
        test_y, reference_scores, candidate_scores = [], [], []
        for patient in selected_test:
            test_ix = by_patient[patient]
            train_patients = patients[patients != patient]
            # Re-sample training patients only; the held-out patient is never fitted.
            selected_train = rng.choice(train_patients, size=len(train_patients), replace=True)
            train_ix = np.concatenate([by_patient[p] for p in selected_train])
            reference_scores.extend(fit_predict(x[train_ix], y[train_ix], x[test_ix], "expression"))
            candidate_scores.extend(fit_predict(x[train_ix], y[train_ix], x[test_ix], method))
            test_y.extend(y[test_ix])
        value = roc_auc_score(test_y, candidate_scores) - roc_auc_score(test_y, reference_scores)
        values.append(value)
        if (draw + 1) % 25 == 0:
            print(f"{method}: completed {draw + 1}/{draws} draws", flush=True)
    return np.asarray(values)


def summarize(values: np.ndarray) -> dict[str, float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return {
        "median_delta_auroc": float(np.median(values)),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "proportion_positive": float(np.mean(values > 0)),
        "proportion_at_or_above_0_05": float(np.mean(values >= 0.05)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=400)
    args = parser.parse_args()
    if args.draws < 100:
        raise ValueError("At least 100 draws are required for this sensitivity analysis")
    OUT.mkdir(parents=True, exist_ok=True)
    x, y, groups, metadata = gse206391()
    original = pd.read_csv(ROOT / "results" / "pri_audit_v21" / "GSE206391_strict_loose_predictions.csv")
    if not np.array_equal(original["patient"].astype(str).to_numpy(), groups.astype(str)):
        raise RuntimeError("Stored predictions no longer align with the reconstructed cohort")
    rows, report = [], {"dataset": "GSE206391", "n_patients": int(len(np.unique(groups))), "n_samples": int(len(y)), "seed": SEED, "draws": args.draws, "design": "Each bootstrap draw resamples paired held-out patients and, for each selected patient, resamples the remaining patients for training. HVG selection, scaling, representation fitting, and the classifier are refit in that training resample."}
    for index, method in enumerate(["pca", "nmf"]):
        conditional = paired_conditional_delta(
            y, groups,
            original[f"strict_{method}_probability"].to_numpy(),
            original["strict_expression_probability"].to_numpy(),
            args.draws, np.random.default_rng(SEED + index),
        )
        nested = nested_refit_delta(x, y, groups, method, args.draws, np.random.default_rng(SEED + 100 + index))
        pd.DataFrame({"draw": np.arange(args.draws), "conditional_saved_prediction_delta": conditional, "nested_full_refit_delta": nested}).to_csv(OUT / f"gse206391_{method}_nested_group_refit_draws.csv", index=False)
        report[method] = {
            "conditional_saved_prediction_bootstrap": summarize(conditional),
            "nested_full_refit_group_bootstrap": summarize(nested),
        }
        rows.append({"comparison": f"{method.upper()} vs matched 2,000-HVG expression", "conditional_95pct_interval": "({:.3f}, {:.3f})".format(*np.quantile(conditional, [0.025, 0.975])), "full_refit_95pct_interval": "({:.3f}, {:.3f})".format(*np.quantile(nested, [0.025, 0.975])), "full_refit_median_delta": float(np.median(nested)), "full_refit_pr_delta_gt_0": float(np.mean(nested > 0)), "full_refit_pr_delta_ge_0_05": float(np.mean(nested >= 0.05))})
    pd.DataFrame(rows).to_csv(OUT / "gse206391_full_refit_sensitivity_summary.csv", index=False)
    (OUT / "gse206391_full_refit_sensitivity.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

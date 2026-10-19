"""Reference sensitivity with candidates fixed across comparator budgets.

The earlier exploratory sweep changed the HVG input for both the candidate and
the expression reference.  That design could not distinguish an effect of the
reference from an effect of changing the candidate itself.  This script fixes
the candidate at a 2,000-HVG input in every outer fold and changes only the
expression-reference HVG budget.

The analysis is a diagnostic.  The prespecified 2,000-HVG matched comparison
remains the primary estimate.
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

OUT = ROOT / "results" / "pri_1_31_reference_sensitivity"
SEED = 20261017
REF_H = (16, 50, 100, 200, 500, 1000, 2000)
CANDIDATES = ("pca", "nmf")
CANDIDATE_H = 2000
N_COMPONENTS = 8


def _top_hvgs(x: np.ndarray, h: int) -> np.ndarray:
    return np.argsort(x.var(axis=0))[::-1][: min(h, x.shape[1])]


def fit_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, method: str, h: int) -> np.ndarray:
    """Fit only on train_x and return a held-out probability vector."""
    selected = _top_hvgs(train_x, h)
    x_train, x_test = train_x[:, selected], test_x[:, selected]
    if method == "expression":
        train_rep, test_rep = x_train, x_test
    elif method == "pca":
        scaler = StandardScaler().fit(x_train)
        train_scaled, test_scaled = scaler.transform(x_train), scaler.transform(x_test)
        n_components = min(N_COMPONENTS, train_scaled.shape[0] - 1, train_scaled.shape[1])
        model = PCA(n_components=n_components, random_state=0)
        train_rep, test_rep = model.fit_transform(train_scaled), model.transform(test_scaled)
    elif method == "nmf":
        n_components = min(N_COMPONENTS, x_train.shape[0] - 1, x_train.shape[1])
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


def outer_oof(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict[str, np.ndarray]:
    """Create fixed-candidate and variable-reference scores on identical outer folds."""
    patients = np.unique(groups)
    scores = {f"reference_{h}": np.full(len(y), np.nan) for h in REF_H}
    scores.update({f"candidate_{m}": np.full(len(y), np.nan) for m in CANDIDATES})
    for patient in patients:
        test = np.flatnonzero(groups == patient)
        train = np.flatnonzero(groups != patient)
        for h in REF_H:
            scores[f"reference_{h}"][test] = fit_predict(x[train], y[train], x[test], "expression", h)
        for method in CANDIDATES:
            scores[f"candidate_{method}"][test] = fit_predict(x[train], y[train], x[test], method, CANDIDATE_H)
    if any(np.isnan(v).any() for v in scores.values()):
        raise RuntimeError("Incomplete out-of-fold predictions")
    return scores


def patient_bootstrap_delta(y: np.ndarray, groups: np.ndarray, candidate: np.ndarray, reference: np.ndarray, draws: int, rng: np.random.Generator) -> np.ndarray:
    patients = np.unique(groups)
    by_patient = {patient: np.flatnonzero(groups == patient) for patient in patients}
    values = []
    for _ in range(draws):
        selected = rng.choice(patients, size=len(patients), replace=True)
        index = np.concatenate([by_patient[p] for p in selected])
        values.append(roc_auc_score(y[index], candidate[index]) - roc_auc_score(y[index], reference[index]))
    return np.asarray(values)


def nested_full_refit_delta(x: np.ndarray, y: np.ndarray, groups: np.ndarray, method: str, reference_h: int, draws: int, rng: np.random.Generator) -> np.ndarray:
    """Patient-level resampling; candidate remains at 2,000 HVGs in every draw."""
    patients = np.unique(groups)
    by_patient = {patient: np.flatnonzero(groups == patient) for patient in patients}
    values = []
    for draw in range(draws):
        held_out = rng.choice(patients, size=len(patients), replace=True)
        labels, cand_scores, ref_scores = [], [], []
        for patient in held_out:
            test = by_patient[patient]
            eligible_train = patients[patients != patient]
            selected_train = rng.choice(eligible_train, size=len(eligible_train), replace=True)
            train = np.concatenate([by_patient[p] for p in selected_train])
            cand_scores.extend(fit_predict(x[train], y[train], x[test], method, CANDIDATE_H))
            ref_scores.extend(fit_predict(x[train], y[train], x[test], "expression", reference_h))
            labels.extend(y[test])
        values.append(roc_auc_score(labels, cand_scores) - roc_auc_score(labels, ref_scores))
        if (draw + 1) % 25 == 0:
            print(f"{method}, h={reference_h}: {draw + 1}/{draws}", flush=True)
    return np.asarray(values)


def interval(values: np.ndarray) -> tuple[float, float]:
    return tuple(float(v) for v in np.quantile(values, [0.025, 0.975]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--full-refit-draws", type=int, default=400)
    args = parser.parse_args()
    if args.bootstrap_draws < 1000 or args.full_refit_draws < 100:
        raise ValueError("Use at least 1,000 conditional and 100 full-refit draws")
    OUT.mkdir(parents=True, exist_ok=True)
    x, y, groups, _ = gse206391()
    scores = outer_oof(x, y, groups)
    pd.DataFrame({"patient": groups, "label": y, **scores}).to_csv(OUT / "gse206391_fixed_candidate_oof_predictions.csv", index=False)

    rows = []
    summary: dict[str, object] = {
        "dataset": "GSE206391", "n_patients": int(len(np.unique(groups))), "n_samples": int(len(y)),
        "candidate_hvg_budget_fixed": CANDIDATE_H, "candidate_components": N_COMPONENTS,
        "reference_hvg_budgets": list(REF_H), "seed": SEED,
        "interpretation": "Exploratory comparator-sensitivity analysis; 2,000-HVG matched comparison remains primary.",
    }
    for method_index, method in enumerate(CANDIDATES):
        candidate = scores[f"candidate_{method}"]
        candidate_auc = float(roc_auc_score(y, candidate))
        method_rows = []
        for reference_h in REF_H:
            reference = scores[f"reference_{reference_h}"]
            reference_auc = float(roc_auc_score(y, reference))
            draws = patient_bootstrap_delta(y, groups, candidate, reference, args.bootstrap_draws, np.random.default_rng(SEED + method_index * 100 + reference_h))
            lo, hi = interval(draws)
            row = {
                "candidate": method.upper() + "-8 fixed at 2,000 HVGs",
                "reference_hvgs": reference_h,
                "candidate_auroc": candidate_auc,
                "reference_auroc": reference_auc,
                "delta_auroc": candidate_auc - reference_auc,
                "conditional_ci95_low": lo,
                "conditional_ci95_high": hi,
                "bootstrap_draws": args.bootstrap_draws,
                "passes_primary_gate": bool(reference_h == 2000 and candidate_auc - reference_auc >= 0.05 and lo > 0),
            }
            rows.append(row)
            method_rows.append(row)
        summary[method] = method_rows

    # Full pipeline refits establish sensitivity at the deliberately weak and matched endpoints.
    full_rows = []
    for method_index, method in enumerate(CANDIDATES):
        for reference_h in (16, 2000):
            values = nested_full_refit_delta(x, y, groups, method, reference_h, args.full_refit_draws, np.random.default_rng(SEED + 1000 + method_index * 100 + reference_h))
            lo, hi = interval(values)
            pd.DataFrame({"draw": np.arange(len(values)), "delta_auroc": values}).to_csv(OUT / f"gse206391_{method}_fixed_candidate_reference_{reference_h}_full_refit_draws.csv", index=False)
            full_rows.append({"candidate": method.upper() + "-8 fixed at 2,000 HVGs", "reference_hvgs": reference_h, "median_delta_auroc": float(np.median(values)), "full_refit_ci95_low": lo, "full_refit_ci95_high": hi, "draws": len(values)})
    pd.DataFrame(rows).to_csv(OUT / "gse206391_fixed_candidate_reference_sensitivity.csv", index=False)
    pd.DataFrame(full_rows).to_csv(OUT / "gse206391_fixed_candidate_reference_full_refit.csv", index=False)
    summary["full_refit_endpoints"] = full_rows
    (OUT / "gse206391_fixed_candidate_reference_sensitivity.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nFull refit endpoints\n", pd.DataFrame(full_rows).to_string(index=False))


if __name__ == "__main__":
    main()

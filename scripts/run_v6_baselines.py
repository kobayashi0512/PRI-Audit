"""Nested leave-one-drug-out baselines for the frozen v6 recovery-order task."""

from __future__ import annotations

import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "motifr_v6"
K_GRID, C_GRID, L1_GRID = (250, 500, 1000, 2000), (0.01, 0.1, 1.0, 10.0), (0.1, 0.5, 0.9)


def load() -> tuple[pd.DataFrame, np.ndarray]:
    meta = pd.read_csv(OUT / "v6_paired_change_metadata.csv")
    matrix = pd.read_parquet(OUT / "v6_paired_change_rank_matrix.parquet").to_numpy(dtype=np.float32)
    meta["row"] = np.arange(len(meta))
    # Each treatment is a domain. Keep placebo separate as a negative-control domain.
    meta["context"] = np.where(meta.drug.eq("Placebo"), "Placebo", meta.drug)
    return meta, matrix


def pair_indices(meta: pd.DataFrame, contexts: list[str]) -> list[tuple[int, int, str]]:
    result = []
    sub = meta.loc[meta.context.isin(contexts)]
    for (context, subject), rows in sub.groupby(["context", "subject"]):
        early = rows.loc[rows.phase.eq("early"), "row"].tolist()
        late = rows.loc[rows.phase.eq("late"), "row"].tolist()
        if len(early) == 1 and len(late) == 1:
            result.append((early[0], late[0], context))
    return result


def select_features(x: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(x.var(axis=0))[-min(k, x.shape[1]):]


def prepare(x: np.ndarray, early: np.ndarray, late: np.ndarray, k: int):
    idx = select_features(x[np.r_[early, late]], k)
    scaler = StandardScaler().fit(x[np.r_[early, late]][:, idx])
    return idx, scaler


def mean_direction_score(x_train, early, late, x_test, k):
    idx, scaler = prepare(x_train, early, late, k)
    direction = scaler.transform(x_train[late][:, idx]).mean(axis=0) - scaler.transform(x_train[early][:, idx]).mean(axis=0)
    return scaler.transform(x_test[:, idx]) @ direction


def pca_score(x_train, early, late, x_test, k):
    idx, scaler = prepare(x_train, early, late, k)
    pca = PCA(n_components=1, random_state=20260730).fit(scaler.transform(x_train[np.r_[early, late]][:, idx]))
    score_train = pca.transform(scaler.transform(x_train[np.r_[early, late]][:, idx])).ravel()
    sign = 1 if score_train[len(early):].mean() > score_train[:len(early)].mean() else -1
    return sign * pca.transform(scaler.transform(x_test[:, idx])).ravel()


def logistic_score(x_train, early, late, x_test, k, c, elastic_l1=None):
    idx, scaler = prepare(x_train, early, late, k)
    train_idx = np.r_[early, late]
    y = np.r_[np.zeros(len(early)), np.ones(len(late))]
    if elastic_l1 is None:
        model = LogisticRegression(C=c, penalty="l2", solver="liblinear", max_iter=5000, random_state=20260730)
    else:
        # Small-n / high-p folds make the default saga stopping criterion needlessly
        # expensive here. A fixed, documented tolerance keeps every nested fold on
        # the same numerical standard and permits checkpointed execution.
        model = LogisticRegression(C=c, penalty="elasticnet", l1_ratio=elastic_l1, solver="saga", max_iter=1000, tol=1e-3, random_state=20260730)
    model.fit(scaler.transform(x_train[train_idx][:, idx]), y)
    return model.decision_function(scaler.transform(x_test[:, idx]))


def order_accuracy(scores: np.ndarray, pairs: list[tuple[int, int, str]], row_to_local: dict[int, int]) -> float:
    valid = [(row_to_local[e], row_to_local[l]) for e, l, _ in pairs if e in row_to_local and l in row_to_local]
    return float(np.mean([scores[l] > scores[e] for e, l in valid])) if valid else np.nan


def inner_select(method: str, x, meta, train_contexts: list[str], candidate_start: int = 0, candidate_end: int | None = None):
    all_candidates = [(k,) for k in K_GRID] if method in {"mean_direction", "pca"} else (
        [(k, c) for k in K_GRID for c in C_GRID] if method == "ridge" else [(k, c, l1) for k in K_GRID for c in C_GRID for l1 in L1_GRID]
    )
    candidates = all_candidates[candidate_start:candidate_end]
    if not candidates:
        raise ValueError("Candidate slice is empty")
    best, best_score = candidates[0], -np.inf
    records = []
    for candidate_no, candidate in enumerate(candidates, start=1):
        fold_scores = []
        for hold in train_contexts:
            inner_train_contexts = [c for c in train_contexts if c != hold]
            train_pairs = pair_indices(meta, inner_train_contexts)
            test_pairs = pair_indices(meta, [hold])
            if not train_pairs or not test_pairs:
                continue
            early = np.array([e for e, _, _ in train_pairs]); late = np.array([l for _, l, _ in train_pairs])
            test_rows = np.array([r for pair in test_pairs for r in pair[:2]])
            if method == "mean_direction": score = mean_direction_score(x, early, late, x[test_rows], *candidate)
            elif method == "pca": score = pca_score(x, early, late, x[test_rows], *candidate)
            elif method == "ridge": score = logistic_score(x, early, late, x[test_rows], *candidate)
            else: score = logistic_score(x, early, late, x[test_rows], *candidate[:2], elastic_l1=candidate[2])
            fold_scores.append(order_accuracy(score, test_pairs, {row: i for i, row in enumerate(test_rows)}))
        value = np.nanmean(fold_scores)
        records.append({"candidate": json.dumps(candidate), "inner_mean_accuracy": float(value)})
        if value > best_score:
            best, best_score = candidate, value
        if candidate_no % 4 == 0 or candidate_no == len(candidates):
            print(f"    tuning={candidate_no}/{len(candidates)} best={best_score:.3f}", flush=True)
    return best, float(best_score), records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer", choices=None, help="Run one held-out context only.")
    parser.add_argument("--method", choices=["mean_direction", "pca", "ridge", "elastic_net"], help="Run one baseline only.")
    parser.add_argument("--tune-start", type=int, default=0, help="Inclusive candidate offset for a checkpointed inner-search shard.")
    parser.add_argument("--tune-end", type=int, default=None, help="Exclusive candidate offset for a checkpointed inner-search shard.")
    parser.add_argument("--finalize", action="store_true", help="Score outer folds from previously checkpointed inner-search candidates.")
    args = parser.parse_args()
    meta, x = load()
    all_contexts = sorted(meta.context.unique())
    contexts = all_contexts.copy()
    methods = ["mean_direction", "pca", "ridge", "elastic_net"]
    if args.outer:
        if args.outer not in all_contexts:
            raise ValueError(f"Unknown outer context {args.outer!r}; available: {all_contexts}")
        contexts = [args.outer]
    if args.method:
        methods = [args.method]
    results = []
    predictions = []
    for hold in contexts:
        print(f"outer={hold}", flush=True)
        # `contexts` may be filtered by --outer for checkpointed execution; the
        # training universe must remain every other audited treatment context.
        train_contexts = [c for c in all_contexts if c != hold]
        train_pairs, test_pairs = pair_indices(meta, train_contexts), pair_indices(meta, [hold])
        early, late = np.array([e for e, _, _ in train_pairs]), np.array([l for _, l, _ in train_pairs])
        test_rows = np.array([r for pair in test_pairs for r in pair[:2]])
        for method in methods:
            print(f"  method={method}", flush=True)
            tuning_file = OUT / f"v6_inner_tuning_{hold}_{method}.csv"
            if args.finalize:
                tuning = pd.read_csv(tuning_file)
                winner = tuning.loc[tuning.inner_mean_accuracy.idxmax()]
                params, inner = tuple(json.loads(winner.candidate)), float(winner.inner_mean_accuracy)
            else:
                params, inner, records = inner_select(method, x, meta, train_contexts, args.tune_start, args.tune_end)
                tune_frame = pd.DataFrame(records)
                tune_frame.insert(0, "held_context", hold)
                tune_frame.insert(1, "method", method)
                tune_frame.insert(2, "candidate_offset", np.arange(args.tune_start, args.tune_start + len(tune_frame)))
                tune_frame.to_csv(tuning_file, mode="a", index=False, header=not tuning_file.exists())
                # A shard never touches an outer test fold; it only extends the
                # auditable nested-tuning ledger.
                if args.tune_end is not None:
                    continue
            if method == "mean_direction": score = mean_direction_score(x, early, late, x[test_rows], *params)
            elif method == "pca": score = pca_score(x, early, late, x[test_rows], *params)
            elif method == "ridge": score = logistic_score(x, early, late, x[test_rows], *params)
            else: score = logistic_score(x, early, late, x[test_rows], *params[:2], elastic_l1=params[2])
            accuracy = order_accuracy(score, test_pairs, {row: i for i, row in enumerate(test_rows)})
            results.append({"held_context": hold, "method": method, "n_pairs": len(test_pairs), "order_accuracy": accuracy, "inner_mean_accuracy": inner, "params": json.dumps(params)})
            for row, value in zip(test_rows, score):
                predictions.append({"held_context": hold, "method": method, "row": row, "score": value})
            # Checkpoint each completed evaluation: long elastic-net selection must never
            # erase completed outer-fold results if an external runner stops mid-job.
            suffix = f"_{hold}_{method}".replace("/", "_")
            pd.DataFrame(results).to_csv(OUT / f"v6_baseline_lodo_checkpoint{suffix}.csv", index=False)
            pd.DataFrame(predictions).to_csv(OUT / f"v6_baseline_lodo_predictions_checkpoint{suffix}.csv", index=False)
    frame = pd.DataFrame(results)
    frame.to_csv(OUT / "v6_baseline_lodo_results.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "v6_baseline_lodo_predictions.csv", index=False)
    if not frame.empty:
        print(frame.pivot(index="held_context", columns="method", values="order_accuracy").round(3).to_string())
    else:
        print("Nested-tuning shard saved; no outer-test scores computed.")


if __name__ == "__main__":
    main()

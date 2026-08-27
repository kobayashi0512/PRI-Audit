"""Nested leave-context-out baselines for the frozen v7 clinical endpoint."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "prismr_v7"
K_GRID, C_GRID, L1_GRID = (250, 500, 1000, 2000), (0.01, 0.1, 1.0, 10.0), (0.1, 0.5, 0.9)


def load():
    meta = pd.read_csv(OUT / "v7_baseline_ls_minus_nl_metadata.csv")
    x = pd.read_parquet(OUT / "v7_baseline_ls_minus_nl_rank.parquet").to_numpy(dtype=np.float32)
    meta["row"] = np.arange(len(meta))
    meta["context"] = meta.cohort + "::" + meta.drug
    # Outcome percentile is formed independently in each context. It is used by
    # regressors only; evaluation always uses the original within-context order.
    meta["train_target"] = meta.groupby("context").response_value.transform(lambda v: rankdata(v, method="average") / (len(v) + 1.0))
    return meta, x


def concordance(scores, rows, meta) -> float:
    frame = meta.iloc[rows].copy()
    frame["score"] = scores
    wins, total = 0, 0
    for _, sub in frame.groupby("context"):
        y, s = sub.response_value.to_numpy(), sub.score.to_numpy()
        for i in range(len(sub)):
            for j in range(i + 1, len(sub)):
                if y[i] == y[j]:
                    continue
                wins += (s[i] - s[j]) * (y[i] - y[j]) > 0
                total += 1
    return float(wins / total) if total else np.nan


def feature_idx(x, rows, k):
    return np.argsort(x[rows].var(axis=0))[-min(k, x.shape[1]):]


def fit_score(method, x, train_rows, test_rows, meta, params):
    k = params[0]
    idx = feature_idx(x, train_rows, k)
    scaler = StandardScaler().fit(x[train_rows][:, idx])
    xt, xv = scaler.transform(x[train_rows][:, idx]), scaler.transform(x[test_rows][:, idx])
    y = meta.iloc[train_rows].train_target.to_numpy()
    if method == "mean_direction":
        direction = xt.T @ (y - y.mean())
        return xv @ direction
    if method == "pca":
        model = PCA(n_components=1, random_state=20260730).fit(xt)
        train_score = model.transform(xt).ravel()
        sign = 1 if np.corrcoef(train_score, y)[0, 1] >= 0 else -1
        return sign * model.transform(xv).ravel()
    if method == "ridge":
        return Ridge(alpha=1 / params[1]).fit(xt, y).predict(xv)
    if method == "pairwise_ranker":
        # This is deliberately PRISM-R without context-invariance or transport:
        # only the mixed-label, within-context pairwise ranking core remains.
        pairs, targets = [], []
        train = meta.iloc[train_rows]
        for _, sub in train.groupby("context"):
            local = sub.row.to_numpy()
            outcome = sub.response_value.to_numpy()
            pos = {row: i for i, row in enumerate(train_rows)}
            for i in range(len(local)):
                for j in range(i + 1, len(local)):
                    if outcome[i] == outcome[j]:
                        continue
                    diff = xt[pos[local[i]]] - xt[pos[local[j]]]
                    target = int(outcome[i] > outcome[j])
                    pairs.extend([diff, -diff])
                    targets.extend([target, 1 - target])
        model = LogisticRegression(C=params[1], penalty="l2", solver="liblinear", fit_intercept=False,
                                   max_iter=3000, random_state=20260730)
        model.fit(np.asarray(pairs), np.asarray(targets))
        return xv @ model.coef_.ravel()
    return ElasticNet(alpha=1 / params[1], l1_ratio=params[2], max_iter=3000, tol=1e-3, random_state=20260730).fit(xt, y).predict(xv)


def candidates(method):
    if method in {"mean_direction", "pca"}:
        return [(k,) for k in K_GRID]
    if method in {"ridge", "pairwise_ranker"}:
        return [(k, c) for k in K_GRID for c in C_GRID]
    return [(k, c, l1) for k in K_GRID for c in C_GRID for l1 in L1_GRID]


def inner_select(method, x, meta, train_contexts):
    best, best_score = None, -np.inf
    for no, params in enumerate(candidates(method), 1):
        fold = []
        for held in train_contexts:
            tr_contexts = [c for c in train_contexts if c != held]
            tr_rows = meta.loc[meta.context.isin(tr_contexts), "row"].to_numpy()
            te_rows = meta.loc[meta.context.eq(held), "row"].to_numpy()
            fold.append(concordance(fit_score(method, x, tr_rows, te_rows, meta, params), te_rows, meta))
        score = float(np.nanmean(fold))
        if score > best_score:
            best, best_score = params, score
        if no % 4 == 0 or no == len(candidates(method)):
            print(f"    tune {no}/{len(candidates(method))}: best={best_score:.3f}", flush=True)
    return best, best_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["mean_direction", "pca", "ridge", "elastic_net", "pairwise_ranker"])
    parser.add_argument("--outer", help="Optional one held-out context")
    args = parser.parse_args()
    meta, x = load()
    all_contexts = sorted(meta.context.unique())
    held_contexts = [args.outer] if args.outer else all_contexts
    if args.outer and args.outer not in all_contexts:
        raise ValueError(f"unknown context: {args.outer}")
    methods = [args.method] if args.method else ["mean_direction", "pca", "ridge", "elastic_net", "pairwise_ranker"]
    results = []
    for held in held_contexts:
        train_contexts = [c for c in all_contexts if c != held]
        tr_rows = meta.loc[meta.context.isin(train_contexts), "row"].to_numpy()
        te_rows = meta.loc[meta.context.eq(held), "row"].to_numpy()
        print(f"outer={held}", flush=True)
        for method in methods:
            print(f"  {method}", flush=True)
            params, inner = inner_select(method, x, meta, train_contexts)
            score = fit_score(method, x, tr_rows, te_rows, meta, params)
            results.append({"held_context": held, "method": method, "n_patients": len(te_rows),
                            "concordance": concordance(score, te_rows, meta), "inner_concordance": inner,
                            "params": json.dumps(params)})
            pd.DataFrame(results).to_csv(OUT / f"v7_baseline_checkpoint_{held}_{method}.csv", index=False)
    frame = pd.DataFrame(results)
    frame.to_csv(OUT / "v7_baseline_loco_results.csv", index=False)
    if not frame.empty:
        print(frame.pivot(index="held_context", columns="method", values="concordance").round(3).to_string())


if __name__ == "__main__":
    main()

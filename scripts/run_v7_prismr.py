"""PRISM-R: partial-order, context-invariant, Sinkhorn-regularized ranker."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "prismr_v7"
STEPS, LR, EPSILON, SEED = 350, 0.05, 0.5, 20260730


def load():
    meta = pd.read_csv(OUT / "v7_baseline_ls_minus_nl_metadata.csv")
    x = pd.read_parquet(OUT / "v7_baseline_ls_minus_nl_rank.parquet").to_numpy(dtype=np.float64)
    meta["row"] = np.arange(len(meta))
    meta["context"] = meta.cohort + "::" + meta.drug
    return meta, x


def feature_idx(x, rows, k):
    return np.argsort(x[rows].var(axis=0))[-k:]


def training_pairs(meta, rows):
    frame = meta.iloc[rows]
    pairs = []
    local = {row: n for n, row in enumerate(rows)}
    for _, sub in frame.groupby("context"):
        rr, yy = sub.row.to_numpy(), sub.response_value.to_numpy()
        for i in range(len(rr)):
            for j in range(i + 1, len(rr)):
                sign = np.sign(yy[i] - yy[j])
                if sign:
                    pairs.append((local[rr[i]], local[rr[j]], sign))
    return pairs


def response_strata(meta, rows):
    """Within-context low/high labels; never compare outcome magnitudes across drugs."""
    frame = meta.iloc[rows]
    out = np.zeros(len(rows), dtype=int)
    local = {row: n for n, row in enumerate(rows)}
    for _, sub in frame.groupby("context"):
        rr, y = sub.row.to_numpy(), sub.response_value.to_numpy()
        # PASI75 already supplies response classes; continuous labels use only
        # their within-context median split for conditional alignment.
        if sub.label_type.iloc[0] == "pasi75_binary":
            high = y > 0
        else:
            # Median of *unique* values avoids an empty high stratum when a
            # small trial has many patients tied at maximal improvement.
            high = y > np.median(np.unique(y))
        out[[local[row] for row in rr]] = high.astype(int)
    return out


def sinkhorn(a, b):
    """Balanced entropic transport plan for one-dimensional score samples."""
    cost = (a[:, None] - b[None, :]) ** 2
    scale = max(float(np.median(cost)), 1e-6)
    kernel = np.exp(-np.clip(cost / (EPSILON * scale), 0, 60)) + 1e-12
    u = np.full(len(a), 1 / len(a)); v = np.full(len(b), 1 / len(b))
    for _ in range(30):
        u = (1 / len(a)) / (kernel @ v + 1e-12)
        v = (1 / len(b)) / (kernel.T @ u + 1e-12)
    return (u[:, None] * kernel) * v[None, :]


def fit_prism(x, meta, rows, params, variant):
    k, alpha, beta, gamma = params
    if variant == "no_invariance": beta = 0.0
    if variant == "no_ot": gamma = 0.0
    idx = feature_idx(x, rows, k)
    scaler = StandardScaler().fit(x[rows][:, idx])
    xt = scaler.transform(x[rows][:, idx])
    pairs = training_pairs(meta, rows)
    contexts = meta.iloc[rows].context.to_numpy()
    groups = [np.flatnonzero(contexts == c) for c in sorted(set(contexts))]
    strata = response_strata(meta, rows)
    valid_groups = [g for g in groups if np.any(strata[g] == 0) and np.any(strata[g] == 1)]
    w = np.zeros(xt.shape[1], dtype=float)
    trace = []
    for step in range(STEPS):
        grad = np.zeros_like(w)
        # Partial-order logistic loss: binary and continuous labels never share
        # a numeric scale, only valid within-context comparisons.
        for i, j, sign in pairs:
            diff = xt[i] - xt[j]
            grad += -sign * expit(-sign * (diff @ w)) * diff
        grad /= max(len(pairs), 1)
        scores = xt @ w
        # Context-invariant *response contrast* penalty. Raw score locations are
        # intentionally not aligned because treatment calibrations may differ.
        if beta:
            contrast, contrast_x = [], []
            for g in valid_groups:
                lo, hi = g[strata[g] == 0], g[strata[g] == 1]
                contrast.append(scores[hi].mean() - scores[lo].mean())
                contrast_x.append(xt[hi].mean(axis=0) - xt[lo].mean(axis=0))
            mean_contrast = float(np.mean(contrast)); mean_x = np.mean(contrast_x, axis=0)
            for value, value_x in zip(contrast, contrast_x):
                grad += 2 * beta * (value - mean_contrast) * (value_x - mean_x)
        # Envelope gradient of conditional balanced entropic OT. Only patients
        # in equal within-context response strata are transported together.
        if gamma:
            for g in valid_groups:
                for level in (0, 1):
                    source = g[strata[g] == level]
                    other = np.concatenate([q[strata[q] == level] for q in valid_groups if contexts[q[0]] != contexts[g[0]]])
                    if not len(source) or not len(other):
                        continue
                    plan = sinkhorn(scores[source], scores[other])
                    delta = scores[source][:, None] - scores[other][None, :]
                    grad += 2 * gamma * np.sum(plan[:, :, None] * delta[:, :, None] * (xt[source][:, None, :] - xt[other][None, :, :]), axis=(0, 1))
        # A bounded smooth update is required for the high-dimensional pairwise
        # objective; L1 sparsification remains the subsequent proximal step.
        grad = np.nan_to_num(grad, nan=0.0, posinf=10.0, neginf=-10.0)
        norm = np.linalg.norm(grad)
        if norm > 10.0:
            grad *= 10.0 / norm
        w -= LR * grad
        w = np.sign(w) * np.maximum(np.abs(w) - LR * alpha, 0.0)
        if not np.isfinite(w).all():
            raise FloatingPointError("non-finite PRISM-R iterate")
        if step in {0, 49, 99, 199, 349}:
            pair_margin = [-s * ((xt[i] - xt[j]) @ w) for i, j, s in pairs]
            trace.append({"step": step + 1, "nonzero": int((w != 0).sum()),
                          "pair_loss_proxy": float(np.mean(np.logaddexp(0, pair_margin)))})
    return idx, scaler, w, trace


def score(x, test_rows, idx, scaler, w):
    return scaler.transform(x[test_rows][:, idx]) @ w


def concordance(scores, rows, meta):
    sub = meta.iloc[rows].copy(); sub["score"] = scores
    wins = total = 0
    for _, group in sub.groupby("context"):
        y, s = group.response_value.to_numpy(), group.score.to_numpy()
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if y[i] != y[j]:
                    wins += (y[i] - y[j]) * (s[i] - s[j]) > 0; total += 1
    return wins / total if total else np.nan


def config_grid(variant):
    base = itertools.product((250, 500), (0.001, 0.01))
    if variant == "full": return [(k, a, .05, .05) for k, a in base]
    if variant == "no_invariance": return [(k, a, 0.0, .05) for k, a in base]
    if variant == "no_ot": return [(k, a, .05, 0.0) for k, a in base]
    return [(k, a, 0.0, 0.0) for k, a in base]


def inner_select(x, meta, contexts, variant):
    best, best_value = None, -np.inf
    for number, params in enumerate(config_grid(variant), 1):
        fold = []
        for held in contexts:
            train_contexts = [c for c in contexts if c != held]
            tr = meta.loc[meta.context.isin(train_contexts), "row"].to_numpy()
            te = meta.loc[meta.context.eq(held), "row"].to_numpy()
            idx, sc, w, _ = fit_prism(x, meta, tr, params, variant)
            fold.append(concordance(score(x, te, idx, sc, w), te, meta))
        value = float(np.nanmean(fold))
        if value > best_value: best, best_value = params, value
        print(f"    tune {number}/4 best={best_value:.3f}", flush=True)
    return best, best_value


def inner_value(x, meta, contexts, params, variant):
    fold = []
    for held in contexts:
        train_contexts = [c for c in contexts if c != held]
        tr = meta.loc[meta.context.isin(train_contexts), "row"].to_numpy()
        te = meta.loc[meta.context.eq(held), "row"].to_numpy()
        idx, sc, w, _ = fit_prism(x, meta, tr, params, variant)
        fold.append(concordance(score(x, te, idx, sc, w), te, meta))
    return float(np.nanmean(fold))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer", required=True)
    parser.add_argument("--variant", choices=["full", "no_invariance", "no_ot", "pairwise_only"], default="full")
    parser.add_argument("--candidate", type=int, help="0-based nested-tuning shard; never scores the outer test domain")
    parser.add_argument("--finalize", action="store_true", help="Select from completed tuning shards and score the outer test domain")
    args = parser.parse_args()
    meta, x = load(); contexts = sorted(meta.context.unique())
    if args.outer not in contexts: raise ValueError(args.outer)
    train_contexts = [c for c in contexts if c != args.outer]
    tr = meta.loc[meta.context.isin(train_contexts), "row"].to_numpy()
    te = meta.loc[meta.context.eq(args.outer), "row"].to_numpy()
    print(f"outer={args.outer} variant={args.variant}", flush=True)
    safe_outer = args.outer.replace("::", "__")
    if args.candidate is not None:
        grid = config_grid(args.variant)
        if args.candidate < 0 or args.candidate >= len(grid): raise ValueError("candidate outside frozen grid")
        params = grid[args.candidate]
        value = inner_value(x, meta, train_contexts, params, args.variant)
        pd.DataFrame([{"candidate": args.candidate, "params": json.dumps(params), "inner_concordance": value}]).to_csv(
            OUT / f"v7_2_prismr_tuning_{safe_outer}_{args.variant}_{args.candidate}.csv", index=False)
        print(json.dumps({"candidate": args.candidate, "params": params, "inner_concordance": value}, indent=2)); return
    if args.finalize:
        shards = [pd.read_csv(OUT / f"v7_2_prismr_tuning_{safe_outer}_{args.variant}_{i}.csv") for i in range(len(config_grid(args.variant)))]
        tuning = pd.concat(shards, ignore_index=True)
        winner = tuning.loc[tuning.inner_concordance.idxmax()]
        params, inner = tuple(json.loads(winner.params)), float(winner.inner_concordance)
    else:
        params, inner = inner_select(x, meta, train_contexts, args.variant)
    idx, sc, w, trace = fit_prism(x, meta, tr, params, args.variant)
    result = {"algorithm_version": "v7.2_conditional", "held_context": args.outer, "variant": args.variant, "n_patients": len(te), "concordance": concordance(score(x, te, idx, sc, w), te, meta), "inner_concordance": inner, "params": json.dumps(params), "nonzero": int((w != 0).sum())}
    pd.DataFrame([result]).to_csv(OUT / f"v7_2_prismr_checkpoint_{safe_outer}_{args.variant}.csv", index=False)
    pd.DataFrame(trace).to_csv(OUT / f"v7_2_prismr_trace_{safe_outer}_{args.variant}.csv", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()

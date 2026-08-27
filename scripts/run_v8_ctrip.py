"""Nested cohort-LOO C-TRIP on deterministic held-out gene blocks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear, nnls

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "trigger_v8"
LAMBDAS = (0.01, 0.1, 1.0, 10.0)


def gene_block(g): return int(hashlib.sha256(g.encode()).hexdigest()[:8], 16) % 5
def cosine(a, b):
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d else np.nan


def robust_prior(x, b, indices, mask, meta):
    """Median within each cohort, then median across cohorts: robust invariant prior."""
    activity = []
    for i in indices:
        a, _ = nnls(b[mask], x[i, mask]); activity.append((meta.cohort.iloc[i], a))
    by_cohort = [np.median([a for c, a in activity if c == cohort], axis=0) for cohort in sorted({c for c, _ in activity})]
    return np.median(np.vstack(by_cohort), axis=0)


def shrunk_activity(d, b, mask, prior, lam):
    design = np.vstack([b[mask], np.sqrt(lam) * np.eye(2)])
    target = np.r_[d[mask], np.sqrt(lam) * prior]
    return lsq_linear(design, target, bounds=(0, np.inf), tol=1e-10).x


def score_context(x, b, meta, fit_indices, test_indices, lam):
    blocks = np.array([gene_block(g) for g in pd.read_csv(OUT / "v8_common_genes.csv").gene])
    out = []
    for held_block in range(5):
        train, test = blocks != held_block, blocks == held_block
        prior = robust_prior(x, b, fit_indices, train, meta)
        for i in test_indices:
            a = shrunk_activity(x[i], b, train, prior, lam)
            out.append(cosine(x[i, test], b[test] @ a))
    return float(np.mean(out))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--outer", required=True); args = parser.parse_args()
    x = pd.read_parquet(OUT / "v8_clinical_ls_minus_nl_rank.parquet").to_numpy(dtype=float)
    b = pd.read_parquet(OUT / "v8_perturbation_dictionary.parquet").to_numpy(dtype=float)[:, :2]
    meta = pd.read_csv(OUT / "v8_clinical_ls_minus_nl_metadata.csv")
    cohorts = sorted(meta.cohort.unique())
    if args.outer not in cohorts: raise ValueError(args.outer)
    outer_test = np.flatnonzero(meta.cohort.eq(args.outer)); train_cohorts = [c for c in cohorts if c != args.outer]
    # Nested selection: each candidate is evaluated by holding out one of the
    # two remaining cohorts; no outer-cohort patients influence lambda choice.
    tuning = []
    for lam in LAMBDAS:
        inner = []
        for held in train_cohorts:
            test = np.flatnonzero(meta.cohort.eq(held)); fit = np.flatnonzero(~meta.cohort.isin([args.outer, held]))
            inner.append(score_context(x, b, meta, fit, test, lam))
        tuning.append({"lambda": lam, "inner_mean_cosine": float(np.mean(inner))})
    best = max(tuning, key=lambda r: r["inner_mean_cosine"])
    fit = np.flatnonzero(meta.cohort.isin(train_cohorts))
    result = {"held_cohort": args.outer, "method": "C-TRIP", "lambda": best["lambda"], "inner_mean_cosine": best["inner_mean_cosine"],
              "held_gene_cosine": score_context(x, b, meta, fit, outer_test, best["lambda"]), "n_patients": len(outer_test)}
    safe = args.outer.replace("::", "__")
    pd.DataFrame(tuning).to_csv(OUT / f"v8_ctrip_tuning_{safe}.csv", index=False)
    pd.DataFrame([result]).to_csv(OUT / f"v8_ctrip_checkpoint_{safe}.csv", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()

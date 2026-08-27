"""Pre-algorithm validation of public perturbation anchors in paired cohorts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "trigger_v8"
SEED, N_BOOT, N_PERM = 20260730, 5000, 2000


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def bootstrap_mean(values, rng):
    means = np.array([rng.choice(values, len(values), replace=True).mean() for _ in range(N_BOOT)])
    return float(values.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def main():
    x = pd.read_parquet(OUT / "v8_clinical_ls_minus_nl_rank.parquet").to_numpy(dtype=float)
    b = pd.read_parquet(OUT / "v8_perturbation_dictionary.parquet").to_numpy(dtype=float)
    meta = pd.read_csv(OUT / "v8_clinical_ls_minus_nl_metadata.csv")
    rng = np.random.default_rng(SEED)
    names = ["IL17", "TNF_high", "IL17_TNF_synergy"]
    rows = []
    for cohort, sub in meta.groupby("cohort"):
        idx = sub.index.to_numpy()
        for j, name in enumerate(names):
            values = np.array([cosine(x[i], b[:, j]) for i in idx])
            observed, low, high = bootstrap_mean(values, rng)
            # Gene-label permutation preserves each program's values and each
            # patient's rank distribution while destroying biological matching.
            null = np.empty(N_PERM)
            cohort_mean = x[idx].mean(axis=0)
            for k in range(N_PERM):
                null[k] = cosine(cohort_mean, b[rng.permutation(b.shape[0]), j])
            observed_cohort = cosine(cohort_mean, b[:, j])
            rows.append({"cohort": cohort, "program": name, "n_patients": len(idx), "patient_mean_cosine": observed,
                         "bootstrap_ci_low": low, "bootstrap_ci_high": high, "cohort_mean_cosine": observed_cohort,
                         "permutation_p_one_sided": float((null >= observed_cohort).mean()), "null_mean": float(null.mean())})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "v8_anchor_patient_bootstrap_permutation.csv", index=False)
    report = {"n_bootstrap": N_BOOT, "n_gene_permutations": N_PERM,
              "criterion": "each primary program must have positive cohort-mean cosine and permutation p<0.05 in >=2 cohorts before C-TRIP implementation"}
    (OUT / "v8_anchor_validation_protocol.json").write_text(json.dumps(report, indent=2) + "\n")
    print(frame.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

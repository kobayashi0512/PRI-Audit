"""Shared cohort loading that reproduces the paper's eligibility rule exactly.

run_pri_audit_v21.eligible_core() keeps ONLY patients that contribute both a
LESIONAL and a NON LESIONAL sample. For GSE206391 that is 48 of 66 samples and
12 of 24 patients. Any experiment that skips this rule gets a different
reference AUROC and is not comparable to the published table.

Every experiment in this directory loads cohorts through here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
V21 = ROOT / "results" / "pri_audit_v21"
DEST = ROOT / "scripts" / "pri132_revision"


def log_cpm(counts: np.ndarray) -> np.ndarray:
    lib = counts.sum(1, keepdims=True)
    lib[lib == 0] = 1.0
    return np.log1p(counts / lib * 1e6)


def gse206391() -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Paired-only GSE206391, matching the paper exactly (48 samples / 12 patients)."""
    d = np.load(V21 / "gse206391_sample_gene_counts.npz", allow_pickle=True)
    counts = d["counts"].astype(float)
    meta = pd.read_csv(V21 / "gse206391_sample_metadata.csv")

    subset = meta[meta.condition.isin(["NON LESIONAL", "LESIONAL"])]
    paired = subset.groupby("patient").condition.nunique()
    keep_patients = paired[paired == 2].index
    keep = (meta.patient.isin(keep_patients)
            & meta.condition.isin(["NON LESIONAL", "LESIONAL"])).to_numpy()

    x = log_cpm(counts[keep])
    y = (meta.loc[keep, "condition"] == "LESIONAL").astype(int).to_numpy()
    groups = meta.loc[keep, "patient"].to_numpy()
    meta_kept = meta.loc[keep].reset_index(drop=True)
    return x, y, groups, meta_kept


def gse173706() -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Paired-only GSE173706 (22 samples / 11 donors)."""
    d = np.load(V21 / "gse173706_sample_gene_counts.npz", allow_pickle=True)
    counts = d["counts"].astype(float)
    meta = pd.read_csv(V21 / "gse173706_sample_metadata.csv")

    subset = meta[meta.condition.isin(["peripheral_normal", "psoriatic"])]
    paired = subset.groupby("donor").condition.nunique()
    donors = paired[paired == 2].index
    keep = (meta.donor.isin(donors)
            & meta.condition.isin(["peripheral_normal", "psoriatic"])).to_numpy()

    x = log_cpm(counts[keep])
    y = (meta.loc[keep, "condition"] == "psoriatic").astype(int).to_numpy()
    groups = meta.loc[keep, "donor"].to_numpy()
    meta_kept = meta.loc[keep].reset_index(drop=True)
    return x, y, groups, meta_kept


def gse151177() -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """GSE151177 source samples (17 samples: 13 psoriasis / 4 control)."""
    d = np.load(V21 / "gse151177_sample_gene_counts.npz", allow_pickle=True)
    counts = d["counts"].astype(float)
    meta = pd.read_csv(V21 / "gse151177_sample_metadata.csv")
    keep = meta.condition.isin(["control", "psoriasis"]).to_numpy()
    x = log_cpm(counts[keep])
    y = (meta.loc[keep, "condition"] == "psoriasis").astype(int).to_numpy()
    groups = meta.loc[keep, "sample"].astype(str).to_numpy()
    return x, y, groups, meta.loc[keep].reset_index(drop=True)


LOADERS = {
    "gse206391": gse206391,
    "gse173706": gse173706,
    "gse151177": gse151177,
}
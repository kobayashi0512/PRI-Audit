"""Frozen independent strict boundary replication in GSE202011 Visium data.

GSE202011 is independent of PRI-Audit development and provides processed 10x
H5 matrices for biologic-naive psoriatic skin.  Its public H5 archive does not
include spot coordinates, so the registered external test is deliberately
limited to the patient-level expression-versus-PCA comparison.  It cannot be
misrepresented as an external spatial-graph validation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_pri_audit_v21 as v21  # noqa: E402


RAW = v21.RAW / "GSE202011_samples"
OUT = v21.OUT
ARCHIVE = v21.RAW / "GSE202011_RAW.tar"
SEED = 20260824

# Prespecified from the GEO series sample list.  ST-14 has two lesional
# sections; the first (GSM6086617 / R1) is selected deterministically before
# any expression matrix is examined.  PSA samples and the unpaired PSO lesion
# are outside the frozen PSO LS/NL estimand.
SAMPLES = [
    ("ST13", "GSM6086615", "LESIONAL"), ("ST13", "GSM6086616", "NON_LESIONAL"),
    ("ST14", "GSM6086617", "LESIONAL"), ("ST14", "GSM6086619", "NON_LESIONAL"),
    ("ST17", "GSM6086625", "LESIONAL"), ("ST17", "GSM6086626", "NON_LESIONAL"),
    ("ST20", "GSM6086630", "LESIONAL"), ("ST20", "GSM6086631", "NON_LESIONAL"),
    ("ST21", "GSM6086632", "LESIONAL"), ("ST21", "GSM6086633", "NON_LESIONAL"),
]


def decode(values: np.ndarray) -> np.ndarray:
    return np.asarray([item.decode() if isinstance(item, bytes) else str(item) for item in values], dtype=str)


def locate(gsm: str) -> Path:
    hits = sorted(RAW.glob(f"{gsm}_*_filtered_feature_bc_matrix.h5"))
    if len(hits) != 1:
        raise FileNotFoundError(f"Expected one H5 for {gsm}; found {[item.name for item in hits]}")
    return hits[0]


def sample_sum(path: Path) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Read 10x H5 directly and return QC-filtered pseudobulk gene counts."""
    with h5py.File(path, "r") as handle:
        group = handle["matrix"]
        shape = tuple(int(value) for value in group["shape"][...])  # genes, spots
        # 10x H5 stores a gene-by-spot compressed-*column* matrix: indptr has
        # one entry per spot, so construct CSC before transposing to spots ×
        # genes.  Treating it as CSR silently changes the pointer invariant.
        matrix = csc_matrix((group["data"][...], group["indices"][...], group["indptr"][...]), shape=shape).T.tocsr()
        genes = decode(group["features"]["id"][...])
        symbols = decode(group["features"]["name"][...])
    total = np.asarray(matrix.sum(axis=1)).ravel()
    detected = np.asarray((matrix > 0).sum(axis=1)).ravel()
    mitochondrial = np.char.startswith(np.char.upper(symbols), "MT-")
    mito_fraction = np.asarray(matrix[:, mitochondrial].sum(axis=1)).ravel() / np.maximum(total, 1)
    keep = (total >= 500) & (detected >= 300) & (mito_fraction <= .20)
    return np.asarray(matrix[keep].sum(axis=0)).ravel().astype(float), genes, int(matrix.shape[0]), int(keep.sum())


def rebuild() -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    rows, counts, genes_ref = [], [], None
    for patient, gsm, condition in SAMPLES:
        path = locate(gsm)
        values, genes, raw_spots, qc_spots = sample_sum(path)
        if genes_ref is None:
            genes_ref = genes
        elif not np.array_equal(genes_ref, genes):
            raise ValueError(f"Gene ordering differs in {path.name}")
        rows.append({"patient": patient, "gsm": gsm, "file": path.name, "condition": condition, "raw_filtered_matrix_spots": raw_spots, "qc_retained_spots": qc_spots, "input_sha256": v21.sha256(path)})
        counts.append(values)
    metadata = pd.DataFrame(rows)
    if metadata.groupby("patient").condition.nunique().ne(2).any():
        raise RuntimeError("Frozen external PSO patient set is not paired")
    matrix = np.vstack(counts)
    np.savez_compressed(OUT / "gse202011_external_sample_gene_counts.npz", counts=matrix, genes=genes_ref)
    metadata.to_csv(OUT / "gse202011_external_sample_metadata.csv", index=False)
    return matrix, metadata, genes_ref


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    counts, metadata, _ = rebuild()
    x = v21.log_cpm(counts)
    y = (metadata.condition == "LESIONAL").astype(int).to_numpy()
    groups = metadata.patient.astype(str).to_numpy()
    result, _ = v21.evaluate_strict_loose(
        "gse202011_external", x, y, groups, metadata, paired=True, allow_nmf=False,
        n_permutations=args.permutations, n_bootstrap=args.bootstrap, out=OUT,
    )
    selected = {row["arm"].replace("strict_", ""): row for row in result["methods"] if row["arm"].startswith("strict_")}
    for name, row in selected.items():
        row["increment_ci95"] = [row["increment_bootstrap_ci95_low"], row["increment_bootstrap_ci95_high"]]
    report = {
        "dataset": "GSE202011", "analysis_role": "independent high-resolution boundary replication", "evaluation_unit": "10 samples from 5 paired PSO patients; leave-one-patient-out",
        "selection_rule": "Pre-specified GEO series sample ledger; ST14 R1 lesion selected over R2 before expression inspection",
        "raw_archive": {"path": str(ARCHIVE.relative_to(v21.ROOT)), "sha256": v21.sha256(ARCHIVE)},
        "spot_qc": "processed 10x filtered matrix; total counts >=500, detected genes >=300, mitochondrial fraction <=0.20",
        "representation_scope": "HVG selection, scaling, PCA and classifier fit only in outer-patient training folds",
        "spatial_graph_status": "not evaluated: GEO H5 matrix archive has no spot-coordinate files; this is not an external spatial-graph test",
        "methods": selected, "n_permutations": args.permutations, "n_bootstrap": args.bootstrap, "seed": v21.SEED,
    }
    (OUT / "gse202011_v21_external_strict.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Streaming QC for GSE173706 supplementary per-cell CSV matrices.

The files are dense CSVs (genes x cells), so this script streams rows and does
not materialize a full matrix. It reports per-sample cell QC and a filtered
cell count for downstream patient-level analyses.
"""
from pathlib import Path
import csv, gzip, re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/raw/GSE173706_samples"
OUT = ROOT / "results/singlecell_v10"
OUT.mkdir(parents=True, exist_ok=True)

# Human mitochondrial genes represented by Ensembl IDs in these CSV files.
MITO_ENSG = {
    "ENSG00000198888", "ENSG00000198763", "ENSG00000198804",
    "ENSG00000198712", "ENSG00000228263", "ENSG00000198899",
    "ENSG00000198938", "ENSG00000198840", "ENSG00000198727",
    "ENSG00000198886", "ENSG00000198786", "ENSG00000198695",
    "ENSG00000198700", "ENSG00000210049", "ENSG00000211459",
}

def one(path):
    with gzip.open(path, "rt", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        n = len(header) - 1
        totals = np.zeros(n, dtype=np.float64)
        detected = np.zeros(n, dtype=np.int32)
        mito = np.zeros(n, dtype=np.float64)
        genes = 0
        for row in reader:
            if not row:
                continue
            gene = row[0].upper()
            vals = np.asarray(row[1:], dtype=np.float64)
            if vals.size != n:
                raise ValueError(f"{path.name}: row has {vals.size}, expected {n}")
            totals += vals
            detected += vals > 0
            if gene in MITO_ENSG or gene.startswith("MT-"):
                mito += vals
            genes += 1
    frac = mito / np.maximum(totals, 1)
    keep = (totals >= 500) & (detected >= 300) & (frac <= 0.20)
    name = path.name.replace(".csv.gz", "")
    # GEO sample IDs precede the biological label (e.g. GSM5277189_PP-30696).
    label = name.split("_", 1)[1] if "_" in name else name
    condition = ("normal_skin" if label.startswith("NS-") else
                 "peripheral_normal" if label.startswith("PN-") else
                 "psoriatic" if label.startswith("PP-") else "unknown")
    # Donor is encoded after the condition prefix; keep V3 as a separate sample
    # because it is not safe to infer pairing without the GEO metadata table.
    donor = re.sub(r"^(NS|PN|PP)-", "", label)
    return {
        "sample": name, "condition": condition, "donor_label": donor,
        "genes": genes, "raw_cells": n, "filtered_cells": int(keep.sum()),
        "retention": float(keep.mean()),
        "median_counts": float(np.median(totals)),
        "median_detected_genes": float(np.median(detected)),
        "median_mito_fraction": float(np.median(frac)),
        "integrity": "passed",
    }

def main():
    paths = sorted(SRC.glob("*.csv.gz"))
    if not paths:
        raise SystemExit("No GSE173706 CSV files found")
    rows = [one(p) for p in paths]
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "gse173706_sample_qc.csv", index=False)
    print(out.to_string(index=False))
    print("\ncondition summary:")
    print(out.groupby("condition")[["raw_cells", "filtered_cells"]].sum())

if __name__ == "__main__":
    main()

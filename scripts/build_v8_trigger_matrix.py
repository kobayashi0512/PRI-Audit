"""Build shared-gene public-data inputs for the C-TRIP feasibility experiment."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "results" / "trigger_v8"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import collapse_duplicate_genes, parse_geo_series_matrix  # noqa: E402
from run_isrc_v2 import load_platform_map  # noqa: E402

SOURCES = {
    "GSE153007": (RAW / "GSE153007_series_matrix.txt.gz", RAW / "GPL6480.annot.gz"),
    "GSE182740": (RAW / "GSE182740_series_matrix.txt.gz", RAW / "GPL570.annot.gz"),
    "GSE57225": (RAW / "GSE57225_series_matrix.txt.gz", RAW / "GPL14550_full.txt"),
    "GSE24767": (RAW / "GSE24767_series_matrix.txt.gz", RAW / "GPL6947.annot.gz"),
}


def expression(cohort):
    path, platform = SOURCES[cohort]
    x, _ = parse_geo_series_matrix(path)
    x.index = x.index.astype(str).map(load_platform_map(platform))
    x = collapse_duplicate_genes(x)
    return x.loc[x.index.notna() & ~x.index.duplicated()].copy()


def r(x):
    return rankdata(x.to_numpy(dtype=float), method="average") / len(x)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(OUT / "v8_paired_psoriasis_metadata.csv")
    perturb = pd.read_csv(OUT / "v8_keratinocyte_perturbation_metadata.csv")
    expr = {k: expression(k) for k in SOURCES}
    genes = sorted(set.intersection(*(set(v.index) for v in expr.values())))
    if len(genes) < 5000:
        raise RuntimeError(f"shared-gene gate failed: {len(genes)}")
    rows, patient_meta = [], []
    for (cohort, subject), sub in meta.groupby(["cohort", "subject"]):
        pivot = sub.set_index("state").accession
        if not {"LS", "NL"}.issubset(pivot.index):
            continue
        x = expr[cohort].loc[genes]
        rows.append((r(x[pivot["LS"]]) - r(x[pivot["NL"]])).astype(np.float32))
        patient_meta.append({"cohort": cohort, "subject": subject, "ls_accession": pivot["LS"], "nl_accession": pivot["NL"]})
    clinical = pd.DataFrame(np.vstack(rows), columns=genes)
    clinical_meta = pd.DataFrame(patient_meta)
    # Perturbation vectors are donor-paired, so donor baseline variability does
    # not enter the experimentally anchored dictionary.
    p = perturb.pivot_table(index="donor", columns="treatment", values="accession", aggfunc="first")
    needed = ["none", "IL17", "10 ng/ml TNFα", "IL17 + 10 ng/ml TNFα"]
    p = p.dropna(subset=needed)
    if len(p) < 3:
        raise RuntimeError("fewer than three complete perturbation donors")
    x = expr["GSE24767"].loc[genes]
    il17 = np.vstack([r(x[row["IL17"]]) - r(x[row["none"]]) for _, row in p.iterrows()])
    tnf = np.vstack([r(x[row["10 ng/ml TNFα"]]) - r(x[row["none"]]) for _, row in p.iterrows()])
    combo = np.vstack([r(x[row["IL17 + 10 ng/ml TNFα"]]) - r(x[row["none"]]) for _, row in p.iterrows()])
    dictionary = np.column_stack([il17.mean(axis=0), tnf.mean(axis=0), combo.mean(axis=0) - il17.mean(axis=0) - tnf.mean(axis=0)]).astype(np.float32)
    clinical.to_parquet(OUT / "v8_clinical_ls_minus_nl_rank.parquet", index=False)
    clinical_meta.to_csv(OUT / "v8_clinical_ls_minus_nl_metadata.csv", index=False)
    pd.DataFrame(dictionary, columns=["IL17", "TNF_high", "IL17_TNF_synergy"]).to_parquet(OUT / "v8_perturbation_dictionary.parquet", index=False)
    pd.DataFrame({"gene": genes}).to_csv(OUT / "v8_common_genes.csv", index=False)
    print(f"clinical={clinical.shape}; dictionary={dictionary.shape}; complete perturbation donors={len(p)}")
    print(clinical_meta.groupby("cohort").size().to_string())


if __name__ == "__main__":
    main()

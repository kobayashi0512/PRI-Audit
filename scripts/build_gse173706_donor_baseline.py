"""Build donor-level matrices and leakage-safe PN-vs-PP baselines for GSE173706."""
from pathlib import Path
import csv, gzip, json, re
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/raw/GSE173706_samples"
OUT = ROOT / "results/singlecell_v10"
OUT.mkdir(parents=True, exist_ok=True)
QC = pd.read_csv(OUT / "gse173706_sample_qc.csv").set_index("sample")
MITO = {"ENSG00000198888","ENSG00000198763","ENSG00000198804","ENSG00000198712","ENSG00000228263","ENSG00000198899","ENSG00000198938","ENSG00000198840","ENSG00000198727","ENSG00000198886","ENSG00000198786","ENSG00000198695","ENSG00000198700","ENSG00000210049","ENSG00000211459"}

def matrix_for(path):
    with gzip.open(path, "rt", newline="") as fh:
        reader = csv.reader(fh); header = next(reader); n = len(header)-1
        totals = np.zeros(n); detected = np.zeros(n, dtype=np.int32); mito = np.zeros(n)
        rows = []
        for row in reader:
            if not row: continue
            gene = row[0].upper(); vals = np.asarray(row[1:], dtype=np.float64)
            totals += vals; detected += vals > 0
            if gene in MITO or gene.startswith("MT-"): mito += vals
            rows.append((gene, vals))
    frac = mito / np.maximum(totals, 1)
    keep = (totals >= 500) & (detected >= 300) & (frac <= .20)
    sums = np.zeros(len(rows), dtype=np.float64)
    genes = []
    for gene, vals in rows:
        genes.append(gene); sums[len(genes)-1] = vals[keep].sum()
    return genes, sums

def main():
    all_genes = None; mats=[]; meta=[]
    for path in sorted(SRC.glob("*.csv.gz")):
        sample = path.name.replace(".csv.gz", "")
        genes, sums = matrix_for(path)
        if all_genes is None: all_genes = genes
        if genes != all_genes: raise ValueError(f"gene order mismatch: {sample}")
        q = QC.loc[sample]
        label = sample.split("_",1)[1]
        condition = "normal_skin" if label.startswith("NS-") else ("peripheral_normal" if label.startswith("PN-") else "psoriatic")
        donor = re.sub(r"^(NS|PN|PP)-", "", label)
        mats.append(sums); meta.append({"sample":sample,"condition":condition,"donor":donor,"filtered_cells":int(q.filtered_cells)})
    counts=np.vstack(mats); meta=pd.DataFrame(meta)
    np.savez_compressed(OUT/"gse173706_sample_gene_counts.npz", counts=counts, genes=np.array(all_genes))
    meta.to_csv(OUT/"gse173706_sample_metadata.csv",index=False)
    # Paired PN/PP donor-level classification only; NS is reserved for context.
    sub=meta[meta.condition.isin(["peripheral_normal","psoriatic"])].copy()
    paired=sub.groupby("donor").condition.nunique(); donors=paired[paired==2].index
    ix=sub.index[sub.donor.isin(donors)].to_numpy(); X=np.log1p(counts[ix]/np.maximum(counts[ix].sum(axis=1,keepdims=True),1)*1e6)
    # high-variance genes selected without using labels
    top=np.argsort(X.var(axis=0))[-min(2000,X.shape[1]):]; X=X[:,top]
    y=(sub.loc[ix,"condition"].to_numpy()=="psoriatic").astype(int); groups=sub.loc[ix,"donor"].to_numpy()
    records=[]
    for g in np.unique(groups):
        tr=groups!=g; te=~tr
        if len(np.unique(y[tr]))<2: continue
        model=make_pipeline(StandardScaler(), LogisticRegression(C=1,max_iter=2000,random_state=0))
        model.fit(X[tr],y[tr]); p=model.predict_proba(X[te])[:,1]
        records.append({"held_out_donor":g,"n_test":int(te.sum()),"auroc":float(roc_auc_score(y[te],p)) if len(np.unique(y[te]))>1 else np.nan,"balanced_accuracy":float(balanced_accuracy_score(y[te],p>.5))})
    pd.DataFrame(records).to_csv(OUT/"gse173706_paired_logistic_lodo.csv",index=False)
    summary={"samples":len(meta),"genes":len(all_genes),"paired_donors":sorted(map(str,donors)),"paired_samples":len(ix),"baseline_mean_lodo_auroc":float(np.nanmean([r["auroc"] for r in records])) if records else None}
    (OUT/"gse173706_donor_baseline_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))

if __name__=="__main__": main()

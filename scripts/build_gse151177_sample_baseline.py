"""Sample-level GSE151177 baseline; cells are aggregated before evaluation."""
from pathlib import Path
import gzip, json
import numpy as np, pandas as pd
from scipy.io import mmread
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]; SRC=ROOT/'data/raw/GSE151177_samples'; OUT=ROOT/'results/singlecell_v10'
QC=pd.read_csv(OUT/'gse151177_sample_qc.csv').set_index('sample')

def load(path):
    stem=path.name.replace('_matrix.mtx.gz','')
    with gzip.open(path,'rb') as fh: mat=mmread(fh).tocsr().astype(np.float32)
    feat=SRC/(stem+'_features.tsv.gz')
    with gzip.open(feat,'rt') as fh: genes=[x.rstrip('\n').split('\t')[0] for x in fh]
    total=np.asarray(mat.sum(axis=0)).ravel(); det=np.asarray((mat>0).sum(axis=0)).ravel()
    mito=np.array([g.upper().startswith('MT-') for g in genes])
    mt=np.asarray(mat[mito].sum(axis=0)).ravel()/np.maximum(total,1)
    keep=(total>=500)&(det>=300)&(mt<=.2)
    return stem, np.asarray(mat[:,keep].sum(axis=1)).ravel(), int(keep.sum())

def main():
    rows=[]; sums=[]; genes=None
    for p in sorted(SRC.glob('*_matrix.mtx.gz')):
        stem, s, kept=load(p); q=QC.loc[stem]
        # Sample-level exclusion is based on the pre-registered raw QC report.
        if float(q.median_mito_fraction)>.2 or float(q.median_detected_genes)<300: continue
        if genes is None:
            with gzip.open(SRC/(stem+'_features.tsv.gz'),'rt') as fh: genes=[x.rstrip('\n').split('\t')[0] for x in fh]
        cond='control' if 'Control' in stem else 'psoriasis'
        rows.append({'sample':stem,'condition':cond,'filtered_cells':kept}); sums.append(s)
    meta=pd.DataFrame(rows); counts=np.vstack(sums)
    np.savez_compressed(OUT/'gse151177_sample_gene_counts.npz',counts=counts,genes=np.array(genes)); meta.to_csv(OUT/'gse151177_sample_metadata.csv',index=False)
    X=np.log1p(counts/np.maximum(counts.sum(axis=1,keepdims=True),1)*1e6); top=np.argsort(X.var(axis=0))[-min(2000,X.shape[1]):]; X=X[:,top]
    y=(meta.condition=='psoriasis').astype(int).to_numpy(); rec=[]
    for i in range(len(y)):
        tr=np.arange(len(y))!=i
        if len(np.unique(y[tr]))<2: continue
        m=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000,random_state=0)); m.fit(X[tr],y[tr]); p=float(m.predict_proba(X[i:i+1])[:,1][0]); rec.append({'held_out_sample':meta.iloc[i]['sample'],'true':int(y[i]),'prob':p})
    r=pd.DataFrame(rec); r['pred']=(r.prob>=.5).astype(int); r.to_csv(OUT/'gse151177_sample_logistic_loo.csv',index=False)
    summary={'samples_used':len(meta),'controls':int((y==0).sum()),'psoriasis':int((y==1).sum()),'filtered_cells':int(meta.filtered_cells.sum()),'loo_auroc':float(roc_auc_score(r.true,r.prob)) if len(r) else None,'loo_balanced_accuracy':float(balanced_accuracy_score(r.true,r.pred)) if len(r) else None,'note':'sample-level feasibility baseline; variance filtering was global and is superseded by strict_multiresolution_benchmarks.json for formal claims'}
    (OUT/'gse151177_sample_baseline_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()

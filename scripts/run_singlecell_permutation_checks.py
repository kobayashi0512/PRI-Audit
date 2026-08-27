"""Permutation sanity checks for the two sample-level feasibility baselines."""
from pathlib import Path
import json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/singlecell_v10'
rng=np.random.default_rng(20260806)

def gse173706(n_perm=25):
    d=np.load(OUT/'gse173706_sample_gene_counts.npz'); X=np.log1p(d['counts']/np.maximum(d['counts'].sum(axis=1,keepdims=True),1)*1e6); top=np.argsort(X.var(axis=0))[-2000:]; X=StandardScaler().fit_transform(X[:,top])
    m=pd.read_csv(OUT/'gse173706_sample_metadata.csv'); sub=m[m.condition.isin(['peripheral_normal','psoriatic'])].copy(); ix=sub.index.to_numpy(); X=X[ix]; groups=sub.donor.to_numpy(); y0=(sub.condition.to_numpy()=='psoriatic').astype(int)
    vals=[]
    for _ in range(n_perm):
        y=rng.permutation(y0); pred=[]; truth=[]
        for g in np.unique(groups):
            tr=groups!=g; te=~tr
            if len(np.unique(y[tr]))<2: continue
            model=make_pipeline(LogisticRegression(C=1,max_iter=1000,random_state=0)); model.fit(X[tr],y[tr]); pred.extend(model.predict_proba(X[te])[:,1]); truth.extend(y[te])
        vals.append(roc_auc_score(truth,pred))
    return {'dataset':'GSE173706','n_perm':n_perm,'mean_auroc':float(np.mean(vals)),'p_ge_observed':float(np.mean(np.array(vals)>=1.0)),'max_auroc':float(np.max(vals))}

def gse151177(n_perm=25):
    d=np.load(OUT/'gse151177_sample_gene_counts.npz'); X=np.log1p(d['counts']/np.maximum(d['counts'].sum(axis=1,keepdims=True),1)*1e6); top=np.argsort(X.var(axis=0))[-2000:]; X=StandardScaler().fit_transform(X[:,top]); m=pd.read_csv(OUT/'gse151177_sample_metadata.csv'); y0=(m.condition=='psoriasis').astype(int).to_numpy(); vals=[]
    for _ in range(n_perm):
        y=rng.permutation(y0); pred=[]; truth=[]
        for i in range(len(y)):
            tr=np.arange(len(y))!=i
            if len(np.unique(y[tr]))<2: continue
            model=make_pipeline(LogisticRegression(C=1,max_iter=1000,random_state=0)); model.fit(X[tr],y[tr]); pred.append(model.predict_proba(X[i:i+1])[:,1][0]); truth.append(y[i])
        vals.append(roc_auc_score(truth,pred))
    return {'dataset':'GSE151177','n_perm':n_perm,'mean_auroc':float(np.mean(vals)),'p_ge_observed':float(np.mean(np.array(vals)>=1.0)),'max_auroc':float(np.max(vals))}

out=[gse173706(),gse151177()]; (OUT/'singlecell_permutation_checks.json').write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps(out,indent=2))

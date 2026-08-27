"""Shuffle cells across patients while preserving per-sample cell counts."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/singlecell_v10'; rng=np.random.default_rng(20260811)
d=np.load(OUT/'gse151177_patient_balanced_neighborhood_v1.npz'); H=d['histograms']; off=d['offsets']; samples=d['samples']; labels=d['labels']; sizes=np.diff(off); cell_labels=np.concatenate([np.full(n,i) for i,n in enumerate(sizes)])

def loo_auc(X):
    pred=[]
    for i in range(len(labels)):
        tr=np.arange(len(labels))!=i; m=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000,random_state=0)); m.fit(X[tr],labels[tr]); pred.append(m.predict_proba(X[i:i+1])[:,1][0])
    return roc_auc_score(labels,pred)

observed=loo_auc(H); null=[]
cell_clusters=np.concatenate([np.repeat(np.arange(H.shape[1])[None,:],0)]) if False else None
# Reconstruct integer cluster memberships from exact histogram counts.
clusters=[]
for i,n in enumerate(sizes):
    counts=np.rint(H[i]*n).astype(int); counts[np.argmax(counts)]+=n-counts.sum(); clusters.extend(np.repeat(np.arange(H.shape[1]),counts))
clusters=np.asarray(clusters)
for _ in range(500):
    z=rng.permutation(clusters); sh=np.zeros_like(H); start=0
    for i,n in enumerate(sizes): sh[i]=np.bincount(z[start:start+n],minlength=H.shape[1])/n; start+=n
    null.append(loo_auc(sh))
np.savez_compressed(OUT/'cell_cross_patient_shuffle_null_distribution.npz', observed=np.array([observed]), null_aurocs=np.asarray(null), seed=np.array([20260811]))
out={'observed_auroc':float(observed),'n_shuffle':len(null),'shuffle_mean':float(np.mean(null)),'shuffle_ci95':[float(np.percentile(null,2.5)),float(np.percentile(null,97.5))],'p_ge_observed':float((1+np.sum(np.array(null)>=observed))/(len(null)+1)),'null_distribution_path':'cell_cross_patient_shuffle_null_distribution.npz'}
(OUT/'cell_cross_patient_shuffle.json').write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps(out,indent=2))

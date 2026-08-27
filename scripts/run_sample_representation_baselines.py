"""PCA/NMF sample-level representation baselines for both single-cell cohorts."""
from pathlib import Path
import json
import numpy as np, pandas as pd
from sklearn.decomposition import PCA, NMF
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/singlecell_v10'

def fit_loo(X,y,groups=None):
    pred=[]; truth=[]
    splits=np.unique(groups) if groups is not None else np.arange(len(y))
    for s in splits:
        te=(groups==s) if groups is not None else (np.arange(len(y))==s); tr=~te
        if len(np.unique(y[tr]))<2: continue
        m=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000,random_state=0)); m.fit(X[tr],y[tr]); pred.extend(m.predict_proba(X[te])[:,1]); truth.extend(y[te])
    return float(roc_auc_score(truth,pred)) if len(set(truth))>1 else float('nan')

def one(stem, paired=False):
    d=np.load(OUT/f'{stem}_sample_gene_counts.npz'); counts=d['counts']; m=pd.read_csv(OUT/f'{stem}_sample_metadata.csv'); X=np.log1p(counts/np.maximum(counts.sum(axis=1,keepdims=True),1)*1e6); top=np.argsort(X.var(axis=0))[-min(2000,X.shape[1]):]; X=X[:,top]
    if paired:
        keep=m.condition.isin(['peripheral_normal','psoriatic']).to_numpy(); m=m[keep].reset_index(drop=True); X=X[keep]; y=(m.condition=='psoriatic').astype(int).to_numpy(); groups=m.donor.to_numpy()
    else:
        y=(m.condition=='psoriasis').astype(int).to_numpy(); groups=None
    pca=PCA(n_components=min(8,X.shape[0]-1,X.shape[1]),random_state=0).fit_transform(StandardScaler().fit_transform(X));
    nonneg=X-X.min()+1e-6; nmf=NMF(n_components=min(8,X.shape[0]-1,X.shape[1]),init='nndsvda',random_state=0,max_iter=500).fit_transform(nonneg)
    return {'dataset':stem,'n_samples':len(y),'pca_loo_auroc':fit_loo(pca,y,groups),'nmf_loo_auroc':fit_loo(nmf,y,groups)}

out=[one('gse173706',True),one('gse151177',False)]; (OUT/'singlecell_representation_baselines.json').write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps(out,indent=2))

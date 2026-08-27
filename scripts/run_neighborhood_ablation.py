"""Ablate neighborhood count and patient balancing on the pilot representation."""
from pathlib import Path
import json
import numpy as np, pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/singlecell_v10'
d=np.load(OUT/'gse151177_patient_balanced_neighborhood_v1.npz'); Z=d['pca']; off=d['offsets']; samples=d['samples']; meta=pd.read_csv(OUT/'gse151177_sample_metadata.csv').set_index('sample').loc[samples]; y=(meta.condition=='psoriasis').astype(int).to_numpy()
rows=[]
for k in [8,16,32]:
    km=MiniBatchKMeans(n_clusters=k,random_state=0,batch_size=2048,n_init=10).fit(Z); lab=km.labels_; H=np.zeros((len(samples),k));
    for i in range(len(samples)): H[i]=np.bincount(lab[off[i]:off[i+1]],minlength=k)/max(off[i+1]-off[i],1)
    for balanced in [True,False]:
        X=H if balanced else H*(off[1:]-off[:-1])[:,None]
        pred=[]; truth=[]
        for i in range(len(y)):
            tr=np.arange(len(y))!=i; m=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000,random_state=0)); m.fit(X[tr],y[tr]); pred.append(m.predict_proba(X[i:i+1])[:,1][0]); truth.append(y[i])
        rows.append({'clusters':k,'patient_balanced':balanced,'loo_auroc':float(roc_auc_score(truth,pred))})
out=pd.DataFrame(rows); out.to_csv(OUT/'patient_balanced_neighborhood_ablation.csv',index=False); print(out.to_string(index=False))

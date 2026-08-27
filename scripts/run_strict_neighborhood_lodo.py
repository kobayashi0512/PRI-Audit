"""Strict outer-LODO neighborhood evaluation using cached cell embeddings."""
from pathlib import Path
import json
import numpy as np, pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/singlecell_v10'
d=np.load(OUT/'gse151177_patient_balanced_neighborhood_v1.npz')
Z,off,samples=d['pca'],d['offsets'],d['samples']
meta=pd.read_csv(OUT/'gse151177_sample_metadata.csv').set_index('sample').loc[samples]
y=(meta.condition=='psoriasis').astype(int).to_numpy()

def hist(km,a,b):
    return np.bincount(km.predict(Z[a:b]),minlength=km.n_clusters)/max(b-a,1)

def run(k):
    pred=[]; truth=[]
    for i in range(len(y)):
        tr=np.arange(len(y))!=i
        train_cells=np.vstack([Z[off[j]:off[j+1]] for j in np.flatnonzero(tr)])
        km=MiniBatchKMeans(n_clusters=k,random_state=0,batch_size=2048,n_init=10).fit(train_cells)
        Htr=np.vstack([hist(km,off[j],off[j+1]) for j in np.flatnonzero(tr)])
        Hte=hist(km,off[i],off[i+1])[None,:]
        model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000,random_state=0))
        model.fit(Htr,y[tr]); pred.append(model.predict_proba(Hte)[:,1][0]); truth.append(y[i])
    return {'clusters':k,'strict_loo_auroc':float(roc_auc_score(truth,pred)),'strict_loo_balanced_accuracy':float(balanced_accuracy_score(truth,np.array(pred)>=.5))}

out=[run(k) for k in [8,16,32]]
(OUT/'strict_neighborhood_lodo_results.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))

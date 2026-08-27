"""Spatial-coordinate shuffle control for the GSE206391 6-NN baseline."""
from pathlib import Path
import json,h5py
import numpy as np,pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/singlecell_v10'; H5=ROOT/'data/raw/GSE206391_Preprocessed_data.h5'; rng=np.random.default_rng(20260811)
def dec(a): return np.array([x.decode() if isinstance(x,bytes) else str(x) for x in a])
with h5py.File(H5,'r') as f:
    X=csr_matrix((f['/X/data'][...],f['/X/indices'][...],f['/X/indptr'][...]),shape=(len(f['/obs/_index']),16685)); obs={k:f['/obs/'+k][...] for k in ['patient','biopsy_type','sample','array_row','array_col','in_tissue']}; cats={k:dec(f['/obs/__categories/'+k][...]) for k in ['patient','biopsy_type','sample']}; mt=f['/raw/var/mt'][...].astype(bool)
def cat(k): return cats[k][obs[k].astype(int)]
meta=pd.DataFrame({'patient':cat('patient'),'condition':cat('biopsy_type'),'sample':cat('sample'),'row':obs['array_row'],'col':obs['array_col'],'in_tissue':obs['in_tissue']}); tot=np.asarray(X.sum(axis=1)).ravel(); det=np.asarray((X>0).sum(axis=1)).ravel(); mtf=np.asarray(X[:,mt].sum(axis=1)).ravel()/np.maximum(tot,1); meta['keep']=(tot>=500)&(det>=300)&(mtf<=.2)&meta.in_tissue.astype(bool)
agg=np.load(OUT/'gse206391_sample_gene_counts.npz'); top=np.argsort(agg['counts'].var(axis=0))[-500:]; records=[]
for sample,g in meta.groupby('sample',sort=True):
    ix=g.index[g.keep].to_numpy(); A=X[ix][:,top].astype(float); A=A.multiply(1/np.maximum(np.asarray(A.sum(axis=1)).ravel(),1)[:,None]).toarray(); coords=g.loc[ix,['row','col']].to_numpy(); neigh=NearestNeighbors(n_neighbors=min(6,len(ix))).fit(coords).kneighbors(return_distance=False); records.append((sample,g.patient.iloc[0],g.condition.iloc[0],A,neigh))
patients=np.array([r[1] for r in records]); cond=np.array([r[2] for r in records]); paired=pd.Series(cond).groupby(pd.Series(patients)).nunique(); donor=paired[paired==2].index.to_numpy(); use=np.isin(patients,donor); y=(cond[use]=='LESIONAL').astype(int); groups=patients[use]
def evaluate(features):
    Z=np.vstack(features)[use]; pred=[];truth=[]
    for g in np.unique(groups):
        tr=groups!=g; te=~tr; m=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1500,random_state=0)); m.fit(Z[tr],y[tr]); pred.extend(m.predict_proba(Z[te])[:,1]); truth.extend(y[te])
    return roc_auc_score(truth,pred)
def feats(shuffle):
    out=[]
    for _,_,_,A,neigh in records:
        B=A[rng.permutation(len(A))] if shuffle else A; out.append(B[neigh].mean(axis=1).mean(axis=0))
    return out
observed=evaluate(feats(False)); null=[evaluate(feats(True)) for _ in range(100)]; np.savez_compressed(OUT/'spatial_coordinate_shuffle_null_distribution.npz', observed=np.array([observed]), null_aurocs=np.asarray(null), seed=np.array([20260811])); result={'observed_auroc':float(observed),'n_shuffle':len(null),'shuffle_mean':float(np.mean(null)),'shuffle_ci95':[float(np.percentile(null,2.5)),float(np.percentile(null,97.5))],'p_ge_observed':float((1+np.sum(np.array(null)>=observed))/(len(null)+1)),'null_distribution_path':'spatial_coordinate_shuffle_null_distribution.npz'}
(OUT/'spatial_coordinate_shuffle.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result,indent=2))

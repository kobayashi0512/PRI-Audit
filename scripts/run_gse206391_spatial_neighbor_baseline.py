"""Spatial kNN smoothing baseline on GSE206391, aggregated per sample."""
from pathlib import Path
import h5py, json
import numpy as np,pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/singlecell_v10'; H5=ROOT/'data/raw/GSE206391_Preprocessed_data.h5'
def dec(a): return np.array([x.decode() if isinstance(x,bytes) else str(x) for x in a])
with h5py.File(H5,'r') as f:
    X=csr_matrix((f['/X/data'][...],f['/X/indices'][...],f['/X/indptr'][...]),shape=(len(f['/obs/_index']),16685))
    obs={k:f['/obs/'+k][...] for k in ['patient','biopsy_type','sample','array_row','array_col','in_tissue']}
    cats={k:dec(f['/obs/__categories/'+k][...]) for k in ['patient','biopsy_type','sample']}
    mt=f['/raw/var/mt'][...].astype(bool)
def cat(k): return cats[k][obs[k].astype(int)]
meta=pd.DataFrame({'patient':cat('patient'),'condition':cat('biopsy_type'),'sample':cat('sample'),'row':obs['array_row'],'col':obs['array_col'],'in_tissue':obs['in_tissue']})
tot=np.asarray(X.sum(axis=1)).ravel(); det=np.asarray((X>0).sum(axis=1)).ravel(); mtfrac=np.asarray(X[:,mt].sum(axis=1)).ravel()/np.maximum(tot,1)
meta['keep']=(tot>=500)&(det>=300)&(mtfrac<=.2)&meta['in_tissue'].astype(bool)
agg=np.load(OUT/'gse206391_sample_gene_counts.npz'); top=np.argsort(agg['counts'].var(axis=0))[-500:]; rows=[]
for sample,g in meta.groupby('sample',sort=True):
    ix=g.index[g.keep].to_numpy(); coords=g.loc[ix,['row','col']].to_numpy(); A=X[ix][:,top].astype(float)
    A=A.multiply(1/np.maximum(np.asarray(A.sum(axis=1)).ravel(),1)[:,None]).toarray()
    if len(ix)>6:
        neigh=NearestNeighbors(n_neighbors=6).fit(coords).kneighbors(return_distance=False)
        F=np.asarray(A[neigh].mean(axis=1)).mean(axis=0)
    else: F=np.asarray(A.mean(axis=0)).ravel()
    rows.append({'sample':sample,'patient':g.patient.iloc[0],'condition':g.condition.iloc[0],**{f'g{i}':v for i,v in enumerate(F)}})
S=pd.DataFrame(rows); sub=S[S.condition.isin(['LESIONAL','NON LESIONAL'])].copy(); paired=sub.groupby('patient').condition.nunique(); donors=paired[paired==2].index; sub=sub[sub.patient.isin(donors)]
Z=sub.filter(regex='^g').to_numpy(); y=(sub.condition=='LESIONAL').astype(int).to_numpy(); groups=sub.patient.to_numpy(); pred=[]; truth=[]
for g in np.unique(groups):
    tr=groups!=g; te=~tr; m=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000,random_state=0)); m.fit(Z[tr],y[tr]); pred.extend(m.predict_proba(Z[te])[:,1]); truth.extend(y[te])
out={'samples':len(S),'paired_patients':len(donors),'features':Z.shape[1],'spatial_knn':6,'lodo_auroc':float(roc_auc_score(truth,pred)),'lodo_balanced_accuracy':float(balanced_accuracy_score(truth,np.array(pred)>=.5))}
(OUT/'gse206391_spatial_neighbor_summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps(out,indent=2))

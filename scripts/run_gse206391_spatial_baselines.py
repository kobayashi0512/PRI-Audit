"""Patient-paired baselines on the public GSE206391 processed H5 matrix."""
from pathlib import Path
import json
import h5py
import numpy as np,pandas as pd
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/singlecell_v10'; H5=ROOT/'data/raw/GSE206391_Preprocessed_data.h5'
def dec(a): return np.array([x.decode() if isinstance(x,bytes) else str(x) for x in a])
with h5py.File(H5,'r') as f:
    data=f['/X/data'][...]; indices=f['/X/indices'][...]; indptr=f['/X/indptr'][...]
    X=csr_matrix((data,indices,indptr),shape=(len(indptr)-1,16685))
    obs={k:f['/obs/'+k][...] for k in ['patient','biopsy_type','sample','specimen','array_row','array_col','in_tissue']}
    cats={k:dec(f['/obs/__categories/'+k][...]) for k in ['patient','biopsy_type','sample','specimen']}
    genes=dec(f['/raw/var/gene_id'][...]); gnames=dec(f['/raw/var/gene_name'][...]); mt=f['/raw/var/mt'][...].astype(bool)
def cat(name,key): return cats[key][obs[key].astype(int)]
meta=pd.DataFrame({'patient':cat('patient','patient'),'condition':cat('biopsy_type','biopsy_type'),'sample':cat('sample','sample'),'specimen':cat('specimen','specimen'),'row':obs['array_row'],'col':obs['array_col'],'in_tissue':obs['in_tissue']})
tot=np.asarray(X.sum(axis=1)).ravel(); det=np.asarray((X>0).sum(axis=1)).ravel(); mtfrac=np.asarray(X[:,mt].sum(axis=1)).ravel()/np.maximum(tot,1)
keep=(tot>=500)&(det>=300)&(mtfrac<=.2)&(meta.in_tissue.astype(bool).to_numpy()); meta['keep']=keep
# equal-weight sample summaries after spot QC
rows=[]; sums=[]
for sample,g in meta.groupby('sample',sort=True):
    ix=g.index[g.keep].to_numpy(); rows.append({'sample':sample,'patient':g.patient.iloc[0],'condition':g.condition.iloc[0],'raw_spots':len(g),'filtered_spots':len(ix)}); sums.append(np.asarray(X[ix].sum(axis=0)).ravel())
sm=pd.DataFrame(rows); C=np.vstack(sums); np.savez_compressed(OUT/'gse206391_sample_gene_counts.npz',counts=C,genes=genes); sm.to_csv(OUT/'gse206391_sample_metadata.csv',index=False)
# Paired LS/NL patient LODO on 2000 unsupervised high-variance genes.
sub=sm[sm.condition.isin(['LESIONAL','NON LESIONAL'])].copy(); paired=sub.groupby('patient').condition.nunique(); donors=paired[paired==2].index; ix=sub.index[sub.patient.isin(donors)].to_numpy(); Z=np.log1p(C[ix]/np.maximum(C[ix].sum(axis=1,keepdims=True),1)*1e6); top=np.argsort(Z.var(axis=0))[-2000:]; Z=Z[:,top]; y=(sub.loc[ix,'condition'].to_numpy()=='LESIONAL').astype(int); groups=sub.loc[ix,'patient'].to_numpy(); pred=[];truth=[]
for g in np.unique(groups):
    tr=groups!=g; te=~tr; m=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000,random_state=0)); m.fit(Z[tr],y[tr]); pred.extend(m.predict_proba(Z[te])[:,1]); truth.extend(y[te])
out={'spots':int(X.shape[0]),'genes':int(X.shape[1]),'samples':len(sm),'filtered_spots':int(keep.sum()),'paired_patients':int(len(donors)),'paired_samples':int(len(ix)),'sample_expression_lodo_auroc':float(roc_auc_score(truth,pred)),'sample_expression_lodo_balanced_accuracy':float(balanced_accuracy_score(truth,np.array(pred)>=.5))}
np.savez_compressed(OUT/'gse206391_spatial_expression_lodo_arrays.npz',labels=np.array(truth),probabilities=np.array(pred))
(OUT/'gse206391_spatial_baseline_summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps(out,indent=2))

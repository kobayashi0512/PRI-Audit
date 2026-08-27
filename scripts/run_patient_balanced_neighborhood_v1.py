"""Exploratory patient-balanced cell-neighborhood representation.

Each sample contributes an equal-mass histogram over shared cell-state
neighborhoods. This prevents samples with more captured cells from dominating.
The current run is a feasibility experiment; final claims require fold-specific
embedding/clustering and an independent cohort.
"""
from pathlib import Path
import gzip, json
import numpy as np, pandas as pd
from scipy.io import mmread
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib

ROOT=Path(__file__).resolve().parents[1]; SRC=ROOT/'data/raw/GSE151177_samples'; OUT=ROOT/'results/singlecell_v10'
QC=pd.read_csv(OUT/'gse151177_sample_qc.csv').set_index('sample')
rng=np.random.default_rng(20260806)

def load_cells(path, gene_idx, max_cells=1500):
    stem=path.name.replace('_matrix.mtx.gz','')
    with gzip.open(path,'rb') as fh: mat=mmread(fh).tocsr().astype(np.float32)
    totals=np.asarray(mat.sum(axis=0)).ravel(); det=np.asarray((mat>0).sum(axis=0)).ravel()
    with gzip.open(SRC/(stem+'_features.tsv.gz'),'rt') as fh: genes=[x.rstrip('\n').split('\t')[0] for x in fh]
    mito=np.array([g.upper().startswith('MT-') for g in genes]); mt=np.asarray(mat[mito].sum(axis=0)).ravel()/np.maximum(totals,1)
    keep=np.flatnonzero((totals>=500)&(det>=300)&(mt<=.2))
    if len(keep)>max_cells: keep=rng.choice(keep,max_cells,replace=False)
    x=mat[gene_idx][:,keep].T.toarray(); x=np.log1p(x/np.maximum(x.sum(axis=1,keepdims=True),1)*1e4)
    return stem,x

def main():
    good=[]
    for p in sorted(SRC.glob('*_matrix.mtx.gz')):
        stem=p.name.replace('_matrix.mtx.gz',''); q=QC.loc[stem]
        if float(q.median_mito_fraction)<=.2 and float(q.median_detected_genes)>=300: good.append(p)
    bulk=np.load(OUT/'gse151177_sample_gene_counts.npz'); genes=list(bulk['genes']); top=np.argsort(bulk['counts'].var(axis=0))[-1000:]
    allx=[]; sample_names=[]; slices=[]
    for p in good:
        stem,x=load_cells(p,top); slices.append((len(allx),len(allx)+len(x))); allx.append(x); sample_names.append(stem)
    X=np.vstack(allx); scaler=StandardScaler().fit(X); pca_model=PCA(n_components=15,random_state=0).fit(scaler.transform(X)); pca=pca_model.transform(scaler.transform(X)); km=MiniBatchKMeans(n_clusters=16,random_state=0,batch_size=2048,n_init=10).fit(pca); lab=km.labels_
    H=np.zeros((len(good),16),float); start=0
    for i,x in enumerate(allx):
        z=lab[start:start+len(x)]; H[i]=np.bincount(z,minlength=16)/max(len(z),1); start+=len(x)
    meta=pd.DataFrame({'sample':sample_names}); meta['condition']=meta['sample'].str.contains('Psoriasis').map({True:'psoriasis',False:'control'}); y=(meta['condition']=='psoriasis').astype(int).to_numpy(); pred=[]; truth=[]
    for i in range(len(y)):
        tr=np.arange(len(y))!=i
        m=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000,random_state=0)); m.fit(H[tr],y[tr]); pred.append(m.predict_proba(H[i:i+1])[:,1][0]); truth.append(y[i])
    offsets=np.cumsum([0]+[len(x) for x in allx])
    np.savez_compressed(OUT/'gse151177_patient_balanced_neighborhood_v1.npz',histograms=H,labels=y,samples=np.array(sample_names),pca=pca,offsets=offsets)
    joblib.dump({'gene_idx':top,'scaler':scaler,'pca':pca_model,'kmeans':km},OUT/'patient_balanced_neighborhood_v1_models.joblib')
    out={'samples':len(good),'cells_used':int(len(X)),'clusters':16,'loo_auroc':float(roc_auc_score(truth,pred)),'loo_balanced_accuracy':float(balanced_accuracy_score(truth,np.array(pred)>=.5)),'representation':'equal-mass per-sample histogram over PCA/KMeans cell neighborhoods','status':'exploratory; clustering/embedding currently unsupervised on all samples'}
    (OUT/'patient_balanced_neighborhood_v1_summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps(out,indent=2))

if __name__=='__main__': main()

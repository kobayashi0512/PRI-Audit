"""TRIAD-Flow v1: trigger + cell-state + local-neighborhood entropic OT."""
from pathlib import Path
import gzip,json,joblib
import numpy as np,pandas as pd
from scipy.io import mmread
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]; SRC=ROOT/'data/raw/GSE151177_samples'; OUT=ROOT/'results/singlecell_v10'; RNG=np.random.default_rng(20260810)
QC=pd.read_csv(OUT/'gse151177_sample_qc.csv').set_index('sample'); dict_df=pd.read_parquet(ROOT/'results/trigger_v8/v8_perturbation_dictionary.parquet'); dict_genes=pd.read_csv(ROOT/'results/trigger_v8/v8_common_genes.csv')['gene'].astype(str).str.upper().to_numpy(); D=dict_df[['IL17','TNF_high','IL17_TNF_synergy']].to_numpy(float); state=joblib.load(OUT/'patient_balanced_neighborhood_v1_models.joblib'); hvg=np.asarray(state['gene_idx'])

def load_sample(p,max_cells=240):
    stem=p.name.replace('_matrix.mtx.gz','')
    with gzip.open(p,'rb') as fh: mat=mmread(fh).tocsr().astype(np.float32)
    with gzip.open(SRC/(stem+'_features.tsv.gz'),'rt') as fh: features=[x.rstrip('\n').split('\t') for x in fh]
    symbols=np.array([(x[1] if len(x)>1 else x[0]).upper() for x in features]); lookup={g:i for i,g in enumerate(symbols)}; pairs=[(lookup[g],j) for j,g in enumerate(dict_genes) if g in lookup]; gi=np.array([x[0] for x in pairs]); dj=np.array([x[1] for x in pairs]); total=np.asarray(mat.sum(axis=0)).ravel(); det=np.asarray((mat>0).sum(axis=0)).ravel(); mito=np.array([g.startswith('MT-') for g in symbols]); mt=np.asarray(mat[mito].sum(axis=0)).ravel()/np.maximum(total,1); keep=np.flatnonzero((total>=500)&(det>=300)&(mt<=.2));
    if len(keep)>max_cells: keep=RNG.choice(keep,max_cells,replace=False)
    x=mat[:,keep]; norm=np.log1p(x.multiply(1/np.maximum(np.asarray(x.sum(axis=0)).ravel(),1)*1e4)).tocsr(); A=norm[gi].T.toarray() @ D[dj]; Z=state['pca'].transform(state['scaler'].transform(norm[hvg].T.toarray())); neigh=NearestNeighbors(n_neighbors=min(10,len(Z))).fit(Z).kneighbors(return_distance=False); local=A[neigh].mean(axis=1)-A; return stem,np.hstack([Z,A,local])

def sinkhorn(A,B,eps=.5,n_iter=150):
    C=((A[:,None,:]-B[None,:,:])**2).sum(axis=2); C=C/(np.median(C[C>0])+1e-8); K=np.exp(-C/eps); a=np.full(len(A),1/len(A)); b=np.full(len(B),1/len(B)); u=np.ones_like(a); v=np.ones_like(b)
    for _ in range(n_iter): u=a/np.maximum(K@v,1e-12); v=b/np.maximum(K.T@u,1e-12)
    P=u[:,None]*K*v[None,:]; P=P/np.maximum(P.sum(),1e-12); flux=(P[:,:,None]*(B[None,:,:]-A[:,None,:])).sum(axis=(0,1)); return float((P*C).sum()),flux

def main():
    meta=pd.read_csv(OUT/'gse151177_sample_metadata.csv'); cells={}; good=[]
    for p in sorted(SRC.glob('*_matrix.mtx.gz')):
        stem=p.name.replace('_matrix.mtx.gz',''); q=QC.loc[stem]
        if float(q.median_mito_fraction)<=.2 and float(q.median_detected_genes)>=300: s,F=load_sample(p); good.append(s); cells[s]=F
    meta=meta[meta['sample'].isin(good)].reset_index(drop=True); y=(meta.condition=='psoriasis').astype(int).to_numpy(); pred=[]; truth=[]; rows=[]
    for ti in range(len(meta)):
        tr=np.arange(len(meta))!=ti; ref=np.vstack([cells[meta.iloc[j]['sample']] for j in np.flatnonzero(tr&(y==0))]); mu=ref.mean(axis=0); sd=ref.std(axis=0)+1e-6; ref=(ref-mu)/sd; feats=[]
        for j in np.flatnonzero(tr): cost,flux=sinkhorn((cells[meta.iloc[j]['sample']]-mu)/sd,ref); feats.append([cost,*flux[15:]])
        cost,flux=sinkhorn((cells[meta.iloc[ti]['sample']]-mu)/sd,ref); m=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000,random_state=0)); m.fit(np.asarray(feats),y[tr]); pred.append(float(m.predict_proba(np.array([[cost,*flux[15:]]]))[:,1][0])); truth.append(int(y[ti])); rows.append({'held_out_sample':meta.iloc[ti]['sample'],'true':int(y[ti]),'cost':cost,'flux_trigger_norm':float(np.linalg.norm(flux[15:]))})
    pd.DataFrame(rows).to_csv(OUT/'triad_trigger_transport_v1_lodo.csv',index=False); out={'samples':len(meta),'cells_per_sample':240,'feature_dimensions':'15 PCA + 3 trigger + 3 local trigger','loo_auroc':float(roc_auc_score(truth,pred)),'loo_balanced_accuracy':float(balanced_accuracy_score(truth,np.asarray(pred)>=.5)),'note':'exploratory v1; state PCA is frozen from the feasibility embedding'}; (OUT/'triad_trigger_transport_v1_summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()

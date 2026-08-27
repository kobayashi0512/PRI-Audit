"""TRIAD-Flow v0: trigger-informed entropic OT on GSE151177 cells.

This is the first executable transport component. It is intentionally limited
to trigger coordinates (IL17/TNF/synergy) so that its contribution is
interpretable before adding graph/GW terms.
"""
from pathlib import Path
import gzip,json
import numpy as np,pandas as pd
from scipy.io import mmread
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]; SRC=ROOT/'data/raw/GSE151177_samples'; OUT=ROOT/'results/singlecell_v10'; RNG=np.random.default_rng(20260810)
QC=pd.read_csv(OUT/'gse151177_sample_qc.csv').set_index('sample')
dict_df=pd.read_parquet(ROOT/'results/trigger_v8/v8_perturbation_dictionary.parquet')
dict_genes=pd.read_csv(ROOT/'results/trigger_v8/v8_common_genes.csv')['gene'].astype(str).str.upper().to_numpy(); D=dict_df[['IL17','TNF_high','IL17_TNF_synergy']].to_numpy(float)

def load_sample(p,max_cells=240):
    stem=p.name.replace('_matrix.mtx.gz','')
    with gzip.open(p,'rb') as fh: mat=mmread(fh).tocsr().astype(np.float32)
    with gzip.open(SRC/(stem+'_features.tsv.gz'),'rt') as fh:
        features=[x.rstrip('\n').split('\t') for x in fh]
    symbols=np.array([(x[1] if len(x)>1 else x[0]).upper() for x in features])
    lookup={g:i for i,g in enumerate(symbols)}; pairs=[(lookup[g],j) for j,g in enumerate(dict_genes) if g in lookup]
    if len(pairs)<1000: raise RuntimeError(f'{stem}: only {len(pairs)} trigger genes mapped')
    gi=np.array([x[0] for x in pairs]); dj=np.array([x[1] for x in pairs]); total=np.asarray(mat.sum(axis=0)).ravel(); det=np.asarray((mat>0).sum(axis=0)).ravel(); mito=np.array([g.startswith('MT-') for g in symbols]); mt=np.asarray(mat[mito].sum(axis=0)).ravel()/np.maximum(total,1); keep=np.flatnonzero((total>=500)&(det>=300)&(mt<=.2))
    if len(keep)>max_cells: keep=RNG.choice(keep,max_cells,replace=False)
    x=mat[gi][:,keep].T.toarray(); x=np.log1p(x/np.maximum(x.sum(axis=1,keepdims=True),1)*1e4); A=x[:,np.arange(len(dj))] @ D[dj]
    return stem,A

def sinkhorn(A,B,eps=.5,n_iter=150):
    C=((A[:,None,:]-B[None,:,:])**2).sum(axis=2); C=C/(np.median(C[C>0])+1e-8); K=np.exp(-C/eps); a=np.full(len(A),1/len(A)); b=np.full(len(B),1/len(B)); u=np.ones_like(a); v=np.ones_like(b)
    for _ in range(n_iter): u=a/np.maximum(K@v,1e-12); v=b/np.maximum(K.T@u,1e-12)
    P=u[:,None]*K*v[None,:]; P=P/np.maximum(P.sum(),1e-12); flux=(P[:,:,None]*(B[None,:,:]-A[:,None,:])).sum(axis=(0,1)); return float((P*C).sum()),flux

def main():
    meta=pd.read_csv(OUT/'gse151177_sample_metadata.csv'); good=[]; cells={}
    for p in sorted(SRC.glob('*_matrix.mtx.gz')):
        stem=p.name.replace('_matrix.mtx.gz',''); q=QC.loc[stem]
        if float(q.median_mito_fraction)<=.2 and float(q.median_detected_genes)>=300:
            s,A=load_sample(p); good.append(s); cells[s]=A
    meta=meta[meta['sample'].isin(good)].reset_index(drop=True); y=(meta.condition=='psoriasis').astype(int).to_numpy(); predictions=[]; truth=[]; fold_rows=[]
    for test_i in range(len(meta)):
        train_i=np.arange(len(meta))!=test_i; train_controls=[cells[meta.iloc[j]['sample']] for j in np.flatnonzero(train_i & (y==0))];
        if not train_controls: continue
        ref=np.vstack(train_controls); train_all=[]; test_feat=None
        # Fold-specific score scaling learned from training cells.
        mu=ref.mean(axis=0); sd=ref.std(axis=0)+1e-6; ref=(ref-mu)/sd
        for j in np.flatnonzero(train_i):
            cost,flux=sinkhorn((cells[meta.iloc[j]['sample']]-mu)/sd,ref); train_all.append([cost,*flux])
        test_cost,test_flux=sinkhorn((cells[meta.iloc[test_i]['sample']]-mu)/sd,ref); test_feat=np.array([[test_cost,*test_flux]])
        model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000,random_state=0)); model.fit(np.asarray(train_all),y[train_i]); predictions.append(float(model.predict_proba(test_feat)[:,1][0])); truth.append(int(y[test_i])); fold_rows.append({'held_out_sample':meta.iloc[test_i]['sample'],'true':int(y[test_i]),'cost':test_cost,'flux_il17':test_flux[0],'flux_tnf':test_flux[1],'flux_synergy':test_flux[2]})
    result=pd.DataFrame(fold_rows); result.to_csv(OUT/'triad_trigger_transport_v0_lodo.csv',index=False); out={'samples':len(meta),'cells_per_sample':240,'mapped_trigger_genes':int(len(dict_genes)),'loo_auroc':float(roc_auc_score(truth,predictions)),'loo_balanced_accuracy':float(balanced_accuracy_score(truth,np.asarray(predictions)>=.5)),'note':'trigger-only entropic OT; graph/GW terms not yet included'}; (OUT/'triad_trigger_transport_v0_summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()

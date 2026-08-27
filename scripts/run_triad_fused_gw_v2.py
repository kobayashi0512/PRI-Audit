"""TRIAD-Flow v2: entropic fused Gromov-Wasserstein feasibility run."""
from pathlib import Path
import json
import sys
import warnings
import numpy as np,pandas as pd,ot
from sklearn.metrics import pairwise_distances,roc_auc_score,balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_triad_trigger_transport_v1 import load_sample, QC, SRC, OUT

def one_transport(A,B):
    # A/B contain state, trigger and local-trigger features; state/trigger
    # distance is the feature term and state distance is the relational term.
    A=(A-A.mean(0))/(A.std(0)+1e-6); B=(B-B.mean(0))/(B.std(0)+1e-6)
    M=pairwise_distances(A,B,metric='sqeuclidean'); C1=pairwise_distances(A,A,metric='euclidean'); C2=pairwise_distances(B,B,metric='euclidean')
    def scale(C):
        pos=C[C>0]; return C/(np.median(pos)+1e-8)
    M,C1,C2=scale(M),scale(C1),scale(C2); p=np.full(len(A),1/len(A)); q=np.full(len(B),1/len(B))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        T=ot.gromov.entropic_fused_gromov_wasserstein(M,C1,C2,p,q,alpha=.5,epsilon=1.0,max_iter=60,tol=1e-5,solver='PGD')
    if not np.isfinite(T).all() or T.sum()<=0: raise RuntimeError('GW transport did not produce a finite coupling')
    flux=(T[:,:,None]*(B[None,:,:]-A[:,None,:])).sum((0,1)); return float((T*M).sum()),flux

def main():
    meta=pd.read_csv(OUT/'gse151177_sample_metadata.csv'); cells={}; good=[]
    for p in sorted(SRC.glob('*_matrix.mtx.gz')):
        stem=p.name.replace('_matrix.mtx.gz',''); q=QC.loc[stem]
        if float(q.median_mito_fraction)<=.2 and float(q.median_detected_genes)>=300:
            s,F=load_sample(p,max_cells=40); good.append(s); cells[s]=F
    meta=meta[meta['sample'].isin(good)].reset_index(drop=True); y=(meta.condition=='psoriasis').astype(int).to_numpy(); pred=[];truth=[]; rows=[]
    for ti in range(len(meta)):
        tr=np.arange(len(meta))!=ti; controls=[cells[meta.iloc[j]['sample']] for j in np.flatnonzero(tr&(y==0))]; ref=np.vstack(controls); feats=[]
        for j in np.flatnonzero(tr):
            c,f=one_transport(cells[meta.iloc[j]['sample']],ref); feats.append([c,*f[15:]])
        c,f=one_transport(cells[meta.iloc[ti]['sample']],ref); model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000,random_state=0)); model.fit(np.asarray(feats),y[tr]); pred.append(float(model.predict_proba(np.array([[c,*f[15:]]]))[:,1][0])); truth.append(int(y[ti])); rows.append({'held_out_sample':meta.iloc[ti]['sample'],'true':int(y[ti]),'transport_cost':c,'trigger_flux_norm':float(np.linalg.norm(f[15:]))})
    pd.DataFrame(rows).to_csv(OUT/'triad_fused_gw_v2_lodo.csv',index=False); out={'samples':len(meta),'cells_per_sample':40,'alpha':.5,'epsilon':1.0,'loo_auroc':float(roc_auc_score(truth,pred)),'loo_balanced_accuracy':float(balanced_accuracy_score(truth,np.asarray(pred)>=.5)),'note':'entropic fused GW feasibility; equal cell counts approximate patient balancing'}; (OUT/'triad_fused_gw_v2_summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()

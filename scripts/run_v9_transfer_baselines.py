"""Expanded v9 transfer baselines on 191 public LS/NL pairs."""
from pathlib import Path
import hashlib
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.linear_model import Ridge

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results'/'trigger_v9'; V8=ROOT/'results'/'trigger_v8'
def block(g): return int(hashlib.sha256(g.encode()).hexdigest()[:8],16)%5
def cos(a,b):
 d=np.linalg.norm(a)*np.linalg.norm(b); return float(a@b/d) if d else np.nan
def fit_group(a,y,groups,eta,ridge):
 z=np.column_stack([np.ones(len(a)),a]); cs=sorted(np.unique(groups)); w=np.array([1/np.sum(groups==c) for c in groups])
 for _ in range(20):
  coef=np.linalg.solve(z.T@(z*w[:,None])+np.diag([0,ridge,ridge]),z.T@(y*w[:,None])); resid=(z@coef-y)**2; loss=np.array([resid[groups==c].mean() for c in cs]); gw=np.exp(np.clip(eta*(loss-loss.mean())/(loss.std()+1e-8),-5,5)); gw/=gw.mean(); w=np.array([gw[cs.index(c)]/np.sum(groups==c) for c in groups])
 return coef
def main():
 parser=argparse.ArgumentParser(); parser.add_argument('--eta',type=float,default=.5); parser.add_argument('--ridge',type=float,default=100.); args=parser.parse_args()
 x=pd.read_parquet(OUT/'v9_expanded_ls_minus_nl_rank.parquet').to_numpy(float); genes=pd.read_csv(OUT/'v9_common_genes.csv').gene.tolist(); blocks=np.array([block(g) for g in genes]); m=pd.read_csv(OUT/'v9_expanded_metadata.csv')
 b0=pd.read_parquet(V8/'v8_perturbation_dictionary.parquet'); g0=pd.read_csv(V8/'v8_common_genes.csv').gene.tolist(); pos={g:i for i,g in enumerate(g0)}; b=b0.loc[[pos[g] for g in genes],['IL17','TNF_high']].to_numpy(float)
 rows=[]
 for held in sorted(m.cohort.unique()):
  tr_i=np.flatnonzero(m.cohort.ne(held)); te_i=np.flatnonzero(m.cohort.eq(held)); groups=m.cohort.iloc[tr_i].to_numpy()
  for hb in range(5):
   tr,te=blocks!=hb,blocks==hb; atr=np.vstack([nnls(b[tr],x[i,tr])[0] for i in tr_i]); ate=np.vstack([nnls(b[tr],x[i,tr])[0] for i in te_i])
   for method,eta in [('pooled',-1),('balanced',0),('groupdro',args.eta)]:
    if method=='pooled':
     model=Ridge(alpha=1).fit(atr,x[tr_i][:,te]); pred=model.intercept_ + ate@model.coef_.T
    else: pred=np.column_stack([np.ones(len(ate)),ate])@fit_group(atr,x[tr_i][:,te],groups,eta,args.ridge)
    for i,p in zip(te_i,pred): rows.append({'cohort':held,'subject':m.subject.iloc[i],'block':hb,'method':method,'cosine':cos(x[i,te],p)})
 frame=pd.DataFrame(rows); frame.to_csv(OUT/f'v9_transfer_baselines_ridge{args.ridge:g}_eta{args.eta:g}.csv',index=False); print(frame.groupby(['cohort','method']).cosine.mean().unstack().round(4).to_string()); print('\nmeans\n'+frame.groupby('method').cosine.mean().round(4).to_string())
if __name__=='__main__': main()

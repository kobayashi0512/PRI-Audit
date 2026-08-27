"""Nested alpha tuning for the expanded pooled transfer baseline."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.linear_model import Ridge
import argparse
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results'/'trigger_v9'; V8=ROOT/'results'/'trigger_v8'; ALPHAS=(0.01,.1,1.,10.,100.)
def block(g): return int(hashlib.sha256(g.encode()).hexdigest()[:8],16)%5
def cos(a,b):
 d=np.linalg.norm(a)*np.linalg.norm(b); return float(a@b/d) if d else np.nan
def load():
 x=pd.read_parquet(OUT/'v9_expanded_ls_minus_nl_rank.parquet').to_numpy(float); genes=pd.read_csv(OUT/'v9_common_genes.csv').gene.tolist(); blocks=np.array([block(g) for g in genes]); m=pd.read_csv(OUT/'v9_expanded_metadata.csv'); b0=pd.read_parquet(V8/'v8_perturbation_dictionary.parquet'); g0=pd.read_csv(V8/'v8_common_genes.csv').gene.tolist(); pos={g:i for i,g in enumerate(g0)}; b=b0.loc[[pos[g] for g in genes],['IL17','TNF_high']].to_numpy(float); return x,blocks,m,b
def score(x,blocks,m,b,train_i,test_i,alpha):
 vals=[]
 for hb in range(5):
  tr,te=blocks!=hb,blocks==hb; atr=np.vstack([nnls(b[tr],x[i,tr])[0] for i in train_i]); ate=np.vstack([nnls(b[tr],x[i,tr])[0] for i in test_i]); model=Ridge(alpha=alpha).fit(atr,x[train_i][:,te]); pred=model.intercept_+ate@model.coef_.T
  vals.extend(cos(x[i,te],p) for i,p in zip(test_i,pred))
 return float(np.mean(vals))
def main():
 parser=argparse.ArgumentParser(); parser.add_argument('--outer'); args=parser.parse_args()
 x,blocks,m,b=load(); rows=[]
 outers=[args.outer] if args.outer else sorted(m.cohort.unique())
 for outer in outers:
  train_cohorts=[c for c in sorted(m.cohort.unique()) if c!=outer]; tuning=[]
  for alpha in ALPHAS:
   vals=[]
   for inner in train_cohorts:
    fit=np.flatnonzero(m.cohort.isin([c for c in train_cohorts if c!=inner])); test=np.flatnonzero(m.cohort.eq(inner)); vals.append(score(x,blocks,m,b,fit,test,alpha))
   tuning.append((alpha,float(np.mean(vals))))
  alpha,best=max(tuning,key=lambda z:z[1]); test=np.flatnonzero(m.cohort.eq(outer)); fit=np.flatnonzero(m.cohort.isin(train_cohorts)); rows.append({'cohort':outer,'alpha':alpha,'inner':best,'held_gene_cosine':score(x,blocks,m,b,fit,test,alpha)})
  pd.DataFrame(rows).to_csv(OUT/f'v9_tuned_pooled_checkpoint_{outer}.csv',index=False)
 frame=pd.DataFrame(rows); frame.to_csv(OUT/'v9_tuned_pooled_results.csv',index=False); print(frame.round(4).to_string(index=False)); print('mean',frame.held_gene_cosine.mean())
if __name__=='__main__': main()

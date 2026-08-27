"""Group-robust perturbation-activity to gene transfer on blocked genes."""

from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from scipy.optimize import nnls
import argparse

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results'/'trigger_v8'
def block(g): return int(hashlib.sha256(g.encode()).hexdigest()[:8],16)%5
def cos(a,b):
 d=np.linalg.norm(a)*np.linalg.norm(b); return float(a@b/d) if d else np.nan

def fit_groupdro(a,y,groups,eta=0.5,steps=20,ridge=1.0):
 z=np.column_stack([np.ones(len(a)),a]); cohorts=sorted(np.unique(groups)); w=np.array([1.0/np.sum(groups==c) for c in groups])
 for _ in range(steps):
  zw=z*w[:,None]; lhs=z.T@zw + ridge*np.diag([0.,1.,1.]); coef=np.linalg.solve(lhs,z.T@(y*w[:,None]))
  resid=(z@coef-y)**2
  loss=np.array([resid[groups==c].mean() for c in cohorts])
  exponent=np.clip(eta*(loss-loss.mean())/(loss.std()+1e-8),-5,5)
  group_weight=np.exp(exponent); group_weight/=group_weight.mean()
  w=np.array([group_weight[cohorts.index(c)]/np.sum(groups==c) for c in groups])
 return coef

def main():
 parser=argparse.ArgumentParser(); parser.add_argument('--eta',type=float,default=0.5); args=parser.parse_args()
 x=pd.read_parquet(OUT/'v8_clinical_ls_minus_nl_rank.parquet').to_numpy(float)
 b=pd.read_parquet(OUT/'v8_perturbation_dictionary.parquet').to_numpy(float)[:,:2]
 genes=pd.read_csv(OUT/'v8_common_genes.csv').gene.tolist(); blocks=np.array([block(g) for g in genes])
 m=pd.read_csv(OUT/'v8_clinical_ls_minus_nl_metadata.csv'); rows=[]
 for held in sorted(m.cohort.unique()):
  train_idx=np.flatnonzero(m.cohort.ne(held)); test_idx=np.flatnonzero(m.cohort.eq(held)); groups=m.cohort.iloc[train_idx].to_numpy()
  for hb in range(5):
   tr,te=blocks!=hb,blocks==hb
   Atrain=np.vstack([nnls(b[tr],x[i,tr])[0] for i in train_idx]); Atest=np.vstack([nnls(b[tr],x[i,tr])[0] for i in test_idx])
   coef=fit_groupdro(Atrain,x[train_idx][:,te],groups,eta=args.eta)
   pred=np.column_stack([np.ones(len(Atest)),Atest])@coef
   for i,score in zip(test_idx,pred): rows.append({'cohort':held,'subject':m.subject.iloc[i],'gene_block':hb,'held_gene_cosine':cos(x[i,te],score)})
 frame=pd.DataFrame(rows); frame.to_csv(OUT/f'v8_groupdro_transfer_eta{args.eta:g}.csv',index=False)
 print(frame.groupby('cohort').held_gene_cosine.mean().round(4).to_string())

if __name__=='__main__': main()

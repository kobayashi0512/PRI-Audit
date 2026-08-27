"""A transparent activity-to-gene transfer baseline for v8 blocked genes."""

from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.linear_model import Ridge

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results'/'trigger_v8'
def block(g): return int(hashlib.sha256(g.encode()).hexdigest()[:8],16)%5
def cos(a,b):
 d=np.linalg.norm(a)*np.linalg.norm(b); return float(a@b/d) if d else np.nan

def main():
 x=pd.read_parquet(OUT/'v8_clinical_ls_minus_nl_rank.parquet').to_numpy(float)
 b=pd.read_parquet(OUT/'v8_perturbation_dictionary.parquet').to_numpy(float)[:,:2]
 genes=pd.read_csv(OUT/'v8_common_genes.csv').gene.tolist(); blocks=np.array([block(g) for g in genes])
 m=pd.read_csv(OUT/'v8_clinical_ls_minus_nl_metadata.csv'); rows=[]
 for held in sorted(m.cohort.unique()):
  train_idx=np.flatnonzero(m.cohort.ne(held)); test_idx=np.flatnonzero(m.cohort.eq(held))
  for hb in range(5):
   tr,te=blocks!=hb,blocks==hb
   Atrain=np.vstack([nnls(b[tr],x[i,tr])[0] for i in train_idx])
   Atest=np.vstack([nnls(b[tr],x[i,tr])[0] for i in test_idx])
   pred=np.zeros((len(test_idx),te.sum()))
   for j,gene in enumerate(np.flatnonzero(te)):
    model=Ridge(alpha=1.0).fit(Atrain,x[train_idx,gene]); pred[:,j]=model.predict(Atest)
   for i,score in zip(test_idx,pred): rows.append({'cohort':held,'subject':m.subject.iloc[i],'gene_block':hb,'held_gene_cosine':cos(x[i,te],score)})
 frame=pd.DataFrame(rows); frame.to_csv(OUT/'v8_transfer_baseline.csv',index=False)
 print(frame.groupby('cohort').held_gene_cosine.mean().round(4).to_string())

if __name__=='__main__': main()

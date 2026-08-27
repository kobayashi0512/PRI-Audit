"""Bootstrap and permutation checks for fixed external neighborhood transfer."""
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/singlecell_v10'; d=np.load(OUT/'gse173706_neighborhood_transfer_arrays.npz',allow_pickle=True); y=d['labels']; p=d['probabilities']; donors=d['donors'].astype(str); unique_donors=np.unique(donors); rng=np.random.default_rng(20260809)
boot=[]
for _ in range(2000):
    sampled=rng.choice(unique_donors,size=len(unique_donors),replace=True); ix=np.concatenate([np.flatnonzero(donors==g) for g in sampled]); boot.append(roc_auc_score(y[ix],p[ix]) if len(np.unique(y[ix]))==2 else np.nan)
perm=[]
for _ in range(2000):
    shuffled=y.copy()
    for g in unique_donors:
        if rng.random()<.5: shuffled[donors==g]=1-shuffled[donors==g]
    perm.append(roc_auc_score(shuffled,p))
observed=float(roc_auc_score(y,p)); np.savez_compressed(OUT/'gse173706_neighborhood_transfer_null_distributions.npz', observed=np.array([observed]), bootstrap_aurocs=np.asarray(boot), permutation_aurocs=np.asarray(perm), seed=np.array([20260809])); out={'observed_auroc':observed,'bootstrap_median':float(np.nanmedian(boot)),'bootstrap_ci95':[float(np.nanpercentile(boot,2.5)),float(np.nanpercentile(boot,97.5))],'permutation_mean':float(np.mean(perm)),'permutation_p_ge_observed':float((1+np.sum(np.array(perm)>=observed))/(len(perm)+1)),'n_samples':int(len(y)),'n_donors':int(len(unique_donors)),'resampling_unit':'donor; paired labels swapped within donor','null_distribution_path':'gse173706_neighborhood_transfer_null_distributions.npz'}
(OUT/'gse173706_neighborhood_transfer_stability.json').write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps(out,indent=2))

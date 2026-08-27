from pathlib import Path
import json,numpy as np
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/singlecell_v10'; d=np.load(OUT/'gse206391_spatial_expression_lodo_arrays.npz'); y=d['labels']; p=d['probabilities']; rng=np.random.default_rng(20260810); obs=roc_auc_score(y,p); boot=[]; perm=[]
for _ in range(2000):
 ix=rng.integers(0,len(y),len(y)); boot.append(roc_auc_score(y[ix],p[ix]) if len(np.unique(y[ix]))==2 else np.nan); perm.append(roc_auc_score(rng.permutation(y),p))
out={'observed_auroc':float(obs),'bootstrap_ci95':[float(np.nanpercentile(boot,2.5)),float(np.nanpercentile(boot,97.5))],'permutation_mean':float(np.mean(perm)),'permutation_p_ge_observed':float(np.mean(np.array(perm)>=obs)),'n_samples':int(len(y))}; (OUT/'gse206391_spatial_baseline_stability.json').write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps(out,indent=2))

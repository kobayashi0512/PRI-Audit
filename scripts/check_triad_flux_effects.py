"""Patient-level effect sizes for TRIAD v0 trigger fluxes."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from scipy.stats import mannwhitneyu
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/singlecell_v10'; d=pd.read_csv(OUT/'triad_trigger_transport_v0_lodo.csv'); rng=np.random.default_rng(20260810); out=[]
for col in ['flux_il17','flux_tnf','flux_synergy']:
    a=d.loc[d.true==1,col].to_numpy(); b=d.loc[d.true==0,col].to_numpy(); obs=float(a.mean()-b.mean()); boot=[]; perm=[]
    for _ in range(5000):
        aa=rng.choice(a,len(a),replace=True); bb=rng.choice(b,len(b),replace=True); boot.append(aa.mean()-bb.mean()); yp=rng.permutation(np.r_[np.ones(len(a)),np.zeros(len(b))]); vals=d[col].to_numpy(); perm.append(vals[yp==1].mean()-vals[yp==0].mean())
    out.append({'axis':col,'mean_psoriasis':float(a.mean()),'mean_control':float(b.mean()),'effect':obs,'bootstrap_ci95':[float(np.percentile(boot,2.5)),float(np.percentile(boot,97.5))],'permutation_p_abs':float(np.mean(np.abs(perm)>=abs(obs))),'mannwhitney_p':float(mannwhitneyu(a,b,alternative='two-sided').pvalue)})
(OUT/'triad_flux_effects.json').write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps(out,indent=2))

"""Corrected, reproducible GSE151177 rerun for PRI-Audit v24.

Uses GEO gene symbols (feature column 2) for mitochondrial filtering. The
primary roster is the 18 emigrating-cell profiles in the GEO series matrix,
then applies the existing source-sample QC rule (median detected genes >=300,
median mitochondrial fraction <=0.20). The five enzyme-digestion profiles are
listed in the roster audit and excluded by protocol. Control05 is retained only
in a labelled QC-threshold sensitivity analysis.
"""
from __future__ import annotations
import csv, gzip, hashlib, json, platform, sys, warnings
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import scipy, sklearn
from scipy.io import mmread
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw/pri_audit_v24/GSE151177_samples'
OUT = ROOT / 'results/pri_audit_v24_gse151177'
SEED = 20261016
N_BOOT = 5000
N_COMPONENTS = 8

# Official GEO roster: 18 emigrating profiles plus five enzyme-digestion profiles.
SAMPLES = {
 'GSM4567877':'Control01','GSM4567878':'Control02','GSM4567879':'Control03',
 'GSM4567880':'Control04','GSM4567881':'Control05','GSM4567883':'Psoriasis01',
 'GSM4567884':'Psoriasis02','GSM4567886':'Psoriasis03','GSM4567888':'Psoriasis04',
 'GSM4567890':'Psoriasis05','GSM4567891':'Psoriasis06','GSM4567893':'Psoriasis07',
 'GSM4567894':'Psoriasis08','GSM4567895':'Psoriasis09','GSM4567896':'Psoriasis10',
 'GSM4567897':'Psoriasis11','GSM4567898':'Psoriasis12','GSM4567899':'Psoriasis13',
}
F_PROFILES = {
 'GSM4567882':'Control05F','GSM4567885':'Psoriasis02F','GSM4567887':'Psoriasis03F',
 'GSM4567889':'Psoriasis04F','GSM4567892':'Psoriasis06F',
}

sys.path.insert(0, str(ROOT / 'scripts'))
import run_pri_audit_v21 as v21


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def read_series_metadata() -> pd.DataFrame:
    """Read official accession, title and protocol fields from GEO matrix."""
    path=RAW.parent/'GSE151177_series_matrix.txt.gz'
    wanted={'Sample_title','Sample_geo_accession'}
    rows={}; characteristics=[]
    with gzip.open(path,'rt',newline='') as f:
        for line in f:
            if line.startswith('!series_matrix_table_begin'): break
            if not line.startswith('!Sample_'): continue
            key, *cells = next(csv.reader([line.rstrip('\n')], delimiter='\t'))
            key=key[1:]
            if key in wanted: rows[key]=cells
            if key=='Sample_characteristics_ch1' and any('cell dissociation:' in str(x).lower() for x in cells):
                characteristics=cells
    acc=rows['Sample_geo_accession']; titles=rows['Sample_title']
    if len(acc)!=23 or len(titles)!=23 or len(characteristics)!=23:
        raise RuntimeError('Unexpected GEO roster size/metadata shape')
    return pd.DataFrame({'gsm':acc,'title':titles,'protocol_raw':characteristics})


def load_sample(gsm: str, name: str):
    stem=f'{gsm}_{name}'
    matrix_path=RAW/f'{stem}_matrix.mtx.gz'
    features_path=RAW/f'{stem}_features.tsv.gz'
    with gzip.open(matrix_path,'rb') as f: matrix=mmread(f).tocsr().astype(np.float32)
    with gzip.open(features_path,'rt') as f:
        rows=[line.rstrip('\n').split('\t') for line in f]
    ids=np.asarray([r[0] for r in rows],dtype=str)
    symbols=np.asarray([r[1] if len(r)>1 else r[0] for r in rows],dtype=str)
    if matrix.shape[0]!=len(ids): raise ValueError(f'{stem}: feature count mismatch')
    mito=np.char.startswith(np.char.upper(symbols),'MT-')
    if int(mito.sum())<10: raise RuntimeError(f'{stem}: mitochondrial symbols absent')
    total=np.asarray(matrix.sum(axis=0)).ravel()
    detected=np.asarray((matrix>0).sum(axis=0)).ravel()
    mt=np.asarray(matrix[mito].sum(axis=0)).ravel()/np.maximum(total,1)
    # Existing per-cell cutoffs, now computed using the official gene symbols.
    keep_cell=(total>=500)&(detected>=300)&(mt<=0.20)
    row={
      'gsm':gsm,'sample':name,
      'condition':'control' if name.startswith('Control') else 'psoriasis',
      'protocol':'Emigrating_cell_method',
      'n_raw_cells':int(matrix.shape[1]),
      'median_total_counts':float(np.median(total)),
      'median_detected_genes':float(np.median(detected)),
      'median_mito_fraction':float(np.median(mt)),
      'per_cell_retained':int(keep_cell.sum()),
      'sample_qc_pass':bool(np.median(detected)>=300 and np.median(mt)<=0.20),
    }
    return row, ids, matrix[:,keep_cell].sum(axis=1).A.ravel().astype(np.float64)


def grouped_bootstrap_delta(y,cand,ref,groups,seed):
    rng=np.random.default_rng(seed); units=np.unique(groups)
    idx={u:np.flatnonzero(groups==u) for u in units}; vals=[]
    for _ in range(N_BOOT):
        chosen=rng.choice(units,size=len(units),replace=True)
        ix=np.concatenate([idx[u] for u in chosen])
        if np.unique(y[ix]).size==2:
            vals.append(float(roc_auc_score(y[ix],cand[ix])-roc_auc_score(y[ix],ref[ix])))
    a=np.asarray(vals)
    return [float(x) for x in np.quantile(a,[.025,.975])],a


def score_folds(x,y,groups,metadata,include_control05=False):
    if include_control05:
        eligible=np.ones(len(y),dtype=bool)
    else:
        eligible=metadata.sample_qc_pass.to_numpy(bool)
    x=x[eligible]; y=y[eligible]; groups=groups[eligible]
    meta=metadata.loc[eligible].reset_index(drop=True)
    if len(np.unique(y))<2: raise RuntimeError('Only one class after source QC')
    methods=('expression','pca','nmf')
    folds=[]
    for group in np.unique(groups):
        test=np.flatnonzero(groups==group); train=np.flatnonzero(groups!=group)
        reps=v21.fit_strict_representations(x[train],x[test],allow_nmf=True)
        folds.append({'train_ix':train,'test_ix':test,'representations':reps})
    yprob={m:np.full(len(y),np.nan) for m in methods}
    yalt={l:{m:np.full(len(y),np.nan) for m in methods} for l in ('ridge','linear_svm')}
    estimators={
      'ridge':lambda:RidgeClassifier(alpha=1.0,class_weight='balanced'),
      'linear_svm':lambda:LinearSVC(C=1.0,class_weight='balanced',dual='auto',max_iter=5000,random_state=0)
    }
    for fold in folds:
        tr=fold['train_ix']; te=fold['test_ix']; yt=y[tr]
        for m in methods:
            a,b=fold['representations'][m]
            logit=make_pipeline(StandardScaler(),__import__('sklearn').linear_model.LogisticRegression(C=1,solver='liblinear',class_weight='balanced',max_iter=2000,random_state=0))
            logit.fit(a,yt); yprob[m][te]=logit.predict_proba(b)[:,1]
            for learner,make_est in estimators.items():
                est=make_pipeline(StandardScaler(),make_est())
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',ConvergenceWarning)
                    est.fit(a,yt)
                yalt[learner][m][te]=est.decision_function(b)
    for m in methods:
        if np.isnan(yprob[m]).any(): raise RuntimeError(f'Missing primary fold scores for {m}')
    all_scores={'logistic':yprob,**yalt}
    rows=[]; pred=meta.copy(); pred['label']=y; pred['heldout_source_sample']=groups
    for learner,by_method in all_scores.items():
        aucs={m:float(roc_auc_score(y,by_method[m])) for m in methods}
        for m in methods:
            pred[f'{learner}_{m}_score']=by_method[m]
            row={'cohort':'GSE151177','roster':'18 emigrating profiles after source QC' if not include_control05 else '18 emigrating profiles incl. Control05 QC sensitivity',
                 'learner':learner,'representation':m,'n_samples':len(y),'n_groups':len(np.unique(groups)),
                 'n_control':int((y==0).sum()),'n_psoriasis':int((y==1).sum()),'auroc':aucs[m]}
            if m!='expression':
                ci,vals=grouped_bootstrap_delta(y,by_method[m],by_method['expression'],groups,SEED+sum(map(ord,learner+m+str(include_control05))))
                row.update({'delta_vs_expression':aucs[m]-aucs['expression'],'delta_ci95_low':ci[0],'delta_ci95_high':ci[1],
                            'n_valid_bootstrap':len(vals),'probability_delta_ge_0_05':float(np.mean(vals>=.05))})
            rows.append(row)
    pred.to_csv(OUT/('predictions_control05_sensitivity.csv' if include_control05 else 'predictions_primary.csv'),index=False)
    return rows


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    official=read_series_metadata()
    mapping={**SAMPLES,**F_PROFILES}
    if set(official.gsm)!=set(mapping): raise RuntimeError('Official 23-GSM roster does not match audit mapping')
    records=[]; counts=[]; genes_ref=None
    for gsm,name in SAMPLES.items():
        row,genes,summed=load_sample(gsm,name)
        if genes_ref is None: genes_ref=genes
        elif not np.array_equal(genes_ref,genes): raise RuntimeError(f'Feature IDs/order mismatch for {gsm}')
        records.append(row); counts.append(summed)
    qc=pd.DataFrame(records)
    for r in records:
        r['sample_qc_pass']=(r['median_detected_genes']>=300 and r['median_mito_fraction']<=0.20)
        r['sample_qc_reason']='included' if r['sample_qc_pass'] else f"median detected genes {r['median_detected_genes']:.0f} < 300"
        r['primary_included']=r['sample_qc_pass']
    qc=pd.DataFrame(records)
    if qc['sample_qc_pass'].sum()!=17: raise RuntimeError(f"Expected 17 sample-QC pass, got {qc['sample_qc_pass'].sum()}")
    qc.to_csv(OUT/'gse151177_primary_source_qc.csv',index=False)
    # Full 23-row inclusion/exclusion roster from official GEO protocol and disease labels.
    full=official.copy()
    full['sample']=full.apply(lambda r: mapping[r.gsm],axis=1)
    full['condition']=full['title'].str.contains('Psoriasis',case=False).map({True:'psoriasis',False:'control'})
    full['dissociation']=full['protocol_raw'].str.extract(r'cell dissociation:\s*([^\"]+)',expand=False)
    full['analysis_role']=full.apply(lambda r:'candidate emigrating profile' if 'Emigrating' in str(r.dissociation) else 'excluded: enzyme-digestion method',axis=1)
    full=full.merge(qc[['gsm','median_total_counts','median_detected_genes','median_mito_fraction','per_cell_retained','sample_qc_pass','sample_qc_reason']],on='gsm',how='left')
    full['primary_included']=full['analysis_role'].eq('candidate emigrating profile') & full.sample_qc_pass.fillna(False)
    full.loc[full.analysis_role.str.startswith('excluded'), 'sample_qc_reason']='excluded by protocol; enzyme-digestion profile'
    full.to_csv(OUT/'gse151177_23gsm_roster_audit.csv',index=False)
    counts=np.vstack(counts)
    meta=qc[['gsm','sample','condition','per_cell_retained','median_detected_genes','median_mito_fraction','sample_qc_pass']].copy()
    np.savez_compressed(OUT/'gse151177_sample_gene_counts_corrected.npz',counts=counts[qc.sample_qc_pass.to_numpy()],genes=genes_ref)
    meta.loc[qc.sample_qc_pass].to_csv(OUT/'gse151177_sample_metadata_corrected.csv',index=False)
    x=v21.log_cpm(counts)
    y=(qc.condition=='psoriasis').to_numpy(int)
    groups=qc['sample'].to_numpy(str)
    primary=score_folds(x,y,groups,qc,False)
    sensitivity=score_folds(x,y,groups,qc,True)
    pd.DataFrame(primary+sensitivity).to_csv(OUT/'summary_primary_and_qc_sensitivity.csv',index=False)
    # Raw inputs and analysis provenance.
    files=sorted(list(RAW.glob('*.gz'))+list(RAW.parent.glob('GSE151177_series_matrix.txt.gz')))
    code_files=[Path(__file__).resolve(), (ROOT/'scripts/run_pri_audit_v21.py').resolve()]
    manifest={'analysis':'PRI-Audit v24 GSE151177 corrected mitochondrial-symbol rerun',
      'run_utc':datetime.now(timezone.utc).isoformat(),
      'script':str(Path(__file__).resolve().relative_to(ROOT)),
      'analysis_code':[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha256(p)} for p in code_files],
      'specification':'Existing v21 thresholds and learner recipe; mitochondrial mask uses gene symbols from GEO features.tsv.gz.',
      'protocol':'18 emigrating profiles are candidates; five enzyme-digestion profiles excluded. The primary analysis applies source-level median detected genes >=300 and median mitochondrial fraction <=0.20; per-cell filters are total counts >=500, detected genes >=300, mitochondrial fraction <=0.20.',
      'sensitivity':'Control05 is included despite failing sample-level QC only as a post-hoc QC robustness check; not the primary estimate.',
      'independent_unit':'GEO source sample; GSE151177 metadata do not provide validated patient IDs.',
      'seed':SEED,'bootstrap_draws':N_BOOT,
      'environment':{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'scikit_learn':sklearn.__version__},
      'inputs':[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha256(p)} for p in files],
      'outputs':{},'sample_counts':{'eligible_emigrating_before_sample_qc':18,'after_sample_qc':int(qc.sample_qc_pass.sum()),'psoriasis':int(((qc.condition=='psoriasis')&qc.sample_qc_pass).sum()),'control':int(((qc.condition=='control')&qc.sample_qc_pass).sum()),'retained_cells':int(qc.loc[qc.sample_qc_pass,'per_cell_retained'].sum())}}
    for p in sorted(OUT.iterdir()):
        if p.is_file() and p.name!='reproducibility_manifest.json': manifest['outputs'][p.name]={'bytes':p.stat().st_size,'sha256':sha256(p)}
    (OUT/'reproducibility_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(pd.DataFrame(primary+sensitivity).to_string(index=False))
    print('QC:',qc[['sample','n_raw_cells','per_cell_retained','median_detected_genes','median_mito_fraction','sample_qc_pass']].to_string(index=False))

if __name__=='__main__': main()

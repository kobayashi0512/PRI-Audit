"""Map GSE173706 cells into the GSE151177 neighborhood model."""
from pathlib import Path
import csv,gzip,json,re
import numpy as np,pandas as pd,joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,balanced_accuracy_score

ROOT=Path(__file__).resolve().parents[1]; SRC=ROOT/'data/raw/GSE173706_samples'; OUT=ROOT/'results/singlecell_v10'
model=joblib.load(OUT/'patient_balanced_neighborhood_v1_models.joblib'); gene_idx=np.asarray(model['gene_idx']); selected=set(gene_idx.tolist()); mito={"ENSG00000198888","ENSG00000198763","ENSG00000198804","ENSG00000198712","ENSG00000228263","ENSG00000198899","ENSG00000198938","ENSG00000198840","ENSG00000198727","ENSG00000198886","ENSG00000198786","ENSG00000198695","ENSG00000198700","ENSG00000210049","ENSG00000211459"}
train=np.load(OUT/'gse151177_patient_balanced_neighborhood_v1.npz'); meta=pd.read_csv(OUT/'gse151177_sample_metadata.csv').set_index('sample').loc[train['samples']]; y=(meta.condition=='psoriasis').astype(int).to_numpy(); clf=LogisticRegression(C=1,max_iter=2000,random_state=0).fit(train['histograms'],y)

def one(path):
    with gzip.open(path,'rt',newline='') as fh:
        reader=csv.reader(fh); header=next(reader); n=len(header)-1; totals=np.zeros(n); det=np.zeros(n,dtype=np.int32); mt=np.zeros(n); x=np.zeros((len(gene_idx),n),dtype=np.float32); row_i=0
        for row in reader:
            if not row: continue
            g=row[0].upper(); vals=np.asarray(row[1:],dtype=np.float32); totals+=vals; det+=vals>0
            if g in mito or g.startswith('MT-'): mt+=vals
            if row_i in selected: x[np.where(gene_idx==row_i)[0][0]]=vals
            row_i+=1
    keep=(totals>=500)&(det>=300)&(mt/np.maximum(totals,1)<=.2); z=np.log1p((x[:,keep]/np.maximum(x[:,keep].sum(axis=0,keepdims=True),1)*1e4).T); emb=model['pca'].transform(model['scaler'].transform(z)); lab=model['kmeans'].predict(emb); h=np.bincount(lab,minlength=16)/max(len(lab),1); return h,int(keep.sum())

rows=[]; hs=[]
for p in sorted(SRC.glob('*.csv.gz')):
    label=p.name.replace('.csv.gz','').split('_',1)[1]; cond='normal_skin' if label.startswith('NS-') else ('peripheral_normal' if label.startswith('PN-') else 'psoriatic'); h,n=one(p); rows.append({'sample':p.name.replace('.csv.gz',''),'condition':cond,'filtered_cells':n}); hs.append(h)
M=pd.DataFrame(rows); H=np.vstack(hs); M['donor']=M['sample'].str.split('_',n=1).str[1].str.replace(r'^(NS|PN|PP)-','',regex=True); eligible=M[M.condition.isin(['peripheral_normal','psoriatic'])]; paired=eligible.groupby('donor').condition.nunique(); paired=paired[paired==2].index; keep=(M.donor.isin(paired)&M.condition.isin(['peripheral_normal','psoriatic'])).to_numpy(); yy=(M.loc[keep,'condition']=='psoriatic').astype(int).to_numpy(); pp=clf.predict_proba(H[keep])[:,1]
out={'samples':len(M),'cells_used':int(M.filtered_cells.sum()),'paired_donors':int(len(paired)),'paired_transfer_samples':int(keep.sum()),'transfer_auroc':float(roc_auc_score(yy,pp)),'transfer_balanced_accuracy':float(balanced_accuracy_score(yy,pp>=.5)),'note':'GSE151177-trained neighborhood model transferred without refitting; evaluation restricted to complete GSE173706 PN/PP donor pairs'}
pd.DataFrame(rows).to_csv(OUT/'gse173706_transferred_neighborhood_samples.csv',index=False)
np.savez_compressed(OUT/'gse173706_neighborhood_transfer_arrays.npz',histograms=H[keep],labels=yy,probabilities=pp,samples=M.loc[keep,'sample'].to_numpy(),donors=M.loc[keep,'donor'].to_numpy())
(OUT/'gse173706_neighborhood_transfer_summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps(out,indent=2))

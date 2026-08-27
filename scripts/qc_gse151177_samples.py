"""QC report for completely downloaded GSE151177 10x sample triples."""
from pathlib import Path
import gzip
import numpy as np
import pandas as pd
from scipy.io import mmread

ROOT=Path(__file__).resolve().parents[1]; SAMPLE=ROOT/'data'/'raw'/'GSE151177_samples'; OUT=ROOT/'results'/'singlecell_v10'; OUT.mkdir(parents=True,exist_ok=True)
def main():
 rows=[]
 for mtx in sorted(SAMPLE.glob('*_matrix.mtx.gz')):
  stem=mtx.name.replace('_matrix.mtx.gz',''); feat=SAMPLE/(stem+'_features.tsv.gz'); bar=SAMPLE/(stem+'_barcodes.tsv.gz')
  try:
   with gzip.open(mtx,'rb') as f: mat=mmread(f).tocsr()
   with gzip.open(feat,'rt') as f: features=[line.rstrip('\n').split('\t') for line in f]
   with gzip.open(bar,'rt') as f: cells=sum(1 for _ in f)
   symbols=np.array([(x[1] if len(x)>1 else x[0]) for x in features]); mito=np.char.startswith(np.char.upper(symbols),'MT-')
   totals=np.asarray(mat.sum(axis=0)).ravel(); detected=np.asarray((mat>0).sum(axis=0)).ravel(); mt=np.asarray(mat[mito].sum(axis=0)).ravel()
   rows.append({'sample':stem,'cells':cells,'genes':mat.shape[0],'nnz':mat.nnz,'median_counts':float(np.median(totals)),'median_detected_genes':float(np.median(detected)),'median_mito_fraction':float(np.median(mt/np.maximum(totals,1))),'integrity':'passed'})
  except Exception as exc: rows.append({'sample':stem,'integrity':f'failed:{type(exc).__name__}'})
 frame=pd.DataFrame(rows); frame.to_csv(OUT/'gse151177_sample_qc.csv',index=False); print(frame.to_string(index=False))
if __name__=='__main__': main()

from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results'/'trigger_v8'
def main():
 pooled=pd.read_csv(OUT/'v8_transfer_baseline.csv').groupby('cohort').held_gene_cosine.mean().rename('pooled_ridge')
 frames=[]
 paths=sorted(OUT.glob('v8_groupdro_transfer_eta*.csv'))
 if (OUT/'v8_groupdro_transfer.csv').exists(): paths.append(OUT/'v8_groupdro_transfer.csv')
 for p in paths:
  if p.name == 'v8_groupdro_transfer.csv':
   eta='0.5'
  else:
   eta=p.stem.split('eta',1)[1]
  s=pd.read_csv(p).groupby('cohort').held_gene_cosine.mean().rename(f'group_weighted_eta_{eta}')
  frames.append(s)
 table=pd.concat([pooled,*frames],axis=1)
 table.to_csv(OUT/'v8_transfer_method_comparison.csv')
 print(table.round(4).to_string())
 print('\nMean across cohorts:\n'+table.mean().round(4).to_string())
if __name__=='__main__': main()

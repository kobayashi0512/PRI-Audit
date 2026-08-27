from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results'/'trigger_v9'
def main():
 f=pd.concat([pd.read_csv(p) for p in sorted(OUT.glob('v9_tuned_pooled_checkpoint_*.csv'))],ignore_index=True)
 f.to_csv(OUT/'v9_tuned_pooled_results.csv',index=False)
 print(f.round(4).to_string(index=False)); print('mean_held_gene_cosine',round(f.held_gene_cosine.mean(),4))
if __name__=='__main__': main()

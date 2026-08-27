"""Build the four frozen main tables for the v12 benchmark manuscript."""
from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; RES=ROOT/'results/singlecell_v10'; OUT=ROOT/'results/paper_v12'; OUT.mkdir(parents=True,exist_ok=True)
def j(name): return json.loads((RES/name).read_text())

bulk=pd.read_csv(ROOT/'results/trigger_v9/v9_expanded_metadata.csv').groupby('cohort').size()
datasets=[{'dataset':k,'modality':'bulk','design':'paired LS/NL','patients_or_pairs':int(v),'cells_or_spots':'NA','role':'cross-cohort ceiling'} for k,v in bulk.items()]
datasets += [
 {'dataset':'GSE151177','modality':'scRNA-seq','design':'control vs psoriasis','patients_or_pairs':18,'cells_or_spots':24753,'role':'development/strict LOO'},
 {'dataset':'GSE173706','modality':'scRNA-seq','design':'PN vs PP donor-paired','patients_or_pairs':11,'cells_or_spots':89280,'role':'external validation'},
 {'dataset':'GSE206391','modality':'spatial','design':'LS/NL patient-paired','patients_or_pairs':12,'cells_or_spots':41801,'role':'spatial validation'},
 {'dataset':'GSE24767','modality':'perturbation','design':'IL17/TNF keratinocyte','patients_or_pairs':'NA','cells_or_spots':'NA','role':'frozen trigger prior'}]
pd.DataFrame(datasets).to_csv(OUT/'table1_datasets.csv',index=False)

methods=[
 ['Strict pseudobulk logistic','patient/sample','expression mean','HVG and scaling refit per outer fold','low','primary baseline'],
 ['Strict PCA/NMF','patient/sample','low-rank expression','HVG, scaling, and factorization refit per outer fold','low','strong baseline'],
 ['Global cell-neighborhood histogram','patient/sample','cell-state proportions','global unsupervised fit','high','exploratory only'],
 ['Strict cell-neighborhood LOO','patient/sample','cell-state proportions','KMeans refit per fold','low-moderate','formal leakage audit'],
 ['Transferred neighborhood','patient/sample','fixed source neighborhoods','no target refit','low','external test'],
 ['Trigger OT v0','patient/sample','IL17/TNF/SYN flux','control reference per fold','low','negative method'],
 ['State+trigger OT v1','patient/sample','PCA+trigger+local state','frozen exploratory PCA','moderate','negative method'],
 ['Fused GW v2','patient/sample','feature+relational transport','fold reference; frozen PCA','moderate','negative method'],
 ['Spatial 6-NN','patient/sample','coordinate-neighbor smoothing','patient LODO; global HVG','moderate','smoothing control'],
 ['Strict spatial graph','patient/sample','Moran-like agreement + local contrast','HVG and graph model refit per outer fold','low','formal spatial test']]
pd.DataFrame(methods,columns=['method','statistical_unit','representation','outer_fit','leakage_risk','paper_role']).to_csv(OUT/'table2_methods_and_leakage.csv',index=False)

strict=j('strict_neighborhood_lodo_results.json'); transfer=j('gse173706_neighborhood_transfer_stability.json'); spatial=j('gse206391_spatial_baseline_stability.json'); spatial_graph=j('gse206391_strict_spatial_graph_benchmark.json')
strict_multi={row['dataset']:row for row in j('strict_multiresolution_benchmarks.json')}
def strict_row(dataset,task,method,label,validation):
    item=strict_multi[dataset]['methods'][method]
    ci=item['group_bootstrap_ci95']
    return [dataset,task,label,'AUROC',item['auroc'],f'{ci[0]:.3f}-{ci[1]:.3f}',validation]
results=[
 strict_row('GSE151177','Control vs psoriasis','pseudobulk','Strict pseudobulk','sample LOO'),
 strict_row('GSE151177','Control vs psoriasis','pca','Strict PCA','sample LOO'),
 strict_row('GSE151177','Control vs psoriasis','nmf','Strict NMF','sample LOO'),
 ['GSE151177','Control vs psoriasis','Global neighborhood','AUROC',j('patient_balanced_neighborhood_v1_summary.json')['loo_auroc'],'NA','optimistic/global fit'],
 *[['GSE151177','Control vs psoriasis',f'Strict neighborhood K={r["clusters"]}','AUROC',r['strict_loo_auroc'],'NA','strict sample LOO'] for r in strict],
 ['GSE151177','Control vs psoriasis','Trigger OT v0','AUROC',j('triad_trigger_transport_v0_summary.json')['loo_auroc'],'NA','sample LOO'],
 ['GSE151177','Control vs psoriasis','State+trigger OT v1','AUROC',j('triad_trigger_transport_v1_summary.json')['loo_auroc'],'NA','sample LOO'],
 ['GSE151177','Control vs psoriasis','Fused GW v2','AUROC',j('triad_fused_gw_v2_summary.json')['loo_auroc'],'NA','sample LOO'],
 strict_row('GSE173706','PN vs PP','pseudobulk','Strict pseudobulk','donor LODO'),
 strict_row('GSE173706','PN vs PP','pca','Strict PCA','donor LODO'),
 strict_row('GSE173706','PN vs PP','nmf','Strict NMF','donor LODO'),
 ['GSE173706','PN vs PP','Transferred neighborhood','AUROC',transfer['observed_auroc'],f'{transfer["bootstrap_ci95"][0]:.3f}-{transfer["bootstrap_ci95"][1]:.3f}','external fixed model'],
 strict_row('GSE206391','LS vs NL','pseudobulk','Strict pseudobulk','patient LODO'),
 strict_row('GSE206391','LS vs NL','pca','Strict PCA','patient LODO'),
 strict_row('GSE206391','LS vs NL','nmf','Strict NMF','patient LODO'),
 ['GSE206391','LS vs NL','Spatial 6-NN smoothing control','AUROC',j('gse206391_spatial_neighbor_summary.json')['lodo_auroc'],'NA','patient LODO; global HVG'],
 ['GSE206391','LS vs NL','Matched 500-gene expression','AUROC',spatial_graph['methods']['expression_500']['auroc'],f'{spatial_graph["methods"]["expression_500"]["group_bootstrap_ci95"][0]:.3f}-{spatial_graph["methods"]["expression_500"]["group_bootstrap_ci95"][1]:.3f}','strict patient LODO'],
 ['GSE206391','LS vs NL','Strict spatial graph','AUROC',spatial_graph['methods']['spatial_graph']['auroc'],f'{spatial_graph["methods"]["spatial_graph"]["group_bootstrap_ci95"][0]:.3f}-{spatial_graph["methods"]["spatial_graph"]["group_bootstrap_ci95"][1]:.3f}','strict patient LODO'],
 ['GSE206391','LS vs NL','Expression + strict spatial graph','AUROC',spatial_graph['methods']['expression_plus_spatial']['auroc'],f'{spatial_graph["methods"]["expression_plus_spatial"]["group_bootstrap_ci95"][0]:.3f}-{spatial_graph["methods"]["expression_plus_spatial"]["group_bootstrap_ci95"][1]:.3f}','strict patient LODO']]
pd.DataFrame(results,columns=['dataset','task','method','metric','value','ci95','validation']).to_csv(OUT/'table3_main_results.csv',index=False)

cell_shuffle=j('cell_cross_patient_shuffle.json'); flux=j('triad_flux_effects.json'); spatial_shuffle=j('spatial_coordinate_shuffle.json') if (RES/'spatial_coordinate_shuffle.json').exists() else None
controls=[
 ['GSE151177','label permutation (500 refits)','strict pseudobulk',strict_multi['GSE151177']['methods']['pseudobulk']['auroc'],strict_multi['GSE151177']['methods']['pseudobulk']['permutation_mean'],strict_multi['GSE151177']['methods']['pseudobulk']['permutation_p_ge_observed'],'signal should vanish'],
 ['GSE173706','paired-label permutation (500 refits)','strict pseudobulk',strict_multi['GSE173706']['methods']['pseudobulk']['auroc'],strict_multi['GSE173706']['methods']['pseudobulk']['permutation_mean'],strict_multi['GSE173706']['methods']['pseudobulk']['permutation_p_ge_observed'],'signal should vanish'],
 ['GSE206391','paired-label permutation (500 refits)','strict pseudobulk',strict_multi['GSE206391']['methods']['pseudobulk']['auroc'],strict_multi['GSE206391']['methods']['pseudobulk']['permutation_mean'],strict_multi['GSE206391']['methods']['pseudobulk']['permutation_p_ge_observed'],'signal should vanish'],
 ['GSE151177','cross-patient cell shuffle','global neighborhood',cell_shuffle['observed_auroc'],cell_shuffle['shuffle_mean'],cell_shuffle['p_ge_observed'],'patient structure should vanish']]
if spatial_shuffle: controls.append(['GSE206391','coordinate shuffle','spatial 6-NN',spatial_shuffle['observed_auroc'],spatial_shuffle['shuffle_mean'],spatial_shuffle['p_ge_observed'],'spatial advantage should vanish'])
controls.append(['GSE206391','paired-label permutation (500 refits)','strict spatial graph',spatial_graph['methods']['spatial_graph']['auroc'],spatial_graph['methods']['spatial_graph']['permutation_mean'],spatial_graph['methods']['spatial_graph']['permutation_p_ge_observed'],'spatial disease signal should vanish'])
controls += [['GSE151177',f'{x["axis"]} flux label permutation','TRIAD v0',x['effect'],0.0,x['permutation_p_abs'],'mechanistic flux test'] for x in flux]
pd.DataFrame(controls,columns=['dataset','control','method','observed_statistic','null_mean_or_reference','p_value','expected_behavior']).to_csv(OUT/'table4_falsification.csv',index=False)

md=['# v12 主表','']
for i,(title,file) in enumerate([('Table 1. Datasets','table1_datasets.csv'),('Table 2. Methods and leakage audit','table2_methods_and_leakage.csv'),('Table 3. Main results','table3_main_results.csv'),('Table 4. Falsification controls','table4_falsification.csv')],1):
    md += [f'## {title}','',pd.read_csv(OUT/file).to_markdown(index=False),'']
(OUT/'main_tables.md').write_text('\n'.join(md),encoding='utf-8')
print('wrote',OUT)

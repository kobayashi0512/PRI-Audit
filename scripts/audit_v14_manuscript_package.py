"""Audit the claim-frozen v14 manuscript architecture and figure package."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RES=ROOT/"results/paper_v14"
MAIN=ROOT/"figures/paper_v14/main"
ED=ROOT/"figures/paper_v14/extended_data"
DOC=ROOT/"docs"


def main():
    checks={}
    main_stems=["fig1_pri_audit_framework","fig2_expression_ceiling","fig3_unsupervised_optimism","fig4_external_transfer","fig5_strict_spatial_value","fig6_falsification_evidence_map"]
    ed_stems=[f"figED{i}_{name}" for i,name in enumerate(["cell_spot_qc","strict_outer_roc","prediction_distributions","bulk_robustness","neighborhood_sensitivity","external_calibration","transport_diagnostics","spatial_graph_diagnostics","complete_null_distributions","hyperparameter_sensitivity"],1)]
    checks["main_figures"]={"pass":all((MAIN/f"{stem}.{ext}").exists() and (MAIN/f"{stem}.{ext}").stat().st_size>0 for stem in main_stems for ext in ("pdf","svg","png")),"figures":len(main_stems),"files":len(list(MAIN.glob("*.*")))}
    checks["extended_data"]={"pass":all((ED/f"{stem}.{ext}").exists() and (ED/f"{stem}.{ext}").stat().st_size>0 for stem in ed_stems for ext in ("pdf","svg","png")),"figures":len(ed_stems),"files":len([x for x in ED.glob("figED*.*")])}
    required=[RES/"main_table1_public_datasets.csv",RES/"main_table2_patient_level_incremental_value.csv",RES/"claim_registry.csv",RES/"figure_source_data_map.csv",RES/"figure_briefs.json",DOC/"v14_claim_freeze.md",DOC/"v14_manuscript_skeleton.md",DOC/"v14_figure_table_manifest.md",DOC/"v14_main_figure_legends.md"]
    checks["manuscript_files"]={"pass":all(x.exists() and x.stat().st_size>0 for x in required),"files":[str(x) for x in required]}
    table1=pd.read_csv(RES/"main_table1_public_datasets.csv"); table2=pd.read_csv(RES/"main_table2_patient_level_incremental_value.csv"); claims=pd.read_csv(RES/"claim_registry.csv")
    delta_error=np.abs((table2.candidate_auroc-table2.reference_auroc)-table2.delta_auroc)
    checks["numeric_consistency"]={"pass":bool((delta_error<1e-10).all() and np.isfinite(table2[["candidate_auroc","reference_auroc","delta_auroc","delta_ci95_low","delta_ci95_high"]]).all().all()),"max_delta_reconstruction_error":float(delta_error.max())}
    checks["content_counts"]={"pass":len(table1)==11 and len(table2)==9 and len(claims)==7,"datasets":int(len(table1)),"incremental_comparisons":int(len(table2)),"frozen_claims":int(len(claims))}
    result={"status":"pass" if all(x["pass"] for x in checks.values()) else "fail","checks":checks}
    (RES/"package_audit.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2)); raise SystemExit(0 if result["status"]=="pass" else 1)


if __name__=="__main__": main()

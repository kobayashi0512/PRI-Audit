"""Collect frozen supplementary plots into the v14 Extended Data order."""
from pathlib import Path
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures/paper_v14/extended_data"
OUT.mkdir(parents=True, exist_ok=True)

mapping = [
    ("figED1_cell_spot_qc", ROOT/"figures/paper_v12/fig2_data_qc"),
    ("figED2_strict_outer_roc", ROOT/"figures/paper_v12/supplement/figS1_strict_roc"),
    ("figED3_prediction_distributions", ROOT/"figures/paper_v12/supplement/figS3_prediction_distributions"),
    ("figED4_bulk_robustness", ROOT/"figures/paper_v12/supplement/figS7_bulk_robustness"),
    ("figED5_neighborhood_sensitivity", ROOT/"figures/paper_v12/supplement/figS4_neighborhood_sensitivity"),
    ("figED6_external_calibration", ROOT/"figures/paper_v13/figED_v13_external_calibration"),
    ("figED7_transport_diagnostics", ROOT/"figures/paper_v12/supplement/figS5_transport_diagnostics"),
    ("figED8_spatial_graph_diagnostics", ROOT/"figures/paper_v12/supplement/figS6_spatial_graph_diagnostics"),
    ("figED9_complete_null_distributions", ROOT/"figures/paper_v13/figED_v13_complete_null_distributions"),
    ("figED10_hyperparameter_sensitivity", ROOT/"figures/paper_v13/figED_v13_sensitivity_matrices"),
]

records=[]
for target, source in mapping:
    files=[]
    for ext in ("pdf","svg","png"):
        src=source.with_suffix(f".{ext}"); dst=OUT/f"{target}.{ext}"
        if not src.exists(): raise FileNotFoundError(src)
        shutil.copy2(src,dst); files.append(str(dst))
    records.append({"extended_data_figure":target,"source_stem":str(source),"files":files})
(OUT/"extended_data_assembly.json").write_text(json.dumps(records,indent=2),encoding="utf-8")
print(json.dumps({"figures":len(records),"files":sum(len(x['files']) for x in records),"output":str(OUT)},indent=2))

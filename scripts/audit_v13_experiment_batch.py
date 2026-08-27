"""Integrity audit for the six v13 completion analyses."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/singlecell_v10"
PAPER = ROOT / "results/paper_v13"
required = {
    "optimism": [OUT/"gse151177_neighborhood_optimism_bootstrap.json", OUT/"gse151177_neighborhood_optimism_bootstrap.csv"],
    "cell_null": [OUT/"cell_cross_patient_shuffle_null_distribution.npz"],
    "label_null": [OUT/"gse206391_strict_spatial_graph_resampling_distributions.npz"],
    "coordinate_null": [OUT/"gse206391_strict_spatial_coordinate_shuffle_distributions.npz"],
    "calibration": [OUT/"gse173706_neighborhood_transfer_calibration.json", OUT/"gse173706_external_calibration_bootstrap.npz"],
    "representative": [OUT/"gse206391_representative_spatial_selection.json", OUT/"gse206391_representative_LS_NL_spatial_map.pdf"],
    "sensitivity": [OUT/"gse206391_strict_spatial_graph_sensitivity_matrix.csv"],
    "figures": [ROOT/"figures/paper_v13/figED_v13_complete_null_distributions.pdf", ROOT/"figures/paper_v13/figED_v13_external_calibration.pdf", ROOT/"figures/paper_v13/figED_v13_optimism_gap_bootstrap.pdf", ROOT/"figures/paper_v13/figED_v13_sensitivity_matrices.pdf"],
}


def main():
    checks = {}
    for key, paths in required.items():
        checks[key] = {"pass": all(p.exists() and p.stat().st_size > 0 for p in paths), "paths": [str(p) for p in paths]}
    z = np.load(OUT/"cell_cross_patient_shuffle_null_distribution.npz"); checks["cell_null"]["n"] = int(len(z["null_aurocs"]))
    z = np.load(OUT/"gse206391_strict_spatial_graph_resampling_distributions.npz"); checks["label_null"]["n"] = int(len(z["label_permutation_spatial_graph"]))
    z = np.load(OUT/"gse206391_strict_spatial_coordinate_shuffle_distributions.npz"); checks["coordinate_null"]["n"] = int(len(z["null_spatial"]))
    sensitivity = pd.read_csv(OUT/"gse206391_strict_spatial_graph_sensitivity_matrix.csv")
    checks["sensitivity"]["n_rows"] = int(len(sensitivity)); checks["sensitivity"]["finite_auroc"] = bool(np.isfinite(sensitivity.auroc).all())
    calibration = json.loads((OUT/"gse173706_neighborhood_transfer_calibration.json").read_text())
    checks["calibration"]["finite_summary"] = bool(all(np.isfinite(calibration["observed"][k]) for k in ("calibration_intercept", "calibration_slope", "brier_score")))
    result = {"status": "pass" if all(x.get("pass", True) and x.get("finite_auroc", True) and x.get("finite_summary", True) for x in checks.values()) else "fail", "checks": checks}
    (PAPER/"v13_experiment_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "pass" else 1)


if __name__ == "__main__":
    main()

"""Calibration diagnostics for frozen external neighborhood transfer."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/singlecell_v10"
SEED = 20260810


def calibration_stats(y, p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=int)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000)
    model.fit(logit, y)
    prob_true, prob_pred = calibration_curve(y, p, n_bins=5, strategy="quantile")
    return {
        "calibration_intercept": float(model.intercept_[0]),
        "calibration_slope": float(model.coef_[0, 0]),
        "brier_score": float(brier_score_loss(y, p)),
        "reliability_predicted": prob_pred.tolist(),
        "reliability_observed": prob_true.tolist(),
        "reliability_n_bins": int(len(prob_true)),
    }


def main(n_boot=5000):
    d = np.load(OUT / "gse173706_neighborhood_transfer_arrays.npz", allow_pickle=True)
    y, p, donors = d["labels"].astype(int), d["probabilities"].astype(float), d["donors"].astype(str)
    observed = calibration_stats(y, p)
    rng = np.random.default_rng(SEED)
    unique = np.unique(donors)
    boot_intercept, boot_slope, boot_brier = [], [], []
    for _ in range(n_boot):
        selected = rng.choice(unique, size=len(unique), replace=True)
        ix = np.concatenate([np.flatnonzero(donors == donor) for donor in selected])
        if np.unique(y[ix]).size < 2:
            continue
        try:
            s = calibration_stats(y[ix], p[ix])
        except ValueError:
            continue
        boot_intercept.append(s["calibration_intercept"])
        boot_slope.append(s["calibration_slope"])
        boot_brier.append(s["brier_score"])
    arrays = {"intercept": np.asarray(boot_intercept), "slope": np.asarray(boot_slope), "brier": np.asarray(boot_brier)}
    np.savez_compressed(OUT / "gse173706_external_calibration_bootstrap.npz", **arrays)
    result = {
        "dataset": "GSE173706",
        "model": "GSE151177 frozen neighborhood transfer",
        "n_samples": int(len(y)), "n_donors": int(len(unique)),
        "resampling_unit": "donor",
        "observed": observed,
        "bootstrap_n": int(len(arrays["slope"])),
        "bootstrap_ci95": {
            key: [float(np.quantile(value, .025)), float(np.quantile(value, .975))]
            for key, value in arrays.items()
        },
    }
    (OUT / "gse173706_neighborhood_transfer_calibration.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame({"sample": d["samples"].astype(str), "donor": donors, "label": y, "probability": p}).to_csv(
        OUT / "gse173706_neighborhood_transfer_calibration_rows.csv", index=False
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

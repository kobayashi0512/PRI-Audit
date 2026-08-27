"""Post-hoc descriptive fixed-score controls declared in final_analysis_boundaries.md."""

from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
RAW, RESULTS = ROOT / "data" / "raw", ROOT / "results" / "ccmac_v3"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import collapse_duplicate_genes, parse_geo_series_matrix, sample_titles  # noqa: E402
from run_isrc_v2 import load_platform_map  # noqa: E402

N_BOOT, SEED = 10000, 20260730


def zrank(x: np.ndarray) -> np.ndarray:
    r = rankdata(np.asarray(x, dtype=float), method="average")
    return (r - r.mean()) / max(r.std(ddof=0), 1e-12)


def unit_columns(x: np.ndarray) -> np.ndarray:
    out = np.asarray(x, dtype=float).copy()
    norm = np.linalg.norm(out, axis=0)
    out[:, norm > 0] /= norm[norm > 0]
    return out


def characteristics(path: Path) -> list[list[str]]:
    rows = []
    with gzip.open(path, "rt", errors="replace") as h:
        for line in h:
            if line.startswith("!Sample_characteristics_ch1"):
                rows.append([x.strip('"') for x in line.rstrip("\n").split("\t")[1:]])
    return rows


def boot_mean(x: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    means = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(N_BOOT)])
    return {"mean": float(x.mean()), "ci_low": float(np.quantile(means,.025)), "ci_high": float(np.quantile(means,.975)), "prob_lt_zero": float((means < 0).mean())}


def v0_score(target: np.ndarray, dictionary: np.ndarray) -> float:
    weight, _ = nnls(dictionary, target)
    return float(1 - np.sum((target - dictionary @ weight) ** 2) / max(np.sum((target - target.mean()) ** 2), 1e-12))


def main() -> None:
    locked = np.load(RESULTS / "locked_model.npz", allow_pickle=True)
    genes = locked["mechanism_genes"].astype(str).tolist()
    dictionary = locked["mechanism_dictionary"]
    mean, scale, theta = locked["mean"], locked["scale"], locked["theta"]
    path = RAW / "GSE85034_series_matrix.txt.gz"
    expression, meta0 = parse_geo_series_matrix(path)
    expression.index = expression.index.astype(str).map(load_platform_map(RAW / "GPL10558.annot.gz"))
    expression = collapse_duplicate_genes(expression)
    titles = sample_titles(meta0)
    treatment = next(row for row in characteristics(path) if row[0].startswith("treatment:"))
    rows = []
    for accession, title, drug in zip(titles, titles.values(), treatment):
        match = re.fullmatch(r"Subject (\d+) (NL|LS|WK ?\d+)", title)
        if match:
            subject, time = match.groups()
            rows.append({"accession": accession, "subject": int(subject), "time": time.replace("WK ", "WK"), "treatment": drug.split(": ",1)[1]})
    meta = pd.DataFrame(rows)
    available = [g for g in genes if g in expression.index]
    positions = [genes.index(g) for g in available]
    full_dictionary = unit_columns(dictionary[positions])
    v0_dictionary = unit_columns(full_dictionary[:, :5])
    records = []
    for row in meta.itertuples():
        target = zrank(expression.loc[available, row.accession].to_numpy())
        feature = target @ full_dictionary
        whole = pd.Series(zrank(expression[row.accession].to_numpy()), index=expression.index)
        records.append({"accession": row.accession, "locked_ridge": float(((feature - mean) / scale) @ theta), "v0_nnls": v0_score(target, v0_dictionary), "nos2_minus_ccl27": float(whole.NOS2 - whole.CCL27)})
    scores = meta.merge(pd.DataFrame(records), on="accession", validate="one_to_one")
    metrics = {"status": "post-hoc descriptive fixed-score comparison", "scores": {}}
    for score in ("locked_ridge", "v0_nnls", "nos2_minus_ccl27"):
        pivot = scores.pivot(index=["subject", "treatment"], columns="time", values=score)
        trajectory = []
        for time in ("WK1", "WK2", "WK4", "WK16"):
            complete = pivot.dropna(subset=["LS", time])
            trajectory.append({"time": time, "n": int(len(complete)), **boot_mean((complete[time] - complete["LS"]).to_numpy())})
        normalized = pivot.dropna(subset=["NL", "LS", "WK16"])
        distance_delta = (np.abs(normalized["WK16"] - normalized["NL"]) - np.abs(normalized["LS"] - normalized["NL"])).to_numpy()
        metrics["scores"][score] = {"trajectory": trajectory, "week16_distance_to_own_nl_change": {"n": int(len(distance_delta)), **boot_mean(distance_delta)}}
    scores.to_csv(RESULTS / "GSE85034_final_fixed_score_controls.csv", index=False)
    (RESULTS / "GSE85034_final_fixed_score_controls.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

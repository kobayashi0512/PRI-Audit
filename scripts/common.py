from __future__ import annotations

import gzip
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def log_cpm(counts: pd.DataFrame, prior_count: float = 0.5) -> pd.DataFrame:
    """Library-size normalized log2 counts-per-million."""
    lib = counts.sum(axis=0)
    if (lib <= 0).any():
        raise ValueError("Encountered a sample with non-positive library size")
    return np.log2((counts + prior_count).div(lib + prior_count, axis=1) * 1e6)


def collapse_duplicate_genes(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.loc[frame.index.notna()]
    frame = frame.loc[frame.index.astype(str).str.len() > 0]
    return frame.groupby(frame.index.astype(str), sort=False).mean()


def paired_effect(
    expression: pd.DataFrame,
    treated: list[str],
    control: list[str],
) -> tuple[pd.Series, pd.DataFrame]:
    if len(treated) != len(control):
        raise ValueError("Treated and control sample lists must be paired")
    delta = pd.DataFrame(
        {
            f"pair_{i + 1}": expression[t] - expression[c]
            for i, (t, c) in enumerate(zip(treated, control))
        }
    )
    return delta.mean(axis=1), delta


def parse_geo_series_matrix(path: Path) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Parse expression matrix and sample metadata lines from a GEO matrix."""
    opener = gzip.open if path.suffix == ".gz" else open
    metadata: dict[str, list[str]] = {}
    data_lines: list[str] = []
    in_table = False
    with opener(path, "rt", errors="replace") as handle:
        for line in handle:
            if line.startswith("!series_matrix_table_begin"):
                in_table = True
                continue
            if line.startswith("!series_matrix_table_end"):
                break
            if in_table:
                data_lines.append(line)
            elif line.startswith("!Sample_"):
                key, *values = line.rstrip("\n").split("\t")
                metadata[key] = [v.strip('"') for v in values]
    if not data_lines:
        raise ValueError(f"No expression table found in {path}")
    from io import StringIO

    expr = pd.read_csv(StringIO("".join(data_lines)), sep="\t", index_col=0)
    expr.columns = [str(c).strip('"') for c in expr.columns]
    return expr, metadata


def hgnc_entrez_map(path: Path) -> dict[str, str]:
    table = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
    table = table.dropna(subset=["entrez_id", "symbol"])
    return dict(zip(table["entrez_id"], table["symbol"]))


def rank_similarity(a: pd.Series, b: pd.Series) -> dict[str, float]:
    common = a.index.intersection(b.index)
    x = a.loc[common]
    y = b.loc[common]
    rho, p = stats.spearmanr(x, y)
    cosine = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
    return {
        "n_genes": int(len(common)),
        "spearman_rho": float(rho),
        "spearman_p": float(p),
        "cosine": cosine,
    }


def signed_set_score(
    disease_effect: pd.Series,
    perturbation_effect: pd.Series,
    top_n: int = 200,
) -> float:
    """Difference between mean disease z-score in perturbation-up and -down genes."""
    common = disease_effect.index.intersection(perturbation_effect.index)
    d = disease_effect.loc[common]
    p = perturbation_effect.loc[common]
    top_n = min(top_n, max(10, len(common) // 4))
    up = p.nlargest(top_n).index
    down = p.nsmallest(top_n).index
    z = (d - d.mean()) / d.std(ddof=1)
    return float(z.loc[up].mean() - z.loc[down].mean())


def permutation_pvalue(
    disease_effect: pd.Series,
    perturbation_effect: pd.Series,
    observed: float,
    top_n: int = 200,
    n_perm: int = 2000,
    seed: int = 20260729,
) -> float:
    common = disease_effect.index.intersection(perturbation_effect.index)
    d = disease_effect.loc[common].to_numpy(dtype=float)
    d = (d - d.mean()) / d.std(ddof=1)
    top_n = min(top_n, max(10, len(common) // 4))
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(n_perm):
        selected = rng.choice(len(common), 2 * top_n, replace=False)
        score = d[selected[:top_n]].mean() - d[selected[top_n:]].mean()
        exceed += abs(score) >= abs(observed)
    return (exceed + 1) / (n_perm + 1)


def bootstrap_specificity(
    pso_delta: pd.DataFrame,
    ad_delta: pd.DataFrame,
    perturbation_effect: pd.Series,
    top_n: int = 200,
    n_boot: int = 2000,
    seed: int = 20260729,
) -> dict[str, float]:
    common = (
        pso_delta.index.intersection(ad_delta.index)
        .intersection(perturbation_effect.index)
    )
    pso_values = pso_delta.loc[common].to_numpy(dtype=float)
    ad_values = ad_delta.loc[common].to_numpy(dtype=float)
    perturbation = perturbation_effect.loc[common].to_numpy(dtype=float)
    top_n = min(top_n, max(10, len(common) // 4))
    order = np.argsort(perturbation)
    down = order[:top_n]
    up = order[-top_n:]
    rng = np.random.default_rng(seed)
    values = []
    batch_size = 200
    for start in range(0, n_boot, batch_size):
        batch = min(batch_size, n_boot - start)
        pso_weights = rng.multinomial(
            pso_values.shape[1],
            np.full(pso_values.shape[1], 1 / pso_values.shape[1]),
            size=batch,
        )
        ad_weights = rng.multinomial(
            ad_values.shape[1],
            np.full(ad_values.shape[1], 1 / ad_values.shape[1]),
            size=batch,
        )
        pso_effects = pso_weights @ pso_values.T / pso_values.shape[1]
        ad_effects = ad_weights @ ad_values.T / ad_values.shape[1]
        pso_z = (pso_effects - pso_effects.mean(axis=1, keepdims=True)) / pso_effects.std(
            axis=1, ddof=1, keepdims=True
        )
        ad_z = (ad_effects - ad_effects.mean(axis=1, keepdims=True)) / ad_effects.std(
            axis=1, ddof=1, keepdims=True
        )
        values.append(
            pso_z[:, up].mean(axis=1)
            - pso_z[:, down].mean(axis=1)
            - ad_z[:, up].mean(axis=1)
            + ad_z[:, down].mean(axis=1)
        )
    arr = np.concatenate(values)
    return {
        "mean": float(arr.mean()),
        "ci_low": float(np.quantile(arr, 0.025)),
        "ci_high": float(np.quantile(arr, 0.975)),
        "prob_gt_zero": float((arr > 0).mean()),
    }


def sample_titles(metadata: dict[str, list[str]]) -> dict[str, str]:
    accessions = metadata.get("!Sample_geo_accession", [])
    titles = metadata.get("!Sample_title", [])
    return dict(zip(accessions, titles))


def patient_id_from_title(title: str) -> str:
    match = re.search(r"(?:AD|PSO)_(\d+)", title)
    if not match:
        raise ValueError(f"Could not parse patient ID from {title!r}")
    return match.group(1)

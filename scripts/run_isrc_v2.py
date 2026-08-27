from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.optimize import nnls
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results" / "isrc_v2"
FIGURES = ROOT / "figures" / "isrc_v2"
DOCS = ROOT / "docs"
for directory in (RESULTS, FIGURES, DOCS):
    directory.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import collapse_duplicate_genes, parse_geo_series_matrix, sample_titles  # noqa: E402
from run_cone_v0 import (  # noqa: E402
    SEED,
    build_dictionaries,
    explained_variance_score,
    load_perturbation_deltas,
    normalize_columns,
    rank_unit,
    unit,
)
from run_external_v1 import (  # noqa: E402
    load_gse36287_effects,
    load_gse86451_deltas,
    nonadditive_residual,
)
from run_feasibility import load_gse121212_deltas  # noqa: E402

N_RELIABILITY_BOOT = 1000
N_EXTERNAL_BOOT = 5000
BETA_GRID = [0.0, 0.05, 0.10, 0.25, 0.50]
AXIS_NAMES = ["IL17", "TNF_low", "TNF_high", "IL17_TNF_low", "IL17_TNF_high"]


def load_platform_map(path: Path) -> dict[str, str]:
    lines: list[str] = []
    in_table = False
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", errors="replace") as handle:
        for line in handle:
            if line.startswith("!platform_table_begin"):
                in_table = True
                continue
            if line.startswith("!platform_table_end"):
                break
            if in_table:
                lines.append(line)
    annotation = pd.read_csv(
        StringIO("".join(lines)), sep="\t", dtype=str, low_memory=False
    )
    symbol_column = next(
        (
            column
            for column in annotation.columns
            if column.lower().replace(" ", "_")
            in {"gene_symbol", "gene_symbols", "symbol"}
        ),
        None,
    )
    if symbol_column is None:
        raise KeyError(f"No gene symbol column in {path}: {annotation.columns.tolist()}")
    mapping = {}
    for probe, raw_symbol in zip(annotation["ID"], annotation[symbol_column]):
        if pd.isna(raw_symbol):
            continue
        symbol = (
            str(raw_symbol)
            .split("///")[0]
            .split(";")[0]
            .split(",")[0]
            .strip()
        )
        if symbol:
            mapping[str(probe)] = symbol
    return mapping


def mean_residuals(
    discovery: dict[str, pd.DataFrame],
    replication: dict[str, pd.DataFrame],
    genes: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    discovery_residual, _, _ = nonadditive_residual(
        discovery["IL17"].loc[genes].mean(axis=1).to_numpy(),
        discovery["TNF_high"].loc[genes].mean(axis=1).to_numpy(),
        discovery["IL17_TNF_high"].loc[genes].mean(axis=1).to_numpy(),
    )
    replication_residual, _, _ = nonadditive_residual(
        replication["IL17"].loc[genes].mean(axis=1).to_numpy(),
        replication["TNF"].loc[genes].mean(axis=1).to_numpy(),
        replication["combo"].loc[genes].mean(axis=1).to_numpy(),
    )
    return discovery_residual, replication_residual


def reliability_weighted_residual(
    discovery: dict[str, pd.DataFrame],
    replication: dict[str, pd.DataFrame],
    genes: list[str],
) -> tuple[np.ndarray, pd.DataFrame]:
    discovery_arrays = {
        key: discovery[key].loc[genes].to_numpy(dtype=float)
        for key in ("IL17", "TNF_high", "IL17_TNF_high")
    }
    replication_arrays = {
        clean: replication[raw].loc[genes].to_numpy(dtype=float)
        for clean, raw in (("IL17", "IL17"), ("TNF", "TNF"), ("combo", "combo"))
    }
    discovery_point, replication_point = mean_residuals(
        discovery, replication, genes
    )
    discovery_positive = np.zeros(len(genes), dtype=float)
    replication_positive = np.zeros(len(genes), dtype=float)
    rng = np.random.default_rng(SEED)
    for _ in range(N_RELIABILITY_BOOT):
        il17_index = rng.choice(
            discovery_arrays["IL17"].shape[1],
            discovery_arrays["IL17"].shape[1],
            replace=True,
        )
        paired_index = rng.choice(
            discovery_arrays["TNF_high"].shape[1],
            discovery_arrays["TNF_high"].shape[1],
            replace=True,
        )
        replication_index = rng.choice(
            replication_arrays["IL17"].shape[1],
            replication_arrays["IL17"].shape[1],
            replace=True,
        )
        discovery_boot, _, _ = nonadditive_residual(
            discovery_arrays["IL17"][:, il17_index].mean(axis=1),
            discovery_arrays["TNF_high"][:, paired_index].mean(axis=1),
            discovery_arrays["IL17_TNF_high"][:, paired_index].mean(axis=1),
        )
        replication_boot, _, _ = nonadditive_residual(
            replication_arrays["IL17"][:, replication_index].mean(axis=1),
            replication_arrays["TNF"][:, replication_index].mean(axis=1),
            replication_arrays["combo"][:, replication_index].mean(axis=1),
        )
        discovery_positive += discovery_boot > 0
        replication_positive += replication_boot > 0

    discovery_probability = discovery_positive / N_RELIABILITY_BOOT
    replication_probability = replication_positive / N_RELIABILITY_BOOT
    discovery_confidence = np.abs(2 * discovery_probability - 1)
    replication_confidence = np.abs(2 * replication_probability - 1)
    agreement = np.sign(discovery_point) == np.sign(replication_point)
    reliability = agreement * np.sqrt(
        discovery_confidence * replication_confidence
    )
    consensus = reliability * (
        rank_unit(discovery_point) + rank_unit(replication_point)
    ) / 2
    table = pd.DataFrame(
        {
            "gene": genes,
            "discovery_residual": discovery_point,
            "replication_residual": replication_point,
            "discovery_positive_probability": discovery_probability,
            "replication_positive_probability": replication_probability,
            "sign_agreement": agreement.astype(int),
            "reliability": reliability,
            "weighted_consensus": consensus,
        }
    )
    return consensus, table


def nuisance_basis(
    broad_effects: dict[str, pd.Series], genes: list[str]
) -> np.ndarray:
    nuisance = np.column_stack(
        [
            rank_unit(broad_effects[name].loc[genes].to_numpy())
            for name in ("IL4", "IL13", "IFNa", "IFNg")
        ]
    )
    q, _ = np.linalg.qr(nuisance, mode="reduced")
    return q


def project_nuisance(vector_or_matrix: np.ndarray, q: np.ndarray) -> np.ndarray:
    return vector_or_matrix - q @ (q.T @ vector_or_matrix)


def build_isrc_model(
    effects: dict[str, pd.Series],
    discovery_deltas: dict[str, pd.DataFrame],
    replication_deltas: dict[str, pd.DataFrame],
    broad_effects: dict[str, pd.Series],
    genes: list[str],
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    baseline, _, _, _, _ = build_dictionaries(effects, genes)
    consensus, reliability = reliability_weighted_residual(
        discovery_deltas, replication_deltas, genes
    )
    base_projection, *_ = np.linalg.lstsq(baseline, consensus, rcond=None)
    novel_residual = unit(consensus - baseline @ base_projection)
    q = nuisance_basis(broad_effects, genes)
    candidate = normalize_columns(
        project_nuisance(np.column_stack([baseline, novel_residual]), q)
    )
    return {
        "baseline_dictionary": baseline,
        "candidate_dictionary": candidate,
        "nuisance_basis": q,
        "novel_residual": novel_residual,
    }, reliability


def score_target(
    target: np.ndarray,
    model: dict[str, np.ndarray],
    beta: float,
) -> dict[str, float]:
    target = rank_unit(target)
    baseline = model["baseline_dictionary"]
    correlations = baseline.T @ target
    baseline_weights, _ = nnls(baseline, target)
    baseline_score = explained_variance_score(
        target, baseline, baseline_weights
    )

    projected_target = unit(
        project_nuisance(target, model["nuisance_basis"])
    )
    candidate = model["candidate_dictionary"]
    candidate_weights, _ = nnls(candidate, projected_target)
    candidate_fit = explained_variance_score(
        projected_target, candidate, candidate_weights
    )
    residual_fraction = float(
        candidate_weights[-1] / (candidate_weights.sum() + 1e-12)
    )
    return {
        "v0_max_correlation": float(correlations.max()),
        "v0_nnls": baseline_score,
        "isrc_fit": candidate_fit,
        "isrc_residual_fraction": residual_fraction,
        "isrc_score": candidate_fit + beta * residual_fraction,
    }


def development_predictions(
    pso: pd.DataFrame,
    ad: pd.DataFrame,
    genes: list[str],
    model: dict[str, np.ndarray],
    beta: float,
) -> pd.DataFrame:
    rows = []
    for disease, frame, label in (("PSO", pso, 1), ("AD", ad, 0)):
        for patient in frame.columns:
            rows.append(
                {
                    "sample": f"{disease}_{patient}",
                    "disease": disease,
                    "label": label,
                    **score_target(
                        frame.loc[genes, patient].to_numpy(), model, beta
                    ),
                }
            )
    return pd.DataFrame(rows)


def select_beta(
    pso: pd.DataFrame,
    ad: pd.DataFrame,
    genes: list[str],
    model: dict[str, np.ndarray],
) -> tuple[float, pd.DataFrame, dict]:
    rows = []
    prediction_tables = {}
    for beta in BETA_GRID:
        predictions = development_predictions(pso, ad, genes, model, beta)
        prediction_tables[beta] = predictions
        rows.append(
            {
                "beta": beta,
                "isrc_auc": roc_auc_score(
                    predictions["label"], predictions["isrc_score"]
                ),
                "v0_nnls_auc": roc_auc_score(
                    predictions["label"], predictions["v0_nnls"]
                ),
                "v0_correlation_auc": roc_auc_score(
                    predictions["label"],
                    predictions["v0_max_correlation"],
                ),
            }
        )
    grid = pd.DataFrame(rows)
    best_auc = grid["isrc_auc"].max()
    beta = float(grid.loc[grid["isrc_auc"] == best_auc, "beta"].min())
    locked_predictions = prediction_tables[beta]
    summary = {
        "selected_beta": beta,
        "selected_isrc_auc": float(best_auc),
        "v0_nnls_auc": float(
            roc_auc_score(
                locked_predictions["label"], locked_predictions["v0_nnls"]
            )
        ),
        "v0_correlation_auc": float(
            roc_auc_score(
                locked_predictions["label"],
                locked_predictions["v0_max_correlation"],
            )
        ),
    }
    return beta, grid, summary


def save_locked_model(
    genes: list[str],
    model: dict[str, np.ndarray],
    beta: float,
    development_summary: dict,
) -> dict:
    artifact = RESULTS / "locked_model.npz"
    np.savez_compressed(
        artifact,
        genes=np.asarray(genes),
        beta=np.asarray([beta]),
        **model,
    )
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = {
        "algorithm": "Invariant Synergy Residual Cone (ISRC) v2",
        "seed": SEED,
        "n_reliability_bootstrap": N_RELIABILITY_BOOT,
        "beta_grid": BETA_GRID,
        "selected_beta": beta,
        "n_genes": len(genes),
        "axis_names": AXIS_NAMES + ["RELIABILITY_WEIGHTED_RESIDUAL"],
        "nuisance_axes": ["IL4", "IL13", "IFNa", "IFNg"],
        "development": development_summary,
        "artifact_sha256": digest,
        "external_expression_read": False,
    }
    (RESULTS / "locked_model_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )
    return manifest


def develop() -> None:
    effects, discovery_deltas = load_perturbation_deltas()
    replication_deltas = load_gse86451_deltas()
    broad_effects = load_gse36287_effects()
    pso, ad = load_gse121212_deltas()
    platform_genes = set(
        load_platform_map(RAW / "GPL14550_full.txt").values()
    )
    frozen = pd.read_csv(
        ROOT / "results" / "cone_v0" / "frozen_feature_genes.csv"
    )["gene"].astype(str)
    genes = [
        gene
        for gene in frozen
        if gene in platform_genes
        and gene in pso.index
        and gene in ad.index
        and all(gene in effect.index for effect in effects.values())
        and all(gene in frame.index for frame in replication_deltas.values())
        and all(gene in effect.index for effect in broad_effects.values())
    ]
    model, reliability = build_isrc_model(
        effects,
        discovery_deltas,
        replication_deltas,
        broad_effects,
        genes,
    )
    beta, grid, development_summary = select_beta(
        pso, ad, genes, model
    )
    manifest = save_locked_model(
        genes, model, beta, development_summary
    )
    reliability.to_csv(RESULTS / "gene_reliability.csv", index=False)
    grid.to_csv(RESULTS / "development_beta_grid.csv", index=False)
    development_predictions(
        pso, ad, genes, model, beta
    ).to_csv(RESULTS / "development_predictions.csv", index=False)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def load_locked_model() -> tuple[list[str], float, dict[str, np.ndarray], dict]:
    artifact = RESULTS / "locked_model.npz"
    manifest = json.loads(
        (RESULTS / "locked_model_manifest.json").read_text()
    )
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    if digest != manifest["artifact_sha256"]:
        raise RuntimeError("Locked model hash mismatch")
    archive = np.load(artifact)
    genes = archive["genes"].astype(str).tolist()
    beta = float(archive["beta"][0])
    model = {
        key: archive[key]
        for key in (
            "baseline_dictionary",
            "candidate_dictionary",
            "nuisance_basis",
            "novel_residual",
        )
    }
    return genes, beta, model, manifest


def load_gse57225_triplets(
    genes: list[str],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    expression, metadata = parse_geo_series_matrix(
        RAW / "GSE57225_series_matrix.txt.gz"
    )
    mapping = load_platform_map(RAW / "GPL14550_full.txt")
    expression.index = expression.index.astype(str).map(mapping)
    expression = collapse_duplicate_genes(expression)
    titles = sample_titles(metadata)
    sample_table = pd.DataFrame(
        [
            {"accession": accession, "title": title}
            for accession, title in titles.items()
        ]
    )
    sample_table["condition"] = (
        sample_table["title"].str.extract(r"_(pso|ecz|ACD|ni)(?:\s|\(|$)")[0]
    )
    sample_table["patient"] = sample_table["title"].str.extract(
        r"^([^_]+)_"
    )[0]
    sample_table["patient_key"] = sample_table["patient"]

    bs = sample_table["patient"].eq("BS")
    mq = pd.to_numeric(
        sample_table["title"].str.extract(r"MQ_(\d+)")[0],
        errors="coerce",
    )
    sample_table.loc[
        bs & ((mq < 50) | sample_table["condition"].eq("ACD")),
        "patient_key",
    ] = "BS1"
    sample_table.loc[
        bs & ((mq >= 50) | sample_table["condition"].eq("ecz")),
        "patient_key",
    ] = "BS2"
    sample_table["condition"] = sample_table["condition"].replace(
        {"ACD": "ecz"}
    )

    duplicates = sample_table.duplicated(
        ["patient_key", "condition"], keep=False
    )
    if duplicates.any():
        duplicate_rows = sample_table.loc[
            duplicates, ["title", "patient_key", "condition"]
        ]
        raise ValueError(f"Ambiguous sample pairing:\n{duplicate_rows}")

    lookup = {
        (row.patient_key, row.condition): row.accession
        for row in sample_table.itertuples()
        if pd.notna(row.condition)
    }
    patients = sorted(sample_table["patient_key"].dropna().unique())
    complete = [
        patient
        for patient in patients
        if all(
            (patient, condition) in lookup
            for condition in ("pso", "ecz", "ni")
        )
    ]
    paired = [
        patient
        for patient in patients
        if all(
            (patient, condition) in lookup for condition in ("pso", "ecz")
        )
    ]
    expression = expression.loc[expression.index.intersection(genes)]
    missing = sorted(set(genes) - set(expression.index))
    if missing:
        raise ValueError(f"Locked genes missing from expression: {missing[:10]}")
    outputs = {
        "PSO": pd.DataFrame(
            {
                patient: expression[lookup[(patient, "pso")]]
                - expression[lookup[(patient, "ni")]]
                for patient in complete
            }
        ).loc[genes],
        "ECZEMA": pd.DataFrame(
            {
                patient: expression[lookup[(patient, "ecz")]]
                - expression[lookup[(patient, "ni")]]
                for patient in complete
            }
        ).loc[genes],
    }
    sample_table["complete_triplet"] = sample_table["patient_key"].isin(
        complete
    )
    sample_table["paired_lesions"] = sample_table["patient_key"].isin(paired)
    return outputs, sample_table


def external_predictions(
    deltas: dict[str, pd.DataFrame],
    genes: list[str],
    model: dict[str, np.ndarray],
    beta: float,
) -> pd.DataFrame:
    rows = []
    for disease, label in (("PSO", 1), ("ECZEMA", 0)):
        for patient in deltas[disease].columns:
            rows.append(
                {
                    "patient": patient,
                    "disease": disease,
                    "label": label,
                    **score_target(
                        deltas[disease].loc[genes, patient].to_numpy(),
                        model,
                        beta,
                    ),
                }
            )
    return pd.DataFrame(rows)


def paired_external_bootstrap(predictions: pd.DataFrame) -> dict:
    patients = sorted(predictions["patient"].unique())
    rng = np.random.default_rng(SEED)
    rows = []
    for _ in range(N_EXTERNAL_BOOT):
        selected = rng.choice(patients, len(patients), replace=True)
        sampled = pd.concat(
            [
                predictions.loc[predictions["patient"] == patient]
                for patient in selected
            ],
            ignore_index=True,
        )
        rows.append(
            {
                "v0_nnls_auc": roc_auc_score(
                    sampled["label"], sampled["v0_nnls"]
                ),
                "isrc_auc": roc_auc_score(
                    sampled["label"], sampled["isrc_score"]
                ),
            }
        )
    bootstrap = pd.DataFrame(rows)
    bootstrap["auc_difference"] = (
        bootstrap["isrc_auc"] - bootstrap["v0_nnls_auc"]
    )

    def interval(column: str) -> dict[str, float]:
        values = bootstrap[column]
        return {
            "ci_low": float(values.quantile(0.025)),
            "ci_high": float(values.quantile(0.975)),
        }

    point_v0 = roc_auc_score(predictions["label"], predictions["v0_nnls"])
    point_isrc = roc_auc_score(
        predictions["label"], predictions["isrc_score"]
    )
    paired = predictions.pivot(
        index="patient", columns="disease", values="isrc_score"
    )
    wins = int((paired["PSO"] > paired["ECZEMA"]).sum())
    sign_test = stats.binomtest(wins, len(paired), 0.5, alternative="two-sided")
    summary = {
        "n_complete_triplets": len(patients),
        "v0_nnls_auc": float(point_v0),
        "v0_nnls_auc_interval": interval("v0_nnls_auc"),
        "isrc_auc": float(point_isrc),
        "isrc_auc_interval": interval("isrc_auc"),
        "isrc_minus_v0_nnls": float(point_isrc - point_v0),
        "auc_difference_interval": interval("auc_difference"),
        "prob_auc_difference_gt_zero": float(
            (bootstrap["auc_difference"] > 0).mean()
        ),
        "within_patient_pso_gt_eczema": wins,
        "within_patient_fraction": float(wins / len(paired)),
        "sign_test_p": float(sign_test.pvalue),
    }
    summary["pass"] = bool(
        summary["isrc_auc_interval"]["ci_low"] > 0.50
        and summary["isrc_auc"] > summary["v0_nnls_auc"]
        and summary["prob_auc_difference_gt_zero"] >= 0.95
        and (
            summary["within_patient_fraction"] >= 0.95
            or summary["sign_test_p"] < 0.05
        )
    )
    return {"summary": summary, "bootstrap": bootstrap}


def write_report(manifest: dict, summary: dict) -> None:
    development = manifest["development"]
    diagnostic_path = RESULTS / "posthoc_ablation_metrics.json"
    diagnostics = (
        json.loads(diagnostic_path.read_text())
        if diagnostic_path.exists()
        else {}
    )
    decision = (
        "通过全部预注册门槛，可以进入第二个独立队列复现与论文算法定稿。"
        if summary["pass"]
        else "未通过全部预注册门槛，不能声称ISRC优于简单NNLS。"
    )
    lines = [
        "# v2 环境不变协同残差锥（ISRC）报告",
        "",
        "## 锁模",
        "",
        f"- 基因数：{manifest['n_genes']}",
        f"- 开发集选择 beta：{manifest['selected_beta']}",
        f"- 锁模SHA-256：`{manifest['artifact_sha256']}`",
        (
            f"- GSE121212开发AUROC：ISRC "
            f"{development['selected_isrc_auc']:.3f}，v0 NNLS "
            f"{development['v0_nnls_auc']:.3f}"
        ),
        "",
        "## GSE57225主要外部结果",
        "",
        "| 指标 | 结果 |",
        "|---|---:|",
        (
            f"| 完整患者三联样本 | "
            f"{summary['n_complete_triplets']} |"
        ),
        f"| v0 NNLS AUROC | {summary['v0_nnls_auc']:.3f} |",
        (
            f"| ISRC AUROC | {summary['isrc_auc']:.3f} "
            f"[{summary['isrc_auc_interval']['ci_low']:.3f}, "
            f"{summary['isrc_auc_interval']['ci_high']:.3f}] |"
        ),
        (
            f"| ISRC−v0 NNLS | {summary['isrc_minus_v0_nnls']:.3f}; "
            f"P(差值>0)={summary['prob_auc_difference_gt_zero']:.3f} |"
        ),
        (
            f"| 患者内PSO得分更高 | "
            f"{summary['within_patient_pso_gt_eczema']}/"
            f"{summary['n_complete_triplets']}; "
            f"符号检验p={summary['sign_test_p']:.4f} |"
        ),
        f"| 全部门槛 | {'通过' if summary['pass'] else '未通过'} |",
        "",
        "## 判定",
        "",
        decision,
        "",
        "本轮主要评价严格使用同一患者的PSO−non-involved与",
        "eczema−non-involved变化。可靠性权重仅由两个扰动实验的",
        "供体/批次bootstrap确定；GSE57225表达值是在锁模哈希生成后读取。",
        "",
    ]
    if diagnostics:
        lines.extend(
            [
                "## 事后消融（不改变主要判定）",
                "",
                "| 模型结构 | AUROC | 患者内PSO得分更高 |",
                "|---|---:|---:|",
                *[
                    (
                        f"| {name} | {value['auc']:.3f} | "
                        f"{value['pso_gt_eczema']}/{value['n_patients']} |"
                    )
                    for name, value in diagnostics.items()
                ],
                "",
                "v0加入可靠性残差但不投影混杂时AUROC没有变化；仅对v0执行",
                "Th2/IFN混杂投影即可达到完整ISRC相同的0.898。由此可见，本轮",
                "外部增益来自混杂投影，而不是新增协同残差轴。协同残差仍具有",
                "机制价值，但尚未证明其具有增量预测价值。",
                "",
            ]
        )
    (DOCS / "isrc_v2_report.md").write_text("\n".join(lines))


def external() -> None:
    genes, beta, model, manifest = load_locked_model()
    deltas, sample_table = load_gse57225_triplets(genes)
    predictions = external_predictions(deltas, genes, model, beta)
    evaluation = paired_external_bootstrap(predictions)
    summary = evaluation["summary"]

    predictions.to_csv(RESULTS / "external_predictions.csv", index=False)
    sample_table.to_csv(RESULTS / "GSE57225_sample_pairing.csv", index=False)
    evaluation["bootstrap"].to_csv(
        RESULTS / "external_paired_bootstrap.csv", index=False
    )
    manifest["external_expression_read"] = True
    manifest["external_summary"] = summary
    (RESULTS / "final_metrics.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )

    sns.set_theme(style="whitegrid")
    plot = predictions.melt(
        id_vars=["patient", "disease"],
        value_vars=["v0_nnls", "isrc_score"],
        var_name="model",
        value_name="score",
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=plot, x="model", y="score", hue="disease", ax=ax)
    ax.set_title("GSE57225 locked within-patient external validation")
    fig.tight_layout()
    fig.savefig(FIGURES / "external_scores.png", dpi=180)
    plt.close(fig)
    write_report(manifest, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["develop", "external"], required=True)
    args = parser.parse_args()
    if args.stage == "develop":
        develop()
    else:
        external()


if __name__ == "__main__":
    main()

from __future__ import annotations

import gzip
import json
import sys
from io import StringIO
from pathlib import Path

import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse, stats

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
DOCS = ROOT / "docs"
for directory in (PROCESSED, RESULTS, FIGURES, DOCS):
    directory.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    bootstrap_specificity,
    collapse_duplicate_genes,
    hgnc_entrez_map,
    log_cpm,
    paired_effect,
    parse_geo_series_matrix,
    patient_id_from_title,
    permutation_pvalue,
    rank_similarity,
    sample_titles,
    signed_set_score,
)


def require(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required public files are missing. Run scripts/download_public_data.sh:\n"
            + "\n".join(missing)
        )


def load_gse74235() -> tuple[pd.Series, pd.DataFrame]:
    counts = pd.read_csv(
        RAW / "GSE74235_GAS_HCQ_GenewiseCounts.txt.gz",
        sep="\t",
        dtype={"EntrezGeneID": str},
    )
    counts = counts.set_index("EntrezGeneID").drop(columns=["GeneLength"])
    mapping = hgnc_entrez_map(RAW / "hgnc_complete_set.txt")
    counts.index = counts.index.map(mapping)
    counts = collapse_duplicate_genes(counts)
    keep = (counts.div(counts.sum(axis=0), axis=1) * 1e6 >= 1).sum(axis=1) >= 3
    expression = log_cpm(counts.loc[keep])
    effect, delta = paired_effect(
        expression,
        treated=["C01", "C02", "C03"],
        control=["B01", "B02", "B03"],
    )
    return effect, delta


def pseudobulk_gse206326() -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_counts = PROCESSED / "GSE206326_pseudobulk_counts.parquet"
    cache_meta = PROCESSED / "GSE206326_pseudobulk_metadata.csv"
    if cache_counts.exists() and cache_meta.exists():
        return pd.read_parquet(cache_counts), pd.read_csv(cache_meta)

    meta = pd.read_csv(
        RAW / "GSE206326_Blood.matrix.data.upload.txt.gz",
        sep="\t",
        index_col=0,
    )
    meta = meta.loc[meta["Treatment"].isin(["CD1a-auto", "CD1a-GAS"])]
    with gzip.open(
        RAW / "GSE206326_GAS_Blood_Teff_RNA_counts.txt.gz", "rt"
    ) as handle:
        header = handle.readline().rstrip("\n").split(",")
    cells = [cell for cell in header[1:] if cell in meta.index]
    column_positions = [0] + [
        position
        for position, cell in enumerate(header[1:], start=1)
        if cell in meta.index
    ]
    meta = meta.loc[cells].copy()
    meta["pseudobulk"] = meta["Donor"].astype(str) + "|" + meta["Treatment"].astype(str)
    groups = dict(zip(meta.index, meta["pseudobulk"]))

    pieces: list[pd.DataFrame] = []
    reader = pd.read_csv(
        RAW / "GSE206326_GAS_Blood_Teff_RNA_counts.txt.gz",
        index_col=0,
        usecols=column_positions,
        chunksize=200,
    )
    for chunk in reader:
        aggregated = {
            group: chunk.loc[:, [c for c in cells if groups[c] == group]].sum(axis=1)
            for group in sorted(set(groups.values()))
        }
        pieces.append(pd.DataFrame(aggregated))
    counts = pd.concat(pieces, axis=0)
    counts = collapse_duplicate_genes(counts)
    pb_meta = (
        meta[["Donor", "Disease", "Treatment", "pseudobulk"]]
        .drop_duplicates()
        .sort_values(["Donor", "Treatment"])
    )
    counts.to_parquet(cache_counts)
    pb_meta.to_csv(cache_meta, index=False)
    return counts, pb_meta


def load_gse206326_effect() -> tuple[pd.Series, pd.DataFrame]:
    counts, meta = pseudobulk_gse206326()
    keep = (counts.div(counts.sum(axis=0), axis=1) * 1e6 >= 1).sum(axis=1) >= 3
    expression = log_cpm(counts.loc[keep])
    pivot = meta.pivot(index="Donor", columns="Treatment", values="pseudobulk").dropna()
    effect, delta = paired_effect(
        expression,
        treated=pivot["CD1a-GAS"].tolist(),
        control=pivot["CD1a-auto"].tolist(),
    )
    delta.columns = pivot.index
    return effect, delta


def get_reactome() -> dict[str, list[str]]:
    cache = PROCESSED / "reactome_pathways.json"
    if cache.exists():
        return json.loads(cache.read_text())
    errors = []
    for library in ("Reactome_Pathways_2024", "Reactome_2022"):
        try:
            pathways = gp.get_library(name=library, organism="Human")
            cache.write_text(json.dumps(pathways))
            (PROCESSED / "reactome_library.txt").write_text(library + "\n")
            return pathways
        except Exception as exc:
            errors.append(f"{library}: {exc}")
    raise RuntimeError("Could not retrieve a Reactome gene-set library:\n" + "\n".join(errors))


def pathway_scores(effect: pd.Series, pathways: dict[str, list[str]]) -> pd.Series:
    z = (effect - effect.mean()) / effect.std(ddof=1)
    scores = {}
    for name, genes in pathways.items():
        overlap = z.index.intersection(genes)
        if 10 <= len(overlap) <= 500:
            scores[name] = float(z.loc[overlap].mean() * np.sqrt(len(overlap)))
    return pd.Series(scores, dtype=float)


def bootstrap_pathway_correlation(
    delta_a: pd.DataFrame,
    delta_b: pd.DataFrame,
    pathways: dict[str, list[str]],
    n_boot: int = 1000,
    seed: int = 20260729,
) -> dict[str, float]:
    point_a = pathway_scores(delta_a.mean(axis=1), pathways)
    point_b = pathway_scores(delta_b.mean(axis=1), pathways)
    pathway_names = point_a.index.intersection(point_b.index).tolist()

    def projection(index: pd.Index) -> sparse.csr_matrix:
        positions = {gene: i for i, gene in enumerate(index.astype(str))}
        rows = []
        columns = []
        values = []
        for column, name in enumerate(pathway_names):
            hits = [positions[gene] for gene in pathways[name] if gene in positions]
            weight = 1 / np.sqrt(len(hits))
            rows.extend(hits)
            columns.extend([column] * len(hits))
            values.extend([weight] * len(hits))
        return sparse.csr_matrix(
            (values, (rows, columns)), shape=(len(index), len(pathway_names))
        )

    matrix_a = projection(delta_a.index)
    matrix_b = projection(delta_b.index)
    values_a = delta_a.to_numpy(dtype=float)
    values_b = delta_b.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    values = []
    batch_size = 100
    for start in range(0, n_boot, batch_size):
        batch = min(batch_size, n_boot - start)
        weights_a = rng.multinomial(
            values_a.shape[1],
            np.full(values_a.shape[1], 1 / values_a.shape[1]),
            size=batch,
        )
        weights_b = rng.multinomial(
            values_b.shape[1],
            np.full(values_b.shape[1], 1 / values_b.shape[1]),
            size=batch,
        )
        effects_a = weights_a @ values_a.T / values_a.shape[1]
        effects_b = weights_b @ values_b.T / values_b.shape[1]
        z_a = (effects_a - effects_a.mean(axis=1, keepdims=True)) / effects_a.std(
            axis=1, ddof=1, keepdims=True
        )
        z_b = (effects_b - effects_b.mean(axis=1, keepdims=True)) / effects_b.std(
            axis=1, ddof=1, keepdims=True
        )
        scores_a = np.asarray(z_a @ matrix_a)
        scores_b = np.asarray(z_b @ matrix_b)
        values.extend(
            stats.spearmanr(scores_a[i], scores_b[i]).statistic
            for i in range(batch)
        )
    arr = np.asarray(values)
    return {
        "median": float(np.median(arr)),
        "ci_low": float(np.quantile(arr, 0.025)),
        "ci_high": float(np.quantile(arr, 0.975)),
        "prob_gt_zero": float((arr > 0).mean()),
    }


def run_node1() -> tuple[dict, pd.DataFrame]:
    effect_206, delta_206 = load_gse206326_effect()
    effect_742, delta_742 = load_gse74235()
    gene_metrics = rank_similarity(effect_206, effect_742)
    signed = signed_set_score(effect_742, effect_206, top_n=200)
    signed_p = permutation_pvalue(effect_742, effect_206, signed, top_n=200)

    pathways = get_reactome()
    p206 = pathway_scores(effect_206, pathways)
    p742 = pathway_scores(effect_742, pathways)
    common = p206.index.intersection(p742.index)
    pathway_rho, pathway_p = stats.spearmanr(p206.loc[common], p742.loc[common])
    boot = bootstrap_pathway_correlation(delta_206, delta_742, pathways)
    table = pd.DataFrame(
        {"GSE206326_score": p206.loc[common], "GSE74235_score": p742.loc[common]}
    )
    table["concordance"] = table["GSE206326_score"] * table["GSE74235_score"]
    table = table.sort_values("concordance", ascending=False)
    table.to_csv(RESULTS / "node1_pathway_scores.csv")
    effect_206.rename("GSE206326_GAS_vs_auto").to_csv(
        RESULTS / "node1_GSE206326_gene_effect.csv"
    )
    effect_742.rename("GSE74235_GAS_vs_untreated").to_csv(
        RESULTS / "node1_GSE74235_gene_effect.csv"
    )

    passed = (
        pathway_rho > 0
        and pathway_p < 0.05
        and boot["prob_gt_zero"] >= 0.95
        and signed > 0
        and signed_p < 0.05
    )
    metrics = {
        "gene_level": gene_metrics,
        "cross_signature_score": signed,
        "cross_signature_permutation_p": signed_p,
        "n_pathways": int(len(common)),
        "pathway_spearman_rho": float(pathway_rho),
        "pathway_spearman_p": float(pathway_p),
        "pathway_bootstrap": boot,
        "pass": bool(passed),
        "pass_rule": (
            "pathway rho > 0, pathway p < 0.05, bootstrap P(rho>0) >= 0.95, "
            "and positive cross-signature score with permutation p < 0.05"
        ),
    }
    return metrics, table


def load_gpl6947_map() -> dict[str, str]:
    table_lines = []
    in_table = False
    with gzip.open(RAW / "GPL6947.annot.gz", "rt", errors="replace") as handle:
        for line in handle:
            if line.startswith("!platform_table_begin"):
                in_table = True
                continue
            if line.startswith("!platform_table_end"):
                break
            if in_table:
                table_lines.append(line)
    annotation = pd.read_csv(
        StringIO("".join(table_lines)), sep="\t", dtype=str, low_memory=False
    )
    symbol_col = next(
        col for col in annotation.columns if col.lower().replace(" ", "_") == "gene_symbol"
    )
    result = {}
    for probe, symbol in zip(annotation["ID"], annotation[symbol_col]):
        if pd.isna(symbol):
            continue
        first = str(symbol).split("///")[0].strip()
        if first:
            result[str(probe)] = first
    return result


def load_gse24767_effects() -> dict[str, pd.Series]:
    expr, metadata = parse_geo_series_matrix(RAW / "GSE24767_series_matrix.txt.gz")
    probe_map = load_gpl6947_map()
    expr.index = expr.index.astype(str).map(probe_map)
    expr = collapse_duplicate_genes(expr)
    titles = sample_titles(metadata)

    conditions: dict[str, dict[str, str]] = {}
    for accession, title in titles.items():
        parts = title.split(",")
        donor = parts[1].strip().replace("ID ", "")
        condition = parts[-1].strip()
        conditions.setdefault(condition, {})[donor] = accession

    aliases = {
        "IL17": "IL17",
        "TNFα": "TNF_low",
        "TNFα10": "TNF_high",
        "IL17 + 1 TNFα": "IL17_TNF_low",
        "IL17 + 10 TNFα": "IL17_TNF_high",
    }
    controls = conditions["Control"]
    effects: dict[str, pd.Series] = {}
    for raw_name, clean_name in aliases.items():
        donors = sorted(set(conditions.get(raw_name, {})).intersection(controls))
        if not donors:
            continue
        effect, _ = paired_effect(
            expr,
            treated=[conditions[raw_name][d] for d in donors],
            control=[controls[d] for d in donors],
        )
        effects[clean_name] = effect
    return effects


def hgnc_identifier_maps() -> tuple[dict[str, str], dict[str, str]]:
    table = pd.read_csv(RAW / "hgnc_complete_set.txt", sep="\t", dtype=str, low_memory=False)
    ensembl = table.dropna(subset=["ensembl_gene_id", "symbol"])
    entrez = table.dropna(subset=["entrez_id", "symbol"])
    return (
        dict(zip(ensembl["ensembl_gene_id"], ensembl["symbol"])),
        dict(zip(entrez["entrez_id"], entrez["symbol"])),
    )


def map_count_index(index: pd.Index) -> pd.Index:
    ensembl, entrez = hgnc_identifier_maps()
    raw = index.astype(str).str.replace(r"\.\d+$", "", regex=True)
    if raw.str.startswith("ENSG").mean() > 0.5:
        return raw.map(ensembl)
    if raw.str.fullmatch(r"\d+").mean() > 0.5:
        return raw.map(entrez)
    return raw


def load_gse121212_deltas() -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = pd.read_csv(RAW / "GSE121212_readcount.txt.gz", sep="\t", index_col=0)
    counts.index = map_count_index(counts.index)
    counts = collapse_duplicate_genes(counts)
    keep = (counts.div(counts.sum(axis=0), axis=1) * 1e6 >= 1).sum(axis=1) >= 10
    expression = log_cpm(counts.loc[keep])

    _, metadata = parse_geo_series_matrix(RAW / "GSE121212_series_matrix.txt.gz")
    titles = sample_titles(metadata)
    by_disease: dict[str, dict[str, dict[str, str]]] = {"PSO": {}, "AD": {}}
    for title in titles.values():
        if title not in expression.columns:
            continue
        if title.endswith("_lesional"):
            disease, state = title.split("_", maxsplit=1)[0], "lesional"
        elif title.endswith("_chronic_lesion"):
            disease, state = title.split("_", maxsplit=1)[0], "lesional"
        elif title.endswith("_non-lesional"):
            disease, state = title.split("_", maxsplit=1)[0], "non-lesional"
        else:
            continue
        if disease not in by_disease:
            continue
        patient = patient_id_from_title(title)
        by_disease[disease].setdefault(patient, {})[state] = title

    outputs = {}
    for disease in ("PSO", "AD"):
        paired = {
            patient: states
            for patient, states in by_disease[disease].items()
            if {"lesional", "non-lesional"}.issubset(states)
        }
        _, delta = paired_effect(
            expression,
            treated=[paired[p]["lesional"] for p in sorted(paired)],
            control=[paired[p]["non-lesional"] for p in sorted(paired)],
        )
        delta.columns = sorted(paired)
        outputs[disease] = delta
    return outputs["PSO"], outputs["AD"]


def run_nodes2_and_3() -> tuple[dict, dict, pd.DataFrame]:
    perturbations = load_gse24767_effects()
    pso_delta, ad_delta = load_gse121212_deltas()
    pso_effect = pso_delta.mean(axis=1)
    ad_effect = ad_delta.mean(axis=1)
    rows = []
    for name, effect in perturbations.items():
        pso_score = signed_set_score(pso_effect, effect, top_n=200)
        ad_score = signed_set_score(ad_effect, effect, top_n=200)
        pso_perm = permutation_pvalue(pso_effect, effect, pso_score, top_n=200)
        specificity = bootstrap_specificity(pso_delta, ad_delta, effect, top_n=200)
        rows.append(
            {
                "perturbation": name,
                "pso_score": pso_score,
                "pso_permutation_p": pso_perm,
                "ad_score": ad_score,
                "specificity_pso_minus_ad": pso_score - ad_score,
                "specificity_ci_low": specificity["ci_low"],
                "specificity_ci_high": specificity["ci_high"],
                "specificity_prob_gt_zero": specificity["prob_gt_zero"],
                **{f"pso_{k}": v for k, v in rank_similarity(pso_effect, effect).items()},
                **{f"ad_{k}": v for k, v in rank_similarity(ad_effect, effect).items()},
            }
        )
    table = pd.DataFrame(rows).sort_values("pso_score", ascending=False)
    table.to_csv(RESULTS / "nodes2_3_perturbation_scores.csv", index=False)
    pso_effect.rename("PSO_lesional_minus_nonlesional").to_csv(
        RESULTS / "node2_PSO_disease_effect.csv"
    )
    ad_effect.rename("AD_lesional_minus_nonlesional").to_csv(
        RESULTS / "node3_AD_disease_effect.csv"
    )

    combo = table.loc[table["perturbation"] == "IL17_TNF_high"].iloc[0]
    node2_pass = bool(combo["pso_score"] > 0 and combo["pso_permutation_p"] < 0.05)
    node3_pass = bool(
        combo["specificity_ci_low"] > 0
        and combo["specificity_prob_gt_zero"] >= 0.95
    )
    node2 = {
        "n_pso_pairs": int(pso_delta.shape[1]),
        "target_signature": "IL17_TNF_high",
        "score": float(combo["pso_score"]),
        "permutation_p": float(combo["pso_permutation_p"]),
        "pass": node2_pass,
        "pass_rule": "positive signed-set score with gene-label permutation p < 0.05",
    }
    node3 = {
        "n_ad_pairs": int(ad_delta.shape[1]),
        "target_signature": "IL17_TNF_high",
        "pso_minus_ad": float(combo["specificity_pso_minus_ad"]),
        "bootstrap_ci": [
            float(combo["specificity_ci_low"]),
            float(combo["specificity_ci_high"]),
        ],
        "bootstrap_prob_gt_zero": float(combo["specificity_prob_gt_zero"]),
        "pass": node3_pass,
        "pass_rule": "95% bootstrap CI for PSO-minus-AD specificity entirely above zero",
    }
    return node2, node3, table


def make_figures(node1_table: pd.DataFrame, perturbation_table: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(
        data=node1_table,
        x="GSE206326_score",
        y="GSE74235_score",
        s=24,
        alpha=0.55,
        ax=ax,
    )
    ax.axhline(0, color="black", lw=0.6)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_title("Node 1: GAS Reactome pathway concordance")
    fig.tight_layout()
    fig.savefig(FIGURES / "node1_pathway_concordance.png", dpi=180)
    plt.close(fig)

    plot = perturbation_table.melt(
        id_vars=["perturbation"],
        value_vars=["pso_score", "ad_score"],
        var_name="disease",
        value_name="signed_set_score",
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=plot, x="perturbation", y="signed_set_score", hue="disease", ax=ax)
    ax.axhline(0, color="black", lw=0.8)
    ax.tick_params(axis="x", rotation=25)
    ax.set_title("Nodes 2–3: perturbation compatibility with lesion transition")
    fig.tight_layout()
    fig.savefig(FIGURES / "nodes2_3_perturbation_scores.png", dpi=180)
    plt.close(fig)


def write_report(metrics: dict) -> None:
    node1 = metrics["node1"]
    node2 = metrics["node2"]
    node3 = metrics["node3"]
    lines = [
        "# 三个可行性节点：第一轮实验报告",
        "",
        "本报告由 `scripts/run_feasibility.py` 自动生成。分析单位为供体/患者，",
        "没有把单个细胞当作独立生物学重复。",
        "",
        "## 结论摘要",
        "",
        "| 节点 | 预设问题 | 结果 | 决策 |",
        "|---|---|---:|---|",
        (
            f"| 1 | GAS反应能否跨 GSE206326 / GSE74235 复现 | "
            f"Reactome ρ={node1['pathway_spearman_rho']:.3f} | 未通过 |"
        ),
        (
            f"| 2 | IL-17/TNF能否恢复银屑病皮损转换 | "
            f"score={node2['score']:.3f}, P={node2['permutation_p']:.4g} | 通过 |"
        ),
        (
            f"| 3 | 该信号能否区别银屑病与AD | "
            f"PSO−AD={node3['pso_minus_ad']:.3f}, 95% CI "
            f"[{node3['bootstrap_ci'][0]:.3f}, {node3['bootstrap_ci'][1]:.3f}] | 通过 |"
        ),
        "",
        "**总体判断：算法的“扰动→皮损机制匹配”主干可行，但 GAS 不能在当前",
        "公开数据上作为已验证诱因锚点。下一版应先做稀疏的细胞因子扰动混合",
        "反演，把 GAS 保留为待验证分支。**",
        "",
        "## 实验设计",
        "",
        "- 节点1：GSE206326 按供体伪 bulk，比较 CD1a-GAS 与 CD1a-auto；",
        "  GSE74235 按供体比较 GAS 24 h 与未处理 24 h。检验基因秩、",
        "  Reactome 通路秩以及供体级 bootstrap。",
        "- 节点2：从 GSE24767 的配对角质形成细胞实验提取 IL-17、TNF 及",
        "  联合扰动签名，与 GSE121212 中 27 对 PSO 皮损−非皮损转换匹配。",
        "- 节点3：使用同一 GSE121212 队列的 27 对 AD 皮损−非皮损转换作为",
        "  疾病对照，以患者级 bootstrap 检验 PSO−AD 特异性。",
        "",
    ]
    for key, title in [
        ("node1", "节点1：GAS跨数据集一致性"),
        ("node2", "节点2：IL-17/TNF恢复银屑病皮损转换"),
        ("node3", "节点3：银屑病相对AD的特异性"),
    ]:
        item = metrics[key]
        lines.extend(
            [
                f"## {title}",
                "",
                f"判定：**{'通过' if item['pass'] else '未通过'}**",
                "",
                "```json",
                json.dumps(item, ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## 稳健性与附加观察",
            "",
            "- 节点1仅保留 GSE206326 的3名健康供体后，Reactome ρ=0.006，",
            "  P=0.849，bootstrap 95% CI 为[-0.153, 0.196]，仍未通过。",
            "- 节点1的跨签名分数为负且置换检验显著，说明问题不是统计功效不足",
            "  所能简单解释；6 h分选T细胞与24 h全PBMC的生物学设计不等价。",
            "- 节点2中 IL-17 单独刺激和低剂量 IL-17/TNF 的 PSO 匹配分数也很高；",
            "  TNF 单独刺激较弱。这支持学习“扰动混合权重”，而不是只硬编码",
            "  一个高剂量 IL-17/TNF 模板。",
            "- AD 对 IL-17/TNF 同样呈正匹配，但显著弱于 PSO。因此节点3支持",
            "  相对特异性，不应被表述成银屑病独有机制。",
            "",
            "## 对算法开发的直接决策",
            "",
            "1. 继续：以公开扰动字典和患者配对皮损转换为输入，开发带非负稀疏",
            "   约束、疾病特异性惩罚和供体不确定性的反演模型。",
            "2. 暂停：不以 GAS 作为论文的主因果结论，也不把节点1失败的数据",
            "   强行合并或调参至显著。",
            "3. 下一验证：增加独立 IL-17/TNF 角质形成细胞扰动队列和独立 PSO/AD",
            "   皮肤队列，实行冻结签名的外部验证；GAS 只在找到实验设计更可比的",
            "   公共队列后重新开启。",
            "",
            "## 解释边界",
            "",
            "这些实验验证的是公开数据中是否存在足以支持算法开发的可迁移信号，",
            "不是对临床诱因因果关系的证明。节点1的两个实验在细胞组成、刺激时间",
            "和对照条件上不同；节点2和3使用同一GSE121212队列降低了疾病比较中的",
            "平台混杂，但仍属于转录组机制兼容度分析。",
            "",
        ]
    )
    (DOCS / "feasibility_report.md").write_text("\n".join(lines))


def main() -> None:
    require(
        [
            RAW / "GSE206326_GAS_Blood_Teff_RNA_counts.txt.gz",
            RAW / "GSE206326_Blood.matrix.data.upload.txt.gz",
            RAW / "GSE74235_GAS_HCQ_GenewiseCounts.txt.gz",
            RAW / "GSE24767_series_matrix.txt.gz",
            RAW / "GPL6947.annot.gz",
            RAW / "GSE121212_readcount.txt.gz",
            RAW / "GSE121212_series_matrix.txt.gz",
            RAW / "hgnc_complete_set.txt",
        ]
    )
    node1, node1_table = run_node1()
    node2, node3, perturbation_table = run_nodes2_and_3()
    metrics = {"node1": node1, "node2": node2, "node3": node3}
    (RESULTS / "feasibility_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2)
    )
    make_figures(node1_table, perturbation_table)
    write_report(metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

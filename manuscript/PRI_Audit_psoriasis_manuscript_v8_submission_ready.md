# PRI-Audit: a falsification-first framework for testing patient-level gain from high-resolution psoriasis transcriptomics

[Author names to be supplied]

[Affiliations to be supplied]

Corresponding author: [Name, postal address, and e-mail to be supplied]

## Abstract

**Background:** High-resolution transcriptomic studies can provide thousands of cells or spatial spots from a small number of patients. Their biological detail is substantial, but it does not itself establish that a derived representation improves prediction for a new patient. We developed PRI-Audit (Patient-level Resolution Incremental-value Audit) to test that specific claim.

**Methods:** PRI-Audit enforces four linked conditions: biological-unit isolation; refitting of all adaptive preprocessing within the outer held-out unit; comparison against an expression reference matched for folds and feature budget; and representation-specific counterfactual diagnostics. We reconstructed three primary high-resolution GEO inputs from raw files, assigned frozen roles to 11 public psoriasis cohorts, and evaluated low-rank, fold-local spot-topic, cell-neighbourhood, and spatial-graph representations with independent-unit bootstrap inference. Diagnostic contrasts included global fitting, random measurement splits, an intentionally unmatched expression comparator, and within-sample coordinate shuffling. A fixed known-truth hierarchy simulation assessed the complete evidence gate under null, coordinate-signal, source-overlap and weak-reference scenarios. An independently selected public Visium cohort was used as a strictly labelled expression/PCA boundary replication.

**Results:** Raw reconstruction was exact for GSE151177 (18 samples × 33,538 genes), GSE173706 (33 × 33,538), and GSE206391 (66 × 16,685). Strict expression AUROC was 0.946, 1.000, and 0.908 in the three respective primary cohorts. PCA and NMF increments over matched expression were +0.018 (95% CI 0.000 to 0.097) and +0.036 (0.000 to 0.156) in GSE151177, 0.000 in GSE173706, and +0.002 (−0.024 to 0.038) and −0.036 (−0.099 to 0.003) in GSE206391. A fold-local eight-topic spot representation and expression-plus-topic fusion in GSE206391 were also lower than dimension-matched expression PCA (ΔAUROC −0.089 and −0.085). Spatial-graph and expression-plus-spatial models had increments of −0.021 (−0.087 to 0.038) and −0.026 (−0.073 to 0.014). Across 400 simulations per condition, the strict gate had zero false-positive declarations under no spatial increment and 0.755–0.903 power for a zero-mean coordinate signal; source-overlapping random cell splits had median AUROC 1.000 whereas patient-LOO performance was near chance. A weakened 16-HVG comparator manufactured an apparent PCA gain of +0.082. Coordinate shuffling left spatial-graph AUROC unchanged (0.870 observed; 0.870 null mean; P=0.563). In the independent five-patient GSE202011 boundary replication, expression and PCA both saturated at AUROC 1.000.

**Conclusions:** In the available public psoriasis cohorts, the tested high-resolution representations did not demonstrate a reproducible incremental patient-level AUROC over matched expression. PRI-Audit does not claim that high-resolution biology is absent. It makes the stronger predictive-gain claim reproducible, falsifiable, and appropriately bounded.

**Keywords:** psoriasis; single-cell RNA sequencing; spatial transcriptomics; patient-level validation; leakage; falsification; reproducibility; public data

## 1. Introduction

Psoriasis is a chronic immune-mediated inflammatory skin disease with a recurrent lesional transcriptional programme involving keratinocyte activation, cytokine signalling, immune recruitment and tissue remodelling [1–3]. Single-cell and spatial technologies have made this programme more anatomically and cellularly specific [4–6]. These technologies are indispensable for mechanism-oriented questions. A different question, however, is often left implicit: does the additional measurement resolution produce a reproducible predictive benefit at the level at which new patients enter a study?

That distinction matters because high-resolution designs are nested. Thousands of cells or spots may originate from a small number of biopsies and patients. Measurements sharing a patient inherit biology, sample processing and batch structure; they are not thousands of independent validation cases. Splitting cells or spots at random can therefore estimate interpolation among measurements from biopsies already represented in training rather than prediction for a new biological unit [7,8]. Leakage can also enter before the supervised learner: feature selection, scaling, factorisation, clustering and graph construction fitted using all samples allow the held-out unit to influence its own representation [9,10].

Even a perfectly isolated candidate model can be judged against an inadequate control. A complex cell-state or spatial representation may be compared with an expression baseline using fewer genes, different folds or a different effective training set. High absolute AUROC then shows disease association, not incremental value created by the added resolution. This is especially consequential in psoriasis because patient-level expression can already be strongly discriminative. A strong expression reference establishes an empirical ceiling that a high-resolution representation must clear to support a claim of predictive increment.

We therefore frame the problem as a falsification task rather than a search for the highest apparent score. PRI-Audit is a reusable evaluation framework that couples: (i) a biological-unit split and resampling boundary, (ii) fold-local construction of every adaptive object, (iii) a paired comparison with a matched expression reference, and (iv) counterfactual nulls tailored to the claimed information source. It evaluates the proposition that a representation produces additional patient-level predictive information, rather than the weaker proposition that high-resolution measurements contain biology.

We apply this framework to 11 public GEO psoriasis datasets with deliberately heterogeneous roles. Three high-resolution cohorts form the primary representation benchmark; seven bulk or longitudinal cohorts provide biological/provenance context; and one in-vitro perturbation dataset is not counted as a patient-prediction fold. The primary claim is narrow: in the public cohorts tested here, have low-rank or spatial representations demonstrated a reproducible patient-level AUROC increment over a comparable expression model?

## 2. Materials and methods

### 2.1. Framework, target estimand, and frozen evidence gate

PRI-Audit was designed as an evaluation algorithm, not as a new clinical classifier. Its inferential unit was the patient, donor, or dataset-defined biological sample. Cells and spots were nested measurements and were never treated as independent test cases. The framework separates data role assignment, fold-local representation construction, matched comparison, counterfactual falsification, and evidence-to-claim translation (Figure 2).

[[FIGURE_2]]

For outer held-out biological group g and representation family m, the representation applied to unit i was

[[EQUATION_1]]

where every adaptive parameter θ, including highly-variable-gene selection, scaling, PCA/NMF factors, cell-neighbourhood dictionaries, graph summaries and classifier coefficients, was fitted using only groups other than g. Candidate and matched expression-reference predictions were then compared on the same held-out observations:

[[EQUATION_2]]

The pre-specified practical threshold was ΔAUROC ≥ 0.05. An incremental predictive claim was permitted only when the point estimate cleared this threshold, the independent-unit bootstrap lower bound was above zero, construction was strict, and the representation-specific counterfactual diagnostic was compatible with the claimed information source:

[[EQUATION_3]]

This conjunction intentionally prevents a disease-associated representation from being translated automatically into an incremental-value claim.

**Algorithm 1. PRI-Audit decision contract.** (1) Register the independent biological unit, candidate family, matched expression reference, practical-value threshold and counterfactual before inspecting results. (2) For each held-out unit, fit every adaptive transformation and classifier only in the remaining units. (3) Emit paired out-of-fold candidate and reference predictions for the identical held-out units. (4) Compute ΔAUROC and resample the independent unit for uncertainty. (5) Apply the counterfactual that breaks the candidate's claimed resolution-specific source while retaining the remaining design. (6) Report an incremental-value claim only if all components of Equation 3 are satisfied; otherwise report the bounded result and preserve the failed test.

### 2.2. Public data roles and raw-input reconstruction

All inputs were de-identified GEO datasets. Their roles were frozen before v21 analysis and are shown in Table 1. GSE151177 was the sample-level single-cell benchmark, GSE173706 the paired-donor single-cell benchmark, and GSE206391 the paired-patient spatial benchmark. GSE202011 was selected independently by metadata compatibility as a five-patient high-resolution expression/PCA boundary replication; its archive contained no spot-coordinate files, so it was never used or described as an external spatial-graph test. The remaining seven bulk/longitudinal datasets and GSE24767 perturbation dataset preserve biological and provenance context, rather than being miscounted as interchangeable classifier folds.

[[TABLE_1]]

We rebuilt the three primary high-resolution inputs from their public raw sources. GSE151177 and GSE173706 were reconstructed as sample-by-gene pseudobulk matrices after the frozen cell-level filtering protocol. GSE206391 was reconstructed from raw spatial matrices with in-tissue filtering. Exact canonical matrix and metadata equality was required before any v21 result was interpreted. Raw source hashes, input paths, software requirements, random seeds, output hashes, and every figure/table source are retained in the reproducibility manifest.

### 2.3. Matched expression, low-rank, and spatial representations

For each held-out unit, 2,000 highly variable genes were selected only in training data. The matched expression reference used the resulting log-transformed sample-level matrix. PCA and NMF used the same fold-specific gene budget, eligible samples, outer folds, and class-balanced logistic-regression downstream learner. This control removes the possibility that a candidate gains merely by receiving a broader feature set or a different split.

The GSE206391 spatial graph used fold-specific highly variable genes, local expression-neighbour agreement, and local contrast on a coordinate graph. A 1,000-feature spatial graph was compared with 1,000-HVG expression; a 1,500-feature expression-plus-spatial representation was compared with 1,500-HVG expression. These paired constructions were the spatial primary tests. The algorithmic modules and their permitted information hand-offs are shown in Figure 3.

[[FIGURE_3]]

For a gene h in sample i and spatial neighbourhood N(u), the graph encoder used a local agreement statistic and local contrast:

[[EQUATION_4]]

[[EQUATION_5]]

The exact feature vectors and fold outputs are preserved, so the spatial claim can be tested without asserting that graph statistics exhaust all possible spatial biology.

### 2.4. Known-truth simulation and fold-local spot-topic comparator

The real datasets cannot reveal whether a decision rule is calibrated when the true incremental value is known. We therefore fixed a hierarchy simulation with 12, 24 or 48 paired patients, two samples per patient and 40 nested measurements per sample. The data-generating process independently controlled aggregate disease expression, a coordinate-associated effect with zero pseudobulk mean, and a sample fingerprint. Four scenarios tested no spatial increment, true coordinate-specific increment, source-overlap without disease signal, and an intentionally weakened expression reference. The complete PRI-Audit gate required ΔAUROC ≥0.05 and a paired patient-bootstrap lower 95% bound above zero. The formal run used 400 repeats for every patient-count/scenario combination; the full protocol and parameters are retained in Supplementary Table S1.

To address whether the real-data comparison was restricted to simple low-rank encoders, we added a pre-specified, fold-local high-resolution spot-topic comparator in GSE206391. For every outer held-patient fold, 500 pseudobulk HVGs and an eight-topic MiniBatchNMF model were fitted only on training-patient spots. Topic activations were averaged within sample. The eight-dimensional topic model was compared with an eight-dimensional fold-local expression PCA, while the 16-dimensional expression-plus-topic fusion was compared with 16-dimensional expression PCA. The primary analysis sampled 300 spots per sample; 150 and 450 spots were held as robustness checks (Supplementary Table S2).

### 2.5. Counterfactual diagnostics and uncertainty

The analysis deliberately contrasted strict evaluation with practices that estimate different quantities. A random cell/spot split used global feature selection and PCA, then split measurement rows randomly. This was retained only to show source-sample overlap and identify its estimand. A global-versus-fully-nested cell-neighbourhood analysis compared K=8, 16 and 32 neighbourhood dictionaries. An intentionally weakened strict 16-HVG expression reference was used as a didactic mismatch sensitivity; it could never satisfy the primary evidence gate.

For spatial specificity, coordinates were shuffled within every sample while preserving counts, labels, outer folds and fold-specific HVGs. This counterfactual tests whether the selected graph representation gains patient-level information from the observed coordinate assignment. It does not test whether spatially varying expression exists. Label-permutation and bootstrap calculations resampled the dataset-specific independent unit, not cells or spots. Primary label-null calculations used 500 permutations; strict increments used 5,000 independent-unit bootstrap draws where reported. The finite-permutation tail probability used the usual correction:

[[EQUATION_6]]

### 2.6. Independent boundary replication and reproducibility

The external candidate-selection ledger was completed before matrix inspection. For GSE202011, five biologic-naive psoriasis lesion/non-lesion pairs were fixed; the first ST14 lesion technical section was selected deterministically. Strict leave-one-patient-out expression and PCA were run without re-tuning. A saturated comparison is reported as a boundary replication, not as proof of equivalence or a high-resolution advantage.

The complete analysis is orchestrated by one submission-reproduction runner. It executes raw reconstruction, strict and diagnostic comparisons, the coordinate null, external boundary analysis, table/figure creation and a final source manifest. A completed 13-step execution log records return code zero for every step. The archive is designed to permit an independent reader to reproduce both positive disease-signal results and the negative incremental-value conclusion.

## 3. Results

### 3.1. The 11-cohort landscape separates biological breadth from predictive replication

Figure 1 provides data-derived views of all 11 public datasets across paired bulk, single-cell, spatial and perturbation scales. It is descriptive: it shows the data structure that motivated the audit, not a model-selection surface. The central design consequence is visible across modalities: thousands of measurements are nested in a substantially smaller number of patients, donors or biological samples. Table 1 therefore preserves each cohort's role and prevents the seven contextual cohorts from being counted as seven extra high-resolution tests.

[[FIGURE_1]]

The raw reconstruction check succeeded before any v21 inference. GSE151177 rebuilt exactly as 18 samples × 33,538 genes, GSE173706 as 33 × 33,538, and GSE206391 as 66 × 16,685. Metadata and canonical count contents were exact. A missing mitochondrial Ensembl identifier discovered during the GSE173706 rebuild was corrected before formal analysis; it is documented rather than silently absorbed. This result turns the reported analysis into a public-input reconstruction rather than an unverifiable continuation of an earlier result directory.

### 3.2. Strict matched comparisons do not demonstrate low-rank or spatial incremental value

Strict expression was strongly discriminative in all three primary high-resolution datasets: AUROC 0.946 in GSE151177, 1.000 in GSE173706, and 0.908 in GSE206391. PCA and NMF did not cross the pre-specified incremental threshold in any primary comparison (Table 2; Figure 4A). In GSE151177, PCA and NMF had point increments of +0.018 and +0.036; their 95% intervals included zero. GSE173706 was comparison-saturated at AUROC 1.000. In GSE206391, PCA was essentially tied with matched expression (+0.002), whereas NMF was lower (−0.036).

[[TABLE_2]]

The fair spatial comparison was similarly negative for incremental patient-level value. The 1,000-feature spatial graph reached AUROC 0.870 versus its matched 1,000-HVG expression reference, for ΔAUROC −0.021 (95% CI −0.087 to 0.038). The 1,500-feature expression-plus-spatial model reached 0.875, for ΔAUROC −0.026 (−0.073 to 0.014) versus matched 1,500-HVG expression. These values do not say that spatial structure is absent; they say that these specific spatial encoders did not demonstrate an additional patient-level discrimination benefit under the specified evaluation boundary.

The fold-local spot-topic comparison did not reverse this boundary. The eight-topic representation reached AUROC 0.804 compared with 0.892 for matched eight-component expression PCA (ΔAUROC −0.089, 95% CI −0.186 to 0.000). Expression-plus-topic fusion reached 0.795 compared with 0.880 for matched 16-component expression PCA (ΔAUROC −0.085, −0.182 to 0.007). At 150 and 450 sampled spots, topic-only increments remained −0.109 and −0.094, and fused increments remained −0.068 and −0.071 (Supplementary Table S2). Thus, this additional low-capacity, strictly fold-local representation did not provide an exception to the matched-expression boundary.

[[FIGURE_4]]

### 3.3. Known-truth simulations validate the evidence gate and expose two failure modes

The controlled hierarchy simulation separates the properties of the audit from the biology of any one cohort (Figure 6; Supplementary Table S1). Under no spatial increment, the strict evidence gate made zero positive declarations in all 400 replicates at each patient count (n=12, 24, 48); the exact 95% Monte-Carlo upper bound for each zero rate was 0.009. When a disease-associated coordinate effect had zero pseudobulk mean, the gate detected it in 0.755, 0.903 and 0.808 of replicates at n=12, 24 and 48, respectively. Mean strict increments closely followed their independently simulated population targets.

The two intentionally invalid contrasts behaved in the opposite direction. With a sample fingerprint but no disease or spatial signal, patient-LOO AUROC was near chance (median 0.507, 0.519 and 0.508), whereas random-cell splitting had median AUROC 1.000 at every patient count because every test cell had a source represented in training. In the weak-reference scenario, candidate and matched expression contained the same information and strict increments remained near zero, but removing the expression-signal block from the comparator manufactured mean apparent gains of +0.459, +0.481 and +0.492. These results do not estimate psoriasis effects; they demonstrate that the evidence gate rejects the two artifacts it is designed to expose while retaining sensitivity to a controlled coordinate-specific increment.

[[FIGURE_6]]

### 3.4. Permissive evaluation can estimate a different and much more optimistic quantity

The diagnostic contrasts show why the matched, source-blocked estimand matters (Table 3; Figure 5A). Under random cell splitting in GSE151177, median cell-level AUROC was 0.999 (50 splits; 2.5–97.5% range 0.998–0.999). Under random spot splitting in GSE206391, median AUROC was 0.970 (0.966–0.974). In both diagnostics, 100% of test measurements came from source samples represented in the training set. These values therefore quantify within-source measurement interpolation, not unseen-patient prediction, and are not compared with PRI-Audit AUROCs.

The fully nested neighbourhood audit was deliberately more nuanced than a universal leakage narrative. With K=8, globally constructed versus fully nested cell-neighbourhood AUROC differed by +0.071 (95% CI 0.000 to 0.222). At K=16 the gap was 0.000 (−0.089 to 0.089), and at K=32 it was −0.054 (−0.197 to 0.000). Global–strict differences were therefore resolution-sensitive. The appropriate conclusion is not that global construction always inflates performance; it is that a global construction is a different analysis and cannot substitute for the pre-specified strict estimator.

An intentionally unmatched comparator demonstrates a second failure mode. In GSE206391, PCA had a negligible +0.002 increment over the correct matched 2,000-HVG expression reference. Replacing that reference with strict 16-HVG expression lowered reference AUROC to 0.828 and created an apparent PCA gain of +0.082 (95% CI +0.002 to +0.219). The difference is a comparator artifact, not support for PCA superiority.

[[TABLE_3]]

### 3.5. Spatial coordinate shuffling does not reveal coordinate-specific incremental value for the selected graph

Coordinate shuffling preserved every expression count, disease label, held-out patient fold and fold-specific HVG set, while permuting location within sample. The spatial-graph AUROC was 0.870; the 500-shuffle null mean was also 0.870 (tail P=0.563). For the combined expression-plus-spatial model, observed AUROC was 0.875 and null mean was 0.879 (tail P=0.677; Figure 5B). Thus, after expression and sampling structure were held fixed, the selected coordinate graph did not show a detectable patient-level contribution from the observed coordinate arrangement.

This is a representation-specific result. It neither invalidates spatial transcriptomics nor establishes that spatial mechanisms are unimportant in psoriasis. It does establish that an attractive spatial map or a high absolute spatial-model score is insufficient evidence for coordinate-specific incremental prediction in this public cohort.

[[FIGURE_5]]

### 3.6. An independent high-resolution boundary replication is saturated rather than favourable to PCA

The externally selected GSE202011 archive provided five paired lesion/non-lesion psoriasis patients. Strict leave-one-patient-out expression and PCA each attained AUROC 1.000; the PCA increment was exactly 0.000 with a paired-patient bootstrap interval of 0.000 to 0.000. The 500 paired-label-permutation P value was 0.064 for each saturated classifier. This result is not a powered equivalence test and cannot establish PCA non-inferiority. It does, however, reproduce the boundary relevant to this manuscript: when expression already separates the available held-out patients perfectly, a low-rank transform has not demonstrated added patient-level information.

GSE202011 is not used as an external spatial validation because the selected public archive did not include coordinate files. The distinction is recorded in the selection ledger and prevents a spatial generalisation from being inferred where the data do not support one.

## 4. Discussion

The central finding is methodological and empirical. In the public psoriasis cohorts considered here, patient-level disease-state expression was strong and high-resolution representations contained interpretable disease-associated structure, yet the tested low-rank and spatial encoders did not demonstrate a reproducible AUROC increment over a matched expression reference. This is not a “high-resolution data are useless” result. It is a test of the narrower, stronger assertion that resolution improves patient-level discrimination beyond a comparator that receives the same patients, folds and feature budget.

PRI-Audit's contribution is the integration of evaluation conditions that are individually familiar but are rarely enforced as one evidence gate for high-resolution transcriptomics. The known-truth simulations establish that this implemented gate has no observed false-positive declarations under the fixed non-incremental hierarchy, detects a controlled coordinate-only increment, and rejects two common artifacts. Nested preprocessing guards the outer test boundary. Matched expression prevents a feature-budget or cohort mismatch from being credited to a representation. Independent-unit bootstrap resists the false precision created by cell/spot counts. Finally, counterfactual diagnostics state what source of information a representation claims to exploit: cell reassignment for compositional representations and coordinate shuffle for coordinate-based graph features. The framework therefore yields a result that can be falsified and reused, rather than an opaque modelling recipe.

The diagnostic experiments make the practical stakes visible. Random measurement splits approach AUROC 1.0 because they repeatedly test measurements from biopsies already present in training. The deliberately weakened spatial expression comparator can manufacture an apparent +0.082 PCA gain. Neither pattern provides evidence that a future patient's disease state will be better predicted. Equally importantly, the fully nested neighbourhood analysis did not produce a fixed-sign global–strict gap across K. We therefore retire historical universal-optimism wording. The evidence supports a more disciplined claim: construction scope changes the estimand and its consequences are representation-resolution dependent.

The spatial conclusion also requires restraint. The coordinate-shuffle null was centred at the observed spatial-graph performance, but this does not prove the absence of spatial biology. It says that the particular graph summary used here did not recover a patient-level coordinate-specific increment after disease expression and sample structure were controlled. Spatial data may still be decisive for compartment-specific mechanisms, tissue architecture, biomarker localisation, or within-biopsy biology; those are different targets from cross-patient lesion discrimination.

Several limitations are inherent. Independent-unit counts remain small, and multiple cohorts are score-saturated; the study cannot establish equivalence between representation families. The 11 cohorts have heterogeneous analytical roles, so they should not be presented as 11 identically powered external prediction tests. GSE202011 strengthens the expression/PCA boundary but lacks coordinates, leaving external spatial-graph transfer untested. The selected practical threshold (+0.05 AUROC) is a transparent decision threshold, not a biological law. Finally, the framework evaluates the implemented representations, not every possible cell-state or spatial model.

The appropriate next step is not post hoc tuning until a complex model exceeds a baseline. It is prospective application of the frozen PRI-Audit contracts to larger, genuinely independent cohorts with pre-specified clinical or mechanistic endpoints. Such studies can test whether a representation gains value in a setting with headroom beyond expression, while retaining the matched-reference and biological-unit safeguards shown here.

## 5. Conclusion

PRI-Audit transforms a broad claim—“higher transcriptomic resolution improves prediction”—into a bounded test. Known-truth simulation confirms that the implemented evidence gate rejects source-overlap and weak-reference artifacts while retaining sensitivity to a controlled coordinate-only increment. Across the available public psoriasis data, disease-state signal was reproducible, but no tested low-rank, spot-topic or spatial representation demonstrated a reproducible patient-level AUROC increment over matched expression. The contribution is therefore a reusable, reproducible and falsifiable method for distinguishing biological detail from established predictive gain.

## Declarations

### Data availability

All analysed datasets are de-identified public Gene Expression Omnibus (GEO) records. The all-input registry includes GSE117239, GSE136757, GSE151177, GSE153007, GSE173706, GSE182740, GSE206391, GSE24767, GSE51440, GSE57225 and GSE85034; GSE202011 was independently added for the frozen boundary replication. The complete cohort-role registry, retrieval paths and raw-input SHA-256 checksums are available in the PRI-Audit repository at https://github.com/kobayashi0512/PRI-Audit (version v1.0.0: https://github.com/kobayashi0512/PRI-Audit/tree/v1.0.0). No controlled participant-level data are redistributed.

### Code availability

Project name: PRI-Audit. Project home page: https://github.com/kobayashi0512/PRI-Audit. Version used in this manuscript: v1.0.0 (https://github.com/kobayashi0512/PRI-Audit/tree/v1.0.0). Operating system: platform independent. Programming language: Python. Other requirements: version-locked packages are listed in reproducibility/requirements-v21.lock. The repository contains the one-command v21/v22 reproduction runner, raw-input manifests, prediction rows, bootstrap arrays, complete null distributions and figure/table source manifests. An archival DOI and the final approved software licence will be added before formal submission.

### Funding

[Funding sources and grant numbers to be supplied by the authors.]

### Author contributions

[Author initials and CRediT roles to be supplied after author order is finalized.]

### Conflicts of interest

[Conflict-of-interest declaration to be supplied by all authors.]

### Ethics statement

This study reanalysed de-identified public datasets and involved no new participant recruitment, intervention, or access to identifiable information. The authors must confirm the final institutional wording before submission.

### Acknowledgements

We thank the original investigators and participants who generated and publicly deposited the transcriptomic datasets used in this benchmark. [Additional acknowledgements to be supplied.]

## References

1. Boehncke WH, Schön MP. Psoriasis. Lancet. 2015;386:983–994. doi:10.1016/S0140-6736(14)61909-7.
2. Lowes MA, Suárez-Fariñas M, Krueger JG. Immunology of psoriasis. Annu Rev Immunol. 2014;32:227–255. doi:10.1146/annurev-immunol-032713-120225.
3. Tian S, Krueger JG, Li K, et al. Meta-analysis derived transcriptome of psoriasis defines core pathogenesis. PLoS One. 2012;7:e44274. doi:10.1371/journal.pone.0044274.
4. Kim J, Lee J, Kim HJ, et al. Single-cell transcriptomics of psoriasis immune populations. J Allergy Clin Immunol. 2021;148:1281–1292. doi:10.1016/j.jaci.2021.04.021.
5. Ma F, Plazyo O, Billi AC, et al. Single cell and spatial sequencing define inflammatory amplification in psoriasis. Nat Commun. 2023;14:3455. doi:10.1038/s41467-023-39020-4.
6. Schäbitz A, Hillig C, Mubarak M, et al. Spatial transcriptomics landscape of lesions from inflammatory skin diseases. Nat Commun. 2022;13:7729. doi:10.1038/s41467-022-35319-w.
7. Zimmerman KD, Espeland MA, Langefeld CD. A practical solution to pseudoreplication bias in single-cell studies. Nat Commun. 2021;12:738. doi:10.1038/s41467-021-21038-1.
8. Squair JW, Gautier M, Kathe C, et al. Confronting false discoveries in single-cell differential expression. Nat Commun. 2021;12:5692. doi:10.1038/s41467-021-25960-2.
9. Ambroise C, McLachlan GJ. Selection bias in gene extraction from microarray data. Proc Natl Acad Sci USA. 2002;99:6562–6566. doi:10.1073/pnas.102102699.
10. Varma S, Simon R. Bias in error estimation when using cross-validation for model selection. BMC Bioinformatics. 2006;7:91. doi:10.1186/1471-2105-7-91.
11. Van Calster B, McLernon DJ, van Smeden M, Wynants L, Steyerberg EW. Calibration: the Achilles heel of predictive analytics. BMC Med. 2019;17:230. doi:10.1186/s12916-019-1466-7.
12. Edgar R, Domrachev M, Lash AE. Gene Expression Omnibus: NCBI gene expression repository. Nucleic Acids Res. 2002;30:207–210. doi:10.1093/nar/30.1.207.

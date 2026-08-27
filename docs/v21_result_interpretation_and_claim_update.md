# PRI-Audit v21 result interpretation and claim update

**Status: frozen after raw-equivalence reconstruction, isolated
500-permutation evaluations, complete spatial coordinate nulls, random
measurement-split diagnostics, and the frozen GSE202011 boundary replication.**

## What v21 validates exactly

The raw public reconstruction is exact relative to the historical analysis
inputs:

| Dataset | Rebuilt samples × features | Metadata | Count matrix |
|---|---:|---|---|
| GSE151177 | 18 × 33,538 | exact | exact |
| GSE173706 | 33 × 33,538 | exact | exact |
| GSE206391 | 66 × 16,685 | exact | exact |

The initially observed GSE173706 mismatch was traced to a missing mitochondrial
Ensembl ID (`ENSG00000198840`) in the new QC set.  It was corrected before any
formal v21 result was interpreted.  The apparent GSE206391 metadata mismatch
was only an in-memory string/integer category representation; canonical CSV
contents were exact.

## Primary matched strict comparison

All values below are held-out AUROC increments against the same strict 2,000-HVG
expression reference.  The primary gate requires increment >=0.05 and a
positive independent-unit bootstrap lower bound.

| Dataset | Strict PCA increment (95% CI) | Strict NMF increment (95% CI) | Interpretation |
|---|---|---|---|
| GSE151177 | +0.018 (0.000, 0.097) | +0.036 (0.000, 0.156) | neither clears the gate |
| GSE173706 | 0.000 (0.000, 0.000) | 0.000 (0.000, 0.000) | expression comparison saturated at AUROC 1.000 |
| GSE206391 | +0.002 (−0.024, 0.038) | −0.036 (−0.099, 0.003) | no incremental evidence |
| GSE47751 external bulk | 0.000 (0.000, 0.000) | not applicable | both expression and PCA saturated at AUROC 1.000 in six baseline pairs |

Thus no low-rank representation satisfies the pre-specified reproducible
incremental claim.  The external GSE47751 result is a biologically consistent,
strict bulk boundary replication; because its comparison is saturated and it
does not contain high-resolution data, it cannot validate a representation
advantage.

## Independent high-resolution boundary replication

The independently selected GSE202011 10x Visium archive supplies five paired
biologic-naive PSO lesional/non-lesional patients (10 samples) after the
pre-specified selection of the first ST14 lesion section over its technical
replicate.  Strict leave-one-patient-out expression and PCA both achieved
AUROC 1.000; the PCA increment was exactly 0.000 (paired-patient bootstrap
95% CI 0.000–0.000; 500 paired-label permutations p=0.064 for each saturated
classifier).  This is a small, saturated boundary replication—not evidence of
PCA superiority and not a powered equivalence study.

The retrieved H5 archive does not contain spot-coordinate files.  It therefore
cannot supply an external spatial-graph test and is never described as one.
This limitation is explicit in the external-candidate ledger.

## Why matched references are non-negotiable

On paired spatial GSE206391, strict PCA versus the matched expression baseline
has an increment of only +0.002.  Replacing that baseline with an intentionally
unmatched strict 16-HVG expression classifier lowers the reference AUROC from
0.908 to 0.828 and creates an apparent PCA gain of +0.082 (paired-patient
bootstrap 95% CI 0.002–0.219).  This is a comparator artifact, not algorithmic
evidence.  It is retained as a clearly labelled didactic sensitivity only.

## Fully nested neighbourhood result supersedes the historical partial check

The v21 neighbourhood implementation refits HVG selection, scaling, PCA, and
K-means in each held-out sample fold.  This is stricter than the historical
check, which only refit the cluster dictionary after a global cell embedding.

| K | Global AUROC | Fully nested AUROC | Global − fully nested | 95% bootstrap CI |
|---:|---:|---:|---:|---|
| 8 | 0.982 | 0.911 | +0.071 | 0.000, 0.222 |
| 16 | 0.964 | 0.964 | 0.000 | −0.089, 0.089 |
| 32 | 0.946 | 1.000 | −0.054 | −0.197, 0.000 |

This sensitivity pattern does **not** support a universal statement that global
neighbourhood construction causes a positive optimism gap.  The older
large-gap result must not remain a main-text claim.  The defensible conclusion
is narrower: neighbourhood performance and the global–strict difference are
resolution-sensitive; no stable incremental patient-level advantage over the
matched expression reference has been demonstrated.

## Fair spatial comparison and complete coordinate null

The spatial analysis now matches feature budgets before comparing methods.  A
1,000-feature spatial graph is compared with 1,000-HVG expression; a
1,500-feature expression-plus-spatial graph is compared with 1,500-HVG
expression.  HVG selection, graph construction and classifier fitting occur
inside each held-out-patient fold.

| GSE206391 representation | Held-out patient AUROC | Increment against matched expression (95% CI) |
|---|---:|---:|
| Spatial graph (1,000 features) | 0.870 | −0.021 (−0.087, 0.038) |
| Expression + spatial (1,500 features) | 0.875 | −0.026 (−0.073, 0.014) |

For the corresponding coordinate-shuffle tests, coordinates were permuted
within every sample while counts, labels, outer folds and fold-specific HVGs
were kept fixed (500 shuffles).  The spatial-graph observed AUROC was 0.870,
versus null mean 0.870 (tail p=0.563); the combined representation was 0.875,
versus null mean 0.879 (tail p=0.677).  These results do **not** license the
claim that spatial organisation is biologically absent.  They show that this
spatial representation has not demonstrated a patient-level increment over
its matched expression reference in the available public cohort.

## Random cell/spot split is an invalid, optimistic estimand

We also ran the common permissive recipe of globally selecting features and
fitting PCA, then randomly splitting measurement rows.  In GSE151177 the
median cell-level AUROC was 0.999 (50 splits; 2.5–97.5% range 0.998–0.999),
and in GSE206391 the median spot-level AUROC was 0.970 (0.966–0.974).  In both
cases **100% of test cells/spots originated from source samples already seen
in training**.  Thus these values estimate within-biopsy measurement
interpolation, not new-sample or new-patient discrimination.  They are
retained as a falsification diagnostic and must never be pooled with the
source-blocked PRI-Audit AUROCs.

## All-11-cohort provenance boundary

`v21_all11_input_registry.csv` and `v21_all11_provenance_manifest.json` now
open and SHA-256 verify all eleven raw GEO inputs, map them to their frozen
analysis scripts, and hash the expected output artifacts.  This is important
because the 11 cohorts are not 11 interchangeable classifier folds: the three
high-resolution cohorts implement the representation benchmark, the seven
bulk cohorts support paired disease/treatment contexts, and GSE24767 is an
in-vitro perturbation prior.  The manuscript must preserve those roles rather
than inflate the independent validation count.

## Consequences for the manuscript

1. Retain the falsification-first framework, exact raw reproducibility, matched
   reference rule, independent-unit inference, and the mismatch-control
   demonstration.
2. Replace any main-text claim of a systematic positive global–strict
   neighbourhood optimism gap with the resolution-sensitivity result above.
3. Demote historical partial strict-neighbourhood outputs to a labelled legacy
   sensitivity, or remove them entirely from the primary paper.
4. Describe GSE47751 as external bulk boundary replication with comparison
   saturation, not as a high-resolution external validation.
5. The current main quantitative figures are
   `figures/paper_v21/fig11_v21_strictness_and_reference_audit.png` and
   `figures/paper_v21/fig12_v21_independence_and_spatial_null.png`.

## Remaining work before an updated full manuscript

* Create the submission one-command runner, exact figure/table source manifest,
  and a clean-environment execution log.
* Regenerate the manuscript results table and captions from v21 outputs.
* Update the claim-freeze document, Methods, Results, Discussion, Extended Data
  table of candidate selection, and data/code availability section.
* Run an adversarial simulated peer review, select journals, and render and
  visually inspect the revised `.docx` after all numerical substitutions are
  complete.

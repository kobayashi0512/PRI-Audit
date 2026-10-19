# PRI-Audit v21: reproducibility and boundary-validation specification

## Purpose

This version turns the frozen PRI-Audit claim into a fully rerunnable experiment
package.  It is **not** a search for a more favourable model.  Its purpose is to
test whether the reported conclusion is stable when the raw public inputs are
rebuilt in a clean result directory, and to make the consequences of loose
evaluation visible with a matched, pre-specified contrast.

## Frozen scientific claim

For public psoriasis datasets, high-resolution representations may contain
disease-associated structure but are not established as reproducibly improving
patient-level discrimination beyond a matched expression reference under
outer-fold refitting, patient/donor-aware splitting, and resolution-appropriate
null tests.

No analysis in this version may tune a representation, feature count,
regularisation constant, or decision threshold to reverse this conclusion.

## Versioned inputs and data scope

| Role | Accession / source | Unit used | Primary comparison |
|---|---|---:|---|
| Single-cell discovery | GSE151177 | one sample per outer fold | psoriasis vs control |
| Paired single-cell replication | GSE173706 | one donor per outer fold | PP vs PN |
| Paired spatial replication | GSE206391 | one patient per outer fold | LS vs NL |
| Existing bulk biological replication | seven pre-existing GEO cohorts | cohort-level | recurrent LS/NL direction |
| Independent high-resolution boundary replication | GSE202011 | one paired PSO patient per outer fold | LS vs NL, strict expression vs PCA |

The new external candidate must be independent of all development and figure
selection.  It must have a documented lesion/normal phenotype, sufficient
individual-level labels to define its unit of resampling, and public raw or
processed data adequate for deterministic retrieval.  Treatment-only and
unpaired before/after studies are retained only for biological-direction
context; they are not called external diagnostic validation.

## Pre-specified experiment A: raw-input reconstruction

1. Rebuild GSE151177 pseudobulk counts from raw 10x matrices, with the existing
   QC rule (library size >=500, detected genes >=300, mitochondrial fraction
   <=0.20) and sample-level QC exclusions from the registered QC reports.
2. Rebuild GSE173706 donor/sample counts from raw cell-by-gene CSV files using
   the same QC rule.
3. Rebuild GSE206391 sample/spot counts from the public H5 matrix using the
   same QC rule and in-tissue requirement.
4. Save SHA-256 hashes, file paths, package versions, exact sample metadata,
   seeds, and output hashes in an isolated `results/pri_audit_v21/` directory.
5. Compare rebuilt counts and strict primary AUROCs to v10.  Exact count hashes
   are expected where input ordering is identical; otherwise numerical equality
   and identical sample metadata are required.

## Pre-specified experiment B: loose-versus-strict evaluation contrast

All comparisons use the same eligible samples, labels, outer group definitions,
classifier (class-balanced logistic regression; `C=1`), and fixed random seed.

| Arm | Feature / representation fitting | What it demonstrates | Formal role |
|---|---|---|---|
| Strict matched expression | HVG selection, scaling and PCA/NMF within each held-out group | correct patient/donor-level reference | primary |
| Strict low-rank | same nested preprocessing, then PCA or NMF | incremental high-level representation | primary |
| Loose low-rank | HVG selection, scaling and low-rank mapping fit once on all eligible samples before group split | information leakage from global representation fitting | diagnostic contrast only |
| Global cell-neighbourhood | cell neighbourhood dictionary fit globally before sample prediction | resolution-induced optimism | diagnostic contrast only |
| Strict cell-neighbourhood | dictionary/refit inside each outer sample fold | patient-level neighbourhood test | primary |
| Mismatched expression sensitivity | deliberately reduced expression reference, clearly labelled non-comparable | why unmatched controls can manufacture a gain | didactic sensitivity only |

Primary estimand: `Delta = AUROC(candidate) - AUROC(strict matched expression)`.
For every arm, report group/patient bootstrap 95% CI and a label-permutation
p-value under the proper independent unit.  The result is interpreted as
support for an incremental claim only when `Delta >= 0.05`, the lower bootstrap
bound is greater than zero, and the representation-specific falsification test
passes.  A loose or mismatched contrast can never satisfy the primary gate.

## Pre-specified experiment C: independent external cohort

The audit will inspect all locally available candidate GEO records before
selection.  Selection is based only on metadata compatibility, availability,
and independence, not outcome performance.  The selected cohort will be
documented in a selection table with every candidate and its exclusion reason.

The selected candidate is GSE202011, an independent Visium archive.  Before
matrix inspection, the audit fixed five PSO LS/NL pairs and selected the first
ST14 lesion section rather than its technical replicate.  The public H5 archive
contains no coordinate files, so this cohort is a strict high-resolution
expression/PCA boundary replication, not an external spatial-graph test.  Its
final result is comparison-saturated (strict expression and PCA AUROC 1.000;
increment 0.000) and is reported with that limitation.

## Deliverables mapped to manuscript

| Deliverable | Target artifact | Manuscript use |
|---|---|---|
| Raw rebuild manifest and input hashes | `results/pri_audit_v21/reproducibility_manifest.json` | Methods / Data availability |
| Rebuild equivalence table | `v21_rebuild_equivalence.csv` | Extended Data Table 1 |
| Loose-versus-strict numerical table | `v21_loose_vs_strict_summary.csv` | Main Table 2 |
| Strictness/matched-reference figure | `figures/paper_v21/fig11_v21_strictness_and_reference_audit.png` | Main Figure 11 |
| Independence/spatial-null figure | `figures/paper_v21/fig12_v21_independence_and_spatial_null.png` | Main Figure 12 |
| External selection audit and result | `v21_external_cohort_audit.csv`, `gse202011_v21_external_strict.json` | Extended Data Table 2 |
| Executable orchestration, source manifest and environment capture | `scripts/run_v21_submission_reproduction.py`, `v21_figure_table_source_manifest.json`, `reproducibility/README.md` | Code availability |

## Stopping and interpretation rules

* Never alter the fixed primary recipe after seeing results.
* Do not describe a treatment-response, cell-level, spot-level, or global-fit
  result as patient-level external diagnostic validation.
* Report negative, null and adverse results with the same precision as positive
  results.
* If a raw source is inaccessible or fails integrity checks, record the reason
  and continue with the remaining pre-specified analysis; do not silently
  substitute a different accession.

> **v1.32 update.** The preceding v21 plan correctly describes the original
> expression/PCA boundary replication. The revision package adds a separate,
> post-v21 spatial-graph boundary analysis after recovering array coordinates by
> an exact barcode join to a versioned Visium V1 map. Its protocol and claim
> boundary are in `docs/pri132_revision_protocol.md`; it does not alter the
> original pre-specified v21 estimand.

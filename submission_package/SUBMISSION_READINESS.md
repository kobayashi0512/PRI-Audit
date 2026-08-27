# PRI-Audit submission-readiness record

## Scientific items

| Item | Status | Evidence / owner action |
|---|---|---|
| Raw high-resolution reconstruction | complete | `results/pri_audit_v21/v21_rebuild_equivalence.csv` |
| Strict versus permissive and unmatched-reference diagnostics | complete | Figure 11 and its source manifest |
| Random cell/spot split diagnostic | complete | Figure 12A; explicitly non-primary |
| Fair spatial graph and 500 coordinate shuffles | complete | Figure 12B and raw distributions |
| All-11 input/result provenance | complete | `v21_all11_provenance_manifest.json` |
| Independent high-resolution boundary replication | complete | GSE202011: five paired PSO patients, strict expression/PCA both AUROC 1.000 and increment 0.000; coordinate files unavailable and explicitly not claimed |
| Final claim freeze and full manuscript rewrite | complete | `manuscript/PRI_Audit_psoriasis_manuscript_v8_submission_ready.docx`; source text and v8 source manifest are hash-linked to v21 artifacts |
| Rendered Word quality assurance | complete | 16-page DOCX rendered to `tmp/pri_audit_v8_docx/rendered_v8_qa3/`; title, equations, all five figures, all three tables, declarations and references visually checked |
| Simulated peer review and rebuttal plan | complete | `submission_package/SIMULATED_PEER_REVIEW.md`; manuscript locations updated below |
| Known-truth calibration/power simulation | complete | 400 replicates × 4 fixed scenarios × 3 independent-patient counts; `results/pri_audit_v22/v22_simulation_summary.csv`, Figure 6 |
| Fold-local spot-topic comparator | complete | GSE206391, 8-topic and expression+topic comparisons against dimension-matched PCA; `gse206391_v22_spot_topic_benchmark.json` |
| Spot-subsampling sensitivity | complete | 150/300/450 sampled spots; all increments remain non-positive; `results/paper_v21/tableS2_v22_spot_topic_sensitivity.csv` |
| v22 manuscript visual QA | complete | `manuscript/PRI_Audit_psoriasis_manuscript_v9_submission_ready.docx` rendered to `tmp/pri_audit_v9_docx/rendered/` and page-level checked |
| Unified v21+v22 release runner | complete | `results/pri_audit_release/complete_reproduction_log.json`: 2026-08-21 09:35–10:27 UTC, 13/13 v21 and 8/8 v22 commands returned 0; 16 v21 and 10 v22 core-output SHA-256 hashes were recorded |
| Clean public-release staging bundle | complete | `/Users/tongyue/Documents/ppf/papers/PRI-Audit-v1.0.0-public/`: 256 curated files, 29.8 MB, zero SHA-256 mismatches, no raw GEO cache, private style reference or Python bytecode |

## Information that must be provided by the authors (never fabricated)

| Field | Required at submission |
|---|---|
| Title, author order, affiliations and corresponding author | yes |
| CRediT contributions and acknowledgements | yes |
| Funding numbers and funder names | if applicable |
| Conflicts of interest | yes, including an explicit “none declared” where appropriate |
| Ethics statement | Public, de-identified GEO reanalysis only; confirm local institutional wording |
| Code archive URL and immutable DOI | yes; create only after the final repository is frozen |
| Data availability wording | GEO accessions, retrieval date, plus archive checksum manifest |

## Required final consistency checks

1. Every AUROC, confidence interval, permutation count and sample/patient count
   in the manuscript must be regenerated from the versioned result files, not
   transcribed from historical drafts.
2. Each quantitative v21 caption must point to the corresponding row in
   `v21_figure_table_source_manifest.csv`; all five figures and three tables
   are additionally linked in `results/paper_v21/v8_submission_source_manifest.csv`.
3. State that 11 GEO cohorts have heterogeneous roles; do not call them 11
   independent patient-level classifier tests.
4. Make “no demonstrated incremental patient-level gain” the conclusion.  Do
   not write that high-resolution biology is absent or that a null result proves
   no causal role for IL-17/TNF.
5. Archive code and this manifest at a versioned commit before assigning a DOI.
6. The target is BMC Bioinformatics, Research Article. Retain the full
   methodological evidence chain; Bioinformatics Advances remains a fallback
   only if the manuscript is deliberately condensed to its live limits.

## Remaining author-owned release actions

The experimental and manuscript package is complete, but it is **not yet
submittable** until authors provide the fields listed above and perform two
external-state actions that cannot be fabricated locally: create a public code
repository/release, then obtain an immutable archive DOI. The manuscript keeps
clear placeholders rather than inventing these details.

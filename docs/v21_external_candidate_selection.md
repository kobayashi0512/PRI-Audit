# PRI-Audit v21 independent-cohort selection record

**Selection frozen before reading the candidate expression matrices.**  The
purpose is not to search for a favourable score.  It is to ask whether a
strict, sample-blocked expression reference remains necessary in an
independent public high-resolution psoriasis cohort.

## Eligibility rules

A candidate qualifies only if it is publicly retrievable, has a documented
psoriasis lesional/non-lesional contrast, identifies the independent patient
unit, and exposes a processed matrix sufficient to reconstruct a deterministic
sample-level representation.  For the spatial test it must also retain spot
coordinates.  A treatment-only, cell-type-sorted, unpaired, or controlled-data
record cannot be relabelled as a patient-level external validation.

## Candidate audit

| Candidate | Decision | Reason fixed before outcome inspection |
|---|---|---|
| GSE202011 | **Selected** | Independent Visium spatial study; public 42.2 MB H5 supplement; biologic-naive PSO LS/NL samples with patient identifiers and coordinates.  Five PSO patients have an evaluable LS/NL pair after deterministic duplicate handling. |
| GSE47751 | Retained as bulk boundary replication | Fresh official GEO retrieval has six paired baseline PN/PP samples, but no high-resolution structure; it cannot validate spatial or cell-level representation gain. |
| GSE228421 | Deferred, not counted as external validation | The GEO record provides a very large processed supplement for five longitudinally treated patients.  Its baseline LS/NL samples are potentially compatible, but it is not used in the single pre-specified external test because GSE202011 was selected first for independent spatial-platform compatibility.  It may be analysed later only as a separately labelled sensitivity. |
| GSE176509 | Excluded | Public, but only two psoriasis donors and index-sorted myeloid cells; it is a cell-subset atlas rather than a whole-skin patient-level benchmark. |
| GSE261704 | Excluded | Local metadata do not provide the required paired psoriasis LS/NL comparison. |
| GSE223799 | Excluded | Dermatologic differential-diagnosis setting; outcome and sampling design differ from LS/NL disease-state benchmarking. |
| GSE314158 | Not added to the frozen v21 external test | Recently released CosMx data are promising but use tissue-microarray cores and a distinct targeted platform; adding it after the primary external test would create an unplanned post-hoc expansion. |

## Frozen GSE202011 analysis plan

1. Retrieve the official `GSE202011_RAW.tar` archive, verify archive integrity
   and SHA-256, and parse only its H5 matrices and series-defined sample names.
2. Retain PSO baseline LS/NL samples; for a replicated lesion, select the
   lexicographically first `R1` H5 file before expression is read.  This rule
   is deterministic and is not outcome-dependent.
3. Construct per-sample pseudobulk expression from QC-passing in-tissue spots.
   Evaluate a 2,000-HVG expression reference and PCA in leave-one-patient-out
   folds with all feature selection/scaling/PCA fit in the training patients.
4. If H5 coordinates are complete, evaluate the same fair-dimensional spatial
   graph comparison as GSE206391; otherwise report the coordinate limitation
   and retain the strict expression/PCA boundary test only.
5. Report every result with patient-pair bootstrap uncertainty.  With at most
   five pairs, this is explicitly a boundary replication, not a powered
   superiority test.

The series record reports 23 matched lesional/non-lesional psoriatic samples,
including technical/section replicates; the frozen test uses only the eligible
PSO patient pairs under the rule above.  The public-series metadata and
supplement are the authority for all final sample counts.

# PRI-Audit reproducibility package

This package reproduces the public-data psoriasis PRI-Audit submission set
without overwriting historical development outputs.  It includes the v21
real-data audit and the v22 submission-closure analyses: a known-truth
hierarchical simulation and a fold-local high-resolution spot-topic
comparison.

## Scope and claim boundary

The primary claim is methodological: disease-associated high-resolution
structure is not by itself evidence of reproducible incremental
*patient-level* discrimination beyond a matched expression reference.  The
package therefore reports strict patient/donor-aware evaluation, global-fit
diagnostic contrasts and representation-specific nulls.  It is not a search
for a more favourable psoriasis classifier.

The complete original real-data specification is
[`docs/pri_audit_v21_experiment_spec.md`](../docs/pri_audit_v21_experiment_spec.md).
The v22 additions are fixed in
[`docs/v22_submission_closure_protocol.md`](../docs/v22_submission_closure_protocol.md).
The frozen interpretation and prohibited overclaims are in
[`docs/v21_result_interpretation_and_claim_update.md`](../docs/v21_result_interpretation_and_claim_update.md).

## Public inputs

The runner verifies and records SHA-256 digests for local public input files.

* **GSE151177** raw 10x matrices — sample-level psoriasis/control reconstruction.
* **GSE173706** raw cell-by-gene matrices — paired PP/PN donor reconstruction.
* **GSE206391** processed public H5 — paired LS/NL spatial sample reconstruction.
* **GSE47751** freshly retrieved official GEO series matrix — independent paired
  bulk boundary replication.
* **GSE202011** official Visium H5 archive — independent paired PSO
  high-resolution boundary replication. The archive lacks coordinate files, so
  it is intentionally restricted to strict expression/PCA comparison.
* **Seven additional bulk GEO cohorts plus GSE24767 perturbation data** — their
  raw input hashes, frozen scientific roles, executable scripts and expected
  artifacts are listed in `v21_all11_provenance_manifest.json`. They are not
  treated as interchangeable patient-level classifier folds.

The original local GSE47751 archive had trailing gzip data and is never used
by the audit.  The runner uses only `GSE47751_series_matrix_fresh.txt.gz`,
which passes `gzip -t`; its hash is recorded in the manifest.

## Environment and one-command rerun

Run from the project root with a clean virtual environment.  The exact package
record is `reproducibility/requirements-v21.lock`; it governs both v21 and v22.

```bash
python -m venv .venv
.venv/bin/python -m pip install -r reproducibility/requirements-v21.lock
.venv/bin/python scripts/run_pri_audit_complete_reproduction.py
```

The umbrella runner invokes the 13-command v21 real-data audit followed by the
8-command v22 closure suite. It writes an overall release log under
`results/pri_audit_release/`.  The lower-level runners can also be invoked
separately when a reviewer needs to inspect one analysis generation:

```bash
.venv/bin/python scripts/run_v21_submission_reproduction.py
.venv/bin/python scripts/run_v22_submission_reproduction.py \
  --replicates 400 --bootstrap 250 --population-patients 240 --workers 4
```

The v21 primary analyses use 500 label/coordinate permutations and up to 5,000
patient bootstrap draws.  v22 uses 400 repeated known-truth simulations per
scenario/patient count and 5,000 patient-bootstrap draws for the primary
spot-topic comparison.  These are intentionally not replaced by a quick mode
in the submission workflow.  Random seeds are fixed in the scripts and all
runtime/platform metadata are embedded in the manifests.

## Outputs

| Artifact | Contents |
|---|---|
| `results/pri_audit_v21/reproducibility_manifest.json` | input/output provenance, SHA-256 hashes and environment |
| `results/pri_audit_v21/v21_submission_reproduction_log.json` | 13-command real-data rerun and output hashes |
| `results/pri_audit_v22/v22_submission_reproduction_log.json` | 8-command closure rerun and output hashes |
| `results/pri_audit_release/complete_reproduction_log.json` | umbrella status and hashes of both command logs |
| `results/pri_audit_v21/v21_loose_vs_strict_summary.csv` | primary strict and global-fit diagnostic contrasts |
| `results/pri_audit_v21/*_strict_loose_predictions.csv` | one held-out probability per sample/donor unit |
| `results/pri_audit_v21/*_strict_loose_nulls.npz` | full label-permutation and bootstrap distributions |
| `results/pri_audit_v21/gse206391_v21_matched_spatial_coordinate_shuffle.*` | complete within-sample coordinate null |
| `results/pri_audit_v22/v22_simulation_*` | fixed-scenario known-truth simulations and specification |
| `results/pri_audit_v22/gse206391_v22_spot_topic_*` | fold-local topic results, predictions and distributions |
| `results/paper_v21/table[1-3]_*.csv` | manuscript-ready primary tables |
| `results/paper_v21/tableS[1-2]_v22_*.csv` | simulation and topic-subsampling supplements |
| `results/paper_v21/*source_manifest.*` | quantitative source links and SHA-256 bindings |

## Interpretation safeguards

1. The `loose_*` arms are diagnostic contrasts only; global representation
   fitting touches held-out sample covariates and is not a primary estimate.
2. The 16-HVG expression comparator is intentionally mismatched. Its results
   illustrate why feature-budget-mismatched references can manufacture an
   apparent increment; it cannot support an algorithmic claim.
3. A random cell/spot split has every test source sample represented in
   training. Its high AUROCs are a deliberate invalid-estimand diagnostic, not
   an alternative validation score.
4. A coordinate-shuffle null addresses the *incremental patient-level value*
   of a coordinate-based representation; a non-significant result does not
   prove that spatial biology is absent.
5. The known-truth simulation establishes properties of the implemented
   evidence gate under its fixed hierarchy; it does not establish a biological
   truth about psoriasis.
6. The spot-topic comparison is a fixed, low-capacity conventional comparator,
   not an exhaustive benchmark of spatial neural networks.

## Manuscript assembly and release

The canonical author-facing draft is
`manuscript/PRI_Audit_psoriasis_RegionSuff_template_submission.docx`; it is
generated from `manuscript/PRI_Audit_psoriasis_manuscript_v8_submission_ready.md`
using `scripts/build_v21_submission_manuscript_docx.py` with the supplied
template profile.  `PRI_Audit_psoriasis_manuscript_v9_submission_ready.docx`
is retained as a profile-independent assembly record.  The user-owned visual
reference document is deliberately excluded from the public repository: it
contributes no manuscript content, results or author metadata, and its
SHA-256 identity is recorded in the template-submission source manifest.

A public release should include source Markdown, the generated DOCX/PDF,
scripts, locked environment, data-derived PNG/PDF figures, tables and both
source manifests.  It must not depend on the private style reference.  The
author-owned release checklist, DOI metadata template and citation template
are in `submission_package/` and the repository root.

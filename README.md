# PRI-Audit

**PRI-Audit** is a falsification-first framework for asking a deliberately
narrow question: does a high-resolution transcriptomic representation provide
*reproducible patient-level discrimination beyond a fair, matched expression
reference*?  The accompanying public-data application is psoriasis, not a
clinical diagnostic product and not a claim that spatial or single-cell biology
is absent.

## What the study establishes

Eleven public GEO cohorts have deliberately different roles.  Three
high-resolution cohorts form the primary patient-level analysis; one
independent Visium cohort is a boundary replication; the remaining cohorts
provide biological context, perturbation evidence or provenance checks.  They
must not be described as 11 interchangeable prediction tests.

PRI-Audit requires all of the following before a candidate representation can
be credited with an incremental patient-level benefit:

1. biological-unit isolation (patients/donors never cross outer folds);
2. fold-local fitting of every adaptive representation step;
3. an expression comparator matched for folds and feature budget;
4. independent-unit uncertainty and representation-specific counterfactual
   diagnostics.

In the current public psoriasis cohorts, the study finds reproducible
disease-state signal but **no tested low-rank, spot-topic or spatial
representation with demonstrated incremental patient-level AUROC over its
matched expression reference**.  Known-truth simulations show that the
implemented gate rejects the fixed no-increment hierarchy while retaining
sensitivity to a zero-mean coordinate-only signal.

## Reproduce the frozen submission package

The public GEO input files are retrieved locally; this repository does not
redistribute controlled or source archives.  From the project root, create a
clean environment using the exact frozen package record and run the complete
non-destructive workflow:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r reproducibility/requirements-v21.lock
.venv/bin/python scripts/fetch_pri_v24_gse151177_inputs.py
.venv/bin/python scripts/run_pri_audit_complete_reproduction.py
```

The full run executes the v21 real-data audit (13 commands), the v22
submission-closure analyses (8 commands), and the corrected v24 GSE151177
rerun. The v24 fetcher checks public GEO files against the input manifest
before analysis. The umbrella log at
`results/pri_audit_release/complete_reproduction_log.json` records each run;
the v21 and v22 logs and the v24 manifest retain execution or input/output
checksums. A submission-grade full run takes appreciably longer than a smoke
test because it retains the planned resampling and null distributions.

For the exact scientific protocol, input roles, checksums and interpretation
boundaries, start with [the reproducibility guide](reproducibility/README.md)
and [`docs/v22_submission_closure_protocol.md`](docs/v22_submission_closure_protocol.md).

## Repository layout

| Directory | Contents |
|---|---|
| `scripts/` | executable reconstruction, modelling, null and manuscript-table scripts |
| `docs/` | frozen specifications, claim boundaries and cohort-role ledger |
| `reproducibility/` | environment lock and end-to-end rerun instructions |
| `results/` | generated numerical outputs, hashes and machine-readable manifests |
| `figures/` | data-derived manuscript figures in PNG and PDF form |
| `manuscript/` | manuscript source and submission-ready Word draft |
| `submission_package/` | journal checklist, cover letter and release templates |

## Data, code and citation

All biological inputs are public GEO records identified in the manuscript and
provenance manifests. The current source snapshot is available at
https://github.com/kobayashi0512/PRI-Audit. The manuscript identifies the
authors and affiliation. The v1.32 release is licensed under MIT and contains
machine-readable citation metadata. The tagged release is the manuscript-linked
code snapshot; a DOI may be added through a future archival service without
changing this tag. Nothing in this repository should be construed as medical
advice or used for individual patient care.


## Revision package v1.32

The v1.32 release adds the independent GSE202011 barcode-to-coordinate graph analysis, grouped full-pipeline resampling, and fixed-candidate reference-sensitivity analysis. Reproduction instructions and result boundaries are in [`docs/pri132_revision_protocol.md`](docs/pri132_revision_protocol.md).

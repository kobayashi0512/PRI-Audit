# Changelog

## v1.0.0 — planned public submission release

This first immutable release is prepared for the PRI-Audit manuscript,
*“PRI-Audit: a falsification-first framework for testing patient-level gain
from high-resolution transcriptomic representations.”*

### Included scientific scope

- v21 strict real-data audit of three primary high-resolution psoriasis
  datasets, with an independent expression/PCA boundary replication.
- Registry and provenance of 11 public GEO cohorts with explicitly
  heterogeneous scientific roles.
- v22 known-truth hierarchical simulation, fold-local spot-topic comparator,
  and spot-subsampling sensitivity analysis.
- Full bootstrap and representation-specific null distributions, prediction
  rows, table/figure source manifests, data-derived figure assets and editable
  Figure 1–3 PowerPoint source.
- A unified reproducibility entry point that writes versioned execution logs
  and SHA-256 output bindings.

### Deliberate exclusions

- GEO source-download cache and other raw archives; these are retrieved from
  their public accessions and verified against the included provenance hashes.
- Private document style references, temporary rendering folders, virtual
  environments, superseded manuscript drafts and unrelated development outputs.

### Before tagging

Replace author, repository, licence and DOI placeholders according to
`submission_package/PUBLIC_RELEASE_CHECKLIST.md`. The `v1.0.0` tag must be
created only after the complete reproducibility log is successful and all
copyright holders approve the chosen licence.

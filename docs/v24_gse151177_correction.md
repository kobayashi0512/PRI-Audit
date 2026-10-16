# GSE151177 v24 correction

The v24 rerun corrects mitochondrial-gene filtering for GSE151177. It uses gene symbols in the GEO `features.tsv.gz` files, where mitochondrial genes begin with `MT-`, rather than relying on Ensembl identifiers. The remaining sample-level and per-cell quality-control thresholds and the v21 learner recipe are unchanged.

The analysis begins with the 18 emigrating-cell profiles listed in the GEO series matrix. Five enzyme-digestion profiles are excluded by protocol. The registered sample-level screen retains 17 profiles: 13 psoriasis and four control source samples. GEO records do not establish verified patient identifiers, so the held-out unit remains the source sample.

## Reproduce

From the repository root, use the pinned v21 environment and run:

```bash
python scripts/fetch_pri_v24_gse151177_inputs.py
python scripts/run_pri_audit_v24_gse151177_corrected.py
```

The fetcher downloads only the public GEO files recorded in the v24 input manifest. It verifies every file against its expected byte count and SHA-256 digest before the analysis starts. The rerun writes predictions, roster and quality-control audits, summary estimates, and an updated manifest under `results/pri_audit_v24_gse151177/`.

The primary subset contains 17 source samples, 13 psoriasis and four controls. The separately labelled Control05 analysis is a quality-control sensitivity check; it is not the primary estimate. The correction does not establish patient-level independence for GSE151177 and does not change the study's other cohort roles or decision rule.

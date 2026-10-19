# PRI-Audit revision package v1.33

This package brings the public repository into correspondence with the revised manuscript. It preserves the paper's single primary question: whether a candidate representation adds patient-level discrimination beyond a fold-matched expression reference.

## Revision analyses

1. **GSE206391 full-pipeline grouped resampling.** `run_nested_group_refit_sensitivity.py` refits HVG selection, scaling, PCA or NMF, and the classifier inside each resampled training set. It reports a sensitivity interval alongside the original conditional saved-prediction interval.
2. **GSE206391 fixed-candidate reference sensitivity.** `run_fixed_candidate_reference_sensitivity.py` fixes each candidate at 2,000 HVGs and varies only the expression-reference budget. It is a revision-stage diagnostic; the 2,000-HVG matched comparison remains the primary estimate.
3. **GSE202011 independent spatial-graph boundary analysis.** `run_gse202011_independent_spatial_graph.py` downloads the ten predeclared GEO H5 matrices, filters spots, and joins their retained barcodes to the versioned Visium V1 barcode-to-array-coordinate map. The join must be one-to-one for every retained barcode. It does not use histology images or image registration.

## Commands

From the repository root, first create the frozen environment and reproduce the original package as described in `reproducibility/README.md`. Then run:

```bash
python scripts/pri132_revision/download_gse202011_inputs.py
python scripts/pri132_revision/run_nested_group_refit_sensitivity.py --draws 400
python scripts/pri132_revision/run_fixed_candidate_reference_sensitivity.py --bootstrap-draws 5000 --full-refit-draws 400
python scripts/pri132_revision/run_gse202011_independent_spatial_graph.py --coordinate-shuffles 500 --bootstrap 5000
```

The GSE202011 GEO archive is not redistributed. Its verified SHA-256 and the expected 10-matrix extraction rule are embedded in the downloader. `visium_v1_barcode_row_col.csv` is the versioned 4,992-barcode coordinate map used for the exact barcode join.

## Claim boundary

The original matched expression comparison is the primary analysis. The full-refit and fixed-candidate analyses quantify sensitivity to resampling and comparator specification. They neither replace the matched primary comparison nor establish a general claim about all spatial or high-resolution models.

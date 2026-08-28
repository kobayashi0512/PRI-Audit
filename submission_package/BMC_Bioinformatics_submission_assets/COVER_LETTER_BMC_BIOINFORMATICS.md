[Date]

Editor-in-Chief / Handling Editor  
*BMC Bioinformatics*

Dear Editor,

Please consider our Research Article, **“PRI-Audit: a falsification-first
framework for testing patient-level gain from high-resolution transcriptomic
representations,”** for publication in *BMC Bioinformatics*.

High-resolution single-cell and spatial transcriptomic studies commonly report
biologically interpretable structure, but this is not the same as a
reproducible patient-level predictive increment beyond a fair expression
reference. PRI-Audit makes that distinction testable. It combines
biological-unit outer-fold isolation, fold-local representation fitting,
feature-budget-matched expression comparators, independent-unit uncertainty
and representation-specific counterfactual nulls in one explicit evidence
gate.

Using public psoriasis transcriptomic cohorts, we show that disease-state
signal is strong while the tested low-rank, spot-topic and spatial encoders do
not demonstrate a reproducible patient-level AUROC increment over matched
expression. Crucially, a fixed known-truth hierarchy simulation shows that the
implemented gate rejects no-increment, source-overlap and weak-reference
artifacts while retaining sensitivity to a coordinate-only signal. The
submission therefore contributes a reusable computational evaluation framework
rather than a claim of a new best classifier or an assertion that spatial
biology is absent.

The work fits *BMC Bioinformatics* because it presents a statistically
specified, reproducible computational framework for transcriptome analysis,
with executable code, locked dependencies, input hashes, patient-level
predictions, complete null distributions and a one-command reproduction
workflow. All biological data are de-identified public GEO records; no new
participant recruitment, intervention or identifiable data access occurred.

[INSERT: confirmation that all authors approved the manuscript, that it is not
under consideration elsewhere, and the final competing-interest statement.]

The public code repository is https://github.com/kobayashi0512/PRI-Audit,
with the frozen manuscript version at
https://github.com/kobayashi0512/PRI-Audit/tree/v1.0.0. The authors will add
the archival DOI for this unchanged tagged release before formal submission.

Thank you for your consideration.

Sincerely,

[Corresponding author name, degree]  
[Affiliation]  
[Postal address]  
[Email]

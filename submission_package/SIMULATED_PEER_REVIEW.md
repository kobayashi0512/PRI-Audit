# Adversarial simulated peer review for PRI-Audit

**Use:** complete this record after the external GSE202011 result and final
manuscript tables are frozen.  The questions below are deliberately written as
the strongest foreseeable reviewer objections; each requires a concrete
manuscript location and artifact, not a rhetorical reply.

| Reviewer attack | Evidence needed in the final paper | Required response / amendment | Status |
|---|---|---|---|
| “This is merely a negative benchmark, not a method.” | Methods §§2.1–2.5; Figures 2–3; `docs/pri_audit_v21_experiment_spec.md` | PRI-Audit is framed as a falsification-first decision rule, with an explicit evidence gate and documented information hand-offs | complete in v8 manuscript |
| “Nested CV and fair baselines are standard; what is original?” | Figure 4B (+0.082 unfair apparent gain), Figure 5A (100% source overlap), Figure 5B (coordinate null), Table 3 | State novelty narrowly: an integrated patient-boundary, matched-reference and source-specific-counterfactual audit—not the invention of cross-validation | complete in v8 manuscript |
| “Eleven datasets are claimed, but only three are high-resolution tests.” | `results/paper_v21/table1_cohort_roles.csv` | Keep roles and independent units explicit; identify GSE202011 as a separate boundary replication | complete |
| “Small patient counts make null results uninformative.” | Table 2; full 500-null/5,000-bootstrap NPZ files; §3.5 GSE202011 saturation notice | Claim lack of *demonstrated* gain, never proof of equivalence or absent biology | complete in v8 manuscript |
| “Random cell/spot split is a strawman.” | Figure 12A; `v21_measurement_random_split_diagnostic.csv` | Keep it didactic and invalid-estimand labelled; never use it as a primary baseline | complete |
| “The spatial null proves only that this one graph is weak.” | `gse206391_v21_matched_spatial_graph.json`; Figure 12B; result interpretation | Bound conclusion to this representation and current public cohort | complete |
| “Global preprocessing did not always help, so the leakage claim is overgeneralised.” | Figure 11C; `v21_nested_neighborhood_strictness.csv` | Retire the universal optimism statement; report resolution sensitivity | complete |
| “External validation was chosen after exploration.” | `docs/v21_external_candidate_selection.md`; `gse202011_v21_external_strict.json`; archive SHA-256 | Include ledger and deterministic ST14-R1 rule in supplement; label future cohorts exploratory | complete |
| “Can a reviewer actually rerun this?” | `run_v21_submission_reproduction.py`; `requirements-v21.lock`; v21 and v8 source manifests; successful 13-step execution log | Deposit immutable code snapshot and DOI before submission | reproducibility complete; DOI/archiving pending author action |

## Decision rule for editorial readiness

The paper is not ready to submit until every `pending` cell above is replaced
by a file path, figure/table number or precise revised sentence.  A reviewer
does not need to agree that the primary result is positive; they need to be
able to verify that the negative incremental conclusion is correctly bounded
and reproducible.

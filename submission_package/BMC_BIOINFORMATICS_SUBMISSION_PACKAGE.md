# BMC Bioinformatics Research Article submission package

## Target and rationale

**Target:** *BMC Bioinformatics*  
**Article type:** Research Article  
**Working title:** *PRI-Audit: a falsification-first framework for testing
patient-level gain from high-resolution transcriptomic representations*

The target is well matched to the manuscript because the journal explicitly
considers novel computational algorithms, software, statistical methods and
machine learning for biological-data modelling, including transcriptome
analysis. The submission should foreground the reusable evaluation contract,
known-truth validation and complete executable evidence, with psoriasis as the
data-rich use case—not as a clinical decision-support claim.

## Files ready for the submission system

| Item | Local source | Status |
|---|---|---|
| Main manuscript | `manuscript/PRI_Audit_psoriasis_manuscript_v9_submission_ready.docx` | ready after author/DOI fields are replaced |
| Editable manuscript source | `manuscript/PRI_Audit_psoriasis_manuscript_v8_submission_ready.md` | ready |
| Main figures | `figures/paper_v19_fig1_dataset_atlas.png`, `figures/paper_v20_fig2_overall_framework.png`, `figures/paper_v20_fig3_algorithm_architecture.png`, `figures/paper_v21/*.png`, `figures/paper_v22/*.png` | ready |
| Figure source deck | `ppt_figures_v1/PRI_Audit_data_integrated_figures_v2_audited.pptx` | ready |
| Main and supplementary tables | `results/paper_v21/table*.csv` | ready |
| Cover letter | `submission_package/COVER_LETTER_BMC_BIOINFORMATICS.md` | replace bracketed fields |
| Reproducibility material | `README.md`, `reproducibility/`, `results/pri_audit_v21/`, `results/pri_audit_v22/` | ready for public release |

## Required final substitutions

1. Complete `AUTHOR_AND_RELEASE_METADATA_FORM.md`.
2. Insert final author list, affiliations, correspondence details, CRediT,
   funding, acknowledgement, ethics wording and competing-interest statement.
3. Obtain the tagged public repository URL and version-specific archive DOI;
   replace their manuscript placeholders.
4. Regenerate the DOCX and source manifest after those substitutions. Do not
   revise numerical claims unless a rerun changes the versioned outputs.
5. On the day of submission, recheck the journal's live pre-submission
   checklist, submission guidelines and fee/funding policy, as these can
   change independently of the manuscript.

## Submission fields: ready wording

**Methods-oriented summary:** PRI-Audit is a falsification-first
patient-level evaluation framework that only credits a high-resolution
transcriptomic representation when it improves beyond a fold- and
feature-budget-matched expression reference under biological-unit isolation,
fold-local fitting, independent-unit uncertainty and a representation-specific
counterfactual null.

**Data availability:** All analysed biological inputs are de-identified public
Gene Expression Omnibus records. The 11-cohort provenance registry and the
separately selected GSE202011 boundary-replication input are documented with
retrieval paths and SHA-256 checksums in the public repository. No controlled
participant-level data are redistributed.

**Code availability:** Source code, fixed protocol, locked environment,
one-command reproduction workflow, generated predictions, complete
bootstrap/null distributions and figure/table source manifests will be openly
available at `[REPOSITORY_URL]`, archived at `[VERSION_SPECIFIC_ARCHIVE_DOI]`.

**Ethics:** This study reanalysed de-identified public datasets and involved no
new participant recruitment, intervention or access to identifiable
information. `[CONFIRM/ADAPT TO INSTITUTIONAL WORDING]`

## Consistency guardrails

- Do not claim a high-resolution classifier exceeded its matched expression
  reference.
- Do not call GSE202011 an external spatial validation: its selected archive
  lacks coordinate files.
- Do not call the 11 cohort-registry entries 11 interchangeable prediction
  tests. The three primary high-resolution cohorts share the core estimand;
  other roles are declared explicitly.
- Describe simulation as validation of the **implemented decision gate** under
  a fixed hierarchy, not evidence about psoriasis biology.
- Do not include a generated tool as an author; disclose any writing/tool use
  only according to the journal's live policy and with author responsibility
  for the final text.

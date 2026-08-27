# Licence decision required before public release

This project cannot receive a real open-source licence without the copyright
holders' approval. No licence text has been fabricated in this staging package.

## Decision to record

All copyright holders should agree on one OSI-approved licence before the
first public tag. For a code-and-reproducibility paper, a permissive licence
such as **MIT** or **BSD-3-Clause** is usually the lowest-friction choice for
reuse; this is a recommendation, not a choice made on the authors' behalf.

Once confirmed:

1. Add the exact approved `LICENSE` file at the repository root.
2. Replace `[OSI_APPROVED_LICENSE_IDENTIFIER]` in `CITATION.cff.template` and
   `[SELECTED_OSI_LICENSE_IDENTIFIER]` in `submission_package/zenodo.json.template`.
3. Record the same licence in the GitHub release and archive metadata.
4. Tag the repository only after the reproducibility run and the licence are
   both final.

Do not use a non-commercial or no-derivatives restriction if the stated goal
is an OSI-approved open-source software release; those restrictions are not
OSI-approved software licences.

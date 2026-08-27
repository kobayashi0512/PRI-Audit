# PRI-Audit public release and DOI checklist

This checklist turns the local, audited package into a public, immutable
software/data-analysis release. It separates actions that can be prepared
locally from actions that require an author-owned GitHub/Zenodo account.

## Already complete locally

- [x] Frozen v21 real-data audit and v22 submission-closure protocols.
- [x] A 13-command v21 runner and an 8-command v22 runner.
- [x] A unified 21-command runner:
  `scripts/run_pri_audit_complete_reproduction.py`.
- [x] Deterministic seeds, locked dependencies, raw-input provenance and
  machine-readable numerical tables.
- [x] Complete primary null/bootstrapping distributions and source manifests.
- [x] Submission-ready Word draft, data-derived figures, and editable Figure
  1–3 PowerPoint source.
- [x] Citation and archive-metadata templates with unambiguous placeholders.

## Prepare the release staging bundle

1. Run the complete workflow in the intended release environment:

   ```bash
   .venv/bin/python scripts/run_pri_audit_complete_reproduction.py
   ```

2. Confirm that `results/pri_audit_release/complete_reproduction_log.json`
   contains two successful commands and that the nested v21/v22 logs contain
   their output hashes.
3. A verified release staging folder has been prepared at
   `/Users/tongyue/Documents/ppf/papers/PRI-Audit-v1.0.0-public/` (256 files,
   29.8 MB, zero manifest hash mismatches). To regenerate it from source after
   any author-metadata change, create a clean, reviewable folder outside the
   working repository:

   ```bash
   .venv/bin/python scripts/build_pri_audit_public_release.py \
     --destination /ABSOLUTE/PATH/PRI-Audit-v1.0.0
   ```

   The builder includes source, selected generated outputs, complete
   null-distribution files, manuscript/figure artifacts and a SHA-256 release
   manifest. It deliberately excludes the public GEO download cache, private
   style references, virtual environments and superseded drafts.
4. In the staging folder, independently run the unified runner after restoring
   the documented public GEO input layout. Keep the resulting log as release
   evidence.

## Author-owned public actions

1. Complete `AUTHOR_AND_RELEASE_METADATA_FORM.md` and obtain approval from all
   authors.
2. Add the authors' chosen real `LICENSE` text and convert
   `CITATION.cff.template` into populated `CITATION.cff`.
3. Create an empty public GitHub repository named `PRI-Audit`. Commit the
   staging folder—not the entire historical workspace—and push the approved
   files.
4. Create the immutable Git tag `v1.0.0` and a matching GitHub release.
5. Archive that tagged release with Zenodo (or the approved institutional
   service), using `zenodo.json.template` as the metadata starting point.
6. Copy the **version-specific archive DOI** (not only a concept DOI) into
   `CITATION.cff`, the repository README, the manuscript Code availability
   section and the BMC submission fields.
7. Re-run the source-manifest checker and review GitHub's displayed README,
   licence, release artifact list and DOI link.

## Do not release

- `data/` GEO source downloads or a redistributed raw archive; cite accessions
  and retain their hashes instead.
- `.venv/`, `.codex_python/`, temporary render directories, private Word style
  references, or prior manuscript drafts.
- Invented author, funder, ethics, conflict or DOI information.

## Publication hand-off

After the DOI exists, replace the four declaration placeholders in the Word
manuscript and regenerate its source manifest. Then attach the manuscript,
cover letter, supplement files and public repository/DOI to the journal
submission system.

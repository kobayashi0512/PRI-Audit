# PRI-Audit v22 submission-closure protocol

## Scope and lock

This protocol adds two analyses required to assess the *evaluation framework*
in the v21 manuscript.  It is not a search for a more favourable psoriasis
classifier.  It was fixed before the v22 runs; it is not represented as an
externally preregistered protocol.

The v21 real-data results remain frozen.  No feature count, classifier
regularisation, decision threshold, or clinical endpoint is selected after
viewing v22 outputs.

## Experiment S1: known-truth hierarchy simulation

### Question

Can the complete PRI-Audit gate distinguish a representation with no extra
patient-level information from one whose coordinate arrangement carries
information that is zero-mean in pseudobulk expression?

### Common data-generating hierarchy

Each simulation has 12, 24, or 48 independent patients.  Every patient has a
paired control and disease sample; every sample has 40 nested cells/spots.
Cell expression comprises: eight aggregate-expression signal dimensions; four
spatial dimensions; 116 nuisance dimensions; patient background; source/sample
fingerprints; and independent cell noise.  A spatial signal changes the
coordinate cross-moment but has exactly zero aggregate sample mean.

### Fixed scenarios

| Scenario | Aggregate expression effect | Spatial effect | Source fingerprint | Intended check |
|---|---:|---:|---:|---|
| `null_increment` | 0.85 | 0.00 | 0.35 | Empirical false-positive rate when candidate space adds no information |
| `spatial_increment` | 0.55 | 0.72 | 0.35 | Empirical power for a coordinate-only increment beyond pseudobulk |
| `source_overlap` | 0.00 | 0.00 | 2.30 | Contrast source-blocked patient prediction with invalid random-cell interpolation |
| `mismatched_reference` | 0.85 | 0.00 | 0.35 | Show an apparent gain caused solely by excluding signal genes from the reference |

In each replicate, a fixed L2 logistic learner is fitted in leave-one-patient
out folds.  The strict candidate is the concatenation of aggregate expression
and the coordinate cross-moment.  The matched reference is aggregate
expression alone.  A gate pass requires `ΔAUROC >= 0.05` **and** a paired
patient-bootstrap lower 95% bound greater than zero.  The formal run uses 400
replicates per scenario and patient count, 250 bootstrap draws per replicate,
and a fixed seed of 20260824.  Monte-Carlo test cohorts are used only to check
the direction of the mean increment, not to reinterpret the bootstrap interval
as a generalisation-confidence interval.

### Primary outputs

* `results/pri_audit_v22/v22_simulation_replicates.csv`
* `results/pri_audit_v22/v22_simulation_summary.csv`
* `results/pri_audit_v22/v22_simulation_specification.json`
* `figures/paper_v22/fig6_v22_known_truth_simulation.{png,pdf}`

## Experiment S2: fold-local spatial spot-topic comparator

### Question

Does a conventional high-resolution spot-composition representation add
patient-level discrimination to a dimension-matched expression representation
in GSE206391?

### Fixed recipe

For each held-out paired patient, select 500 pseudobulk HVGs using training
patients only.  Fit an eight-topic MiniBatchNMF on up to 300 spots per
training sample; transform held-patient spots; average their topic activations
within sample.  Compare the eight-dimensional topic representation with an
eight-dimensional fold-local expression PCA.  Compare the 16-dimensional
expression-plus-topic representation with a 16-dimensional fold-local
expression PCA.  All classifiers are fixed class-balanced L2 logistic
regressions.  Uncertainty uses 5,000 paired-patient bootstrap draws and the
label null uses 500 within-patient label flips.

The topic model is purposely a low-capacity, auditable comparator suitable for
12 independent patients.  It is **not** presented as an exhaustive benchmark
of all spatial neural networks.  Sensitivity to 150 and 450 sampled spots is
supplementary and cannot replace the 300-spot primary result.

### Primary outputs

* `results/pri_audit_v22/gse206391_v22_spot_topic_benchmark.json`
* `results/pri_audit_v22/gse206391_v22_spot_topic_predictions.csv`
* `results/pri_audit_v22/gse206391_v22_spot_topic_distributions.npz`

## Interpretation guardrails

1. Simulation establishes properties of the implemented decision contract,
   not truth about psoriasis biology.
2. The spot-topic result is a representation-specific patient-level comparison
   and does not claim that spatial structure is absent.
3. A saturated or non-significant cohort is never interpreted as evidence of
   equivalence.
4. The v22 results can narrow or strengthen the methods claim but cannot be
   used to change the frozen real-data conclusion into a positive classifier
   claim.

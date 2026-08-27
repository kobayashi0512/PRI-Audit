"""Controlled simulation study for the PRI-Audit submission revision.

The real cohorts establish what happened in the available psoriasis data but
cannot reveal whether an evaluation pipeline has a calibrated false-positive
rate.  This script supplies that missing *known-truth* experiment.  It does
not use any real outcome to tune a method.

Each simulated patient contributes a paired control/disease sample, each
sample contains repeated cells/spots, and the data-generating process can add
three separable signals:

* a patient-shared expression effect (captured by pseudobulk expression),
* a zero-mean coordinate-associated effect (captured only by a spatial
  cross-moment), and
* a source fingerprint that makes random measurement splits interpolate
  already-seen samples without improving held-out-patient prediction.

The predeclared simulation questions are therefore: (i) whether the strict,
matched audit spuriously declares a gain under no spatial increment; (ii)
whether it has power when such an increment is present; (iii) whether a
measurement split can produce apparent performance from source overlap; and
(iv) whether weakening an otherwise equivalent expression comparator creates
an apparent high-resolution gain.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

# Each replicate is numerically independent.  Limiting native BLAS threads
# avoids oversubscription when the formal Monte-Carlo run uses a few workers.
for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_thread_variable, "1")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "pri_audit_v22"
SEED = 20260824
ALPHA_GATE = 0.05


@dataclass(frozen=True)
class Scenario:
    """Fixed, interpretable parameters of a simulation condition."""

    name: str
    expression_effect: float
    spatial_effect: float
    source_fingerprint: float
    weak_reference: bool = False


SCENARIOS = (
    Scenario("null_increment", expression_effect=0.85, spatial_effect=0.0, source_fingerprint=0.35),
    Scenario("spatial_increment", expression_effect=0.55, spatial_effect=0.72, source_fingerprint=0.35),
    Scenario("source_overlap", expression_effect=0.0, spatial_effect=0.0, source_fingerprint=2.30),
    Scenario("mismatched_reference", expression_effect=0.85, spatial_effect=0.0, source_fingerprint=0.35, weak_reference=True),
)


@dataclass
class SimulatedCohort:
    expression: np.ndarray
    spatial: np.ndarray
    labels: np.ndarray
    patients: np.ndarray
    cell_expression: np.ndarray
    cell_labels: np.ndarray
    cell_sources: np.ndarray


def _classifier() -> object:
    """Fixed classifier used throughout the real-data PRI-Audit workflow."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, solver="liblinear", class_weight="balanced", max_iter=2000, random_state=0),
    )


def simulate_cohort(n_patients: int, scenario: Scenario, rng: np.random.Generator, n_cells: int = 40) -> SimulatedCohort:
    """Generate a paired patient/cell hierarchy with a known spatial truth.

    The first eight genes contain a non-spatial disease expression effect.  A
    different four-gene block has a condition-dependent *coordinate* effect
    whose mean is zero, so pseudobulk cannot recover it.  The final genes are
    nuisance dimensions.  Each sample additionally has a fixed high-dimensional
    source fingerprint, which random cell splitting can learn but patient-LOO
    evaluation cannot reuse.
    """
    # A moderately high-dimensional nuisance block lets source fingerprints be
    # memorised in a random-cell split even when the number of source samples
    # is larger than in the smallest public cohort.
    n_expression_signal, n_spatial_signal, n_nuisance = 8, 4, 116
    p = n_expression_signal + n_spatial_signal + n_nuisance
    n_samples = n_patients * 2
    labels = np.tile(np.array([0, 1], dtype=int), n_patients)
    patients = np.repeat(np.arange(n_patients), 2)
    patient_background = rng.normal(0.0, 0.45, size=(n_patients, p))
    source_background = rng.normal(0.0, scenario.source_fingerprint, size=(n_samples, p))

    expression = np.empty((n_samples, p), dtype=float)
    spatial = np.empty((n_samples, n_spatial_signal), dtype=float)
    cells: list[np.ndarray] = []
    cell_labels: list[np.ndarray] = []
    cell_sources: list[np.ndarray] = []

    # Exactly balanced coordinates make the spatial effect identically
    # zero-mean at the sample aggregate level.
    base_coordinate = np.tile(np.array([-1.0, 1.0]), n_cells // 2)
    for sample in range(n_samples):
        y = labels[sample]
        coordinate = rng.permutation(base_coordinate)
        values = (
            patient_background[patients[sample]][None, :]
            + source_background[sample][None, :]
            + rng.normal(0.0, 0.65, size=(n_cells, p))
        )
        values[:, :n_expression_signal] += scenario.expression_effect * y
        values[:, n_expression_signal:n_expression_signal + n_spatial_signal] += (
            scenario.spatial_effect * y * coordinate[:, None]
        )
        expression[sample] = values.mean(axis=0)
        spatial[sample] = (
            values[:, n_expression_signal:n_expression_signal + n_spatial_signal] * coordinate[:, None]
        ).mean(axis=0)
        cells.append(values)
        cell_labels.append(np.full(n_cells, y, dtype=int))
        cell_sources.append(np.full(n_cells, sample, dtype=int))

    return SimulatedCohort(
        expression=expression,
        spatial=spatial,
        labels=labels,
        patients=patients,
        cell_expression=np.vstack(cells),
        cell_labels=np.concatenate(cell_labels),
        cell_sources=np.concatenate(cell_sources),
    )


def crossfit_probability(features: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Leave-one-patient-out prediction with all fitted objects fold-local."""
    probability = np.full(labels.size, np.nan)
    for held in np.unique(groups):
        test = groups == held
        train = ~test
        model = _classifier()
        model.fit(features[train], labels[train])
        probability[test] = model.predict_proba(features[test])[:, 1]
    if np.isnan(probability).any():
        raise RuntimeError("Cross-fitted predictions were incomplete")
    return probability


def random_measurement_split_probability(cohort: SimulatedCohort, rng: np.random.Generator) -> tuple[float, float]:
    """Return random-cell split AUROC and its source-overlap fraction.

    This deliberately estimates a different estimand and is retained only as
    an invalid-estimand diagnostic, mirroring the real-data manuscript.
    """
    n = cohort.cell_labels.size
    permuted = rng.permutation(n)
    cut = int(round(n * 0.70))
    train_ix, test_ix = permuted[:cut], permuted[cut:]
    model = _classifier()
    model.fit(cohort.cell_expression[train_ix], cohort.cell_labels[train_ix])
    probability = model.predict_proba(cohort.cell_expression[test_ix])[:, 1]
    seen_sources = np.unique(cohort.cell_sources[train_ix])
    overlap = float(np.mean(np.isin(cohort.cell_sources[test_ix], seen_sources)))
    return float(roc_auc_score(cohort.cell_labels[test_ix], probability)), overlap


def bootstrap_delta(
    labels: np.ndarray,
    candidate_probability: np.ndarray,
    reference_probability: np.ndarray,
    groups: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[float, float]:
    """Paired patient bootstrap interval for the held-out AUROC increment."""
    unique = np.unique(groups)
    by_group = {group: np.flatnonzero(groups == group) for group in unique}
    values: list[float] = []
    for _ in range(n_bootstrap):
        selected = rng.choice(unique, size=unique.size, replace=True)
        ix = np.concatenate([by_group[group] for group in selected])
        values.append(
            roc_auc_score(labels[ix], candidate_probability[ix])
            - roc_auc_score(labels[ix], reference_probability[ix])
        )
    return tuple(float(value) for value in np.quantile(np.asarray(values), [0.025, 0.975]))


def population_increment(
    train: SimulatedCohort,
    scenario: Scenario,
    rng: np.random.Generator,
    population_patients: int,
    use_weak_reference: bool,
) -> float:
    """Approximate each training replicate's out-of-sample target increment."""
    test = simulate_cohort(population_patients, scenario, rng)
    candidate_train = np.hstack([train.expression, train.spatial])
    candidate_test = np.hstack([test.expression, test.spatial])
    if use_weak_reference:
        # The weak comparator intentionally excludes the known expression-signal
        # block.  The candidate has no extra information in this scenario.
        reference_train = train.expression[:, 12:]
        reference_test = test.expression[:, 12:]
    else:
        reference_train = train.expression
        reference_test = test.expression
    candidate = _classifier().fit(candidate_train, train.labels)
    reference = _classifier().fit(reference_train, train.labels)
    candidate_auc = roc_auc_score(test.labels, candidate.predict_proba(candidate_test)[:, 1])
    reference_auc = roc_auc_score(test.labels, reference.predict_proba(reference_test)[:, 1])
    return float(candidate_auc - reference_auc)


def run_replicate(
    n_patients: int,
    scenario: Scenario,
    seed: int,
    n_bootstrap: int,
    population_patients: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    cohort = simulate_cohort(n_patients, scenario, rng)
    expression_probability = crossfit_probability(cohort.expression, cohort.labels, cohort.patients)
    candidate_probability = crossfit_probability(
        np.hstack([cohort.expression, cohort.spatial]), cohort.labels, cohort.patients
    )
    strict_delta = float(
        roc_auc_score(cohort.labels, candidate_probability) - roc_auc_score(cohort.labels, expression_probability)
    )
    strict_ci_low, strict_ci_high = bootstrap_delta(
        cohort.labels, candidate_probability, expression_probability, cohort.patients, rng, n_bootstrap
    )
    population_delta = population_increment(
        cohort, scenario, rng, population_patients, use_weak_reference=False
    )

    random_auc, source_overlap = random_measurement_split_probability(cohort, rng)
    sample_auc = float(roc_auc_score(cohort.labels, expression_probability))

    weak_delta = np.nan
    weak_ci_low = np.nan
    weak_ci_high = np.nan
    weak_population_delta = np.nan
    if scenario.weak_reference:
        weak_probability = crossfit_probability(cohort.expression[:, 12:], cohort.labels, cohort.patients)
        # The candidate here deliberately uses only the same pseudobulk
        # information as the matched expression reference; the contrast is
        # unfair solely because the reference is weakened.
        matched_probability = expression_probability
        weak_delta = float(
            roc_auc_score(cohort.labels, matched_probability) - roc_auc_score(cohort.labels, weak_probability)
        )
        weak_ci_low, weak_ci_high = bootstrap_delta(
            cohort.labels, matched_probability, weak_probability, cohort.patients, rng, n_bootstrap
        )
        weak_population_delta = population_increment(
            cohort, scenario, rng, population_patients, use_weak_reference=True
        )

    return {
        "scenario": scenario.name,
        "n_patients": n_patients,
        "seed": seed,
        "strict_expression_auroc": float(roc_auc_score(cohort.labels, expression_probability)),
        "strict_candidate_auroc": float(roc_auc_score(cohort.labels, candidate_probability)),
        "strict_increment": strict_delta,
        "strict_increment_ci95_low": strict_ci_low,
        "strict_increment_ci95_high": strict_ci_high,
        "strict_gate_pass": bool(strict_delta >= ALPHA_GATE and strict_ci_low > 0.0),
        "strict_population_increment": population_delta,
        "random_measurement_split_auroc": random_auc,
        "random_measurement_test_source_overlap": source_overlap,
        "patient_loo_expression_auroc": sample_auc,
        "weak_reference_increment": weak_delta,
        "weak_reference_increment_ci95_low": weak_ci_low,
        "weak_reference_increment_ci95_high": weak_ci_high,
        "weak_reference_population_increment": weak_population_delta,
        "weak_reference_gate_pass": bool(
            not np.isnan(weak_delta) and weak_delta >= ALPHA_GATE and weak_ci_low > 0.0
        ),
    }


def run_job(job: tuple[Scenario, int, int, int, int, int]) -> dict[str, object]:
    """Pickle-friendly worker wrapper retaining deterministic job ordering."""
    scenario, n_patients, seed, n_bootstrap, population_patients, replicate = job
    record = run_replicate(n_patients, scenario, seed, n_bootstrap, population_patients)
    record["replicate"] = replicate
    return record


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (scenario, n_patients), frame in raw.groupby(["scenario", "n_patients"], sort=False):
        row: dict[str, object] = {
            "scenario": scenario,
            "n_patients": int(n_patients),
            "replicates": int(len(frame)),
            "strict_mean_increment": float(frame.strict_increment.mean()),
            "strict_mean_population_increment": float(frame.strict_population_increment.mean()),
            "strict_gate_rate": float(frame.strict_gate_pass.mean()),
            "median_patient_loo_expression_auroc": float(frame.patient_loo_expression_auroc.median()),
            "median_random_measurement_split_auroc": float(frame.random_measurement_split_auroc.median()),
            "mean_random_measurement_source_overlap": float(frame.random_measurement_test_source_overlap.mean()),
        }
        weak = frame.weak_reference_increment.dropna()
        if not weak.empty:
            row.update({
                "weak_reference_mean_increment": float(weak.mean()),
                "weak_reference_gate_rate": float(frame.weak_reference_gate_pass.mean()),
                "weak_reference_mean_population_increment": float(frame.weak_reference_population_increment.dropna().mean()),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the predeclared PRI-Audit known-truth simulations.")
    parser.add_argument("--replicates", type=int, default=400)
    parser.add_argument("--bootstrap", type=int, default=250)
    parser.add_argument("--population-patients", type=int, default=240)
    parser.add_argument("--patient-counts", type=int, nargs="+", default=[12, 24, 48])
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.replicates < 20 or args.bootstrap < 100:
        raise ValueError("Use at least 20 Monte-Carlo replicates and 100 bootstrap draws.")

    OUT.mkdir(parents=True, exist_ok=True)
    seed_sequence = np.random.SeedSequence(args.seed)
    jobs = [
        (scenario, n_patients, replicate)
        for scenario in SCENARIOS
        for n_patients in args.patient_counts
        for replicate in range(args.replicates)
    ]
    child_sequences = seed_sequence.spawn(len(jobs))
    worker_jobs = [
        (scenario, n_patients, int(child.generate_state(1, dtype=np.uint64)[0]), args.bootstrap, args.population_patients, replicate)
        for (scenario, n_patients, replicate), child in zip(jobs, child_sequences)
    ]
    records: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for job_number, record in enumerate(executor.map(run_job, worker_jobs, chunksize=1), start=1):
            records.append(record)
            if job_number % max(1, len(jobs) // 20) == 0 or job_number == len(jobs):
                print(f"completed {job_number}/{len(jobs)} simulation replicates", flush=True)

    raw = pd.DataFrame(records)
    summary = summarize(raw)
    raw.to_csv(OUT / "v22_simulation_replicates.csv", index=False)
    summary.to_csv(OUT / "v22_simulation_summary.csv", index=False)
    spec = {
        "analysis_version": "pri_audit_v22_simulation",
        "seed": args.seed,
        "replicates_per_condition": args.replicates,
        "patient_counts": args.patient_counts,
        "paired_samples_per_patient": 2,
        "repeated_measurements_per_sample": 40,
        "bootstrap_draws": args.bootstrap,
        "population_patients_for_monte_carlo_target": args.population_patients,
        "strict_gate": "increment >= 0.05 and patient-bootstrap lower 95% bound > 0",
        "scenarios": [scenario.__dict__ for scenario in SCENARIOS],
        "interpretation": {
            "null_increment": "strict gate rate is the empirical false-positive rate for a non-incremental spatial representation",
            "spatial_increment": "strict gate rate is empirical power for a zero-mean coordinate-specific signal beyond pseudobulk expression",
            "source_overlap": "random-cell AUROC is an invalid-estimand diagnostic because test cells have sources represented in training",
            "mismatched_reference": "candidate and matched expression contain the same information; any weak-reference gain is manufactured by the comparator",
        },
    }
    (OUT / "v22_simulation_specification.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

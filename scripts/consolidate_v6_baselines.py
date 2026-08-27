"""Consolidate checkpointed v6 baseline evaluations without refitting any model."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "motifr_v6"


def main() -> None:
    files = sorted(OUT.glob("v6_baseline_lodo_checkpoint_*.csv"))
    frames = [pd.read_csv(path) for path in files if path.stat().st_size]
    if not frames:
        raise FileNotFoundError("No baseline checkpoints found")
    combined = pd.concat(frames, ignore_index=True)
    required = {"held_context", "method", "n_pairs", "order_accuracy", "inner_mean_accuracy", "params"}
    if not required.issubset(combined.columns):
        raise ValueError(f"Malformed baseline checkpoints: {set(combined.columns)}")
    # Repeated rows arise because each checkpoint contains all earlier completed
    # folds from the same invocation. Equality across repeats is an audit check,
    # not an opportunity to select a favorable result.
    keys = ["held_context", "method"]
    conflicts = combined.groupby(keys).order_accuracy.nunique()
    if (conflicts > 1).any():
        raise ValueError(f"Conflicting checkpoints: {conflicts[conflicts > 1].to_dict()}")
    final = combined.drop_duplicates(keys, keep="last").sort_values(keys).reset_index(drop=True)
    if len(final) != 28:
        raise ValueError(f"Expected 7 contexts x 4 methods = 28 evaluations, got {len(final)}")
    final.to_csv(OUT / "v6_baseline_lodo_results.csv", index=False)
    pivot = final.pivot(index="held_context", columns="method", values="order_accuracy")
    pivot.to_csv(OUT / "v6_baseline_lodo_accuracy_table.csv")
    summary = final.groupby("method", as_index=False).agg(
        mean_order_accuracy=("order_accuracy", "mean"),
        minimum_order_accuracy=("order_accuracy", "min"),
    )
    summary.to_csv(OUT / "v6_baseline_lodo_summary.csv", index=False)
    print(pivot.round(3).to_string())
    print("\n" + summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

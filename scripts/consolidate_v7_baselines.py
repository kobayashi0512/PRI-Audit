"""Audit and consolidate checkpointed v7 baseline results without refitting."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "prismr_v7"


def main() -> None:
    frames = [pd.read_csv(p) for p in OUT.glob("v7_baseline_checkpoint_*.csv")]
    data = pd.concat(frames, ignore_index=True)
    keys = ["held_context", "method"]
    conflicts = data.groupby(keys).concordance.nunique()
    if (conflicts > 1).any():
        raise ValueError(f"conflicting checkpoint values: {conflicts[conflicts > 1].to_dict()}")
    final = data.drop_duplicates(keys).sort_values(keys).reset_index(drop=True)
    if len(final) != 40:
        raise ValueError(f"expected 8 contexts x 5 methods, found {len(final)}")
    final.to_csv(OUT / "v7_baseline_loco_results.csv", index=False)
    table = final.pivot(index="held_context", columns="method", values="concordance")
    table.to_csv(OUT / "v7_baseline_loco_concordance_table.csv")
    active = final.loc[~final.held_context.str.endswith("::Placebo")]
    summary = active.groupby("method", as_index=False).agg(mean_active_concordance=("concordance", "mean"), min_active_concordance=("concordance", "min"))
    summary.to_csv(OUT / "v7_baseline_active_summary.csv", index=False)
    print(table.round(3).to_string())
    print("\nActive contexts only:\n" + summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Construct the four manuscript Big Two subsets from scored dialogue records.

Input records must include a stable identifier and z-scored Big Five columns:
agreeableness, conscientiousness, neuroticism, extraversion, and openness.
The script deliberately produces only the four target corners reported in the
paper. It does not include archival or unreported target groups.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "id", "agreeableness", "conscientiousness", "neuroticism", "extraversion", "openness"
}
FULL_BUDGET = 2941


def select(frame: pd.DataFrame, size: int = FULL_BUDGET) -> dict[str, pd.DataFrame]:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    work = frame.copy()
    work["stability"] = (
        work["agreeableness"] + work["conscientiousness"] - work["neuroticism"]
    ) / 3.0
    work["plasticity"] = (work["extraversion"] + work["openness"]) / 2.0
    trait_columns = list(REQUIRED_COLUMNS - {"id"})
    work = work.loc[(work[trait_columns].abs() <= 4).all(axis=1)]

    high_high = work.assign(rank=work["stability"] + work["plasticity"]).nlargest(size, "rank")
    high_low = work.loc[(work["stability"] > 1) & (work["plasticity"] < -1)].copy()
    high_low = high_low.assign(rank=high_low["stability"] - high_low["plasticity"]).nlargest(size, "rank")
    low_high = work.loc[(work["stability"] < -1) & (work["plasticity"] > 1)].copy()
    low_high = low_high.assign(rank=low_high["plasticity"] - low_high["stability"]).nlargest(size, "rank")
    low_low = work.loc[(work["stability"] < -1) & (work["plasticity"] < -1)].copy()
    low_low = low_low.assign(rank=low_low["stability"] + low_low["plasticity"]).nsmallest(size, "rank")

    return {
        "stability_high_plasticity_high": high_high,
        "stability_high_plasticity_low": high_low,
        "stability_low_plasticity_high": low_high,
        "stability_low_plasticity_low": low_low,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--budget", type=int, default=FULL_BUDGET)
    args = parser.parse_args()

    subsets = select(pd.read_csv(args.input_csv), args.budget)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, subset in subsets.items():
        subset.drop(columns="rank").to_csv(args.output_dir / f"{name}.csv", index=False)
        print(f"{name}: {len(subset)} rows")


if __name__ == "__main__":
    main()

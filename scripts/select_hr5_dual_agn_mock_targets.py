#!/usr/bin/env python3
"""Select six well-resolved HR5 dual AGN systems at one output."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output-number", type=int, default=296)
    parser.add_argument("--number", type=int, default=6)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--galaxy-ids", type=Path, required=True)
    return parser.parse_args()


def select_targets(pairs: pd.DataFrame, output_number: int, number: int) -> pd.DataFrame:
    selected = pairs.loc[
        (pairs["output_number"] == output_number)
        & (pairs["pair_class"] == "dual")
        & (pairs["host_relation"] == "distinct PSB galaxies in one FoF halo")
        & (pairs["hr5_100_star_particle_selection"] == 1)
        & (pairs["fable_selection_analogue"] == 1)
        & (pairs["pair_system_multiplicity"] == 2)
    ].sort_values("separation_pkpc")
    if len(selected) < number:
        raise ValueError(f"Only {len(selected)} systems satisfy the selection")
    ranks = np.rint(np.linspace(0, len(selected) - 1, number)).astype(int)
    if len(np.unique(ranks)) != number:
        raise ValueError("The separation ranks are not unique")
    result = selected.iloc[ranks].copy().reset_index(drop=True)
    result.insert(0, "panel", [chr(ord("a") + index) for index in range(number)])
    return result


def main() -> None:
    args = parse_args()
    pairs = pd.read_csv(args.pairs)
    targets = select_targets(pairs, args.output_number, args.number)
    args.table.parent.mkdir(parents=True, exist_ok=True)
    targets.to_csv(args.table, index=False)
    galaxy_ids = np.unique(
        np.concatenate(
            [
                targets["primary_galaxy_gid"].to_numpy(dtype=np.int64),
                targets["secondary_galaxy_gid"].to_numpy(dtype=np.int64),
            ]
        )
    )
    args.galaxy_ids.write_text(
        "".join(f"{galaxy_id}\n" for galaxy_id in galaxy_ids), encoding="ascii"
    )
    print(
        targets[
            [
                "panel",
                "primary_sink_id",
                "secondary_sink_id",
                "separation_pkpc",
                "primary_galaxy_gid",
                "secondary_galaxy_gid",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()

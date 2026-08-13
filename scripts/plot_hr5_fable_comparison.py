#!/usr/bin/env python3
"""Plot the redshift evolution of HR5 capture--host timing for FABLE cuts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = ("#0072B2", "#D55E00", "#009E73", "#7A5195")


def _all_event_bounds(bounds: dict[str, object]) -> tuple[float, float]:
    """Return fractions with the same all-selected-event denominator as FABLE."""

    return (
        float(bounds["all_event_no_added_host_delay_lower_fraction"]),
        float(bounds["all_event_no_added_host_delay_upper_fraction"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "/scratch/kjhan/Hydro/HR5/FoFPSB/02_without_dc/Ver2/"
            "Derived_Sink_Hosts/canonical_v1/capture_host_descendants/"
            "hr5_possible_binary_capture_host_descendants.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/hr5/hr5_fable_capture_host_comparison.pdf"),
    )
    args = parser.parse_args()

    data = json.loads(args.summary.read_text(encoding="utf-8"))
    groups = data["by_host_assignment_output"]
    redshift: list[float] = []
    count: list[int] = []
    lower_fraction: list[float] = []
    upper_fraction: list[float] = []
    unresolved_fraction: list[float] = []
    for output in sorted(groups, key=int):
        group = groups[output]
        analogue = group["fable_event_selection_analogue"]
        number = int(analogue["possible_binary_capture_count"])
        if number == 0:
            continue
        bounds = analogue["timing_fraction_bounds"]
        lower, upper = _all_event_bounds(bounds)
        if lower is None or upper is None:
            continue
        evolution = group["by_selection_output"][str(output)]
        redshift.append(float(evolution["redshift"]))
        count.append(number)
        lower_fraction.append(lower)
        upper_fraction.append(upper)
        unresolved_fraction.append(
            int(bounds["unresolved_host_time_count"]) / number
        )

    redshift_array = np.asarray(redshift)
    order = np.argsort(redshift_array)
    coordinate = np.log10(1.0 + redshift_array[order])
    lower_array = np.asarray(lower_fraction)[order]
    upper_array = np.asarray(upper_fraction)[order]
    unresolved_array = np.asarray(unresolved_fraction)[order]
    count_array = np.asarray(count)[order]

    figure, axes = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True)
    axes[0].plot(coordinate, count_array, color="black", marker="o", markersize=4)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("events per stored interval")

    axes[1].fill_between(
        coordinate,
        lower_array,
        upper_array,
        color=COLORS[0],
        alpha=0.2,
        linewidth=0,
        label="HR5 timing bounds",
    )
    axes[1].vlines(
        coordinate,
        lower_array,
        upper_array,
        color=COLORS[0],
        linewidth=0.7,
        alpha=0.45,
    )
    axes[1].plot(
        coordinate,
        lower_array,
        marker="o",
        markersize=3.5,
        linewidth=1,
        color=COLORS[0],
    )
    axes[1].plot(
        coordinate,
        upper_array,
        marker="o",
        markerfacecolor="white",
        markersize=3.5,
        linewidth=1,
        color=COLORS[0],
    )
    axes[1].plot(
        coordinate,
        unresolved_array,
        color=COLORS[1],
        marker="s",
        markersize=3.5,
        linewidth=1,
        linestyle="--",
        label="unresolved host time",
    )
    benchmark = data["published_fable_benchmark"]
    fable_fraction = float(benchmark["no_added_host_delay_fraction"])
    fable_interval = np.asarray(benchmark["no_added_host_delay_wilson_68"])
    axes[1].axhspan(
        fable_interval[0],
        fable_interval[1],
        color="0.75",
        alpha=0.45,
        linewidth=0,
    )
    axes[1].axhline(
        fable_fraction,
        color="0.35",
        linewidth=1,
        linestyle="--",
        label="FABLE no added host delay",
    )
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("fraction of selected events")
    axes[1].set_xlabel(r"$\log_{10}(1+z)$")
    axes[1].legend(loc="best", fontsize=8, frameon=False)
    for axis in axes:
        axis.tick_params(direction="in", which="both", top=True, right=True)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()

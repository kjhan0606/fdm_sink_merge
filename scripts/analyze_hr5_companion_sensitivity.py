#!/usr/bin/env python3
"""Measure host-delay bounds under stricter assigned-companion criteria."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_EVENT_TABLE = Path(
    "/scratch/kjhan/Hydro/HR5/FoFPSB/02_without_dc/Ver2/"
    "Derived_Sink_Hosts/canonical_v1/capture_host_descendants/"
    "hr5_possible_binary_capture_host_descendants.csv"
)
DEFAULT_OUTPUT_DIRECTORY = Path("results/hr5")
FABLE_SELECTED_EVENT_COUNT = 10_716
FABLE_NO_ADDED_HOST_DELAY_COUNT = 513


def _wilson_interval(success: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    fraction = success / total
    scale = 1.0 + 1.0 / total
    centre = (fraction + 1.0 / (2.0 * total)) / scale
    half_width = (
        np.sqrt(fraction * (1.0 - fraction) / total + 1.0 / (4.0 * total**2))
        / scale
    )
    return float(centre - half_width), float(centre + half_width)


def _timing_fraction_bounds(
    counter: Counter[str], total: int
) -> dict[str, object]:
    certain = (
        counter["common_host_before_later_possible_binary_capture"]
        + counter["common_descendant_before_possible_binary_capture"]
    )
    overlap = counter["time_intervals_overlap"]
    unresolved = counter["host_time_unresolved"]
    if total <= 0:
        raise ValueError("At least one selected event is required")
    return {
        "certain_no_added_host_delay_count": certain,
        "interval_overlap_count": overlap,
        "unresolved_host_time_count": unresolved,
        "all_event_no_added_host_delay_lower_fraction": certain / total,
        "all_event_no_added_host_delay_upper_fraction": (
            certain + overlap + unresolved
        )
        / total,
        "certain_no_added_host_delay_wilson_68": _wilson_interval(certain, total),
    }


def _selection_flags(row: dict[str, str]) -> dict[str, bool]:
    fable = row["fable_selection_analogue"] == "1"
    unique = fable and row["unique_assigned_companion"] == "1"
    phase_space = (
        unique and row["last_resolved_speed_below_point_mass_escape"] == "1"
    )
    return {"all": fable, "unique": unique, "phase_space": phase_space}


def _summary_rows(event_table: Path) -> list[dict[str, object]]:
    counters = {name: Counter() for name in ("all", "unique", "phase_space")}
    counts = Counter()
    with event_table.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "fable_selection_analogue",
            "unique_assigned_companion",
            "last_resolved_speed_below_point_mass_escape",
            "capture_host_time_order",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("The event table lacks assigned-companion diagnostics")
        for row in reader:
            for name, selected in _selection_flags(row).items():
                if not selected:
                    continue
                counts[name] += 1
                counters[name].update([row["capture_host_time_order"]])

    rows: list[dict[str, object]] = []
    labels = {
        "all": "all assigned companions",
        "unique": "unique assignment",
        "phase_space": "unique and point-mass bound",
    }
    for name in ("all", "unique", "phase_space"):
        total = counts[name]
        counter = counters[name]
        bounds = _timing_fraction_bounds(counter, total)
        rows.append(
            {
                "selection": name,
                "selection_label": labels[name],
                "event_count": total,
                "certain_no_added_host_delay_count": bounds[
                    "certain_no_added_host_delay_count"
                ],
                "interval_overlap_count": bounds["interval_overlap_count"],
                "unresolved_host_time_count": bounds[
                    "unresolved_host_time_count"
                ],
                "capture_precedes_host_count": (
                    counter["possible_binary_capture_before_common_descendant"]
                    + counter[
                        "possible_binary_capture_before_last_resolved_distinct_hosts"
                    ]
                ),
                "no_added_host_delay_lower_fraction": bounds[
                    "all_event_no_added_host_delay_lower_fraction"
                ],
                "no_added_host_delay_upper_fraction": bounds[
                    "all_event_no_added_host_delay_upper_fraction"
                ],
                "certain_fraction_wilson_68_lower": bounds[
                    "certain_no_added_host_delay_wilson_68"
                ][0],
                "certain_fraction_wilson_68_upper": bounds[
                    "certain_no_added_host_delay_wilson_68"
                ][1],
            }
        )
    return rows


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(path: Path, rows: list[dict[str, object]]) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 7.0,
            "mathtext.fontset": "stix",
            "axes.labelsize": 7.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "pdf.fonttype": 42,
        }
    )
    labels = [
        "all",
        "unique",
        "unique +\npoint-mass bound",
    ]
    count = np.asarray([row["event_count"] for row in rows], dtype=np.float64)
    lower = np.asarray(
        [row["no_added_host_delay_lower_fraction"] for row in rows],
        dtype=np.float64,
    )
    upper = np.asarray(
        [row["no_added_host_delay_upper_fraction"] for row in rows],
        dtype=np.float64,
    )
    x = np.arange(len(rows))
    figure, axes = plt.subplots(1, 2, figsize=(7.15, 2.65))
    figure.subplots_adjust(wspace=0.34, bottom=0.23, left=0.09, right=0.98, top=0.97)
    axes[0].bar(x, count, width=0.55, color=("#777777", "#0072B2", "#D55E00"))
    axes[0].set_yscale("log")
    axes[0].set_ylabel("number of selected events")
    axes[0].set_xticks(x, labels)
    for position, value in zip(x, count, strict=True):
        axes[0].text(position, value * 1.15, f"{int(value):,}", ha="center", va="bottom")

    midpoint = 0.5 * (lower + upper)
    axes[1].errorbar(
        x,
        midpoint,
        yerr=np.vstack((midpoint - lower, upper - midpoint)),
        fmt="o",
        mfc="white",
        color="#0072B2",
        ms=4.5,
        lw=1.0,
        capsize=2.0,
        label="HR5 timing bounds",
    )
    fable = FABLE_NO_ADDED_HOST_DELAY_COUNT / FABLE_SELECTED_EVENT_COUNT
    axes[1].axhline(
        fable,
        color="0.35",
        ls="--",
        lw=1.0,
        label="FABLE",
    )
    axes[1].set_ylim(0.0, 1.04)
    axes[1].set_ylabel("fraction requiring no added host delay")
    axes[1].set_xticks(x, labels)
    axes[1].legend(frameon=False, loc="center left", handlelength=1.4)
    for label, axis in zip(("(a)", "(b)"), axes, strict=True):
        axis.text(0.03, 0.96, label, transform=axis.transAxes, va="top")
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def analyze(event_table: Path, output_directory: Path) -> dict[str, object]:
    rows = _summary_rows(event_table)
    output_directory.mkdir(parents=True, exist_ok=True)
    table_path = output_directory / "hr5_fable_companion_sensitivity.csv"
    figure_path = output_directory / "hr5_fable_companion_sensitivity.pdf"
    summary_path = output_directory / "hr5_fable_companion_sensitivity.json"
    _write_rows(table_path, rows)
    _plot(figure_path, rows)
    summary: dict[str, object] = {
        "selection": (
            "FABLE SMBH and host-stellar-mass thresholds applied to HR5"
        ),
        "hierarchy": rows,
        "phase_space_note": (
            "The strict sample requires a unique assigned companion and a "
            "relative speed no greater than the escape speed generated by the "
            "two SMBH point masses at the last resolved output. The host "
            "potential and unresolved matter are omitted."
        ),
        "published_fable_no_added_host_delay_fraction": (
            FABLE_NO_ADDED_HOST_DELAY_COUNT / FABLE_SELECTED_EVENT_COUNT
        ),
        "published_fable_count": FABLE_SELECTED_EVENT_COUNT,
        "source_event_table": str(event_table),
        "table_path": str(table_path),
        "figure_path": str(figure_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-table", type=Path, default=DEFAULT_EVENT_TABLE)
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    args = parser.parse_args()
    print(json.dumps(analyze(args.event_table, args.output_directory), indent=2))


if __name__ == "__main__":
    main()

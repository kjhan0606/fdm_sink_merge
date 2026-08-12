#!/usr/bin/env python3
"""Plot host-galaxy outcomes for matched dual- and single-AGN pairs."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM


COLORS = {"dual": "#0072B2", "offset": "#D55E00"}
LABELS = {"dual": "dual AGN", "offset": "single AGN"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _join_bounds(row: dict[str, str], threshold_gyr: float) -> tuple[float, float]:
    status = row["host_track_status"]
    if status == "same_host_at_selection":
        return 1.0, 1.0
    if status == "common_descendant":
        lower_time = float(row["common_descendant_delay_lower_gyr"])
        upper_time = float(row["common_descendant_delay_upper_gyr"])
        return float(upper_time <= threshold_gyr), float(lower_time <= threshold_gyr)
    if status == "right_censored":
        resolved_time = float(row["common_descendant_delay_lower_gyr"])
        if resolved_time >= threshold_gyr:
            return 0.0, 0.0
    return 0.0, 1.0


def _measure(
    rows: list[dict[str, str]],
    threshold_gyr: float,
    bootstrap_count: int,
    final_redshift: float,
) -> list[dict[str, float | int]]:
    grouped: dict[tuple[int, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        grouped[(int(row["selection_output"]), row["match_id"])][
            row["matched_pair_class"]
        ] = row
    rng = np.random.default_rng(20260812)
    cosmology = FlatLambdaCDM(H0=68.4, Om0=0.3, Tcmb0=2.725)
    final_age_gyr = float(cosmology.age(final_redshift).value)
    result: list[dict[str, float | int]] = []
    for output in sorted({key[0] for key in grouped}):
        pairs = [pair for (number, _), pair in grouped.items() if number == output]
        if not pairs:
            continue
        redshift = float(pairs[0]["dual"]["selection_redshift"])
        available_followup_gyr = final_age_gyr - float(cosmology.age(redshift).value)
        size = len(pairs)
        same = {
            pair_class: np.asarray(
                [
                    pair[pair_class]["host_track_status"]
                    == "same_host_at_selection"
                    for pair in pairs
                ],
                dtype=np.float64,
            )
            for pair_class in ("dual", "offset")
        }
        joined = {
            pair_class: np.asarray(
                [_join_bounds(pair[pair_class], threshold_gyr) for pair in pairs]
            )
            for pair_class in ("dual", "offset")
        }
        bootstrap_same = {pair_class: [] for pair_class in same}
        bootstrap_join_lower = {pair_class: [] for pair_class in same}
        bootstrap_join_upper = {pair_class: [] for pair_class in same}
        bootstrap_lower: list[float] = []
        bootstrap_upper: list[float] = []
        for _ in range(bootstrap_count):
            index = rng.integers(0, size, size=size)
            for pair_class in same:
                bootstrap_same[pair_class].append(float(np.mean(same[pair_class][index])))
                bootstrap_join_lower[pair_class].append(
                    float(np.mean(joined[pair_class][index, 0]))
                )
                bootstrap_join_upper[pair_class].append(
                    float(np.mean(joined[pair_class][index, 1]))
                )
            bootstrap_lower.append(
                float(np.mean(joined["dual"][index, 0] - joined["offset"][index, 1]))
            )
            bootstrap_upper.append(
                float(np.mean(joined["dual"][index, 1] - joined["offset"][index, 0]))
            )
        row: dict[str, float | int] = {
            "output": output,
            "redshift": redshift,
            "matched_system_count": size,
            "join_threshold_gyr": threshold_gyr,
            "available_followup_gyr": available_followup_gyr,
            "adequate_followup_for_join_threshold": int(
                available_followup_gyr >= threshold_gyr
            ),
            "dual_minus_single_joined_fraction_lower": float(
                np.mean(joined["dual"][:, 0] - joined["offset"][:, 1])
            ),
            "dual_minus_single_joined_fraction_upper": float(
                np.mean(joined["dual"][:, 1] - joined["offset"][:, 0])
            ),
            "dual_minus_single_joined_fraction_bootstrap_outer_lower": float(
                np.quantile(bootstrap_lower, 0.16)
            ),
            "dual_minus_single_joined_fraction_bootstrap_outer_upper": float(
                np.quantile(bootstrap_upper, 0.84)
            ),
        }
        for pair_class in ("dual", "offset"):
            row[f"{pair_class}_same_host_fraction"] = float(np.mean(same[pair_class]))
            row[f"{pair_class}_same_host_bootstrap_16"] = float(
                np.quantile(bootstrap_same[pair_class], 0.16)
            )
            row[f"{pair_class}_same_host_bootstrap_84"] = float(
                np.quantile(bootstrap_same[pair_class], 0.84)
            )
            row[f"{pair_class}_joined_fraction_lower"] = float(
                np.mean(joined[pair_class][:, 0])
            )
            row[f"{pair_class}_joined_fraction_upper"] = float(
                np.mean(joined[pair_class][:, 1])
            )
            row[f"{pair_class}_joined_fraction_bootstrap_outer_lower"] = float(
                np.quantile(bootstrap_join_lower[pair_class], 0.16)
            )
            row[f"{pair_class}_joined_fraction_bootstrap_outer_upper"] = float(
                np.quantile(bootstrap_join_upper[pair_class], 0.84)
            )
        result.append(row)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--table",
        type=Path,
        default=Path(
            "/scratch/kjhan/Hydro/HR5/FoFPSB/02_without_dc/Ver2/"
            "Derived_Sink_Hosts/canonical_v1/matched_pair_host_descendants/"
            "hr5_matched_agn_pair_host_descendants.csv"
        ),
    )
    parser.add_argument("--join-threshold-gyr", type=float, default=0.5)
    parser.add_argument("--final-redshift", type=float, default=0.6253607831404921)
    parser.add_argument("--bootstrap-count", type=int, default=2000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/hr5/hr5_matched_pair_host_evolution.pdf"),
    )
    args = parser.parse_args()
    if (
        args.join_threshold_gyr <= 0.0
        or args.bootstrap_count < 1
        or args.final_redshift < 0.0
    ):
        parser.error("The time threshold and bootstrap count must be positive")

    measured = _measure(
        _read_csv(args.table),
        args.join_threshold_gyr,
        args.bootstrap_count,
        args.final_redshift,
    )
    measured.sort(key=lambda row: float(row["redshift"]))
    followup = [
        row
        for row in measured
        if int(row["adequate_followup_for_join_threshold"])
    ]
    followup_redshift = np.asarray([float(row["redshift"]) for row in followup])

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.1, 3.0))
    for pair_class, offset in (("dual", -0.02), ("offset", 0.02)):
        lower = np.asarray(
            [float(row[f"{pair_class}_joined_fraction_lower"]) for row in followup]
        )
        upper = np.asarray(
            [float(row[f"{pair_class}_joined_fraction_upper"]) for row in followup]
        )
        outer_lower = np.asarray(
            [
                float(row[f"{pair_class}_joined_fraction_bootstrap_outer_lower"])
                for row in followup
            ]
        )
        outer_upper = np.asarray(
            [
                float(row[f"{pair_class}_joined_fraction_bootstrap_outer_upper"])
                for row in followup
            ]
        )
        value = 0.5 * (lower + upper)
        axes[0].errorbar(
            followup_redshift + offset,
            value,
            yerr=np.vstack((value - outer_lower, outer_upper - value)),
            color=COLORS[pair_class],
            marker="o" if pair_class == "dual" else "s",
            mfc="white",
            markersize=4,
            linewidth=0.9,
            capsize=2,
            label=LABELS[pair_class],
        )
        axes[0].vlines(
            followup_redshift + offset,
            lower,
            upper,
            color=COLORS[pair_class],
            linewidth=3,
            alpha=0.35,
        )
    axes[0].set_xlabel(r"$z$")
    axes[0].set_ylabel(
        rf"fraction joined by {args.join_threshold_gyr:g} Gyr"
    )
    axes[0].set_ylim(bottom=0.0)
    axes[0].legend(frameon=False, loc="upper right")
    axes[0].text(0.43, 0.94, "(a)", transform=axes[0].transAxes, va="top")

    lower = np.asarray(
        [float(row["dual_minus_single_joined_fraction_lower"]) for row in followup]
    )
    upper = np.asarray(
        [float(row["dual_minus_single_joined_fraction_upper"]) for row in followup]
    )
    outer_lower = np.asarray(
        [
            float(row["dual_minus_single_joined_fraction_bootstrap_outer_lower"])
            for row in followup
        ]
    )
    outer_upper = np.asarray(
        [
            float(row["dual_minus_single_joined_fraction_bootstrap_outer_upper"])
            for row in followup
        ]
    )
    centre = 0.5 * (lower + upper)
    axes[1].errorbar(
        followup_redshift,
        centre,
        yerr=np.vstack((centre - outer_lower, outer_upper - centre)),
        color="#009E73",
        marker="D",
        mfc="white",
        markersize=4,
        linewidth=0.9,
        capsize=2,
    )
    axes[1].vlines(
        followup_redshift, lower, upper, color="#009E73", linewidth=3, alpha=0.45
    )
    axes[1].axhline(0.0, color="0.4", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel(r"$z$")
    axes[1].set_ylabel(
        rf"dual $-$ single joined fraction by {args.join_threshold_gyr:g} Gyr"
    )
    axes[1].text(0.43, 0.94, "(b)", transform=axes[1].transAxes, va="top")
    for axis in axes:
        axis.tick_params(direction="in", which="both", top=True, right=True)
    figure.tight_layout(w_pad=1.4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    plt.close(figure)

    table_path = args.output.with_suffix(".csv")
    with table_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(measured[0]))
        writer.writeheader()
        writer.writerows(measured)


if __name__ == "__main__":
    main()

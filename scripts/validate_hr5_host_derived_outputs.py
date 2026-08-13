#!/usr/bin/env python3
"""Validate the host demographics and assigned-companion sensitivity results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DEFAULT_RESULT_ROOT = Path("results/hr5")
EXPECTED_OUTPUT_COUNT = 17
EXPECTED_SPATIAL_PAIR_COUNT = 15_946
EXPECTED_CLASSIFIABLE_PAIR_COUNT = 15_940
EXPECTED_DISTINCT_HOST_PAIR_COUNT = 14_698
EXPECTED_SENSITIVITY_COUNTS = (25_494, 22_451, 33)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _valid_pdf(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 10_000:
        return False
    with path.open("rb") as stream:
        if stream.read(5) != b"%PDF-":
            return False
        stream.seek(max(0, path.stat().st_size - 1024))
        return b"%%EOF" in stream.read()


def validate(result_root: Path) -> dict[str, object]:
    dual_root = result_root / "dual_agn"
    demographics_path = dual_root / "hr5_dual_agn_host_demographics.csv"
    fit_path = dual_root / "hr5_dual_agn_host_modified_schechter_parameters.csv"
    demographics_figure = dual_root / "hr5_dual_agn_host_demographics.pdf"
    sensitivity_path = result_root / "hr5_fable_companion_sensitivity.csv"
    sensitivity_figure = result_root / "hr5_fable_companion_sensitivity.pdf"
    matched_path = result_root / "hr5_matched_pair_host_evolution_fable.csv"
    matched_figure = result_root / "hr5_matched_pair_host_evolution_fable.pdf"

    demographics = _read_csv(demographics_path)
    if len(demographics) != EXPECTED_OUTPUT_COUNT:
        raise ValueError("Host demographics do not contain all MkAGN outputs")
    spatial_count = sum(int(row["spatial_pair_count"]) for row in demographics)
    classifiable_count = sum(
        int(row["pair_count_with_two_psb_hosts"]) for row in demographics
    )
    distinct_count = sum(int(row["distinct_host_pair_count"]) for row in demographics)
    if (
        spatial_count != EXPECTED_SPATIAL_PAIR_COUNT
        or classifiable_count != EXPECTED_CLASSIFIABLE_PAIR_COUNT
        or distinct_count != EXPECTED_DISTINCT_HOST_PAIR_COUNT
    ):
        raise ValueError("Host-demographic totals differ from the canonical catalogue")
    for row in demographics:
        density = float(row["distinct_host_number_density_cmpc3"])
        error = float(row["distinct_host_number_density_jackknife_error_cmpc3"])
        count = int(row["distinct_host_pair_count"])
        if count > 0 and (density <= 0.0 or not np.isfinite(error) or error < 0.0):
            raise ValueError("A detected distinct-host population has invalid density")

    fit = _read_csv(fit_path)
    if {row["population"] for row in fit} != {"spatial_pair", "distinct_host"}:
        raise ValueError("Modified Schechter table has incomplete population coverage")
    for row in fit:
        values = np.asarray(
            [
                float(row["phi_star_cmpc3"]),
                float(row["z_star"]),
                float(row["alpha"]),
                float(row["beta"]),
                float(row["rms_log10_residual"]),
            ]
        )
        if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("Modified Schechter fit contains invalid parameters")

    sensitivity = _read_csv(sensitivity_path)
    counts = tuple(int(row["event_count"]) for row in sensitivity)
    if counts != EXPECTED_SENSITIVITY_COUNTS:
        raise ValueError("Assigned-companion hierarchy has unexpected event counts")
    if not counts[0] >= counts[1] >= counts[2]:
        raise ValueError("Assigned-companion selections are not nested")
    for row in sensitivity:
        lower = float(row["no_added_host_delay_lower_fraction"])
        upper = float(row["no_added_host_delay_upper_fraction"])
        if not 0.0 <= lower <= upper <= 1.0:
            raise ValueError("Assigned-companion timing bounds are invalid")

    matched = _read_csv(matched_path)
    if len(matched) != 3 or sum(int(row["matched_system_count"]) for row in matched) != 281:
        raise ValueError("FABLE-analogue matched host sample is incomplete")
    for row in matched:
        lower = float(row["dual_minus_single_joined_fraction_lower"])
        upper = float(row["dual_minus_single_joined_fraction_upper"])
        if not lower <= 0.0 <= upper:
            raise ValueError("A matched host difference no longer includes zero")

    figures = (demographics_figure, sensitivity_figure, matched_figure)
    if not all(_valid_pdf(path) for path in figures):
        raise ValueError("At least one host-analysis figure is incomplete")
    return {
        "validated": True,
        "host_demographic_output_count": len(demographics),
        "spatial_pair_count": spatial_count,
        "pair_count_with_two_psb_hosts": classifiable_count,
        "distinct_host_pair_count": distinct_count,
        "companion_sensitivity_event_counts": list(counts),
        "matched_host_system_count": 281,
        "figure_paths": [str(path) for path in figures],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/hr5/hr5_host_derived_validation.json"),
    )
    args = parser.parse_args()
    result = validate(args.result_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

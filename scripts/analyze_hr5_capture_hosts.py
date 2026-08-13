#!/usr/bin/env python3
"""Compare HR5 possible binary captures with mergers of their PSB hosts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from astropy.cosmology import FlatLambdaCDM

from analyze_hr5_host_descendants import _redshifts, _summary, _tree_paths, trace_pairs
from fdm_smbh_delay.hr5 import (
    HOST_RELATION_LABELS,
    classify_sink_pair_hosts,
    read_mkagn_snapshot,
    read_sink_host_catalog,
)

from analyze_hr5_dual_agn_hosts import FABLE_MINIMUM_HOST_STELLAR_MASS_MSUN


DEFAULT_HR5_ROOT = Path("/scratch/kjhan/Hydro/HR5/FoFPSB/02_without_dc/Ver2")
DEFAULT_CANONICAL_ROOT = DEFAULT_HR5_ROOT / "Derived_Sink_Hosts" / "canonical_v1"

FABLE_NO_ADDED_HOST_DELAY_COUNT = 513
FABLE_SELECTED_EVENT_COUNT = 10_716


def _wilson_interval(success: int, total: int, z_score: float = 1.0) -> list[float]:
    """Return a Wilson binomial interval, using one standard deviation by default."""

    if total <= 0:
        return [float("nan"), float("nan")]
    fraction = success / total
    scale = 1.0 + z_score**2 / total
    centre = (fraction + z_score**2 / (2.0 * total)) / scale
    half_width = (
        z_score
        * np.sqrt(
            fraction * (1.0 - fraction) / total
            + z_score**2 / (4.0 * total**2)
        )
        / scale
    )
    return [centre - half_width, centre + half_width]


def _timing_fraction_bounds(counter: Counter[str], total: int) -> dict[str, object]:
    certain_no_delay = (
        counter["common_host_before_later_possible_binary_capture"]
        + counter["common_descendant_before_possible_binary_capture"]
    )
    overlap = counter["time_intervals_overlap"]
    unresolved = counter["host_time_unresolved"]
    resolved = total - unresolved
    result: dict[str, object] = {
        "event_count": total,
        "certain_no_added_host_delay_count": certain_no_delay,
        "interval_overlap_count": overlap,
        "unresolved_host_time_count": unresolved,
        "resolved_timing_count": resolved,
        "all_event_no_added_host_delay_lower_fraction": (
            certain_no_delay / total if total else None
        ),
        "all_event_no_added_host_delay_upper_fraction": (
            (certain_no_delay + overlap + unresolved) / total if total else None
        ),
        "certain_no_added_host_delay_wilson_68": (
            _wilson_interval(certain_no_delay, total) if total else None
        ),
    }
    if resolved > 0:
        result.update(
            {
                "resolved_no_added_host_delay_lower_fraction": (
                    certain_no_delay / resolved
                ),
                "resolved_no_added_host_delay_upper_fraction": (
                    (certain_no_delay + overlap) / resolved
                ),
            }
        )
    else:
        result.update(
            {
                "resolved_no_added_host_delay_lower_fraction": None,
                "resolved_no_added_host_delay_upper_fraction": None,
            }
        )
    return result


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _manifest_batches(
    rows: list[dict[str, str]], maximum_event_count: int
) -> list[list[dict[str, str]]]:
    """Group host outputs without splitting the events from one output."""

    batches: list[list[dict[str, str]]] = []
    current: list[dict[str, str]] = []
    current_count = 0
    for row in rows:
        count = int(row["possible_binary_capture_count"])
        if current and current_count + count > maximum_event_count:
            batches.append(current)
            current = []
            current_count = 0
        current.append(row)
        current_count += count
    if current:
        batches.append(current)
    return batches


def _receiver_validation_index(
    path: Path | None,
) -> dict[tuple[int, int, int], dict[str, str]]:
    if path is None or not path.is_file():
        return {}
    result: dict[tuple[int, int, int], dict[str, str]] = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = (
                int(row["sink_id"]),
                int(row["receiver_id"]),
                int(row["assigned_capture_output"]),
            )
            result[key] = row
    return result


def _host_value(
    hosts: np.ndarray,
    row: np.ndarray,
    field: str,
    fill: float,
) -> np.ndarray:
    value = np.full(row.size, fill, dtype=np.float64)
    found = row >= 0
    value[found] = hosts[field][row[found]]
    return value


def _agn_pair_state(
    snapshot_path: Path | None,
    first_id: np.ndarray,
    second_id: np.ndarray,
    dimensionless_hubble: float,
    luminosity_threshold_erg_s: float,
) -> dict[str, np.ndarray]:
    size = first_id.size
    missing = {
        "first_lbol": np.full(size, np.nan),
        "second_lbol": np.full(size, np.nan),
        "first_eddington_ratio": np.full(size, np.nan),
        "second_eddington_ratio": np.full(size, np.nan),
        "state": np.full(size, "no MkAGN measurement", dtype=object),
    }
    if snapshot_path is None or not snapshot_path.is_file():
        return missing
    _, _, records = read_mkagn_snapshot(snapshot_path)
    required = {"sink_id", "mass", "Lbol"}
    if records.dtype.names is None or not required.issubset(records.dtype.names):
        return missing

    order = np.argsort(records["sink_id"])
    sorted_id = np.asarray(records["sink_id"][order], dtype=np.int64)

    def values(identifier: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        position = np.searchsorted(sorted_id, identifier)
        found = position < sorted_id.size
        found[found] &= sorted_id[position[found]] == identifier[found]
        luminosity = np.full(identifier.size, np.nan)
        mass = np.full(identifier.size, np.nan)
        luminosity[found] = records["Lbol"][order[position[found]]]
        mass[found] = records["mass"][order[position[found]]] / dimensionless_hubble
        eddington_ratio = luminosity / (1.26e38 * mass)
        return luminosity, eddington_ratio, found

    first_lbol, first_eddington, first_found = values(first_id)
    second_lbol, second_eddington, second_found = values(second_id)
    measured = first_found & second_found
    first_active = measured & np.isfinite(first_lbol) & (
        first_lbol >= luminosity_threshold_erg_s
    )
    second_active = measured & np.isfinite(second_lbol) & (
        second_lbol >= luminosity_threshold_erg_s
    )
    state = np.full(size, "SMBH missing from MkAGN snapshot", dtype=object)
    state[measured & ~first_active & ~second_active] = "neither SMBH active"
    state[first_active ^ second_active] = "one SMBH active"
    state[first_active & second_active] = "both SMBHs active"
    return {
        "first_lbol": first_lbol,
        "second_lbol": second_lbol,
        "first_eddington_ratio": first_eddington,
        "second_eddington_ratio": second_eddington,
        "state": state,
    }


def _event_rows(
    manifest_row: dict[str, str],
    dimensionless_hubble: float,
    receiver_validation: dict[tuple[int, int, int], dict[str, str]],
    agn_snapshot_path: Path | None,
    luminosity_threshold_erg_s: float,
) -> tuple[list[dict[str, str]], dict[tuple[int, int, int], dict[str, str]]]:
    events = _read_csv(Path(manifest_row["capture_event_path"]))
    hosts = read_sink_host_catalog(Path(manifest_row["host_catalogue_path"]))
    first_id = np.asarray([int(row["sink_id"]) for row in events], dtype=np.int64)
    second_id = np.asarray([int(row["receiver_id"]) for row in events], dtype=np.int64)
    relation, first_row, second_row = classify_sink_pair_hosts(first_id, second_id, hosts)
    first_gid = _host_value(hosts, first_row, "galaxy_gid", -1).astype(np.int64)
    second_gid = _host_value(hosts, second_row, "galaxy_gid", -1).astype(np.int64)
    first_fof = _host_value(hosts, first_row, "fof_index", -1).astype(np.int64)
    second_fof = _host_value(hosts, second_row, "fof_index", -1).astype(np.int64)
    first_stars = _host_value(hosts, first_row, "host_stellar_count", -1).astype(np.int64)
    second_stars = _host_value(hosts, second_row, "host_stellar_count", -1).astype(np.int64)
    first_stellar_mass = _host_value(
        hosts, first_row, "host_stellar_mass_msun_h", np.nan
    ) / dimensionless_hubble
    second_stellar_mass = _host_value(
        hosts, second_row, "host_stellar_mass_msun_h", np.nan
    ) / dimensionless_hubble
    first_total_mass = _host_value(
        hosts, first_row, "host_total_mass_msun_h", np.nan
    ) / dimensionless_hubble
    second_total_mass = _host_value(
        hosts, second_row, "host_total_mass_msun_h", np.nan
    ) / dimensionless_hubble
    agn = _agn_pair_state(
        agn_snapshot_path,
        first_id,
        second_id,
        dimensionless_hubble,
        luminosity_threshold_erg_s,
    )

    output = int(manifest_row["output"])
    redshift = float(manifest_row["redshift"])
    cosmology = FlatLambdaCDM(H0=68.4, Om0=0.3, Tcmb0=2.725)
    host_time = float(cosmology.age(redshift).value)
    capture_index: dict[tuple[int, int, int], dict[str, str]] = {}
    result: list[dict[str, str]] = []
    for index, event in enumerate(events):
        first_mass = float(event["minor_mass_last_resolved_msun"])
        second_mass = float(event["receiver_mass_last_resolved_msun"])
        hr5_100_star_particle_selection = (
            first_mass >= 1.0e6
            and second_mass >= 1.0e6
            and first_stars[index] >= 100
            and second_stars[index] >= 100
        )
        fable_analogue = (
            first_mass >= 1.0e6
            and second_mass >= 1.0e6
            and first_stellar_mass[index] >= FABLE_MINIMUM_HOST_STELLAR_MASS_MSUN
            and second_stellar_mass[index] >= FABLE_MINIMUM_HOST_STELLAR_MASS_MSUN
        )
        row = dict(event)
        diagnostic = receiver_validation.get(
            (
                int(first_id[index]),
                int(second_id[index]),
                int(event["assigned_capture_output"]),
            )
        )
        if diagnostic is None:
            diagnostic_values = {
                "last_resolved_pair_separation_pkpc": "nan",
                "last_resolved_relative_speed_kms": "nan",
                "last_resolved_speed_to_point_mass_escape_ratio": "nan",
                "last_resolved_speed_below_point_mass_escape": "0",
                "simultaneous_assignment_multiplicity": "-1",
                "unique_assigned_companion": "0",
            }
        else:
            speed_ratio = float(diagnostic["speed_to_escape_ratio_last_resolved"])
            multiplicity = int(diagnostic["simultaneous_assignment_multiplicity"])
            diagnostic_values = {
                "last_resolved_pair_separation_pkpc": diagnostic[
                    "last_resolved_separation_pkpc"
                ],
                "last_resolved_relative_speed_kms": diagnostic[
                    "relative_speed_last_resolved_kms"
                ],
                "last_resolved_speed_to_point_mass_escape_ratio": str(speed_ratio),
                "last_resolved_speed_below_point_mass_escape": str(
                    int(np.isfinite(speed_ratio) and speed_ratio <= 1.0)
                ),
                "simultaneous_assignment_multiplicity": str(multiplicity),
                "unique_assigned_companion": str(int(multiplicity == 1)),
            }
        row.update(
            {
                "selection_output": str(output),
                "selection_redshift": str(redshift),
                "primary_sink_id": str(first_id[index]),
                "secondary_sink_id": str(second_id[index]),
                "primary_galaxy_gid": str(first_gid[index]),
                "secondary_galaxy_gid": str(second_gid[index]),
                "primary_fof_index": str(first_fof[index]),
                "secondary_fof_index": str(second_fof[index]),
                "host_relation": str(HOST_RELATION_LABELS[relation[index]]),
                "primary_host_stellar_mass_msun": str(first_stellar_mass[index]),
                "secondary_host_stellar_mass_msun": str(second_stellar_mass[index]),
                "primary_host_total_mass_msun": str(first_total_mass[index]),
                "secondary_host_total_mass_msun": str(second_total_mass[index]),
                "primary_host_stellar_particle_count": str(first_stars[index]),
                "secondary_host_stellar_particle_count": str(second_stars[index]),
                "hr5_100_star_particle_selection": str(
                    int(hr5_100_star_particle_selection)
                ),
                "fable_selection_analogue": str(int(fable_analogue)),
                "primary_lbol_erg_s": str(agn["first_lbol"][index]),
                "secondary_lbol_erg_s": str(agn["second_lbol"][index]),
                "primary_eddington_ratio": str(
                    agn["first_eddington_ratio"][index]
                ),
                "secondary_eddington_ratio": str(
                    agn["second_eddington_ratio"][index]
                ),
                "agn_pair_state": str(agn["state"][index]),
                **diagnostic_values,
            }
        )
        result.append(row)
        key_ids = sorted((int(first_id[index]), int(second_id[index])))
        capture_index[(output, key_ids[0], key_ids[1])] = {
            "assigned_capture_output": event["assigned_capture_output"],
            "capture_delay_lower_gyr": str(
                max(0.0, float(event["last_resolved_cosmic_time_gyr"]) - host_time)
            ),
            "capture_delay_upper_gyr": str(
                max(0.0, float(event["assigned_capture_cosmic_time_gyr"]) - host_time)
            ),
            "pair_class": "possible binary capture",
        }
    return result, capture_index


def _evolution_rows(by_output: dict[str, object]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for output in sorted(by_output, key=int):
        summary = by_output[output]
        evolution = summary["by_selection_output"][output]
        analogue = summary["fable_event_selection_analogue"]
        timing = analogue["capture_host_time_order"]
        bounds = analogue["timing_fraction_bounds"]
        activity = analogue["agn_pair_state"]
        diagnostics = analogue["assigned_companion_diagnostics"]
        result.append(
            {
                "host_assignment_output": int(output),
                "redshift": float(evolution["redshift"]),
                "possible_binary_capture_count": int(summary["pair_count"]),
                "fable_selection_analogue_count": int(
                    analogue["possible_binary_capture_count"]
                ),
                "both_smbhs_active_count": int(activity.get("both SMBHs active", 0)),
                "one_smbh_active_count": int(activity.get("one SMBH active", 0)),
                "neither_smbh_active_count": int(
                    activity.get("neither SMBH active", 0)
                ),
                "no_mkagn_measurement_count": int(
                    activity.get("no MkAGN measurement", 0)
                ),
                "smbh_missing_from_mkagn_count": int(
                    activity.get("SMBH missing from MkAGN snapshot", 0)
                ),
                "certain_no_added_host_delay_count": int(
                    bounds["certain_no_added_host_delay_count"]
                ),
                "interval_overlap_count": int(bounds["interval_overlap_count"]),
                "unresolved_host_time_count": int(
                    bounds["unresolved_host_time_count"]
                ),
                "possible_binary_capture_before_common_descendant_count": int(
                    timing.get("possible_binary_capture_before_common_descendant", 0)
                ),
                "possible_binary_capture_before_last_resolved_distinct_hosts_count": int(
                    timing.get(
                        "possible_binary_capture_before_last_resolved_distinct_hosts",
                        0,
                    )
                ),
                "all_event_no_added_host_delay_lower_fraction": bounds[
                    "all_event_no_added_host_delay_lower_fraction"
                ],
                "all_event_no_added_host_delay_upper_fraction": bounds[
                    "all_event_no_added_host_delay_upper_fraction"
                ],
                "unique_assigned_companion_count": int(
                    diagnostics["unique_assignment_count"]
                ),
                "speed_below_point_mass_escape_count": int(
                    diagnostics["speed_below_point_mass_escape_count"]
                ),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    parser.add_argument(
        "--capture-host-root",
        type=Path,
        default=DEFAULT_CANONICAL_ROOT / "capture_hosts",
    )
    parser.add_argument(
        "--tree-root", type=Path, default=DEFAULT_HR5_ROOT / "Galaxy_Merging"
    )
    parser.add_argument("--outputs", nargs="+", type=int)
    parser.add_argument("--dimensionless-hubble", type=float, default=0.684)
    parser.add_argument("--luminosity-threshold-erg-s", type=float, default=1.0e43)
    parser.add_argument("--maximum-events-per-trace", type=int, default=50_000)
    parser.add_argument(
        "--receiver-validation",
        type=Path,
        default=Path(
            "results/hr5/receiver_validation/hr5_receiver_validation.csv"
        ),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_CANONICAL_ROOT / "capture_host_descendants",
    )
    args = parser.parse_args()
    if args.maximum_events_per_trace < 1:
        parser.error("--maximum-events-per-trace must be positive")

    manifest = _read_csv(args.capture_host_root / "hr5_capture_host_manifest.csv")
    selected = [
        row
        for row in manifest
        if row["capture_event_status"] == "complete"
        and row["host_catalogue_status"] == "complete"
    ]
    if args.outputs:
        requested = set(args.outputs)
        selected = [row for row in selected if int(row["output"]) in requested]
        missing = requested - {int(row["output"]) for row in selected}
        if missing:
            parser.error(f"Incomplete capture-host inputs for outputs {sorted(missing)}")
    if not selected:
        raise ValueError("No complete possible-binary-capture host inputs")

    receiver_validation = _receiver_validation_index(args.receiver_validation)
    canonical_manifest = {
        int(row["output"]): row
        for row in _read_csv(args.canonical_root / "hr5_output_manifest.csv")
    }

    tree_outputs, tree_paths = _tree_paths(args.tree_root)
    redshift = _redshifts(
        args.canonical_root / "hr5_output_manifest.csv", tree_outputs
    )
    args.output_directory.mkdir(parents=True, exist_ok=True)
    table_path = args.output_directory / "hr5_possible_binary_capture_host_descendants.csv"
    temporary_table = table_path.with_suffix(".csv.tmp")
    writer: csv.DictWriter | None = None
    stream = temporary_table.open("w", newline="")
    total_status: Counter[str] = Counter()
    fable_status: Counter[str] = Counter()
    time_order: Counter[str] = Counter()
    fable_time_order: Counter[str] = Counter()
    by_output: dict[str, object] = {}
    pair_count = 0
    fable_count = 0
    fable_unique_assignment_count = 0
    fable_speed_below_point_mass_escape_count = 0
    agn_pair_state: Counter[str] = Counter()
    fable_agn_pair_state: Counter[str] = Counter()
    fable_timing_by_agn_state: dict[str, Counter[str]] = {}
    try:
        batches = _manifest_batches(selected, args.maximum_events_per_trace)
        for batch_number, batch in enumerate(batches, start=1):
            print(
                f"Tracing capture-host batch {batch_number} of {len(batches)}",
                flush=True,
            )
            rows: list[dict[str, str]] = []
            capture_index: dict[tuple[int, int, int], dict[str, str]] = {}
            for manifest_row in batch:
                output = int(manifest_row["output"])
                mkagn_text = canonical_manifest.get(output, {}).get("mkagn_path", "")
                output_rows, output_capture_index = _event_rows(
                    manifest_row,
                    args.dimensionless_hubble,
                    receiver_validation,
                    Path(mkagn_text) if mkagn_text else None,
                    args.luminosity_threshold_erg_s,
                )
                overlap = capture_index.keys() & output_capture_index.keys()
                if overlap:
                    raise ValueError("Duplicate possible binary capture pair in one batch")
                rows.extend(output_rows)
                capture_index.update(output_capture_index)
            traced_batch = trace_pairs(
                rows,
                tree_outputs,
                tree_paths,
                redshift,
                capture_index,
            )
            traced_by_output: dict[int, list[dict[str, object]]] = {}
            for row in traced_batch:
                traced_by_output.setdefault(int(row["selection_output"]), []).append(row)
            for manifest_row in batch:
                output = int(manifest_row["output"])
                traced = traced_by_output[output]
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(traced[0]))
                    writer.writeheader()
                writer.writerows(traced)
                summary = _summary(traced, tree_outputs)
                pair_count += len(traced)
                total_status.update(row["host_track_status"] for row in traced)
                time_order.update(row["capture_host_time_order"] for row in traced)
                agn_pair_state.update(row["agn_pair_state"] for row in traced)
                fable_rows = [
                    row
                    for row in traced
                    if int(row["fable_selection_analogue"]) == 1
                ]
                fable_count += len(fable_rows)
                fable_unique_assignment_count += sum(
                    int(row["unique_assigned_companion"]) for row in fable_rows
                )
                fable_speed_below_point_mass_escape_count += sum(
                    int(row["last_resolved_speed_below_point_mass_escape"])
                    for row in fable_rows
                )
                fable_status.update(row["host_track_status"] for row in fable_rows)
                fable_agn_pair_state.update(
                    row["agn_pair_state"] for row in fable_rows
                )
                fable_time_order.update(
                    row["capture_host_time_order"] for row in fable_rows
                )
                for row in fable_rows:
                    state = str(row["agn_pair_state"])
                    fable_timing_by_agn_state.setdefault(state, Counter()).update(
                        [str(row["capture_host_time_order"])]
                    )
                summary["fable_event_selection_analogue"] = {
                    "possible_binary_capture_count": len(fable_rows),
                    "agn_pair_state": dict(
                        Counter(row["agn_pair_state"] for row in fable_rows)
                    ),
                    "host_track_status": dict(
                        Counter(row["host_track_status"] for row in fable_rows)
                    ),
                    "capture_host_time_order": dict(
                        Counter(row["capture_host_time_order"] for row in fable_rows)
                    ),
                }
                summary["fable_event_selection_analogue"][
                    "timing_fraction_bounds"
                ] = _timing_fraction_bounds(
                    Counter(row["capture_host_time_order"] for row in fable_rows),
                    len(fable_rows),
                )
                summary["fable_event_selection_analogue"][
                    "assigned_companion_diagnostics"
                ] = {
                    "unique_assignment_count": sum(
                        int(row["unique_assigned_companion"])
                        for row in fable_rows
                    ),
                    "speed_below_point_mass_escape_count": sum(
                        int(row["last_resolved_speed_below_point_mass_escape"])
                        for row in fable_rows
                    ),
                }
                by_output[str(output)] = summary
    finally:
        stream.close()
    temporary_table.replace(table_path)

    combined = {
        "possible_binary_capture_count": pair_count,
        "host_track_status": dict(total_status),
        "fable_selection_analogue_possible_binary_capture_count": fable_count,
        "fable_selection_analogue_host_track_status": dict(fable_status),
        "fable_selection_analogue_capture_host_time_order": dict(fable_time_order),
        "fable_selection_analogue_timing_fraction_bounds": _timing_fraction_bounds(
            fable_time_order, fable_count
        ),
        "fable_host_resolution_note": (
            "The numerical stellar-mass threshold matches the published FABLE "
            "value, but HR5 uses total PSB stellar mass rather than mass within "
            "twice the stellar half-mass radius."
        ),
        "fable_minimum_host_stellar_mass_msun": FABLE_MINIMUM_HOST_STELLAR_MASS_MSUN,
        "fable_selection_analogue_assigned_companion_diagnostics": {
            "receiver_validation_row_count": len(receiver_validation),
            "unique_assignment_count": fable_unique_assignment_count,
            "speed_below_point_mass_escape_count": (
                fable_speed_below_point_mass_escape_count
            ),
        },
        "capture_host_time_order": dict(time_order),
        "luminosity_threshold_erg_s": args.luminosity_threshold_erg_s,
        "agn_pair_state": dict(agn_pair_state),
        "fable_selection_analogue_agn_pair_state": dict(fable_agn_pair_state),
        "fable_selection_analogue_timing_by_agn_state": {
            state: {
                "capture_host_time_order": dict(counter),
                "timing_fraction_bounds": _timing_fraction_bounds(
                    counter, sum(counter.values())
                ),
            }
            for state, counter in sorted(fable_timing_by_agn_state.items())
        },
        "assigned_companion_diagnostic_note": (
            "The speed criterion uses only the mutual point-mass SMBH potential "
            "at the last resolved output. It tests the legacy companion assignment "
            "and is not a complete binding criterion in the host potential."
        ),
        "published_fable_benchmark": {
            "selected_numerical_bh_merger_count": FABLE_SELECTED_EVENT_COUNT,
            "no_added_host_delay_count": FABLE_NO_ADDED_HOST_DELAY_COUNT,
            "no_added_host_delay_fraction": (
                FABLE_NO_ADDED_HOST_DELAY_COUNT / FABLE_SELECTED_EVENT_COUNT
            ),
            "no_added_host_delay_wilson_68": _wilson_interval(
                FABLE_NO_ADDED_HOST_DELAY_COUNT, FABLE_SELECTED_EVENT_COUNT
            ),
            "median_macrophysical_delay_gyr": 1.3,
            "host_pair_not_merged_by_z0_fraction": 0.29,
            "source": "Buttigieg et al. 2025, MNRAS, 542, 2019",
        },
        "by_host_assignment_output": by_output,
        "caveat": (
            "The surviving SMBH is assigned from the legacy distance and mass "
            "criteria because the sink histories do not store the companion "
            "selected by the simulation."
        ),
    }
    summary_path = args.output_directory / "hr5_possible_binary_capture_host_descendants.json"
    temporary_summary = summary_path.with_suffix(".json.tmp")
    evolution_path = args.output_directory / "hr5_fable_capture_host_evolution.csv"
    temporary_evolution = evolution_path.with_suffix(".csv.tmp")
    evolution_rows = _evolution_rows(by_output)
    with temporary_evolution.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(evolution_rows[0]))
        writer.writeheader()
        writer.writerows(evolution_rows)
    temporary_summary.write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
    temporary_evolution.replace(evolution_path)
    temporary_summary.replace(summary_path)
    print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()

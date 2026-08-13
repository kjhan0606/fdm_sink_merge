#!/usr/bin/env python3
"""Validate the completed HR5 capture-host and FABLE comparison products."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


DEFAULT_CANONICAL_ROOT = Path(
    "/scratch/kjhan/Hydro/HR5/FoFPSB/02_without_dc/Ver2/"
    "Derived_Sink_Hosts/canonical_v1"
)
EXPECTED_OUTPUT_COUNT = 121
EXPECTED_EVENT_COUNT = 576_278
EXPECTED_REQUEST_COUNT = 1_128_422


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _csv_data_row_count(path: Path) -> int:
    with path.open(newline="") as stream:
        return max(0, sum(1 for _ in stream) - 1)


def _counter_total(counter: dict[str, object]) -> int:
    return sum(int(value) for value in counter.values())


def validate(
    canonical_root: Path,
    figure_path: Path,
) -> dict[str, object]:
    manifest_path = canonical_root / "capture_hosts/hr5_capture_host_manifest.csv"
    table_path = (
        canonical_root
        / "capture_host_descendants/hr5_possible_binary_capture_host_descendants.csv"
    )
    descendant_summary_path = table_path.with_suffix(".json")
    evolution_path = table_path.parent / "hr5_fable_capture_host_evolution.csv"
    manifest = _read_csv(manifest_path)
    if len(manifest) != EXPECTED_OUTPUT_COUNT:
        raise ValueError(f"Expected {EXPECTED_OUTPUT_COUNT} host outputs")
    manifest_outputs = {int(row["output"]) for row in manifest}
    if len(manifest_outputs) != EXPECTED_OUTPUT_COUNT:
        raise ValueError("Capture-host manifest contains duplicate outputs")
    if any(row["capture_event_status"] != "complete" for row in manifest):
        raise ValueError("At least one partitioned capture-event table is incomplete")
    if any(row["host_catalogue_status"] != "complete" for row in manifest):
        raise ValueError("At least one capture-host catalogue is incomplete")
    event_count = sum(int(row["possible_binary_capture_count"]) for row in manifest)
    request_count = sum(int(row["requested_sink_count"]) for row in manifest)
    if event_count != EXPECTED_EVENT_COUNT or request_count != EXPECTED_REQUEST_COUNT:
        raise ValueError("Capture-host manifest totals do not match the source catalogue")

    filtered_output_count = 0
    filtered_hosted_sink_count = 0
    for row in manifest:
        if row["host_catalogue_source"] != "capture_filtered_host_catalogue":
            continue
        filtered_output_count += 1
        extraction_summary_path = Path(row["extraction_summary_path"])
        host_path = Path(row["host_catalogue_path"])
        if not extraction_summary_path.is_file() or not host_path.is_file():
            raise ValueError("A filtered host catalogue lacks its extraction record")
        extraction = json.loads(extraction_summary_path.read_text(encoding="utf-8"))
        if int(extraction["requested_sink_count"]) != int(row["requested_sink_count"]):
            raise ValueError("Filtered host request count differs from the manifest")
        selected = int(extraction["selected_sink_count"])
        if selected > int(extraction["requested_sink_count"]):
            raise ValueError("A filtered catalogue contains more sinks than requested")
        for field in (
            "duplicate_sink_count",
            "particle_count_mismatches",
            "host_sink_mass_mismatches",
            "metadata_sample_mismatches",
        ):
            if int(extraction[field]) != 0:
                raise ValueError(f"Filtered host extraction has nonzero {field}")
        if _csv_data_row_count(host_path) != selected:
            raise ValueError("Filtered host table row count differs from extraction record")
        filtered_hosted_sink_count += selected

    summary = json.loads(descendant_summary_path.read_text(encoding="utf-8"))
    if int(summary["possible_binary_capture_count"]) != EXPECTED_EVENT_COUNT:
        raise ValueError("Descendant summary has an incomplete event count")
    for field in ("host_track_status", "capture_host_time_order", "agn_pair_state"):
        if _counter_total(summary[field]) != EXPECTED_EVENT_COUNT:
            raise ValueError(f"Descendant summary has an incomplete {field} count")
    manifest_event_count = {
        int(row["output"]): int(row["possible_binary_capture_count"])
        for row in manifest
    }
    groups = summary["by_host_assignment_output"]
    if len(groups) != EXPECTED_OUTPUT_COUNT:
        raise ValueError("Descendant summary does not contain every host output")
    if {int(output) for output in groups} != manifest_outputs:
        raise ValueError("Descendant summary and host manifest cover different outputs")
    summary_group_event_count = {
        int(output): int(group["pair_count"]) for output, group in groups.items()
    }
    if summary_group_event_count != manifest_event_count:
        raise ValueError("Descendant summary event counts differ from the manifest")
    diagnostics = summary["fable_selection_analogue_assigned_companion_diagnostics"]
    if int(diagnostics["receiver_validation_row_count"]) != EXPECTED_EVENT_COUNT:
        raise ValueError("Assigned-companion diagnostics were not loaded completely")
    benchmark = summary["published_fable_benchmark"]
    if (
        int(benchmark["selected_numerical_bh_merger_count"]) != 10_716
        or int(benchmark["no_added_host_delay_count"]) != 513
    ):
        raise ValueError("FABLE benchmark values have changed unexpectedly")

    required_fields = {
        "selection_output",
        "primary_sink_id",
        "secondary_sink_id",
        "host_track_status",
        "capture_host_time_order",
        "fable_selection_analogue",
        "agn_pair_state",
        "simultaneous_assignment_multiplicity",
        "unique_assigned_companion",
    }
    descendant_count = 0
    diagnosed_count = 0
    table_event_count: Counter[int] = Counter()
    table_fable_count = 0
    with table_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not required_fields.issubset(reader.fieldnames):
            raise ValueError("Descendant table is missing required physical quantities")
        for row in reader:
            descendant_count += 1
            diagnosed_count += int(row["simultaneous_assignment_multiplicity"]) >= 0
            table_event_count[int(row["selection_output"])] += 1
            table_fable_count += int(row["fable_selection_analogue"])
    if descendant_count != EXPECTED_EVENT_COUNT:
        raise ValueError("Descendant table row count is incomplete")
    if diagnosed_count != EXPECTED_EVENT_COUNT:
        raise ValueError("Some possible binary captures lack companion diagnostics")
    if dict(table_event_count) != manifest_event_count:
        raise ValueError("Descendant table event counts differ from the manifest")
    summary_fable_count = int(
        summary["fable_selection_analogue_possible_binary_capture_count"]
    )
    if table_fable_count != summary_fable_count:
        raise ValueError("FABLE-selection count differs between table and summary")
    for field in (
        "fable_selection_analogue_host_track_status",
        "fable_selection_analogue_capture_host_time_order",
        "fable_selection_analogue_agn_pair_state",
    ):
        if _counter_total(summary[field]) != summary_fable_count:
            raise ValueError(f"Descendant summary has an incomplete {field} count")
    summary_group_fable_count = {
        int(output): int(group["fable_event_selection_analogue"]["possible_binary_capture_count"])
        for output, group in groups.items()
    }
    if sum(summary_group_fable_count.values()) != summary_fable_count:
        raise ValueError("Per-output FABLE-selection counts differ from the summary")

    evolution = _read_csv(evolution_path)
    if len(evolution) != EXPECTED_OUTPUT_COUNT:
        raise ValueError("Compact FABLE evolution table has incomplete output coverage")
    evolution_outputs = {int(row["host_assignment_output"]) for row in evolution}
    if evolution_outputs != manifest_outputs:
        raise ValueError("Compact FABLE evolution table covers different outputs")
    if sum(int(row["possible_binary_capture_count"]) for row in evolution) != event_count:
        raise ValueError("Compact evolution event count differs from the manifest")
    evolution_event_count = {
        int(row["host_assignment_output"]): int(row["possible_binary_capture_count"])
        for row in evolution
    }
    if evolution_event_count != manifest_event_count:
        raise ValueError("Compact evolution per-output counts differ from the manifest")
    if sum(int(row["fable_selection_analogue_count"]) for row in evolution) != table_fable_count:
        raise ValueError("Compact evolution FABLE count differs from the event table")
    evolution_fable_count = {
        int(row["host_assignment_output"]): int(row["fable_selection_analogue_count"])
        for row in evolution
    }
    if evolution_fable_count != summary_group_fable_count:
        raise ValueError("Compact evolution per-output FABLE counts differ from the summary")
    for row in evolution:
        selected_count = int(row["fable_selection_analogue_count"])
        lower_text = row["all_event_no_added_host_delay_lower_fraction"]
        upper_text = row["all_event_no_added_host_delay_upper_fraction"]
        if selected_count == 0:
            if lower_text or upper_text:
                raise ValueError("Zero-count evolution row contains timing fractions")
            continue
        if not lower_text or not upper_text:
            raise ValueError("Selected evolution events lack timing fractions")
        lower = float(lower_text)
        upper = float(upper_text)
        if not 0.0 <= lower <= upper <= 1.0:
            raise ValueError("Compact evolution table contains invalid timing bounds")
    if not figure_path.is_file() or figure_path.stat().st_size < 10_000:
        raise ValueError("FABLE comparison figure is absent or unexpectedly small")
    with figure_path.open("rb") as stream:
        if stream.read(5) != b"%PDF-":
            raise ValueError("FABLE comparison figure is not a PDF")
        stream.seek(max(0, figure_path.stat().st_size - 1024))
        if b"%%EOF" not in stream.read():
            raise ValueError("FABLE comparison PDF is incomplete")

    return {
        "validated": True,
        "host_output_count": len(manifest),
        "possible_binary_capture_count": event_count,
        "unique_output_sink_request_count": request_count,
        "filtered_host_output_count": filtered_output_count,
        "filtered_hosted_sink_count": filtered_hosted_sink_count,
        "descendant_table_row_count": descendant_count,
        "assigned_companion_diagnostic_count": diagnosed_count,
        "fable_selection_analogue_possible_binary_capture_count": summary_fable_count,
        "evolution_table_path": str(evolution_path),
        "evolution_table_row_count": len(evolution),
        "figure_path": str(figure_path),
        "figure_size_bytes": figure_path.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("results/hr5/hr5_fable_capture_host_comparison.pdf"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/hr5/hr5_fable_capture_host_validation.json"),
    )
    args = parser.parse_args()
    result = validate(args.canonical_root, args.figure)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

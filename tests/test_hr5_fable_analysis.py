from __future__ import annotations

import csv
import importlib.util
import json
import struct
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from fdm_smbh_delay.hr5 import MKAGN_DTYPE


def _load_analysis():
    repository = Path(__file__).resolve().parents[1]
    scripts = repository / "scripts"
    sys.path.insert(0, str(scripts))
    path = scripts / "analyze_hr5_capture_hosts.py"
    spec = importlib.util.spec_from_file_location("analyze_hr5_capture_hosts", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_matched_analysis():
    repository = Path(__file__).resolve().parents[1]
    scripts = repository / "scripts"
    sys.path.insert(0, str(scripts))
    path = scripts / "analyze_hr5_matched_pair_hosts.py"
    spec = importlib.util.spec_from_file_location("analyze_hr5_matched_pair_hosts", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_matched_plotting():
    repository = Path(__file__).resolve().parents[1]
    path = repository / "scripts" / "plot_hr5_matched_pair_hosts.py"
    spec = importlib.util.spec_from_file_location("plot_hr5_matched_pair_hosts", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_pair_host_builder():
    repository = Path(__file__).resolve().parents[1]
    path = repository / "scripts" / "build_hr5_agn_pair_host_dataset.py"
    spec = importlib.util.spec_from_file_location(
        "build_hr5_agn_pair_host_dataset", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_fable_validator():
    repository = Path(__file__).resolve().parents[1]
    path = repository / "scripts" / "validate_hr5_fable_outputs.py"
    spec = importlib.util.spec_from_file_location("validate_hr5_fable_outputs", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_fable_plotting():
    repository = Path(__file__).resolve().parents[1]
    path = repository / "scripts" / "plot_hr5_fable_comparison.py"
    spec = importlib.util.spec_from_file_location("plot_hr5_fable_comparison", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_timing_fraction_bounds_keep_interval_and_tracking_uncertainty() -> None:
    analysis = _load_analysis()
    counter = Counter(
        {
            "common_host_before_later_possible_binary_capture": 1,
            "common_descendant_before_possible_binary_capture": 1,
            "time_intervals_overlap": 3,
            "possible_binary_capture_before_common_descendant": 2,
            "host_time_unresolved": 2,
            "possible_binary_capture_before_last_resolved_distinct_hosts": 1,
        }
    )

    result = analysis._timing_fraction_bounds(counter, 10)

    assert result["certain_no_added_host_delay_count"] == 2
    assert result["resolved_timing_count"] == 8
    assert result["resolved_no_added_host_delay_lower_fraction"] == pytest.approx(0.25)
    assert result["resolved_no_added_host_delay_upper_fraction"] == pytest.approx(0.625)
    assert result["all_event_no_added_host_delay_lower_fraction"] == pytest.approx(0.2)
    assert result["all_event_no_added_host_delay_upper_fraction"] == pytest.approx(0.7)


def test_fable_plot_uses_all_selected_events_as_denominator() -> None:
    plotting = _load_fable_plotting()
    bounds = {
        "all_event_no_added_host_delay_lower_fraction": 0.2,
        "all_event_no_added_host_delay_upper_fraction": 0.7,
        "resolved_no_added_host_delay_lower_fraction": 0.25,
        "resolved_no_added_host_delay_upper_fraction": 0.625,
    }

    assert plotting._all_event_bounds(bounds) == (0.2, 0.7)


def test_capture_host_evolution_rows_preserve_counts_and_bounds() -> None:
    analysis = _load_analysis()
    by_output = {
        "20": {
            "pair_count": 5,
            "by_selection_output": {"20": {"redshift": 3.0}},
            "fable_event_selection_analogue": {
                "possible_binary_capture_count": 4,
                "agn_pair_state": {
                    "both SMBHs active": 1,
                    "no MkAGN measurement": 3,
                },
                "capture_host_time_order": {
                    "possible_binary_capture_before_common_descendant": 1
                },
                "timing_fraction_bounds": {
                    "certain_no_added_host_delay_count": 1,
                    "interval_overlap_count": 1,
                    "unresolved_host_time_count": 1,
                    "all_event_no_added_host_delay_lower_fraction": 0.25,
                    "all_event_no_added_host_delay_upper_fraction": 0.75,
                },
                "assigned_companion_diagnostics": {
                    "unique_assignment_count": 2,
                    "speed_below_point_mass_escape_count": 1,
                },
            },
        }
    }

    row = analysis._evolution_rows(by_output)[0]

    assert row["host_assignment_output"] == 20
    assert row["possible_binary_capture_count"] == 5
    assert row["fable_selection_analogue_count"] == 4
    assert row["both_smbhs_active_count"] == 1
    assert row["no_mkagn_measurement_count"] == 3
    assert row["all_event_no_added_host_delay_lower_fraction"] == 0.25
    assert row["all_event_no_added_host_delay_upper_fraction"] == 0.75


def test_capture_manifest_batches_do_not_split_an_output() -> None:
    analysis = _load_analysis()
    rows = [
        {"output": "20", "possible_binary_capture_count": "4"},
        {"output": "21", "possible_binary_capture_count": "5"},
        {"output": "22", "possible_binary_capture_count": "12"},
        {"output": "23", "possible_binary_capture_count": "3"},
    ]

    batches = analysis._manifest_batches(rows, 10)

    assert [[row["output"] for row in batch] for batch in batches] == [
        ["20", "21"],
        ["22"],
        ["23"],
    ]


def test_agn_pair_state_uses_saved_luminosities(tmp_path: Path) -> None:
    analysis = _load_analysis()
    path = tmp_path / "agn.00020.dat"
    records = np.zeros(2, dtype=MKAGN_DTYPE)
    records["sink_id"] = [1, 2]
    records["mass"] = [6.84e7, 6.84e7]
    records["Lbol"] = [1.0e44, 1.0e42]
    path.write_bytes(struct.pack("<ddi", 3.0, 1.0e5, 2) + records.tobytes())

    result = analysis._agn_pair_state(
        path,
        np.asarray([1, 2, 3]),
        np.asarray([2, 1, 2]),
        0.684,
        1.0e43,
    )

    assert result["state"].tolist() == [
        "one SMBH active",
        "one SMBH active",
        "SMBH missing from MkAGN snapshot",
    ]
    assert result["first_eddington_ratio"][0] > 0.0


def test_matched_delay_order_respects_both_interval_boundaries() -> None:
    analysis = _load_matched_analysis()
    rows = [
        {
            "match_id": "0",
            "selection_output": "20",
            "matched_pair_class": "dual",
            "host_track_status": "common_descendant",
            "common_descendant_delay_lower_gyr": 0.1,
            "common_descendant_delay_upper_gyr": 0.2,
            "fable_selection_analogue": "1",
        },
        {
            "match_id": "0",
            "selection_output": "20",
            "matched_pair_class": "offset",
            "host_track_status": "common_descendant",
            "common_descendant_delay_lower_gyr": 0.4,
            "common_descendant_delay_upper_gyr": 0.5,
            "fable_selection_analogue": "1",
        },
        {
            "match_id": "1",
            "selection_output": "20",
            "matched_pair_class": "dual",
            "host_track_status": "same_host_at_selection",
            "fable_selection_analogue": "0",
        },
        {
            "match_id": "1",
            "selection_output": "20",
            "matched_pair_class": "offset",
            "host_track_status": "common_descendant",
            "common_descendant_delay_lower_gyr": 0.0,
            "common_descendant_delay_upper_gyr": 0.1,
            "fable_selection_analogue": "1",
        },
    ]

    result = analysis._matched_delay_order(rows)

    assert result["matched_system_count"] == 2
    assert result["both_pairs_in_fable_selection_analogue_count"] == 1
    assert result["fable_selection_analogue"]["matched_system_count"] == 1
    assert result["fable_selection_analogue"]["same_host_at_selection"][
        "matched_system_count"
    ] == 1
    assert result["host_delay_order"] == {
        "dual_hosts_join_earlier": 1,
        "delay_intervals_overlap": 1,
    }


def test_join_bounds_use_resolved_part_of_right_censoring_interval() -> None:
    plotting = _load_matched_plotting()

    unresolved = {
        "host_track_status": "right_censored",
        "common_descendant_delay_lower_gyr": "0.2",
    }
    resolved_non_join = {
        "host_track_status": "right_censored",
        "common_descendant_delay_lower_gyr": "0.8",
    }

    assert plotting._join_bounds(unresolved, 0.5) == (0.0, 1.0)
    assert plotting._join_bounds(resolved_non_join, 0.5) == (0.0, 0.0)


def test_correlated_output_filter_keeps_the_later_snapshot() -> None:
    analysis = _load_matched_analysis()
    rows = [
        {"output_number": "88", "redshift": "2.8586251056769725"},
        {"output_number": "89", "redshift": "2.847835993381463"},
        {"output_number": "117", "redshift": "1.4988100669710724"},
    ]

    retained, excluded = analysis._remove_correlated_outputs(rows, 0.05)

    assert {int(row["output_number"]) for row in retained} == {89, 117}
    assert excluded == [88]


def test_pair_host_builder_records_system_multiplicity_and_relative_speed() -> None:
    builder = _load_pair_host_builder()
    pairs = {
        "id_1": np.asarray([1, 2, 10]),
        "id_2": np.asarray([2, 3, 11]),
        "is_dual": np.asarray([True, True, False]),
        "velocity_1_kms": np.asarray([[0.0, 0.0, 0.0]] * 3),
        "velocity_2_kms": np.asarray(
            [[3.0, 4.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 12.0]]
        ),
    }

    builder._add_system_information(pairs)

    assert pairs["pair_system_multiplicity"].tolist() == [3, 3, 2]
    assert pairs["dual_system_multiplicity"].tolist() == [3, 3, 0]
    assert pairs["relative_speed_kms"].tolist() == [5.0, 0.0, 12.0]


def test_fable_output_validator_cross_checks_manifest_table_and_figure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validator = _load_fable_validator()
    monkeypatch.setattr(validator, "EXPECTED_OUTPUT_COUNT", 2)
    monkeypatch.setattr(validator, "EXPECTED_EVENT_COUNT", 2)
    monkeypatch.setattr(validator, "EXPECTED_REQUEST_COUNT", 4)
    host_root = tmp_path / "capture_hosts"
    descendant_root = tmp_path / "capture_host_descendants"
    host_root.mkdir()
    descendant_root.mkdir()
    with (host_root / "hr5_capture_host_manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "output",
                "possible_binary_capture_count",
                "requested_sink_count",
                "capture_event_status",
                "host_catalogue_status",
                "host_catalogue_source",
                "extraction_summary_path",
                "host_catalogue_path",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "output": 20,
                "possible_binary_capture_count": 1,
                "requested_sink_count": 2,
                "capture_event_status": "complete",
                "host_catalogue_status": "complete",
                "host_catalogue_source": "full_canonical_host_catalogue",
                "extraction_summary_path": "",
                "host_catalogue_path": "",
            }
        )
        extraction_summary = tmp_path / "filtered_extraction.json"
        filtered_hosts = tmp_path / "filtered_hosts.csv"
        extraction_summary.write_text(
            json.dumps(
                {
                    "requested_sink_count": 2,
                    "selected_sink_count": 1,
                    "duplicate_sink_count": 0,
                    "particle_count_mismatches": 0,
                    "host_sink_mass_mismatches": 0,
                    "metadata_sample_mismatches": 0,
                }
            ),
            encoding="utf-8",
        )
        filtered_hosts.write_text("sink_id\n1\n", encoding="utf-8")
        writer.writerow(
            {
                "output": 21,
                "possible_binary_capture_count": 1,
                "requested_sink_count": 2,
                "capture_event_status": "complete",
                "host_catalogue_status": "complete",
                "host_catalogue_source": "capture_filtered_host_catalogue",
                "extraction_summary_path": extraction_summary,
                "host_catalogue_path": filtered_hosts,
            }
        )
    table = descendant_root / "hr5_possible_binary_capture_host_descendants.csv"
    fields = (
        "selection_output",
        "primary_sink_id",
        "secondary_sink_id",
        "host_track_status",
        "capture_host_time_order",
        "fable_selection_analogue",
        "agn_pair_state",
        "simultaneous_assignment_multiplicity",
        "unique_assigned_companion",
    )
    with table.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        rows = [dict.fromkeys(fields, "0") for _ in range(2)]
        for output, row in zip((20, 21), rows):
            row["selection_output"] = str(output)
            row["simultaneous_assignment_multiplicity"] = "1"
        rows[0]["fable_selection_analogue"] = "1"
        writer.writerows(rows)
    table.with_suffix(".json").write_text(
        json.dumps(
            {
                "possible_binary_capture_count": 2,
                "host_track_status": {"0": 2},
                "capture_host_time_order": {"0": 2},
                "agn_pair_state": {"0": 2},
                "fable_selection_analogue_possible_binary_capture_count": 1,
                "fable_selection_analogue_host_track_status": {"0": 1},
                "fable_selection_analogue_capture_host_time_order": {"0": 1},
                "fable_selection_analogue_agn_pair_state": {"0": 1},
                "by_host_assignment_output": {
                    "20": {
                        "pair_count": 1,
                        "fable_event_selection_analogue": {
                            "possible_binary_capture_count": 1
                        },
                    },
                    "21": {
                        "pair_count": 1,
                        "fable_event_selection_analogue": {
                            "possible_binary_capture_count": 0
                        },
                    },
                },
                "fable_selection_analogue_assigned_companion_diagnostics": {
                    "receiver_validation_row_count": 2
                },
                "published_fable_benchmark": {
                    "selected_numerical_bh_merger_count": 10716,
                    "no_added_host_delay_count": 513,
                },
            }
        ),
        encoding="utf-8",
    )
    evolution = descendant_root / "hr5_fable_capture_host_evolution.csv"
    with evolution.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "host_assignment_output",
                "possible_binary_capture_count",
                "fable_selection_analogue_count",
                "all_event_no_added_host_delay_lower_fraction",
                "all_event_no_added_host_delay_upper_fraction",
            ),
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "host_assignment_output": output,
                    "possible_binary_capture_count": 1,
                    "fable_selection_analogue_count": int(output == 20),
                    "all_event_no_added_host_delay_lower_fraction": (
                        0.2 if output == 20 else ""
                    ),
                    "all_event_no_added_host_delay_upper_fraction": (
                        0.7 if output == 20 else ""
                    ),
                }
                for output in (20, 21)
            ]
        )
    figure = tmp_path / "figure.pdf"
    figure.write_bytes(b"%PDF-1.4\n" + b"0" * 10_000 + b"\n%%EOF\n")

    result = validator.validate(tmp_path, figure)

    assert result["validated"] is True
    assert result["descendant_table_row_count"] == 2
    assert result["assigned_companion_diagnostic_count"] == 2
    assert result["evolution_table_row_count"] == 2
    assert result["filtered_host_output_count"] == 1
    assert result["filtered_hosted_sink_count"] == 1

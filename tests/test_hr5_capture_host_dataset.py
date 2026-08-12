from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest


def _load_builder():
    repository = Path(__file__).resolve().parents[1]
    path = repository / "scripts" / "build_hr5_capture_host_dataset.py"
    spec = importlib.util.spec_from_file_location("build_hr5_capture_host_dataset", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_reporter():
    repository = Path(__file__).resolve().parents[1]
    path = repository / "scripts" / "report_hr5_capture_host_status.py"
    spec = importlib.util.spec_from_file_location("report_hr5_capture_host_status", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, fields: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)


def test_prepare_capture_targets_groups_unique_pair_members(tmp_path: Path) -> None:
    builder = _load_builder()
    capture = tmp_path / "captures.csv"
    _write_csv(
        capture,
        ["sink_id", "receiver_id", "last_resolved_output"],
        [[2, 8, 20], [3, 8, 20], [4, 9, 22]],
    )
    manifest = tmp_path / "manifest.csv"
    _write_csv(
        manifest,
        [
            "output",
            "redshift",
            "host_extraction_ready",
            "galfind_path",
            "galaxy_list_path",
            "galaxy_count",
            "galaxy_link_path",
            "sink_host_catalog_path",
        ],
        [
            [20, 3.0, True, "/canonical/FoF.00020/data", "/canonical/FoF.00020/list", 7, "/tree/20", tmp_path / "full20.csv"],
            [21, 2.0, True, "/canonical/FoF.00021/data", "/canonical/FoF.00021/list", 9, "/tree/21", tmp_path / "full21.csv"],
        ],
    )

    rows = builder.prepare_capture_targets(capture, manifest, tmp_path / "derived")

    assert [row["output"] for row in rows] == [20, 21]
    assert rows[0]["possible_binary_capture_count"] == 2
    assert rows[0]["requested_sink_count"] == 3
    assert Path(rows[0]["sink_id_path"]).read_text() == "2\n3\n8\n"
    assert rows[1]["last_resolved_outputs"] == "22"
    assert rows[1]["maximum_output_lag"] == 1
    assert ".mine" not in str(rows[0]["galfind_path"])

    builder.partition_capture_events(capture, rows)
    with Path(rows[1]["capture_event_path"]).open(newline="") as stream:
        partitioned = list(csv.DictReader(stream))
    assert len(partitioned) == 1
    assert partitioned[0]["host_assignment_output"] == "21"
    assert partitioned[0]["last_resolved_output"] == "22"


def test_extraction_summary_rejects_internal_mismatches() -> None:
    builder = _load_builder()
    row = {"galaxy_count": 9, "requested_sink_count": 4}
    summary = {
        "psb_galaxy_count": 9,
        "requested_sink_count": 4,
        "selected_sink_count": 3,
        "duplicate_sink_count": 0,
        "particle_count_mismatches": 0,
        "host_sink_mass_mismatches": 0,
        "metadata_sample_mismatches": 0,
    }

    builder._validate_extraction_summary(row, summary, "00020")
    summary["host_sink_mass_mismatches"] = 1

    with pytest.raises(ValueError, match="host_sink_mass_mismatches"):
        builder._validate_extraction_summary(row, summary, "00020")


def test_status_report_counts_files_instead_of_stale_manifest_flags(
    tmp_path: Path,
) -> None:
    reporter = _load_reporter()
    canonical = tmp_path / "canonical"
    capture_root = canonical / "capture_hosts"
    capture_root.mkdir(parents=True)
    host = capture_root / "host.csv"
    event = capture_root / "event.csv"
    host.write_text("sink_id\n1\n")
    event.write_text("sink_id\n1\n")
    _write_csv(
        capture_root / "hr5_capture_host_manifest.csv",
        [
            "host_catalogue_path",
            "capture_event_path",
            "possible_binary_capture_count",
            "requested_sink_count",
        ],
        [[host, event, 3, 5], [capture_root / "missing.csv", event, 4, 6]],
    )

    result = reporter.report(canonical, tmp_path / "repository")

    assert result["manifest_output_count"] == 2
    assert result["capture_event_output_count"] == 2
    assert result["host_catalogue_output_count"] == 1
    assert result["host_complete_possible_binary_capture_count"] == 3
    assert result["host_complete_requested_sink_count"] == 5
    assert result["validated"] is False

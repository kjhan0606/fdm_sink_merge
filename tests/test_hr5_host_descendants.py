from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from fdm_smbh_delay.hr5_galaxies import GALAXY_LINK_DTYPE, GALAXY_LINK_HEADER


def _load_analysis():
    repository = Path(__file__).resolve().parents[1]
    path = repository / "scripts" / "analyze_hr5_host_descendants.py"
    spec = importlib.util.spec_from_file_location("analyze_hr5_host_descendants", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_links(
    path: Path,
    redshift: float,
    rows: list[tuple[int, int, int, int]],
) -> None:
    records = np.zeros(len(rows), dtype=GALAXY_LINK_DTYPE)
    for index, (now_gid, descendant_array_id, next_gid, link_flag) in enumerate(rows):
        records[index]["array_id"] = index
        records[index]["now_gid"] = now_gid
        records[index]["next_gid"] = next_gid
        records[index]["descendant_array_id"] = descendant_array_id
        records[index]["link_flag"] = link_flag
    path.write_bytes(GALAXY_LINK_HEADER.pack(redshift, len(records)) + records.tobytes())


def test_trace_pairs_brackets_common_descendant_and_capture_order(tmp_path: Path) -> None:
    analysis = _load_analysis()
    paths = [tmp_path / f"GalaxyLinkedList.{output:05d}" for output in (20, 21, 22)]
    _write_links(paths[0], 3.0, [(0, 0, 0, 4), (1, 1, 1, 4)])
    _write_links(paths[1], 2.0, [(2, 0, 0, 4), (3, 1, 1, 4)])
    _write_links(paths[2], 1.0, [(5, -999, 5, 4), (5, -999, 5, 4)])
    rows = [
        {
            "selection_output": "20",
            "primary_sink_id": "10",
            "secondary_sink_id": "11",
            "primary_galaxy_gid": "0",
            "secondary_galaxy_gid": "1",
            "fable_selection_analogue": "1",
            "host_relation": "distinct PSB galaxies in one FoF halo",
        }
    ]
    capture = {
        (20, 10, 11): {
            "assigned_capture_output": "21",
            "capture_delay_lower_gyr": "0.01",
            "capture_delay_upper_gyr": "0.02",
            "pair_class": "dual",
        }
    }

    result = analysis.trace_pairs(
        rows,
        [20, 21, 22],
        paths,
        np.array([3.0, 2.0, 1.0]),
        capture,
    )[0]

    assert result["host_track_status"] == "common_descendant"
    assert result["common_descendant_output"] == 22
    assert result["common_descendant_gid"] == 5
    assert result["common_descendant_delay_lower_gyr"] > 0.0
    assert result["common_descendant_delay_upper_gyr"] > result[
        "common_descendant_delay_lower_gyr"
    ]
    assert (
        result["capture_host_time_order"]
        == "possible_binary_capture_before_common_descendant"
    )


def test_time_order_keeps_touching_intervals_unresolved() -> None:
    analysis = _load_analysis()

    assert (
        analysis._time_order("common_descendant", 0.1, 0.2, 0.3, 0.4)
        == "common_descendant_before_possible_binary_capture"
    )
    assert (
        analysis._time_order("common_descendant", 0.3, 0.4, 0.1, 0.2)
        == "possible_binary_capture_before_common_descendant"
    )
    assert (
        analysis._time_order("common_descendant", 0.2, 0.3, 0.1, 0.2)
        == "time_intervals_overlap"
    )
    assert (
        analysis._time_order("right_censored", 0.5, np.nan, 0.1, 0.2)
        == "possible_binary_capture_before_last_resolved_distinct_hosts"
    )

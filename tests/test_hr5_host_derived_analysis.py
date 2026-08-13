from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_script(name: str):
    repository = Path(__file__).resolve().parents[1]
    scripts = repository / "scripts"
    sys.path.insert(0, str(scripts))
    path = scripts / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_pair_key_is_independent_of_primary_label() -> None:
    analysis = _load_script("analyze_hr5_dual_agn_host_demographics")

    assert analysis._pair_key(7, 3) == (3, 7)
    assert analysis._pair_key(3, 7) == (3, 7)


def test_pair_fraction_jackknife_uses_pair_midpoints() -> None:
    analysis = _load_script("analyze_hr5_dual_agn_host_demographics")
    denominator = np.asarray([True, True, True, True])
    numerator = np.asarray([True, False, True, False])
    region = np.asarray([0, 0, 1, 1])

    fraction, error = analysis._jackknife_pair_ratio(
        numerator, denominator, region, 2
    )

    assert fraction == pytest.approx(0.5)
    assert error == pytest.approx(0.0)


def test_pair_fraction_rejects_numerator_outside_denominator() -> None:
    analysis = _load_script("analyze_hr5_dual_agn_host_demographics")

    with pytest.raises(ValueError, match="subset"):
        analysis._jackknife_pair_ratio(
            np.asarray([True, False]),
            np.asarray([False, True]),
            np.asarray([0, 1]),
            2,
        )


def test_companion_sensitivity_applies_a_nested_selection(tmp_path: Path) -> None:
    analysis = _load_script("analyze_hr5_companion_sensitivity")
    path = tmp_path / "events.csv"
    fields = [
        "fable_selection_analogue",
        "unique_assigned_companion",
        "last_resolved_speed_below_point_mass_escape",
        "capture_host_time_order",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            [
                {
                    fields[0]: "1",
                    fields[1]: "1",
                    fields[2]: "1",
                    fields[3]: "common_host_before_later_possible_binary_capture",
                },
                {
                    fields[0]: "1",
                    fields[1]: "1",
                    fields[2]: "0",
                    fields[3]: "time_intervals_overlap",
                },
                {
                    fields[0]: "1",
                    fields[1]: "0",
                    fields[2]: "1",
                    fields[3]: "possible_binary_capture_before_common_descendant",
                },
                {
                    fields[0]: "0",
                    fields[1]: "1",
                    fields[2]: "1",
                    fields[3]: "common_host_before_later_possible_binary_capture",
                },
            ]
        )

    rows = analysis._summary_rows(path)

    assert [row["event_count"] for row in rows] == [3, 2, 1]
    assert rows[0]["no_added_host_delay_lower_fraction"] == pytest.approx(1.0 / 3.0)
    assert rows[0]["no_added_host_delay_upper_fraction"] == pytest.approx(2.0 / 3.0)
    assert rows[2]["no_added_host_delay_lower_fraction"] == pytest.approx(1.0)

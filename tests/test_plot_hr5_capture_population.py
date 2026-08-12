from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "plot_hr5_capture_population.py"
)
SPEC = importlib.util.spec_from_file_location("plot_hr5_capture_population", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT)


def test_read_events_uses_directly_recorded_removed_mass(tmp_path: Path) -> None:
    path = tmp_path / "captures.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "assigned_capture_output",
                "assigned_capture_redshift",
                "mass_ratio_last_resolved",
                "minor_mass_last_resolved_msun",
                "chirp_mass_last_resolved_msun",
            )
        )
        writer.writerow((17, 2.5, 0.25, 3.0e5, 8.0e5))

    output, redshift, mass_ratio, removed_mass = SCRIPT._read_events(path)

    assert np.array_equal(output, np.array([17]))
    assert np.array_equal(redshift, np.array([2.5]))
    assert np.array_equal(mass_ratio, np.array([0.25]))
    assert np.array_equal(removed_mass, np.array([3.0e5]))

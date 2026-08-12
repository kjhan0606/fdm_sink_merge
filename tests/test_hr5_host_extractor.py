from __future__ import annotations

import csv
import json
import shutil
import struct
import subprocess
from pathlib import Path

import pytest


METADATA = struct.Struct("<6i11d")
SINK = struct.Struct("<20di4x")


def _metadata(count: tuple[int, int, int, int, int, int], values: list[float]) -> bytes:
    return METADATA.pack(*count, *values)


def _sink(sink_id: int, mass: float, position: tuple[float, float, float]) -> bytes:
    values = [0.0] * 20
    values[0:3] = position
    values[3:6] = (10.0, 20.0, 30.0)
    values[6] = mass
    return SINK.pack(*values, sink_id)


def test_hr5_host_extractor_reads_direct_and_background_membership(tmp_path: Path) -> None:
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("A C compiler is required for the legacy HR5 extractor test")

    repository = Path(__file__).resolve().parents[1]
    executable = tmp_path / "extract_hr5_sink_hosts"
    subprocess.run(
        [
            compiler,
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-o",
            str(executable),
            str(repository / "tools" / "extract_hr5_sink_hosts.c"),
            "-lm",
        ],
        check=True,
    )

    halo = _metadata((2, 1, 1, 2, 1, 5), [30.0] + [0.0] * 10)
    host_values = [80.0, 20.0, 10.0, 30.0, 20.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    first_host = _metadata((1, 1, 2, 1, 5, 0), host_values)
    empty_host = _metadata((0, 0, 0, 0, 0, 0), [0.0] * 11)

    catalog_list = tmp_path / "GALCATALOG.LIST.00007"
    catalog_list.write_bytes(halo + first_host + empty_host)

    galfind = tmp_path / "GALFIND.DATA.00007"
    galfind.write_bytes(
        halo
        + first_host
        + bytes(128)
        + bytes(128)
        + _sink(42, 10.0, (1.1, 2.2, 3.3))
        + _sink(91, 20.0, (4.4, 5.5, 6.6))
        + bytes(128)
        + empty_host
    )

    background_values = [5.0, 0.0, 0.0, 5.0] + [0.0] * 7
    background = tmp_path / "background_ptl.00007"
    background.write_bytes(
        _metadata((0, 0, 1, 0, 1, 0), background_values)
        + _sink(105, 5.0, (7.7, 8.8, 9.9))
    )

    output = tmp_path / "hosts.csv"
    completed = subprocess.run(
        [
            str(executable),
            "--data",
            str(galfind),
            "--list",
            str(catalog_list),
            "--background",
            str(background),
            "--output",
            str(output),
            "--output-number",
            "7",
            "--redshift",
            "3.25",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(completed.stdout)
    assert summary == {
        "output": 7,
        "redshift": 3.25,
        "fof_halo_count": 1,
        "psb_galaxy_count": 2,
        "hosted_sink_count": 2,
        "background_sink_count": 1,
        "requested_sink_count": 0,
        "selected_sink_count": 3,
        "duplicate_sink_count": 0,
        "particle_count_mismatches": 0,
        "host_sink_mass_mismatches": 0,
        "metadata_sample_mismatches": 0,
        "maximum_sink_id": 105,
    }

    with output.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["sink_id"]) for row in rows] == [42, 91, 105]
    assert [int(row["galaxy_gid"]) for row in rows] == [0, 0, -1]
    assert [int(row["psb_index"]) for row in rows] == [0, 0, -1]
    assert [int(row["background"]) for row in rows] == [0, 0, 1]
    assert float(rows[0]["host_stellar_mass_msun_h"]) == pytest.approx(20.0)
    assert float(rows[1]["sink_mass_msun_h"]) == pytest.approx(20.0)
    assert float(rows[2]["sink_x_cmpc_h"]) == pytest.approx(7.7)
    assert int(rows[0]["host_dm_count"]) == 1
    assert int(rows[0]["host_gas_count"]) == 1
    assert int(rows[0]["host_stellar_count"]) == 1
    assert int(rows[0]["host_particle_count"]) == 5

    selected_ids = tmp_path / "selected_sink_ids.txt"
    selected_ids.write_text("91\n105\n")
    selected_output = tmp_path / "selected_hosts.csv"
    selected_output.with_suffix(".csv.tmp").write_text("interrupted extraction\n")
    selected = subprocess.run(
        [
            str(executable),
            "--data",
            str(galfind),
            "--list",
            str(catalog_list),
            "--background",
            str(background),
            "--sink-ids",
            str(selected_ids),
            "--output",
            str(selected_output),
            "--output-number",
            "7",
            "--redshift",
            "3.25",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    selected_summary = json.loads(selected.stdout)
    assert selected_summary["requested_sink_count"] == 2
    assert selected_summary["selected_sink_count"] == 2
    with selected_output.open(newline="") as stream:
        selected_rows = list(csv.DictReader(stream))
    assert [int(row["sink_id"]) for row in selected_rows] == [91, 105]
    assert not selected_output.with_suffix(".csv.tmp").exists()

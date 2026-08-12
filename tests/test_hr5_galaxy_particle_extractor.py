from __future__ import annotations

import csv
import shutil
import struct
import subprocess
from pathlib import Path

import pytest


METADATA = struct.Struct("<6i11d")


def _metadata(count: tuple[int, int, int, int, int, int], values: list[float]) -> bytes:
    return METADATA.pack(*count, *values)


def _gas() -> bytes:
    record = bytearray(128)
    struct.pack_into("<4d", record, 0, 1.0, 2.0, 3.0, 0.01)
    struct.pack_into("<d", record, 48, 42.0)
    struct.pack_into("<f", record, 56, 1.5e6)
    struct.pack_into("<f", record, 60, 0.012)
    struct.pack_into("<i", record, 76, 15)
    struct.pack_into("<f", record, 80, 12.5)
    return bytes(record)


def _star() -> bytes:
    record = bytearray(128)
    struct.pack_into("<3d", record, 0, 4.0, 5.0, 6.0)
    struct.pack_into("<d", record, 48, 20.0)
    struct.pack_into("<d", record, 64, -1.25)
    struct.pack_into("<d", record, 72, 0.025)
    struct.pack_into("<d", record, 80, 7.5e-14)
    struct.pack_into("<i", record, 120, 14)
    return bytes(record)


def test_selected_galaxy_particle_extractor_reads_gas_and_stars(tmp_path: Path) -> None:
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("A C compiler is required for the legacy HR5 extractor test")
    repository = Path(__file__).resolve().parents[1]
    executable = tmp_path / "extract_hr5_galaxy_particles"
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
            str(repository / "tools" / "extract_hr5_galaxy_particles.c"),
            "-lm",
        ],
        check=True,
    )

    halo = _metadata((2, 0, 0, 0, 0, 0), [0.0] * 11)
    empty = _metadata((0, 0, 0, 0, 0, 0), [0.0] * 11)
    values = [33.5, 1.0, 12.5, 0.0, 20.0] + [0.0] * 6
    selected = _metadata((1, 1, 0, 1, 3, 0), values)
    catalog_list = tmp_path / "GALCATALOG.LIST.00001"
    catalog_list.write_bytes(halo + empty + selected)
    galfind = tmp_path / "GALFIND.DATA.00001"
    galfind.write_bytes(
        halo + empty + selected + bytes(128) + _gas() + _star()
    )
    galaxy_ids = tmp_path / "galaxy_ids.txt"
    galaxy_ids.write_text("1\n")
    output = tmp_path / "particles.csv"
    completed = subprocess.run(
        [
            str(executable),
            "--data",
            str(galfind),
            "--list",
            str(catalog_list),
            "--galaxy-ids",
            str(galaxy_ids),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Extracted 1 galaxies, 1 gas cells, and 1 star particles" in completed.stderr
    assert "Gas-mass mismatches: 0; stellar-mass mismatches: 0" in completed.stderr
    with output.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["particle_type"] for row in rows] == ["gas", "star"]
    assert [int(row["galaxy_gid"]) for row in rows] == [1, 1]
    assert float(rows[0]["mass_msun_h"]) == pytest.approx(12.5)
    assert float(rows[0]["metallicity"]) == pytest.approx(0.012)
    assert float(rows[0]["density_code"]) == pytest.approx(42.0)
    assert int(rows[0]["level"]) == 15
    assert float(rows[1]["mass_msun_h"]) == pytest.approx(20.0)
    assert float(rows[1]["formation_time"]) == pytest.approx(-1.25)
    assert float(rows[1]["metallicity"]) == pytest.approx(0.025)
    assert int(rows[1]["level"]) == 14

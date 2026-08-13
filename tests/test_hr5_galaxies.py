from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from fdm_smbh_delay.hr5_galaxies import (
    GALAXY_LINK_DTYPE,
    GALAXY_LINK_HEADER,
    map_galaxy_descendants,
    open_galaxy_links,
)


def _records(rows: list[tuple[int, ...]]) -> np.ndarray:
    result = np.zeros(len(rows), dtype=GALAXY_LINK_DTYPE)
    for index, row in enumerate(rows):
        (
            result[index]["array_id"],
            result[index]["now_gid"],
            result[index]["descendant_array_id"],
            result[index]["link_flag"],
        ) = row
    return result


def test_open_galaxy_links_validates_native_layout(tmp_path: Path) -> None:
    path = tmp_path / "GalaxyLinkedList.00020"
    records = _records([(0, 0, 2, 4), (1, 1, 3, 4)])
    path.write_bytes(GALAXY_LINK_HEADER.pack(13.5, 2) + records.tobytes())

    header, mapped = open_galaxy_links(path)

    assert header.redshift == pytest.approx(13.5)
    assert header.record_count == 2
    assert mapped.dtype == GALAXY_LINK_DTYPE
    assert mapped[1]["now_gid"] == 1


def test_map_galaxy_descendants_handles_multiple_tracers_and_major_branch() -> None:
    current = _records(
        [
            (0, 0, 1, 4),
            (1, 1, 2, 2),
            (2, 1, 3, 4),
            (3, 2, -999, 4),
        ]
    )
    following = _records(
        [
            (0, 0, -999, 4),
            (1, 7, -999, 4),
            (2, 8, -999, 2),
            (3, 9, -999, 4),
        ]
    )

    descendant, status = map_galaxy_descendants(
        current, following, np.array([0, 1, 2, 3])
    )

    assert descendant.tolist() == [7, 9, -1, -1]
    assert status.tolist() == [0, 1, 3, 2]


def test_map_galaxy_descendants_prefers_dominant_progenitor_record() -> None:
    current = _records([(0, 0, 0, 4), (1, 0, 1, 4)])
    current[1]["status_flag"] = 8
    following = _records([(0, 4, -999, 4), (1, 5, -999, 4)])

    descendant, status = map_galaxy_descendants(
        current, following, np.array([0])
    )

    assert descendant.tolist() == [5]
    assert status.tolist() == [1]


def test_open_galaxy_links_rejects_truncated_file(tmp_path: Path) -> None:
    path = tmp_path / "GalaxyLinkedList.00020"
    path.write_bytes(struct.pack("<fq", 13.5, 2) + bytes(GALAXY_LINK_DTYPE.itemsize))
    with pytest.raises(ValueError, match="Unexpected HR5 galaxy-link size"):
        open_galaxy_links(path)

"""Readers and descendant tracking for the native HR5 galaxy links."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


GALAXY_LINK_HEADER = struct.Struct("<fq")
GALAXY_LINK_DTYPE = np.dtype(
    [
        ("mbp_id", "<i8"),
        ("array_id", "<i8"),
        ("next_gid", "<i8"),
        ("now_gid", "<i8"),
        ("progenitor_array_id", "<i8"),
        ("descendant_array_id", "<i8"),
        ("mass_msun_h", "<f4"),
        ("status_flag", "<i2"),
        ("link_flag", "<i2"),
    ]
)
MAJOR_BRANCH_FLAG = 0x04
MAJOR_PROGENITOR_FLAG = 0x08
NO_LINK = -999


@dataclass(frozen=True)
class GalaxyLinkHeader:
    """Header of one ``GalaxyLinkedList.NNNNN`` file."""

    redshift: float
    record_count: int


def read_galaxy_link_header(path: Path) -> GalaxyLinkHeader:
    """Read and validate the header and size of an HR5 galaxy-link file."""

    with path.open("rb") as stream:
        payload = stream.read(GALAXY_LINK_HEADER.size)
    if len(payload) != GALAXY_LINK_HEADER.size:
        raise ValueError(f"Incomplete HR5 galaxy-link header in {path}")
    redshift, count = GALAXY_LINK_HEADER.unpack(payload)
    expected_size = GALAXY_LINK_HEADER.size + count * GALAXY_LINK_DTYPE.itemsize
    if count < 0 or path.stat().st_size != expected_size:
        raise ValueError(
            f"Unexpected HR5 galaxy-link size for {path}. Expected "
            f"{expected_size} bytes and found {path.stat().st_size} bytes."
        )
    return GalaxyLinkHeader(float(redshift), int(count))


def open_galaxy_links(path: Path) -> tuple[GalaxyLinkHeader, np.memmap]:
    """Memory-map one native HR5 galaxy-link catalogue without copying it."""

    header = read_galaxy_link_header(path)
    records = np.memmap(
        path,
        dtype=GALAXY_LINK_DTYPE,
        mode="r",
        offset=GALAXY_LINK_HEADER.size,
        shape=(header.record_count,),
    )
    return header, records


def _lower_bound_now_gid(records: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Find lower bounds without materializing the strided ``now_gid`` field."""

    query = np.asarray(values, dtype=np.int64)
    lower = np.zeros(query.size, dtype=np.int64)
    upper = np.full(query.size, records.size, dtype=np.int64)
    while np.any(lower < upper):
        middle = lower + (upper - lower) // 2
        active = lower < upper
        middle_value = np.empty(query.size, dtype=np.int64)
        middle_value[active] = records["now_gid"][middle[active]]
        move_right = active & (middle_value < query)
        lower[move_right] = middle[move_right] + 1
        upper[active & ~move_right] = middle[active & ~move_right]
    return lower


def map_galaxy_descendants(
    current: np.ndarray,
    following: np.ndarray,
    galaxy_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map selected HR5 galaxies to the following available output.

    The native link catalogue can store several most-bound-particle records for
    one galaxy.  All valid records normally enter the same descendant galaxy.
    When they do not, the record marked as the dominant progenitor resolves the
    descendant.  A unique branch carrying the major-branch flag provides a
    fallback. Remaining splits are reported as ambiguous rather than being
    assigned silently.

    Status codes are ``0`` for an exact unique descendant, ``1`` for a unique
    major-branch descendant, ``2`` when the galaxy is absent, ``3`` when no
    valid descendant survives, and ``4`` for an ambiguous split.
    """

    requested = np.asarray(galaxy_ids, dtype=np.int64)
    descendants = np.full(requested.shape, -1, dtype=np.int64)
    status = np.full(requested.shape, 2, dtype=np.int8)
    if requested.size == 0:
        return descendants, status

    flat_requested = requested.ravel()
    left_edge = _lower_bound_now_gid(current, flat_requested)
    flat_descendants = descendants.ravel()
    flat_status = status.ravel()
    for index, galaxy_id in enumerate(flat_requested):
        left = int(left_edge[index])
        if left >= current.size or current["now_gid"][left] != galaxy_id:
            continue
        right = left + 1
        while right < current.size and current["now_gid"][right] == galaxy_id:
            right += 1

        rows = current[left:right]
        next_index = np.asarray(rows["descendant_array_id"], dtype=np.int64)
        valid = (next_index >= 0) & (next_index < following.size)
        if not np.any(valid):
            flat_status[index] = 3
            continue
        next_gid = np.asarray(following["now_gid"][next_index[valid]], dtype=np.int64)
        unique = np.unique(next_gid)
        if unique.size == 1:
            flat_descendants[index] = unique[0]
            flat_status[index] = 0
            continue

        major_progenitor = valid & (
            (rows["status_flag"] & MAJOR_PROGENITOR_FLAG) != 0
        )
        if np.count_nonzero(major_progenitor) == 1:
            major_index = int(rows["descendant_array_id"][major_progenitor][0])
            flat_descendants[index] = int(following["now_gid"][major_index])
            flat_status[index] = 1
            continue

        major = valid & ((rows["link_flag"] & MAJOR_BRANCH_FLAG) != 0)
        if np.any(major):
            major_gid = np.unique(
                following["now_gid"][
                    np.asarray(rows["descendant_array_id"][major], dtype=np.int64)
                ]
            )
            if major_gid.size == 1:
                flat_descendants[index] = major_gid[0]
                flat_status[index] = 1
                continue
        flat_status[index] = 4
    return descendants, status

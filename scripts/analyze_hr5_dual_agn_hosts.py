#!/usr/bin/env python3
"""Classify spatially selected active SMBH pairs by their direct HR5 hosts."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

from fdm_smbh_delay.hr5 import (
    HOST_RELATION_LABELS,
    classify_sink_pair_hosts,
    find_dual_agn_pairs,
    lookup_sink_hosts,
    read_mkagn_snapshot,
    read_sink_host_catalog,
)


FABLE_DIMENSIONLESS_HUBBLE = 0.679
FABLE_INITIAL_BARYON_MASS_MSUN_H = 6.4e6
FABLE_MINIMUM_HOST_STELLAR_MASS_MSUN = (
    100.0 * FABLE_INITIAL_BARYON_MASS_MSUN_H / FABLE_DIMENSIONLESS_HUBBLE
)


def _output_number(path: Path) -> int | None:
    match = re.search(r"\.(\d{5})(?:\.[^.]+)?$", path.name)
    return int(match.group(1)) if match else None


def _host_value(records: np.ndarray, row: np.ndarray, field: str, fill: float) -> np.ndarray:
    value = np.full(row.size, fill, dtype=np.float64)
    found = row >= 0
    value[found] = records[field][row[found]]
    return value


def _write_pair_table(
    path: Path,
    pairs: dict[str, np.ndarray],
    hosts: np.ndarray,
    relation: np.ndarray,
    first_row: np.ndarray,
    second_row: np.ndarray,
    dimensionless_hubble: float,
) -> None:
    first_gid = _host_value(hosts, first_row, "galaxy_gid", -1).astype(np.int64)
    second_gid = _host_value(hosts, second_row, "galaxy_gid", -1).astype(np.int64)
    first_fof = _host_value(hosts, first_row, "fof_index", -1).astype(np.int64)
    second_fof = _host_value(hosts, second_row, "fof_index", -1).astype(np.int64)
    first_stellar_mass = _host_value(
        hosts, first_row, "host_stellar_mass_msun_h", np.nan
    ) / dimensionless_hubble
    second_stellar_mass = _host_value(
        hosts, second_row, "host_stellar_mass_msun_h", np.nan
    ) / dimensionless_hubble
    first_total_mass = _host_value(
        hosts, first_row, "host_total_mass_msun_h", np.nan
    ) / dimensionless_hubble
    second_total_mass = _host_value(
        hosts, second_row, "host_total_mass_msun_h", np.nan
    ) / dimensionless_hubble
    first_stellar_count = _host_value(
        hosts, first_row, "host_stellar_count", -1
    ).astype(np.int64)
    second_stellar_count = _host_value(
        hosts, second_row, "host_stellar_count", -1
    ).astype(np.int64)
    hr5_100_star_particle_selection = (
        (pairs["mass_1_msun"] >= 1.0e6)
        & (pairs["mass_2_msun"] >= 1.0e6)
        & (first_stellar_count >= 100)
        & (second_stellar_count >= 100)
    )
    fable_selection_analogue = (
        (pairs["mass_1_msun"] >= 1.0e6)
        & (pairs["mass_2_msun"] >= 1.0e6)
        & (first_stellar_mass >= FABLE_MINIMUM_HOST_STELLAR_MASS_MSUN)
        & (second_stellar_mass >= FABLE_MINIMUM_HOST_STELLAR_MASS_MSUN)
    )

    fields = (
        "primary_sink_id",
        "secondary_sink_id",
        "separation_pkpc",
        "primary_smbh_mass_msun",
        "secondary_smbh_mass_msun",
        "primary_lbol_erg_s",
        "secondary_lbol_erg_s",
        "host_relation",
        "primary_galaxy_gid",
        "secondary_galaxy_gid",
        "primary_fof_index",
        "secondary_fof_index",
        "primary_host_stellar_mass_msun",
        "secondary_host_stellar_mass_msun",
        "primary_host_total_mass_msun",
        "secondary_host_total_mass_msun",
        "primary_host_stellar_particle_count",
        "secondary_host_stellar_particle_count",
        "hr5_100_star_particle_selection",
        "fable_selection_analogue",
    )
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        for index in range(relation.size):
            writer.writerow(
                (
                    int(pairs["id_1"][index]),
                    int(pairs["id_2"][index]),
                    float(pairs["separation_pkpc"][index]),
                    float(pairs["mass_1_msun"][index]),
                    float(pairs["mass_2_msun"][index]),
                    float(pairs["luminosity_1_erg_s"][index]),
                    float(pairs["luminosity_2_erg_s"][index]),
                    HOST_RELATION_LABELS[relation[index]],
                    int(first_gid[index]),
                    int(second_gid[index]),
                    int(first_fof[index]),
                    int(second_fof[index]),
                    float(first_stellar_mass[index]),
                    float(second_stellar_mass[index]),
                    float(first_total_mass[index]),
                    float(second_total_mass[index]),
                    int(first_stellar_count[index]),
                    int(second_stellar_count[index]),
                    int(hr5_100_star_particle_selection[index]),
                    int(fable_selection_analogue[index]),
                )
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("host_catalog", type=Path)
    parser.add_argument("agn_snapshot", type=Path)
    parser.add_argument("--output-directory", type=Path, default=Path("results"))
    parser.add_argument("--dimensionless-hubble", type=float, default=0.684)
    parser.add_argument("--box-size-cmpc-h", type=float, default=717.229040)
    parser.add_argument("--luminosity-threshold-erg-s", type=float, default=1.0e43)
    parser.add_argument("--minimum-separation-pkpc", type=float, default=0.5)
    parser.add_argument("--maximum-separation-pkpc", type=float, default=30.0)
    args = parser.parse_args()

    hosts = read_sink_host_catalog(args.host_catalog)
    redshift, _, records = read_mkagn_snapshot(args.agn_snapshot)
    pairs = find_dual_agn_pairs(
        records,
        redshift,
        args.dimensionless_hubble,
        luminosity_threshold_erg_s=args.luminosity_threshold_erg_s,
        minimum_separation_pkpc=args.minimum_separation_pkpc,
        maximum_separation_pkpc=args.maximum_separation_pkpc,
        box_size_cmpc_over_h=args.box_size_cmpc_h,
    )
    relation, first_row, second_row = classify_sink_pair_hosts(
        pairs["id_1"], pairs["id_2"], hosts
    )

    active = np.isfinite(records["Lbol"]) & (
        records["Lbol"] >= args.luminosity_threshold_erg_s
    )
    active_row = lookup_sink_hosts(records["sink_id"][active], hosts)
    active_found = active_row >= 0
    active_background = np.zeros(active_row.size, dtype=bool)
    active_background[active_found] = hosts["background"][active_row[active_found]] == 1

    relation_count = {
        str(label): int(np.count_nonzero(relation == code))
        for code, label in enumerate(HOST_RELATION_LABELS)
    }
    classifiable = relation >= 2
    distinct_host = relation >= 3
    first_stellar_count = _host_value(
        hosts, first_row, "host_stellar_count", -1
    ).astype(np.int64)
    second_stellar_count = _host_value(
        hosts, second_row, "host_stellar_count", -1
    ).astype(np.int64)
    first_stellar_mass = _host_value(
        hosts, first_row, "host_stellar_mass_msun_h", np.nan
    ) / args.dimensionless_hubble
    second_stellar_mass = _host_value(
        hosts, second_row, "host_stellar_mass_msun_h", np.nan
    ) / args.dimensionless_hubble
    hr5_100_star_particle_selection = (
        (pairs["mass_1_msun"] >= 1.0e6)
        & (pairs["mass_2_msun"] >= 1.0e6)
        & (first_stellar_count >= 100)
        & (second_stellar_count >= 100)
    )
    fable_selection_analogue = (
        (pairs["mass_1_msun"] >= 1.0e6)
        & (pairs["mass_2_msun"] >= 1.0e6)
        & (first_stellar_mass >= FABLE_MINIMUM_HOST_STELLAR_MASS_MSUN)
        & (second_stellar_mass >= FABLE_MINIMUM_HOST_STELLAR_MASS_MSUN)
    )
    summary = {
        "output": _output_number(args.agn_snapshot),
        "redshift": redshift,
        "luminosity_threshold_erg_s": args.luminosity_threshold_erg_s,
        "separation_pkpc": [
            args.minimum_separation_pkpc,
            args.maximum_separation_pkpc,
        ],
        "active_smbh_count": int(np.count_nonzero(active)),
        "active_smbh_with_direct_psb_host": int(
            np.count_nonzero(active_found & ~active_background)
        ),
        "active_smbh_in_fof_background": int(np.count_nonzero(active_background)),
        "active_smbh_without_direct_psb_host": int(np.count_nonzero(~active_found)),
        "spatially_selected_active_pair_count": int(relation.size),
        "host_relation_count": relation_count,
        "pair_count_with_two_psb_hosts": int(np.count_nonzero(classifiable)),
        "distinct_host_dual_agn_candidate_count": int(np.count_nonzero(distinct_host)),
        "distinct_host_fraction_among_pairs_with_two_psb_hosts": (
            float(np.count_nonzero(distinct_host) / np.count_nonzero(classifiable))
            if np.any(classifiable)
            else None
        ),
        "fable_selection_analogue": {
            "minimum_smbh_mass_msun": 1.0e6,
            "minimum_host_stellar_mass_msun": FABLE_MINIMUM_HOST_STELLAR_MASS_MSUN,
            "host_resolution_note": (
                "The threshold matches FABLE's published stellar-mass value, but "
                "HR5 uses total PSB stellar mass rather than mass within twice the "
                "stellar half-mass radius."
            ),
            "pair_count": int(np.count_nonzero(fable_selection_analogue)),
            "distinct_host_pair_count": int(
                np.count_nonzero(fable_selection_analogue & distinct_host)
            ),
        },
        "hr5_100_star_particle_selection": {
            "minimum_smbh_mass_msun": 1.0e6,
            "minimum_host_stellar_particle_count": 100,
            "pair_count": int(np.count_nonzero(hr5_100_star_particle_selection)),
            "distinct_host_pair_count": int(
                np.count_nonzero(hr5_100_star_particle_selection & distinct_host)
            ),
        },
    }

    args.output_directory.mkdir(parents=True, exist_ok=True)
    output = summary["output"]
    tag = f"{output:05d}" if output is not None else "unknown"
    pair_path = args.output_directory / f"hr5_dual_agn_hosts.{tag}.csv"
    summary_path = args.output_directory / f"hr5_dual_agn_hosts.{tag}.json"
    _write_pair_table(
        pair_path,
        pairs,
        hosts,
        relation,
        first_row,
        second_row,
        args.dimensionless_hubble,
    )
    with summary_path.open("w") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

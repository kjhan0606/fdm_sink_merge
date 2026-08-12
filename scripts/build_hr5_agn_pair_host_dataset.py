#!/usr/bin/env python3
"""Build a direct-host catalogue for dual and single-AGN SMBH pairs in HR5."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from analyze_hr5_dual_agn_hosts import FABLE_MINIMUM_HOST_STELLAR_MASS_MSUN
from fdm_smbh_delay.hr5 import (
    HOST_RELATION_LABELS,
    classify_sink_pair_hosts,
    find_agn_pair_population,
    pair_component_labels,
    pair_component_multiplicity,
    read_mkagn_snapshot,
    read_sink_host_catalog,
)


DEFAULT_HR5_ROOT = Path("/scratch/kjhan/Hydro/HR5/FoFPSB/02_without_dc/Ver2")
DEFAULT_CANONICAL_ROOT = DEFAULT_HR5_ROOT / "Derived_Sink_Hosts" / "canonical_v1"


FIELDS = (
    "output_number",
    "redshift",
    "pair_class",
    "primary_sink_id",
    "secondary_sink_id",
    "primary_active",
    "secondary_active",
    "separation_pkpc",
    "primary_mass_msun",
    "secondary_mass_msun",
    "mass_ratio",
    "primary_lbol_erg_s",
    "secondary_lbol_erg_s",
    "primary_lhx_erg_s",
    "secondary_lhx_erg_s",
    "primary_eddington_ratio",
    "secondary_eddington_ratio",
    "dual_system_multiplicity",
    "pair_system_multiplicity",
    "pair_system_label",
    "relative_speed_kms",
    "primary_position_x_cmpc_h",
    "primary_position_y_cmpc_h",
    "primary_position_z_cmpc_h",
    "secondary_position_x_cmpc_h",
    "secondary_position_y_cmpc_h",
    "secondary_position_z_cmpc_h",
    "primary_velocity_x_kms",
    "primary_velocity_y_kms",
    "primary_velocity_z_kms",
    "secondary_velocity_x_kms",
    "secondary_velocity_y_kms",
    "secondary_velocity_z_kms",
    "host_relation",
    "primary_galaxy_gid",
    "secondary_galaxy_gid",
    "primary_fof_index",
    "secondary_fof_index",
    "primary_host_stellar_mass_msun",
    "secondary_host_stellar_mass_msun",
    "primary_host_gas_mass_msun",
    "secondary_host_gas_mass_msun",
    "primary_host_total_mass_msun",
    "secondary_host_total_mass_msun",
    "primary_host_stellar_particle_count",
    "secondary_host_stellar_particle_count",
    "fable_selection_analogue",
    "hr5_100_star_particle_selection",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _host_value(
    hosts: np.ndarray,
    row: np.ndarray,
    field: str,
    fill: float,
) -> np.ndarray:
    value = np.full(row.size, fill, dtype=np.float64)
    found = row >= 0
    value[found] = hosts[field][row[found]]
    return value


def _add_system_information(pairs: dict[str, np.ndarray]) -> None:
    label, multiplicity, _, _ = pair_component_labels(pairs["id_1"], pairs["id_2"])
    pairs["pair_system_label"] = label
    pairs["pair_system_multiplicity"] = multiplicity
    pairs["relative_speed_kms"] = np.linalg.norm(
        pairs["velocity_2_kms"] - pairs["velocity_1_kms"], axis=1
    )
    dual = pairs["is_dual"]
    dual_multiplicity, _, _ = pair_component_multiplicity(
        pairs["id_1"][dual], pairs["id_2"][dual]
    )
    pairs["dual_system_multiplicity"] = np.zeros(dual.size, dtype=np.int64)
    pairs["dual_system_multiplicity"][dual] = dual_multiplicity


def _output_rows(
    output: int,
    redshift: float,
    mkagn_path: Path,
    host_path: Path,
    dimensionless_hubble: float,
    luminosity_threshold_erg_s: float,
    minimum_smbh_mass_msun: float,
    box_size_cmpc_h: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    snapshot_redshift, _, records = read_mkagn_snapshot(mkagn_path)
    if not np.isclose(snapshot_redshift, redshift, rtol=0.0, atol=1.0e-8):
        raise ValueError(f"Redshift mismatch at output {output:05d}")
    pairs = find_agn_pair_population(
        records,
        redshift,
        dimensionless_hubble,
        luminosity_threshold_erg_s=luminosity_threshold_erg_s,
        minimum_mass_msun=minimum_smbh_mass_msun,
        box_size_cmpc_over_h=box_size_cmpc_h,
    )
    _add_system_information(pairs)
    hosts = read_sink_host_catalog(host_path)
    relation, first_row, second_row = classify_sink_pair_hosts(
        pairs["id_1"], pairs["id_2"], hosts
    )
    first_gid = _host_value(hosts, first_row, "galaxy_gid", -1).astype(np.int64)
    second_gid = _host_value(hosts, second_row, "galaxy_gid", -1).astype(np.int64)
    first_fof = _host_value(hosts, first_row, "fof_index", -1).astype(np.int64)
    second_fof = _host_value(hosts, second_row, "fof_index", -1).astype(np.int64)
    first_stars = _host_value(hosts, first_row, "host_stellar_count", -1).astype(
        np.int64
    )
    second_stars = _host_value(hosts, second_row, "host_stellar_count", -1).astype(
        np.int64
    )

    def host_mass(field: str) -> tuple[np.ndarray, np.ndarray]:
        return (
            _host_value(hosts, first_row, field, np.nan) / dimensionless_hubble,
            _host_value(hosts, second_row, field, np.nan) / dimensionless_hubble,
        )

    first_stellar_mass, second_stellar_mass = host_mass("host_stellar_mass_msun_h")
    first_gas_mass, second_gas_mass = host_mass("host_gas_mass_msun_h")
    first_total_mass, second_total_mass = host_mass("host_total_mass_msun_h")
    fable = (
        (pairs["mass_1_msun"] >= 1.0e6)
        & (pairs["mass_2_msun"] >= 1.0e6)
        & (first_stellar_mass >= FABLE_MINIMUM_HOST_STELLAR_MASS_MSUN)
        & (second_stellar_mass >= FABLE_MINIMUM_HOST_STELLAR_MASS_MSUN)
    )
    hr5_resolved = (
        (pairs["mass_1_msun"] >= 1.0e6)
        & (pairs["mass_2_msun"] >= 1.0e6)
        & (first_stars >= 100)
        & (second_stars >= 100)
    )

    rows: list[dict[str, object]] = []
    for index in range(pairs["id_1"].size):
        row = {
            "output_number": output,
            "redshift": redshift,
            "pair_class": "dual" if pairs["is_dual"][index] else "offset",
            "primary_sink_id": int(pairs["id_1"][index]),
            "secondary_sink_id": int(pairs["id_2"][index]),
            "primary_active": int(pairs["active_1"][index]),
            "secondary_active": int(pairs["active_2"][index]),
            "separation_pkpc": pairs["separation_pkpc"][index],
            "primary_mass_msun": pairs["mass_1_msun"][index],
            "secondary_mass_msun": pairs["mass_2_msun"][index],
            "mass_ratio": pairs["mass_2_msun"][index] / pairs["mass_1_msun"][index],
            "primary_lbol_erg_s": pairs["lbol_1_erg_s"][index],
            "secondary_lbol_erg_s": pairs["lbol_2_erg_s"][index],
            "primary_lhx_erg_s": pairs["lhx_1_erg_s"][index],
            "secondary_lhx_erg_s": pairs["lhx_2_erg_s"][index],
            "primary_eddington_ratio": pairs["eddington_ratio_1"][index],
            "secondary_eddington_ratio": pairs["eddington_ratio_2"][index],
            "dual_system_multiplicity": int(
                pairs["dual_system_multiplicity"][index]
            ),
            "pair_system_multiplicity": int(
                pairs["pair_system_multiplicity"][index]
            ),
            "pair_system_label": int(pairs["pair_system_label"][index]),
            "relative_speed_kms": pairs["relative_speed_kms"][index],
            "primary_position_x_cmpc_h": pairs["position_1_cmpc_over_h"][index, 0],
            "primary_position_y_cmpc_h": pairs["position_1_cmpc_over_h"][index, 1],
            "primary_position_z_cmpc_h": pairs["position_1_cmpc_over_h"][index, 2],
            "secondary_position_x_cmpc_h": pairs["position_2_cmpc_over_h"][index, 0],
            "secondary_position_y_cmpc_h": pairs["position_2_cmpc_over_h"][index, 1],
            "secondary_position_z_cmpc_h": pairs["position_2_cmpc_over_h"][index, 2],
            "primary_velocity_x_kms": pairs["velocity_1_kms"][index, 0],
            "primary_velocity_y_kms": pairs["velocity_1_kms"][index, 1],
            "primary_velocity_z_kms": pairs["velocity_1_kms"][index, 2],
            "secondary_velocity_x_kms": pairs["velocity_2_kms"][index, 0],
            "secondary_velocity_y_kms": pairs["velocity_2_kms"][index, 1],
            "secondary_velocity_z_kms": pairs["velocity_2_kms"][index, 2],
            "host_relation": HOST_RELATION_LABELS[relation[index]],
            "primary_galaxy_gid": int(first_gid[index]),
            "secondary_galaxy_gid": int(second_gid[index]),
            "primary_fof_index": int(first_fof[index]),
            "secondary_fof_index": int(second_fof[index]),
            "primary_host_stellar_mass_msun": first_stellar_mass[index],
            "secondary_host_stellar_mass_msun": second_stellar_mass[index],
            "primary_host_gas_mass_msun": first_gas_mass[index],
            "secondary_host_gas_mass_msun": second_gas_mass[index],
            "primary_host_total_mass_msun": first_total_mass[index],
            "secondary_host_total_mass_msun": second_total_mass[index],
            "primary_host_stellar_particle_count": int(first_stars[index]),
            "secondary_host_stellar_particle_count": int(second_stars[index]),
            "fable_selection_analogue": int(fable[index]),
            "hr5_100_star_particle_selection": int(hr5_resolved[index]),
        }
        rows.append(row)
    summary = {
        "output": output,
        "redshift": redshift,
        "active_smbh_count": int(pairs["active_count"]),
        "pair_count": len(rows),
        "dual_pair_count": int(np.count_nonzero(pairs["is_dual"])),
        "single_agn_pair_count": int(np.count_nonzero(pairs["is_offset"])),
        "host_relation": dict(Counter(row["host_relation"] for row in rows)),
        "fable_selection_analogue_count": int(np.count_nonzero(fable)),
        "hr5_100_star_particle_selection_count": int(np.count_nonzero(hr5_resolved)),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    parser.add_argument("--outputs", nargs="+", type=int)
    parser.add_argument("--dimensionless-hubble", type=float, default=0.684)
    parser.add_argument("--luminosity-threshold-erg-s", type=float, default=1.0e43)
    parser.add_argument("--minimum-smbh-mass-msun", type=float, default=1.0e6)
    parser.add_argument("--box-size-cmpc-h", type=float, default=717.229040)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_CANONICAL_ROOT / "agn_pair_hosts",
    )
    args = parser.parse_args()
    if (
        args.dimensionless_hubble <= 0.0
        or args.luminosity_threshold_erg_s <= 0.0
        or args.minimum_smbh_mass_msun <= 0.0
        or args.box_size_cmpc_h <= 0.0
    ):
        parser.error("Physical thresholds and scale parameters must be positive")

    manifest = [
        row
        for row in _read_csv(args.canonical_root / "hr5_output_manifest.csv")
        if row["mkagn_path"]
        and row["sink_host_catalog_path"]
        and Path(row["sink_host_catalog_path"]).is_file()
    ]
    if args.outputs:
        requested = set(args.outputs)
        manifest = [row for row in manifest if int(row["output"]) in requested]
        missing = requested - {int(row["output"]) for row in manifest}
        if missing:
            parser.error(f"No complete MkAGN and direct-host data for {sorted(missing)}")
    if not manifest:
        raise ValueError("No complete MkAGN and direct-host outputs")

    args.output_directory.mkdir(parents=True, exist_ok=True)
    table_path = args.output_directory / "hr5_agn_pair_hosts_mbh_ge_1e6.csv"
    summary_path = args.output_directory / "hr5_agn_pair_hosts_mbh_ge_1e6.json"
    temporary_table = table_path.with_suffix(".csv.tmp")
    temporary_summary = summary_path.with_suffix(".json.tmp")
    summaries: list[dict[str, object]] = []
    total_rows = 0
    with temporary_table.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for manifest_row in sorted(manifest, key=lambda row: int(row["output"])):
            output = int(manifest_row["output"])
            print(f"Building AGN pair hosts at output {output:05d}", flush=True)
            rows, summary = _output_rows(
                output,
                float(manifest_row["redshift"]),
                Path(manifest_row["mkagn_path"]),
                Path(manifest_row["sink_host_catalog_path"]),
                args.dimensionless_hubble,
                args.luminosity_threshold_erg_s,
                args.minimum_smbh_mass_msun,
                args.box_size_cmpc_h,
            )
            writer.writerows(rows)
            summaries.append(summary)
            total_rows += len(rows)
    result = {
        "selection": {
            "minimum_smbh_mass_msun": args.minimum_smbh_mass_msun,
            "luminosity_threshold_erg_s": args.luminosity_threshold_erg_s,
            "separation_pkpc": [0.5, 30.0],
            "fable_minimum_host_stellar_mass_msun": (
                FABLE_MINIMUM_HOST_STELLAR_MASS_MSUN
            ),
        },
        "output_count": len(summaries),
        "pair_count": total_rows,
        "by_output": summaries,
        "provenance": {
            "host_assignment": "direct membership in an HR5 PSB galaxy",
            "capture_companion_used": False,
            "excluded_path_patterns": ["*.mine", "*.test", "*.try"],
        },
    }
    temporary_summary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary_table.replace(table_path)
    temporary_summary.replace(summary_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

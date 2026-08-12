#!/usr/bin/env python3
"""Export HR5 stellar sources, dusty AMR cells, and AGN for SKIRT."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fdm_smbh_delay.hr5_mock_observation import (
    formed_stellar_mass_msun,
    read_hr5_age_table,
    read_ramses_info,
    stellar_age_gyr,
)


SKIRT_IMPORT_REFERENCE = "https://skirt.ugent.be/root/_user_import_snap.html"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--particles", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--ramses-info", type=Path, required=True)
    parser.add_argument("--age-table", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stellar-smoothing-pc", type=float, default=300.0)
    parser.add_argument("--dust-to-metal-ratio", type=float, default=0.4)
    parser.add_argument(
        "--maximum-dust-temperature-k-per-mu", type=float, default=1.0e6
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_position_pc(
    coordinates: np.ndarray,
    center: np.ndarray,
    conversion_pkpc_per_cmpc_h: float,
) -> np.ndarray:
    return (coordinates - center[np.newaxis, :]) * conversion_pkpc_per_cmpc_h * 1000.0


def save_table(path: Path, data: np.ndarray, header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, data, fmt="%.10e", header=header, comments="# ")


def main() -> None:
    args = parse_args()
    targets = pd.read_csv(args.targets)
    if len(targets) != 6 or list(targets["panel"]) != list("abcdef"):
        raise ValueError("The target table must contain panels a through f")
    redshift = float(targets["redshift"].iloc[0])
    info = read_ramses_info(args.ramses_info)
    dimensionless_hubble = info["H0"] / 100.0
    conversion_pkpc_per_cmpc_h = 1000.0 / (
        dimensionless_hubble * (1.0 + redshift)
    )
    usecols = [
        "galaxy_gid",
        "particle_type",
        "x_cmpc_h",
        "y_cmpc_h",
        "z_cmpc_h",
        "mass_msun_h",
        "metallicity",
        "formation_time",
        "initial_mass_code",
        "cell_size_cmpc_h",
        "density_code",
        "temperature_code",
    ]
    particles = pd.read_csv(args.particles, usecols=usecols)
    stars_all = particles.loc[particles["particle_type"] == "star"].copy()
    _, conformal_time, lookback_time = read_hr5_age_table(args.age_table)
    stars_all["age_gyr"] = stellar_age_gyr(
        stars_all["formation_time"].to_numpy(),
        info["time"],
        conformal_time,
        lookback_time,
    )
    stars_all["initial_mass_msun"] = formed_stellar_mass_msun(
        stars_all["initial_mass_code"].to_numpy(), info["unit_l"], info["unit_d"]
    )

    args.output_directory.mkdir(parents=True, exist_ok=True)
    panels: list[dict[str, object]] = []
    for target in targets.itertuples(index=False):
        gids = [int(target.primary_galaxy_gid), int(target.secondary_galaxy_gid)]
        stars = stars_all.loc[stars_all["galaxy_gid"].isin(gids)]
        gas = particles.loc[
            (particles["particle_type"] == "gas") & particles["galaxy_gid"].isin(gids)
        ].copy()
        center = 0.5 * np.array(
            [
                target.primary_position_x_cmpc_h + target.secondary_position_x_cmpc_h,
                target.primary_position_y_cmpc_h + target.secondary_position_y_cmpc_h,
                target.primary_position_z_cmpc_h + target.secondary_position_z_cmpc_h,
            ],
            dtype=np.float64,
        )
        star_position = relative_position_pc(
            stars[["x_cmpc_h", "y_cmpc_h", "z_cmpc_h"]].to_numpy(),
            center,
            conversion_pkpc_per_cmpc_h,
        )
        star_table = np.column_stack(
            [
                star_position,
                np.full(len(stars), args.stellar_smoothing_pc),
                stars["initial_mass_msun"].to_numpy(),
                stars["metallicity"].to_numpy(),
                stars["age_gyr"].to_numpy(),
            ]
        )
        star_path = args.output_directory / f"panel_{target.panel}_stars.txt"
        save_table(
            star_path,
            star_table,
            "\n".join(
                [
                    "HR5 stellar particles for a SKIRT ParticleSource and an SSP SED family",
                    "Column 1: position x (pc)",
                    "Column 2: position y (pc)",
                    "Column 3: position z (pc)",
                    "Column 4: smoothing length (pc)",
                    "Column 5: initial mass (Msun)",
                    "Column 6: metallicity (1)",
                    "Column 7: age (Gyr)",
                ]
            ),
        )

        temperature_measure = np.divide(
            gas["temperature_code"].to_numpy(),
            gas["density_code"].to_numpy(),
            out=np.full(len(gas), np.inf),
            where=gas["density_code"].to_numpy() > 0.0,
        )
        dust_mass = (
            gas["mass_msun_h"].to_numpy()
            / dimensionless_hubble
            * gas["metallicity"].to_numpy()
            * args.dust_to_metal_ratio
        )
        selected = (
            (temperature_measure <= args.maximum_dust_temperature_k_per_mu)
            & np.isfinite(dust_mass)
            & (dust_mass > 0.0)
        )
        gas = gas.loc[selected]
        dust_mass = dust_mass[selected]
        cell_center = relative_position_pc(
            gas[["x_cmpc_h", "y_cmpc_h", "z_cmpc_h"]].to_numpy(),
            center,
            conversion_pkpc_per_cmpc_h,
        )
        cell_width_pc = (
            gas["cell_size_cmpc_h"].to_numpy()
            * conversion_pkpc_per_cmpc_h
            * 1000.0
        )
        half_width = 0.5 * cell_width_pc[:, np.newaxis]
        dust_table = np.column_stack(
            [cell_center - half_width, cell_center + half_width, dust_mass]
        )
        dust_path = args.output_directory / f"panel_{target.panel}_dust_cells.txt"
        save_table(
            dust_path,
            dust_table,
            "\n".join(
                [
                    "HR5 dusty AMR cells for a SKIRT CellMedium with massType=Mass",
                    "Column 1: box xmin (pc)",
                    "Column 2: box ymin (pc)",
                    "Column 3: box zmin (pc)",
                    "Column 4: box xmax (pc)",
                    "Column 5: box ymax (pc)",
                    "Column 6: box zmax (pc)",
                    "Column 7: dust mass (Msun)",
                ]
            ),
        )

        agn_position = relative_position_pc(
            np.array(
                [
                    [
                        target.primary_position_x_cmpc_h,
                        target.primary_position_y_cmpc_h,
                        target.primary_position_z_cmpc_h,
                    ],
                    [
                        target.secondary_position_x_cmpc_h,
                        target.secondary_position_y_cmpc_h,
                        target.secondary_position_z_cmpc_h,
                    ],
                ]
            ),
            center,
            conversion_pkpc_per_cmpc_h,
        )
        agn_table = np.column_stack(
            [
                agn_position,
                [target.primary_lbol_erg_s, target.secondary_lbol_erg_s],
                [target.primary_sink_id, target.secondary_sink_id],
            ]
        )
        agn_path = args.output_directory / f"panel_{target.panel}_agn.txt"
        save_table(
            agn_path,
            agn_table,
            "\n".join(
                [
                    "HR5 AGN positions and bolometric luminosities; assign a chosen AGN SED in SKIRT",
                    "Column 1: position x (pc)",
                    "Column 2: position y (pc)",
                    "Column 3: position z (pc)",
                    "Column 4: bolometric luminosity (erg/s)",
                    "Column 5: sink identifier (1)",
                ]
            ),
        )
        files = {
            "stars": star_path,
            "dust_cells": dust_path,
            "agn": agn_path,
        }
        panels.append(
            {
                "panel": target.panel,
                "primary_sink_id": int(target.primary_sink_id),
                "secondary_sink_id": int(target.secondary_sink_id),
                "center_cmpc_h": center.tolist(),
                "stellar_particle_count": int(len(stars)),
                "stellar_initial_mass_msun": float(stars["initial_mass_msun"].sum()),
                "dust_cell_count": int(len(gas)),
                "dust_mass_msun": float(dust_mass.sum()),
                "files": {
                    name: {
                        "path": str(path),
                        "sha256": sha256(path),
                        "bytes": path.stat().st_size,
                    }
                    for name, path in files.items()
                },
            }
        )

    manifest = {
        "status": "complete",
        "scientific_status": "three-dimensional transfer input; SKIRT execution pending",
        "output_number": int(targets["output_number"].iloc[0]),
        "redshift": redshift,
        "coordinate_origin": "midpoint between the two active SMBHs in each panel",
        "coordinate_units": "proper pc",
        "observer_direction": "+z axis looking toward the origin",
        "stellar_source": {
            "smoothing_length_pc": args.stellar_smoothing_pc,
            "columns": "SKIRT ParticleSource with an SSP family parameterized by initial mass, metallicity, and age",
        },
        "dust_medium": {
            "geometry": "native axis-aligned HR5 AMR leaf cells",
            "mass_type": "integrated dust mass",
            "dust_to_metal_ratio": args.dust_to_metal_ratio,
            "maximum_temperature_measure_k_per_mu": args.maximum_dust_temperature_k_per_mu,
            "temperature_measure": "temperature_code divided by density_code",
            "skirt_configuration": "CellMedium massType=Mass, massFraction=1, importMetallicity=false, importTemperature=false",
        },
        "agn_source": {
            "status": "positions and bolometric luminosities exported",
            "remaining_choice": "AGN SED family, anisotropy, torus, and variability model",
        },
        "reference": SKIRT_IMPORT_REFERENCE,
        "panels": panels,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

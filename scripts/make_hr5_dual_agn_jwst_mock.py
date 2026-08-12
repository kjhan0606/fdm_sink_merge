#!/usr/bin/env python3
"""Make a six-panel, dust-free JWST/F200W morphology mock for HR5 dual AGN."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

from fdm_smbh_delay.hr5_mock_observation import (
    convolve_with_psf,
    deposit_bilinear,
    load_psf,
)


PSF_REFERENCE = (
    "https://jwst-docs.stsci.edu/jwst-near-infrared-camera/"
    "nircam-performance/nircam-point-spread-functions"
)
PSF_LIBRARY = "https://stsci.app.box.com/v/jwst-simulated-psf-library"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--particles", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--psf", type=Path, required=True)
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--output-fits", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--dimensionless-hubble", type=float, default=0.684)
    parser.add_argument("--omega-m", type=float, default=0.3)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--psf-extension", default="DET_SAMP")
    parser.add_argument("--stellar-smoothing-pkpc", type=float, default=0.3)
    parser.add_argument("--draft-watermark", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def projected_offsets(
    x: np.ndarray | float,
    y: np.ndarray | float,
    center_x: float,
    center_y: float,
    conversion_pkpc_per_cmpc_h: float,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        (np.asarray(x) - center_x) * conversion_pkpc_per_cmpc_h,
        (np.asarray(y) - center_y) * conversion_pkpc_per_cmpc_h,
    )


def asinh_normalize(images: list[np.ndarray]) -> tuple[list[np.ndarray], float, float]:
    positive = np.concatenate([image[image > 0.0] for image in images])
    if not len(positive):
        raise ValueError("All projected stellar-mass maps are empty")
    lower = float(np.percentile(positive, 2.0))
    upper = float(np.percentile(positive, 99.7))
    softening = max(lower, upper / 500.0)
    denominator = np.arcsinh(upper / softening)
    normalized = [
        np.clip(np.arcsinh(image / softening) / denominator, 0.0, 1.0)
        for image in images
    ]
    return normalized, lower, upper


def main() -> None:
    args = parse_args()
    targets = pd.read_csv(args.targets)
    if len(targets) != 6 or list(targets["panel"]) != list("abcdef"):
        raise ValueError("The target table must contain panels a through f")
    redshifts = targets["redshift"].unique()
    if len(redshifts) != 1:
        raise ValueError("All targets must come from one redshift")
    redshift = float(redshifts[0])
    psf, pixel_scale_arcsec, psf_metadata = load_psf(args.psf, args.psf_extension)
    cosmology = FlatLambdaCDM(H0=100.0 * args.dimensionless_hubble, Om0=args.omega_m)
    pkpc_per_arcsec = float(cosmology.kpc_proper_per_arcmin(redshift).value / 60.0)
    pixel_pkpc = pixel_scale_arcsec * pkpc_per_arcsec
    field_pkpc = args.image_size * pixel_pkpc
    conversion = 1000.0 / (args.dimensionless_hubble * (1.0 + redshift))
    half_field = 0.5 * field_pkpc
    edges = np.linspace(-half_field, half_field, args.image_size + 1)
    pixel_area = pixel_pkpc**2

    usecols = [
        "galaxy_gid",
        "particle_type",
        "x_cmpc_h",
        "y_cmpc_h",
        "z_cmpc_h",
        "mass_msun_h",
        "metallicity",
    ]
    particles = pd.read_csv(args.particles, usecols=usecols)
    required_galaxies = set(
        targets["primary_galaxy_gid"].astype(int)
    ) | set(targets["secondary_galaxy_gid"].astype(int))
    available_galaxies = set(particles["galaxy_gid"].astype(int).unique())
    if required_galaxies != available_galaxies:
        raise ValueError("The extracted particle galaxies do not match the targets")

    intrinsic_host_maps: list[np.ndarray] = []
    raw_host_maps: list[np.ndarray] = []
    metal_maps: list[np.ndarray] = []
    agn_impulse_maps: list[np.ndarray] = []
    agn_maps: list[np.ndarray] = []
    panel_metadata: list[dict[str, object]] = []
    global_log_lbol = np.log10(
        np.concatenate(
            [targets["primary_lbol_erg_s"], targets["secondary_lbol_erg_s"]]
        )
    )
    lbol_min = float(global_log_lbol.min())
    lbol_max = float(global_log_lbol.max())

    for _, target in targets.iterrows():
        gids = [int(target["primary_galaxy_gid"]), int(target["secondary_galaxy_gid"])]
        pair_particles = particles.loc[particles["galaxy_gid"].isin(gids)]
        stars = pair_particles.loc[pair_particles["particle_type"] == "star"]
        gas = pair_particles.loc[pair_particles["particle_type"] == "gas"]
        center_x = 0.5 * (
            float(target["primary_position_x_cmpc_h"])
            + float(target["secondary_position_x_cmpc_h"])
        )
        center_y = 0.5 * (
            float(target["primary_position_y_cmpc_h"])
            + float(target["secondary_position_y_cmpc_h"])
        )
        star_x, star_y = projected_offsets(
            stars["x_cmpc_h"].to_numpy(),
            stars["y_cmpc_h"].to_numpy(),
            center_x,
            center_y,
            conversion,
        )
        gas_x, gas_y = projected_offsets(
            gas["x_cmpc_h"].to_numpy(),
            gas["y_cmpc_h"].to_numpy(),
            center_x,
            center_y,
            conversion,
        )
        stellar_mass = stars["mass_msun_h"].to_numpy() / args.dimensionless_hubble
        gas_metal_mass = (
            gas["mass_msun_h"].to_numpy()
            * gas["metallicity"].to_numpy()
            / args.dimensionless_hubble
        )
        stellar_histogram = np.histogram2d(
            star_y, star_x, bins=(edges, edges), weights=stellar_mass
        )[0]
        metal_histogram = np.histogram2d(
            gas_y, gas_x, bins=(edges, edges), weights=gas_metal_mass
        )[0]
        intrinsic_host = gaussian_filter(
            stellar_histogram,
            sigma=args.stellar_smoothing_pkpc / pixel_pkpc,
            mode="constant",
        )
        host_map = convolve_with_psf(intrinsic_host, psf) / pixel_area
        metal_map = metal_histogram / pixel_area

        primary_x, primary_y = projected_offsets(
            float(target["primary_position_x_cmpc_h"]),
            float(target["primary_position_y_cmpc_h"]),
            center_x,
            center_y,
            conversion,
        )
        secondary_x, secondary_y = projected_offsets(
            float(target["secondary_position_x_cmpc_h"]),
            float(target["secondary_position_y_cmpc_h"]),
            center_x,
            center_y,
            conversion,
        )
        primary_log_lbol = float(np.log10(target["primary_lbol_erg_s"]))
        secondary_log_lbol = float(np.log10(target["secondary_lbol_erg_s"]))
        impulses = np.zeros_like(host_map)
        for x_position, y_position, log_lbol in (
            (float(primary_x), float(primary_y), primary_log_lbol),
            (float(secondary_x), float(secondary_y), secondary_log_lbol),
        ):
            visual_weight = 0.35 + 0.65 * (log_lbol - lbol_min) / (lbol_max - lbol_min)
            deposit_bilinear(impulses, x_position, y_position, field_pkpc, visual_weight)
        agn_map = convolve_with_psf(impulses, psf)
        projected_separation = float(
            np.hypot(primary_x - secondary_x, primary_y - secondary_y)
        )
        if projected_separation > float(target["separation_pkpc"]) + 1.0e-6:
            raise ValueError("Projected separation exceeds the three-dimensional separation")
        if max(abs(primary_x), abs(primary_y), abs(secondary_x), abs(secondary_y)) >= half_field:
            raise ValueError("An active SMBH lies outside its image")
        inside = (
            (np.abs(star_x) < half_field)
            & (np.abs(star_y) < half_field)
        )
        intrinsic_host_maps.append(intrinsic_host)
        raw_host_maps.append(host_map)
        metal_maps.append(metal_map)
        agn_impulse_maps.append(impulses)
        agn_maps.append(agn_map)
        panel_metadata.append(
            {
                "panel": str(target["panel"]),
                "primary_sink_id": int(target["primary_sink_id"]),
                "secondary_sink_id": int(target["secondary_sink_id"]),
                "primary_galaxy_gid": gids[0],
                "secondary_galaxy_gid": gids[1],
                "separation_3d_pkpc": float(target["separation_pkpc"]),
                "separation_projected_pkpc": projected_separation,
                "primary_log10_lbol_erg_s": primary_log_lbol,
                "secondary_log10_lbol_erg_s": secondary_log_lbol,
                "star_particle_count": int(len(stars)),
                "star_particle_count_in_field": int(inside.sum()),
                "stellar_mass_fraction_in_field": float(
                    stellar_mass[inside].sum() / stellar_mass.sum()
                ),
                "gas_cell_count": int(len(gas)),
                "primary_xy_pkpc": [float(primary_x), float(primary_y)],
                "secondary_xy_pkpc": [float(secondary_x), float(secondary_y)],
            }
        )

    normalized_hosts, lower_scale, upper_scale = asinh_normalize(raw_host_maps)
    agn_peak = max(float(image.max()) for image in agn_maps)
    normalized_agn = [np.sqrt(np.clip(image / agn_peak, 0.0, 1.0)) for image in agn_maps]
    host_colormap = LinearSegmentedColormap.from_list(
        "hr5_host",
        [
            (0.0, "#020510"),
            (0.18, "#0d1730"),
            (0.48, "#70402d"),
            (0.78, "#d49a5b"),
            (1.0, "#fff1ce"),
        ],
    )

    plt.rcParams.update(
        {
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "font.family": "serif",
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.9), sharex=True, sharey=True)
    extent = (-half_field, half_field, -half_field, half_field)
    for index, (axis, target, host_image, agn_image, details) in enumerate(
        zip(axes.flat, targets.itertuples(index=False), normalized_hosts, normalized_agn,
            panel_metadata, strict=True)
    ):
        host_rgb = host_colormap(host_image)[..., :3]
        cyan = np.zeros_like(host_rgb)
        cyan[..., 0] = 0.20 * agn_image
        cyan[..., 1] = 0.95 * agn_image
        cyan[..., 2] = 1.00 * agn_image
        composite = np.clip(
            host_rgb * (1.0 - 0.55 * agn_image[..., np.newaxis]) + 1.25 * cyan,
            0.0,
            1.0,
        )
        axis.imshow(composite, origin="lower", extent=extent, interpolation="nearest")
        primary_xy = details["primary_xy_pkpc"]
        secondary_xy = details["secondary_xy_pkpc"]
        axis.scatter(
            [primary_xy[0]], [primary_xy[1]], s=28, facecolors="none",
            edgecolors="#5ee8ff", linewidths=0.8, zorder=5,
        )
        axis.scatter(
            [secondary_xy[0]], [secondary_xy[1]], s=28, facecolors="none",
            edgecolors="#f58cff", linewidths=0.8, zorder=5,
        )
        axis.text(
            0.025, 0.965, f"({target.panel})", transform=axis.transAxes,
            ha="left", va="top", color="white", fontsize=7.0,
        )
        axis.text(
            0.975,
            0.965,
            rf"$r_{{\rm 3D}}={target.separation_pkpc:.1f}$ pkpc"
            "\n"
            rf"$\log L_{{\rm bol}}={np.log10(target.primary_lbol_erg_s):.1f},"
            rf"{np.log10(target.secondary_lbol_erg_s):.1f}$",
            transform=axis.transAxes,
            ha="right",
            va="top",
            color="white",
            fontsize=6.2,
            linespacing=1.15,
            bbox={"facecolor": "#000000", "edgecolor": "none", "alpha": 0.36, "pad": 1.5},
        )
        axis.set_xlim(-half_field, half_field)
        axis.set_ylim(-half_field, half_field)
        axis.set_xticks([-20, 0, 20])
        axis.set_yticks([-20, 0, 20])
        axis.tick_params(
            color="white", labelcolor="black", direction="in", length=2.5, width=0.6
        )
        for spine in axis.spines.values():
            spine.set_color("white")
            spine.set_linewidth(0.5)
        if index >= 3:
            axis.set_xlabel(r"$x$ (pkpc)", color="black")
        if index % 3 == 0:
            axis.set_ylabel(r"$y$ (pkpc)", color="black")
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.085, top=0.995, wspace=0.035, hspace=0.055)
    if args.draft_watermark:
        fig.text(
            0.5, 0.50, "DRAFT", ha="center", va="center", rotation=28,
            fontsize=70, color="white", alpha=0.10, weight="bold", zorder=20,
        )

    for path in (args.output_pdf, args.output_png, args.output_fits, args.metadata):
        path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_pdf, dpi=300)
    fig.savefig(args.output_png, dpi=300)
    plt.close(fig)

    primary_header = fits.Header()
    primary_header["REDSHIFT"] = redshift
    primary_header["FILTER"] = "F200W"
    primary_header["PIXSCALE"] = (pixel_scale_arcsec, "arcsec pixel-1")
    primary_header["PIXKPC"] = (pixel_pkpc, "proper kpc pixel-1")
    primary_header["DUST"] = (False, "No attenuation applied")
    primary_header["LOSDIR"] = ("simulation z", "line-of-sight direction")
    hdus: list[fits.ImageHDU | fits.PrimaryHDU] = [fits.PrimaryHDU(header=primary_header)]
    psf_header = fits.Header()
    psf_header["BUNIT"] = "fraction"
    psf_header["INTEGRAL"] = float(psf.sum())
    hdus.append(fits.ImageHDU(psf.astype(np.float32), header=psf_header, name="PSF"))
    for details, intrinsic_host, host_map, metal_map, agn_impulses, agn_map in zip(
        panel_metadata,
        intrinsic_host_maps,
        raw_host_maps,
        metal_maps,
        agn_impulse_maps,
        agn_maps,
        strict=True,
    ):
        panel = str(details["panel"]).upper()
        common = fits.Header()
        common["PANEL"] = panel.lower()
        common["SINKPRI"] = details["primary_sink_id"]
        common["SINKSEC"] = details["secondary_sink_id"]
        common["R3DPKPC"] = details["separation_3d_pkpc"]
        common["R2DPKPC"] = details["separation_projected_pkpc"]
        intrinsic_header = common.copy()
        intrinsic_header["BUNIT"] = "Msun pixel-1"
        intrinsic_header["PSFCONV"] = False
        host_header = common.copy()
        host_header["BUNIT"] = "Msun pkpc-2"
        host_header["PSFCONV"] = True
        metal_header = common.copy()
        metal_header["BUNIT"] = "Msun pkpc-2"
        metal_header["PSFCONV"] = False
        agn_intrinsic_header = common.copy()
        agn_intrinsic_header["BUNIT"] = "display weight"
        agn_intrinsic_header["PSFCONV"] = False
        agn_header = common.copy()
        agn_header["BUNIT"] = "display weight"
        agn_header["PSFCONV"] = True
        hdus.append(
            fits.ImageHDU(
                intrinsic_host.astype(np.float32),
                header=intrinsic_header,
                name=f"HOSTINT_{panel}",
            )
        )
        hdus.append(
            fits.ImageHDU(
                host_map.astype(np.float32), header=host_header, name=f"HOST_{panel}"
            )
        )
        hdus.append(
            fits.ImageHDU(
                metal_map.astype(np.float32), header=metal_header, name=f"METAL_{panel}"
            )
        )
        hdus.append(
            fits.ImageHDU(
                agn_impulses.astype(np.float32),
                header=agn_intrinsic_header,
                name=f"AGNINT_{panel}",
            )
        )
        hdus.append(
            fits.ImageHDU(
                agn_map.astype(np.float32), header=agn_header, name=f"AGN_{panel}"
            )
        )
    fits.HDUList(hdus).writeto(args.output_fits, overwrite=True, checksum=True)

    metadata = {
        "status": "complete",
        "output_number": int(targets["output_number"].iloc[0]),
        "redshift": redshift,
        "selection": {
            "number_of_systems": 6,
            "method": "equally spaced separation ranks after the physical and resolution cuts",
            "pair_class": "dual",
            "host_relation": "distinct PSB galaxies in one FoF halo",
            "pair_system_multiplicity": 2,
            "hr5_100_star_particle_selection": True,
            "fable_selection_analogue": True,
        },
        "projection": {
            "line_of_sight": "simulation z axis",
            "field_pkpc": field_pkpc,
            "image_size_pixels": args.image_size,
            "pixel_scale_arcsec": pixel_scale_arcsec,
            "pixel_scale_pkpc": pixel_pkpc,
            "pkpc_per_arcsec": pkpc_per_arcsec,
            "dimensionless_hubble": args.dimensionless_hubble,
            "omega_m": args.omega_m,
        },
        "host_light_model": {
            "quantity": "projected stellar mass with a spatially constant mass-to-light ratio",
            "units_before_display_scaling": "Msun pkpc^-2",
            "display_scaling": "common asinh scaling for all six panels",
            "display_lower_msun_pkpc2": lower_scale,
            "display_upper_msun_pkpc2": upper_scale,
            "stellar_particle_gaussian_sigma_pkpc": args.stellar_smoothing_pkpc,
            "attenuation_applied": False,
            "interpretation": "dust-free morphology proxy, not calibrated F200W photometry",
        },
        "agn_overlay": {
            "positions": "HR5 SMBH positions",
            "kernel": "same F200W PSF as the host map",
            "amplitude": "logarithmically compressed visual scaling of bolometric luminosity",
            "photometric_interpretation": False,
            "primary_ring_color": "cyan",
            "secondary_ring_color": "magenta",
        },
        "gas_layer": {
            "quantity": "projected gas metal mass per physical area",
            "saved_to_fits": True,
            "used_as_attenuation": False,
            "reason": "a dust-to-metal relation, opacity curve, and foreground geometry remain unspecified",
        },
        "psf": {
            **psf_metadata,
            "filter": "F200W",
            "source_file": str(args.psf),
            "sha256": sha256(args.psf),
            "official_documentation": PSF_REFERENCE,
            "official_library": PSF_LIBRARY,
            "published_simulated_fwhm_arcsec": 0.064,
        },
        "intermediate_layers": {
            "PSF": "normalized detector-sampled F200W kernel",
            "HOSTINT_[A-F]": "intrinsic smoothed stellar-mass map before PSF convolution",
            "HOST_[A-F]": "PSF-convolved stellar-mass surface-density map",
            "METAL_[A-F]": "projected gas metal-mass surface-density map",
            "AGNINT_[A-F]": "unresolved AGN display impulses before PSF convolution",
            "AGN_[A-F]": "PSF-convolved AGN display layer",
        },
        "panels": panel_metadata,
        "outputs": {
            "pdf": str(args.output_pdf),
            "png": str(args.output_png),
            "fits": str(args.output_fits),
        },
    }
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "field_pkpc": field_pkpc, "panels": panel_metadata}, indent=2))


if __name__ == "__main__":
    main()

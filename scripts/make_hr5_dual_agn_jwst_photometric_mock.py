#!/usr/bin/env python3
"""Make a calibrated F200W quick-look image for six HR5 dual AGN systems."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

from fdm_smbh_delay.hr5_mock_observation import (
    ForegroundScreenResult,
    agn_power_law_band_flux_njy,
    apply_foreground_screen,
    convolve_with_psf,
    deposit_bilinear,
    dust_opacity_at_wavelength,
    formed_stellar_mass_msun,
    load_draine_dust_curve,
    load_psf,
    load_throughput_curve,
    observed_band_flux_njy_from_lnu,
    project_square_amr_cells,
    read_hr5_age_table,
    read_ramses_info,
    stellar_age_gyr,
)


F200W_THROUGHPUT_REFERENCE = (
    "https://jwst-docs.stsci.edu/jwst-near-infrared-camera/"
    "nircam-instrumentation/nircam-filters"
)
DRAINE_DUST_REFERENCE = "https://www.astro.princeton.edu/~draine/dust/dustmix.html"
FSPS_REFERENCE = "https://github.com/cconroy20/fsps"
AGN_SED_REFERENCE = "https://doi.org/10.1086/506525"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--particles", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--ramses-info", type=Path, required=True)
    parser.add_argument("--age-table", type=Path, required=True)
    parser.add_argument("--sps-home", type=Path, required=True)
    parser.add_argument("--sps-cache", type=Path, required=True)
    parser.add_argument("--throughput", type=Path, required=True)
    parser.add_argument("--dust-curve", type=Path, required=True)
    parser.add_argument("--psf", type=Path, required=True)
    parser.add_argument("--psf-extension", default="DET_SAMP")
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--output-fits", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--stellar-smoothing-pkpc", type=float, default=0.3)
    parser.add_argument("--dust-to-metal-ratio", type=float, default=0.4)
    parser.add_argument(
        "--maximum-dust-temperature-k-per-mu", type=float, default=1.0e6
    )
    parser.add_argument("--scattering-sigma-pixel", type=float, default=2.0)
    parser.add_argument("--agn-bolometric-correction-5100", type=float, default=10.3)
    parser.add_argument("--agn-alpha-nu", type=float, default=-0.5)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(path: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


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


def build_or_load_fsps_grid(
    sps_home: Path,
    cache_path: Path,
    redshift: float,
    luminosity_distance_cm: float,
    throughput_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    throughput = load_throughput_curve(throughput_path)
    expected = {
        "fsps_revision": git_revision(sps_home),
        "throughput_sha256": sha256(throughput_path),
        "redshift": redshift,
        "luminosity_distance_cm": luminosity_distance_cm,
    }
    if cache_path.is_file():
        with np.load(cache_path) as cache:
            metadata = json.loads(str(cache["metadata_json"]))
            if all(metadata.get(key) == value for key, value in expected.items()):
                return (
                    cache["log_age_year"],
                    cache["log_metallicity"],
                    cache["log10_flux_njy_per_formed_msun"],
                    metadata,
                )

    os.environ["SPS_HOME"] = str(sps_home)
    import fsps  # Imported only after SPS_HOME is defined.

    population = fsps.StellarPopulation(
        compute_vega_mags=False,
        zcontinuous=0,
        sfh=0,
        add_neb_emission=False,
        dust_type=0,
        dust2=0.0,
    )
    log_age_year = np.asarray(population.ssp_ages, dtype=np.float64)
    metallicity = np.asarray(population.zlegend, dtype=np.float64)
    log_metallicity = np.log10(metallicity)
    band_flux = np.empty((len(metallicity), len(log_age_year)), dtype=np.float64)
    wavelength: np.ndarray | None = None
    for index in range(len(metallicity)):
        wavelength, spectra = population.get_spectrum(zmet=index + 1, tage=0.0)
        band_flux[index] = observed_band_flux_njy_from_lnu(
            wavelength,
            spectra,
            redshift,
            luminosity_distance_cm,
            throughput,
        )
    if wavelength is None or np.any(~np.isfinite(band_flux)) or np.any(band_flux <= 0.0):
        raise RuntimeError("FSPS returned an invalid F200W luminosity grid")
    metadata = {
        **expected,
        "python_fsps_version": fsps.__version__,
        "libraries": [
            value.decode("ascii") if isinstance(value, bytes) else str(value)
            for value in population.libraries
        ],
        "solar_metallicity": float(population.solar_metallicity),
        "metallicity_grid": metallicity.tolist(),
        "log_age_year_grid": log_age_year.tolist(),
        "spectrum_units": "Lsun Hz^-1 per solar mass formed",
        "band_flux_units": "nJy per solar mass formed",
        "filter_integration": "photon-weighted mean Fnu",
        "reference": FSPS_REFERENCE,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        log_age_year=log_age_year,
        log_metallicity=log_metallicity,
        log10_flux_njy_per_formed_msun=np.log10(band_flux),
        metadata_json=json.dumps(metadata, sort_keys=True),
    )
    return log_age_year, log_metallicity, np.log10(band_flux), metadata


def stellar_f200w_flux_njy(
    stellar_age: np.ndarray,
    metallicity: np.ndarray,
    formed_mass: np.ndarray,
    log_age_grid: np.ndarray,
    log_metallicity_grid: np.ndarray,
    log_flux_grid: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    age_year = np.clip(stellar_age * 1.0e9, 10.0 ** log_age_grid[0], 10.0 ** log_age_grid[-1])
    clipped_metallicity = np.clip(
        metallicity, 10.0 ** log_metallicity_grid[0], 10.0 ** log_metallicity_grid[-1]
    )
    interpolator = RegularGridInterpolator(
        (log_metallicity_grid, log_age_grid),
        log_flux_grid,
        bounds_error=True,
    )
    coordinates = np.column_stack(
        [np.log10(clipped_metallicity), np.log10(age_year)]
    )
    flux = 10.0 ** interpolator(coordinates) * formed_mass
    clipping = {
        "age_below_grid": int(np.sum(stellar_age * 1.0e9 < 10.0 ** log_age_grid[0])),
        "age_above_grid": int(np.sum(stellar_age * 1.0e9 > 10.0 ** log_age_grid[-1])),
        "metallicity_below_grid": int(
            np.sum(metallicity < 10.0 ** log_metallicity_grid[0])
        ),
        "metallicity_above_grid": int(
            np.sum(metallicity > 10.0 ** log_metallicity_grid[-1])
        ),
    }
    return flux, clipping


def transfer_budget(
    intrinsic: np.ndarray,
    transfer: ForegroundScreenResult,
    observed: np.ndarray,
) -> dict[str, float]:
    return {
        "intrinsic_njy": float(intrinsic.sum()),
        "direct_njy": float(transfer.direct.sum()),
        "absorbed_njy": float(transfer.absorbed.sum()),
        "scattered_before_field_crop_njy": float(
            transfer.scattered_before_redistribution.sum()
        ),
        "scattered_inside_field_njy": float(transfer.scattered_in_field.sum()),
        "emergent_before_psf_njy": float(transfer.emergent.sum()),
        "observed_after_psf_field_crop_njy": float(observed.sum()),
    }


def display_scale(images: list[np.ndarray]) -> tuple[list[np.ndarray], float, float]:
    positive = np.concatenate([image[image > 0.0] for image in images])
    if not len(positive):
        raise ValueError("Every calibrated image is empty")
    lower = float(np.percentile(positive, 1.0))
    upper = float(np.percentile(positive, 99.9))
    softening = max(lower, upper / 1000.0)
    denominator = np.arcsinh(upper / softening)
    return (
        [
            np.clip(np.arcsinh(image / softening) / denominator, 0.0, 1.0)
            for image in images
        ],
        lower,
        upper,
    )


def main() -> None:
    args = parse_args()
    targets = pd.read_csv(args.targets)
    if len(targets) != 6 or list(targets["panel"]) != list("abcdef"):
        raise ValueError("The target table must contain panels a through f")
    redshifts = targets["redshift"].unique()
    if len(redshifts) != 1:
        raise ValueError("All targets must come from one redshift")
    redshift = float(redshifts[0])

    info = read_ramses_info(args.ramses_info)
    expected_scale_factor = 1.0 / (1.0 + redshift)
    if not np.isclose(info["aexp"], expected_scale_factor, rtol=2.0e-4):
        raise ValueError("The RAMSES output and target redshifts do not agree")
    dimensionless_hubble = info["H0"] / 100.0
    cosmology = FlatLambdaCDM(H0=info["H0"], Om0=info["omega_m"])
    luminosity_distance_cm = float(cosmology.luminosity_distance(redshift).cgs.value)
    throughput = load_throughput_curve(args.throughput)
    psf, pixel_scale_arcsec, psf_metadata = load_psf(args.psf, args.psf_extension)
    pkpc_per_arcsec = float(cosmology.kpc_proper_per_arcmin(redshift).value / 60.0)
    pixel_pkpc = pixel_scale_arcsec * pkpc_per_arcsec
    field_pkpc = args.image_size * pixel_pkpc
    half_field = 0.5 * field_pkpc
    pixel_area_pkpc2 = pixel_pkpc**2
    conversion = 1000.0 / (dimensionless_hubble * (1.0 + redshift))
    edges = np.linspace(-half_field, half_field, args.image_size + 1)

    log_age_grid, log_metallicity_grid, log_flux_grid, fsps_metadata = (
        build_or_load_fsps_grid(
            args.sps_home,
            args.sps_cache,
            redshift,
            luminosity_distance_cm,
            args.throughput,
        )
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
        "density_code",
        "temperature_code",
        "cell_size_cmpc_h",
    ]
    particles = pd.read_csv(args.particles, usecols=usecols)
    required_galaxies = set(targets["primary_galaxy_gid"].astype(int)) | set(
        targets["secondary_galaxy_gid"].astype(int)
    )
    if required_galaxies != set(particles["galaxy_gid"].astype(int).unique()):
        raise ValueError("The particle table does not contain exactly the target galaxies")

    stars_all = particles.loc[particles["particle_type"] == "star"].copy()
    _, conformal_time, lookback_time = read_hr5_age_table(args.age_table)
    ages = stellar_age_gyr(
        stars_all["formation_time"].to_numpy(),
        info["time"],
        conformal_time,
        lookback_time,
    )
    formed_mass = formed_stellar_mass_msun(
        stars_all["initial_mass_code"].to_numpy(), info["unit_l"], info["unit_d"]
    )
    stellar_flux, clipping = stellar_f200w_flux_njy(
        ages,
        stars_all["metallicity"].to_numpy(),
        formed_mass,
        log_age_grid,
        log_metallicity_grid,
        log_flux_grid,
    )
    stars_all["stellar_age_gyr"] = ages
    stars_all["formed_mass_msun"] = formed_mass
    stars_all["f200w_flux_njy"] = stellar_flux

    dust_wavelength, dust_absorption, dust_scattering, _ = load_draine_dust_curve(
        args.dust_curve
    )
    rest_pivot_micron = throughput.pivot_wavelength_micron / (1.0 + redshift)
    kappa_abs, kappa_sca = dust_opacity_at_wavelength(
        rest_pivot_micron, dust_wavelength, dust_absorption, dust_scattering
    )
    solar_mass_g = 1.988409870698051e33
    pkpc_cm = 3.085677581491367e21

    layers: list[dict[str, np.ndarray]] = []
    panels: list[dict[str, object]] = []
    for _, target in targets.iterrows():
        gids = [int(target["primary_galaxy_gid"]), int(target["secondary_galaxy_gid"])]
        stars = stars_all.loc[stars_all["galaxy_gid"].isin(gids)]
        gas = particles.loc[
            (particles["particle_type"] == "gas") & particles["galaxy_gid"].isin(gids)
        ]
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
        stellar_histogram = np.histogram2d(
            star_y,
            star_x,
            bins=(edges, edges),
            weights=stars["f200w_flux_njy"].to_numpy(),
        )[0]
        stellar_intrinsic = gaussian_filter(
            stellar_histogram,
            sigma=args.stellar_smoothing_pkpc / pixel_pkpc,
            mode="constant",
        )

        thermal_measure = np.divide(
            gas["temperature_code"].to_numpy(),
            gas["density_code"].to_numpy(),
            out=np.full(len(gas), np.inf),
            where=gas["density_code"].to_numpy() > 0.0,
        )
        dust_survives = thermal_measure <= args.maximum_dust_temperature_k_per_mu
        dust_mass_msun = (
            gas["mass_msun_h"].to_numpy()
            / dimensionless_hubble
            * gas["metallicity"].to_numpy()
            * args.dust_to_metal_ratio
            * dust_survives
        )
        gas_cell_width_pkpc = gas["cell_size_cmpc_h"].to_numpy() * conversion
        dust_histogram = project_square_amr_cells(
            gas_x,
            gas_y,
            gas_cell_width_pkpc,
            dust_mass_msun,
            edges,
        )
        dust_surface_density = (
            dust_histogram * solar_mass_g / (pixel_area_pkpc2 * pkpc_cm**2)
        )
        tau_abs = kappa_abs * dust_surface_density
        tau_sca = kappa_sca * dust_surface_density

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
        agn_intrinsic = np.zeros_like(stellar_intrinsic)
        primary_agn_flux = agn_power_law_band_flux_njy(
            float(target["primary_lbol_erg_s"]),
            redshift,
            luminosity_distance_cm,
            throughput,
            args.agn_bolometric_correction_5100,
            args.agn_alpha_nu,
        )
        secondary_agn_flux = agn_power_law_band_flux_njy(
            float(target["secondary_lbol_erg_s"]),
            redshift,
            luminosity_distance_cm,
            throughput,
            args.agn_bolometric_correction_5100,
            args.agn_alpha_nu,
        )
        deposit_bilinear(
            agn_intrinsic, float(primary_x), float(primary_y), field_pkpc, primary_agn_flux
        )
        deposit_bilinear(
            agn_intrinsic,
            float(secondary_x),
            float(secondary_y),
            field_pkpc,
            secondary_agn_flux,
        )
        star_transfer = apply_foreground_screen(
            stellar_intrinsic, tau_abs, tau_sca, args.scattering_sigma_pixel
        )
        agn_transfer = apply_foreground_screen(
            agn_intrinsic, tau_abs, tau_sca, args.scattering_sigma_pixel
        )
        stellar_observed = convolve_with_psf(star_transfer.emergent, psf)
        agn_observed = convolve_with_psf(agn_transfer.emergent, psf)
        total_observed = stellar_observed + agn_observed

        intrinsic_total = float(stellar_intrinsic.sum() + agn_intrinsic.sum())
        closed_total = float(
            star_transfer.direct.sum()
            + star_transfer.absorbed.sum()
            + star_transfer.scattered_before_redistribution.sum()
            + agn_transfer.direct.sum()
            + agn_transfer.absorbed.sum()
            + agn_transfer.scattered_before_redistribution.sum()
        )
        if not np.isclose(intrinsic_total, closed_total, rtol=1.0e-12, atol=1.0e-12):
            raise RuntimeError("The foreground-screen energy budget does not close")
        panel = str(target["panel"])
        panels.append(
            {
                "panel": panel,
                "primary_sink_id": int(target["primary_sink_id"]),
                "secondary_sink_id": int(target["secondary_sink_id"]),
                "primary_galaxy_gid": gids[0],
                "secondary_galaxy_gid": gids[1],
                "separation_3d_pkpc": float(target["separation_pkpc"]),
                "separation_projected_pkpc": float(
                    np.hypot(primary_x - secondary_x, primary_y - secondary_y)
                ),
                "primary_xy_pkpc": [float(primary_x), float(primary_y)],
                "secondary_xy_pkpc": [float(secondary_x), float(secondary_y)],
                "primary_agn_intrinsic_f200w_njy": primary_agn_flux,
                "secondary_agn_intrinsic_f200w_njy": secondary_agn_flux,
                "star_particle_count": int(len(stars)),
                "gas_cell_count": int(len(gas)),
                "dust_surviving_gas_cell_count": int(dust_survives.sum()),
                "dust_mass_inside_projected_field_msun": float(dust_histogram.sum()),
                "tau_abs_percentiles": np.percentile(tau_abs, [50, 90, 99, 100]).tolist(),
                "tau_sca_percentiles": np.percentile(tau_sca, [50, 90, 99, 100]).tolist(),
                "stellar_budget": transfer_budget(
                    stellar_intrinsic, star_transfer, stellar_observed
                ),
                "agn_budget": transfer_budget(agn_intrinsic, agn_transfer, agn_observed),
                "screen_energy_closure_relative": (
                    closed_total - intrinsic_total
                )
                / intrinsic_total,
            }
        )
        layers.append(
            {
                "stellar_intrinsic": stellar_intrinsic,
                "agn_intrinsic": agn_intrinsic,
                "dust_surface_density": dust_surface_density,
                "tau_abs": tau_abs,
                "tau_sca": tau_sca,
                "stellar_direct": star_transfer.direct,
                "agn_direct": agn_transfer.direct,
                "stellar_scattered": star_transfer.scattered_in_field,
                "agn_scattered": agn_transfer.scattered_in_field,
                "total_absorbed": star_transfer.absorbed + agn_transfer.absorbed,
                "total_emergent": star_transfer.emergent + agn_transfer.emergent,
                "stellar_observed": stellar_observed,
                "agn_observed": agn_observed,
                "total_observed": total_observed,
            }
        )

    normalized_total, display_lower, display_upper = display_scale(
        [layer["total_observed"] for layer in layers]
    )
    normalized_agn = [
        np.divide(
            layer["agn_observed"],
            layer["total_observed"],
            out=np.zeros_like(layer["agn_observed"]),
            where=layer["total_observed"] > 0.0,
        )
        for layer in layers
    ]
    colormap = LinearSegmentedColormap.from_list(
        "f200w",
        ["#020510", "#111c35", "#75432d", "#d49a5b", "#fff1ce"],
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
    for index, (axis, target, intensity, agn_fraction, panel) in enumerate(
        zip(
            axes.flat,
            targets.itertuples(index=False),
            normalized_total,
            normalized_agn,
            panels,
            strict=True,
        )
    ):
        rgb = colormap(intensity)[..., :3]
        cyan = np.zeros_like(rgb)
        cyan[..., 0] = 0.15 * intensity * agn_fraction
        cyan[..., 1] = 0.90 * intensity * agn_fraction
        cyan[..., 2] = 1.00 * intensity * agn_fraction
        composite = np.clip(rgb * (1.0 - 0.6 * agn_fraction[..., None]) + cyan, 0.0, 1.0)
        axis.imshow(composite, origin="lower", extent=extent, interpolation="nearest")
        primary_xy = panel["primary_xy_pkpc"]
        secondary_xy = panel["secondary_xy_pkpc"]
        axis.scatter(
            [primary_xy[0]],
            [primary_xy[1]],
            s=28,
            facecolors="none",
            edgecolors="#5ee8ff",
            linewidths=0.8,
        )
        axis.scatter(
            [secondary_xy[0]],
            [secondary_xy[1]],
            s=28,
            facecolors="none",
            edgecolors="#f58cff",
            linewidths=0.8,
        )
        axis.text(
            0.025,
            0.965,
            f"({target.panel})",
            transform=axis.transAxes,
            ha="left",
            va="top",
            color="white",
            fontsize=7.0,
        )
        axis.text(
            0.975,
            0.965,
            rf"$r_{{\rm 3D}}={target.separation_pkpc:.1f}$ pkpc",
            transform=axis.transAxes,
            ha="right",
            va="top",
            color="white",
            fontsize=6.2,
            bbox={"facecolor": "#000000", "edgecolor": "none", "alpha": 0.36, "pad": 1.5},
        )
        axis.set_xlim(-half_field, half_field)
        axis.set_ylim(-half_field, half_field)
        axis.set_xticks([-20, 0, 20])
        axis.set_yticks([-20, 0, 20])
        axis.tick_params(color="white", labelcolor="black", direction="in", length=2.5)
        for spine in axis.spines.values():
            spine.set_color("white")
            spine.set_linewidth(0.5)
        if index >= 3:
            axis.set_xlabel(r"$x$ (pkpc)")
        if index % 3 == 0:
            axis.set_ylabel(r"$y$ (pkpc)")
    fig.subplots_adjust(
        left=0.075,
        right=0.995,
        bottom=0.085,
        top=0.995,
        wspace=0.035,
        hspace=0.055,
    )

    for output in (args.output_pdf, args.output_png, args.output_fits, args.metadata):
        output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_pdf, dpi=300)
    fig.savefig(args.output_png, dpi=300)
    plt.close(fig)

    primary_header = fits.Header()
    primary_header["REDSHIFT"] = redshift
    primary_header["FILTER"] = "F200W"
    primary_header["BUNIT"] = "nJy pixel-1"
    primary_header["PIXSCALE"] = (pixel_scale_arcsec, "arcsec pixel-1")
    primary_header["PIXKPC"] = (pixel_pkpc, "proper kpc pixel-1")
    primary_header["DUST"] = True
    primary_header["DUSTGEOM"] = "screen"
    primary_header["SCATTER"] = "image-plane"
    primary_header["LOSDIR"] = "simulation z"
    hdus: list[fits.PrimaryHDU | fits.ImageHDU] = [fits.PrimaryHDU(header=primary_header)]
    psf_header = fits.Header()
    psf_header["BUNIT"] = "fraction"
    psf_header["INTEGRAL"] = float(psf.sum())
    hdus.append(fits.ImageHDU(psf.astype(np.float32), header=psf_header, name="PSF"))
    extension_definitions = [
        ("stellar_intrinsic", "STARIN", "nJy pixel-1"),
        ("agn_intrinsic", "AGNIN", "nJy pixel-1"),
        ("dust_surface_density", "DUST", "g cm-2"),
        ("tau_abs", "TAUA", "dimensionless"),
        ("tau_sca", "TAUS", "dimensionless"),
        ("stellar_direct", "STARDIR", "nJy pixel-1"),
        ("agn_direct", "AGNDIR", "nJy pixel-1"),
        ("stellar_scattered", "STARSCAT", "nJy pixel-1"),
        ("agn_scattered", "AGNSCAT", "nJy pixel-1"),
        ("total_absorbed", "ABSORB", "nJy pixel-1"),
        ("total_emergent", "EMERG", "nJy pixel-1"),
        ("stellar_observed", "STAROBS", "nJy pixel-1"),
        ("agn_observed", "AGNOBS", "nJy pixel-1"),
        ("total_observed", "TOTALOBS", "nJy pixel-1"),
    ]
    for panel, layer in zip(panels, layers, strict=True):
        common = fits.Header()
        common["PANEL"] = panel["panel"]
        common["SINKPRI"] = panel["primary_sink_id"]
        common["SINKSEC"] = panel["secondary_sink_id"]
        common["R3DPKPC"] = panel["separation_3d_pkpc"]
        for key, prefix, unit in extension_definitions:
            header = common.copy()
            header["BUNIT"] = unit
            hdus.append(
                fits.ImageHDU(
                    layer[key].astype(np.float32),
                    header=header,
                    name=f"{prefix}_{panel['panel'].upper()}",
                )
            )
    fits.HDUList(hdus).writeto(args.output_fits, overwrite=True, checksum=True)

    metadata = {
        "status": "complete",
        "scientific_status": "foreground-screen and image-plane single-scattering quick look",
        "output_number": int(targets["output_number"].iloc[0]),
        "redshift": redshift,
        "cosmology": {
            "H0_km_s_mpc": info["H0"],
            "omega_m": info["omega_m"],
            "luminosity_distance_cm": luminosity_distance_cm,
        },
        "projection": {
            "line_of_sight": "simulation z axis",
            "field_pkpc": field_pkpc,
            "image_size_pixel": args.image_size,
            "pixel_scale_arcsec": pixel_scale_arcsec,
            "pixel_scale_pkpc": pixel_pkpc,
        },
        "stellar_emission": {
            "model": "FSPS simple stellar populations evaluated at each particle age and metallicity",
            "formed_mass_source": "RAMSES initial stellar mass and output unit conversion",
            "fsps": fsps_metadata,
            "particle_grid_clipping": clipping,
            "stellar_smoothing_pkpc": args.stellar_smoothing_pkpc,
        },
        "agn_emission": {
            "model": "unobscured power law normalized at rest-frame 5100 Angstrom",
            "alpha_nu": args.agn_alpha_nu,
            "bolometric_correction_5100": args.agn_bolometric_correction_5100,
            "reference": AGN_SED_REFERENCE,
            "caveat": "single-template preview without a torus, variability, or orientation distribution",
        },
        "throughput": {
            "source_file": str(args.throughput),
            "sha256": sha256(args.throughput),
            "pivot_wavelength_observed_micron": throughput.pivot_wavelength_micron,
            "pivot_wavelength_rest_micron": rest_pivot_micron,
            "integration": "photon-weighted mean Fnu",
            "reference": F200W_THROUGHPUT_REFERENCE,
        },
        "dust": {
            "model": "Draine Milky Way R_V=3.1 carbonaceous-silicate mixture",
            "source_file": str(args.dust_curve),
            "sha256": sha256(args.dust_curve),
            "dust_to_metal_ratio": args.dust_to_metal_ratio,
            "maximum_temperature_measure_k_per_mu": args.maximum_dust_temperature_k_per_mu,
            "temperature_measure": "temperature_code divided by density_code",
            "projected_cell_model": "axis-aligned square top hats rounded upward to the detector-pixel scale",
            "kappa_abs_cm2_g_at_rest_pivot": kappa_abs,
            "kappa_sca_cm2_g_at_rest_pivot": kappa_sca,
            "reference": DRAINE_DUST_REFERENCE,
        },
        "transfer": {
            "geometry": "total projected dust column placed in a foreground screen",
            "direct": "I0 exp[-(tau_abs+tau_sca)]",
            "scattering": "single-interaction budget redistributed by a normalized image-plane Gaussian",
            "scattering_sigma_pixel": args.scattering_sigma_pixel,
            "caveat": "the screen does not retain the relative line-of-sight depths of sources and dust",
            "permitted_use": "data-flow, flux-budget, and morphology preview",
            "excluded_claims": [
                "physical obscured dual-AGN fraction",
                "viewing-angle distribution",
                "multiple-scattering color prediction",
            ],
        },
        "psf": {
            **psf_metadata,
            "source_file": str(args.psf),
            "sha256": sha256(args.psf),
            "application_order": "after dust transfer and observer-band integration",
        },
        "display": {
            "common_asinh_scale": True,
            "lower_njy_pixel": display_lower,
            "upper_njy_pixel": display_upper,
            "display_transform_applied_to_fits": False,
        },
        "panels": panels,
        "outputs": {
            "pdf": str(args.output_pdf),
            "png": str(args.output_png),
            "fits": str(args.output_fits),
            "metadata": str(args.metadata),
        },
    }
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "complete",
                "output_fits": str(args.output_fits),
                "panels": panels,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

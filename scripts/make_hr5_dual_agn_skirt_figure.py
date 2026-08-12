#!/usr/bin/env python3
"""Render the six HR5 dual-AGN F200W images from SKIRT products."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from matplotlib.colors import LinearSegmentedColormap

from fdm_smbh_delay.hr5_mock_observation import convolve_with_psf, load_psf


COMPONENTS = ("transparent", "primarydirect", "primaryscattered", "total")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--psf", type=Path, required=True)
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--output-fits", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--field-pkpc", type=float, default=55.5086998969992)
    parser.add_argument("--arrow-length-pkpc", type=float, default=3.5)
    parser.add_argument("--arrow-tip-gap-pixels", type=float, default=25.0)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--dimensionless-hubble", type=float, default=0.684)
    parser.add_argument("--omega-matter", type=float, default=0.3)
    parser.add_argument("--angular-scale-arcsec", type=float, default=1.0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_skirt_image(path: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path) as hdus:
        data = np.asarray(hdus[0].data, dtype=np.float64)
        header = hdus[0].header.copy()
    if data.ndim == 3 and data.shape[0] == 1:
        data = data[0]
    if data.ndim != 2 or not np.all(np.isfinite(data)):
        raise ValueError(f"Invalid SKIRT image: {path}")
    return data, header


def arrow_geometry(
    points_xy: np.ndarray, length: float, tip_gap: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return opposite arrow tails perpendicular to the projected pair axis."""

    points = np.asarray(points_xy, dtype=np.float64)
    if points.shape != (2, 2) or length <= 0.0 or tip_gap < 0.0:
        raise ValueError("Two AGN positions, positive length, and non-negative gap are required")
    pair_axis = points[1] - points[0]
    norm = float(np.linalg.norm(pair_axis))
    if norm == 0.0:
        raise ValueError("The projected AGN positions coincide")
    perpendicular = np.array([-pair_axis[1], pair_axis[0]]) / norm
    sides = np.vstack((perpendicular, -perpendicular))
    tips = points + tip_gap * sides
    tails = tips + length * sides
    return tails, tips


def display_scale(images: list[np.ndarray]) -> tuple[list[np.ndarray], float, float]:
    positive = np.concatenate([image[image > 0.0] for image in images])
    if not positive.size:
        raise ValueError("All SKIRT images are empty")
    lower = float(np.percentile(positive, 1.0))
    upper = float(np.percentile(positive, 99.99))
    lower = max(lower, upper * 1.0e-6)
    log_lower = np.log10(lower)
    log_upper = np.log10(upper)
    scaled = [
        np.clip(
            (np.log10(np.clip(image, lower, upper)) - log_lower) / (log_upper - log_lower),
            0.0,
            1.0,
        )
        for image in images
    ]
    return scaled, lower, upper


def latex_scientific(value: float) -> str:
    if value <= 0.0 or not np.isfinite(value):
        raise ValueError("A positive finite luminosity is required")
    exponent = int(np.floor(np.log10(value)))
    mantissa = value / 10.0**exponent
    return rf"{mantissa:.1f}\times10^{{{exponent}}}"


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    run_manifest_path = args.run_directory / "hr5_dual_agn_skirt_dust_run.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    panels = manifest["panels"]
    if [str(panel["panel"]) for panel in panels] != list("abcdef"):
        raise ValueError("Input manifest must contain panels a through f")
    redshift = float(manifest["redshift"])
    cosmology = FlatLambdaCDM(
        H0=100.0 * args.dimensionless_hubble,
        Om0=args.omega_matter,
    )
    pkpc_per_arcsec = float(
        cosmology.kpc_proper_per_arcmin(redshift).value / 60.0
    )
    angular_scale_pkpc = args.angular_scale_arcsec * pkpc_per_arcsec

    psf, psf_scale, psf_metadata = load_psf(args.psf)
    panel_layers: list[dict[str, np.ndarray]] = []
    panel_headers: list[fits.Header] = []
    panel_metadata: list[dict[str, object]] = []
    for panel in panels:
        name = str(panel["panel"])
        output_directory = args.run_directory / f"panel_{name}_dust"
        layers: dict[str, np.ndarray] = {}
        headers: dict[str, fits.Header] = {}
        input_files: dict[str, dict[str, object]] = {}
        for component in COMPONENTS:
            path = output_directory / f"panel_{name}_dust_f200w_{component}.fits"
            image, header = read_skirt_image(path)
            layers[component] = image
            headers[component] = header
            input_files[component] = {"path": str(path), "sha256": sha256(path)}
        if len({layer.shape for layer in layers.values()}) != 1:
            raise ValueError(f"SKIRT component dimensions differ in panel {name}")
        if not np.isclose(float(headers["total"]["CDELT1"]), psf_scale, atol=1.0e-8):
            raise ValueError(f"SKIRT and PSF pixel scales differ in panel {name}")

        layers["observed"] = np.clip(convolve_with_psf(layers["total"], psf), 0.0, None)
        closure_denominator = max(float(np.sum(np.abs(layers["total"]))), np.finfo(float).tiny)
        closure = float(
            np.sum(np.abs(layers["total"] - layers["primarydirect"] - layers["primaryscattered"]))
            / closure_denominator
        )

        agn_path = Path(panel["files"]["agn"]["path"])
        agn = np.loadtxt(agn_path, comments="#", ndmin=2)
        points = agn[:, :2] / 1000.0
        arrow_tip_gap_pkpc = args.arrow_tip_gap_pixels * args.field_pkpc / args.image_size
        tails, tips = arrow_geometry(points, args.arrow_length_pkpc, arrow_tip_gap_pkpc)
        pair_axis = points[1] - points[0]
        arrow_dot_products = [
            float(np.dot(pair_axis, tips[index] - tails[index])) for index in range(2)
        ]
        arrow_vectors = tips - tails
        opposition_cosine = float(
            np.dot(arrow_vectors[0], arrow_vectors[1])
            / (np.linalg.norm(arrow_vectors[0]) * np.linalg.norm(arrow_vectors[1]))
        )
        separation_3d = float(np.linalg.norm(agn[1, :3] - agn[0, :3]) / 1000.0)
        separation_projected = float(np.linalg.norm(pair_axis))
        image_y, image_x = np.indices(layers["observed"].shape)
        outside_agn = np.ones(layers["observed"].shape, dtype=bool)
        agn_peaks: list[float] = []
        for x_pkpc, y_pkpc in points:
            x_pixel = (x_pkpc / args.field_pkpc + 0.5) * args.image_size - 0.5
            y_pixel = (y_pkpc / args.field_pkpc + 0.5) * args.image_size - 0.5
            radius_squared = (image_x - x_pixel) ** 2 + (image_y - y_pixel) ** 2
            agn_peaks.append(float(layers["observed"][radius_squared <= 1.5**2].max()))
            outside_agn &= radius_squared > 8.0**2
        brightest_non_agn = float(layers["observed"][outside_agn].max())
        panel_layers.append(layers)
        panel_headers.append(headers["total"])
        panel_metadata.append(
            {
                "panel": name,
                "primary_sink_id": int(panel["primary_sink_id"]),
                "secondary_sink_id": int(panel["secondary_sink_id"]),
                "separation_3d_pkpc": separation_3d,
                "separation_projected_pkpc": separation_projected,
                "agn_xy_pkpc": points.tolist(),
                "agn_lbol_erg_s": agn[:, 3].tolist(),
                "agn_peak_mjy_sr": agn_peaks,
                "brightest_outside_agn_mjy_sr": brightest_non_agn,
                "agn_peak_to_brightest_outside_agn": [
                    peak / brightest_non_agn for peak in agn_peaks
                ],
                "arrow_tails_xy_pkpc": tails.tolist(),
                "arrow_tips_xy_pkpc": tips.tolist(),
                "arrow_pair_axis_dot_products_pkpc2": arrow_dot_products,
                "arrow_opposition_cosine": opposition_cosine,
                "component_closure_l1_relative": closure,
                "input_files": input_files,
            }
        )

    observed = [layers["observed"] for layers in panel_layers]
    normalized, display_lower, display_upper = display_scale(observed)
    colormap = LinearSegmentedColormap.from_list(
        "f200w_skirt",
        ["#01030a", "#0b1730", "#473040", "#a45b3d", "#f3bd72", "#fff6d8"],
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
    half_field = args.field_pkpc / 2.0
    extent = (-half_field, half_field, -half_field, half_field)
    for index, (axis, panel, image, meta) in enumerate(
        zip(axes.flat, panels, normalized, panel_metadata, strict=True)
    ):
        axis.imshow(colormap(image), origin="lower", extent=extent, interpolation="nearest")
        tails = np.asarray(meta["arrow_tails_xy_pkpc"])
        tips = np.asarray(meta["arrow_tips_xy_pkpc"])
        for arrow_number, (tail, tip) in enumerate(zip(tails, tips, strict=True), start=1):
            annotation = axis.annotate(
                "",
                xy=tip,
                xytext=tail,
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": "#e9fbff",
                    "linewidth": 1.05,
                    "mutation_scale": 8.5,
                    "shrinkA": 0.0,
                    "shrinkB": 0.0,
                },
                annotation_clip=True,
                zorder=8,
            )
            annotation.arrow_patch.set_path_effects(
                [path_effects.Stroke(linewidth=2.4, foreground="#02040a"), path_effects.Normal()]
            )
            axis.text(
                tail[0],
                tail[1],
                str(arrow_number),
                ha="center",
                va="center",
                color="white",
                fontsize=5.6,
                fontweight="bold",
                bbox={
                    "boxstyle": "circle,pad=0.16",
                    "facecolor": "#02040a",
                    "edgecolor": "#e9fbff",
                    "linewidth": 0.65,
                    "alpha": 0.92,
                },
                clip_on=True,
                zorder=9,
            )
        axis.text(
            0.025,
            0.965,
            f"({panel['panel']})",
            transform=axis.transAxes,
            ha="left",
            va="top",
            color="white",
            fontsize=7.0,
            path_effects=[path_effects.withStroke(linewidth=1.5, foreground="black")],
        )
        axis.text(
            0.975,
            0.965,
            rf"$r_{{\rm 3D}}={meta['separation_3d_pkpc']:.1f}$ pkpc",
            transform=axis.transAxes,
            ha="right",
            va="top",
            color="white",
            fontsize=6.2,
            bbox={"facecolor": "#000000", "edgecolor": "none", "alpha": 0.38, "pad": 1.4},
        )
        luminosities = meta["agn_lbol_erg_s"]
        axis.text(
            0.025,
            0.035,
            rf"$L_{{\rm bol,1}}={latex_scientific(float(luminosities[0]))}$"
            "\n"
            rf"$L_{{\rm bol,2}}={latex_scientific(float(luminosities[1]))}$"
            "\n"
            r"$[\mathrm{erg\,s^{-1}}]$",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            color="white",
            fontsize=5.4,
            linespacing=1.05,
            bbox={"facecolor": "#000000", "edgecolor": "none", "alpha": 0.38, "pad": 1.4},
        )
        scale_x1, scale_y = 24.0, -23.5
        scale_x0 = scale_x1 - angular_scale_pkpc
        axis.plot(
            [scale_x0, scale_x1],
            [scale_y, scale_y],
            color="white",
            linewidth=1.4,
            path_effects=[path_effects.Stroke(linewidth=2.4, foreground="black"), path_effects.Normal()],
        )
        axis.text(
            0.5 * (scale_x0 + scale_x1),
            scale_y + 1.1,
            rf"${args.angular_scale_arcsec:g}^{{\prime\prime}}$",
            color="white",
            fontsize=6.2,
            ha="center",
            va="bottom",
            path_effects=[path_effects.withStroke(linewidth=1.4, foreground="black")],
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
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.085, top=0.995, wspace=0.035, hspace=0.055)

    for output in (args.output_pdf, args.output_png, args.output_fits, args.metadata):
        output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_pdf, dpi=300)
    fig.savefig(args.output_png, dpi=300)
    plt.close(fig)

    primary = fits.Header()
    primary["REDSHIFT"] = float(manifest["redshift"])
    primary["FILTER"] = "F200W"
    primary["BUNIT"] = "MJy/sr"
    primary["PIXSCALE"] = (psf_scale, "arcsec pixel-1")
    primary["FIELDPK"] = (args.field_pkpc, "proper kpc")
    primary["KPCPARC"] = (pkpc_per_arcsec, "proper kpc arcsec-1")
    primary["DUSTRT"] = True
    primary["PSFCONV"] = True
    primary["ARROWS"] = "perpendicular"
    hdus: list[fits.PrimaryHDU | fits.ImageHDU] = [fits.PrimaryHDU(header=primary)]
    psf_header = fits.Header()
    psf_header["BUNIT"] = "fraction"
    psf_header["INTEGRAL"] = float(psf.sum())
    hdus.append(fits.ImageHDU(psf.astype(np.float32), header=psf_header, name="PSF"))
    extension_names = {
        "transparent": "TRANSP",
        "primarydirect": "DIRECT",
        "primaryscattered": "SCATTER",
        "total": "TOTAL",
        "observed": "OBSERVED",
    }
    for panel, layers, source_header, meta in zip(
        panels, panel_layers, panel_headers, panel_metadata, strict=True
    ):
        for component, prefix in extension_names.items():
            header = source_header.copy()
            header["PANEL"] = str(panel["panel"])
            header["SINKPRI"] = int(panel["primary_sink_id"])
            header["SINKSEC"] = int(panel["secondary_sink_id"])
            header["R3DPKPC"] = float(meta["separation_3d_pkpc"])
            header["R2DPKPC"] = float(meta["separation_projected_pkpc"])
            hdus.append(
                fits.ImageHDU(
                    layers[component].astype(np.float32),
                    header=header,
                    name=f"{prefix}_{str(panel['panel']).upper()}",
                )
            )
    fits.HDUList(hdus).writeto(args.output_fits, overwrite=True, checksum=True)

    metadata = {
        "status": "complete",
        "scientific_status": "three-dimensional Monte Carlo dust transfer followed by PSF convolution",
        "redshift": float(manifest["redshift"]),
        "filter": "JWST/NIRCam F200W",
        "observer_direction": "simulation z axis",
        "agn_angular_emission": run_manifest["agn_angular_emission"],
        "photon_packets_per_panel": run_manifest["packets"],
        "dust_transfer": {
            "engine": "SKIRT 9",
            "component_identity": "total = direct primary emission + scattered primary emission",
            "thermal_dust_emission": False,
            "reason": "observer-frame F200W samples rest-frame near-infrared emission at this redshift",
        },
        "psf": {
            "path": str(args.psf),
            "sha256": sha256(args.psf),
            "pixel_scale_arcsec": psf_scale,
            "normalized_sum": float(psf.sum()),
            **psf_metadata,
        },
        "display": {
            "field_pkpc": args.field_pkpc,
            "pkpc_per_arcsec": pkpc_per_arcsec,
            "angular_scale_bar_arcsec": args.angular_scale_arcsec,
            "angular_scale_bar_pkpc": angular_scale_pkpc,
            "lower_positive_percentile_mjy_sr": display_lower,
            "upper_positive_percentile_mjy_sr": display_upper,
            "transform": "base-10 logarithm",
            "lower_percentile": 1.0,
            "upper_percentile": 99.99,
            "common_scaling_for_all_panels": True,
        },
        "agn_markers": {
            "form": "arrows only",
            "orientation": "perpendicular to the projected line joining the two AGN",
            "direction": "opposite arrow directions, each ending at one AGN",
            "length_pkpc": args.arrow_length_pkpc,
            "tip_gap_pixels": args.arrow_tip_gap_pixels,
            "tip_gap_pkpc": args.arrow_tip_gap_pixels * args.field_pkpc / args.image_size,
            "agn_position_circles": False,
            "numbered_tail_labels": True,
        },
        "panels": panel_metadata,
        "outputs": {
            "pdf": str(args.output_pdf),
            "png": str(args.output_png),
            "fits": str(args.output_fits),
        },
    }
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate the PSF response used by the HR5 dual-AGN mock image."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.colors import LogNorm

from fdm_smbh_delay.hr5_mock_observation import delta_source_diagnostic, load_psf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--psf", type=Path, required=True)
    parser.add_argument("--psf-extension", default="DET_SAMP")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-fits", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    psf, pixel_scale_arcsec, psf_metadata = load_psf(args.psf, args.psf_extension)
    metrics, layers = delta_source_diagnostic(
        psf, image_shape=(args.image_size, args.image_size)
    )
    metrics["fwhm_x_arcsec"] = float(metrics["fwhm_x_pixel"]) * pixel_scale_arcsec
    metrics["fwhm_y_arcsec"] = float(metrics["fwhm_y_pixel"]) * pixel_scale_arcsec
    report = {
        "status": "pass"
        if float(metrics["maximum_absolute_residual"]) < 1.0e-12
        else "fail",
        "psf": {
            **psf_metadata,
            "source_file": str(args.psf),
            "sha256": sha256(args.psf),
        },
        "delta_source": metrics,
    }
    if report["status"] != "pass":
        raise RuntimeError("The delta-source PSF regression test failed")

    for output in (args.output_json, args.output_fits, args.output_png):
        output.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    primary = fits.Header()
    primary["PSFFILE"] = args.psf.name
    primary["PSFSHA"] = report["psf"]["sha256"]
    primary["PIXSCALE"] = (pixel_scale_arcsec, "arcsec pixel-1")
    primary["INFLUX"] = 1.0
    primary["OUTFLUX"] = metrics["cropped_output_flux"]
    primary["MAXRESID"] = metrics["maximum_absolute_residual"]
    hdus: list[fits.PrimaryHDU | fits.ImageHDU] = [fits.PrimaryHDU(header=primary)]
    for name in ("delta", "convolved", "expected", "residual"):
        hdus.append(fits.ImageHDU(layers[name].astype(np.float32), name=name.upper()))
    fits.HDUList(hdus).writeto(args.output_fits, overwrite=True, checksum=True)

    floor = max(float(layers["convolved"].max()) * 1.0e-7, 1.0e-15)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45))
    axes[0].imshow(
        np.clip(layers["convolved"], floor, None),
        origin="lower",
        norm=LogNorm(vmin=floor, vmax=float(layers["convolved"].max())),
        cmap="magma",
    )
    axes[0].set_title("Unit point source")
    center_y = float(metrics["centroid_y_pixel"])
    center_x = float(metrics["centroid_x_pixel"])
    yy, xx = np.indices(layers["convolved"].shape, dtype=np.float64)
    radius = np.hypot(xx - center_x, yy - center_y)
    bins = np.arange(0.0, 65.0, 0.5)
    indices = np.digitize(radius.ravel(), bins)
    radial = np.array(
        [
            np.mean(layers["convolved"].ravel()[indices == index])
            if np.any(indices == index)
            else np.nan
            for index in range(1, len(bins))
        ]
    )
    axes[1].semilogy(0.5 * (bins[:-1] + bins[1:]), radial, color="#2455a4")
    axes[1].set_xlabel("Radius (pixel)")
    axes[1].set_ylabel("Mean normalized intensity")
    axes[1].set_ylim(floor, float(layers["convolved"].max()) * 1.5)
    residual_limit = max(float(np.max(np.abs(layers["residual"]))), 1.0e-18)
    axes[2].imshow(
        layers["residual"],
        origin="lower",
        cmap="coolwarm",
        vmin=-residual_limit,
        vmax=residual_limit,
    )
    axes[2].set_title("Convolution residual")
    for axis in axes:
        axis.tick_params(labelsize=7)
        if axis is not axes[1]:
            axis.set_xlim(args.image_size / 2 - 16, args.image_size / 2 + 16)
            axis.set_ylim(args.image_size / 2 - 16, args.image_size / 2 + 16)
    fig.tight_layout()
    fig.savefig(args.output_png, dpi=250)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

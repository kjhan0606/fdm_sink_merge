#!/usr/bin/env python3
"""Validate calibrated HR5 F200W intermediate products and flux budgets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from astropy.io import fits

from fdm_smbh_delay.hr5_mock_observation import (
    apply_foreground_screen,
    delta_source_diagnostic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fits", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def close(first: float, second: float, rtol: float = 2.0e-5) -> bool:
    return bool(np.isclose(first, second, rtol=rtol, atol=1.0e-8))


def main() -> None:
    args = parse_args()
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    failures: list[str] = []
    checks: dict[str, object] = {}
    with fits.open(args.fits, checksum=True) as hdus:
        checksum_ok = all(
            hdu.verify_checksum() == 1 for hdu in hdus if "CHECKSUM" in hdu.header
        )
        checks["fits_checksums"] = checksum_ok
        if not checksum_ok:
            failures.append("A FITS checksum failed")
        psf = np.asarray(hdus["PSF"].data, dtype=np.float64)
        psf /= psf.sum()
        psf_metrics, _ = delta_source_diagnostic(
            psf, image_shape=hdus["TOTALOBS_A"].data.shape
        )
        checks["delta_source"] = psf_metrics
        if float(psf_metrics["maximum_absolute_residual"]) >= 1.0e-12:
            failures.append("The stored PSF fails the delta-source test")

        panel_checks: dict[str, object] = {}
        for panel_metadata in metadata["panels"]:
            panel = panel_metadata["panel"].upper()
            required = [
                f"STARIN_{panel}",
                f"AGNIN_{panel}",
                f"DUST_{panel}",
                f"TAUA_{panel}",
                f"TAUS_{panel}",
                f"STARDIR_{panel}",
                f"AGNDIR_{panel}",
                f"STARSCAT_{panel}",
                f"AGNSCAT_{panel}",
                f"ABSORB_{panel}",
                f"EMERG_{panel}",
                f"STAROBS_{panel}",
                f"AGNOBS_{panel}",
                f"TOTALOBS_{panel}",
            ]
            missing = [name for name in required if name not in hdus]
            if missing:
                failures.append(f"Panel {panel} lacks {missing}")
                continue
            arrays = {
                name: np.asarray(hdus[name].data, dtype=np.float64) for name in required
            }
            if any(np.any(~np.isfinite(array)) or np.any(array < 0.0) for array in arrays.values()):
                failures.append(f"Panel {panel} contains a negative or non-finite value")
            combined_matches = np.allclose(
                arrays[f"TOTALOBS_{panel}"],
                arrays[f"STAROBS_{panel}"] + arrays[f"AGNOBS_{panel}"],
                rtol=1.0e-6,
                atol=1.0e-5,
            )
            emergent_matches = np.allclose(
                arrays[f"EMERG_{panel}"],
                arrays[f"STARDIR_{panel}"]
                + arrays[f"AGNDIR_{panel}"]
                + arrays[f"STARSCAT_{panel}"]
                + arrays[f"AGNSCAT_{panel}"],
                rtol=1.0e-6,
                atol=1.0e-5,
            )
            if not combined_matches:
                failures.append(f"Panel {panel} combined observed layer is inconsistent")
            if not emergent_matches:
                failures.append(f"Panel {panel} emergent layer is inconsistent")

            budgets_ok = True
            for component, intrinsic_name, observed_name in (
                ("stellar_budget", f"STARIN_{panel}", f"STAROBS_{panel}"),
                ("agn_budget", f"AGNIN_{panel}", f"AGNOBS_{panel}"),
            ):
                budget = panel_metadata[component]
                closure = (
                    budget["direct_njy"]
                    + budget["absorbed_njy"]
                    + budget["scattered_before_field_crop_njy"]
                )
                budgets_ok &= close(budget["intrinsic_njy"], closure, rtol=1.0e-11)
                budgets_ok &= close(
                    float(arrays[intrinsic_name].sum()), budget["intrinsic_njy"]
                )
                budgets_ok &= close(
                    float(arrays[observed_name].sum()),
                    budget["observed_after_psf_field_crop_njy"],
                )
            if not budgets_ok:
                failures.append(f"Panel {panel} has an inconsistent flux budget")

            intrinsic = arrays[f"STARIN_{panel}"] + arrays[f"AGNIN_{panel}"]
            zeros = np.zeros_like(intrinsic)
            zero_dust = apply_foreground_screen(intrinsic, zeros, zeros, 2.0)
            zero_dust_ok = np.array_equal(zero_dust.emergent, intrinsic)
            zero_scattering = apply_foreground_screen(
                intrinsic, arrays[f"TAUA_{panel}"], zeros, 2.0
            )
            zero_scattering_ok = (
                float(zero_scattering.scattered_in_field.sum()) == 0.0
                and np.array_equal(zero_scattering.emergent, zero_scattering.direct)
            )
            if not zero_dust_ok:
                failures.append(f"Panel {panel} fails the zero-dust gate")
            if not zero_scattering_ok:
                failures.append(f"Panel {panel} fails the zero-scattering gate")
            panel_checks[panel.lower()] = {
                "combined_observed_reproducible": combined_matches,
                "emergent_layers_reproducible": emergent_matches,
                "flux_budgets_close": budgets_ok,
                "zero_dust_returns_intrinsic": zero_dust_ok,
                "zero_scattering_returns_absorption_only": zero_scattering_ok,
            }
        checks["panels"] = panel_checks

    report = {
        "status": "pass" if not failures else "fail",
        "fits": str(args.fits),
        "metadata": str(args.metadata),
        "checks": checks,
        "failures": failures,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

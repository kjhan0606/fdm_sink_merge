#!/usr/bin/env python3
"""Measure host-confirmed dual AGN abundances and their redshift evolution."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from fdm_smbh_delay.hr5 import (
    find_agn_pair_population,
    fit_redshift_rate,
    read_mkagn_snapshot,
    redshift_rate_model,
    spatial_jackknife_pair_statistics,
)


DEFAULT_AGN_DIRECTORY = Path(
    "/home/kjhan/BACKUP/GalFinder/"
    "SRC(FoF_PSB_Free_Ver2_Dev)/SRC(AGN)/SRC(MkAGN)/HR5_AGN_DATA"
)
DEFAULT_CANONICAL_ROOT = Path(
    "/scratch/kjhan/Hydro/HR5/FoFPSB/02_without_dc/Ver2/"
    "Derived_Sink_Hosts/canonical_v1"
)
DEFAULT_OUTPUT_DIRECTORY = Path("results/hr5/dual_agn")
FIDUCIAL_VOLUME_CMPC3 = 1.087e7


def _pair_key(first: int, second: int) -> tuple[int, int]:
    """Return an order-independent pair identifier."""

    return (first, second) if first < second else (second, first)


def _read_host_pair_index(path: Path) -> dict[tuple[int, int], dict[str, str]]:
    result: dict[tuple[int, int], dict[str, str]] = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = _pair_key(
                int(row["primary_sink_id"]), int(row["secondary_sink_id"])
            )
            if key in result:
                raise ValueError(f"Duplicate active SMBH pair in {path}: {key}")
            result[key] = row
    return result


def _pair_midpoint_region(
    first_x: np.ndarray,
    second_x: np.ndarray,
    box_size: float,
    region_count: int,
) -> np.ndarray:
    delta = np.asarray(second_x, dtype=np.float64) - np.asarray(
        first_x, dtype=np.float64
    )
    delta -= box_size * np.rint(delta / box_size)
    midpoint = np.mod(np.asarray(first_x, dtype=np.float64) + 0.5 * delta, box_size)
    return np.minimum(
        (midpoint * region_count / box_size).astype(np.int64), region_count - 1
    )


def _jackknife_pair_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    pair_region: np.ndarray,
    region_count: int,
) -> tuple[float, float]:
    """Return a pair fraction and its delete-one-region jackknife error."""

    selected_numerator = np.asarray(numerator, dtype=bool)
    selected_denominator = np.asarray(denominator, dtype=bool)
    region = np.asarray(pair_region, dtype=np.int64)
    if (
        selected_numerator.shape != selected_denominator.shape
        or selected_numerator.shape != region.shape
    ):
        raise ValueError("Pair selections and regions must have matching shapes")
    if np.any(selected_numerator & ~selected_denominator):
        raise ValueError("The numerator must be a subset of the denominator")
    total_denominator = int(np.count_nonzero(selected_denominator))
    total_numerator = int(np.count_nonzero(selected_numerator))
    fraction = (
        total_numerator / total_denominator if total_denominator else float("nan")
    )
    denominator_by_region = np.bincount(
        region[selected_denominator], minlength=region_count
    )
    numerator_by_region = np.bincount(
        region[selected_numerator], minlength=region_count
    )
    retained_denominator = total_denominator - denominator_by_region
    samples = np.divide(
        total_numerator - numerator_by_region,
        retained_denominator,
        out=np.full(region_count, np.nan),
        where=retained_denominator > 0,
    )
    finite = samples[np.isfinite(samples)]
    if finite.size < 2:
        return fraction, float("nan")
    error = np.sqrt(
        (finite.size - 1.0)
        / finite.size
        * np.sum((finite - np.mean(finite)) ** 2)
    )
    return fraction, float(error)


def _fit_population(
    redshift: np.ndarray,
    density: np.ndarray,
    count: np.ndarray,
    population: str,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    fitted = fit_redshift_rate(redshift, density, count)
    if not fitted.success:
        raise ValueError(f"Modified Schechter fit failed for {population}")
    usable = (count >= 3) & np.isfinite(density) & (density > 0.0)
    evaluation = np.expm1(
        np.linspace(
            np.log1p(float(np.min(redshift[usable]))),
            np.log1p(float(np.max(redshift[usable]))),
            500,
        )
    )
    curve = redshift_rate_model(
        evaluation,
        fitted.phi_star,
        fitted.z_star,
        fitted.alpha,
        fitted.beta,
    )
    measured = redshift_rate_model(
        redshift[usable],
        fitted.phi_star,
        fitted.z_star,
        fitted.alpha,
        fitted.beta,
    )
    residual = np.log10(measured / density[usable])
    parameters: dict[str, object] = {
        "population": population,
        "phi_star_cmpc3": fitted.phi_star,
        "z_star": fitted.z_star,
        "alpha": fitted.alpha,
        "beta": fitted.beta,
        "fit_point_count": fitted.n_bin,
        "minimum_fit_redshift": float(np.min(redshift[usable])),
        "maximum_fit_redshift": float(np.max(redshift[usable])),
        "rms_log10_residual": float(np.sqrt(np.mean(residual**2))),
        "maximum_absolute_log10_residual": float(np.max(np.abs(residual))),
    }
    return parameters, evaluation, curve


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(
    path: Path,
    rows: list[dict[str, object]],
    fits: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 7.0,
            "mathtext.fontset": "stix",
            "axes.labelsize": 7.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "pdf.fonttype": 42,
        }
    )
    ordered = sorted(rows, key=lambda row: float(row["redshift"]))
    redshift = np.asarray([row["redshift"] for row in ordered], dtype=np.float64)
    figure, axes = plt.subplots(1, 2, figsize=(7.15, 2.65))
    figure.subplots_adjust(wspace=0.31, bottom=0.18, left=0.09, right=0.98, top=0.97)
    density_specs = (
        (
            "spatial_pair",
            "spatial_pair_count",
            "spatial_pair_number_density_cmpc3",
            "spatial_pair_number_density_jackknife_error_cmpc3",
            "all spatial pairs",
            "#777777",
            "s",
        ),
        (
            "distinct_host",
            "distinct_host_pair_count",
            "distinct_host_number_density_cmpc3",
            "distinct_host_number_density_jackknife_error_cmpc3",
            "pairs in distinct PSB galaxies",
            "#0072B2",
            "o",
        ),
    )
    for population, count_field, value_field, error_field, label, color, marker in density_specs:
        count = np.asarray([row[count_field] for row in ordered], dtype=np.int64)
        value = np.asarray([row[value_field] for row in ordered], dtype=np.float64)
        error = np.asarray([row[error_field] for row in ordered], dtype=np.float64)
        detected = count > 0
        axes[0].errorbar(
            redshift[detected],
            value[detected],
            yerr=error[detected],
            fmt=marker,
            ms=4.0,
            mfc="white",
            color=color,
            lw=0.8,
            capsize=1.4,
            ls="none",
            label=label,
        )
        fit_redshift, fit_density = fits[population]
        axes[0].plot(fit_redshift, fit_density, color=color, lw=1.1)

    fraction_specs = (
        (
            "distinct_host_fraction",
            "distinct_host_fraction_jackknife_error",
            "fiducial",
            "#0072B2",
            "o",
        ),
        (
            "fable_distinct_host_fraction",
            "fable_distinct_host_fraction_jackknife_error",
            "FABLE mass limits",
            "#D55E00",
            "s",
        ),
        (
            "hr5_100_star_distinct_host_fraction",
            "hr5_100_star_distinct_host_fraction_jackknife_error",
            r"$N_\star\geq100$ per host",
            "#009E73",
            "D",
        ),
    )
    for value_field, error_field, label, color, marker in fraction_specs:
        value = np.asarray([row[value_field] for row in ordered], dtype=np.float64)
        error = np.asarray([row[error_field] for row in ordered], dtype=np.float64)
        usable = np.isfinite(value)
        axes[1].errorbar(
            redshift[usable],
            value[usable],
            yerr=error[usable],
            fmt=marker,
            ms=4.0,
            mfc="white",
            color=color,
            lw=0.8,
            capsize=1.4,
            ls="none",
            label=label,
        )

    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"$z$")
    axes[0].set_ylabel(r"$n_{\rm pair}$ [cMpc$^{-3}$]")
    axes[0].legend(frameon=False, loc="lower right", handlelength=1.0)
    axes[1].set_xlabel(r"$z$")
    axes[1].set_ylabel("fraction in distinct PSB galaxies")
    axes[1].set_ylim(-0.04, 1.04)
    axes[1].legend(frameon=False, loc="lower right", handlelength=1.0)
    for label, axis in zip(("(a)", "(b)"), axes, strict=True):
        axis.minorticks_on()
        axis.text(0.03, 0.96, label, transform=axis.transAxes, va="top")
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def analyze(
    agn_directory: Path,
    canonical_root: Path,
    output_directory: Path,
    dimensionless_hubble: float,
    box_size_cmpc_over_h: float,
    volume_cmpc3: float,
    region_count: int,
) -> dict[str, object]:
    evolution_path = canonical_root / "hr5_dual_agn_host_evolution.csv"
    with evolution_path.open(newline="") as stream:
        evolution = list(csv.DictReader(stream))
    rows: list[dict[str, object]] = []
    for evolution_row in evolution:
        output = int(evolution_row["output"])
        redshift, _, records = read_mkagn_snapshot(
            agn_directory / f"agn.{output:05d}.dat"
        )
        pairs = find_agn_pair_population(
            records,
            redshift,
            dimensionless_hubble,
            luminosity_threshold_erg_s=1.0e43,
            minimum_mass_msun=1.0e4,
            box_size_cmpc_over_h=box_size_cmpc_over_h,
        )
        dual = np.asarray(pairs["is_dual"], dtype=bool)
        host_index = _read_host_pair_index(
            canonical_root
            / f"output_{output:05d}"
            / f"hr5_dual_agn_hosts.{output:05d}.csv"
        )
        relation = np.full(dual.size, "not a dual AGN", dtype=object)
        fable = np.zeros(dual.size, dtype=bool)
        hr5_100_star = np.zeros(dual.size, dtype=bool)
        matched_keys: set[tuple[int, int]] = set()
        for index in np.flatnonzero(dual):
            key = _pair_key(int(pairs["id_1"][index]), int(pairs["id_2"][index]))
            if key not in host_index:
                raise ValueError(f"Host relation is absent for output {output}: {key}")
            host_row = host_index[key]
            matched_keys.add(key)
            relation[index] = host_row["host_relation"]
            fable[index] = host_row["fable_selection_analogue"] == "1"
            hr5_100_star[index] = host_row["hr5_100_star_particle_selection"] == "1"
        if matched_keys != set(host_index):
            raise ValueError(f"Host and MkAGN pair tables differ at output {output}")

        classifiable = dual & (relation != "no direct PSB assignment")
        distinct = classifiable & (relation != "same PSB galaxy")
        pair_region = _pair_midpoint_region(
            pairs["position_1_cmpc_over_h"][:, 0],
            pairs["position_2_cmpc_over_h"][:, 0],
            box_size_cmpc_over_h,
            region_count,
        )
        spatial = spatial_jackknife_pair_statistics(
            pairs["active_position_x_cmpc_over_h"],
            pairs["position_1_cmpc_over_h"][:, 0],
            pairs["position_2_cmpc_over_h"][:, 0],
            dual,
            volume_cmpc3,
            box_size_cmpc_over_h,
            region_count=region_count,
        )
        host_confirmed = spatial_jackknife_pair_statistics(
            pairs["active_position_x_cmpc_over_h"],
            pairs["position_1_cmpc_over_h"][:, 0],
            pairs["position_2_cmpc_over_h"][:, 0],
            distinct,
            volume_cmpc3,
            box_size_cmpc_over_h,
            region_count=region_count,
        )
        distinct_fraction, distinct_error = _jackknife_pair_ratio(
            distinct, classifiable, pair_region, region_count
        )
        fable_fraction, fable_error = _jackknife_pair_ratio(
            distinct & fable, classifiable & fable, pair_region, region_count
        )
        star_fraction, star_error = _jackknife_pair_ratio(
            distinct & hr5_100_star,
            classifiable & hr5_100_star,
            pair_region,
            region_count,
        )
        row: dict[str, object] = {
            "output": output,
            "redshift": redshift,
            "active_smbh_count": int(pairs["active_count"]),
            "spatial_pair_count": int(np.count_nonzero(dual)),
            "spatial_pair_number_density_cmpc3": float(spatial["number_density"]),
            "spatial_pair_number_density_jackknife_error_cmpc3": float(
                spatial["number_density_jackknife_error"]
            ),
            "pair_count_with_two_psb_hosts": int(np.count_nonzero(classifiable)),
            "same_host_pair_count": int(
                np.count_nonzero(classifiable & (relation == "same PSB galaxy"))
            ),
            "distinct_host_pair_count": int(np.count_nonzero(distinct)),
            "distinct_host_number_density_cmpc3": float(
                host_confirmed["number_density"]
            ),
            "distinct_host_number_density_jackknife_error_cmpc3": float(
                host_confirmed["number_density_jackknife_error"]
            ),
            "distinct_host_fraction": distinct_fraction,
            "distinct_host_fraction_jackknife_error": distinct_error,
            "fable_pair_count": int(np.count_nonzero(classifiable & fable)),
            "fable_distinct_host_pair_count": int(np.count_nonzero(distinct & fable)),
            "fable_distinct_host_fraction": fable_fraction,
            "fable_distinct_host_fraction_jackknife_error": fable_error,
            "hr5_100_star_pair_count": int(
                np.count_nonzero(classifiable & hr5_100_star)
            ),
            "hr5_100_star_distinct_host_pair_count": int(
                np.count_nonzero(distinct & hr5_100_star)
            ),
            "hr5_100_star_distinct_host_fraction": star_fraction,
            "hr5_100_star_distinct_host_fraction_jackknife_error": star_error,
        }
        rows.append(row)
        print(
            f"Output {output:05d}, z={redshift:.3f}: "
            f"{row['distinct_host_pair_count']}/{row['spatial_pair_count']} "
            "active pairs occupy distinct PSB galaxies",
            flush=True,
        )

    redshift = np.asarray([row["redshift"] for row in rows], dtype=np.float64)
    fit_results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    parameters: list[dict[str, object]] = []
    for population, count_field, density_field in (
        (
            "spatial_pair",
            "spatial_pair_count",
            "spatial_pair_number_density_cmpc3",
        ),
        (
            "distinct_host",
            "distinct_host_pair_count",
            "distinct_host_number_density_cmpc3",
        ),
    ):
        parameter, evaluation, curve = _fit_population(
            redshift,
            np.asarray([row[density_field] for row in rows], dtype=np.float64),
            np.asarray([row[count_field] for row in rows], dtype=np.int64),
            population,
        )
        parameters.append(parameter)
        fit_results[population] = (evaluation, curve)

    output_directory.mkdir(parents=True, exist_ok=True)
    measurement_path = output_directory / "hr5_dual_agn_host_demographics.csv"
    parameter_path = (
        output_directory
        / "hr5_dual_agn_host_modified_schechter_parameters.csv"
    )
    fit_path = output_directory / "hr5_dual_agn_host_redshift_fits.csv"
    figure_path = output_directory / "hr5_dual_agn_host_demographics.pdf"
    _write_rows(measurement_path, rows)
    _write_rows(parameter_path, parameters)
    fit_rows: list[dict[str, object]] = []
    for population, (evaluation, curve) in fit_results.items():
        fit_rows.extend(
            {
                "population": population,
                "redshift": float(z),
                "number_density_fit_cmpc3": float(value),
            }
            for z, value in zip(evaluation, curve, strict=True)
        )
    _write_rows(fit_path, fit_rows)
    _plot(figure_path, rows, fit_results)

    summary: dict[str, object] = {
        "output_count": len(rows),
        "spatial_pair_count": sum(int(row["spatial_pair_count"]) for row in rows),
        "pair_count_with_two_psb_hosts": sum(
            int(row["pair_count_with_two_psb_hosts"]) for row in rows
        ),
        "same_host_pair_count": sum(int(row["same_host_pair_count"]) for row in rows),
        "distinct_host_pair_count": sum(
            int(row["distinct_host_pair_count"]) for row in rows
        ),
        "spatial_jackknife_region_count": region_count,
        "volume_cmpc3": volume_cmpc3,
        "modified_schechter_parameters": parameters,
        "measurement_path": str(measurement_path),
        "fit_path": str(fit_path),
        "figure_path": str(figure_path),
    }
    (output_directory / "hr5_dual_agn_host_demographics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agn-directory", type=Path, default=DEFAULT_AGN_DIRECTORY)
    parser.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    parser.add_argument("--dimensionless-hubble", type=float, default=0.684)
    parser.add_argument("--box-size-cmpc-over-h", type=float, default=717.229040)
    parser.add_argument("--volume-cmpc3", type=float, default=FIDUCIAL_VOLUME_CMPC3)
    parser.add_argument("--spatial-region-count", type=int, default=8)
    args = parser.parse_args()
    summary = analyze(
        args.agn_directory,
        args.canonical_root,
        args.output_directory,
        args.dimensionless_hubble,
        args.box_size_cmpc_over_h,
        args.volume_cmpc3,
        args.spatial_region_count,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

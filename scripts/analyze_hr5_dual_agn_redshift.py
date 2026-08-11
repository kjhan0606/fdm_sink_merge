#!/usr/bin/env python3
"""Measure the redshift evolution of close active SMBH pairs in HR5."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from fdm_smbh_delay.hr5 import (
    find_agn_pair_population,
    locally_weighted_logarithmic_trend,
    pair_component_multiplicity,
    read_mkagn_snapshot,
    spatial_jackknife_pair_statistics,
)


DEFAULT_AGN_DIRECTORY = Path(
    "/home/kjhan/BACKUP/GalFinder/"
    "SRC(FoF_PSB_Free_Ver2_Dev)/SRC(AGN)/SRC(MkAGN)/HR5_AGN_DATA"
)
DEFAULT_OUTPUT_DIRECTORY = Path("results/hr5/dual_agn")
FIDUCIAL_VOLUME_CMPC3 = 1.087e7
COLORS = {"bol43": "#D55E00", "bol44": "#0072B2", "hx42": "#009E73"}
MARKERS = {"bol43": "o", "bol44": "s", "hx42": "D"}
LINE_STYLES = {"bol43": "-", "bol44": "--", "hx42": ":"}
LABELS = {
    "bol43": r"$L_{\rm bol}\geq10^{43}\,{\rm erg\,s^{-1}}$",
    "bol44": r"$L_{\rm bol}\geq10^{44}\,{\rm erg\,s^{-1}}$",
    "hx42": r"$L_{2-10\,{\rm keV}}\geq10^{42}\,{\rm erg\,s^{-1}}$",
}
OUTPUT_PATTERN = re.compile(r"agn\.(\d{5})\.dat$")


def _available_outputs(directory: Path) -> tuple[int, ...]:
    output_numbers = []
    for path in directory.glob("agn.*.dat"):
        match = OUTPUT_PATTERN.fullmatch(path.name)
        if match is not None:
            output_numbers.append(int(match.group(1)))
    return tuple(sorted(output_numbers))


def _selection_statistics(
    pairs: dict[str, np.ndarray],
    volume_cmpc3: float,
    box_size_cmpc_over_h: float,
    spatial_region_count: int,
) -> dict[str, float | int]:
    dual = pairs["is_dual"]
    active_count = int(pairs["active_count"])
    pair_count = int(np.count_nonzero(dual))
    spatial = spatial_jackknife_pair_statistics(
        pairs["active_position_x_cmpc_over_h"],
        pairs["position_1_cmpc_over_h"][:, 0],
        pairs["position_2_cmpc_over_h"][:, 0],
        dual,
        volume_cmpc3,
        box_size_cmpc_over_h,
        region_count=spatial_region_count,
    )
    if pair_count:
        _, member, multiplicity = pair_component_multiplicity(
            pairs["id_1"][dual], pairs["id_2"][dual]
        )
    else:
        member = np.empty(0, dtype=np.int64)
        multiplicity = np.empty(0, dtype=np.int64)
    pure_member_count = int(np.count_nonzero(multiplicity == 2))
    return {
        "active_agn_count": active_count,
        "dual_pair_count": pair_count,
        "dual_pair_number_density_cmpc3": float(spatial["number_density"]),
        "dual_pair_number_density_jackknife_error_cmpc3": float(
            spatial["number_density_jackknife_error"]
        ),
        "dual_pair_fraction": float(spatial["pair_fraction"]),
        "dual_pair_fraction_jackknife_error": float(
            spatial["pair_fraction_jackknife_error"]
        ),
        "dual_member_count": int(member.size),
        "dual_member_fraction": member.size / active_count if active_count else np.nan,
        "dual_member_count_error_fraction": (
            np.sqrt(member.size) / active_count if active_count else np.nan
        ),
        "pure_dual_member_count": pure_member_count,
        "pure_dual_member_fraction": (
            pure_member_count / active_count if active_count else np.nan
        ),
        "pure_dual_member_count_error_fraction": (
            np.sqrt(pure_member_count) / active_count if active_count else np.nan
        ),
    }


def _write_measurements(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fit_one_quantity(
    rows: list[dict[str, object]],
    selection: str,
    value_field: str,
    error_field: str,
    evaluation_redshift: np.ndarray,
    neighbor_count: int,
) -> np.ndarray:
    selected_rows = [row for row in rows if row["selection"] == selection]
    redshift = np.asarray([row["redshift"] for row in selected_rows], dtype=np.float64)
    value = np.asarray([row[value_field] for row in selected_rows], dtype=np.float64)
    error = np.asarray([row[error_field] for row in selected_rows], dtype=np.float64)
    count = np.asarray([row["dual_pair_count"] for row in selected_rows], dtype=np.int64)
    usable = (count >= 3) & np.isfinite(error)
    result = locally_weighted_logarithmic_trend(
        redshift[usable],
        value[usable],
        evaluation_redshift,
        None,
        neighbor_count=neighbor_count,
        degree=2,
    )
    if np.any(usable):
        result[
            (evaluation_redshift < np.min(redshift[usable]))
            | (evaluation_redshift > np.max(redshift[usable]))
        ] = np.nan
    return result


def _make_fits(
    rows: list[dict[str, object]], neighbor_count: int
) -> tuple[np.ndarray, dict[tuple[str, str], np.ndarray]]:
    positive_redshift = np.asarray(
        [
            row["redshift"]
            for row in rows
            if row["selection"] == "bol43" and row["dual_pair_count"] >= 3
        ],
        dtype=np.float64,
    )
    if positive_redshift.size < 4:
        raise ValueError("At least four snapshots with three pairs are required")
    evaluation_redshift = np.expm1(
        np.linspace(
            np.log1p(np.min(positive_redshift)),
            np.log1p(np.max(positive_redshift)),
            500,
        )
    )
    fits: dict[tuple[str, str], np.ndarray] = {}
    for selection in LABELS:
        fits[(selection, "density")] = _fit_one_quantity(
            rows,
            selection,
            "dual_pair_number_density_cmpc3",
            "dual_pair_number_density_jackknife_error_cmpc3",
            evaluation_redshift,
            neighbor_count,
        )
        fits[(selection, "pair_fraction")] = _fit_one_quantity(
            rows,
            selection,
            "dual_pair_fraction",
            "dual_pair_fraction_jackknife_error",
            evaluation_redshift,
            neighbor_count,
        )
    for value_field, error_field in (
        ("dual_member_fraction", "dual_member_count_error_fraction"),
        ("pure_dual_member_fraction", "pure_dual_member_count_error_fraction"),
    ):
        fits[("bol43", value_field)] = _fit_one_quantity(
            rows,
            "bol43",
            value_field,
            error_field,
            evaluation_redshift,
            neighbor_count,
        )
    return evaluation_redshift, fits


def _write_fits(
    path: Path,
    evaluation_redshift: np.ndarray,
    fits: dict[tuple[str, str], np.ndarray],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "redshift",
                "selection",
                "dual_pair_number_density_fit_cmpc3",
                "dual_pair_fraction_fit",
                "dual_member_fraction_fit",
                "pure_dual_member_fraction_fit",
            )
        )
        for selection in LABELS:
            member = fits.get(
                (selection, "dual_member_fraction"),
                np.full(evaluation_redshift.size, np.nan),
            )
            pure_member = fits.get(
                (selection, "pure_dual_member_fraction"),
                np.full(evaluation_redshift.size, np.nan),
            )
            for values in zip(
                evaluation_redshift,
                fits[(selection, "density")],
                fits[(selection, "pair_fraction")],
                member,
                pure_member,
            ):
                writer.writerow((values[0], selection, *values[1:]))


def _plot(
    path: Path,
    rows: list[dict[str, object]],
    evaluation_redshift: np.ndarray,
    fits: dict[tuple[str, str], np.ndarray],
    volume_cmpc3: float,
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
    figure, axes = plt.subplots(1, 2, figsize=(7.15, 2.65))
    figure.subplots_adjust(wspace=0.31, bottom=0.18, left=0.09, right=0.98, top=0.97)
    for selection in LABELS:
        selected_rows = [row for row in rows if row["selection"] == selection]
        redshift = np.asarray([row["redshift"] for row in selected_rows])
        count = np.asarray([row["dual_pair_count"] for row in selected_rows])
        density = np.asarray(
            [row["dual_pair_number_density_cmpc3"] for row in selected_rows]
        )
        density_error = np.asarray(
            [
                row["dual_pair_number_density_jackknife_error_cmpc3"]
                for row in selected_rows
            ]
        )
        detected = count > 0
        axes[0].errorbar(
            redshift[detected],
            density[detected],
            yerr=density_error[detected],
            fmt=MARKERS[selection],
            color=COLORS[selection],
            mfc="white",
            ms=3.8,
            lw=0.8,
            capsize=1.4,
            ls="none",
            label=LABELS[selection],
            zorder=3,
        )
        if np.any(~detected):
            upper_limit = -np.log(0.05) / volume_cmpc3
            axes[0].plot(
                redshift[~detected],
                np.full(np.count_nonzero(~detected), upper_limit),
                marker="v",
                color=COLORS[selection],
                mfc="white",
                ms=3.4,
                ls="none",
                zorder=2,
            )
        axes[0].plot(
            evaluation_redshift,
            fits[(selection, "density")],
            color=COLORS[selection],
            lw=1.1,
            ls=LINE_STYLES[selection],
        )

        fraction = np.asarray([row["dual_pair_fraction"] for row in selected_rows])
        fraction_error = np.asarray(
            [row["dual_pair_fraction_jackknife_error"] for row in selected_rows]
        )
        usable = np.isfinite(fraction) & (fraction > 0.0)
        axes[1].errorbar(
            redshift[usable],
            fraction[usable],
            yerr=fraction_error[usable],
            fmt=MARKERS[selection],
            color=COLORS[selection],
            mfc="white",
            ms=3.8,
            lw=0.8,
            capsize=1.4,
            ls="none",
            zorder=3,
        )
        axes[1].plot(
            evaluation_redshift,
            fits[(selection, "pair_fraction")],
            color=COLORS[selection],
            lw=1.1,
            ls=LINE_STYLES[selection],
        )
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    for axis in axes:
        axis.set_xlabel(r"$z$")
        axis.minorticks_on()
    axes[0].set_ylabel(r"$n_{\rm dual}$ [cMpc$^{-3}$]")
    axes[1].set_ylabel(r"$N_{\rm pair}/N_{\rm AGN}$")
    axes[0].legend(frameon=False, loc="upper right", handlelength=1.0)
    axes[0].text(0.03, 0.96, "(a)", transform=axes[0].transAxes, va="top")
    axes[1].text(0.03, 0.96, "(b)", transform=axes[1].transAxes, va="top")
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def analyze(
    agn_directory: Path,
    output_directory: Path,
    outputs: tuple[int, ...],
    dimensionless_hubble: float,
    box_size_cmpc_over_h: float,
    volume_cmpc3: float,
    spatial_region_count: int,
    neighbor_count: int,
) -> None:
    selection_specs = {
        "bol43": ("Lbol", 1.0e43),
        "bol44": ("Lbol", 1.0e44),
        "hx42": ("LhX", 1.0e42),
    }
    rows: list[dict[str, object]] = []
    for output_number in outputs:
        path = agn_directory / f"agn.{output_number:05d}.dat"
        redshift, _, records = read_mkagn_snapshot(path)
        for selection, (luminosity_field, threshold) in selection_specs.items():
            pairs = find_agn_pair_population(
                records,
                redshift,
                dimensionless_hubble,
                luminosity_threshold_erg_s=threshold,
                luminosity_field=luminosity_field,
                minimum_mass_msun=1.0e4,
                box_size_cmpc_over_h=box_size_cmpc_over_h,
            )
            statistics = _selection_statistics(
                pairs,
                volume_cmpc3,
                box_size_cmpc_over_h,
                spatial_region_count,
            )
            rows.append(
                {
                    "output_number": output_number,
                    "redshift": redshift,
                    "record_size_bytes": records.dtype.itemsize,
                    "selection": selection,
                    "luminosity_field": luminosity_field,
                    "luminosity_threshold_erg_s": threshold,
                    **statistics,
                }
            )
        fiducial = rows[-3]
        print(
            f"Output {output_number:05d}, z={redshift:.3f}: "
            f"{fiducial['active_agn_count']:,} active SMBHs and "
            f"{fiducial['dual_pair_count']:,} close pairs",
            flush=True,
        )

    evaluation_redshift, fits = _make_fits(rows, neighbor_count)
    output_directory.mkdir(parents=True, exist_ok=True)
    measurement_path = output_directory / "hr5_dual_agn_redshift_evolution.csv"
    fit_path = output_directory / "hr5_dual_agn_redshift_local_fits.csv"
    _write_measurements(measurement_path, rows)
    _write_fits(fit_path, evaluation_redshift, fits)
    _plot(
        output_directory / "hr5_dual_agn_redshift_evolution.pdf",
        rows,
        evaluation_redshift,
        fits,
        volume_cmpc3,
    )
    metadata = {
        "outputs": list(outputs),
        "snapshot_count": len(outputs),
        "selection": {
            "minimum_smbh_mass_msun": 1.0e4,
            "minimum_separation_pkpc": 0.5,
            "maximum_separation_pkpc": 30.0,
            "luminosity_selections": {
                key: {"field": field, "threshold_erg_s": threshold}
                for key, (field, threshold) in selection_specs.items()
            },
        },
        "uncertainty": {
            "method": "spatial jackknife along the long axis",
            "region_count": spatial_region_count,
        },
        "local_fit": {
            "coordinate": "log(1 + redshift)",
            "quantity": "natural logarithm of the measured abundance",
            "polynomial_degree": 2,
            "neighbor_count": neighbor_count,
            "distance_weight": "tricube",
            "measurement_weight": (
                "none; jackknife uncertainties are shown on the measurements"
            ),
            "minimum_dual_pair_count": 3,
        },
        "zero_count_upper_limit": {
            "confidence": 0.95,
            "poisson_events": float(-np.log(0.05)),
        },
    }
    (output_directory / "hr5_dual_agn_redshift_evolution.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agn-directory", type=Path, default=DEFAULT_AGN_DIRECTORY)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--outputs", nargs="+", type=int)
    parser.add_argument("--dimensionless-hubble", type=float, default=0.684)
    parser.add_argument("--box-size-cmpc-over-h", type=float, default=717.229040)
    parser.add_argument("--volume-cmpc3", type=float, default=FIDUCIAL_VOLUME_CMPC3)
    parser.add_argument("--spatial-region-count", type=int, default=8)
    parser.add_argument("--neighbor-count", type=int, default=7)
    args = parser.parse_args()
    outputs = tuple(args.outputs) if args.outputs else _available_outputs(args.agn_directory)
    if not outputs:
        raise ValueError(f"No MkAGN snapshots found in {args.agn_directory}")
    analyze(
        args.agn_directory,
        args.output_directory,
        outputs,
        args.dimensionless_hubble,
        args.box_size_cmpc_over_h,
        args.volume_cmpc3,
        args.spatial_region_count,
        args.neighbor_count,
    )


if __name__ == "__main__":
    main()

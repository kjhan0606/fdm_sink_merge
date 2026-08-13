#!/usr/bin/env python3
"""Compare host-galaxy evolution for matched dual- and single-AGN pairs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from astropy.cosmology import FlatLambdaCDM
from scipy.optimize import linear_sum_assignment, minimize
from scipy.stats import binomtest

from analyze_hr5_dual_agn_hosts import (
    FABLE_MINIMUM_HOST_STELLAR_MASS_MSUN,
    _host_value,
)
from analyze_hr5_host_descendants import _redshifts, _tree_paths, trace_pairs
from fdm_smbh_delay.hr5 import (
    HOST_RELATION_LABELS,
    classify_sink_pair_hosts,
    match_population_by_properties,
    read_sink_host_catalog,
)


DEFAULT_HR5_ROOT = Path("/scratch/kjhan/Hydro/HR5/FoFPSB/02_without_dc/Ver2")
DEFAULT_CANONICAL_ROOT = DEFAULT_HR5_ROOT / "Derived_Sink_Hosts" / "canonical_v1"
HOST_FEATURE_NAMES = (
    "log10_primary_smbh_mass",
    "log10_smbh_mass_ratio",
    "log10_separation",
    "log10_relative_speed_plus_10",
    "log10_primary_host_stellar_mass",
    "log10_secondary_host_stellar_mass",
    "log10_primary_host_gas_to_stellar_ratio",
    "log10_secondary_host_gas_to_stellar_ratio",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _pair_key(
    output: int, pair_class: str, first_sink: int, second_sink: int
) -> tuple[int, str, int, int]:
    low, high = sorted((first_sink, second_sink))
    return output, pair_class, low, high


def _pair_population_index(
    path: Path,
) -> dict[tuple[int, str, int, int], dict[str, str]]:
    result: dict[tuple[int, str, int, int], dict[str, str]] = {}
    for row in _read_csv(path):
        key = _pair_key(
            int(row["output_number"]),
            row["pair_class"],
            int(row["primary_sink_id"]),
            int(row["secondary_sink_id"]),
        )
        if key in result:
            raise ValueError(f"Duplicate SMBH pair in {path}: {key}")
        result[key] = row
    return result


def _matched_rows(
    matched_path: Path,
    population_path: Path,
    outputs: set[int] | None,
) -> list[dict[str, str]]:
    population = _pair_population_index(population_path)
    result: list[dict[str, str]] = []
    for match_id, match in enumerate(_read_csv(matched_path)):
        output = int(match["output_number"])
        if outputs is not None and output not in outputs:
            continue
        definitions = (
            (
                "dual",
                int(match["dual_primary_sink_id"]),
                int(match["dual_secondary_sink_id"]),
            ),
            (
                "offset",
                int(match["offset_primary_sink_id"]),
                int(match["offset_secondary_sink_id"]),
            ),
        )
        for pair_class, first_sink, second_sink in definitions:
            key = _pair_key(output, pair_class, first_sink, second_sink)
            if key not in population:
                raise ValueError(f"Matched pair is absent from the pair population: {key}")
            source = dict(population[key])
            source.update(
                {
                    "match_id": str(match_id),
                    "matched_pair_class": pair_class,
                    "standardized_match_distance": match[
                        "standardized_match_distance"
                    ],
                }
            )
            result.append(source)
    return result


def _population_rows(path: Path, outputs: set[int] | None) -> list[dict[str, str]]:
    rows = _read_csv(path)
    if outputs is None:
        return rows
    return [row for row in rows if int(row["output_number"]) in outputs]


def _remove_correlated_outputs(
    rows: list[dict[str, str]], minimum_separation_gyr: float
) -> tuple[list[dict[str, str]], list[int]]:
    if minimum_separation_gyr <= 0.0:
        return rows, []
    redshift = {
        int(row["output_number"]): float(row["redshift"])
        for row in rows
    }
    cosmology = FlatLambdaCDM(H0=68.4, Om0=0.3, Tcmb0=2.725)
    cosmic_time = {
        output: float(cosmology.age(value).value)
        for output, value in redshift.items()
    }
    retained: list[int] = []
    excluded: list[int] = []
    for output in sorted(cosmic_time, key=cosmic_time.get, reverse=True):
        if any(
            abs(cosmic_time[output] - cosmic_time[other]) < minimum_separation_gyr
            for other in retained
        ):
            excluded.append(output)
        else:
            retained.append(output)
    admitted = set(retained)
    return [row for row in rows if int(row["output_number"]) in admitted], sorted(excluded)


def _assign_hosts(
    rows: list[dict[str, str]],
    canonical_root: Path,
    dimensionless_hubble: float,
) -> list[dict[str, str]]:
    by_output: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_output[int(row["output_number"])].append(row)
    result: list[dict[str, str]] = []
    for output, selected in sorted(by_output.items()):
        path = canonical_root / f"output_{output:05d}" / f"sink_hosts.{output:05d}.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        hosts = read_sink_host_catalog(path)
        first_id = np.asarray(
            [int(row["primary_sink_id"]) for row in selected], dtype=np.int64
        )
        second_id = np.asarray(
            [int(row["secondary_sink_id"]) for row in selected], dtype=np.int64
        )
        relation, first_row, second_row = classify_sink_pair_hosts(
            first_id, second_id, hosts
        )
        first_gid = _host_value(hosts, first_row, "galaxy_gid", -1).astype(np.int64)
        second_gid = _host_value(hosts, second_row, "galaxy_gid", -1).astype(np.int64)
        first_stellar_mass = _host_value(
            hosts, first_row, "host_stellar_mass_msun_h", np.nan
        ) / dimensionless_hubble
        second_stellar_mass = _host_value(
            hosts, second_row, "host_stellar_mass_msun_h", np.nan
        ) / dimensionless_hubble
        first_stellar_count = _host_value(
            hosts, first_row, "host_stellar_count", -1
        ).astype(np.int64)
        second_stellar_count = _host_value(
            hosts, second_row, "host_stellar_count", -1
        ).astype(np.int64)
        first_gas_mass = _host_value(
            hosts, first_row, "host_gas_mass_msun_h", np.nan
        ) / dimensionless_hubble
        second_gas_mass = _host_value(
            hosts, second_row, "host_gas_mass_msun_h", np.nan
        ) / dimensionless_hubble
        for index, source in enumerate(selected):
            row = dict(source)
            fable_analogue = (
                float(source["primary_mass_msun"]) >= 1.0e6
                and float(source["secondary_mass_msun"]) >= 1.0e6
                and first_stellar_mass[index] >= FABLE_MINIMUM_HOST_STELLAR_MASS_MSUN
                and second_stellar_mass[index] >= FABLE_MINIMUM_HOST_STELLAR_MASS_MSUN
            )
            row.update(
                {
                    "selection_output": str(output),
                    "selection_redshift": source["redshift"],
                    "primary_galaxy_gid": str(first_gid[index]),
                    "secondary_galaxy_gid": str(second_gid[index]),
                    "host_relation": str(HOST_RELATION_LABELS[relation[index]]),
                    "primary_host_stellar_mass_msun": str(first_stellar_mass[index]),
                    "secondary_host_stellar_mass_msun": str(
                        second_stellar_mass[index]
                    ),
                    "primary_host_gas_mass_msun": str(first_gas_mass[index]),
                    "secondary_host_gas_mass_msun": str(second_gas_mass[index]),
                    "primary_host_stellar_particle_count": str(
                        first_stellar_count[index]
                    ),
                    "secondary_host_stellar_particle_count": str(
                        second_stellar_count[index]
                    ),
                    "fable_selection_analogue": str(int(fable_analogue)),
                }
            )
            result.append(row)
    return result


def _standardized_mean_difference(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    numerator = np.mean(first, axis=0) - np.mean(second, axis=0)
    denominator = np.sqrt(
        0.5 * (np.var(first, axis=0, ddof=1) + np.var(second, axis=0, ddof=1))
    )
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0.0,
    )


def _host_features(row: dict[str, str]) -> np.ndarray:
    first_stars = float(row["primary_host_stellar_mass_msun"])
    second_stars = float(row["secondary_host_stellar_mass_msun"])
    first_gas = float(row["primary_host_gas_mass_msun"])
    second_gas = float(row["secondary_host_gas_mass_msun"])
    return np.asarray(
        (
            np.log10(float(row["primary_mass_msun"])),
            np.log10(float(row["mass_ratio"])),
            np.log10(float(row["separation_pkpc"])),
            np.log10(float(row["relative_speed_kms"]) + 10.0),
            np.log10(first_stars),
            np.log10(second_stars),
            np.log10((first_gas + 1.0e6) / (first_stars + 1.0e6)),
            np.log10((second_gas + 1.0e6) / (second_stars + 1.0e6)),
        )
    )
def _host_informed_match(
    rows: list[dict[str, str]],
    match_caliper: float,
    include_same_host_at_selection: bool = False,
    require_fable_selection: bool = False,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    result: list[dict[str, str]] = []
    summary: dict[str, object] = {
        "feature_names": HOST_FEATURE_NAMES,
        "by_output": {},
    }
    next_match_id = 0
    outputs = sorted({int(row["output_number"]) for row in rows})
    for output in outputs:
        admitted_relations = {
            "distinct PSB galaxies in one FoF halo",
            "distinct FoF haloes",
        }
        if include_same_host_at_selection:
            admitted_relations.add("same PSB galaxy")
        available = [
            row
            for row in rows
            if int(row["output_number"]) == output
            and int(row["pair_system_multiplicity"]) == 2
            and row["host_relation"] in admitted_relations
            and (
                not require_fable_selection
                or int(row["fable_selection_analogue"]) == 1
            )
        ]
        dual = [row for row in available if row["pair_class"] == "dual"]
        offset = [row for row in available if row["pair_class"] == "offset"]
        if not dual or not offset:
            continue
        dual_features = np.asarray([_host_features(row) for row in dual])
        offset_features = np.asarray([_host_features(row) for row in offset])
        finite_dual = np.all(np.isfinite(dual_features), axis=1)
        finite_offset = np.all(np.isfinite(offset_features), axis=1)
        dual = [row for row, keep in zip(dual, finite_dual) if keep]
        offset = [row for row, keep in zip(offset, finite_offset) if keep]
        dual_features = dual_features[finite_dual]
        offset_features = offset_features[finite_offset]
        dual_index, offset_index, distance = match_population_by_properties(
            dual_features, offset_features
        )
        within_caliper = distance <= match_caliper
        dual_index = dual_index[within_caliper]
        offset_index = offset_index[within_caliper]
        distance = distance[within_caliper]
        if distance.size == 0:
            summary["by_output"][str(output)] = {
                "dual_pair_count_before_matching": len(dual),
                "single_agn_pair_count_before_matching": len(offset),
                "matched_system_count": 0,
                "match_caliper": match_caliper,
                "pair_count_excluded_by_caliper": min(len(dual), len(offset)),
                "exclusion_reason": "no pair passes the match caliper",
            }
            continue
        for pair_index, (first, second) in enumerate(zip(dual_index, offset_index)):
            for pair_class, source in (
                ("dual", dual[int(first)]),
                ("offset", offset[int(second)]),
            ):
                row = dict(source)
                row.update(
                    {
                        "match_id": str(next_match_id),
                        "matched_pair_class": pair_class,
                        "standardized_match_distance": str(distance[pair_index]),
                    }
                )
                result.append(row)
            next_match_id += 1
        matched_dual = dual_features[dual_index]
        matched_offset = offset_features[offset_index]
        summary["by_output"][str(output)] = {
            "dual_pair_count_before_matching": len(dual),
            "single_agn_pair_count_before_matching": len(offset),
            "matched_system_count": int(dual_index.size),
            "match_caliper": match_caliper,
            "pair_count_excluded_by_caliper": int(
                min(len(dual), len(offset)) - dual_index.size
            ),
            "absolute_standardized_mean_difference_before": np.abs(
                _standardized_mean_difference(dual_features, offset_features)
            ).tolist(),
            "absolute_standardized_mean_difference_after": np.abs(
                _standardized_mean_difference(matched_dual, matched_offset)
            ).tolist(),
            "match_distance_q16_q50_q84": np.quantile(
                distance, [0.16, 0.5, 0.84]
            ).tolist(),
        }
    return result, summary


def _propensity_score_match(
    rows: list[dict[str, str]],
    match_caliper: float,
    maximum_absolute_smd: float,
    regularization_c: float,
    include_same_host_at_selection: bool = False,
    require_fable_selection: bool = False,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    """Match within redshift using a regularized host-property propensity score."""

    result: list[dict[str, str]] = []
    summary: dict[str, object] = {
        "feature_names": HOST_FEATURE_NAMES,
        "propensity_regularization_c": regularization_c,
        "maximum_absolute_standardized_mean_difference": maximum_absolute_smd,
        "by_output": {},
    }
    next_match_id = 0
    outputs = sorted({int(row["output_number"]) for row in rows})
    for output in outputs:
        admitted_relations = {
            "distinct PSB galaxies in one FoF halo",
            "distinct FoF haloes",
        }
        if include_same_host_at_selection:
            admitted_relations.add("same PSB galaxy")
        available = [
            row
            for row in rows
            if int(row["output_number"]) == output
            and int(row["pair_system_multiplicity"]) == 2
            and row["host_relation"] in admitted_relations
            and (
                not require_fable_selection
                or int(row["fable_selection_analogue"]) == 1
            )
        ]
        dual = [row for row in available if row["pair_class"] == "dual"]
        offset = [row for row in available if row["pair_class"] == "offset"]
        if not dual or not offset:
            continue
        dual_features = np.asarray([_host_features(row) for row in dual])
        offset_features = np.asarray([_host_features(row) for row in offset])
        finite_dual = np.all(np.isfinite(dual_features), axis=1)
        finite_offset = np.all(np.isfinite(offset_features), axis=1)
        dual = [row for row, keep in zip(dual, finite_dual) if keep]
        offset = [row for row, keep in zip(offset, finite_offset) if keep]
        dual_features = dual_features[finite_dual]
        offset_features = offset_features[finite_offset]
        pooled = np.vstack((dual_features, offset_features))
        centre = np.mean(pooled, axis=0)
        scale = np.std(pooled, axis=0, ddof=1)
        scale[(~np.isfinite(scale)) | (scale == 0.0)] = 1.0
        standardized = (pooled - centre) / scale
        response = np.concatenate((np.ones(len(dual)), np.zeros(len(offset))))

        def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
            linear = parameters[0] + standardized @ parameters[1:]
            residual = 1.0 / (1.0 + np.exp(-np.clip(linear, -700.0, 700.0))) - response
            value = np.sum(np.logaddexp(0.0, linear) - response * linear)
            value += 0.5 / regularization_c * np.sum(parameters[1:] ** 2)
            gradient = np.concatenate(
                (
                    np.asarray([np.sum(residual)]),
                    standardized.T @ residual + parameters[1:] / regularization_c,
                )
            )
            return float(value), gradient

        fit = minimize(
            objective,
            np.zeros(standardized.shape[1] + 1),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 2000, "ftol": 1.0e-12},
        )
        if not fit.success:
            raise RuntimeError(f"Propensity fit failed at output {output}: {fit.message}")
        logit = fit.x[0] + standardized @ fit.x[1:]
        logit_scale = float(np.std(logit, ddof=1))
        if not np.isfinite(logit_scale) or logit_scale == 0.0:
            logit_scale = 1.0
        dual_logit = logit[: len(dual)]
        offset_logit = logit[len(dual) :]
        distance = np.abs(dual_logit[:, None] - offset_logit[None, :]) / logit_scale
        dual_index, offset_index = linear_sum_assignment(distance)
        within_caliper = distance[dual_index, offset_index] <= match_caliper
        dual_index = dual_index[within_caliper]
        offset_index = offset_index[within_caliper]
        selected_distance = distance[dual_index, offset_index]
        output_summary: dict[str, object] = {
            "dual_pair_count_before_matching": len(dual),
            "single_agn_pair_count_before_matching": len(offset),
            "candidate_matched_system_count": int(dual_index.size),
            "match_caliper_in_logit_standard_deviations": match_caliper,
            "pair_count_excluded_by_caliper": int(
                min(len(dual), len(offset)) - dual_index.size
            ),
            "absolute_standardized_mean_difference_before": np.abs(
                _standardized_mean_difference(dual_features, offset_features)
            ).tolist(),
        }
        if dual_index.size < 2:
            output_summary.update(
                {
                    "matched_system_count": 0,
                    "exclusion_reason": "fewer than two pairs pass the match caliper",
                }
            )
            summary["by_output"][str(output)] = output_summary
            continue
        matched_dual = dual_features[dual_index]
        matched_offset = offset_features[offset_index]
        after = np.abs(_standardized_mean_difference(matched_dual, matched_offset))
        output_summary["absolute_standardized_mean_difference_after"] = after.tolist()
        output_summary["match_distance_q16_q50_q84"] = np.quantile(
            selected_distance, [0.16, 0.5, 0.84]
        ).tolist()
        if np.max(after) > maximum_absolute_smd:
            output_summary.update(
                {
                    "matched_system_count": 0,
                    "exclusion_reason": "post-match covariate imbalance",
                }
            )
            summary["by_output"][str(output)] = output_summary
            continue
        output_summary["matched_system_count"] = int(dual_index.size)
        summary["by_output"][str(output)] = output_summary
        for pair_index, (first, second) in enumerate(zip(dual_index, offset_index)):
            for pair_class, source in (
                ("dual", dual[int(first)]),
                ("offset", offset[int(second)]),
            ):
                row = dict(source)
                row.update(
                    {
                        "match_id": str(next_match_id),
                        "matched_pair_class": pair_class,
                        "standardized_match_distance": str(
                            selected_distance[pair_index]
                        ),
                    }
                )
                result.append(row)
            next_match_id += 1
    if not result:
        raise ValueError("No redshift output passes the matching-quality criteria")
    return result, summary


def _delay_interval(row: dict[str, object]) -> tuple[float, float] | None:
    if row["host_track_status"] == "same_host_at_selection":
        return 0.0, 0.0
    if row["host_track_status"] != "common_descendant":
        return None
    return (
        float(row["common_descendant_delay_lower_gyr"]),
        float(row["common_descendant_delay_upper_gyr"]),
    )


def _matched_delay_order(rows: list[dict[str, object]]) -> dict[str, object]:
    by_match: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        by_match[str(row["match_id"])][str(row["matched_pair_class"])] = row
    order = Counter()
    common_delay_difference_lower: list[float] = []
    common_delay_difference_upper: list[float] = []
    fable_pair_count = 0
    for pair in by_match.values():
        if set(pair) != {"dual", "offset"}:
            raise ValueError("Each matched system must contain a dual and offset pair")
        dual = pair["dual"]
        offset = pair["offset"]
        if int(dual["fable_selection_analogue"]) and int(
            offset["fable_selection_analogue"]
        ):
            fable_pair_count += 1
        dual_interval = _delay_interval(dual)
        offset_interval = _delay_interval(offset)
        if dual_interval is None or offset_interval is None:
            order["unresolved"] += 1
            continue
        difference_lower = dual_interval[0] - offset_interval[1]
        difference_upper = dual_interval[1] - offset_interval[0]
        common_delay_difference_lower.append(difference_lower)
        common_delay_difference_upper.append(difference_upper)
        if difference_upper < 0.0:
            order["dual_hosts_join_earlier"] += 1
        elif difference_lower > 0.0:
            order["dual_hosts_join_later"] += 1
        else:
            order["delay_intervals_overlap"] += 1
    summary: dict[str, object] = {
        "matched_system_count": len(by_match),
        "both_pairs_in_fable_selection_analogue_count": fable_pair_count,
        "host_delay_order": dict(order),
    }

    def same_host_comparison(
        pairs: list[dict[str, dict[str, object]]],
    ) -> dict[str, object]:
        contingency: Counter[str] = Counter()
        for pair in pairs:
            dual_same = pair["dual"]["host_track_status"] == "same_host_at_selection"
            offset_same = pair["offset"]["host_track_status"] == "same_host_at_selection"
            if dual_same and offset_same:
                contingency["both_same_host"] += 1
            elif dual_same:
                contingency["dual_only_same_host"] += 1
            elif offset_same:
                contingency["single_only_same_host"] += 1
            else:
                contingency["neither_same_host"] += 1
        total = len(pairs)
        dual_count = contingency["both_same_host"] + contingency["dual_only_same_host"]
        offset_count = (
            contingency["both_same_host"] + contingency["single_only_same_host"]
        )
        discordant = (
            contingency["dual_only_same_host"]
            + contingency["single_only_same_host"]
        )
        probability = (
            binomtest(
                contingency["dual_only_same_host"], discordant, 0.5
            ).pvalue
            if discordant
            else 1.0
        )
        return {
            "matched_system_count": total,
            "contingency": dict(contingency),
            "dual_same_host_fraction": dual_count / total if total else None,
            "single_agn_same_host_fraction": offset_count / total if total else None,
            "dual_minus_single_same_host_fraction": (
                (dual_count - offset_count) / total if total else None
            ),
            "mcnemar_exact_two_sided_p": probability,
        }

    matched_pairs = list(by_match.values())
    summary["same_host_at_selection"] = same_host_comparison(matched_pairs)
    by_output: dict[str, object] = {}
    outputs = sorted(
        {int(pair["dual"]["selection_output"]) for pair in matched_pairs}
    )
    for output in outputs:
        selected = [
            pair
            for pair in matched_pairs
            if int(pair["dual"]["selection_output"]) == output
        ]
        by_output[str(output)] = same_host_comparison(selected)
    summary["same_host_at_selection_by_output"] = by_output
    fable_pairs = [
        pair
        for pair in matched_pairs
        if int(pair["dual"]["fable_selection_analogue"])
        and int(pair["offset"]["fable_selection_analogue"])
    ]
    fable_by_output: dict[str, object] = {}
    for output in outputs:
        selected = [
            pair
            for pair in fable_pairs
            if int(pair["dual"]["selection_output"]) == output
        ]
        if selected:
            fable_by_output[str(output)] = same_host_comparison(selected)
    summary["fable_selection_analogue"] = {
        "matched_system_count": len(fable_pairs),
        "same_host_at_selection": same_host_comparison(fable_pairs),
        "same_host_at_selection_by_output": fable_by_output,
    }
    if common_delay_difference_lower:
        summary["dual_minus_single_host_delay_bounds_gyr"] = {
            "lower_q16_q50_q84": np.quantile(
                common_delay_difference_lower, [0.16, 0.5, 0.84]
            ).tolist(),
            "upper_q16_q50_q84": np.quantile(
                common_delay_difference_upper, [0.16, 0.5, 0.84]
            ).tolist(),
        }
    return summary


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    by_class: dict[str, object] = {}
    for pair_class in ("dual", "offset"):
        selected = [row for row in rows if row["matched_pair_class"] == pair_class]
        delay = [interval for row in selected if (interval := _delay_interval(row))]
        class_summary: dict[str, object] = {
            "pair_count": len(selected),
            "host_relation": dict(Counter(row["host_relation"] for row in selected)),
            "host_track_status": dict(
                Counter(row["host_track_status"] for row in selected)
            ),
            "fable_selection_analogue_count": sum(
                int(row["fable_selection_analogue"]) for row in selected
            ),
        }
        if delay:
            lower = np.asarray([value[0] for value in delay])
            upper = np.asarray([value[1] for value in delay])
            class_summary["host_joining_delay_bounds_gyr"] = {
                "lower_q16_q50_q84": np.quantile(lower, [0.16, 0.5, 0.84]).tolist(),
                "upper_q16_q50_q84": np.quantile(upper, [0.16, 0.5, 0.84]).tolist(),
            }
        by_class[pair_class] = class_summary
    return {
        "pair_count": len(rows),
        "by_agn_pair_class": by_class,
        "matched_comparison": _matched_delay_order(rows),
        "interpretation": (
            "The dual and single-AGN pairs were matched before host tracing. "
            "This comparison does not use the legacy assigned capture companion."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matched-pairs",
        type=Path,
        default=Path("results/hr5/dual_agn/hr5_dual_offset_matched_pairs.csv"),
    )
    parser.add_argument(
        "--pair-population",
        type=Path,
        default=(
            DEFAULT_CANONICAL_ROOT
            / "agn_pair_hosts"
            / "hr5_agn_pair_hosts_mbh_ge_1e6.csv"
        ),
    )
    parser.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    parser.add_argument(
        "--tree-root", type=Path, default=DEFAULT_HR5_ROOT / "Galaxy_Merging"
    )
    parser.add_argument("--outputs", nargs="+", type=int)
    parser.add_argument(
        "--matching",
        choices=("propensity", "host", "legacy"),
        default="propensity",
        help=(
            "Use host-property propensity matching, standardized host-property "
            "distance, or reproduce the previous four-property match."
        ),
    )
    parser.add_argument("--match-caliper", type=float)
    parser.add_argument("--maximum-absolute-smd", type=float, default=0.2)
    parser.add_argument("--propensity-regularization-c", type=float, default=0.3)
    parser.add_argument("--minimum-snapshot-separation-gyr", type=float, default=0.05)
    parser.add_argument(
        "--include-same-host-at-selection",
        action="store_true",
        help="Include close active SMBH pairs already assigned to one PSB galaxy.",
    )
    parser.add_argument(
        "--require-fable-selection",
        action="store_true",
        help="Require each dual and single-AGN pair to pass the FABLE mass analogue.",
    )
    parser.add_argument("--dimensionless-hubble", type=float, default=0.684)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_CANONICAL_ROOT / "matched_pair_host_descendants",
    )
    args = parser.parse_args()
    if args.match_caliper is None:
        args.match_caliper = 0.2 if args.matching == "propensity" else 1.0
    if (
        args.match_caliper <= 0.0
        or args.maximum_absolute_smd <= 0.0
        or args.propensity_regularization_c <= 0.0
        or args.minimum_snapshot_separation_gyr < 0.0
    ):
        parser.error("Matching scale parameters must be positive")

    selected_outputs = set(args.outputs) if args.outputs else None
    if args.matching in {"propensity", "host"}:
        population = _population_rows(args.pair_population, selected_outputs)
        population, correlated_outputs = _remove_correlated_outputs(
            population, args.minimum_snapshot_separation_gyr
        )
        hosted_population = _assign_hosts(
            population, args.canonical_root, args.dimensionless_hubble
        )
        if args.matching == "propensity":
            hosted, matching_summary = _propensity_score_match(
                hosted_population,
                args.match_caliper,
                args.maximum_absolute_smd,
                args.propensity_regularization_c,
                args.include_same_host_at_selection,
                args.require_fable_selection,
            )
        else:
            hosted, matching_summary = _host_informed_match(
                hosted_population,
                args.match_caliper,
                args.include_same_host_at_selection,
                args.require_fable_selection,
            )
    else:
        matched = _matched_rows(
            args.matched_pairs, args.pair_population, selected_outputs
        )
        hosted = _assign_hosts(matched, args.canonical_root, args.dimensionless_hubble)
        matching_summary = {
            "method": "previous four-property match",
            "features": [
                "primary SMBH mass",
                "SMBH mass ratio",
                "separation",
                "relative speed",
            ],
        }
        correlated_outputs = []
    tree_outputs, tree_paths = _tree_paths(args.tree_root)
    redshift = _redshifts(
        args.canonical_root / "hr5_output_manifest.csv", tree_outputs
    )
    traced = trace_pairs(hosted, tree_outputs, tree_paths, redshift, {})

    args.output_directory.mkdir(parents=True, exist_ok=True)
    table_path = args.output_directory / "hr5_matched_agn_pair_host_descendants.csv"
    summary_path = args.output_directory / "hr5_matched_agn_pair_host_descendants.json"
    temporary_table = table_path.with_suffix(".csv.tmp")
    temporary_summary = summary_path.with_suffix(".json.tmp")
    with temporary_table.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(traced[0]))
        writer.writeheader()
        writer.writerows(traced)
    summary = _summary(traced)
    summary["matching"] = {
        "method": args.matching,
        "initial_host_relation": (
            "one or two directly assigned PSB galaxies"
            if args.include_same_host_at_selection
            else "two distinct directly assigned PSB galaxies"
        ),
        "minimum_snapshot_separation_gyr": args.minimum_snapshot_separation_gyr,
        "outputs_excluded_as_temporally_correlated": correlated_outputs,
        "fable_mass_selection_required_before_matching": (
            args.require_fable_selection
        ),
        **matching_summary,
    }
    temporary_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary_table.replace(table_path)
    temporary_summary.replace(summary_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

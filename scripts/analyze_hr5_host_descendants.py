#!/usr/bin/env python3
"""Trace the host galaxies of HR5 active SMBH pairs to common descendants."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from astropy.cosmology import FlatLambdaCDM

from fdm_smbh_delay.hr5_galaxies import (
    map_galaxy_descendants,
    open_galaxy_links,
)


DEFAULT_HR5_ROOT = Path("/scratch/kjhan/Hydro/HR5/FoFPSB/02_without_dc/Ver2")
DEFAULT_DERIVED_ROOT = DEFAULT_HR5_ROOT / "Derived_Sink_Hosts" / "canonical_v1"
DESCENDANT_STATUS = {
    2: "host_not_in_link_catalogue",
    3: "host_track_ended",
    4: "ambiguous_descendant",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _pair_key(output: int, first_sink: int, second_sink: int) -> tuple[int, int, int]:
    low, high = sorted((first_sink, second_sink))
    return output, low, high


def _capture_index(path: Path | None) -> dict[tuple[int, int, int], dict[str, str]]:
    if path is None or not path.is_file():
        return {}
    result: dict[tuple[int, int, int], dict[str, str]] = {}
    for row in _read_csv(path):
        key = _pair_key(
            int(row["output_number"]),
            int(row["primary_sink_id"]),
            int(row["secondary_sink_id"]),
        )
        result[key] = row
    return result


def _tree_paths(tree_root: Path) -> tuple[list[int], list[Path]]:
    paths: dict[int, Path] = {}
    for path in tree_root.glob("GalaxyLinkedList.[0-9][0-9][0-9][0-9][0-9]"):
        output = int(path.name.rsplit(".", maxsplit=1)[1])
        paths[output] = path
    outputs = sorted(paths)
    return outputs, [paths[output] for output in outputs]


def _redshifts(manifest_path: Path, outputs: list[int]) -> np.ndarray:
    manifest = {
        int(row["output"]): float(row["redshift"])
        for row in _read_csv(manifest_path)
        if row["redshift"]
    }
    missing = [output for output in outputs if output not in manifest]
    if missing:
        raise ValueError(f"No redshift in the HR5 manifest for outputs {missing}")
    return np.asarray([manifest[output] for output in outputs], dtype=np.float64)


def _load_pair_rows(derived_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    pattern = "output_*/hr5_dual_agn_hosts.[0-9][0-9][0-9][0-9][0-9].csv"
    for path in sorted(derived_root.glob(pattern)):
        output = int(path.stem.rsplit(".", maxsplit=1)[1])
        summary_path = path.with_suffix(".json")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for row in _read_csv(path):
            row["selection_output"] = str(output)
            row["selection_redshift"] = str(summary["redshift"])
            rows.append(row)
    return rows


def _time_order(
    host_status: str,
    host_lower: float,
    host_upper: float,
    capture_lower: float,
    capture_upper: float,
) -> str:
    if not np.isfinite(capture_upper):
        return "no_later_possible_binary_capture"
    if host_status == "same_host_at_selection":
        return "common_host_before_later_possible_binary_capture"
    if host_status == "common_descendant":
        if host_upper < capture_lower:
            return "common_descendant_before_possible_binary_capture"
        if capture_upper < host_lower:
            return "possible_binary_capture_before_common_descendant"
        return "time_intervals_overlap"
    if host_status == "right_censored" and capture_upper < host_lower:
        return "possible_binary_capture_before_last_resolved_distinct_hosts"
    return "host_time_unresolved"


def trace_pairs(
    rows: list[dict[str, str]],
    outputs: list[int],
    paths: list[Path],
    redshift: np.ndarray,
    capture_rows: dict[tuple[int, int, int], dict[str, str]],
) -> list[dict[str, object]]:
    output_index = {output: index for index, output in enumerate(outputs)}
    selection_output = np.asarray(
        [int(row["selection_output"]) for row in rows], dtype=np.int64
    )
    missing = sorted(set(selection_output) - set(outputs))
    if missing:
        raise ValueError(f"No galaxy-link catalogue for selection outputs {missing}")
    start = np.asarray([output_index[int(value)] for value in selection_output])
    first_gid = np.asarray(
        [int(row["primary_galaxy_gid"]) for row in rows], dtype=np.int64
    )
    second_gid = np.asarray(
        [int(row["secondary_galaxy_gid"]) for row in rows], dtype=np.int64
    )
    current_first = first_gid.copy()
    current_second = second_gid.copy()
    status = np.full(len(rows), "active", dtype=object)
    common_output = np.full(len(rows), -1, dtype=np.int64)
    common_gid = np.full(len(rows), -1, dtype=np.int64)
    lower_delay = np.full(len(rows), np.nan, dtype=np.float64)
    upper_delay = np.full(len(rows), np.nan, dtype=np.float64)
    used_major_branch = np.zeros(len(rows), dtype=bool)

    unclassifiable = (first_gid < 0) | (second_gid < 0)
    status[unclassifiable] = "unclassifiable_host"
    same = (~unclassifiable) & (first_gid == second_gid)
    status[same] = "same_host_at_selection"
    common_output[same] = selection_output[same]
    common_gid[same] = first_gid[same]
    lower_delay[same] = 0.0
    upper_delay[same] = 0.0

    cosmology = FlatLambdaCDM(H0=68.4, Om0=0.3, Tcmb0=2.725)
    cosmic_time = np.asarray(cosmology.age(redshift).value, dtype=np.float64)

    for tree_index in range(len(outputs) - 1):
        active = np.flatnonzero((status == "active") & (start <= tree_index))
        if active.size == 0:
            continue
        _, current = open_galaxy_links(paths[tree_index])
        _, following = open_galaxy_links(paths[tree_index + 1])
        requested = np.unique(
            np.concatenate((current_first[active], current_second[active]))
        )
        descendant, link_status = map_galaxy_descendants(
            current, following, requested
        )
        first_position = np.searchsorted(requested, current_first[active])
        second_position = np.searchsorted(requested, current_second[active])
        first_status = link_status[first_position]
        second_status = link_status[second_position]
        first_next = descendant[first_position]
        second_next = descendant[second_position]
        used_major_branch[active] |= (first_status == 1) | (second_status == 1)

        failed = (first_status >= 2) | (second_status >= 2)
        if np.any(failed):
            failed_rows = active[failed]
            combined = np.maximum(first_status[failed], second_status[failed])
            status[failed_rows] = [DESCENDANT_STATUS[int(code)] for code in combined]
        succeeded = ~failed
        if not np.any(succeeded):
            del current, following
            continue

        succeeded_rows = active[succeeded]
        current_first[succeeded_rows] = first_next[succeeded]
        current_second[succeeded_rows] = second_next[succeeded]
        joined = current_first[succeeded_rows] == current_second[succeeded_rows]
        if np.any(joined):
            joined_rows = succeeded_rows[joined]
            status[joined_rows] = "common_descendant"
            common_output[joined_rows] = outputs[tree_index + 1]
            common_gid[joined_rows] = current_first[joined_rows]
            lower_delay[joined_rows] = (
                cosmic_time[tree_index] - cosmic_time[start[joined_rows]]
            )
            upper_delay[joined_rows] = (
                cosmic_time[tree_index + 1] - cosmic_time[start[joined_rows]]
            )
        del current, following

    censored = status == "active"
    status[censored] = "right_censored"
    lower_delay[censored] = cosmic_time[-1] - cosmic_time[start[censored]]

    result: list[dict[str, object]] = []
    for index, source in enumerate(rows):
        row: dict[str, object] = dict(source)
        row.update(
            {
                "host_track_status": status[index],
                "common_descendant_output": int(common_output[index]),
                "common_descendant_redshift": (
                    float(redshift[output_index[common_output[index]]])
                    if common_output[index] >= 0
                    else np.nan
                ),
                "common_descendant_gid": int(common_gid[index]),
                "common_descendant_delay_lower_gyr": float(lower_delay[index]),
                "common_descendant_delay_upper_gyr": float(upper_delay[index]),
                "major_branch_resolution_used": int(used_major_branch[index]),
            }
        )
        capture = capture_rows.get(
            _pair_key(
                int(selection_output[index]),
                int(source["primary_sink_id"]),
                int(source["secondary_sink_id"]),
            )
        )
        capture_output = -1
        capture_lower = np.nan
        capture_upper = np.nan
        pair_class = ""
        if capture is not None:
            capture_output = int(capture["assigned_capture_output"])
            capture_lower = float(capture["capture_delay_lower_gyr"])
            capture_upper = float(capture["capture_delay_upper_gyr"])
            pair_class = capture.get("pair_class", "")
        row.update(
            {
                "active_pair_class": pair_class,
                "assigned_capture_output": capture_output,
                "capture_delay_lower_gyr": float(capture_lower),
                "capture_delay_upper_gyr": float(capture_upper),
                "capture_host_time_order": _time_order(
                    str(status[index]),
                    float(lower_delay[index]),
                    float(upper_delay[index]),
                    float(capture_lower),
                    float(capture_upper),
                ),
            }
        )
        result.append(row)
    return result


def _summary(rows: list[dict[str, object]], outputs: list[int]) -> dict[str, object]:
    distinct_relations = {
        "distinct PSB galaxies in one FoF halo",
        "distinct FoF haloes",
    }
    fable = [
        row
        for row in rows
        if int(row["fable_selection_analogue"]) == 1
        and row["host_relation"] in distinct_relations
    ]
    captured = [row for row in rows if int(row["assigned_capture_output"]) >= 0]

    def bounds(sample: list[dict[str, object]]) -> dict[str, object]:
        common = [row for row in sample if row["host_track_status"] == "common_descendant"]
        if not common:
            return {"common_descendant_count": 0}
        lower = np.asarray(
            [row["common_descendant_delay_lower_gyr"] for row in common],
            dtype=np.float64,
        )
        upper = np.asarray(
            [row["common_descendant_delay_upper_gyr"] for row in common],
            dtype=np.float64,
        )
        return {
            "common_descendant_count": len(common),
            "lower_bound_gyr_q16_q50_q84": np.quantile(lower, [0.16, 0.5, 0.84]).tolist(),
            "upper_bound_gyr_q16_q50_q84": np.quantile(upper, [0.16, 0.5, 0.84]).tolist(),
        }

    selection_outputs = sorted({int(row["selection_output"]) for row in rows})
    evolution: dict[str, object] = {}
    for output in selection_outputs:
        selected = [row for row in rows if int(row["selection_output"]) == output]
        selected_fable = [row for row in fable if int(row["selection_output"]) == output]
        evolution[str(output)] = {
            "redshift": float(selected[0]["selection_redshift"]),
            "pair_count": len(selected),
            "host_track_status": dict(
                Counter(row["host_track_status"] for row in selected)
            ),
            "common_descendant_delay_bounds": bounds(selected),
            "fable_selection_analogue_pair_count": len(selected_fable),
            "fable_selection_analogue_common_descendant_delay_bounds": bounds(selected_fable),
        }
    return {
        "pair_count": len(rows),
        "tree_output_count": len(outputs),
        "tree_output_range": [outputs[0], outputs[-1]],
        "host_track_status": dict(Counter(row["host_track_status"] for row in rows)),
        "common_descendant_delay_bounds": bounds(rows),
        "fable_selection_analogue": {
            "minimum_smbh_mass_msun": 1.0e6,
            "minimum_host_stellar_mass_msun": 100.0 * 6.4e6 / 0.679,
            "host_relation": "distinct PSB galaxies",
            "host_resolution_note": (
                "The threshold matches FABLE's published stellar-mass value, but "
                "HR5 uses total PSB stellar mass rather than mass within twice "
                "the stellar half-mass radius."
            ),
            "pair_count": len(fable),
            "host_track_status": dict(
                Counter(row["host_track_status"] for row in fable)
            ),
            "common_descendant_delay_bounds": bounds(fable),
        },
        "pair_count_with_later_possible_binary_capture": len(captured),
        "capture_host_time_order": dict(
            Counter(row["capture_host_time_order"] for row in captured)
        ),
        "by_selection_output": evolution,
        "interpretation": {
            "common_descendant": (
                "First output in which the two directly assigned PSB galaxies "
                "enter the same descendant in the HR5 galaxy links."
            ),
            "time_interval": (
                "The common-descendant time is bracketed by the adjacent available "
                "galaxy outputs."
            ),
            "right_censoring": (
                "Distinct host tracks that persist to output 296 are right-censored "
                "at z=0.625."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-root", type=Path, default=DEFAULT_DERIVED_ROOT)
    parser.add_argument(
        "--tree-root", type=Path, default=DEFAULT_HR5_ROOT / "Galaxy_Merging"
    )
    parser.add_argument(
        "--capture-pairs",
        type=Path,
        default=Path("results/hr5/dual_agn/hr5_dual_agn_pairs.csv"),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_DERIVED_ROOT / "host_descendants",
    )
    args = parser.parse_args()

    rows = _load_pair_rows(args.derived_root)
    if not rows:
        raise ValueError(f"No completed active-pair host catalogues under {args.derived_root}")
    outputs, paths = _tree_paths(args.tree_root)
    if not outputs:
        raise ValueError(f"No HR5 galaxy-link catalogues under {args.tree_root}")
    redshift = _redshifts(args.derived_root / "hr5_output_manifest.csv", outputs)
    result = trace_pairs(
        rows,
        outputs,
        paths,
        redshift,
        _capture_index(args.capture_pairs),
    )

    args.output_directory.mkdir(parents=True, exist_ok=True)
    table_path = args.output_directory / "hr5_active_pair_host_descendants.csv"
    summary_path = args.output_directory / "hr5_active_pair_host_descendants.json"
    temporary_table = table_path.with_suffix(".csv.tmp")
    temporary_summary = summary_path.with_suffix(".json.tmp")
    with temporary_table.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    summary = _summary(result, outputs)
    temporary_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary_table.replace(table_path)
    temporary_summary.replace(summary_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

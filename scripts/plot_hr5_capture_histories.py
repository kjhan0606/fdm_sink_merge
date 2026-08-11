#!/usr/bin/env python3
"""Plot capture histories for two massive SMBHs in Horizon Run 5."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from fdm_smbh_delay.hr5 import (
    HEADER_DTYPE,
    SINK_DTYPE,
    read_mkagn_snapshot,
    read_tree_header,
)


DEFAULT_TREE = Path(
    "/home/kjhan/BACKUP/GalFinder/"
    "SRC(FoF_PSB_Free_Ver2_Dev)/SRC(AGN)/BinarySMBH/"
    "Sink_Merging_Tree.dat.Updated"
)
DEFAULT_FINAL_AGN = Path(
    "/home/kjhan/BACKUP/GalFinder/"
    "SRC(FoF_PSB_Free_Ver2_Dev)/SRC(AGN)/SRC(MkAGN)/"
    "HR5_AGN_DATA/agn.00296.dat"
)
DEFAULT_CAPTURE_CATALOG = Path("results/hr5/hr5_capture_catalog.csv")
DEFAULT_DUAL_CATALOG = Path("results/hr5/dual_agn/hr5_dual_agn_pairs.csv")
DEFAULT_OUTPUT_DIRECTORY = Path("results/hr5/capture_histories")
BOX_SIZE_CMPC = 1048.5
MAJOR_MASS_RATIO = 0.1
TRACK_RADIUS_PKPC = 50.0
TRACK_WINDOW_GYR = 1.0
MASS_SCALE_LOG_MIN = 4.0
MASS_SCALE_LOG_MAX = 10.0


def _plot_settings() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.0,
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.8,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "legend.title_fontsize": 7.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _panel_label(axis: plt.Axes, label: str) -> None:
    label_text = axis.text2D(
        0.04,
        0.96,
        label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        color="black",
        fontweight="bold",
        zorder=50,
    )
    label_text.set_path_effects(
        [path_effects.withStroke(linewidth=1.6, foreground="white")]
    )


def _read_capture_catalog(path: Path) -> dict[str, np.ndarray]:
    names = (
        "sink_id",
        "receiver_id",
        "last_resolved_history_index",
        "assigned_capture_history_index",
        "last_resolved_output",
        "assigned_capture_output",
        "last_resolved_redshift",
        "assigned_capture_redshift",
        "minor_mass_last_resolved_msun",
        "receiver_mass_last_resolved_msun",
    )
    with path.open(encoding="utf-8") as stream:
        header = next(csv.reader(stream))
    missing = sorted(set(names) - set(header))
    if missing:
        raise ValueError(f"The capture catalogue is missing columns: {missing}")
    values = np.loadtxt(
        path,
        delimiter=",",
        skiprows=1,
        usecols=tuple(header.index(name) for name in names),
        dtype=np.float64,
    )
    result = {name: values[:, index] for index, name in enumerate(names)}
    for name in names[:6]:
        result[name] = result[name].astype(np.int64)
    return result


def _read_sink_records(path: Path, sink_ids: set[int]) -> dict[int, np.void]:
    records: dict[int, np.void] = {}
    with path.open("rb") as stream:
        for sink_id in sorted(sink_ids):
            stream.seek(HEADER_DTYPE.itemsize + (sink_id - 1) * SINK_DTYPE.itemsize)
            value = np.fromfile(stream, dtype=SINK_DTYPE, count=1)
            if value.size != 1 or int(value["sink_id"][0]) != sink_id:
                raise ValueError(f"Could not read sink particle {sink_id}")
            records[sink_id] = value[0]
    return records


def _minimum_image(delta: np.ndarray, box_size_cmpc: float) -> np.ndarray:
    return delta - box_size_cmpc * np.rint(delta / box_size_cmpc)


def _most_massive_final_sink(path: Path, dimensionless_hubble: float) -> tuple[int, float, float]:
    redshift, _, records = read_mkagn_snapshot(path)
    index = int(np.nanargmax(records["mass"]))
    return (
        int(records["sink_id"][index]),
        float(records["mass"][index] / dimensionless_hubble),
        redshift,
    )


def _most_massive_dual_member(path: Path, output_number: int) -> dict[str, float | int]:
    best: dict[str, float | int] | None = None
    with path.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if int(row["output_number"]) != output_number or row["pair_class"] != "dual":
                continue
            for member, companion in (("primary", "secondary"), ("secondary", "primary")):
                mass = float(row[f"{member}_mass_msun"])
                if best is None or mass > float(best["mass_msun"]):
                    best = {
                        "sink_id": int(row[f"{member}_sink_id"]),
                        "companion_id": int(row[f"{companion}_sink_id"]),
                        "mass_msun": mass,
                        "companion_mass_msun": float(row[f"{companion}_mass_msun"]),
                        "output_number": output_number,
                        "redshift": float(row["redshift"]),
                        "separation_pkpc": float(row["separation_pkpc"]),
                        "lbol_erg_s": float(row[f"{member}_lbol_erg_s"]),
                        "companion_lbol_erg_s": float(row[f"{companion}_lbol_erg_s"]),
                    }
    if best is None:
        raise ValueError(f"No dual AGN pair is available at output {output_number}")
    return best


def _children_by_parent(catalog: dict[str, np.ndarray]) -> dict[int, list[int]]:
    children: dict[int, list[int]] = defaultdict(list)
    for row, parent in enumerate(catalog["receiver_id"]):
        children[int(parent)].append(row)
    return children


def _collect_capture_tree(
    root_id: int,
    catalog: dict[str, np.ndarray],
    children: dict[int, list[int]],
) -> tuple[set[int], list[int]]:
    nodes = {root_id}
    rows: list[int] = []
    stack = [root_id]
    while stack:
        parent = stack.pop()
        for row in children.get(parent, []):
            child = int(catalog["sink_id"][row])
            rows.append(row)
            if child not in nodes:
                nodes.add(child)
                stack.append(child)
    return nodes, rows


def _maximum_tree_depth(
    root_id: int,
    catalog: dict[str, np.ndarray],
    children: dict[int, list[int]],
) -> int:
    maximum_depth = 0
    stack = [(root_id, 0)]
    while stack:
        parent, depth = stack.pop()
        maximum_depth = max(maximum_depth, depth)
        for row in children.get(parent, []):
            stack.append((int(catalog["sink_id"][row]), depth + 1))
    return maximum_depth


def _tree_positions(
    root_id: int,
    rows: list[int],
    catalog: dict[str, np.ndarray],
) -> tuple[dict[int, float], dict[int, list[int]]]:
    edge_for_child = {int(catalog["sink_id"][row]): row for row in rows}
    child_ids: dict[int, list[int]] = defaultdict(list)
    for row in rows:
        child_ids[int(catalog["receiver_id"][row])].append(
            int(catalog["sink_id"][row])
        )
    for parent in child_ids:
        child_ids[parent].sort(
            key=lambda child: (
                int(catalog["assigned_capture_history_index"][edge_for_child[child]]),
                -float(catalog["minor_mass_last_resolved_msun"][edge_for_child[child]]),
            )
        )
    positions: dict[int, float] = {}
    next_leaf = 0.0

    def assign(node: int) -> float:
        nonlocal next_leaf
        descendants = child_ids.get(node, [])
        if not descendants:
            positions[node] = next_leaf
            next_leaf += 1.0
            return positions[node]
        descendant_positions = [assign(child) for child in descendants]
        positions[node] = 0.5 * (
            min(descendant_positions) + max(descendant_positions)
        )
        return positions[node]

    assign(root_id)
    root_position = positions[root_id]
    for node in positions:
        positions[node] -= root_position
    return positions, child_ids


def _cosmic_time_transform(
    redshift: np.ndarray,
    cosmic_time: np.ndarray,
):
    def time_to_redshift(time_gyr: np.ndarray) -> np.ndarray:
        return np.interp(time_gyr, cosmic_time, redshift)

    def redshift_to_time(value: np.ndarray) -> np.ndarray:
        return np.interp(value, redshift[::-1], cosmic_time[::-1])

    return time_to_redshift, redshift_to_time


def _mass_marker_area(mass_msun: float | np.ndarray) -> float | np.ndarray:
    logarithmic_mass = np.log10(np.maximum(mass_msun, 10.0**MASS_SCALE_LOG_MIN))
    normalized_mass = np.clip(
        (logarithmic_mass - MASS_SCALE_LOG_MIN)
        / (MASS_SCALE_LOG_MAX - MASS_SCALE_LOG_MIN),
        0.0,
        1.0,
    )
    return 10.0 + 70.0 * normalized_mass


def _plot_capture_tree(
    output: Path,
    root_id: int,
    root_mass_msun: float,
    nodes: set[int],
    rows: list[int],
    catalog: dict[str, np.ndarray],
    records: dict[int, np.void],
    redshift: np.ndarray,
    cosmic_time: np.ndarray,
    maximum_redshift: float,
) -> dict[str, float | int]:
    positions, _ = _tree_positions(root_id, rows, catalog)
    row_for_child = {int(catalog["sink_id"][row]): row for row in rows}
    major_color = "#26456E"
    minor_color = "#B5B5B5"
    figure = plt.figure(figsize=(7.15, 5.1))
    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=(0.91, 0.09),
        left=0.075,
        right=0.94,
        bottom=0.09,
        top=0.88,
        wspace=0.28,
    )
    axis = figure.add_subplot(grid[0, 0])
    mass_axis = figure.add_subplot(grid[0, 1])
    lower_time = float(np.interp(maximum_redshift, redshift[::-1], cosmic_time[::-1]))
    final_time = float(cosmic_time[-1])
    branch_color: dict[int, str] = {}
    branch_alpha: dict[int, float] = {}
    mass_markers: dict[tuple[int, int], float] = {}

    for node in sorted(nodes):
        state = np.asarray(records[node]["state"][: redshift.size], dtype=np.float64)
        active = np.flatnonzero(state[:, 0] > 0.0)
        if active.size == 0:
            continue
        begin = max(lower_time, float(cosmic_time[active[0]]))
        if node == root_id:
            end = final_time
            color = "#D55E00"
            alpha = 1.0
            width = 2.7
            zorder = 15
        else:
            row = row_for_child[node]
            last_index = int(catalog["last_resolved_history_index"][row])
            end = float(cosmic_time[last_index])
            mass = float(catalog["minor_mass_last_resolved_msun"][row])
            parent_mass = float(catalog["receiver_mass_last_resolved_msun"][row])
            mass_ratio = min(mass, parent_mass) / max(mass, parent_mass)
            color = major_color if mass_ratio >= MAJOR_MASS_RATIO else minor_color
            alpha = 0.9 if mass_ratio >= MAJOR_MASS_RATIO else 0.28
            width = 0.65 + 1.35 * np.sqrt(min(1.0, mass_ratio))
            zorder = 7 if mass_ratio >= MAJOR_MASS_RATIO else 2
        branch_color[node] = color
        branch_alpha[node] = alpha
        if end >= lower_time:
            axis.plot(
                [positions[node], positions[node]],
                [begin, end],
                color=color,
                lw=width,
                alpha=alpha,
                solid_capstyle="round",
                zorder=zorder,
            )
        birth_index = int(active[0])
        if cosmic_time[birth_index] >= lower_time:
            mass_markers[(node, birth_index)] = float(state[birth_index, 0])
        if node != root_id:
            last_index = int(catalog["last_resolved_history_index"][row_for_child[node]])
            if cosmic_time[last_index] >= lower_time:
                mass_markers[(node, last_index)] = float(state[last_index, 0])

    major_count = 0
    for row in rows:
        child = int(catalog["sink_id"][row])
        parent = int(catalog["receiver_id"][row])
        last_index = int(catalog["last_resolved_history_index"][row])
        capture_index = int(catalog["assigned_capture_history_index"][row])
        lower = float(cosmic_time[last_index])
        upper = float(cosmic_time[capture_index])
        if upper < lower_time:
            continue
        child_mass = float(catalog["minor_mass_last_resolved_msun"][row])
        parent_mass = float(catalog["receiver_mass_last_resolved_msun"][row])
        mass_ratio = min(child_mass, parent_mass) / max(child_mass, parent_mass)
        major = mass_ratio >= MAJOR_MASS_RATIO
        major_count += int(major)
        color = major_color if major else minor_color
        alpha = 0.9 if major else 0.28
        width = 0.65 + 1.35 * np.sqrt(min(1.0, mass_ratio))
        axis.plot(
            [positions[child], positions[parent]],
            [max(lower, lower_time), upper],
            color=color,
            ls=(0, (2.0, 1.4)),
            lw=width,
            alpha=alpha,
            zorder=8 if major else 3,
        )
        parent_state = np.asarray(records[parent]["state"][: redshift.size])
        mass_markers[(parent, capture_index)] = float(parent_state[capture_index, 0])

    for (node, state_index), mass_msun in mass_markers.items():
        axis.scatter(
            [positions[node]],
            [cosmic_time[state_index]],
            s=_mass_marker_area(mass_msun),
            facecolor="white",
            edgecolor=branch_color[node],
            linewidth=0.75,
            alpha=max(branch_alpha[node], 0.48),
            zorder=12,
        )

    axis.set_ylim(lower_time, final_time)
    axis.set_ylabel("cosmic time [Gyr]")
    axis.set_xticks([])
    axis.tick_params(axis="x", which="both", bottom=False, top=False)
    axis.spines["bottom"].set_visible(False)
    axis.spines["top"].set_visible(False)
    axis.grid(axis="y", color="#D9D9D9", lw=0.45, alpha=0.7)
    time_to_redshift, redshift_to_time = _cosmic_time_transform(redshift, cosmic_time)
    redshift_axis = axis.secondary_yaxis(
        "right", functions=(time_to_redshift, redshift_to_time)
    )
    redshift_axis.set_ylabel("redshift")
    redshift_axis.set_yticks((6.0, 4.0, 3.0, 2.0, 1.0, float(redshift[-1])))
    position_span = max(positions.values()) - min(positions.values())
    axis.text(
        positions[root_id] - 0.015 * position_span,
        final_time - 0.015 * (final_time - lower_time),
        rf"ID {root_id}",
        color="#D55E00",
        ha="right",
        va="top",
        fontsize=7.0,
        fontweight="bold",
    )
    legend = (
        Line2D([0], [0], color="#D55E00", lw=2.7, label="main branch"),
        Line2D([0], [0], color=major_color, lw=1.5, label=rf"$q\geq{MAJOR_MASS_RATIO:g}$"),
        Line2D([0], [0], color=minor_color, lw=1.2, alpha=0.55, label=rf"$q<{MAJOR_MASS_RATIO:g}$"),
        Line2D([0], [0], color="#555555", lw=1.0, ls=(0, (2.0, 1.4)), label="assigned capture interval"),
    )
    figure.legend(
        handles=legend,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.48, 0.985),
        ncol=4,
        columnspacing=1.5,
        handletextpad=0.7,
    )
    scale_log_mass = np.array((4.0, 6.0, 8.0, 10.0))
    mass_axis.scatter(
        np.full(scale_log_mass.size, 0.42),
        scale_log_mass,
        s=_mass_marker_area(10.0**scale_log_mass),
        facecolor="white",
        edgecolor="#4A4A4A",
        linewidth=0.75,
        clip_on=False,
    )
    mass_axis.set_xlim(0.0, 1.0)
    mass_axis.set_ylim(MASS_SCALE_LOG_MIN - 0.6, MASS_SCALE_LOG_MAX + 0.6)
    mass_axis.set_xticks([])
    mass_axis.set_yticks(scale_log_mass)
    mass_axis.set_yticklabels(tuple(rf"$10^{{{int(value)}}}$" for value in scale_log_mass))
    mass_axis.yaxis.tick_right()
    mass_axis.yaxis.set_label_position("right")
    mass_axis.set_ylabel(r"SMBH mass [$M_\odot$]", labelpad=7.0)
    mass_axis.tick_params(axis="y", which="major", length=0, pad=3.0)
    for spine in mass_axis.spines.values():
        spine.set_visible(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)
    return {
        "root_sink_id": root_id,
        "root_mass_msun": root_mass_msun,
        "node_count": len(nodes),
        "assigned_capture_count": len(rows),
        "major_capture_count": major_count,
        "mass_marker_count": len(mass_markers),
        "maximum_redshift_shown": maximum_redshift,
    }


def _unwrap_root(position: np.ndarray, box_size_cmpc: float) -> np.ndarray:
    step = _minimum_image(np.diff(position, axis=0), box_size_cmpc)
    return np.vstack((np.zeros(3), np.cumsum(step, axis=0)))


def _last_near_segment(
    indices: np.ndarray,
    separation_pkpc: np.ndarray,
    cosmic_time: np.ndarray,
    maximum_radius_pkpc: float,
    maximum_duration_gyr: float,
) -> np.ndarray:
    if indices.size == 0:
        return indices
    end_time = cosmic_time[indices[-1]]
    start = indices.size - 1
    while start > 0:
        previous = start - 1
        if separation_pkpc[previous] > maximum_radius_pkpc:
            break
        if end_time - cosmic_time[indices[previous]] > maximum_duration_gyr:
            break
        start = previous
    return indices[start:]


def _style_3d_axis(axis: plt.Axes) -> None:
    axis.set_proj_type("ortho")
    axis.grid(True)
    for coordinate_axis in (axis.xaxis, axis.yaxis, axis.zaxis):
        coordinate_axis.set_major_locator(MaxNLocator(nbins=4))
        coordinate_axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        coordinate_axis.pane.set_edgecolor((0.65, 0.65, 0.65, 0.55))
        coordinate_axis._axinfo["grid"]["color"] = (0.75, 0.75, 0.75, 0.35)
        coordinate_axis._axinfo["grid"]["linewidth"] = 0.4


def _plot_dual_trajectory(
    output: Path,
    root: dict[str, float | int],
    direct_rows: list[int],
    catalog: dict[str, np.ndarray],
    records: dict[int, np.void],
    output_number: np.ndarray,
    redshift: np.ndarray,
    cosmic_time: np.ndarray,
    box_size_cmpc: float,
) -> dict[str, float | int]:
    root_id = int(root["sink_id"])
    companion_id = int(root["companion_id"])
    selection_index = int(np.flatnonzero(output_number == int(root["output_number"]))[0])
    root_state = np.asarray(records[root_id]["state"][: redshift.size], dtype=np.float64)
    root_active = np.flatnonzero(root_state[:, 0] > 0.0)
    root_position = root_state[root_active, 1:4]
    root_unwrapped_active = _unwrap_root(root_position, box_size_cmpc)
    root_unwrapped = np.full((redshift.size, 3), np.nan)
    root_unwrapped[root_active] = root_unwrapped_active
    origin = root_unwrapped[selection_index].copy()
    root_unwrapped -= origin

    capture_redshifts = catalog["last_resolved_redshift"][direct_rows]
    redshift_norm = Normalize(
        vmin=float(np.min(capture_redshifts)),
        vmax=float(np.max(capture_redshifts)),
    )
    color_map = plt.get_cmap("cividis_r")
    figure = plt.figure(figsize=(7.15, 4.45))
    first_axis = figure.add_subplot(1, 2, 1, projection="3d")
    second_axis = figure.add_subplot(1, 2, 2, projection="3d")
    _style_3d_axis(first_axis)
    _style_3d_axis(second_axis)
    first_axis.view_init(elev=22.0, azim=-58.0)
    second_axis.view_init(elev=25.0, azim=-52.0)

    first_axis.plot(
        root_unwrapped[root_active, 0],
        root_unwrapped[root_active, 1],
        root_unwrapped[root_active, 2],
        color="#D55E00",
        lw=2.5,
        zorder=20,
        label=f"SMBH {root_id}",
    )
    relative_tracks: list[np.ndarray] = []
    plotted_track_count = 0
    for row in sorted(
        direct_rows,
        key=lambda value: int(catalog["assigned_capture_history_index"][value]),
    ):
        child_id = int(catalog["sink_id"][row])
        child_state = np.asarray(records[child_id]["state"][: redshift.size], dtype=np.float64)
        capture_index = int(catalog["assigned_capture_history_index"][row])
        shared = np.flatnonzero(
            (root_state[:capture_index, 0] > 0.0)
            & (child_state[:capture_index, 0] > 0.0)
        )
        if shared.size == 0:
            continue
        delta = _minimum_image(
            child_state[shared, 1:4] - root_state[shared, 1:4], box_size_cmpc
        )
        relative_pkpc = delta * (1000.0 / (1.0 + redshift[shared, None]))
        separation = np.linalg.norm(relative_pkpc, axis=1)
        segment_indices = _last_near_segment(
            shared,
            separation,
            cosmic_time,
            TRACK_RADIUS_PKPC,
            TRACK_WINDOW_GYR,
        )
        lookup = {int(index): number for number, index in enumerate(shared)}
        local = np.asarray([lookup[int(index)] for index in segment_indices])
        relative_segment = relative_pkpc[local]
        absolute_segment = root_unwrapped[segment_indices] + delta[local]
        child_mass = float(catalog["minor_mass_last_resolved_msun"][row])
        parent_mass = float(catalog["receiver_mass_last_resolved_msun"][row])
        mass_ratio = min(child_mass, parent_mass) / max(child_mass, parent_mass)
        capture_redshift = float(catalog["last_resolved_redshift"][row])
        highlighted = child_id == companion_id
        color = "#CC79A7" if highlighted else color_map(redshift_norm(capture_redshift))
        alpha = 1.0 if highlighted else 0.28 + 0.62 * min(1.0, np.sqrt(mass_ratio / 0.1))
        width = 2.2 if highlighted else 0.65 + 1.8 * min(1.0, np.sqrt(mass_ratio))
        first_axis.plot(
            absolute_segment[:, 0],
            absolute_segment[:, 1],
            absolute_segment[:, 2],
            color=color,
            lw=width,
            alpha=alpha,
            zorder=12 if highlighted else 5,
        )
        second_axis.plot(
            relative_segment[:, 0],
            relative_segment[:, 1],
            relative_segment[:, 2],
            color=color,
            lw=width,
            alpha=alpha,
            zorder=12 if highlighted else 5,
        )
        for axis, point in (
            (first_axis, absolute_segment[-1]),
            (second_axis, relative_segment[-1]),
        ):
            axis.scatter(
                [point[0]],
                [point[1]],
                [point[2]],
                s=13.0 if highlighted else 6.0,
                facecolors="white",
                edgecolors=color,
                linewidths=0.7,
                alpha=max(alpha, 0.55),
                depthshade=False,
                zorder=25,
            )
        relative_tracks.append(relative_segment)
        plotted_track_count += 1

    selection_root = root_unwrapped[selection_index]
    companion_state = np.asarray(
        records[companion_id]["state"][: redshift.size], dtype=np.float64
    )
    selection_delta = _minimum_image(
        companion_state[selection_index, 1:4] - root_state[selection_index, 1:4],
        box_size_cmpc,
    )
    selection_relative = selection_delta * 1000.0 / (1.0 + redshift[selection_index])
    selection_companion = selection_root + selection_delta
    first_axis.scatter(
        [selection_root[0]], [selection_root[1]], [selection_root[2]],
        marker="*", s=58.0, color="#D55E00", edgecolor="white", linewidth=0.6,
        depthshade=False, zorder=35,
    )
    first_axis.scatter(
        [selection_companion[0]], [selection_companion[1]], [selection_companion[2]],
        marker="D", s=24.0, color="#CC79A7", edgecolor="white", linewidth=0.6,
        depthshade=False, zorder=35,
    )
    second_axis.scatter(
        [0.0], [0.0], [0.0], marker="*", s=64.0, color="#D55E00",
        edgecolor="white", linewidth=0.6, depthshade=False, zorder=35,
    )
    second_axis.scatter(
        [selection_relative[0]], [selection_relative[1]], [selection_relative[2]],
        marker="D", s=25.0, color="#CC79A7", edgecolor="white", linewidth=0.6,
        depthshade=False, zorder=35,
    )
    second_axis.plot(
        [0.0, selection_relative[0]],
        [0.0, selection_relative[1]],
        [0.0, selection_relative[2]],
        color="#CC79A7", ls=(0, (2.0, 1.4)), lw=1.0, zorder=22,
    )

    first_axis.set_xlabel(r"$X-X_{\rm sel}$ [cMpc]", labelpad=3.0)
    first_axis.set_ylabel(r"$Y-Y_{\rm sel}$ [cMpc]", labelpad=3.0)
    first_axis.set_zlabel(r"$Z-Z_{\rm sel}$ [cMpc]", labelpad=0.0)
    second_axis.set_xlabel(r"$\Delta x$ [pkpc]", labelpad=3.0)
    second_axis.set_ylabel(r"$\Delta y$ [pkpc]", labelpad=3.0)
    second_axis.set_zlabel(r"$\Delta z$ [pkpc]", labelpad=0.0)
    _panel_label(first_axis, "(a)")
    _panel_label(second_axis, "(b)")
    first_axis.legend(
        handles=(
            Line2D([0], [0], color="#D55E00", lw=2.5, label=f"SMBH {root_id}"),
            Line2D([0], [0], color="#777777", lw=1.0, label=f"removed SMBHs assigned to {root_id}"),
            Line2D([0], [0], marker="D", color="none", markerfacecolor="#CC79A7", markeredgecolor="white", label=f"dual companion {companion_id}"),
        ),
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.94),
    )
    if relative_tracks:
        limit = TRACK_RADIUS_PKPC
        second_axis.set_xlim(-limit, limit)
        second_axis.set_ylim(-limit, limit)
        second_axis.set_zlim(-limit, limit)
        second_axis.set_box_aspect((1.0, 1.0, 1.0))
    root_extent = np.ptp(root_unwrapped[root_active], axis=0)
    first_axis.set_box_aspect(np.maximum(root_extent, 0.18 * np.max(root_extent)))
    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=redshift_norm, cmap=color_map),
        ax=(first_axis, second_axis),
        pad=0.11,
        fraction=0.026,
        shrink=0.72,
    )
    colorbar.set_label("redshift at final resolved output")
    figure.subplots_adjust(left=0.015, right=0.83, bottom=0.08, top=0.98, wspace=0.10)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight", pad_inches=0.035)
    plt.close(figure)
    return {
        "root_sink_id": root_id,
        "root_mass_msun_at_selection": float(root["mass_msun"]),
        "selection_output": int(root["output_number"]),
        "selection_redshift": float(root["redshift"]),
        "dual_companion_id": companion_id,
        "dual_companion_mass_msun": float(root["companion_mass_msun"]),
        "dual_separation_pkpc": float(root["separation_pkpc"]),
        "direct_assigned_capture_count": len(direct_rows),
        "plotted_track_count": plotted_track_count,
        "track_radius_pkpc": TRACK_RADIUS_PKPC,
        "track_window_gyr": TRACK_WINDOW_GYR,
    }


def make_figures(
    tree_path: Path,
    final_agn_path: Path,
    capture_catalog_path: Path,
    dual_catalog_path: Path,
    output_directory: Path,
    dual_output: int,
    dimensionless_hubble: float,
    box_size_cmpc: float,
) -> None:
    _plot_settings()
    header = read_tree_header(tree_path)
    nstep = int(header["nstep"])
    redshift = np.asarray(header["redshift"][:nstep], dtype=np.float64)
    output_number = np.asarray(header["output_number"][:nstep], dtype=np.int64)
    cosmology = FlatLambdaCDM(
        H0=float(header["h0"]),
        Om0=float(header["omega_m"]),
        Tcmb0=2.7255,
    )
    cosmic_time = np.asarray(cosmology.age(redshift).value)
    catalog = _read_capture_catalog(capture_catalog_path)
    children = _children_by_parent(catalog)

    final_root_id, final_root_mass, final_redshift = _most_massive_final_sink(
        final_agn_path, dimensionless_hubble
    )
    tree_nodes, tree_rows = _collect_capture_tree(final_root_id, catalog, children)
    tree_records = _read_sink_records(tree_path, tree_nodes)
    output_directory.mkdir(parents=True, exist_ok=True)
    tree_summary = _plot_capture_tree(
        output_directory / "hr5_most_massive_capture_tree.pdf",
        final_root_id,
        final_root_mass,
        tree_nodes,
        tree_rows,
        catalog,
        tree_records,
        redshift,
        cosmic_time,
        maximum_redshift=7.0,
    )
    tree_summary["final_redshift"] = final_redshift
    tree_summary["direct_assigned_capture_count"] = len(
        children.get(final_root_id, [])
    )
    tree_summary["maximum_tree_depth"] = _maximum_tree_depth(
        final_root_id, catalog, children
    )
    tree_summary["minimum_assigned_capture_redshift"] = float(
        np.min(catalog["assigned_capture_redshift"][tree_rows])
    )
    tree_summary["maximum_assigned_capture_redshift"] = float(
        np.max(catalog["assigned_capture_redshift"][tree_rows])
    )

    dual_root = _most_massive_dual_member(dual_catalog_path, dual_output)
    direct_rows = children.get(int(dual_root["sink_id"]), [])
    trajectory_ids = {int(dual_root["sink_id"]), int(dual_root["companion_id"])}
    trajectory_ids.update(int(catalog["sink_id"][row]) for row in direct_rows)
    trajectory_records = _read_sink_records(tree_path, trajectory_ids)
    trajectory_summary = _plot_dual_trajectory(
        output_directory / "hr5_massive_dual_agn_trajectories.pdf",
        dual_root,
        direct_rows,
        catalog,
        trajectory_records,
        output_number,
        redshift,
        cosmic_time,
        box_size_cmpc,
    )

    summary = {
        "capture_tree": tree_summary,
        "dual_agn_trajectories": trajectory_summary,
        "interpretation": {
            "links": "Every branch connection uses the surviving SMBH assigned by the distance and mass criteria.",
            "capture": "The plotted links represent possible numerical binary captures rather than verified capture partners or physical coalescences.",
            "activity": "The dual-AGN classification applies at the selected output and does not supply a continuous luminosity history.",
        },
    }
    (output_directory / "hr5_capture_histories_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, default=DEFAULT_TREE)
    parser.add_argument("--final-agn", type=Path, default=DEFAULT_FINAL_AGN)
    parser.add_argument("--capture-catalog", type=Path, default=DEFAULT_CAPTURE_CATALOG)
    parser.add_argument("--dual-catalog", type=Path, default=DEFAULT_DUAL_CATALOG)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--dual-output", type=int, default=117)
    parser.add_argument("--dimensionless-hubble", type=float, default=0.684)
    parser.add_argument("--box-size-cmpc", type=float, default=BOX_SIZE_CMPC)
    args = parser.parse_args()
    make_figures(
        args.tree,
        args.final_agn,
        args.capture_catalog,
        args.dual_catalog,
        args.output_directory,
        args.dual_output,
        args.dimensionless_hubble,
        args.box_size_cmpc,
    )


if __name__ == "__main__":
    main()

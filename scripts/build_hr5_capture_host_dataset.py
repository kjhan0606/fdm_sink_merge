#!/usr/bin/env python3
"""Prepare and extract direct HR5 hosts for all possible binary captures."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import shutil
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path


DEFAULT_HR5_ROOT = Path("/scratch/kjhan/Hydro/HR5/FoFPSB/02_without_dc/Ver2")
DEFAULT_CANONICAL_ROOT = DEFAULT_HR5_ROOT / "Derived_Sink_Hosts" / "canonical_v1"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def prepare_capture_targets(
    capture_catalog: Path,
    manifest_path: Path,
    output_root: Path,
) -> list[dict[str, object]]:
    """Write unique sink-identifier selections for each last-resolved output."""

    capture_rows: list[tuple[int, int, int]] = []
    with capture_catalog.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"sink_id", "receiver_id", "last_resolved_output"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Capture catalogue is missing columns: {sorted(required)}")
        for row in reader:
            capture_rows.append(
                (
                    int(row["last_resolved_output"]),
                    int(row["sink_id"]),
                    int(row["receiver_id"]),
                )
            )

    manifest = {int(row["output"]): row for row in _read_csv(manifest_path)}
    available_outputs = sorted(
        output
        for output, row in manifest.items()
        if row["host_extraction_ready"] == "True"
        and row.get("galaxy_count", "")
        and row.get("galaxy_link_path", "")
    )
    if not available_outputs:
        raise ValueError("The canonical manifest contains no direct-host outputs")
    identifiers: dict[int, set[int]] = defaultdict(set)
    event_count: dict[int, int] = defaultdict(int)
    source_outputs: dict[int, set[int]] = defaultdict(set)
    for last_output, sink_id, receiver_id in capture_rows:
        position = bisect.bisect_right(available_outputs, last_output) - 1
        if position < 0:
            raise ValueError(
                f"No direct-host output precedes capture history output {last_output}"
            )
        host_output = available_outputs[position]
        identifiers[host_output].update((sink_id, receiver_id))
        event_count[host_output] += 1
        source_outputs[host_output].add(last_output)
    output_root.mkdir(parents=True, exist_ok=True)
    target_root = output_root / "targets"
    target_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for output in sorted(identifiers):
        if output not in manifest:
            raise ValueError(f"Output {output} is absent from the canonical HR5 manifest")
        source = manifest[output]
        tag = f"{output:05d}"
        target_path = target_root / f"sink_ids.{tag}.txt"
        temporary_target = target_path.with_suffix(".txt.tmp")
        temporary_target.write_text(
            "".join(f"{sink_id}\n" for sink_id in sorted(identifiers[output])),
            encoding="ascii",
        )
        temporary_target.replace(target_path)
        derived = output_root / f"output_{tag}"
        event_path = derived / f"possible_binary_captures.{tag}.csv"
        filtered_path = derived / f"sink_hosts_for_captures.{tag}.csv"
        full_path = Path(source["sink_host_catalog_path"])
        if full_path.is_file():
            host_path = full_path
            host_source = "full_canonical_host_catalogue"
            status = "complete"
        else:
            host_path = filtered_path
            host_source = "capture_filtered_host_catalogue"
            status = "complete" if filtered_path.is_file() else "missing"
        rows.append(
            {
                "output": output,
                "redshift": float(source["redshift"]),
                "possible_binary_capture_count": event_count[output],
                "last_resolved_outputs": ";".join(
                    str(value) for value in sorted(source_outputs[output])
                ),
                "maximum_output_lag": max(source_outputs[output]) - output,
                "requested_sink_count": len(identifiers[output]),
                "sink_id_path": str(target_path),
                "capture_event_path": str(event_path),
                "capture_event_status": "complete" if event_path.is_file() else "missing",
                "galfind_path": source["galfind_path"],
                "galaxy_list_path": source["galaxy_list_path"],
                "galaxy_count": int(source["galaxy_count"]),
                "host_catalogue_path": str(host_path),
                "host_catalogue_source": host_source,
                "host_catalogue_status": status,
                "extraction_summary_path": str(derived / "sink_host_extraction.json"),
            }
        )
    return rows


def partition_capture_events(
    capture_catalog: Path,
    rows: list[dict[str, object]],
) -> None:
    """Partition the capture catalogue by its preceding galaxy output."""

    source_to_host: dict[int, int] = {}
    row_by_output = {int(row["output"]): row for row in rows}
    for row in rows:
        host_output = int(row["output"])
        for value in str(row["last_resolved_outputs"]).split(";"):
            source = int(value)
            if source in source_to_host:
                raise ValueError(f"Capture output {source} maps to more than one host output")
            source_to_host[source] = host_output

    temporary: dict[int, Path] = {}
    final: dict[int, Path] = {}
    completed = False
    try:
        with capture_catalog.open(newline="") as source, ExitStack() as stack:
            reader = csv.DictReader(source)
            if reader.fieldnames is None:
                raise ValueError(f"No header in {capture_catalog}")
            fields = ["host_assignment_output", *reader.fieldnames]
            writers: dict[int, csv.DictWriter] = {}
            for output, row in row_by_output.items():
                path = Path(str(row["capture_event_path"]))
                path.parent.mkdir(parents=True, exist_ok=True)
                temp = path.with_suffix(".csv.tmp")
                stream = stack.enter_context(temp.open("w", newline=""))
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writers[output] = writer
                temporary[output] = temp
                final[output] = path
            for event in reader:
                last_output = int(event["last_resolved_output"])
                if last_output not in source_to_host:
                    raise ValueError(f"No host output for capture output {last_output}")
                host_output = source_to_host[last_output]
                writers[host_output].writerow(
                    {"host_assignment_output": host_output, **event}
                )
        for output, temp in temporary.items():
            temp.replace(final[output])
        completed = True
    finally:
        if not completed:
            for temp in temporary.values():
                temp.unlink(missing_ok=True)


def _write_manifest(rows: list[dict[str, object]], output_root: Path) -> None:
    path = output_root / "hr5_capture_host_manifest.csv"
    json_path = output_root / "hr5_capture_host_manifest.json"
    temporary = path.with_suffix(".csv.tmp")
    temporary_json = json_path.with_suffix(".json.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary_json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    temporary_json.replace(json_path)


def _compile_extractor(repository: Path, executable: Path) -> None:
    compiler = shutil.which("cc")
    if compiler is None:
        raise RuntimeError("No C compiler is available")
    executable.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            compiler,
            "-std=c11",
            "-O3",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-fopenmp",
            "-o",
            str(executable),
            str(repository / "tools" / "extract_hr5_sink_hosts.c"),
            "-lm",
        ],
        check=True,
    )


def _validate_extraction_summary(
    row: dict[str, object], summary: dict[str, object], tag: str
) -> None:
    if int(summary["psb_galaxy_count"]) != int(row["galaxy_count"]):
        raise ValueError(f"Galaxy-count mismatch at output {tag}")
    if int(summary["requested_sink_count"]) != int(row["requested_sink_count"]):
        raise ValueError(f"Sink-selection mismatch at output {tag}")
    if int(summary["selected_sink_count"]) > int(summary["requested_sink_count"]):
        raise ValueError(f"Selected more sinks than requested at output {tag}")
    for field in (
        "duplicate_sink_count",
        "particle_count_mismatches",
        "host_sink_mass_mismatches",
        "metadata_sample_mismatches",
    ):
        if int(summary[field]) != 0:
            raise ValueError(f"{field} is nonzero at output {tag}")


def _extract(row: dict[str, object], executable: Path, threads: int, force: bool) -> None:
    if row["host_catalogue_source"] == "full_canonical_host_catalogue":
        return
    output = int(row["output"])
    tag = f"{output:05d}"
    path = Path(str(row["host_catalogue_path"]))
    if path.is_file() and not force:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = path.parent / "sink_host_extraction.json"
    log_path = path.parent / "sink_host_extraction.log"
    command = [
        str(executable),
        "--data",
        str(row["galfind_path"]),
        "--list",
        str(row["galaxy_list_path"]),
        "--sink-ids",
        str(row["sink_id_path"]),
        "--output",
        str(path),
        "--output-number",
        str(output),
        "--redshift",
        f"{float(row['redshift']):.17g}",
        "--threads",
        str(threads),
    ]
    if force:
        command.append("--force")
    with summary_path.open("w") as stdout, log_path.open("w") as stderr:
        subprocess.run(command, check=True, stdout=stdout, stderr=stderr)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    _validate_extraction_summary(row, summary, tag)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture-catalog",
        type=Path,
        default=Path("results/hr5/hr5_capture_catalog.csv"),
    )
    parser.add_argument(
        "--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_CANONICAL_ROOT / "capture_hosts",
    )
    parser.add_argument("--outputs", nargs="+", type=int)
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--partition-events", action="store_true")
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.jobs < 1:
        parser.error("--jobs must be positive")

    rows = prepare_capture_targets(
        args.capture_catalog,
        args.canonical_root / "hr5_output_manifest.csv",
        args.output_root,
    )
    _write_manifest(rows, args.output_root)
    if args.partition_events:
        partition_capture_events(args.capture_catalog, rows)
        rows = prepare_capture_targets(
            args.capture_catalog,
            args.canonical_root / "hr5_output_manifest.csv",
            args.output_root,
        )
        _write_manifest(rows, args.output_root)
    selected = rows
    if args.outputs:
        requested = set(args.outputs)
        selected = [row for row in rows if int(row["output"]) in requested]
        missing = requested - {int(row["output"]) for row in selected}
        if missing:
            parser.error(f"No possible binary captures at outputs {sorted(missing)}")

    if args.extract:
        repository = Path(__file__).resolve().parents[1]
        executable = args.output_root / "bin" / "extract_hr5_sink_hosts"
        _compile_extractor(repository, executable)
        def run_one(row: dict[str, object]) -> None:
            print(
                f"Extracting capture hosts at output {int(row['output']):05d}",
                flush=True,
            )
            _extract(row, executable, args.threads, args.force)

        if args.jobs == 1:
            for row in selected:
                run_one(row)
        else:
            with ThreadPoolExecutor(max_workers=args.jobs) as executor:
                list(executor.map(run_one, selected))
        rows = prepare_capture_targets(
            args.capture_catalog,
            args.canonical_root / "hr5_output_manifest.csv",
            args.output_root,
        )
        _write_manifest(rows, args.output_root)

    print(
        json.dumps(
            {
                "output_count": len(rows),
                "possible_binary_capture_count": sum(
                    int(row["possible_binary_capture_count"]) for row in rows
                ),
                "unique_output_sink_request_count": sum(
                    int(row["requested_sink_count"]) for row in rows
                ),
                "complete_output_count": sum(
                    row["host_catalogue_status"] == "complete" for row in rows
                ),
                "output_root": str(args.output_root),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

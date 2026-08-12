#!/usr/bin/env python3
"""Inventory HR5 outputs and build reproducible sink--host data products."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_HR5_ROOT = Path(
    "/scratch/kjhan/Hydro/HR5/FoFPSB/02_without_dc/Ver2"
)
DEFAULT_AGN_DIRECTORY = Path(
    "/home/kjhan/BACKUP/GalFinder/"
    "SRC(FoF_PSB_Free_Ver2_Dev)/SRC(AGN)/SRC(MkAGN)/HR5_AGN_DATA"
)
DEFAULT_INFO_DIRECTORY = Path(
    "/scratch/kjhan/Hydro/HR5/SRC(Anal)/C/SRC(HFIND_RAMSES)/"
    "SRC(FoF_PSB_Free_Ver2_Dev)/SRC(AGN)/BinarySMBH/data"
)
GALAXY_INFO_BYTES = 616
MKAGN_HEADER = struct.Struct("<ddi")
FLOAT32 = struct.Struct("<f")
GALAXY_LINK_HEADER = struct.Struct("<fq")
GALAXY_LINK_RECORD_BYTES = 56
OUTPUT_PATTERN = re.compile(r"(\d{5})")


@dataclass
class OutputRecord:
    output: int
    redshift: float | None
    redshift_source: str
    scale_factor: float | None
    ramses_info_path: str
    mkagn_path: str
    mkagn_bytes: int | None
    mkagn_record_count: int | None
    mkagn_record_bytes: int | None
    galfind_path: str
    galfind_bytes: int | None
    galaxy_list_path: str
    galaxy_list_bytes: int | None
    background_path: str
    background_bytes: int | None
    galaxy_catalog_path: str
    galaxy_catalog_bytes: int | None
    galaxy_count: int | None
    galaxy_link_path: str
    galaxy_link_bytes: int | None
    galaxy_link_redshift: float | None
    galaxy_link_record_count: int | None
    next_galaxy_link_output: int | None
    host_extraction_ready: bool
    dual_agn_host_analysis_ready: bool
    derived_directory: str
    sink_host_catalog_path: str
    sink_host_catalog_status: str
    dual_agn_summary_path: str
    dual_agn_summary_status: str


def _path_text(path: Path | None) -> str:
    return str(path) if path is not None else ""


def _file_size(path: Path | None) -> int | None:
    return path.stat().st_size if path is not None and path.is_file() else None


def _output_number(path: Path) -> int | None:
    match = OUTPUT_PATTERN.search(path.name)
    return int(match.group(1)) if match else None


def _discover_numbered(directory: Path, pattern: str) -> dict[int, Path]:
    result: dict[int, Path] = {}
    if not directory.is_dir():
        return result
    for path in directory.glob(pattern):
        output = _output_number(path)
        if output is not None and 0 <= output <= 296:
            result[output] = path
    return result


def _read_mkagn_header(path: Path) -> tuple[float, int, int]:
    with path.open("rb") as stream:
        payload = stream.read(MKAGN_HEADER.size)
    if len(payload) != MKAGN_HEADER.size:
        raise ValueError(f"Incomplete MkAGN header in {path}")
    redshift, _, count = MKAGN_HEADER.unpack(payload)
    if not math.isfinite(redshift) or redshift < 0.0 or count <= 0:
        raise ValueError(f"Invalid MkAGN header in {path}")
    payload_bytes = path.stat().st_size - MKAGN_HEADER.size
    if payload_bytes % count:
        raise ValueError(f"MkAGN payload is not divisible by its record count in {path}")
    return redshift, count, payload_bytes // count


def _read_link_redshift(path: Path) -> float | None:
    with path.open("rb") as stream:
        payload = stream.read(FLOAT32.size)
    if len(payload) != FLOAT32.size:
        return None
    redshift = float(FLOAT32.unpack(payload)[0])
    return redshift if math.isfinite(redshift) and 0.0 <= redshift < 100.0 else None


def _read_link_header(path: Path) -> tuple[float | None, int]:
    with path.open("rb") as stream:
        payload = stream.read(GALAXY_LINK_HEADER.size)
    if len(payload) != GALAXY_LINK_HEADER.size:
        raise ValueError(f"Incomplete GalaxyLinkedList header in {path}")
    redshift, count = GALAXY_LINK_HEADER.unpack(payload)
    expected_bytes = GALAXY_LINK_HEADER.size + count * GALAXY_LINK_RECORD_BYTES
    if count < 0 or path.stat().st_size != expected_bytes:
        raise ValueError(
            f"Unexpected GalaxyLinkedList size for {path}. Expected "
            f"{expected_bytes} bytes and found {path.stat().st_size} bytes."
        )
    valid_redshift = (
        float(redshift)
        if math.isfinite(redshift) and 0.0 <= redshift < 100.0
        else None
    )
    return valid_redshift, int(count)


def _read_scale_factor(path: Path) -> float:
    with path.open(encoding="ascii") as stream:
        for line in stream:
            if line.lstrip().startswith("aexp") and "=" in line:
                scale_factor = float(line.split("=", maxsplit=1)[1])
                if math.isfinite(scale_factor) and 0.0 < scale_factor <= 1.0:
                    return scale_factor
                break
    raise ValueError(f"No valid scale factor in {path}")


def _canonical_fof_files(fof_root: Path, output: int) -> tuple[Path | None, ...]:
    tag = f"{output:05d}"
    directory = fof_root / f"FoF.{tag}"
    if not directory.is_dir():
        return None, None, None
    data = directory / f"GALFIND.DATA.{tag}"
    catalog_list = directory / f"GALCATALOG.LIST.{tag}"
    background = directory / f"background_ptl.{tag}"
    return (
        data if data.is_file() else None,
        catalog_list if catalog_list.is_file() else None,
        background if background.is_file() else None,
    )


def inventory(
    hr5_root: Path,
    agn_directory: Path,
    info_directory: Path,
    output_root: Path,
) -> list[OutputRecord]:
    fof_root = hr5_root / "FoF_Data"
    tree_root = hr5_root / "Galaxy_Merging"
    mkagn = _discover_numbered(agn_directory, "agn.[0-9][0-9][0-9][0-9][0-9].dat")
    galaxy_catalog = _discover_numbered(tree_root, "galaxy_catalog.[0-9]*.dat")
    galaxy_link = _discover_numbered(tree_root, "GalaxyLinkedList.[0-9]*")
    ramses_info = _discover_numbered(info_directory, "info_[0-9][0-9][0-9][0-9][0-9].txt")

    canonical_fof_outputs: set[int] = set()
    if fof_root.is_dir():
        for directory in fof_root.iterdir():
            match = re.fullmatch(r"FoF\.(\d{5})", directory.name)
            if match and 0 <= int(match.group(1)) <= 296:
                canonical_fof_outputs.add(int(match.group(1)))

    outputs = sorted(
        set(mkagn)
        | set(galaxy_catalog)
        | set(galaxy_link)
        | set(ramses_info)
        | canonical_fof_outputs
    )
    records: list[OutputRecord] = []
    link_outputs = sorted(galaxy_link)
    next_link_output = {
        output: link_outputs[index + 1] if index + 1 < len(link_outputs) else None
        for index, output in enumerate(link_outputs)
    }
    for output in outputs:
        tag = f"{output:05d}"
        agn_path = mkagn.get(output)
        link_path = galaxy_link.get(output)
        catalog_path = galaxy_catalog.get(output)
        info_path = ramses_info.get(output)
        data_path, list_path, background_path = _canonical_fof_files(fof_root, output)

        redshift: float | None = None
        redshift_source = ""
        scale_factor = _read_scale_factor(info_path) if info_path is not None else None
        mkagn_count: int | None = None
        mkagn_record_bytes: int | None = None
        if agn_path is not None:
            redshift, mkagn_count, mkagn_record_bytes = _read_mkagn_header(agn_path)
            redshift_source = "MkAGN"
        elif scale_factor is not None:
            redshift = 1.0 / scale_factor - 1.0
            redshift_source = "RAMSES info"
        elif link_path is not None:
            redshift = _read_link_redshift(link_path)
            if redshift is not None:
                redshift_source = "GalaxyLinkedList"

        galaxy_count: int | None = None
        if catalog_path is not None:
            catalog_bytes = catalog_path.stat().st_size
            if catalog_bytes % GALAXY_INFO_BYTES:
                raise ValueError(
                    f"Galaxy catalogue size is not divisible by {GALAXY_INFO_BYTES}: "
                    f"{catalog_path}"
                )
            galaxy_count = catalog_bytes // GALAXY_INFO_BYTES

        link_redshift: float | None = None
        link_record_count: int | None = None
        if link_path is not None:
            link_redshift, link_record_count = _read_link_header(link_path)

        derived = output_root / f"output_{tag}"
        host_path = derived / f"sink_hosts.{tag}.csv"
        host_tmp = Path(f"{host_path}.tmp")
        summary_path = derived / f"hr5_dual_agn_hosts.{tag}.json"
        if host_path.is_file():
            host_status = "complete"
        elif host_tmp.is_file():
            host_status = "partial"
        else:
            host_status = "missing"
        summary_status = "complete" if summary_path.is_file() else "missing"
        host_ready = data_path is not None and list_path is not None and redshift is not None
        dual_ready = host_ready and agn_path is not None
        records.append(
            OutputRecord(
                output=output,
                redshift=redshift,
                redshift_source=redshift_source,
                scale_factor=scale_factor,
                ramses_info_path=_path_text(info_path),
                mkagn_path=_path_text(agn_path),
                mkagn_bytes=_file_size(agn_path),
                mkagn_record_count=mkagn_count,
                mkagn_record_bytes=mkagn_record_bytes,
                galfind_path=_path_text(data_path),
                galfind_bytes=_file_size(data_path),
                galaxy_list_path=_path_text(list_path),
                galaxy_list_bytes=_file_size(list_path),
                background_path=_path_text(background_path),
                background_bytes=_file_size(background_path),
                galaxy_catalog_path=_path_text(catalog_path),
                galaxy_catalog_bytes=_file_size(catalog_path),
                galaxy_count=galaxy_count,
                galaxy_link_path=_path_text(link_path),
                galaxy_link_bytes=_file_size(link_path),
                galaxy_link_redshift=link_redshift,
                galaxy_link_record_count=link_record_count,
                next_galaxy_link_output=next_link_output.get(output),
                host_extraction_ready=host_ready,
                dual_agn_host_analysis_ready=dual_ready,
                derived_directory=str(derived),
                sink_host_catalog_path=str(host_path),
                sink_host_catalog_status=host_status,
                dual_agn_summary_path=str(summary_path),
                dual_agn_summary_status=summary_status,
            )
        )
    return records


def write_manifest(records: list[OutputRecord], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    rows = [asdict(record) for record in records]
    csv_path = output_root / "hr5_output_manifest.csv"
    json_path = output_root / "hr5_output_manifest.json"
    temporary_csv = csv_path.with_suffix(".csv.tmp")
    temporary_json = json_path.with_suffix(".json.tmp")
    with temporary_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary_json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    temporary_csv.replace(csv_path)
    temporary_json.replace(json_path)


def _compile_extractor(repository: Path, executable: Path) -> None:
    compiler = shutil.which("cc")
    if compiler is None:
        raise RuntimeError("No C compiler is available")
    source = repository / "tools" / "extract_hr5_sink_hosts.c"
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
            str(source),
            "-lm",
        ],
        check=True,
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_extraction(
    record: OutputRecord,
    executable: Path,
    threads: int,
    include_background: bool,
    force: bool,
) -> None:
    if record.redshift is None:
        raise ValueError(f"Output {record.output} has no valid redshift")
    derived = Path(record.derived_directory)
    derived.mkdir(parents=True, exist_ok=True)
    host_path = Path(record.sink_host_catalog_path)
    summary_path = derived / "sink_host_extraction.json"
    log_path = derived / "sink_host_extraction.log"
    if host_path.is_file() and not force:
        return
    command = [
        str(executable),
        "--data",
        record.galfind_path,
        "--list",
        record.galaxy_list_path,
        "--output",
        str(host_path),
        "--output-number",
        str(record.output),
        "--redshift",
        f"{record.redshift:.17g}",
        "--threads",
        str(threads),
    ]
    if include_background and record.background_path:
        command.extend(["--background", record.background_path])
    if force:
        command.append("--force")
    with summary_path.open("w") as stdout, log_path.open("w") as stderr:
        subprocess.run(command, check=True, stdout=stdout, stderr=stderr)
    summary = _read_json(summary_path)
    if record.galaxy_count is not None and summary["psb_galaxy_count"] != record.galaxy_count:
        raise ValueError(
            f"Output {record.output} has {summary['psb_galaxy_count']} PSB galaxies "
            f"in GALFIND.DATA but {record.galaxy_count} in galaxy_catalog"
        )


def _run_host_analysis(record: OutputRecord, repository: Path) -> None:
    derived = Path(record.derived_directory)
    command = [
        sys.executable,
        str(repository / "scripts" / "analyze_hr5_dual_agn_hosts.py"),
        record.sink_host_catalog_path,
        record.mkagn_path,
        "--output-directory",
        str(derived),
    ]
    environment = dict(os.environ)
    source_path = str(repository / "src")
    environment["PYTHONPATH"] = (
        source_path
        if not environment.get("PYTHONPATH")
        else source_path + ":" + environment["PYTHONPATH"]
    )
    log_path = derived / "dual_agn_host_analysis.log"
    with log_path.open("w") as stream:
        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT, env=environment)


def write_evolution_summary(records: list[OutputRecord], output_root: Path) -> None:
    rows: list[dict[str, Any]] = []
    for record in records:
        path = Path(record.dual_agn_summary_path)
        if not path.is_file():
            continue
        summary = _read_json(path)
        relation = summary["host_relation_count"]
        rows.append(
            {
                "output": summary["output"],
                "redshift": summary["redshift"],
                "active_smbh_count": summary["active_smbh_count"],
                "active_smbh_with_direct_psb_host": summary[
                    "active_smbh_with_direct_psb_host"
                ],
                "spatially_selected_active_pair_count": summary[
                    "spatially_selected_active_pair_count"
                ],
                "pair_without_two_direct_psb_hosts": relation[
                    "no direct PSB assignment"
                ]
                + relation["sink outside a PSB galaxy"],
                "same_psb_galaxy_pair_count": relation["same PSB galaxy"],
                "distinct_psb_same_fof_pair_count": relation[
                    "distinct PSB galaxies in one FoF halo"
                ],
                "distinct_fof_pair_count": relation["distinct FoF haloes"],
                "pair_count_with_two_psb_hosts": summary[
                    "pair_count_with_two_psb_hosts"
                ],
                "distinct_host_dual_agn_candidate_count": summary[
                    "distinct_host_dual_agn_candidate_count"
                ],
                "distinct_host_fraction_among_pairs_with_two_psb_hosts": summary[
                    "distinct_host_fraction_among_pairs_with_two_psb_hosts"
                ],
                "fable_selection_analogue_pair_count": summary[
                    "fable_selection_analogue"
                ]["pair_count"],
                "fable_selection_analogue_distinct_host_pair_count": summary[
                    "fable_selection_analogue"
                ]["distinct_host_pair_count"],
                "hr5_100_star_particle_pair_count": summary[
                    "hr5_100_star_particle_selection"
                ]["pair_count"],
                "hr5_100_star_particle_distinct_host_pair_count": summary[
                    "hr5_100_star_particle_selection"
                ]["distinct_host_pair_count"],
            }
        )
    rows.sort(key=lambda row: row["redshift"], reverse=True)
    path = output_root / "hr5_dual_agn_host_evolution.csv"
    temporary = path.with_suffix(".csv.tmp")
    if rows:
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    elif path.exists():
        path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hr5-root", type=Path, default=DEFAULT_HR5_ROOT)
    parser.add_argument("--agn-directory", type=Path, default=DEFAULT_AGN_DIRECTORY)
    parser.add_argument("--info-directory", type=Path, default=DEFAULT_INFO_DIRECTORY)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_HR5_ROOT / "Derived_Sink_Hosts" / "canonical_v1",
    )
    parser.add_argument("--outputs", nargs="+", type=int)
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--include-background", action="store_true")
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")

    repository = Path(__file__).resolve().parents[1]
    records = inventory(
        args.hr5_root, args.agn_directory, args.info_directory, args.output_root
    )
    write_manifest(records, args.output_root)
    selected = [record for record in records if record.mkagn_path]
    if args.outputs:
        requested = set(args.outputs)
        selected = [record for record in selected if record.output in requested]
        missing = requested - {record.output for record in selected}
        if missing:
            parser.error(f"No MkAGN data for outputs {sorted(missing)}")

    if args.extract:
        executable = args.output_root / "bin" / "extract_hr5_sink_hosts"
        _compile_extractor(repository, executable)
        for record in selected:
            if not record.host_extraction_ready:
                raise ValueError(f"Output {record.output} lacks a canonical FoF/PSB input")
            print(f"Extracting output {record.output:05d} at z={record.redshift:.8g}", flush=True)
            _run_extraction(
                record,
                executable,
                args.threads,
                args.include_background,
                args.force,
            )
            records = inventory(
                args.hr5_root, args.agn_directory, args.info_directory, args.output_root
            )
            write_manifest(records, args.output_root)

    if args.analyze:
        for record in records:
            if not record.mkagn_path:
                continue
            if args.outputs and record.output not in set(args.outputs):
                continue
            if not Path(record.sink_host_catalog_path).is_file():
                raise ValueError(f"Output {record.output} has no completed sink-host catalog")
            print(f"Analyzing output {record.output:05d}", flush=True)
            _run_host_analysis(record, repository)
            records = inventory(
                args.hr5_root, args.agn_directory, args.info_directory, args.output_root
            )
            write_manifest(records, args.output_root)

    records = inventory(
        args.hr5_root, args.agn_directory, args.info_directory, args.output_root
    )
    write_manifest(records, args.output_root)
    write_evolution_summary(records, args.output_root)
    mkagn_count = sum(bool(record.mkagn_path) for record in records)
    ready_count = sum(record.dual_agn_host_analysis_ready for record in records)
    complete_count = sum(record.dual_agn_summary_status == "complete" for record in records)
    print(
        json.dumps(
            {
                "manifest_output_count": len(records),
                "mkagn_output_count": mkagn_count,
                "dual_agn_host_input_count": ready_count,
                "dual_agn_host_analysis_complete_count": complete_count,
                "output_root": str(args.output_root),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

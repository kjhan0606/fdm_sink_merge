#!/usr/bin/env python3
"""Report completion of the HR5 capture-host and FABLE products."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


DEFAULT_CANONICAL_ROOT = Path(
    "/scratch/kjhan/Hydro/HR5/FoFPSB/02_without_dc/Ver2/"
    "Derived_Sink_Hosts/canonical_v1"
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def report(canonical_root: Path, repository: Path) -> dict[str, object]:
    capture_root = canonical_root / "capture_hosts"
    manifest_path = capture_root / "hr5_capture_host_manifest.csv"
    rows = _read_csv(manifest_path)
    complete = [row for row in rows if Path(row["host_catalogue_path"]).is_file()]
    event_complete = [row for row in rows if Path(row["capture_event_path"]).is_file()]
    temporary = list(capture_root.glob("output_*/*.tmp"))
    descendant_root = canonical_root / "capture_host_descendants"
    products = {
        "event_table": descendant_root
        / "hr5_possible_binary_capture_host_descendants.csv",
        "summary": descendant_root
        / "hr5_possible_binary_capture_host_descendants.json",
        "evolution_table": descendant_root / "hr5_fable_capture_host_evolution.csv",
        "figure": repository / "results/hr5/hr5_fable_capture_host_comparison.pdf",
        "validation": repository
        / "results/hr5/hr5_fable_capture_host_validation.json",
    }
    validation: dict[str, object] | None = None
    if products["validation"].is_file():
        validation = json.loads(products["validation"].read_text(encoding="utf-8"))
    return {
        "report_time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "manifest_output_count": len(rows),
        "capture_event_output_count": len(event_complete),
        "host_catalogue_output_count": len(complete),
        "host_complete_possible_binary_capture_count": sum(
            int(row["possible_binary_capture_count"]) for row in complete
        ),
        "host_complete_requested_sink_count": sum(
            int(row["requested_sink_count"]) for row in complete
        ),
        "temporary_output_count": len(temporary),
        "products": {
            name: {
                "path": str(path),
                "exists": path.is_file(),
                "size_bytes": path.stat().st_size if path.is_file() else 0,
            }
            for name, path in products.items()
        },
        "validated": bool(validation and validation.get("validated") is True),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    parser.add_argument(
        "--repository", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.dumps(report(args.canonical_root, args.repository), indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(args.output)
    print(payload, end="")


if __name__ == "__main__":
    main()

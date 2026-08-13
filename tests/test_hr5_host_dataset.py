from __future__ import annotations

import importlib.util
import json
import struct
import sys
from pathlib import Path

import pytest


def _load_builder():
    repository = Path(__file__).resolve().parents[1]
    path = repository / "scripts" / "build_hr5_host_dataset.py"
    spec = importlib.util.spec_from_file_location("build_hr5_host_dataset", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_inventory_prefers_canonical_fof_and_mkagn_redshift(tmp_path: Path) -> None:
    builder = _load_builder()
    hr5 = tmp_path / "HR5"
    fof = hr5 / "FoF_Data" / "FoF.00117"
    alternative = hr5 / "FoF_Data" / "FoF.00117.mine"
    tree = hr5 / "Galaxy_Merging"
    agn = tmp_path / "AGN"
    derived = tmp_path / "derived"
    for directory in (fof, alternative, tree, agn):
        directory.mkdir(parents=True, exist_ok=True)

    (fof / "GALFIND.DATA.00117").write_bytes(b"canonical")
    (fof / "GALCATALOG.LIST.00117").write_bytes(b"list")
    (fof / "background_ptl.00117").write_bytes(b"background")
    (alternative / "GALFIND.DATA.00117").write_bytes(b"alternative")
    (alternative / "GALCATALOG.LIST.00117").write_bytes(b"list")
    (tree / "galaxy_catalog.00117.dat").write_bytes(bytes(2 * 616))
    (tree / "GalaxyLinkedList.00117").write_bytes(struct.pack("<fq", 1.6, 0))
    (agn / "agn.00117.dat").write_bytes(
        struct.pack("<ddi", 1.4988100669710724, 1.0, 2) + bytes(2 * 360)
    )

    records = builder.inventory(hr5, agn, tmp_path / "missing_info", derived)
    assert len(records) == 1
    record = records[0]
    assert record.output == 117
    assert record.redshift == 1.4988100669710724
    assert record.redshift_source == "MkAGN"
    assert record.mkagn_record_count == 2
    assert record.mkagn_record_bytes == 360
    assert record.galfind_path == str(fof / "GALFIND.DATA.00117")
    assert record.galaxy_count == 2
    assert record.galaxy_link_record_count == 0
    assert record.host_extraction_ready
    assert record.dual_agn_host_analysis_ready
    assert str(alternative) not in record.galfind_path

    builder.write_manifest(records, derived)
    manifest = json.loads((derived / "hr5_output_manifest.json").read_text())
    assert manifest[0]["output"] == 117


def test_inventory_uses_link_redshift_without_mkagn(tmp_path: Path) -> None:
    builder = _load_builder()
    hr5 = tmp_path / "HR5"
    tree = hr5 / "Galaxy_Merging"
    tree.mkdir(parents=True)
    (tree / "galaxy_catalog.00042.dat").write_bytes(bytes(616))
    (tree / "GalaxyLinkedList.00042").write_bytes(struct.pack("<fq", 6.9602, 0))

    records = builder.inventory(
        hr5,
        tmp_path / "missing_agn",
        tmp_path / "missing_info",
        tmp_path / "derived",
    )
    assert len(records) == 1
    assert records[0].redshift_source == "GalaxyLinkedList"
    assert records[0].redshift == pytest.approx(6.9602)
    assert not records[0].dual_agn_host_analysis_ready

#!/usr/bin/env python3
"""Configure and run SKIRT F200W transfer for the six HR5 dual AGN systems."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

from fdm_smbh_delay.hr5_mock_observation import load_throughput_curve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--throughput", type=Path, required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--skirt", type=Path, default=Path("skirt"))
    parser.add_argument("--panels", default="abcdef")
    parser.add_argument("--packets", type=float, default=2.0e6)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--field-pkpc", type=float, default=55.5086998969992)
    parser.add_argument("--mode", choices=("dust", "dust-free"), default="dust")
    parser.add_argument("--agn-beam-opening-angle-deg", type=float, default=30.0)
    parser.add_argument("--emulate", action="store_true")
    parser.add_argument("--configure-only", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def xml_path(path: Path) -> str:
    return escape(str(path.resolve()), {'"': "&quot;"})


def write_band_file(source: Path, destination: Path) -> tuple[float, float]:
    curve = load_throughput_curve(source)
    selected = curve.throughput > 0.0
    wavelength = curve.wavelength_micron[selected]
    throughput = curve.throughput[selected]
    if len(wavelength) < 2:
        raise ValueError("F200W throughput has fewer than two positive samples")
    destination.parent.mkdir(parents=True, exist_ok=True)
    header = "\n".join(
        [
            "JWST NIRCam F200W observer-frame throughput",
            "Column 1: wavelength (micron)",
            "Column 2: transmission (1)",
        ]
    )
    np.savetxt(
        destination,
        np.column_stack((wavelength, throughput)),
        fmt="%.10e",
        header=header,
        comments="# ",
    )
    return float(wavelength.min()), float(wavelength.max())


def dust_domain(dust_path: Path) -> tuple[np.ndarray, np.ndarray]:
    cells = np.loadtxt(dust_path, comments="#", usecols=range(6), ndmin=2)
    lower = np.min(cells[:, :3], axis=0)
    upper = np.max(cells[:, 3:6], axis=0)
    width = upper - lower
    padding = np.maximum(0.01 * width, 1.0)
    return lower - padding, upper + padding


def point_source(
    position: np.ndarray, luminosity_erg_s: float, beam_opening_angle_deg: float
) -> str:
    return f"""
                    <PointSource positionX="{position[0]:.10e} pc" positionY="{position[1]:.10e} pc"
                                 positionZ="{position[2]:.10e} pc">
                        <sed type="SED">
                            <QuasarSED/>
                        </sed>
                        <normalization type="LuminosityNormalization">
                            <IntegratedLuminosityNormalization wavelengthRange="All"
                                integratedLuminosity="{luminosity_erg_s:.10e} erg/s"/>
                        </normalization>
                        <angularDistribution type="AngularDistribution">
                            <ConicalAngularDistribution
                                openingAngle="{beam_opening_angle_deg:.10g} deg"
                                symmetryX="0" symmetryY="0" symmetryZ="1"/>
                        </angularDistribution>
                        <polarizationProfile type="PolarizationProfile">
                            <NoPolarizationProfile/>
                        </polarizationProfile>
                    </PointSource>"""


def render_ski(
    panel: dict[str, object],
    band_path: Path,
    redshift: float,
    hubble: float,
    omega_m: float,
    band_limits_observed: tuple[float, float],
    packets: float,
    image_size: int,
    field_pkpc: float,
    mode: str,
    agn_beam_opening_angle_deg: float,
) -> str:
    files = panel["files"]
    assert isinstance(files, dict)
    star_path = Path(files["stars"]["path"])
    dust_path = Path(files["dust_cells"]["path"])
    agn_path = Path(files["agn"]["path"])
    agn = np.loadtxt(agn_path, comments="#", ndmin=2)
    if agn.shape != (2, 5):
        raise ValueError(f"{agn_path} must contain exactly two AGN")
    rest_min = band_limits_observed[0] / (1.0 + redshift)
    rest_max = band_limits_observed[1] / (1.0 + redshift)
    if not 0.0 < agn_beam_opening_angle_deg <= 90.0:
        raise ValueError("AGN beam half-opening angle must lie in (0, 90] degrees")
    sources = "".join(
        point_source(row[:3], float(row[3]), agn_beam_opening_angle_deg) for row in agn
    )

    if mode == "dust":
        lower, upper = dust_domain(dust_path)
        medium = f"""
            <mediumSystem type="MediumSystem">
                <MediumSystem>
                    <photonPacketOptions type="PhotonPacketOptions">
                        <PhotonPacketOptions forceScattering="true" minWeightReduction="1e4"
                                             minScattEvents="0" pathLengthBias="0.5"/>
                    </photonPacketOptions>
                    <media type="Medium">
                        <CellMedium filename="{xml_path(dust_path)}" massType="Mass" massFraction="1"
                                    importMetallicity="false" importTemperature="false"
                                    importVelocity="false" importMagneticField="false"
                                    importVariableMixParams="false">
                            <materialMix type="MaterialMix">
                                <WeingartnerDraineDustMix environment="MilkyWay" numSilicateSizes="10"
                                    numGraphiteSizes="10" numPAHSizes="5"/>
                            </materialMix>
                        </CellMedium>
                    </media>
                    <samplingOptions type="SamplingOptions">
                        <SamplingOptions numDensitySamples="100" numPropertySamples="1"/>
                    </samplingOptions>
                    <grid type="SpatialGrid">
                        <PolicyTreeSpatialGrid treeType="OctTree"
                            minX="{lower[0]:.10e} pc" maxX="{upper[0]:.10e} pc"
                            minY="{lower[1]:.10e} pc" maxY="{upper[1]:.10e} pc"
                            minZ="{lower[2]:.10e} pc" maxZ="{upper[2]:.10e} pc">
                            <policy type="TreePolicy">
                                <SiteListTreePolicy minLevel="2" maxLevel="12" numExtraLevels="0"/>
                            </policy>
                        </PolicyTreeSpatialGrid>
                    </grid>
                </MediumSystem>
            </mediumSystem>"""
        simulation_mode = "ExtinctionOnly"
    else:
        medium = ""
        simulation_mode = "NoMedium"

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!-- HR5 dual AGN F200W transfer with SKIRT -->
<skirt-simulation-hierarchy type="MonteCarloSimulation" format="9">
    <MonteCarloSimulation userLevel="Expert" simulationMode="{simulation_mode}" numPackets="{packets:.10e}">
        <random type="Random">
            <Random seed="0"/>
        </random>
        <units type="Units">
            <ExtragalacticUnits wavelengthOutputStyle="Wavelength" fluxOutputStyle="Frequency"/>
        </units>
        <cosmology type="Cosmology">
            <FlatUniverseCosmology redshift="{redshift:.12g}" reducedHubbleConstant="{hubble:.12g}"
                                   matterDensityFraction="{omega_m:.12g}"/>
        </cosmology>
        <sourceSystem type="SourceSystem">
            <SourceSystem minWavelength="{rest_min:.10e} micron" maxWavelength="{rest_max:.10e} micron"
                          sourceBias="0.5">
                <sources type="Source">
                    <ParticleSource filename="{xml_path(star_path)}" importVelocity="false"
                                    importVelocityDispersion="false" importCurrentMass="false"
                                    importBias="false">
                        <smoothingKernel type="SmoothingKernel">
                            <CubicSplineSmoothingKernel/>
                        </smoothingKernel>
                        <sedFamily type="SEDFamily">
                            <FSPSSEDFamily imf="Kroupa"/>
                        </sedFamily>
                    </ParticleSource>{sources}
                </sources>
            </SourceSystem>
        </sourceSystem>{medium}
        <instrumentSystem type="InstrumentSystem">
            <InstrumentSystem>
                <defaultWavelengthGrid type="WavelengthGrid">
                    <ConfigurableBandWavelengthGrid>
                        <bands type="Band">
                            <FileBand filename="{xml_path(band_path)}"/>
                        </bands>
                    </ConfigurableBandWavelengthGrid>
                </defaultWavelengthGrid>
                <instruments type="Instrument">
                    <FrameInstrument instrumentName="f200w" distance="0 Mpc"
                                     inclination="0 deg" azimuth="0 deg" roll="90 deg"
                                     fieldOfViewX="{field_pkpc:.10e} kpc" numPixelsX="{image_size}"
                                     centerX="0 pc" fieldOfViewY="{field_pkpc:.10e} kpc"
                                     numPixelsY="{image_size}" centerY="0 pc"
                                     recordComponents="true" numScatteringLevels="0"
                                     recordPolarization="false" recordStatistics="true"/>
                </instruments>
            </InstrumentSystem>
        </instrumentSystem>
        <probeSystem type="ProbeSystem">
            <ProbeSystem/>
        </probeSystem>
    </MonteCarloSimulation>
</skirt-simulation-hierarchy>
"""


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    redshift = float(manifest["redshift"])
    hubble = 0.684
    omega_m = 0.3
    args.run_directory.mkdir(parents=True, exist_ok=True)
    band_path = args.run_directory / "f200w_observer_band.txt"
    band_limits = write_band_file(args.throughput, band_path)
    requested = set(args.panels)
    available = {str(panel["panel"]): panel for panel in manifest["panels"]}
    if not requested or not requested <= available.keys():
        raise ValueError("Requested panels must be selected from a through f")

    products: list[dict[str, object]] = []
    for panel_name in args.panels:
        panel = available[panel_name]
        ski_path = args.run_directory / f"panel_{panel_name}_{args.mode}.ski"
        output_directory = args.run_directory / f"panel_{panel_name}_{args.mode}"
        output_directory.mkdir(parents=True, exist_ok=True)
        ski_path.write_text(
            render_ski(
                panel,
                band_path,
                redshift,
                hubble,
                omega_m,
                band_limits,
                args.packets,
                args.image_size,
                args.field_pkpc,
                args.mode,
                args.agn_beam_opening_angle_deg,
            ),
            encoding="utf-8",
        )
        command = [
            str(args.skirt),
            "-t",
            str(args.threads),
            "-o",
            str(output_directory),
        ]
        if args.emulate:
            command.append("-e")
        command.append(str(ski_path))
        if not args.configure_only:
            subprocess.run(command, check=True)
        products.append(
            {
                "panel": panel_name,
                "mode": args.mode,
                "ski": str(ski_path),
                "ski_sha256": sha256(ski_path),
                "output_directory": str(output_directory),
                "command": command,
            }
        )

    run_manifest = {
        "status": "configured" if args.configure_only else "complete",
        "emulation": args.emulate,
        "input_manifest": str(args.input_manifest),
        "input_manifest_sha256": sha256(args.input_manifest),
        "throughput": str(args.throughput),
        "throughput_sha256": sha256(args.throughput),
        "band_file": str(band_path),
        "band_file_sha256": sha256(band_path),
        "redshift": redshift,
        "cosmology": {"h": hubble, "omega_m": omega_m},
        "packets": args.packets,
        "threads": args.threads,
        "image_size": args.image_size,
        "field_pkpc": args.field_pkpc,
        "agn_angular_emission": {
            "model": "observer-aligned biconical beam",
            "half_opening_angle_deg": args.agn_beam_opening_angle_deg,
            "axis": "simulation z axis",
            "on_axis_intensity_relative_to_isotropic": float(
                1.0 / (1.0 - np.cos(np.deg2rad(args.agn_beam_opening_angle_deg)))
            ),
            "total_bolometric_luminosity_conserved": True,
        },
        "products": products,
    }
    manifest_path = args.run_directory / f"hr5_dual_agn_skirt_{args.mode}_run.json"
    manifest_path.write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(run_manifest, indent=2))


if __name__ == "__main__":
    main()

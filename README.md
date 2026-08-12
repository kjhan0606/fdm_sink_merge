# FDM SMBH Delay

`fdm-smbh-delay` estimates the unresolved delay between a parsec-scale
supermassive-black-hole (SMBH) binary and a separation of `0.01 pc` inside a
fuzzy-dark-matter (FDM) soliton.

It is the delay-model component of the working paper **Pulsar Timing Array in
Fuzzy Dark Matter Model**, which connects SMBH coalescence delays, cosmological
event populations, and PTA signals. The manuscript is maintained separately in
the Overleaf repository recorded in
[`docs/project_links.md`](docs/project_links.md).

The package is a **parameterized toy model and unresolved inspiral
prescription**, not a first-principles prediction of physical coalescence.
Published separation curves provide comparison cases, but the adopted soliton
profiles still require calibration against the corresponding simulations. In
particular, a merger of sink particles in lagRamses is a resolution-scale
numerical event and must not be called an SMBH coalescence.

## v0.1 scope

The first model integrates

```text
1 pc --(static soliton + analytic FDM wave drag)--> 0.01 pc
```

It does not yet predict the preceding numerical-radius-to-parsec inspiral. A
Peters gravitational-wave time is available below the FDM interval. The orbit
calculation returns a crossing time or a censored timeout, an orbital time
series, energy and momentum transfer, and validity flags.

The scientific design and lagRamses context are preserved in
[`docs/HANDOFF.md`](docs/HANDOFF.md).
Repository locations, including the Overleaf paper remote, are recorded in
[`docs/project_links.md`](docs/project_links.md).

## Install

Python 3.11 or newer is required.

```bash
python -m pip install -e '.[dev]'
pytest
```

The public configuration interface requires explicit units. Bare numeric
values are rejected for dimensional quantities.

## Run

```bash
fdm-smbh-delay configs/lagramses_m22_example.yaml --output results/m22_example
```

or, without installing the console script:

```bash
python scripts/run_case.py configs/lagramses_m22_example.yaml \
  --output results/m22_example
```

The output directory contains:

- `summary.json`: status, delay, energy error, validity flags, and provenance
- `timeseries.csv`: orbit, energy, angular momentum, density, and drag metrics
- `config.yaml`: exact input configuration used for the calculation

`timeout` is a valid censored physical result, not a numerical failure.
The transfer ledger is not yet a live wavefunction update; the coupling and
double-counting rules are documented in
[`docs/wave_energy_coupling.md`](docs/wave_energy_coupling.md).

To combine a sink time with all three physical intervals, use:

```bash
fdm-smbh-compose \
  --z-sink 1.0 \
  --fdm-summary results/case/summary.json \
  --kpc-to-pc-delay "100 Myr" \
  --gw-delay "3 Myr"
```

The command returns no `true_merge_time_myr` if an interval is missing,
invalid, or censored. Missing physics is never interpreted as zero delay.

## Example configuration

```yaml
model:
  name: wave_df_3d
  alpha_df: 0.341
  drag: true
  fdm_bulk_velocity: ["0 km/s", "0 km/s", "0 km/s"]

binary:
  M1: "5.0e7 Msun"
  M2: "5.0e7 Msun"
  separation: "1 pc"
  eccentricity: 0.0
  orbit: circular

fdm:
  particle_mass: "1.0e-21 eV"
  soliton_mass: "1.0e9 Msun"
  mass_definition: total_profile
  core_radius: "2 pc"
  profile: schive_fit

integration:
  stop_separation: "0.01 pc"
  max_time: "20 Myr"
  output_samples: 1000
  rtol: 1.0e-9
  atol: 1.0e-12
```

The example core radius is a schema demonstration, not a validated literature
fiducial value. Published comparisons require matching each paper's soliton
mass and core-radius definitions.

## Development status

The repository currently implements the static-soliton three-dimensional orbit
calculation, FDM energy and momentum transfer, Peters gravitational-wave
times, conservative composition of the physical coalescence time, and a source
study of the lagRamses conditions for a merger of sink particles. Koo and Boey separation curves
are included as comparison cases. Calibration against their simulated density
histories, the numerical-radius-to-parsec inspiral, an evolving FDM
wavefunction, and the cosmological PTA population remain future work.

## Horizon Run 5 comparison sample

The legacy HR5 sink tree can be regenerated as an explicit catalog of binary
captures with interval-censored times.
The two output boundaries are retained because a disappearing sink is last
resolved at output `i-1`, whereas the assigned surviving SMBH is selected from
the population at output `i`. The files retain `receiver_id` as a historical
field name.

```bash
python scripts/extract_hr5_capture_catalog.py \
  '/home/kjhan/BACKUP/GalFinder/SRC(FoF_PSB_Free_Ver2_Dev)/SRC(AGN)/BinarySMBH/Sink_Merging_Tree.dat.Updated'

python scripts/reproduce_hr5_original_figures.py --rebuild-cache
python scripts/validate_hr5_capture_receivers.py
python scripts/analyze_hr5_dual_agn_redshift.py
python scripts/analyze_hr5_dual_agn.py
python scripts/plot_hr5_capture_histories.py
```

The first command produces 576,278 binary-capture intervals from 1,688,677 sink
histories. The second command redraws Figures 1--13 of the earlier HR5 draft.
The third command tests the assigned surviving SMBHs against the phase-space
states and consecutive MkAGN outputs. The fourth command measures the
redshift evolution of spatially selected dual AGN candidates in all available
MkAGN snapshots. It also fits a local quadratic model at every plotted
redshift. The fifth command performs the detailed active-pair analysis at
outputs 89, 117, and 296, estimates spatial variance with an eight-region
jackknife, and constructs matched pure two-member comparisons with interval
and right censoring. The sixth command draws the assigned capture hierarchy of
the most massive final SMBH and the three-dimensional trajectories associated
with the most massive member of the dual-AGN sample at output 117.

The disappearing sinks are measured directly, but the assigned surviving SMBHs
come from distance and mass criteria rather than direct records of the capture
partners. The possible binary captures are not physical SMBH coalescences. The event
definitions, validation of assigned companions, active-pair selection, and literature
comparison are documented in
[`docs/hr5_reproduction.md`](docs/hr5_reproduction.md).

The original FoF/PSB galaxy-finder products can also recover the direct
association between a sink particle and its host galaxy. The optimized reader
skips the dark matter, gas, and star payloads and reads only the saved sink
records:

```bash
cc -std=c11 -O3 -Wall -Wextra -Werror -fopenmp \
  -o /tmp/extract_hr5_sink_hosts tools/extract_hr5_sink_hosts.c -lm

/tmp/extract_hr5_sink_hosts \
  --data /scratch/kjhan/Hydro/HR5/FoFPSB/02_without_dc/Ver2/FoF_Data/FoF.00117/GALFIND.DATA.00117 \
  --list /scratch/kjhan/Hydro/HR5/FoFPSB/02_without_dc/Ver2/FoF_Data/FoF.00117/GALCATALOG.LIST.00117 \
  --output /scratch/kjhan/Hydro/HR5/FoFPSB/02_without_dc/Ver2/Derived_Sink_Hosts/sink_hosts.00117.csv \
  --output-number 117 --redshift 1.4988132 --threads 8
```

The byte layout, validation criteria, and current output coverage are given in
[`docs/hr5_reproduction.md`](docs/hr5_reproduction.md#direct-association-of-sink-particles-with-psb-galaxies).

Inventory all 297 RAMSES outputs and build canonical sink--host products for
the 17 outputs that also have MkAGN records with:

```bash
PYTHONPATH=src python scripts/build_hr5_host_dataset.py
PYTHONPATH=src python scripts/build_hr5_host_dataset.py \
  --extract --analyze --threads 32

PYTHONPATH=src python scripts/analyze_hr5_host_descendants.py

PYTHONPATH=src python scripts/build_hr5_agn_pair_host_dataset.py

PYTHONPATH=src python scripts/build_hr5_capture_host_dataset.py --partition-events
PYTHONPATH=src python scripts/build_hr5_capture_host_dataset.py \
  --extract --jobs 4 --threads 8
PYTHONPATH=src python scripts/analyze_hr5_capture_hosts.py
python scripts/plot_hr5_fable_comparison.py

PYTHONPATH=src python scripts/analyze_hr5_matched_pair_hosts.py
python scripts/plot_hr5_matched_pair_hosts.py

PYTHONPATH=src python scripts/analyze_hr5_matched_pair_hosts.py \
  --require-fable-selection \
  --output-directory /scratch/kjhan/Hydro/HR5/FoFPSB/02_without_dc/Ver2/Derived_Sink_Hosts/canonical_v1/matched_pair_host_descendants_fable
```

The batch calculation accepts only directories named exactly `FoF.NNNNN`.
The active-pair host calculation follows both directly assigned PSB galaxies
through all 129 available galaxy outputs. It records the interval in which they first acquire
a common descendant, retains systems that remain distinct at `z=0.625` as
right-censored, and applies the FABLE SMBH-mass threshold together with an HR5
host-stellar-mass analogue. The stellar-mass value is the same as FABLE's
published cut, although HR5 uses total PSB stellar mass rather than the FABLE
aperture within twice the stellar half-mass radius. A separate flag retains the
HR5 threshold of 100 directly counted stellar particles.
The AGN-pair host builder applies the same SMBH-mass, luminosity, and separation
selection at all 17 MkAGN outputs. It records dual and single-AGN pairs together
with both SMBHs' phase-space quantities and the directly assigned stellar and
gas properties of both PSB galaxies. This systematic table is the default input
to the matched host analysis and does not use the historical assigned capture
companion.
The capture-host builder groups all 576,278 possible binary captures by the
latest available preceding galaxy output and writes the 1,128,422 unique
output--SMBH requests needed for an event-level host calculation. The second
builder command generates the missing filtered host catalogues without reading
directories whose names contain `.mine`, `.test`, or `.try`.
The following analysis measures whether the two assigned PSB hosts acquire a
common descendant before or after each possible binary capture. It keeps both
time intervals and ambiguous galaxy links explicit. At the 17 outputs with
MkAGN data, it also records both SMBH luminosities and Eddington ratios and
separates events with two, one, or no active SMBHs.
The plotting command writes the redshift-dependent event count and timing
fractions, together with FABLE's published no-added-delay fraction, to
`results/hr5/hr5_fable_capture_host_comparison.pdf`.
The descendant analysis also writes the compact 121-row table
`capture_host_descendants/hr5_fable_capture_host_evolution.csv` alongside the
event-level catalogue and JSON summary.
The HR5 fractions use every event that passes the FABLE-selection analogue as
the denominator, matching the denominator of the published FABLE fraction.
Their lower and upper limits retain interval overlap and unresolved host times.
The event counts are counts per stored output interval rather than rates.
The matched analysis provides an assigned-companion-independent comparison of
dual AGN with single-AGN controls. It starts with two distinct, directly
identified PSB hosts and uses a host-property propensity score. Outputs that
fail the stated post-match balance criterion are excluded. The analysis retains
interval bounds for the time at which the two hosts acquire a common descendant.
The first figure is
`results/hr5/hr5_matched_pair_host_evolution.pdf`.
The final command repeats the calculation after imposing the FABLE SMBH and
host-stellar-mass thresholds before matching.
The FABLE comparison is defined in
[`docs/hr5_fable_comparison.md`](docs/hr5_fable_comparison.md).

The output-296 host particles can also be rendered as a six-panel NIRCam F200W
morphology mock for representative dual AGN systems. The calculation uses the
official STScI F200W PSF and retains a separate projected gas metal-mass layer,
but the displayed host light is deliberately dust-free and non-photometric.
The target selection, particle extraction, attenuation requirements, and
output layers are documented in
[`docs/hr5_dual_agn_jwst_mock.md`](docs/hr5_dual_agn_jwst_mock.md).
The same document describes the unit-source PSF regression test, the separate
FSPS-calibrated F200W quick look, its foreground-screen dust and scattering
preview, and the final three-dimensional SKIRT calculation. The SKIRT image
uses the HR5 stellar and AMR dust geometry, bolometrically normalized quasar
spectra in observer-aligned 30-degree biconical beams, $10^7$ photon packets
per system, and the detector-sampled F200W PSF. A common base-10 logarithmic
surface-brightness scale is used for the six panels.
Oppositely directed arrows perpendicular to each projected AGN-pair axis mark
the two active nuclei. The original non-photometric image remains the
regression baseline.

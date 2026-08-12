# Horizon Run 5 binary-capture and active SMBH pair analysis

## Data provenance

The comparison sample is derived from the legacy HR5 sink tree

```text
/home/kjhan/BACKUP/GalFinder/SRC(FoF_PSB_Free_Ver2_Dev)/SRC(AGN)/BinarySMBH/Sink_Merging_Tree.dat.Updated
```

The file contains 1,688,677 fixed sink histories over 278 outputs. The extraction
finds 576,278 binary captures from disappearing SMBH particles. Of these events,
576,036 have positive masses for both objects at the last resolved output.

The companion-selection calculation follows `mkmerging.c`. A sink
that is present at output `i-1` and absent at output `i` is matched to a sink at
`i`. The search radius starts at the nearest-survivor distance, increases by
0.002 cMpc, and stops at 0.5 cMpc. Within each radius, the most massive survivor
with at least twice the removed sink mass is selected. This procedure writes
`mergeid` and `mergeistep` into `Sink_Merging_Tree.dat.Updated`; it does not read
a partner identifier from a RAMSES event record. The surviving SMBH is therefore
an assigned companion rather than a directly recorded participant in the
numerical binary capture. The catalogue keeps `receiver_id` as the historical
field name.

## Output-step convention

Every event is an interval. No single column silently mixes the two boundaries.

| Quantity | Output | Interpretation |
|---|---:|---|
| disappearing sink mass, position, velocity | `i-1` | last resolved state |
| mass of the assigned surviving SMBH used for mass ratio and chirp mass | `i-1` | two-object mass estimate before disappearance |
| identifier of the surviving SMBH | selected at `i` | assigned from the surviving population |
| mass of the surviving SMBH at the assigned output | `i` | post-disappearance diagnostic, not used in chirp mass |
| binary-capture time | between `i-1` and `i` | interval-censored event |
| assigned capture time and redshift | `i` | upper boundary of the interval |

The catalog stores both history indices, output numbers, redshifts, cosmic
times, and the interval width. A later delay condition such as capture within
1 Gyr is accepted only when the upper boundary meets the condition. If the
1 Gyr boundary falls inside the output interval, the case is recorded
separately.

## Validation of assigned companions

`scripts/validate_hr5_capture_receivers.py` reads the assigned surviving SMBH at both
interval boundaries and compares its last-resolved state with the disappearing
sink. The two-body diagnostic uses

```text
v_escape = sqrt(2 G (M_disappearing + M_surviving) / separation)
```

All 576,278 associations satisfy the 0.5 cMpc search limit and the companion
mass factor of two because those conditions define the legacy selection. Of
576,277 associations with finite last-resolved phase-space quantities, only 41
have `v_relative <= v_escape`. The fraction is `7.1146e-5`. The median
separation at the last common output is 4.614 pkpc, and the median relative
speed is 218.2 km/s. Also, 6.54 percent of the possible binary captures assign the
same surviving SMBH to at least one other disappearance in the same output.

The locally available consecutive MkAGN outputs span 20 through 26. Only one
sink disappears across those intervals. Reconstructing its companion from the
two adjacent MkAGN snapshots reproduces the legacy tree assignment exactly.
This single event tests the implementation but does not independently validate
the full association catalogue.

The phase-space result separates two measurements. Sink disappearance and its
output interval come directly from the histories. The identity of the assigned companion, pair
mass, chirp mass, and relative orbit remain conditional on the legacy
association until direct records of the merger of two sink particles are restored. The validation
products are written under `results/hr5/receiver_validation/`.

## Regeneration

```bash
python scripts/extract_hr5_capture_catalog.py \
  '/home/kjhan/BACKUP/GalFinder/SRC(FoF_PSB_Free_Ver2_Dev)/SRC(AGN)/BinarySMBH/Sink_Merging_Tree.dat.Updated'

python scripts/reproduce_hr5_original_figures.py --rebuild-cache
python scripts/validate_hr5_capture_receivers.py
python scripts/analyze_hr5_dual_agn_redshift.py
python scripts/analyze_hr5_dual_agn.py
python scripts/plot_hr5_capture_histories.py
```

The full tree can also be rebuilt from consecutive MkAGN snapshots with
`scripts/build_hr5_binary_capture_history.py`. The local MkAGN directory does
not contain all outputs from 19 through 296. It has a consecutive sequence from
20 through 26 and sparse later snapshots. The builder refuses missing output
intervals by default so that an incomplete sequence cannot be interpreted as a
physical disappearance history.

## Original figures

`scripts/reproduce_hr5_original_figures.py` redraws the 13 figures in the
earlier draft with external PDF graphics. It also writes
`hr5_original_figure_validation.json`. The validation compares the regenerated
values with numerical statements in the draft. The original point series are
not available in a machine-readable table, so an exact pointwise residual
cannot be claimed.

`scripts/plot_hr5_capture_population.py` writes the capture-rate and chirp-mass
distribution to `hr5_capture_population.pdf`. The lower horizontal axes use
`log10(1 + z_cap)`, and the upper axes give the corresponding redshift. The
two-dimensional counts use equal intervals in `log10(1 + z_cap)`. The
mass-ratio probability density is written separately to
`hr5_capture_mass_ratio.pdf`.

`scripts/plot_hr5_capture_histories.py` writes two diagnostic figures under
`results/hr5/capture_histories/`. The first traces the hierarchy assigned to
SMBH 11799, which is the most massive SMBH at the final HR5 output. The
hierarchy contains 121 sink particles and 120 assigned links. Twenty-eight
removed SMBHs are assigned directly to SMBH 11799, and the deepest branch has
four links. The second figure follows SMBH 33570, the most massive member of
the dual-AGN sample at output 117, together with the 16 removed SMBHs assigned
directly to it. The figure gives root-centered physical coordinates for the
final continuous portion of each path within 50 pkpc and during the preceding
1 Gyr. Each coordinate is fitted against cosmic time with a polynomial of
degree no greater than three within a smoothing spline. The combined target
residual in three dimensions is 3 pkpc. The fit uses a lower degree when fewer
than four positions are available and is omitted for a single position. Every
fit is limited to the sampled time interval and is constrained to the measured
end points. Marker area gives SMBH mass while marker and curve colors give
redshift. The dual classification applies only at output 117. The fitted paths
do not verify the physical capture partner and do not establish SMBH
coalescence.

In the hierarchy, circle area scales linearly with the logarithm of the
recorded SMBH mass and the interior color follows the same mass scale. Circles
mark formation after $z=7$, the final resolved output before numerical
removal, and the surviving SMBH in the next output after an assigned capture.
Repeated captures at one output share a single mass marker for the surviving
SMBH. The scale at right contains 13 circles in intervals of 0.5 dex and
labels each full decade from $10^4$ through $10^{10}\,M_\odot$.

The regenerated seed-rate peak, massive-SMBH captured-mass fraction, and
fixed-delay rate normalization satisfy the reported or visually readable
benchmarks. The legacy cumulative count for the $10^{6}\,M_\odot$ chirp-mass
threshold does not satisfy the value stated in the earlier draft. The
validation file keeps the discrepancy visible rather than tuning a hidden
normalization.

Figures 12 and 13 use the physical all-sky differential comoving volume with
`Omega=4 pi sr`, equivalent to 41,253 square degrees. The earlier draft wrote an
additional factor of one third in the shell expression. The validation output
retains that legacy normalization for comparison, but it is not used in the new
PTA source counts. Counts for another solid angle follow
`N(Omega)=N_all-sky Omega/(4 pi)`.

Figures 10 and 11 show count-statistical uncertainties in the fitted redshift
distribution. Each redshift-bin count is independently resampled from a
Poisson distribution 200 times. The four distribution parameters are refitted
for every realization. Symbols mark the bootstrap medians, and error bars span
the 16th through 84th percentiles. The calculation does not include cosmic
variance, output-time uncertainty, or ambiguity in the assignment of the
surviving SMBH.

Figures 8 and 9, which appear as Figures 11 and 12 in the current JKAS draft,
show the same count-statistical uncertainty for each measured redshift-bin
rate. The vertical bars span the 16th through 84th percentiles of 200
independent Poisson realizations. The measured rates remain at the symbol
positions. The numerical values are written to
`hr5_fixed_delay_rate_bootstrap.csv`.

Figures 9 and 10 use 6-point labels, ticks, panel letters, and legends after the
requested 40 percent reduction. Figure 3 uses 6-point axis labels and tick
labels while retaining 10-point panel letters and legends. The remaining
manuscript graphics use 10-point text, open geometric symbols, and a
color-vision-deficiency-safe palette. Line styles duplicate the color encoding
for grayscale reproduction. Legends remain inside the axes and occupy regions
separated from the measured curves and uncertainty intervals.

## Active SMBH pair selection and measurements

The redshift-evolution measurement uses all 17 available MkAGN snapshots from
output 20 through output 296. The detailed active-pair calculation that
includes projected observables, later possible binary captures, and matched
single-AGN pairs uses outputs 89, 117, and 296.
Both members of a dual AGN candidate satisfy the adopted bolometric-luminosity
threshold. Exactly one member of a single-AGN pair satisfies the threshold.
Every pair has a three-dimensional physical separation between 0.5 and
30 pkpc. Three fractional measures are retained because published measurements
use different denominators. `N_pair/N_AGN` counts pair edges.
`N_member/N_AGN` counts every unique active SMBH with an active companion. The
pure-dual measure removes members of connected systems containing three or
more active SMBHs.

The available snapshots use three historical record lengths of 200, 336, and
360 bytes. Every layout stores bolometric luminosity. The two longer layouts
also store hard-X-ray luminosity while the 200-byte layout requires the
bolometric correction implemented by the original MkAGN calculation. Applying
the same correction to output 89 reproduces all 161,626 positive saved
hard-X-ray luminosities to a maximum relative difference of
`5.66e-16`.

The fiducial selection first yields a nonzero pair count at redshift 8.666.
The measured number density reaches `6.430e-4 cMpc^-3` at redshift 3.394 and
decreases to `3.588e-6 cMpc^-3` at redshift 0.625. The files
`hr5_dual_agn_redshift_evolution.csv` and
`hr5_dual_agn_redshift_fits.csv` contain the direct measurements and the
displayed fits. The number densities follow the modified Schechter-like form
`n_X(z) = phi_star (z/z_star)^alpha exp[-(z/z_star)^beta]`. The file
`hr5_dual_agn_redshift_modified_schechter_parameters.csv` gives the four global
parameters and the logarithmic residuals for the active-SMBH and dual-AGN
number densities. The fits use outputs with at least three dual AGN candidates
and remain confined to the sampled redshift interval. The fractions retain a
locally weighted quadratic interpolation through the seven nearest qualifying
outputs. Zero-count outputs are shown as 95 percent Poisson upper limits.

| redshift | active AGN | dual pairs | number density [cMpc^-3] | pair/AGN | member/AGN | pure member/AGN |
|---:|---:|---:|---:|---:|---:|---:|
| 2.848 | 77,805 | 2,532 | 2.329e-4 | 0.03254 | 0.06110 | 0.05609 |
| 1.499 | 29,446 | 364 | 3.349e-5 | 0.01236 | 0.02343 | 0.02187 |
| 0.625 | 8,179 | 39 | 3.588e-6 | 0.00477 | 0.00954 | 0.00954 |

The table adopts `Lbol >= 1e43 erg s^-1`. The calculation also measures
`Lbol >= 1e44 erg s^-1` and hard-X-ray `L(2-10 keV) >= 1e42 erg s^-1`
selections. Eight equal slabs along the long axis give spatial jackknife
errors on the fiducial number densities of `1.743e-5`, `3.874e-6`, and
`5.808e-7 cMpc^-3` at the three redshifts. A separate comparison imposes
`M_BH >= 1e6 Msun` on both members. At redshift 2.848 it contains 387
dual-active and 574 single-active pairs. At redshift 1.499 it contains 295
dual-active and 1,336 single-active pairs.

The projected-selection calculation follows each physically associated
three-dimensional pair over 128 deterministic sightlines. It applies
`0.5 <= r_p <= 30 pkpc` and includes peculiar velocity plus Hubble flow in
`Delta v_los`. For the fiducial dual population, the mean retained fractions
are 0.888 and 0.990 at redshift 2.848 for velocity limits of 300 and 600 km/s.
The corresponding values at redshift 1.499 are 0.869 and 0.986. These fractions
measure the viewing-angle retention of three-dimensional pairs. They do not
include unrelated foreground or background objects in an observational
cylinder.

The controlled binary-capture comparison retains systems with exactly two
SMBHs in the mass-limited population. It standardizes `log10(M_primary)`,
`log10(mass ratio)`, `log10(separation)`, and
`log10(relative speed + 10 km/s)`, then finds a unique minimum-distance
assignment. At redshift 2.848, all 326 dual AGN candidates are matched to
326 of 470 single-AGN pairs. The maximum absolute standardized
mean difference decreases from 0.328 to 0.088. The interval-censored 1 Gyr
binary-capture fractions are 0.893--0.902 and 0.859--0.862. Their difference is
bounded by 0.031--0.043.

At redshift 1.499, all 243 dual AGN candidates are matched to 243 of 1,135
single-AGN pairs. The maximum absolute standardized mean
difference decreases from 0.486 to 0.031. The corresponding binary-capture
fractions are 0.749--0.753 and 0.683--0.695. Their difference is bounded by
0.053--0.070. The output-296 population is right-censored at the selection time
because no later HR5 output is present. These are possible binary captures,
not direct records of the partner or physical coalescences.

Neutral-hydrogen columns are available along six cardinal sightlines at
outputs 89 and 296. The analysis reports the fraction of active sightlines
above `N_H = 1e23` and `1e24 cm^-2`. It does not convert these columns into a
survey detection probability because that step requires a spectral and
instrument response model.

The local MkAGN products contain zero-valued galaxy identifiers and host mass
fields. The original FoF/PSB products have now been located under
`/scratch/kjhan/Hydro/HR5/FoFPSB/02_without_dc/Ver2`. A direct host assignment
is possible whenever `GALFIND.DATA`, its matching `GALCATALOG.LIST`, and the
optional `background_ptl` file are present. Canonical FoF/PSB files exist for
all 17 outputs with MkAGN records, including outputs 89, 117, and 296 used for
the detailed active-pair calculation. Only directories named exactly
`FoF.NNNNN` are used.

The output files are `hr5_dual_agn_summary.json`,
`hr5_dual_agn_capture_cdf.csv`, `hr5_dual_agn_pairs.csv`, and
`hr5_dual_offset_pairs_mbh_ge_1e6.csv`. Matched systems are written to
`hr5_dual_offset_matched_pairs.csv`. Columns whose historical names begin with
`receiver_` identify the surviving SMBH assigned by the distance and mass
criteria. They do not identify the partner selected by the simulation or
establish physical SMBH coalescence.

Output 296 at redshift 0.625 contains 8,179 SMBHs above the fiducial bolometric
limit. Their median Eddington ratio is 0.0410, and the central 68 percent
interval is 0.0158--0.271. The fraction above an Eddington ratio of 0.1 is
0.288. The luminosity comparison is written to
`hr5_eddington_luminosity_z0p625.pdf`.

## Literature comparison

Published dual AGN fractions depend strongly on luminosity, projected or
three-dimensional separation, obscuration, and the denominator used for the
fraction. Direct numerical comparison therefore requires matched selections.

- [Liu et al. 2011](https://arxiv.org/abs/1104.0950) measured kpc-scale AGN
  pairs in SDSS and reported a few-percent pair fraction over 5--100 kpc.
- [Van Wassenhove et al. 2012](https://arxiv.org/abs/1111.0223) showed that
  simultaneous activity becomes more common at small separations, while the
  observable phase remains short.
- [Steinborn et al. 2016](https://arxiv.org/abs/1510.08465) measured a roughly
  one-percent dual AGN fraction at redshift two in Magneticum for a bolometric
  threshold of 1e43 erg s^-1.
- [Capelo et al. 2017](https://arxiv.org/abs/1611.09244) quantified how viewing
  angle, luminosity threshold, and separation change the observable duration
  in controlled galaxy-merger calculations.
- [Rosas-Guevara et al. 2019](https://arxiv.org/abs/1805.01479) connected dual
  AGN below 30 kpc to the EAGLE galaxy-merger population and found percent-level
  fractions for a hard X-ray selection.
- [Volonteri et al. 2022](https://arxiv.org/abs/2112.07193) followed Horizon-AGN
  duals to later black-hole mergers and found a strong but selection-dependent
  connection between 4--30 kpc pairs and later merger events.
- [Chen et al. 2023](https://arxiv.org/abs/2208.04970) found a three-percent
  dual fraction at redshifts two to three in ASTRID and showed that many bright,
  close pairs merge within 500 Myr.
- [Puerto-Sanchez et al. 2025](https://arxiv.org/abs/2411.15297) compared nine
  cosmological simulations with a common selection and found that predicted
  dual fractions and number densities retain substantial inter-simulation
  variation.
- [Saeedzadeh et al. 2024](https://arxiv.org/abs/2403.17076) followed
  Romulus25 dual AGNs and found rapidly evolving, slowly evolving, and
  nonmerging histories.
- [Buttigieg et al. 2025](https://arxiv.org/abs/2504.17549) showed that
  premature black hole mergers in FABLE can precede completion of the host
  merger by several Gyr and can alter the predicted massive-binary population.
- [Chen et al. 2025](https://arxiv.org/abs/2512.16844) applied observational
  selection functions to ASTRID and found that roughly 30--70 percent of the
  selected dual AGNs coalesce within about 1 Gyr.

The HR5 values lie within the broad range of these studies. The common
luminosity and separation cuts allow a selection-level comparison. Direct
membership in a PSB galaxy now permits the distinct-galaxy cut used in the
recent multi-simulation comparison. Obscuration, survey response, and the
numerical capture prescription remain separate physical uncertainties.

## Direct association of sink particles with PSB galaxies

`GALFIND.DATA` stores one `HaloInfo` record for each FoF halo. Each halo record
is followed by its `SubInfo` records and by the dark matter, gas, sink, and star
particles assigned to each PSB galaxy. The sink identifier therefore provides
a direct host assignment rather than a nearest-galaxy association.

The original Intel compiler saved `DmType`, `GasType`, and `StarType` records
with 128 bytes per particle. `SinkType` occupies 168 bytes and stores its
32-bit identifier at byte 160. A current GCC build gives a different size for
`GasType`, so the extractor uses the measured legacy byte sizes explicitly.

Build the extractor with:

```bash
cc -std=c11 -O3 -Wall -Wextra -Werror -fopenmp \
  -o /tmp/extract_hr5_sink_hosts tools/extract_hr5_sink_hosts.c -lm
```

For output 117, run:

```bash
/tmp/extract_hr5_sink_hosts \
  --data /scratch/kjhan/Hydro/HR5/FoFPSB/02_without_dc/Ver2/FoF_Data/FoF.00117/GALFIND.DATA.00117 \
  --list /scratch/kjhan/Hydro/HR5/FoFPSB/02_without_dc/Ver2/FoF_Data/FoF.00117/GALCATALOG.LIST.00117 \
  --output /scratch/kjhan/Hydro/HR5/FoFPSB/02_without_dc/Ver2/Derived_Sink_Hosts/sink_hosts.00117.csv \
  --output-number 117 \
  --redshift 1.4988132 \
  --threads 8
```

The table records `sink_id`, the sequential PSB-galaxy identifier
`galaxy_gid`, the FoF and PSB indices, sink and host masses, positions, and
velocities, together with the dark-matter, gas, sink, stellar, and total
particle counts of the host. The extractor verifies the particle count in each PSB record, the
sum of the sink masses assigned to each host, duplicate sink identifiers,
sampled metadata shared by the two galaxy-finder files, and the final byte
offset in every input file.

The optional `--background background_ptl.NNNNN` argument also extracts sinks
that do not belong to a PSB galaxy and marks them with `galaxy_gid = -1` and
`background = 1`. A complete background scan is substantially more expensive
because the file contains one variable-length record for every FoF halo. The
direct-host calculation therefore omits this option. A sink absent from the
result has no direct PSB assignment at that output; it is not assigned to the
nearest galaxy.

## Descendants of the directly assigned host galaxies

The 129 files named `GalaxyLinkedList.NNNNN` cover outputs 18--123 and 23
additional outputs through output 296. Each native record stores a
most-bound-particle link, the PSB-galaxy identifier at the current output, and
the array index of its record at the following available galaxy output. A PSB
galaxy may have several tracer records. Their valid links normally enter one
descendant. If they enter more than one galaxy, the analysis accepts a unique
record marked as the dominant progenitor by the native tree calculation. A
unique major-branch link provides a secondary resolution, while any remaining
case stays ambiguous.

Run:

```bash
PYTHONPATH=src python scripts/analyze_hr5_host_descendants.py
```

The calculation follows both host galaxies of every spatially selected active
SMBH pair. It records the first output in which the two tracks enter one common
descendant. The preceding and current galaxy outputs define the time interval.
Systems with two distinct tracks at output 296 are right-censored at
`z=0.625`. They are not counted as host galaxies that never merge. The table
also joins the interval of a later possible binary capture for the same assigned
SMBH pair when that event exists in the active-pair catalogue. The sink
histories do not identify the companion selected by the simulation.

# HR5 comparison with FABLE

## Scientific objective

The central measurement will connect three events that are usually treated
separately:

1. two active SMBHs occupy distinct PSB galaxies,
2. the two PSB galaxies acquire a common descendant, and
3. one of the two sink particles is removed in a numerical binary capture.

This sequence allows us to measure whether a numerical binary capture precedes
or follows the merger of its host galaxies. The same data also determine how
AGN activity changes while the galaxies approach each other.

Buttigieg et al. (2025) used FABLE to show that numerical BH mergers often
precede the merger of the host subhaloes. Their selected sample contains 10,716
events after requiring both BH masses to exceed (10^6\,M_\odot) and the stellar
mass of each host to exceed 100 initial baryonic resolution masses. Only about
5 percent of these events
require no additional galactic-scale delay. Their median added delay is
1.3 Gyr, while 29 percent of the host pairs do not merge by (z=0). These
published values define the direct comparison sample, not priors imposed on
HR5.

References:

- [Buttigieg et al. 2025, MNRAS, 542, 2019](https://doi.org/10.1093/mnras/staf1336)
- [Buttigieg et al. 2026, arXiv:2607.05208](https://arxiv.org/abs/2607.05208)
- [Lee et al. 2021, ApJ, 908, 11](https://doi.org/10.3847/1538-4357/abd08b)
- [Volonteri et al. 2022, MNRAS, 514, 640](https://doi.org/10.1093/mnras/stac1217)

The numerical values transcribed from the two FABLE studies are stored in
`configs/fable_reference.json`. Keeping this reference separate from the HR5
measurements prevents a published FABLE count from being mistaken for an HR5
result.

## Complementary numerical leverage

| Property | HR5 material used here | FABLE analysis |
|---|---:|---:|
| sampled region | elongated high-resolution region | periodic cube |
| effective comoving volume | (1.087\times10^7\) cMpc\(^3\) | (3.19\times10^6\) cMpc\(^3\) for (100h^{-1}\) cMpc and (h=0.679\) |
| stored galaxy outputs | 129 | 136 |
| final redshift | 0.625 | 0 |
| numerical SMBH events before common cuts | 576,278 possible binary captures | 91,879 numerical BH mergers |
| direct numerical partner identifier | unavailable | available |
| host assignment | direct PSB membership | subhalo membership |

The HR5 effective volume is about 3.4 times the FABLE volume. This difference
improves the sampling of rare massive systems, but it is not a sufficient claim
of superiority. Buttigieg et al. (2026) find that FABLE-100 already converges
for the gravitational-wave background near the most constraining PTA
frequency. The additional HR5 volume must therefore be used to resolve trends
with mass, redshift, environment, and AGN activity rather than to repeat a
volume-convergence argument. FABLE has the decisive advantages of direct
numerical merger partners and evolution to (z=0). The HR5 analysis must use
interval censoring at its final output and must repeat every measurement under
stricter tests of the assigned companion.

## Common selection

The primary comparison will use identical measurable cuts wherever the two
simulations provide the same quantity:

| Quantity | Common selection |
|---|---:|
| SMBH mass | both (M_\mathrm{BH}\geq10^6\,M_\odot) |
| host resolution | both host stellar masses exceed (100\times6.4\times10^6h^{-1}\,M_\odot) |
| bolometric luminosity | both (L_\mathrm{bol}\geq10^{43}\) erg s\(^{-1}\) |
| physical separation | 0.5--30 pkpc |
| host relation | distinct PSB galaxies |

The SMBH-mass threshold matches the FABLE cut. The primary HR5 comparison also
uses the numerical value of FABLE's stellar-mass threshold,
(100\times6.4\times10^6h^{-1}\,M_\odot). The aperture remains different:
FABLE measures stellar mass within twice the stellar half-mass radius, whereas
HR5 provides the total stellar mass assigned to a PSB galaxy. A secondary HR5
sample requires at least 100 directly counted stellar particles and measures
the sensitivity to this aperture-independent resolution cut. The luminosity
and separation cuts define the HR5 dual-AGN sample and
must be imposed on any FABLE dual-AGN catalogue used in a numerical comparison.
Results will also be reported without the luminosity cut so that the dynamics
are not conditioned on simultaneous activity.

Raw counts will not be compared. HR5 samples an elongated high-resolution
region and ends at (z=0.625), whereas the FABLE calculation uses a periodic
box and continues to (z=0). Event densities, conditional fractions, and
censored time distributions are the appropriate common observables.

## Event definitions in HR5

`Binary capture` denotes the removal of a sink particle by the numerical
prescription. It does not denote physical SMBH coalescence. The available sink
histories do not store the identity of the partner selected by the simulation.
An assigned companion therefore remains conditional on the legacy distance and
mass criteria.

For a candidate pair with two direct PSB assignments, define

\[
\Delta t_\mathrm{gal}=t_\mathrm{common\ descendant}-t_\mathrm{capture}.
\]

A positive value indicates that the numerical binary capture precedes the
appearance of a common descendant for the two galaxies. A negative value means
that the host galaxies merge first. Both event times are interval-censored, so
the catalogue must retain lower and upper bounds rather than a single adopted
time. A system whose galaxies remain distinct at the final HR5 output is
right-censored at (z=0.625); it is not classified as a non-merger.

The primary sample will require the assigned sink pair to occupy two directly
identified PSB galaxies at the last output in which both sinks are present.
Secondary samples will add the relative-velocity test, a common FoF halo, and a
minimum stellar-particle count. The dependence on these restrictions will show
how much of the result follows from uncertain companion assignment.

## Measurements

### 1. Redshift evolution of host-confirmed dual AGN

For every available MkAGN output, measure the abundance of close active pairs
in the same PSB galaxy, in two PSB galaxies within one FoF halo, and in distinct
FoF haloes. Separate pure pairs from systems containing three or more active
SMBHs. Report the number density and the fraction relative to both active SMBHs
and all close SMBH pairs.

### 2. Time ordering of galaxy mergers and numerical binary captures

Follow both PSB identifiers through `GalaxyLinkedList`. Record the first output
at which they share a descendant and compare this interval with the numerical
binary-capture interval. Measure the signed distribution of

\[
\Delta t_\mathrm{gal}
\]

with interval-censored survival methods. Report the fractions for capture
first, host merger first, the same output interval, and right-censored host
pairs. Repeat the measurement with the FABLE mass threshold and the HR5
host-resolution analogue.

### 3. AGN activity along the merger sequence

Measure bolometric luminosity, Eddington ratio, gas fraction of the host,
stellar-mass ratio, and SMBH-mass ratio as functions of time relative to the
host merger and numerical binary capture. Compare simultaneous activity with
single activity after matching in redshift, separation, both SMBH masses, both
host stellar masses, and host gas fraction. This tests whether enhanced dual
activity marks a particular part of the merger sequence without interpreting
the association as causal.

### 4. Dependence on environment and mass

Measure the delay distribution in bins of primary SMBH mass, chirp mass, SMBH
mass ratio, host stellar-mass ratio, FoF mass, and redshift. HR5's effective
high-resolution volume provides improved statistics for rare massive systems
and dense environments. The elongated geometry requires spatial resampling by
subvolume rather than a cubic-box variance estimate.

### 5. Consequences for gravitational-wave populations

Shift only events whose numerical binary capture precedes the host merger.
Keep host pairs that do not reach a common descendant by (z=0.625) as
censored. Recompute the event density in chirp-mass--redshift space before and
after the measured galactic-scale delay. This result can be compared directly
with the FABLE correction. A later calculation may add the independently
modeled interval from binary capture to sub-pc separation and the subsequent
FDM and gravitational-wave intervals. A fixed 1-Gyr hardening delay will not be
adopted as the physical HR5 prediction.

## Comparative contribution

The useful distinction from FABLE is not merely a larger catalogue. HR5 can
combine a large effective volume with direct PSB membership, 129 stored galaxy
outputs, detailed sink positions and velocities, and AGN luminosities. This
permits a joint measurement of host identity, activity, galaxy-merger timing,
and numerical binary capture over the same systems. FABLE records the merger
partner directly but repositions black holes and does not follow their
velocities dynamically. HR5 retains phase-space histories, although its legacy
companion identifier was assigned after the simulation. The primary result
must therefore be a hierarchy: all sink removals, removals with two identified
hosts, unique companion assignments, and assignments that pass a stated
phase-space test. Dependence on that hierarchy measures the uncertainty instead
of hiding it.

FABLE has already propagated macrophysical delays into a PTA calculation and,
in its 2026 analysis, varied a uniform unresolved hardening time from 0.1 to
5 Gyr. Repeating that calculation with another fixed delay would add little.
The comparative advantage of HR5 is to condition the delay on the measured
SMBH masses, host masses, environment, redshift, and AGN state. The same
catalogue can then show whether the systems that dominate the PTA strain occupy
the same part of the merger sequence as observable dual AGN.

The exact HR5 sink-motion and binary-capture parameters must be recovered from
the production configuration before claiming a difference from FABLE's
repositioning prescription. The source contains explicit sink positions and
velocities and a velocity-dependent capture option, but source capability alone
does not establish which options were active in HR5.

## Assigned-companion-independent test

The primary dual-AGN test does not use the companion assigned to a removed sink
particle. It begins with the mass-limited close-pair catalogue and uses the PSB
galaxy identified directly for each SMBH. The initial sample requires two
distinct PSB galaxies, so a close pair already assigned to one galaxy is not
called a dual AGN. Dual AGN are matched one-to-one to systems with only one
active SMBH in eight measured quantities: both SMBH masses, physical
separation, relative speed, both host stellar masses, and both host
gas-to-stellar mass ratios. Regularized propensity scores define the matches. A
caliper of 0.2 standard deviations in score rejects poor matches, and an output
is retained only when every post-match standardized mean difference is below
0.2. The two PSB galaxies are then followed through the native galaxy links.

The complete mass-limited sample contains 383 matched systems at redshifts
3.394, 2.848, and 1.499. Output 88 is removed because it lies only about 0.01
Gyr before output 89 and would repeat many systems. Outputs at redshifts 4.074
and 0.625 do not satisfy the balance criterion. In the sample that also imposes
the FABLE SMBH and host-stellar-mass thresholds before matching, 281 systems
remain: 70, 81, and 130 at the three retained redshifts.

For the FABLE-selection analogue, interval censoring bounds the fractions whose
two PSB galaxies join within 0.5 Gyr to 0.929--0.986 for dual AGN and
0.943--0.986 for single-AGN controls at redshift 3.394. The corresponding bounds
are 0.852--0.951 and 0.790--0.926 at redshift 2.848, and 0.669--0.800 and
0.677--0.823 at redshift 1.499. At every redshift, the interval for the paired
difference includes zero. Bootstrap sampling does not change that conclusion.
The available HR5 data therefore do not show that simultaneous activity changes
the probability that already close, distinct PSB galaxies acquire a common
descendant within 0.5 Gyr.

For a right-censored host pair, the resolved follow-up interval is used
directly: a pair known to remain distinct for at least 0.5 Gyr is a definite
non-joining system at that threshold. This result is independent of the legacy
companion assignment. The event-level comparison with FABLE remains a separate
catalogue audit because the HR5 sink histories do not record the numerical
capture partner directly.

## Data products

The canonical data set is stored under

```text
/scratch/kjhan/Hydro/HR5/FoFPSB/02_without_dc/Ver2/
Derived_Sink_Hosts/canonical_v1
```

`hr5_output_manifest.csv` and `hr5_output_manifest.json` record the RAMSES
output, scale factor, redshift, MkAGN path, canonical FoF/PSB paths, galaxy
catalogue, galaxy-link file, number of link records, following galaxy output,
input sizes, and completion state. Directories
named `output_NNNNN` contain the sink--host table, extraction validation,
dual-AGN host table, and summary for one output. The combined redshift table is
`hr5_dual_agn_host_evolution.csv`.

The completed active-pair host catalogue contains 15,946 close active SMBH
pairs. Direct PSB assignments are available for both SMBHs in 15,940 systems.
Of these, 1,242 already occupy one PSB galaxy and 14,698 occupy two distinct PSB
galaxies. A total of 13,370 distinct-host systems acquire a common descendant in
a later saved output. The selection using the FABLE SMBH and host-stellar-mass
thresholds contains 1,254 systems in distinct PSB galaxies, of which 1,145
acquire a common descendant. These counts describe a sample selected by AGN
activity and separation. They are not the FABLE numerical-merger sample and
must not be compared as raw event counts.

The host-confirmed dual-AGN demographics are measured with

```bash
PYTHONPATH=src python scripts/analyze_hr5_dual_agn_host_demographics.py
```

The calculation separates spatially selected active SMBH pairs from pairs in
two distinct PSB galaxies. It estimates the uncertainty with an eight-region
spatial jackknife and fits both densities with the modified Schechter form used
for the active-SMBH abundance. The distinct-host fit gives
`phi = 6.689e-4 cMpc^-3`, `z_star = 3.787`, `alpha = 3.263`, and
`beta = 2.796`, with an RMS residual of 0.281 dex. At redshifts 3.394, 2.848,
1.499, and 0.625, the fractions of classifiable pairs in distinct PSB galaxies
are 0.929, 0.951, 0.777, and 0.487, respectively.

Run

```bash
PYTHONPATH=src python scripts/analyze_hr5_host_descendants.py
```

after the 17 active-SMBH outputs are complete. The calculation reads all 129
native `GalaxyLinkedList` outputs from output 18 through output 296 without
copying their full contents. The resulting
`host_descendants/hr5_active_pair_host_descendants.csv` retains the original
pair and host quantities, the first common descendant, lower and upper bounds
on its delay, the later possible binary-capture interval when one is available
for the assigned SMBH pair, and the ordering of those two intervals. The
accompanying JSON file gives counts for the full sample and the FABLE-selection
analogue.

The event-level preparation is

```bash
PYTHONPATH=src python scripts/build_hr5_capture_host_dataset.py --partition-events
PYTHONPATH=src python scripts/analyze_hr5_capture_hosts.py
python scripts/plot_hr5_fable_comparison.py
```

The assigned-companion-independent dual-AGN test is

```bash
PYTHONPATH=src python scripts/build_hr5_agn_pair_host_dataset.py
PYTHONPATH=src python scripts/analyze_hr5_matched_pair_hosts.py
python scripts/plot_hr5_matched_pair_hosts.py
PYTHONPATH=src python scripts/analyze_hr5_matched_pair_hosts.py \
  --require-fable-selection \
  --output-directory /scratch/kjhan/Hydro/HR5/FoFPSB/02_without_dc/Ver2/Derived_Sink_Hosts/canonical_v1/matched_pair_host_descendants_fable
python scripts/plot_hr5_matched_pair_hosts.py \
  --table /scratch/kjhan/Hydro/HR5/FoFPSB/02_without_dc/Ver2/Derived_Sink_Hosts/canonical_v1/matched_pair_host_descendants_fable/hr5_matched_agn_pair_host_descendants.csv \
  --output results/hr5/hr5_matched_pair_host_evolution_fable.pdf
```

It writes the matched catalogue and time-bounded summary below
`matched_pair_host_descendants/` in the canonical scratch data set. The figure
and its plotted values are written to
`results/hr5/hr5_matched_pair_host_evolution.pdf` and
`results/hr5/hr5_matched_pair_host_evolution.csv`.

The first command constructs the mass-limited dual- and single-AGN population
at every complete MkAGN output. It stores both SMBH positions, velocities,
luminosities, and Eddington ratios together with the directly assigned stellar,
gas, and total masses of both PSB galaxies. Its output is
`agn_pair_hosts/hr5_agn_pair_hosts_mbh_ge_1e6.csv` in the canonical scratch data
set. The table contains 5,059 close pairs across all 17 outputs, including the
outputs with no qualifying pair, and does not contain a legacy assigned
companion. Qualifying pairs first appear at redshift 4.946 under the
(10^6\,M_\odot) SMBH-mass threshold.

It covers all 576,278 possible binary captures. Their last resolved sink
outputs map onto 121 galaxy outputs and require 1,128,422 unique
output--SMBH combinations. For captures between the sparsely stored late-time
galaxy outputs, the calculation uses the latest preceding output and records
an output lag of as many as 13. This offset remains part of the event-time
interval. It is not treated as an exact simultaneous host assignment. The
filtered extraction is started explicitly with
`--extract --jobs 4 --threads 8`. Four outputs are read independently, while
each extractor uses eight threads for the sink blocks in one output.
`analyze_hr5_capture_hosts.py` then assigns both SMBHs to their PSB galaxies,
follows those galaxies to a common descendant, and classifies the ordering of
the common-descendant and possible-binary-capture intervals. Its FABLE-selection
analogue requires both SMBH masses to exceed (10^6\,M_\odot) and both directly
assigned hosts to exceed FABLE's published stellar-mass threshold. HR5 uses
total PSB stellar mass rather than the FABLE aperture within twice the stellar
half-mass radius. A second flag retains the requirement of at least 100 HR5
stellar particles in each host. Same-host systems remain in this event-level
sample because they provide the analogue of FABLE events that require no
additional host-merger delay.

The descendant calculation groups complete host-assignment outputs into each
pass through the 129 galaxy-link outputs. Events from one host-assignment output
are never split between passes. On the production node, the configured maximum
of 600,000 events admits the full catalogue in one pass and avoids repeated
access to about 129 GB of galaxy-link records. This grouping changes only the
memory use and repeated reading; each pair retains its own selection output and
interval bounds.

Where an MkAGN snapshot exists at the host-assignment output, the event table
also records both bolometric luminosities and Eddington ratios. Each event is
classified as having two, one, or no active SMBHs at
(L_\mathrm{bol}\geq10^{43}\,\mathrm{erg\,s^{-1}}). The capture--host timing
fractions are then reported separately for these activity states. This is the
principal extension beyond the published FABLE delay measurement: it connects
the numerical event, host-galaxy evolution, and the electromagnetic state of
both SMBHs without assigning AGN activity to an event from a population-average
duty cycle.

The event table also joins the legacy companion diagnostic. It records whether
the same assigned companion is used for several removals in one output and
whether the last resolved relative speed is below the escape speed generated by
the two SMBH point masses. The latter is deliberately a diagnostic rather than
a complete binding criterion because it omits the host potential and unresolved
matter around each SMBH.

The sensitivity to the assigned companion is measured with

```bash
PYTHONPATH=src python scripts/analyze_hr5_companion_sensitivity.py
```

Within the FABLE mass analogue, the complete assigned catalogue contains
25,494 possible binary captures. Requiring an assignment that is unique in the
saved output leaves 22,451 systems. Requiring both uniqueness and
`v_rel <= v_esc` for the two SMBH point masses leaves 33 systems. The respective
fractions whose hosts require no additional joining time are bounded by
0.637--0.885, 0.647--0.887, and 0.788--1.000. The last selection is not a full
binding test because it omits the host potential and unresolved matter.

The comparison figure uses all events that pass the FABLE-selection analogue
as the denominator of the HR5 timing fractions. The lower limit counts only
systems whose hosts certainly joined before the possible binary capture. The
upper limit also includes overlapping timing intervals and unresolved host
times. This denominator matches the published FABLE fraction of 513 among
10,716 selected events. The upper panel reports events per stored HR5 output
interval and must not be interpreted as a rate without division by the interval
width and effective volume. The shaded HR5 region joins the lower and upper
timing limits. No midpoint is treated as a measured fraction.

`capture_host_descendants/hr5_fable_capture_host_evolution.csv` is the compact
redshift table for this comparison. Its 121 rows record the event count, FABLE
selection count, AGN states, time-order counts, timing bounds, and
assigned-companion diagnostics at each host-assignment output. The full
event-level table and nested JSON summary remain the authoritative sources for
individual systems and aggregate cross-checks.

Run

```bash
PYTHONPATH=src python scripts/validate_hr5_host_derived_outputs.py
```

after all derived calculations. The validator checks the host-demographic
totals, modified Schechter fit, nested companion selections, matched host
sample, and publication figures.

Only directories named exactly `FoF.NNNNN` are admitted. Results in directories
whose names contain `.mine`, `.test`, or `.try` are excluded.

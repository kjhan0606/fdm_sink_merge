# HR5 dual-AGN JWST morphology mock at z=0.625

The six-panel figure uses the HR5 PSB-galaxy particles at output 296 rather
than an analytic galaxy profile. The selection first requires a dual active
SMBH system in two distinct PSB galaxies within one FoF halo, exactly two SMBHs
in the close-pair system, at least 100 star particles in each host, and the
FABLE comparison cuts. Six systems are then sampled at equally spaced ranks in
three-dimensional separation. This gives separations from 8.49 to 29.95 pkpc
without selecting systems by visual appearance.

The line of sight is the simulation z axis. Star-particle masses are projected
onto a 256 by 256 grid, smoothed with a Gaussian kernel of 0.3 pkpc to suppress
particle sampling, and convolved with the official STScI NIRCam F200W PSF. The
PSF has a detector pixel scale of 0.0311 arcsec and an STScI simulated FWHM of
0.064 arcsec. At z=0.62536, one pixel spans 0.2168 pkpc for the adopted HR5
cosmology. Each panel therefore covers 55.51 pkpc on a side. The same surface
density scaling is used in all panels.

The host image assumes a spatially constant mass-to-light ratio. It is a
dust-free morphology proxy, not calibrated F200W photometry. Cyan and magenta
rings mark the primary and secondary active SMBHs. A point source at each SMBH
position is convolved with the same F200W PSF. Its display amplitude uses a
logarithmically compressed form of the HR5 bolometric luminosity and must not
be interpreted as an F200W flux.

## Attenuation

An attenuation map is not required to display the stellar morphology and the
two active nuclei. It is required before interpreting colors, flux ratios,
detection probabilities, or obscured dual-AGN fractions. The extractor saves
the gas-cell positions, masses, and metallicities. The FITS product also stores
the projected gas metal-mass surface density for every panel. This is only an
intermediate quantity. A physical attenuation calculation still requires a
dust-to-metal relation, a wavelength-dependent opacity, the foreground gas
geometry for each stellar and nuclear source, and an AGN spectral model.

## Products

- `results/hr5/hr5_dual_agn_jwst_f200w_z0p625.pdf`
- `results/hr5/hr5_dual_agn_jwst_f200w_z0p625.png`
- `results/hr5/hr5_dual_agn_jwst_f200w_z0p625.json`
- `mock_observations/output_00296/hr5_dual_agn_jwst_f200w_z0p625.fits`

The JSON file records the target identifiers, three-dimensional and projected
separations, particle counts, field coverage, cosmology, display transforms,
PSF checksum, and all non-photometric assumptions. The FITS file contains the
convolved stellar-mass map, projected gas metal-mass map, and AGN point-source
layer for every panel.

The F200W PSF was downloaded from the
[STScI simulated PSF library](https://stsci.app.box.com/v/jwst-simulated-psf-library).
Its SHA-256 checksum is
`3b98266507af8e10157c515ae47ed3b9ec357a5d39d59e004cd780689873b2fb`.
The detector scale and FWHM follow the
[NIRCam PSF documentation](https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-performance/nircam-point-spread-functions).

## Instrumental regression test

`scripts/validate_hr5_jwst_psf.py` places a unit-flux delta source at the
centre of the 256 by 256 science grid. It applies the same normalized PSF and
the same `same`-field convolution used for the HR5 image. The convolved source
matches direct placement of the cropped kernel to a maximum absolute residual
of (5.6\times10^{-17}). The field retains 0.997904 of the input flux. The
remaining 0.002096 lies outside the field. The measured detector-sampled FWHM
is 0.0743 arcsec along one image axis and 0.0751 arcsec along the other.

The regression products are

- `results/hr5/hr5_jwst_f200w_psf_validation.json`
- `results/hr5/hr5_jwst_f200w_psf_validation.png`
- `mock_observations/hr5_jwst_f200w_psf_validation.fits`

The original morphology generator now also retains the intrinsic stellar map,
the unconvolved AGN impulses, the PSF, and the convolved host and AGN layers.
The displayed morphology and the three original numerical layers remain
unchanged.

## Calibrated F200W quick look

`scripts/make_hr5_dual_agn_jwst_photometric_mock.py` provides a separate
photometric calculation. It does not replace the dust-free morphology
baseline. Stellar birth times are converted to ages with the HR5 conformal-time
table. Initial stellar masses follow the RAMSES mass units stored with output
296. FSPS then assigns an intrinsic spectrum to each particle from its age and
metallicity. The calculation uses python-fsps 0.4.7 with the MIST isochrones and
MILES spectral library. The matching FSPS data revision is
`82a873508d500ca353bbb922459bf928498f7a72`.

The band integration uses the detector-averaged, in-flight NIRCam F200W total
throughput released by STScI as version 7.0. Its SHA-256 checksum is
`9a607820a5cfec29965d9f6e1e956afbd3a32bfa439453fa18c0a8fac0eee7a5`.
The measured pivot wavelength is 1.98760 micron. It corresponds to 1.22287
micron in the rest frame at the target redshift. Flux densities are
photon-weighted means and are stored in nJy per detector pixel.

The AGN preview uses an unobscured power law with
(L_\nu\propto\nu^{-0.5}). It is normalized with
(L_{\rm bol}/[\lambda L_\lambda(5100\,{\rm Angstrom})]=10.3), following the
mean type-1 quasar SED scale of Richards et al. (2006). This single template
does not include an obscuring torus, orientation dependence, or variability.

### Foreground-screen preview

The fast dust calculation adopts a dust-to-metal mass ratio of 0.4. Dust is
removed from cells whose stored thermal measure
`temperature_code/density_code` exceeds (10^6\) K per mean molecular weight.
Each AMR leaf cell is projected as an axis-aligned square rather than deposited
as a point. Absorption and scattering opacities are taken from the Draine
Milky-Way (R_V=3.1) carbonaceous-silicate mixture. The opacity-table checksum
is `b56680cc38b85f051f20c4405303e8c480cc9bec714fd5ba722a257a40ae840c`.
At the rest-frame F200W pivot, the adopted absorption and scattering opacities
are 4436 and 6037 square centimetres per gram of dust.

The screen removes direct light with
(\exp[-(\tau_{\rm abs}+\tau_{\rm sca})]). A single-interaction scattering
budget is redistributed with a normalized two-pixel image-plane Gaussian.
This width is an effective image-plane parameter. It is not a
Henyey--Greenstein asymmetry parameter. The screen intentionally supplies a
fast data-flow and flux-budget test. It does not preserve the relative depth
of stars, AGN, and dust and cannot determine the physical obscured fraction or
its viewing-angle dependence.

The calibrated products are

- `results/hr5/hr5_dual_agn_jwst_f200w_photometric_screen_z0p625.pdf`
- `results/hr5/hr5_dual_agn_jwst_f200w_photometric_screen_z0p625.png`
- `results/hr5/hr5_dual_agn_jwst_f200w_photometric_screen_z0p625.json`
- `mock_observations/output_00296/hr5_dual_agn_jwst_f200w_photometric_screen_z0p625.fits`
- `results/hr5/hr5_dual_agn_jwst_f200w_photometric_screen_validation.json`

The FITS file retains intrinsic stellar and AGN light, dust surface density,
absorption and scattering optical depths, direct and scattered light,
absorbed light, total emergent light, the PSF-convolved stellar and AGN
images, and their sum. The display transform is absent from every physical
array. All six panels pass the zero-dust, zero-scattering, flux-budget,
component-sum, FITS-checksum, and delta-source gates.

## Three-dimensional transfer with SKIRT

`scripts/export_hr5_dual_agn_skirt_inputs.py` writes a separate source and
medium catalogue for every panel. Stellar files follow the SKIRT
`ParticleSource` column convention with position, smoothing length, initial
mass, metallicity, and age. Dust files retain the native cuboidal bounds and
integrated dust mass of every surviving HR5 AMR leaf cell. AGN files retain
the two positions, bolometric luminosities, and sink identifiers. The origin
is the midpoint of the two active SMBHs. Coordinates are proper parsecs and
the declared observer lies on the positive simulation z axis.

The export manifest is
`results/hr5/hr5_dual_agn_skirt_input_manifest_z0p625.json`. The catalogues are
under `mock_observations/output_00296/skirt_inputs` in the HR5 analysis area.
SKIRT v9.0 was built from revision `1facef2` in
`/scratch/kjhan/software/SKIRT`, together with version 8 of the required Core
resource pack.

`scripts/run_hr5_dual_agn_skirt.py` imports the HR5 stellar particles and AMR
dust cells without collapsing their line-of-sight coordinates. Stellar spectra
come from the Kroupa-IMF FSPS family distributed with SKIRT. Each active SMBH
is represented by the SKIRT quasar spectrum normalized to its HR5 bolometric
luminosity. The final optical-quasar view adopts an observer-aligned biconical
beam with a half-opening angle of 30 degrees. Conservation of the total
bolometric luminosity then raises the on-axis specific intensity by a factor of
7.464 relative to isotropic emission. The medium uses the surviving dust mass
in each native cuboidal cell and the Milky-Way Weingartner--Draine grain
mixture. Forced scattering retains both direct and multiply scattered F200W
light. Thermal dust emission is not included because observer-frame F200W
samples 1.22 micron in the rest frame at this redshift.

The final calculation launches $10^7$ photon packets for each of the six
systems. SKIRT applies the observer-frame STScI F200W band response. The total
surface-brightness maps are then convolved with the detector-sampled in-flight
F200W PSF. `scripts/make_hr5_dual_agn_skirt_figure.py` applies one base-10
logarithmic display scale to all panels and leaves the physical FITS arrays
untransformed. Its lower and upper limits are the 1st and 99.99th percentiles
of all positive pixels in the six images. The AGN positions are indicated only
by arrows. For each pair, the two arrows lie perpendicular to the projected
separation vector and point in opposite directions toward the nuclei. Each
arrowhead stops 25 detector pixels, or 5.42 pkpc, before the corresponding AGN
coordinate so that it does not obscure the PSF core. This placement avoids
suggesting that an arrow represents an orbital velocity or force. Each panel
also gives the two HR5 bolometric luminosities in erg per second. Numbered
circles at the arrow tails associate each nucleus with the corresponding
luminosity entry. Every panel shows a 1 arcsec scale bar. In the HR5
cosmology at redshift 0.625, this angle corresponds to 6.972 pkpc.

The final products are

- `results/hr5/hr5_dual_agn_jwst_f200w_skirt_z0p625.pdf`
- `results/hr5/hr5_dual_agn_jwst_f200w_skirt_z0p625.png`
- `results/hr5/hr5_dual_agn_jwst_f200w_skirt_z0p625.json`
- `mock_observations/output_00296/hr5_dual_agn_jwst_f200w_skirt_z0p625.fits`

The FITS product retains the transparent, direct, scattered, total, and
PSF-convolved maps for every system. The relative residual in the identity
between the total image and the sum of its direct and scattered components is
between $2.11\times10^{-8}$ and $3.63\times10^{-8}$ across the six panels. The
dot product between every arrow vector and its projected pair axis is zero to
floating-point precision, while the cosine between the two arrow vectors is
(-1).

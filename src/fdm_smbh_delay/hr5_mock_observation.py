"""Physical and instrumental helpers for HR5 mock observations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.constants import L_sun
from astropy.io import fits
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve


@dataclass(frozen=True)
class ImageMoments:
    """Flux, centroid, and second moments of a non-negative image."""

    flux: float
    centroid_x_pixel: float
    centroid_y_pixel: float
    sigma_x_pixel: float
    sigma_y_pixel: float


@dataclass(frozen=True)
class ThroughputCurve:
    """Dimensionless photon throughput on an observed wavelength grid."""

    wavelength_micron: np.ndarray
    throughput: np.ndarray

    @property
    def pivot_wavelength_micron(self) -> float:
        numerator = np.trapezoid(
            self.throughput * self.wavelength_micron, self.wavelength_micron
        )
        denominator = np.trapezoid(
            self.throughput / self.wavelength_micron, self.wavelength_micron
        )
        return float(np.sqrt(numerator / denominator))


@dataclass(frozen=True)
class ForegroundScreenResult:
    """Direct, absorbed, and single-scattering image-plane light budgets."""

    direct: np.ndarray
    absorbed: np.ndarray
    scattered_before_redistribution: np.ndarray
    scattered_in_field: np.ndarray
    emergent: np.ndarray


def load_psf(
    path: Path, extension: str = "DET_SAMP"
) -> tuple[np.ndarray, float, dict[str, object]]:
    """Load and normalize a two-dimensional PSF image from a FITS extension."""

    with fits.open(path) as hdus:
        hdu = hdus[extension]
        psf = np.asarray(hdu.data, dtype=np.float64)
        pixel_scale = float(hdu.header["PIXELSCL"])
        metadata: dict[str, object] = {
            "extension": extension,
            "shape": list(psf.shape),
            "pixel_scale_arcsec": pixel_scale,
            "oversample": int(hdu.header.get("OVERSAMP", 1)),
            "original_sum": float(psf.sum()),
        }
    if (
        psf.ndim != 2
        or not np.all(np.isfinite(psf))
        or np.any(psf < 0.0)
        or float(psf.sum()) <= 0.0
    ):
        raise ValueError("The PSF image is invalid")
    psf /= psf.sum()
    return psf, pixel_scale, metadata


def deposit_bilinear(
    image: np.ndarray,
    x_physical: float,
    y_physical: float,
    field_physical: float,
    weight: float,
) -> None:
    """Deposit a point source while retaining its subpixel centroid."""

    if image.ndim != 2 or image.shape[0] != image.shape[1]:
        raise ValueError("The destination image must be a square two-dimensional array")
    if not np.isfinite(weight):
        raise ValueError("The point-source weight must be finite")
    size = image.shape[0]
    x_pixel = (x_physical / field_physical + 0.5) * size - 0.5
    y_pixel = (y_physical / field_physical + 0.5) * size - 0.5
    x0 = int(np.floor(x_pixel))
    y0 = int(np.floor(y_pixel))
    for dy in (0, 1):
        for dx in (0, 1):
            xx = x0 + dx
            yy = y0 + dy
            if 0 <= xx < size and 0 <= yy < size:
                fraction = (1.0 - abs(x_pixel - xx)) * (1.0 - abs(y_pixel - yy))
                image[yy, xx] += weight * fraction


def convolve_with_psf(image: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """Convolve an image with a unit-integral PSF and retain the input shape."""

    if image.ndim != 2 or psf.ndim != 2:
        raise ValueError("The source image and PSF must both be two-dimensional")
    if not np.isclose(float(psf.sum()), 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("The PSF must have unit integral")
    return fftconvolve(image, psf, mode="same")


def image_moments(image: np.ndarray) -> ImageMoments:
    """Measure the flux, centroid, and axis-aligned second moments."""

    array = np.asarray(image, dtype=np.float64)
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError("The image must be a finite two-dimensional array")
    flux = float(array.sum())
    if flux <= 0.0:
        raise ValueError("The image must have positive total flux")
    yy, xx = np.indices(array.shape, dtype=np.float64)
    centroid_x = float(np.sum(xx * array) / flux)
    centroid_y = float(np.sum(yy * array) / flux)
    sigma_x = float(np.sqrt(np.sum((xx - centroid_x) ** 2 * array) / flux))
    sigma_y = float(np.sqrt(np.sum((yy - centroid_y) ** 2 * array) / flux))
    return ImageMoments(flux, centroid_x, centroid_y, sigma_x, sigma_y)


def _half_maximum_width(profile: np.ndarray) -> float:
    """Return a linearly interpolated full width at half maximum in pixels."""

    values = np.asarray(profile, dtype=np.float64)
    peak_index = int(np.argmax(values))
    half = 0.5 * float(values[peak_index])
    left_candidates = np.flatnonzero(values[:peak_index] < half)
    right_candidates = np.flatnonzero(values[peak_index + 1 :] < half)
    if not len(left_candidates) or not len(right_candidates):
        raise ValueError("The profile does not cross half maximum on both sides")
    left_low = int(left_candidates[-1])
    left_high = left_low + 1
    right_high = peak_index + 1 + int(right_candidates[0])
    right_low = right_high - 1

    def crossing(low: int, high: int) -> float:
        slope = float(values[high] - values[low])
        if slope == 0.0:
            return 0.5 * (low + high)
        return low + (half - float(values[low])) / slope

    return crossing(right_low, right_high) - crossing(left_low, left_high)


def psf_fwhm_pixels(psf_image: np.ndarray) -> tuple[float, float]:
    """Measure horizontal and vertical FWHM through the brightest pixel."""

    peak_y, peak_x = np.unravel_index(np.argmax(psf_image), psf_image.shape)
    return (
        _half_maximum_width(psf_image[peak_y, :]),
        _half_maximum_width(psf_image[:, peak_x]),
    )


def shifted_kernel_crop(
    psf: np.ndarray, image_shape: tuple[int, int], source_pixel: tuple[int, int]
) -> np.ndarray:
    """Place a PSF analytically using SciPy's ``same``-convolution convention."""

    height, width = image_shape
    source_y, source_x = source_pixel
    start_y = (psf.shape[0] - 1) // 2
    start_x = (psf.shape[1] - 1) // 2
    result = np.zeros(image_shape, dtype=np.float64)
    for output_y in range(height):
        kernel_y = output_y + start_y - source_y
        if not 0 <= kernel_y < psf.shape[0]:
            continue
        output_x0 = max(0, source_x - start_x)
        output_x1 = min(width, source_x - start_x + psf.shape[1])
        kernel_x0 = output_x0 + start_x - source_x
        kernel_x1 = output_x1 + start_x - source_x
        result[output_y, output_x0:output_x1] = psf[
            kernel_y, kernel_x0:kernel_x1
        ]
    return result


def delta_source_diagnostic(
    psf: np.ndarray, image_shape: tuple[int, int] = (256, 256)
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Verify the response to a centered unit-flux point source."""

    source_pixel = (image_shape[0] // 2, image_shape[1] // 2)
    delta = np.zeros(image_shape, dtype=np.float64)
    delta[source_pixel] = 1.0
    convolved = convolve_with_psf(delta, psf)
    expected = shifted_kernel_crop(psf, image_shape, source_pixel)
    residual = convolved - expected
    moments = image_moments(convolved)
    fwhm_x, fwhm_y = psf_fwhm_pixels(convolved)
    yy, xx = np.indices(image_shape, dtype=np.float64)
    radius = np.hypot(xx - moments.centroid_x_pixel, yy - moments.centroid_y_pixel)
    encircled = {
        str(aperture): float(convolved[radius <= aperture].sum())
        for aperture in (1, 2, 4, 8, 16, 32)
    }
    metrics: dict[str, object] = {
        "input_flux": 1.0,
        "full_kernel_flux": float(psf.sum()),
        "cropped_output_flux": moments.flux,
        "cropped_flux_loss": 1.0 - moments.flux,
        "source_pixel_yx": list(source_pixel),
        "centroid_x_pixel": moments.centroid_x_pixel,
        "centroid_y_pixel": moments.centroid_y_pixel,
        "centroid_offset_x_pixel": moments.centroid_x_pixel - source_pixel[1],
        "centroid_offset_y_pixel": moments.centroid_y_pixel - source_pixel[0],
        "sigma_x_pixel": moments.sigma_x_pixel,
        "sigma_y_pixel": moments.sigma_y_pixel,
        "fwhm_x_pixel": fwhm_x,
        "fwhm_y_pixel": fwhm_y,
        "maximum_absolute_residual": float(np.max(np.abs(residual))),
        "rms_residual": float(np.sqrt(np.mean(residual**2))),
        "encircled_flux_within_pixel_radius": encircled,
    }
    return metrics, {
        "delta": delta,
        "convolved": convolved,
        "expected": expected,
        "residual": residual,
    }


def load_throughput_curve(path: Path) -> ThroughputCurve:
    """Load a two-column wavelength-throughput table."""

    table = np.loadtxt(path, skiprows=1)
    if (
        table.ndim != 2
        or table.shape[1] < 2
        or len(table) < 3
        or not np.all(np.isfinite(table[:, :2]))
        or np.any(np.diff(table[:, 0]) <= 0.0)
        or np.any(table[:, 1] < 0.0)
        or float(table[:, 1].max()) <= 0.0
    ):
        raise ValueError("The throughput table is invalid")
    return ThroughputCurve(table[:, 0].copy(), table[:, 1].copy())


def photon_weighted_mean_fnu(
    wavelength_micron: np.ndarray,
    throughput: np.ndarray,
    fnu: np.ndarray,
) -> np.ndarray:
    """Average spectral flux density for a photon-counting detector."""

    wavelength = np.asarray(wavelength_micron, dtype=np.float64)
    response = np.asarray(throughput, dtype=np.float64)
    spectral_flux = np.asarray(fnu, dtype=np.float64)
    if spectral_flux.shape[-1] != len(wavelength):
        raise ValueError("The final spectral axis must match the throughput grid")
    denominator = np.trapezoid(response / wavelength, wavelength)
    if denominator <= 0.0:
        raise ValueError("The throughput integral must be positive")
    return np.trapezoid(
        spectral_flux * response / wavelength, wavelength, axis=-1
    ) / denominator


def interpolate_spectrum(
    wavelength_input: np.ndarray,
    values: np.ndarray,
    wavelength_output: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate one or more spectra along their final axis."""

    wavelength = np.asarray(wavelength_input, dtype=np.float64)
    output = np.asarray(wavelength_output, dtype=np.float64)
    spectra = np.asarray(values, dtype=np.float64)
    if spectra.shape[-1] != len(wavelength):
        raise ValueError("The final spectral axis must match the input wavelength grid")
    if output.min() < wavelength.min() or output.max() > wavelength.max():
        raise ValueError("The requested wavelengths extend beyond the spectrum")
    upper = np.searchsorted(wavelength, output, side="right")
    upper = np.clip(upper, 1, len(wavelength) - 1)
    lower = upper - 1
    fraction = (output - wavelength[lower]) / (wavelength[upper] - wavelength[lower])
    return spectra[..., lower] * (1.0 - fraction) + spectra[..., upper] * fraction


def observed_band_flux_njy_from_lnu(
    wavelength_rest_angstrom: np.ndarray,
    lnu_lsun_hz: np.ndarray,
    redshift: float,
    luminosity_distance_cm: float,
    throughput: ThroughputCurve,
) -> np.ndarray:
    """Convert a rest-frame luminosity spectrum to observed band flux density."""

    rest_wavelength = throughput.wavelength_micron * 1.0e4 / (1.0 + redshift)
    band_lnu_lsun_hz = interpolate_spectrum(
        wavelength_rest_angstrom, lnu_lsun_hz, rest_wavelength
    )
    fnu_cgs = (
        (1.0 + redshift)
        * band_lnu_lsun_hz
        * L_sun.cgs.value
        / (4.0 * np.pi * luminosity_distance_cm**2)
    )
    return photon_weighted_mean_fnu(
        throughput.wavelength_micron, throughput.throughput, fnu_cgs
    ) / 1.0e-32


def agn_power_law_band_flux_njy(
    bolometric_luminosity_erg_s: float,
    redshift: float,
    luminosity_distance_cm: float,
    throughput: ThroughputCurve,
    bolometric_correction_5100: float = 10.3,
    alpha_nu: float = -0.5,
) -> float:
    """Estimate an unobscured AGN band flux from a normalized power law."""

    if bolometric_luminosity_erg_s <= 0.0 or bolometric_correction_5100 <= 0.0:
        raise ValueError("The luminosity and bolometric correction must be positive")
    speed_of_light_angstrom_s = 2.99792458e18
    reference_wavelength_angstrom = 5100.0
    reference_frequency = speed_of_light_angstrom_s / reference_wavelength_angstrom
    reference_lnu = (
        bolometric_luminosity_erg_s
        / bolometric_correction_5100
        / reference_frequency
    )
    rest_wavelength_angstrom = (
        throughput.wavelength_micron * 1.0e4 / (1.0 + redshift)
    )
    rest_frequency = speed_of_light_angstrom_s / rest_wavelength_angstrom
    lnu = reference_lnu * (rest_frequency / reference_frequency) ** alpha_nu
    fnu_cgs = (
        (1.0 + redshift)
        * lnu
        / (4.0 * np.pi * luminosity_distance_cm**2)
    )
    return float(
        photon_weighted_mean_fnu(
            throughput.wavelength_micron, throughput.throughput, fnu_cgs
        )
        / 1.0e-32
    )


def read_hr5_age_table(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read scale factor, conformal time, and look-back time from the HR5 table."""

    table = np.loadtxt(path, skiprows=1)
    if table.ndim != 2 or table.shape[1] < 3 or np.any(np.diff(table[:, 1]) <= 0.0):
        raise ValueError("The HR5 age table is invalid")
    return table[:, 0], table[:, 1], table[:, 2]


def stellar_age_gyr(
    formation_time: np.ndarray,
    snapshot_time: float,
    conformal_time: np.ndarray,
    lookback_time_gyr: np.ndarray,
) -> np.ndarray:
    """Convert HR5 stellar birth times to ages at a stored output."""

    birth = np.asarray(formation_time, dtype=np.float64)
    minimum = float(conformal_time.min())
    maximum = float(conformal_time.max())
    tolerance = 1.0e-10
    if (
        np.any(birth < minimum - tolerance)
        or np.any(birth > snapshot_time + tolerance)
        or not minimum <= snapshot_time <= maximum
    ):
        raise ValueError("A stellar or snapshot time lies outside the HR5 age table")
    birth_lookback = np.interp(
        np.clip(birth, minimum, snapshot_time), conformal_time, lookback_time_gyr
    )
    snapshot_lookback = float(
        np.interp(snapshot_time, conformal_time, lookback_time_gyr)
    )
    age = birth_lookback - snapshot_lookback
    if np.any(age < -1.0e-8):
        raise ValueError("A stellar age is negative")
    return np.clip(age, 0.0, None)


def read_ramses_info(path: Path) -> dict[str, float]:
    """Read scalar entries from a RAMSES ``info`` file."""

    values: dict[str, float] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        if "=" not in line:
            continue
        key, raw_value = line.split("=", maxsplit=1)
        try:
            values[key.strip()] = float(raw_value.strip().split()[0])
        except ValueError:
            continue
    required = {"time", "aexp", "unit_l", "unit_d", "H0", "omega_m"}
    if not required.issubset(values):
        raise ValueError(f"The RAMSES info file lacks {sorted(required - values.keys())}")
    return values


def formed_stellar_mass_msun(
    initial_mass_code: np.ndarray, unit_length_cm: float, unit_density_g_cm3: float
) -> np.ndarray:
    """Convert the RAMSES initial stellar mass to solar masses."""

    solar_mass_g = 1.988409870698051e33
    return (
        np.asarray(initial_mass_code, dtype=np.float64)
        * unit_density_g_cm3
        * unit_length_cm**3
        / solar_mass_g
    )


def load_draine_dust_curve(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read wavelength, absorption opacity, scattering opacity, and albedo."""

    rows: list[list[float]] = []
    for line in path.read_text(encoding="ascii").splitlines():
        fields = line.split()
        if len(fields) < 6:
            continue
        try:
            values = [float(value) for value in fields[:6]]
        except ValueError:
            continue
        rows.append(values)
    table = np.asarray(rows, dtype=np.float64)
    if table.ndim != 2 or table.shape[1] != 6 or len(table) < 10:
        raise ValueError("The Draine dust table is invalid")
    order = np.argsort(table[:, 0])
    wavelength = table[order, 0]
    albedo = table[order, 1]
    absorption = table[order, 4]
    scattering = absorption * albedo / np.clip(1.0 - albedo, 1.0e-12, None)
    return wavelength, absorption, scattering, albedo


def dust_opacity_at_wavelength(
    wavelength_micron: float,
    dust_wavelength_micron: np.ndarray,
    absorption_opacity_cm2_g: np.ndarray,
    scattering_opacity_cm2_g: np.ndarray,
) -> tuple[float, float]:
    """Logarithmically interpolate dust absorption and scattering opacities."""

    if not dust_wavelength_micron.min() <= wavelength_micron <= dust_wavelength_micron.max():
        raise ValueError("The wavelength lies outside the dust opacity table")
    log_wavelength = np.log(dust_wavelength_micron)
    coordinate = np.log(wavelength_micron)
    absorption = float(
        np.exp(np.interp(coordinate, log_wavelength, np.log(absorption_opacity_cm2_g)))
    )
    scattering = float(
        np.exp(
            np.interp(
                coordinate,
                log_wavelength,
                np.log(np.clip(scattering_opacity_cm2_g, np.finfo(float).tiny, None)),
            )
        )
    )
    return absorption, scattering


def project_square_amr_cells(
    x_physical: np.ndarray,
    y_physical: np.ndarray,
    cell_width_physical: np.ndarray,
    weight: np.ndarray,
    edges_physical: np.ndarray,
) -> np.ndarray:
    """Project axis-aligned AMR cells as normalized square top hats."""

    x = np.asarray(x_physical, dtype=np.float64)
    y = np.asarray(y_physical, dtype=np.float64)
    width = np.asarray(cell_width_physical, dtype=np.float64)
    values = np.asarray(weight, dtype=np.float64)
    edges = np.asarray(edges_physical, dtype=np.float64)
    if not (x.shape == y.shape == width.shape == values.shape):
        raise ValueError("AMR cell coordinates, widths, and weights must have one shape")
    if np.any(width <= 0.0) or np.any(values < 0.0):
        raise ValueError("AMR cell widths must be positive and weights non-negative")
    pixel_width = np.diff(edges)
    if len(edges) < 2 or not np.allclose(pixel_width, pixel_width[0]):
        raise ValueError("The image edges must define a uniform grid")
    result = np.zeros((len(edges) - 1, len(edges) - 1), dtype=np.float64)
    for cell_width in np.unique(width):
        selected = width == cell_width
        impulses = np.histogram2d(
            y[selected], x[selected], bins=(edges, edges), weights=values[selected]
        )[0]
        kernel_size = max(1, int(np.ceil(cell_width / pixel_width[0])))
        kernel = np.full(
            (kernel_size, kernel_size), 1.0 / kernel_size**2, dtype=np.float64
        )
        result += fftconvolve(impulses, kernel, mode="same")
    result[result < 0.0] = 0.0
    return result


def apply_foreground_screen(
    intrinsic_image: np.ndarray,
    absorption_optical_depth: np.ndarray,
    scattering_optical_depth: np.ndarray,
    scattering_sigma_pixel: float,
) -> ForegroundScreenResult:
    """Apply an explicitly approximate foreground screen and scatter kernel."""

    intrinsic = np.asarray(intrinsic_image, dtype=np.float64)
    tau_abs = np.asarray(absorption_optical_depth, dtype=np.float64)
    tau_sca = np.asarray(scattering_optical_depth, dtype=np.float64)
    if intrinsic.shape != tau_abs.shape or intrinsic.shape != tau_sca.shape:
        raise ValueError("The source and optical-depth maps must have identical shapes")
    if (
        np.any(intrinsic < 0.0)
        or np.any(tau_abs < 0.0)
        or np.any(tau_sca < 0.0)
        or scattering_sigma_pixel < 0.0
    ):
        raise ValueError("Source intensity, optical depth, and scatter width must be non-negative")
    tau_extinction = tau_abs + tau_sca
    direct = intrinsic * np.exp(-np.clip(tau_extinction, 0.0, 700.0))
    interacted = intrinsic - direct
    albedo = np.divide(
        tau_sca,
        tau_extinction,
        out=np.zeros_like(tau_extinction),
        where=tau_extinction > 0.0,
    )
    scattered_before = interacted * albedo
    absorbed = interacted - scattered_before
    if scattering_sigma_pixel == 0.0:
        scattered_in_field = scattered_before.copy()
    else:
        scattered_in_field = gaussian_filter(
            scattered_before, sigma=scattering_sigma_pixel, mode="constant"
        )
    return ForegroundScreenResult(
        direct=direct,
        absorbed=absorbed,
        scattered_before_redistribution=scattered_before,
        scattered_in_field=scattered_in_field,
        emergent=direct + scattered_in_field,
    )

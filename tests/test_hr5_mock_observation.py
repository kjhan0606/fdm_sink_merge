from __future__ import annotations

import numpy as np
import pytest

from fdm_smbh_delay.hr5_mock_observation import (
    ThroughputCurve,
    apply_foreground_screen,
    agn_power_law_band_flux_njy,
    convolve_with_psf,
    delta_source_diagnostic,
    deposit_bilinear,
    dust_opacity_at_wavelength,
    formed_stellar_mass_msun,
    image_moments,
    photon_weighted_mean_fnu,
    project_square_amr_cells,
    psf_fwhm_pixels,
    stellar_age_gyr,
)


def test_bilinear_deposition_conserves_flux_and_centroid() -> None:
    image = np.zeros((8, 8), dtype=np.float64)
    deposit_bilinear(image, x_physical=0.0, y_physical=0.0, field_physical=8.0, weight=3.0)
    moments = image_moments(image)
    assert moments.flux == pytest.approx(3.0)
    assert moments.centroid_x_pixel == pytest.approx(3.5)
    assert moments.centroid_y_pixel == pytest.approx(3.5)


def test_delta_source_matches_direct_kernel_placement() -> None:
    coordinates = np.arange(-4, 5, dtype=np.float64)
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    psf = np.exp(-0.5 * (xx**2 + yy**2))
    psf /= psf.sum()
    metrics, layers = delta_source_diagnostic(psf, image_shape=(32, 32))
    assert metrics["full_kernel_flux"] == pytest.approx(1.0)
    assert metrics["cropped_output_flux"] == pytest.approx(1.0)
    assert metrics["maximum_absolute_residual"] < 1.0e-14
    assert np.allclose(layers["convolved"], layers["expected"], atol=1.0e-14)


def test_psf_fwhm_for_gaussian() -> None:
    coordinates = np.arange(-16, 17, dtype=np.float64)
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    sigma = 2.0
    psf = np.exp(-0.5 * (xx**2 + yy**2) / sigma**2)
    measured_x, measured_y = psf_fwhm_pixels(psf)
    expected = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma
    assert measured_x == pytest.approx(expected, rel=0.015)
    assert measured_y == pytest.approx(expected, rel=0.015)


def test_convolution_rejects_unnormalized_psf() -> None:
    with pytest.raises(ValueError, match="unit integral"):
        convolve_with_psf(np.ones((3, 3)), np.ones((3, 3)))


def test_photon_weighted_mean_preserves_constant_fnu() -> None:
    wavelength = np.linspace(1.0, 2.0, 100)
    throughput = np.exp(-0.5 * ((wavelength - 1.5) / 0.2) ** 2)
    result = photon_weighted_mean_fnu(wavelength, throughput, np.full(100, 3.5))
    assert result == pytest.approx(3.5)


def test_agn_band_flux_scales_with_luminosity() -> None:
    throughput = ThroughputCurve(
        np.linspace(1.8, 2.2, 100), np.ones(100, dtype=np.float64)
    )
    first = agn_power_law_band_flux_njy(1.0e44, 0.6, 1.0e28, throughput)
    second = agn_power_law_band_flux_njy(2.0e44, 0.6, 1.0e28, throughput)
    assert second == pytest.approx(2.0 * first)


def test_hr5_stellar_age_uses_lookback_difference() -> None:
    conformal = np.array([-3.0, -2.0, -1.0, 0.0])
    lookback = np.array([9.0, 6.0, 3.0, 0.0])
    age = stellar_age_gyr(np.array([-3.0, -1.5]), -1.0, conformal, lookback)
    assert np.allclose(age, [6.0, 1.5])


def test_formed_stellar_mass_uses_ramses_mass_unit() -> None:
    solar_mass_g = 1.988409870698051e33
    result = formed_stellar_mass_msun(np.array([2.0]), 10.0, solar_mass_g / 1000.0)
    assert result[0] == pytest.approx(2.0)


def test_dust_opacity_log_interpolation() -> None:
    wavelength = np.array([1.0, 2.0])
    absorption = np.array([100.0, 400.0])
    scattering = np.array([25.0, 100.0])
    k_abs, k_sca = dust_opacity_at_wavelength(
        np.sqrt(2.0), wavelength, absorption, scattering
    )
    assert k_abs == pytest.approx(200.0)
    assert k_sca == pytest.approx(50.0)


def test_zero_dust_returns_intrinsic_image() -> None:
    source = np.arange(9, dtype=np.float64).reshape(3, 3)
    result = apply_foreground_screen(
        source, np.zeros_like(source), np.zeros_like(source), 2.0
    )
    assert np.array_equal(result.direct, source)
    assert np.array_equal(result.emergent, source)
    assert float(result.absorbed.sum()) == 0.0
    assert float(result.scattered_in_field.sum()) == 0.0


def test_foreground_screen_energy_closes_before_scatter_crop() -> None:
    source = np.ones((16, 16), dtype=np.float64)
    result = apply_foreground_screen(
        source,
        np.full_like(source, 0.7),
        np.full_like(source, 0.3),
        1.5,
    )
    recovered = (
        result.direct.sum()
        + result.absorbed.sum()
        + result.scattered_before_redistribution.sum()
    )
    assert recovered == pytest.approx(source.sum())
    assert result.scattered_in_field.sum() < result.scattered_before_redistribution.sum()


def test_square_amr_projection_conserves_central_cell_mass() -> None:
    edges = np.arange(-8.5, 9.5, 1.0)
    image = project_square_amr_cells(
        np.array([0.0]),
        np.array([0.0]),
        np.array([3.0]),
        np.array([7.0]),
        edges,
    )
    assert image.sum() == pytest.approx(7.0)
    assert np.sum(image > 0.0) >= 9

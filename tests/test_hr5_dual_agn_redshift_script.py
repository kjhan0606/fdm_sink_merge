from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from fdm_smbh_delay.hr5 import redshift_rate_model


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "analyze_hr5_dual_agn_redshift.py"
)
SPEC = importlib.util.spec_from_file_location("analyze_hr5_dual_agn_redshift", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT)


def test_modified_schechter_density_fit_recovers_synthetic_curve() -> None:
    redshift = np.geomspace(0.4, 9.0, 20)
    expected = (7.0e-4, 4.3, 2.2, 3.1)
    density = redshift_rate_model(redshift, *expected)
    rows = [
        {
            "selection": "bol43",
            "redshift": float(z),
            "dual_pair_count": 100,
            "dual_pair_number_density_cmpc3": float(value),
        }
        for z, value in zip(redshift, density, strict=True)
    ]
    evaluation = np.array([0.2, 0.4, 1.0, 9.0, 10.0])

    fitted, parameters = SCRIPT._fit_modified_schechter_density(
        rows,
        "bol43",
        "dual_agn",
        "dual_pair_number_density_cmpc3",
        evaluation,
    )

    assert np.isnan(fitted[[0, -1]]).all()
    assert fitted[1:-1] == pytest.approx(
        redshift_rate_model(evaluation[1:-1], *expected), rel=2.0e-4
    )
    assert parameters["phi_star_cmpc3"] == pytest.approx(expected[0], rel=2.0e-4)
    assert parameters["z_star"] == pytest.approx(expected[1], rel=2.0e-4)
    assert parameters["alpha"] == pytest.approx(expected[2], rel=2.0e-4)
    assert parameters["beta"] == pytest.approx(expected[3], rel=2.0e-4)
    assert parameters["rms_log10_residual"] < 1.0e-8

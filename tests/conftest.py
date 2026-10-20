"""Constructed numerical fixtures for the public calibration tests."""

from __future__ import annotations

import numpy as np
import pytest

from rionmass.constants import amu
from rionmass.constants import me as me_mass


def make_row(
    name: str,
    Z: int,
    A: int,
    Q: int,
    flag: str,
    x: float,
    xerr: float,
    target_moq: float,
    target_moq_err: float,
    e_bind: float = 0.0,
) -> list:
    """Choose the mass excess so the ordinary conversion returns target_moq."""
    ME = target_moq * amu * Q - A * amu + Q * me_mass - e_bind
    MEerr = target_moq_err * amu * Q
    return [
        name,
        Z,
        A,
        A - Z,
        Q,
        flag,
        x,
        xerr,
        0.0,
        0.0,
        ME,
        MEerr,
        e_bind,
        target_moq,
        target_moq_err,
        "",
        name,
        f"{name}{Q}+",
    ]


@pytest.fixture
def linear_calibrants() -> list[list]:
    """Six noiseless references on m/q = a0 + a1*z, z = (x-300)/200."""
    a0, a1 = 2.0, 0.001
    xs = np.linspace(100.0, 500.0, 6)
    return [
        make_row(f"C{i}", 6, 12 + i, 6, "Y", float(xv), 0.01, a0 + a1 * ((xv - 300.0) / 200.0), 1e-6)
        for i, xv in enumerate(xs)
    ]


@pytest.fixture
def linear_calibration_params() -> tuple[float, float]:
    """The ``(a0, a1)`` the ``linear_calibrants`` fixture was built from."""
    return 2.0, 0.001


@pytest.fixture
def noisy_calibrants() -> list[list]:
    """Eight scattered references with a fixed seed for uncertainty checks."""
    rng = np.random.default_rng(20260813)
    a0, a1 = -1.5, 0.02
    xs = np.linspace(50.0, 450.0, 8)
    sigma = 5e-6
    rows = []
    for i, xv in enumerate(xs):
        z = (xv - 250.0) / 200.0
        true_moq = a0 + a1 * z
        noisy_moq = true_moq + rng.normal(0.0, sigma)
        rows.append(make_row(f"N{i}", 26, 56 + i, 26, "Y", float(xv), 0.05, noisy_moq, sigma))
    return rows


@pytest.fixture
def row_factory():
    """Expose :func:`make_row` as a fixture, for tests that need custom rows."""
    return make_row


@pytest.fixture
def unknown_row_on_line() -> list:
    """One unknown ion whose AME m/q matches the ``linear_calibrants`` line exactly."""
    # At z=0 the predicted m/q is the intercept.
    a0 = 2.0
    x = 300.0
    return make_row("U0", 92, 238, 92, "N", x, 0.01, a0, 1e-6)


@pytest.fixture
def invented_run(monkeypatch, tmp_path, linear_calibrants, unknown_row_on_line):
    from rionmass.api import run_full_calibration_workflow
    from rionmass.configuration import CalibrationConfig, ODSImportConfig
    from rionmass.electronic import ElectronicInputs

    rows = [*linear_calibrants, unknown_row_on_line]
    data = np.array([[r[0], r[1], r[4], r[5], r[6], r[7], 0, 0, "", r[16], r[17]] for r in rows], dtype=object)
    ame = [
        {"Element": r[0], "Z": r[1], "A": r[2], "ME (keV)": r[10] * 1000, "ME_error (keV)": r[11] * 1000} for r in rows
    ]
    monkeypatch.setattr(
        "rionmass.api.build_data_from_ods", lambda **kw: (data, {"x_label": "frequency", "x_unit": "Hz"})
    )
    monkeypatch.setattr("rionmass.api.Read_AME_barion", lambda: ame)
    source = tmp_path / "invented.ods"
    source.write_text("Invented ODS test input; parsing is stubbed.")
    ods = ODSImportConfig(source, 0, unmeasured_list=("U092+",), exclusion_list=(1,), harmonic=126)
    inputs = ElectronicInputs.from_corrections(
        {(6, 6): 0, (92, 92): 0}, covariance_MeV2=np.array([[1e-6, np.nan], [np.nan, 4e-6]])
    )
    calibration = CalibrationConfig(
        electronic_inputs=inputs,
        p=2,
        compare_orders=(2, 3),
        loo_anchor_species=("C06+",),
        electronic_method="refit",
        electronic_workers=2,
        isomer_excitation_energy={"invented_isomer": (0.1, 0.001)},
    )
    result = run_full_calibration_workflow(ods, calibration)
    return ods, calibration, result

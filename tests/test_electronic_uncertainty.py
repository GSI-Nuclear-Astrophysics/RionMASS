"""Shared-input electronic covariance and cancellation checks.

Constructed observables and published reference inputs isolate the
propagation formula and its agreement with refitted perturbations.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from rionmass.api import export_result_tables, run_full_calibration_workflow
from rionmass.configuration import CalibrationConfig, ODSImportConfig
from rionmass.electronic import (
    ElectronicInputs,
    binding_energy_inputs,
    electronic_mass_trials,
    ionization_energy_inputs,
    propagate_electronic_uncertainty,
    propagate_partial_covariance,
    read_electronic_inputs,
)
from rionmass.fitting import fit_calibration, predict_unknown_table
from rionmass.io import fetch_nist_electron_binding_table
from rionmass.uncertainty import leave_one_out_self_calibration


def test_2007_table_has_unknown_uncertainty():
    # Public reference values, not measurement data.
    table = [{"Z": 79, "TotBEn": 517015, "2e-ion": 184793}]
    inputs = binding_energy_inputs(table, [(79, 77)])
    assert inputs.correction(79, 77) == pytest.approx(0.332222)
    assert np.isnan(inputs.covariance_MeV2).all()
    assert inputs.correction_table()["uncertainty_known"].tolist() == [False]
    assert np.isnan(inputs.correction_table()["sigma_MeV"].iloc[0])


def test_nist_marginal_uncertainties_do_not_imply_independence():
    table = fetch_nist_electron_binding_table()
    inputs = binding_energy_inputs(table, [(79, 77)], "nist")
    sigmas = dict(zip(inputs.labels, inputs.primitive_table()["sigma_MeV"], strict=True))
    assert sigmas["B(Z=79,n=79)"] == pytest.approx(1100e-6)
    assert sigmas["B(Z=79,n=2)"] == pytest.approx(0.4e-6)
    assert np.isnan(inputs.correction_table()["sigma_MeV"].iloc[0])
    independent = binding_energy_inputs(table, [(79, 77)], "nist", assume_independent=True)
    assert independent.correction_table()["sigma_MeV"].iloc[0] == pytest.approx(np.hypot(1100, 0.4) / 1e6)
    assert "assumed independent" in independent.source


def test_bare_nist_ion_needs_only_neutral_uncertainty():
    inputs = binding_energy_inputs(fetch_nist_electron_binding_table(), [(79, 79)], "nist")
    assert inputs.correction_table()["sigma_MeV"].iloc[0] == pytest.approx(1100e-6)


def test_missing_nist_uncertainty_remains_unknown():
    table = {"6": [{"Ion Charge": "0", "Binding Energy (eV)": "1000", "Uncertainty (eV)": ""}]}
    inputs = binding_energy_inputs(table, [(6, 6)], "nist", assume_independent=True)
    assert np.isnan(inputs.covariance_MeV2[0, 0])


def test_explicit_total_binding_covariance_includes_subtraction_cross_term():
    table = {
        "6": [
            {"Ion Charge": "0", "Binding Energy (eV)": "1000"},
            {"Ion Charge": "+5", "Binding Energy (eV)": "500"},
        ]
    }
    inputs = binding_energy_inputs(table, [(6, 5)], "nist")
    cov = np.array([[4, 3], [3, 9]], dtype=float) * 1e-12
    evaluated = inputs.with_covariance(cov, "constructed covariance example")
    assert evaluated.correction_table()["sigma_MeV"].iloc[0] == pytest.approx(np.sqrt(4 + 9 - 6) * 1e-6)


def test_ionization_sum_and_source_groups():
    # Invented IE values/uncertainties, supplied source groups are explicit.
    table = {
        "6": [
            {"Ion Charge": "0", "Ionization Energy (eV)": "10", "Uncertainty (eV)": "1"},
            {"Ion Charge": "+1", "Ionization Energy (eV)": "20", "Uncertainty (eV)": "2"},
            {"Ion Charge": "+2", "Ionization Energy (eV)": "30", "Uncertainty (eV)": "3"},
        ]
    }
    groups = {"IE(Z=6,q=0)": "A", "IE(Z=6,q=1)": "A", "IE(Z=6,q=2)": "B"}
    inputs = ionization_energy_inputs(table, [(6, 2), (6, 3)], source_groups=groups)
    assert inputs.correction(6, 2) == pytest.approx(30e-6)
    assert inputs.correction(6, 3) == pytest.approx(60e-6)
    corrections = inputs.correction_table()
    assert corrections["sigma_MeV"].to_numpy() == pytest.approx([3e-6, np.sqrt(18) * 1e-6])
    cov = propagate_partial_covariance(inputs.response, inputs.covariance_MeV2, inputs.response)
    assert cov[0, 1] == pytest.approx(9e-12)


def test_total_binding_table_cannot_be_mistaken_for_individual_ionization_energies():
    with pytest.raises(KeyError, match="Ionization Energy"):
        ionization_energy_inputs(fetch_nist_electron_binding_table(), [(6, 1)])


def test_unknown_unused_input_does_not_contaminate_known_projection():
    covariance = np.array([[1.0, np.nan], [np.nan, np.nan]])
    jacobian = np.array([[2.0, 0], [0, 3.0]])
    result = propagate_partial_covariance(jacobian, covariance, jacobian)
    assert result[0, 0] == 4
    assert np.isnan(result[1, 1]) and np.isnan(result[0, 1])


def test_inconsistent_partial_covariance_cannot_hide_a_negative_variance():
    # A finite invalid principal block with unknown cross terms elsewhere.
    covariance = np.array([[1, -0.9, -0.9, np.nan], [-0.9, 1, -0.9, np.nan], [-0.9, -0.9, 1, np.nan], [np.nan] * 4])
    jacobian = np.array([[1, 1, 1, 0]])
    with pytest.raises(ValueError, match="negative projected variance"):
        propagate_partial_covariance(jacobian, covariance, jacobian)


def test_json_individual_ionization_energies(tmp_path):
    path = tmp_path / "invented_ionization_inputs.json"
    path.write_text(
        json.dumps(
            {
                "format": "ionization_energies",
                "table": {
                    "6": [
                        {"Ion Charge": "0", "Ionization Energy (eV)": "10", "Uncertainty (eV)": "1"},
                        {"Ion Charge": "1", "Ionization Energy (eV)": "20", "Uncertainty (eV)": "2"},
                    ]
                },
                "assume_independent": True,
                "source": "invented atomic inputs",
            }
        )
    )
    inputs = read_electronic_inputs(path, [(6, 2)])
    assert inputs.correction(6, 2) == pytest.approx(30e-6)
    assert inputs.correction_table()["sigma_MeV"].iloc[0] == pytest.approx(np.sqrt(5) * 1e-6)


@pytest.mark.parametrize(
    "cov",
    [
        [[-1, 0], [0, 1]],
        [[1, 2], [2, 1]],
        [[1, 0], [1, 1]],
        [[1, float("inf")], [float("inf"), 1]],
    ],
)
def test_invalid_covariance_is_rejected(cov):
    with pytest.raises(ValueError):
        ElectronicInputs.from_corrections({(6, 6): 0, (8, 8): 0}, covariance_MeV2=np.array(cov))


def test_zero_variance_is_known_zero_not_missing():
    inputs = ElectronicInputs.from_corrections({(6, 6): 0, (8, 8): 0}, sigmas_MeV={(6, 6): 0})
    assert inputs.covariance_MeV2[0].tolist() == [0, 0]
    assert np.isnan(inputs.covariance_MeV2[1, 1])


def test_non_psd_known_block_is_rejected_even_with_unknown_other_inputs():
    covariance = np.full((4, 4), np.nan)
    covariance[:3, :3] = -0.9
    np.fill_diagonal(covariance[:3, :3], 1)
    with pytest.raises(ValueError, match="positive semidefinite"):
        ElectronicInputs.from_corrections({(6, 6): 0, (8, 8): 0, (10, 10): 0, (12, 12): 0}, covariance_MeV2=covariance)


def test_json_correction_covariance_follows_entry_order(tmp_path):
    path = tmp_path / "toy_electronic.json"
    path.write_text(
        json.dumps(
            {
                "corrections": [{"Z": 8, "q": 8, "I_q_MeV": 0}, {"Z": 6, "q": 6, "I_q_MeV": 0}],
                "covariance_MeV2": [[4e-6, 1e-6], [1e-6, 1e-6]],
            }
        )
    )
    inputs = read_electronic_inputs(path, [(6, 6), (8, 8)])
    assert inputs.ions == ((6, 6), (8, 8))
    np.testing.assert_allclose(inputs.covariance_MeV2, [[1e-6, 1e-6], [1e-6, 4e-6]], rtol=1e-12)


def test_common_electronic_error_cancels_for_identical_element_and_charge(linear_calibrants):
    unknown = list(linear_calibrants[2])
    fit = fit_calibration(linear_calibrants, p=2)
    # All rows share (Z,q)=(6,6); its unknown absolute error cancels exactly
    # between calibration intercept and atomic recovery.
    inputs = ElectronicInputs.from_corrections({(6, 6): 0})
    result = propagate_electronic_uncertainty(fit, linear_calibrants, [unknown], inputs)
    assert result.mass_jacobian_keV_per_MeV[0, 0] == 0
    assert result.mass_covariance_keV2.iloc[0, 0] == 0
    assert np.isnan(result.coefficient_covariance.iloc[0, 0])


def test_correlated_mass_and_coefficient_propagation(linear_calibrants, unknown_row_on_line):
    fit = fit_calibration(linear_calibrants, p=2)
    second = list(unknown_row_on_line)
    second[6] = 340
    covariance = np.array([[1e-6, 1e-6], [1e-6, 4e-6]])
    inputs = ElectronicInputs.from_corrections({(6, 6): 0, (92, 92): 0}, covariance_MeV2=covariance)
    original_cov = fit.cov.copy()
    result = propagate_electronic_uncertainty(fit, linear_calibrants, [unknown_row_on_line, second], inputs)
    expected_j = np.array([[1000 * 92 / 6, -1000]] * 2)
    expected_cov = expected_j @ covariance @ expected_j.T
    np.testing.assert_allclose(result.mass_jacobian_keV_per_MeV, expected_j, rtol=1e-12)
    np.testing.assert_allclose(result.mass_covariance_keV2, expected_cov, rtol=1e-12)
    np.testing.assert_allclose(
        result.mass_coefficient_covariance,
        expected_j @ covariance @ result.coefficient_jacobian.T,
        rtol=1e-12,
        atol=1e-15,
    )
    np.testing.assert_array_equal(fit.cov, original_cov)  # separate component, not double counted
    assert result.total_mass_covariance_keV2.to_numpy() == pytest.approx(
        result.conditional_mass_covariance_keV2.to_numpy() + expected_cov
    )
    _, detail, _ = predict_unknown_table(fit, [unknown_row_on_line, second])
    np.testing.assert_allclose(np.diag(result.conditional_mass_covariance_keV2), detail["ME_total_err_keV"] ** 2)
    difference = np.array([1.0, -1.0])
    assert difference @ result.mass_covariance_keV2.to_numpy() @ difference == pytest.approx(0, abs=1e-10)


def test_covariance_inflation_does_not_inflate_electronic_response(noisy_calibrants, unknown_row_on_line):
    # Noisy fixture references share (Z,q)=(26,26), unknown=(92,92).
    plain = fit_calibration(noisy_calibrants, p=2)
    inflated = fit_calibration(noisy_calibrants, p=2, inflate_cov_by_redchi2=True)
    inputs = ElectronicInputs.from_corrections({(26, 26): 0, (92, 92): 0}, covariance_MeV2=np.diag([1e-6, 4e-6]))
    p = propagate_electronic_uncertainty(plain, noisy_calibrants, [unknown_row_on_line], inputs)
    i = propagate_electronic_uncertainty(inflated, noisy_calibrants, [unknown_row_on_line], inputs)
    np.testing.assert_allclose(p.mass_covariance_keV2, i.mass_covariance_keV2, rtol=1e-12)


def test_incomplete_inputs_produce_unknown_total_not_zero(linear_calibrants, unknown_row_on_line):
    fit = fit_calibration(linear_calibrants, p=2)
    inputs = ElectronicInputs.from_corrections({(6, 6): 0, (92, 92): 0})
    result = propagate_electronic_uncertainty(fit, linear_calibrants, [unknown_row_on_line], inputs)
    assert np.isnan(result.total_mass_covariance_keV2.iloc[0, 0])
    assert np.isfinite(result.conditional_mass_covariance_keV2.iloc[0, 0])


def test_refit_jacobian_matches_exact_linear_response_and_step_variation(linear_calibrants, unknown_row_on_line):
    fit = fit_calibration(linear_calibrants, p=2)
    inputs = ElectronicInputs.from_corrections(
        {(6, 6): 0, (92, 92): 0}, sigmas_MeV={(6, 6): 0.001, (92, 92): 0.002}, assume_independent=True
    )
    analytic = propagate_electronic_uncertainty(fit, linear_calibrants, [unknown_row_on_line], inputs)
    for step in (0.001, 0.002):
        numerical = propagate_electronic_uncertainty(
            fit, linear_calibrants, [unknown_row_on_line], inputs, method="refit", step_MeV=step, workers=2
        )
        np.testing.assert_allclose(
            numerical.mass_jacobian_keV_per_MeV, analytic.mass_jacobian_keV_per_MeV, rtol=1e-8, atol=1e-4
        )


def test_parallel_trials_reuse_shared_draws(linear_calibrants, unknown_row_on_line):
    fit = fit_calibration(linear_calibrants, p=2)
    inputs = ElectronicInputs.from_corrections(
        {(6, 6): 0, (92, 92): 0}, sigmas_MeV={(6, 6): 0.001, (92, 92): 0.002}, assume_independent=True
    )
    serial = electronic_mass_trials(fit, linear_calibrants, [unknown_row_on_line], inputs, replicates=32, seed=17)
    parallel = electronic_mass_trials(
        fit, linear_calibrants, [unknown_row_on_line], inputs, replicates=32, seed=17, workers=2
    )
    np.testing.assert_array_equal(serial, parallel)
    draws = np.random.default_rng(17).multivariate_normal(np.zeros(2), inputs.covariance_MeV2, size=32)
    _, central, _ = predict_unknown_table(fit, [unknown_row_on_line])
    expected = central["ME_state_keV"].to_numpy()[None, :] + draws @ np.array([[1000 * 92 / 6, -1000]]).T
    np.testing.assert_allclose(serial, expected, rtol=0, atol=1e-5)


def test_trials_reject_missing_covariance(linear_calibrants, unknown_row_on_line):
    fit = fit_calibration(linear_calibrants, p=2)
    inputs = ElectronicInputs.from_corrections({(6, 6): 0, (92, 92): 0})
    with pytest.raises(ValueError, match="complete electronic covariance"):
        electronic_mass_trials(fit, linear_calibrants, [unknown_row_on_line], inputs, replicates=2)


def test_iterative_weight_refit_jacobian_matches_shared_trials(row_factory):
    # Constructed positive m/q curve with unequal charges and visible x errors.
    xs = np.linspace(1900000, 2000000, 8)
    refs = []
    for i, x in enumerate(xs):
        z = (x - 1950000) / 50000
        y = 2 + 0.001 * z + 0.0001 * z * z + 2e-5 * np.cos(i)
        refs.append(row_factory(f"T{i}", 6 + i, 12 + 2 * i, 6 + i, "Y", float(x), 2000, float(y), 5e-6, e_bind=0.2))
    target = row_factory("TU", 20, 40, 20, "N", 1952000, 0, 2, 5e-6, e_bind=0.2)
    corrections = {(r[1], r[4]): 0.2 for r in [*refs, target]}
    inputs = ElectronicInputs.from_corrections(corrections, covariance_MeV2=np.eye(len(corrections)) * 1e-6)
    fit = fit_calibration(refs, p=3)
    numerical = propagate_electronic_uncertainty(fit, refs, [target], inputs, method="refit", workers=2)
    smaller = propagate_electronic_uncertainty(fit, refs, [target], inputs, method="refit", step_MeV=0.0005)
    np.testing.assert_allclose(
        numerical.mass_jacobian_keV_per_MeV, smaller.mass_jacobian_keV_per_MeV, rtol=1e-6, atol=1e-4
    )
    trials = electronic_mass_trials(fit, refs, [target], inputs, replicates=64, seed=18, workers=2)
    draws = np.random.default_rng(18).multivariate_normal(np.zeros(len(inputs.labels)), inputs.covariance_MeV2, size=64)
    _, central, _ = predict_unknown_table(fit, [target])
    first_order = central["ME_state_keV"].to_numpy()[None, :] + draws @ numerical.mass_jacobian_keV_per_MeV.T
    # Reweighting makes this estimator nonlinear. Its remainder should scale
    # quadratically as the input perturbation is reduced, not vanish identically.
    remainder = trials - first_order
    small_inputs = inputs.with_covariance(inputs.covariance_MeV2 * 0.01, "tenfold smaller perturbations")
    small_trials = electronic_mass_trials(fit, refs, [target], small_inputs, replicates=64, seed=18, workers=2)
    small_first_order = (
        central["ME_state_keV"].to_numpy()[None, :] + 0.1 * draws @ numerical.mass_jacobian_keV_per_MeV.T
    )
    small_remainder = small_trials - small_first_order
    assert np.max(np.abs(small_remainder)) < 0.03 * np.max(np.abs(remainder)) + 1e-6
    np.testing.assert_allclose(small_trials, small_first_order, rtol=0, atol=0.00005)


def test_loo_preserves_common_cancellation(linear_calibrants):
    inputs = ElectronicInputs.from_corrections({(6, 6): 0})
    result, _, _ = leave_one_out_self_calibration(linear_calibrants, p=2, electronic_inputs=inputs)
    assert np.all(result["electronic_uncertainty_known"].to_numpy())
    assert (result["ME_electronic_err_keV"] == 0).all()
    np.testing.assert_allclose(result.attrs["electronic_covariance_keV2"], 0)


@pytest.mark.parametrize("known", [True, False])
def test_workflow_exports_conditional_and_electronic_covariance(
    monkeypatch,
    tmp_path,
    linear_calibrants,
    unknown_row_on_line,
    known,
):
    rows = [*linear_calibrants, unknown_row_on_line]
    data = np.array([[r[0], r[1], r[4], r[5], r[6], r[7], 0, 0, "", r[16], r[17]] for r in rows], dtype=object)
    ame = [
        {"Element": r[0], "Z": r[1], "A": r[2], "ME (keV)": r[10] * 1000, "ME_error (keV)": r[11] * 1000} for r in rows
    ]
    monkeypatch.setattr(
        "rionmass.api.build_data_from_ods", lambda **kwargs: (data, {"x_label": "frequency", "x_unit": "Hz"})
    )
    monkeypatch.setattr("rionmass.api.Read_AME_barion", lambda: ame)
    inputs = ElectronicInputs.from_corrections(
        {(6, 6): 0, (92, 92): 0},
        covariance_MeV2=np.diag([1e-6, 4e-6]) if known else np.full((2, 2), np.nan),
    )
    result = run_full_calibration_workflow(
        ODSImportConfig("invented.ods", 0),
        CalibrationConfig(electronic_inputs=inputs, p=2, compare_orders=(2,)),
    )
    assert result.electronic_uncertainty_complete == known
    assert result.unknown_df["electronic_uncertainty_known"].iloc[0] == known
    assert np.isfinite(result.unknown_df["ME_conditional_err_keV"].iloc[0])
    assert np.isfinite(result.unknown_df["ME_total_err_keV"].iloc[0]) == known
    files = export_result_tables(result, str(tmp_path))
    assert any(p.endswith("electronic_input_covariance_MeV2.csv") for p in files)
    covariance = pd.read_csv(tmp_path / "mass_covariance_keV2.csv", index_col=0)
    assert np.isfinite(covariance.iloc[0, 0]) == known


@pytest.mark.parametrize(
    "source,independent,known", [("file", False, False), ("nist", False, False), ("nist", True, True)]
)
def test_workflow_binding_table_defaults(
    monkeypatch, linear_calibrants, unknown_row_on_line, source, independent, known
):
    # All calibration rows and atomic table entries in this test are invented.
    rows = [*linear_calibrants, unknown_row_on_line]
    data = np.array([[r[0], r[1], r[4], r[5], r[6], r[7], 0, 0, "", r[16], r[17]] for r in rows], dtype=object)
    ame = [
        {"Element": r[0], "Z": r[1], "A": r[2], "ME (keV)": r[10] * 1000, "ME_error (keV)": r[11] * 1000} for r in rows
    ]
    monkeypatch.setattr(
        "rionmass.api.build_data_from_ods", lambda **kwargs: (data, {"x_label": "frequency", "x_unit": "Hz"})
    )
    monkeypatch.setattr("rionmass.api.Read_AME_barion", lambda: ame)
    monkeypatch.setattr(
        "rionmass.api.Read_elbien_file", lambda path: [{"Z": 6, "TotBEn": 1000}, {"Z": 92, "TotBEn": 1000}]
    )
    monkeypatch.setattr(
        "rionmass.api.fetch_nist_electron_binding_table",
        lambda path: {
            "6": [{"Ion Charge": "0", "Binding Energy (eV)": "1000", "Uncertainty (eV)": "1"}],
            "92": [{"Ion Charge": "0", "Binding Energy (eV)": "1000", "Uncertainty (eV)": "2"}],
        },
    )
    result = run_full_calibration_workflow(
        ODSImportConfig("invented.ods", 0),
        CalibrationConfig(
            elbien_source=source,
            elbien_path="invented.dat",
            electronic_assume_independent=independent,
            p=2,
            compare_orders=(2,),
        ),
    )
    assert result.electronic_uncertainty_complete == known
    assert result.unknown_df["electronic_uncertainty_known"].iloc[0] == known
    if known:
        # Identical reference charges: reference I/q response is constant.
        assert result.unknown_df["ME_electronic_err_keV"].iloc[0] == pytest.approx(np.hypot(92 / 6, 2) * 0.001)

"""Anchor validation policies, using invented data only."""

import numpy as np
import pandas as pd
import pytest

from rionmass.configuration import CalibrationConfig, ODSImportConfig
from rionmass.processing import get_isomer_metadata
from rionmass.uncertainty import (
    compare_polynomial_orders,
    estimate_extra_systematic_from_loo,
    leave_one_out_self_calibration,
)


@pytest.mark.parametrize("policy", ["retain_in_training", "remove_from_validation"])
def test_anchor_training_and_excluded_holdout(linear_calibrants, policy):
    anchor = get_isomer_metadata(linear_calibrants[0])[2]
    ordinary, _, _ = leave_one_out_self_calibration(linear_calibrants, p=2)
    loo, legacy, fits = leave_one_out_self_calibration(
        linear_calibrants, p=2, loo_anchor_species=(anchor,), loo_anchor_policy=policy
    )
    assert len(loo) == len(legacy) == len(fits) == 6
    assert loo.used_for_extra_scatter.tolist() == [False, True, True, True, True, True]
    assert loo.loo_role.iloc[0] == "Anchor diagnostic only"
    assert loo.loo_extrapolation.iloc[0]
    np.testing.assert_allclose(loo.iloc[0].ME_state_keV, ordinary.iloc[0].ME_state_keV)
    for fit in fits[1:]:
        assert (anchor in fit.labels) == (policy == "retain_in_training")
    assert loo.loo_training_species.iloc[1].split(";") == fits[1].labels
    result = estimate_extra_systematic_from_loo(loo, p=2)
    assert result["N"] == 5
    assert result["dof"] == 3
    assert result["n_excluded"] == 1
    assert result["loo_anchor_species"] == (anchor,)
    assert result["loo_anchor_policy"] == policy
    assert result["extra_sys_keV"] == 0
    np.testing.assert_allclose(result["chi2_raw"], np.sum(loo.loo_pull_no_extra.to_numpy()[1:] ** 2))


def test_anchor_residual_does_not_set_scatter_but_other_residuals_do():
    loo = pd.DataFrame(
        {
            "Delta_ME_state_keV": [1e6, 1, -1, 2, -2],
            "loo_sigma_no_extra_keV": [1, 1, 1, 1, 1],
            "used_for_extra_scatter": [False, True, True, True, True],
        }
    )
    result = estimate_extra_systematic_from_loo(loo, p=2)
    assert result["chi2_raw"] == 10
    assert result["extra_sys_keV"] == pytest.approx(2)
    assert result["table"].extra_sys_keV.iloc[0] == 0
    loo.loc[0, "Delta_ME_state_keV"] = -1e9
    assert estimate_extra_systematic_from_loo(loo, p=2)["extra_sys_keV"] == pytest.approx(2)


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"loo_anchor_species": ("missing",)}, "exactly one reference"),
        ({"loo_anchor_species": ("C06+", "C06+")}, "distinct species"),
        ({"loo_anchor_species": "C06+"}, "distinct species"),
        ({"loo_anchor_species": ("C06+",), "loo_anchor_policy": "wrong"}, "loo_anchor_policy"),
        ({"loo_anchor_species": ("C06+", "C16+", "C26+", "C36+")}, "Too few"),
    ],
)
def test_invalid_anchors_rejected(linear_calibrants, kwargs, message):
    with pytest.raises(ValueError, match=message):
        leave_one_out_self_calibration(linear_calibrants, p=2, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"loo_anchor_species": ("A", "A")},
        {"loo_anchor_species": "A"},
        {"loo_anchor_species": (" A",)},
        {"loo_anchor_policy": "wrong"},
    ],
)
def test_invalid_anchor_configuration(kwargs):
    with pytest.raises(ValueError, match="loo_anchor"):
        CalibrationConfig(elbien_source="nist", **kwargs)


def test_order_comparison_uses_same_validation_policy(linear_calibrants):
    table = compare_polynomial_orders(linear_calibrants, p_values=(2, 3, 5), loo_anchor_species=("C06+",))
    assert table.loc[table.p_terms == 2, "n_loo"].iloc[0] == 5
    assert table.loc[table.p_terms == 3, "n_loo"].iloc[0] == 5
    assert table.loc[table.p_terms == 2, "loo_anchor_policy"].iloc[0] == "retain_in_training"
    assert "Too few" in table.loc[table.p_terms == 5, "error"].iloc[0]


@pytest.mark.parametrize("policy", ["retain_in_training", "remove_from_validation"])
def test_workflow_preserves_full_fit_and_exports_policy(
    monkeypatch, tmp_path, linear_calibrants, unknown_row_on_line, policy
):
    from rionmass.api import export_result_tables, run_full_calibration_workflow
    from rionmass.electronic import ElectronicInputs

    rows = [*linear_calibrants, unknown_row_on_line]
    data = np.array([[r[0], r[1], r[4], r[5], r[6], r[7], 0, 0, "", r[16], r[17]] for r in rows], dtype=object)
    ame = [
        {"Element": r[0], "Z": r[1], "A": r[2], "ME (keV)": r[10] * 1000, "ME_error (keV)": r[11] * 1000} for r in rows
    ]
    monkeypatch.setattr(
        "rionmass.api.build_data_from_ods", lambda **kwargs: (data, {"x_label": "frequency", "x_unit": "Hz"})
    )
    monkeypatch.setattr("rionmass.api.Read_AME_barion", lambda: ame)
    inputs = ElectronicInputs.from_corrections({(6, 6): 0, (92, 92): 0}, covariance_MeV2=np.diag([1e-6, 4e-6]))
    ordinary = run_full_calibration_workflow(
        ODSImportConfig("invented.ods", 0), CalibrationConfig(electronic_inputs=inputs, p=2, compare_orders=(2,))
    )
    anchored = run_full_calibration_workflow(
        ODSImportConfig("invented.ods", 0),
        CalibrationConfig(
            electronic_inputs=inputs, p=2, compare_orders=(2,), loo_anchor_species=("C06+",), loo_anchor_policy=policy
        ),
    )
    np.testing.assert_array_equal(ordinary.fit.coeff, anchored.fit.coeff)
    np.testing.assert_array_equal(ordinary.fit.cov, anchored.fit.cov)
    assert anchored.fit.labels == ordinary.fit.labels
    assert anchored.fit.sigma_eff_y[0] > 0
    np.testing.assert_array_equal(ordinary.unknown_df.ME_state_keV, anchored.unknown_df.ME_state_keV)
    np.testing.assert_array_equal(ordinary.unknown_df.ME_stat_err_keV, anchored.unknown_df.ME_stat_err_keV)
    assert ordinary.electronic is not None and anchored.electronic is not None
    np.testing.assert_array_equal(ordinary.electronic.mass_covariance_keV2, anchored.electronic.mass_covariance_keV2)
    written = export_result_tables(anchored, str(tmp_path))
    assert len(written) == len(set(written))
    assert pd.read_csv(tmp_path / "loo_validation_summary.csv").loo_anchor_policy.iloc[0] == policy
    table = pd.read_csv(tmp_path / "loo_reference_results.csv")
    assert table["Validation use"].iloc[0] == "Anchor diagnostic only"
    assert "ΔME vs AME2020 (keV)" in table
    detailed = pd.read_csv(tmp_path / "leave_one_out_with_extra_sys.csv")
    assert detailed.used_for_extra_scatter.tolist() == [False, True, True, True, True, True]


def test_cli_parses_anchor_options():
    from rionmass.cli import _build_parser

    args = _build_parser().parse_args(
        [
            "run",
            "--ods",
            "invented.ods",
            "--sheet-index",
            "0",
            "--elbien-source",
            "nist",
            "--loo-anchor",
            "A",
            "--loo-anchor",
            "B",
            "--loo-anchor-policy",
            "remove_from_validation",
        ]
    )
    assert args.loo_anchor == ["A", "B"]
    assert args.loo_anchor_policy == "remove_from_validation"

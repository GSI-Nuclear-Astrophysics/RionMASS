"""Regression checks on invented masses only; no experimental measurements."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rionmass.results import final_mass_results, ion_display_label, loo_reference_results


@pytest.fixture
def invented_results():
    return pd.DataFrame(
        {
            "species_id": ["12C6+", "14N7+"],
            "E_iso_keV": [0.0, 0.0],
            "ME_state_keV": [10.0, 20.0],
            "ME_AME_state_keV": [11.0, 18.0],
            "Delta_ME_state_keV": [-1.0, 2.0],
            "ME_AME_state_err_keV": [2.0, 3.0],
            "ME_stat_err_keV": [0.5, 0.6],
            "ME_extra_sys_keV": [1.0, 1.0],
            "ME_electronic_err_keV": [np.nan, np.nan],
            "ME_conditional_err_keV": [1.1, 1.2],
            "ME_total_err_keV": [np.nan, np.nan],
        }
    )


def test_readable_results_keep_unknown_inputs_and_numeric_precision(invented_results):
    before = invented_results.copy(deep=True)
    table = final_mass_results(invented_results)
    assert table.Ion.tolist() == ["¹²C⁶⁺", "¹⁴N⁷⁺"]
    assert table["ME (keV)"].tolist() == [10.0, 20.0]
    assert table["ΔME vs AME2020 (keV)"].tolist() == [-1.0, 2.0]
    assert table["σ AME2020 (keV)"].tolist() == [2.0, 3.0]
    assert table["σ fit (keV)"].tolist() == [0.5, 0.6]
    assert table["σ LOO extra (keV)"].tolist() == [1.0, 1.0]
    assert table["σ combined (keV)"].isna().to_numpy().all()
    assert table["σ conditional (keV)"].tolist() == [1.1, 1.2]
    pd.testing.assert_frame_equal(before, invented_results)
    assert ion_display_label("14mN7+") == "¹⁴ᵐN⁷⁺"


@pytest.mark.parametrize("name", ["plot_unknown_mass_excess_vs_ame", "plot_unknown_mass_excess_residuals"])
@pytest.mark.parametrize("case", ["conditional", "mixed", "missing", "empty", "only_isomer"])
def test_unknown_plots_retain_masses_with_incomplete_errors(invented_results, name, case):
    pytest.importorskip("matplotlib")
    from matplotlib.container import ErrorbarContainer
    from matplotlib.figure import Figure

    from rionmass import plotting

    df = invented_results.copy()
    if case == "mixed":
        df.loc[0, "ME_total_err_keV"] = 1.3
    elif case == "missing":
        df = df.drop(columns=["ME_conditional_err_keV"])
    elif case == "empty":
        df = pd.DataFrame()
    elif case == "only_isomer":
        df["species_id"] = ["12mC6+", "14mN7+"]
    ax = Figure().add_subplot(111)
    getattr(plotting, name)(df, ax=ax, ground_only=True)
    if case in ("empty", "only_isomer"):
        assert any("No finite unknown" in t.get_text() for t in ax.texts)
    else:
        labels = ax.get_legend_handles_labels()[1]
        expected = "uncertainty unavailable" if case == "missing" else "conditional uncertainties"
        assert expected in labels
        if case == "mixed":
            assert "combined uncertainties" in labels
        data_points = [container.lines[0] for container in ax.containers if isinstance(container, ErrorbarContainer)]
        assert sum(np.asarray(line.get_xdata()).size for line in data_points) == 2


def test_low_full_fit_chi_square_can_coexist_with_excess_loo_scatter(row_factory):
    from rionmass.fitting import fit_calibration
    from rionmass.uncertainty import estimate_extra_systematic_from_loo, leave_one_out_self_calibration

    sigma = 1e-5
    refs = [
        row_factory(f"T{i}", 6, 12 + i, 6, "Y", float(i), 0.0, 2 + i * 0.001 + offset * sigma, sigma)
        for i, offset in enumerate([0.6, -0.6, -0.6, 0.6])
    ]
    fit = fit_calibration(refs, p=2)
    loo, _, _ = leave_one_out_self_calibration(refs, p=2)
    result = estimate_extra_systematic_from_loo(loo, p=2)
    assert fit.red_chi2 == pytest.approx(0.72, rel=1e-6)
    assert result["red_chi2_raw"] > 1
    assert result["extra_sys_keV"] > 0


def test_loo_results_separate_prediction_and_comparison_errors(invented_results):
    df = invented_results.copy()
    # LOO low-level predictions contain conditional fit errors, not the AME
    # comparison error or separately propagated electronic component.
    df["ME_total_err_keV"] = [3.0, 4.0]
    df["ME_electronic_err_keV"] = [4.0, np.nan]
    df["loo_sigma_no_extra_keV"] = [np.sqrt(29), 5.0]
    df["loo_sigma_with_extra_keV"] = [np.sqrt(30), np.sqrt(26)]
    df["loo_role"] = ["Anchor diagnostic only", "Validation reference"]
    df["loo_extrapolation"] = [True, False]
    df["electronic_uncertainty_known"] = [True, False]
    out = loo_reference_results(df)
    assert out["ΔME vs AME2020 (keV)"].tolist() == [-1.0, 2.0]
    assert out["σ AME2020 (keV)"].tolist() == [2.0, 3.0]
    assert out["σ prediction conditional (keV)"].tolist() == [3.0, 4.0]
    assert out["σ prediction combined (keV)"].iloc[0] == 5.0
    assert np.isnan(out["σ prediction combined (keV)"].iloc[1])
    assert out["Validation use"].iloc[0] == "Anchor diagnostic only"
    assert out["Held-out geometry"].tolist() == ["Extrapolation", "Interpolation"]


def test_empty_result_views_keep_column_contract():
    assert "σ AME2020 (keV)" in final_mass_results(pd.DataFrame())
    assert "Validation use" in loo_reference_results(pd.DataFrame())

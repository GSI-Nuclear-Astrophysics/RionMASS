"""Calibration, conditioning and uncertainty checks.

Exact-line fixtures use abs=1e-9 for coefficients and chi-square < 1e-6,
allowing floating-point and iterative-solver rounding.
"""

from __future__ import annotations

import numpy as np
import pytest

from rionmass.fitting import MassCalibration, fit_calibration, predict_unknown_table


class TestFitCalibration:
    def test_recovers_exact_linear_coefficients(self, linear_calibrants, linear_calibration_params):
        a0, a1 = linear_calibration_params
        fit = fit_calibration(linear_calibrants, p=2, observable_label="freq", observable_unit="Hz")
        assert fit.coeff[0] == pytest.approx(a0, abs=1e-9)
        assert fit.coeff[1] == pytest.approx(a1, abs=1e-9)
        assert fit.chi2 < 1e-6
        assert fit.dof == len(linear_calibrants) - 2

    def test_flow_of_information_significance_sums_to_p(self, linear_calibrants):
        fit = fit_calibration(linear_calibrants, p=2)
        assert np.sum(fit.significance) == pytest.approx(fit.p, abs=1e-9)

    def test_too_few_calibrants_raises(self, linear_calibrants):
        with pytest.raises(ValueError, match="Need at least p="):
            fit_calibration(linear_calibrants[:1], p=3)

    def test_centering_choice_does_not_change_fitted_curve(self, linear_calibrants):
        # Changing the polynomial basis must preserve physical predictions.
        fit_auto = fit_calibration(linear_calibrants, p=2)
        fit_manual = fit_calibration(linear_calibrants, p=2, x0=0.0, xscale=1.0)
        x_probe = 275.0
        y_auto = fit_auto.predict_moq(x_probe)["moq"]
        y_manual = fit_manual.predict_moq(x_probe)["moq"]
        assert float(y_auto) == pytest.approx(float(y_manual), rel=1e-8)

    def test_noisy_fit_chi2_is_finite_and_positive_dof(self, noisy_calibrants):
        fit = fit_calibration(noisy_calibrants, p=2)
        assert fit.dof == len(noisy_calibrants) - 2
        assert np.isfinite(fit.chi2)
        assert fit.chi2 >= 0.0

    def test_default_does_not_inflate_covariance_by_redchi2(self, noisy_calibrants):
        # Covariance inflation is explicitly opt-in.
        fit_default = fit_calibration(noisy_calibrants, p=2)
        fit_inflated = fit_calibration(noisy_calibrants, p=2, inflate_cov_by_redchi2=True)
        if fit_default.red_chi2 > 1.0:
            assert np.trace(fit_inflated.cov) > np.trace(fit_default.cov)
        else:
            assert np.trace(fit_inflated.cov) == pytest.approx(np.trace(fit_default.cov))


class TestPredictUnknownTable:
    def test_predicts_known_value_on_the_line(self, linear_calibrants, linear_calibration_params, unknown_row_on_line):
        a0, _a1 = linear_calibration_params
        fit = fit_calibration(linear_calibrants, p=2)
        _rows, detail_df, influence = predict_unknown_table(fit, [unknown_row_on_line])
        assert detail_df["MoQ_fit"].iloc[0] == pytest.approx(a0, abs=1e-8)
        assert detail_df["Delta_ME_state_keV"].iloc[0] == pytest.approx(0.0, abs=1e-3)
        assert list(influence.keys()) == ["U092+"]

    def test_extra_sys_added_in_quadrature(self, linear_calibrants, unknown_row_on_line):
        fit = fit_calibration(linear_calibrants, p=2)
        _rows0, df0, _ = predict_unknown_table(fit, [unknown_row_on_line], extra_sys_keV=0.0)
        _rows1, df1, _ = predict_unknown_table(fit, [unknown_row_on_line], extra_sys_keV=10.0)
        stat = df0["ME_total_err_keV"].iloc[0]
        total = df1["ME_total_err_keV"].iloc[0]
        assert total == pytest.approx((stat**2 + 10.0**2) ** 0.5, rel=1e-9)


class TestMassCalibrationCompatWrapper:
    def test_default_return_shape_is_three_tuple(self, linear_calibrants, unknown_row_on_line):
        result = MassCalibration(linear_calibrants, [unknown_row_on_line], False, p=2, A0_Y=None, iterationMax=50)
        assert len(result) == 3
        rows, coeff, chi2 = result
        assert len(rows) == 1
        assert len(coeff) == 2

    def test_a0_y_is_accepted_and_ignored(self, linear_calibrants, unknown_row_on_line):
        # A0_Y is retained for API compatibility; this fit needs no seed.
        r1 = MassCalibration(linear_calibrants, [unknown_row_on_line], False, p=2, A0_Y=[1, 2], iterationMax=50)
        r2 = MassCalibration(linear_calibrants, [unknown_row_on_line], False, p=2, A0_Y=[999, -999], iterationMax=50)
        assert r1[1] == pytest.approx(r2[1])


def test_covariance_inflation_preserves_reference_response(linear_calibrants):
    """An uncertainty-scale choice must preserve Audi flow and prediction weights."""
    from rionmass.constants import amu
    from rionmass.models import IDX_ME_AME, IDX_Q

    references = [list(row) for row in linear_calibrants]
    references[2][IDX_ME_AME] += 1e-4 * references[2][IDX_Q] * amu
    nominal = fit_calibration(references, p=2)
    inflated = fit_calibration(references, p=2, inflate_cov_by_redchi2=True)
    assert nominal.red_chi2 > 1
    np.testing.assert_allclose(inflated.coeff, nominal.coeff, rtol=0, atol=0)
    np.testing.assert_allclose(inflated.cov, nominal.cov * nominal.red_chi2)
    np.testing.assert_allclose(inflated.R, nominal.R, rtol=0, atol=0)
    np.testing.assert_allclose(inflated.significance, nominal.significance)
    np.testing.assert_allclose(inflated.flow_coeff.sum(axis=0), np.ones(nominal.p))
    assert inflated.significance.sum() == pytest.approx(nominal.p)

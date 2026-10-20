"""LOO extra scatter, order comparison and reference-removal checks.

The root-solver check uses rel=1e-6 on the resulting reduced statistic.
"""

from __future__ import annotations

import numpy as np
import pytest

from rionmass.processing import get_isomer_metadata
from rionmass.uncertainty import (
    calibrant_geometry_sweep,
    compare_polynomial_orders,
    estimate_extra_systematic_from_loo,
    leave_one_out_self_calibration,
)


class TestLeaveOneOutSelfCalibration:
    def test_one_row_per_calibrant(self, linear_calibrants):
        loo_df, legacy_rows, fits = leave_one_out_self_calibration(linear_calibrants, p=2)
        assert len(loo_df) == len(linear_calibrants)
        assert len(legacy_rows) == len(linear_calibrants)
        assert len(fits) == len(linear_calibrants)

    def test_exact_line_gives_near_zero_residuals(self, linear_calibrants):
        # Each holdout leaves the exact generating line identifiable.
        loo_df, _, _ = leave_one_out_self_calibration(linear_calibrants, p=2)
        assert np.allclose(loo_df["Delta_ME_state_keV"], 0.0, atol=1e-3)


class TestEstimateExtraSystematic:
    def test_no_inflation_needed_for_exact_line(self, linear_calibrants):
        loo_df, _, _ = leave_one_out_self_calibration(linear_calibrants, p=2)
        result = estimate_extra_systematic_from_loo(loo_df, p=2)
        assert result["extra_sys_keV"] == pytest.approx(0.0, abs=1e-6)
        assert result["red_chi2_raw"] <= 1.0 + 1e-6

    def test_root_solve_makes_red_chi2_equal_one(self, noisy_calibrants):
        loo_df, _, _ = leave_one_out_self_calibration(noisy_calibrants, p=2)
        result = estimate_extra_systematic_from_loo(loo_df, p=2)
        if result["red_chi2_raw"] > 1.0:
            assert result["red_chi2_with_sys"] == pytest.approx(1.0, rel=1e-6)
            assert result["extra_sys_keV"] > 0.0
        else:
            assert result["extra_sys_keV"] == 0.0

    def test_table_has_augmented_columns(self, linear_calibrants):
        loo_df, _, _ = leave_one_out_self_calibration(linear_calibrants, p=2)
        result = estimate_extra_systematic_from_loo(loo_df, p=2)
        table = result["table"]
        assert "loo_sigma_with_extra_keV" in table.columns
        assert "loo_pull_with_extra" in table.columns
        assert len(table) == len(loo_df)


class TestComparePolynomialOrders:
    def test_skips_underdetermined_orders(self, linear_calibrants):
        # Underdetermined orders are skipped.
        df = compare_polynomial_orders(linear_calibrants, p_values=[2, 3, 8])
        assert set(df["p_terms"]) == {2, 3}

    def test_lower_order_sufficient_for_exact_line(self, linear_calibrants):
        df = compare_polynomial_orders(linear_calibrants, p_values=[2, 3])
        row_p2 = df[df["p_terms"] == 2].iloc[0]
        assert row_p2["rms_residual_keV"] < 1.0


class TestCalibrantGeometrySweep:
    def test_zero_delta_when_remaining_calibrants_still_exact(self, linear_calibrants, unknown_row_on_line):
        # Removing one exact-line reference must preserve the target prediction.
        candidate = get_isomer_metadata(linear_calibrants[0])[2]
        result = calibrant_geometry_sweep(linear_calibrants, [unknown_row_on_line], candidate_labels=[candidate], p=2)
        assert len(result) == 1
        row = result.iloc[0]
        assert not row["skipped"]
        assert row["n_calibrants_baseline"] == 6
        assert row["n_calibrants_without_candidate"] == 5
        assert row["delta_ME_state_keV"] == pytest.approx(0.0, abs=1e-6)
        assert row["delta_extra_sys_keV"] == pytest.approx(0.0, abs=1e-6)

    def test_skips_when_removal_underdetermines_fit(self, row_factory, unknown_row_on_line):
        # Removing a reference from a saturated fit leaves no identifiable solution.
        calibrants = [
            row_factory("A0", 6, 12, 6, "Y", 100.0, 0.01, 2.0, 1e-6),
            row_factory("A1", 6, 13, 6, "Y", 200.0, 0.01, 2.1, 1e-6),
            row_factory("A2", 6, 14, 6, "Y", 300.0, 0.01, 2.2, 1e-6),
        ]
        result = calibrant_geometry_sweep(calibrants, [unknown_row_on_line], candidate_labels=["A06+"], p=3)
        assert len(result) == 1
        row = result.iloc[0]
        assert row["skipped"]
        assert row["n_calibrants_without_candidate"] == 2
        assert np.isnan(row["delta_ME_state_keV"])
        assert np.isnan(row["delta_extra_sys_keV"])

    def test_raises_on_unknown_candidate_label(self, linear_calibrants, unknown_row_on_line):
        with pytest.raises(ValueError, match="not found in Nuclei_Y"):
            calibrant_geometry_sweep(linear_calibrants, [unknown_row_on_line], candidate_labels=["does-not-exist"], p=2)

    def test_one_row_per_candidate_times_unknown(self, linear_calibrants, unknown_row_on_line, row_factory):
        second_unknown = row_factory("U1", 92, 239, 92, "N", 340.0, 0.01, 2.0007, 1e-6)
        labels = [get_isomer_metadata(linear_calibrants[i])[2] for i in (0, 1)]
        result = calibrant_geometry_sweep(
            linear_calibrants, [unknown_row_on_line, second_unknown], candidate_labels=labels, p=2
        )
        assert len(result) == 2 * 2
        assert set(result["candidate_removed"]) == set(labels)

    def test_delta_ME_valid_but_delta_sys_nan_at_boundary(self, row_factory, unknown_row_on_line):
        # With exactly p remaining references, masses remain identifiable but LOO scatter does not.
        calibrants = [
            row_factory("B0", 6, 12, 6, "Y", 100.0, 0.01, 2.0, 1e-6),
            row_factory("B1", 6, 13, 6, "Y", 200.0, 0.01, 2.1, 1e-6),
            row_factory("B2", 6, 14, 6, "Y", 300.0, 0.01, 2.2, 1e-6),
            row_factory("B3", 6, 15, 6, "Y", 400.0, 0.01, 2.3, 1e-6),
        ]
        result = calibrant_geometry_sweep(calibrants, [unknown_row_on_line], candidate_labels=["B06+"], p=3)
        assert len(result) == 1
        row = result.iloc[0]
        assert not row["skipped"]
        assert row["n_calibrants_without_candidate"] == 3
        assert not np.isnan(row["delta_ME_state_keV"])
        assert np.isnan(row["delta_extra_sys_keV"])

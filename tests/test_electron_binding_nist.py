"""Offline checks against the bundled public NIST total-binding table."""

from __future__ import annotations

import pytest

from rionmass.io import GetBindingEnergyFromNIST, fetch_nist_electron_binding_table


@pytest.fixture(scope="module")
def nist_table():
    return fetch_nist_electron_binding_table()


class TestFetchNistElectronBindingTable:
    def test_covers_many_elements_offline(self, nist_table):
        # Loose lower bound so the test survives a future cache update.
        assert len(nist_table) >= 100

    def test_carbon_entries_present_and_ordered(self, nist_table):
        carbon = nist_table["6"]
        assert len(carbon) == 6  # neutral (Ion Charge 0) through C5+ (H-like)
        assert carbon[0]["Ion Charge"] == "0"


class TestGetBindingEnergyFromNIST:
    def test_bare_carbon_matches_neutral_atom_total(self, nist_table):
        value_mev = GetBindingEnergyFromNIST(nist_table, Z=6, Q=6)
        assert value_mev == pytest.approx(1030.1085e-6, rel=1e-12)

    def test_missing_element_raises(self, nist_table):
        with pytest.raises(KeyError, match="Z=999"):
            GetBindingEnergyFromNIST(nist_table, Z=999, Q=1)

    @pytest.mark.parametrize("charge", [-1, 999])
    def test_out_of_range_charge_raises(self, nist_table, charge):
        with pytest.raises(ValueError, match="0 <= Q <= Z"):
            GetBindingEnergyFromNIST(nist_table, Z=6, Q=charge)

    @pytest.mark.parametrize(
        "Z,Q,neutral_eV,residual_eV",
        [(6, 1, 1030.1085, 1018.8482), (6, 5, 1030.1085, 489.99320779), (79, 77, 517100, 184770.4)],
    )
    def test_nonbare_ions_use_neutral_minus_actual_charge(self, nist_table, Z, Q, neutral_eV, residual_eV):
        # Values from the bundled NIST total-binding cache, in eV.
        expected = (neutral_eV - residual_eV) / 1e6
        assert GetBindingEnergyFromNIST(nist_table, Z, Q) == pytest.approx(expected, rel=1e-12)

    def test_entry_order_does_not_affect_lookup(self, nist_table):
        reordered = {"6": list(reversed(nist_table["6"]))}
        assert GetBindingEnergyFromNIST(reordered, 6, 1) == pytest.approx(11.2603e-6, rel=1e-12)

    def test_value_is_positive_for_a_bare_heavy_ion(self, nist_table):
        # A bare ion uses the full neutral-atom binding energy.
        value_mev = GetBindingEnergyFromNIST(nist_table, Z=78, Q=78)
        assert value_mev > 0.0

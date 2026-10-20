"""Physical convention checks independent of optional ODS/network packages.

The Au and Pb total-binding values below are reference-data excerpts from
ElBiEn_2007.dat (09.09.2007): TotBEn and columns labelled by *bound-electron
count*. Its header cites Rodrigues et al., ADNDT 86 (2004) 117-233, with
older one-/two-electron tables. Nuclear masses and frequency coordinates
in the integration test are invented, not experimental results.

Lookup arithmetic uses rel=1e-12. Mass recovery uses abs=1e-5 keV to allow
cancellation of rest energies of order 2e5 MeV in float64 arithmetic.
"""

from __future__ import annotations

import numpy as np
import pytest

from rionmass.constants import amu, me
from rionmass.fitting import fit_calibration, predict_unknown_table
from rionmass.io import GetBindingEnergy, GetBindingEnergyFromNIST
from rionmass.processing import ame_moq_for_nuclear_state, build_nuclei_from_data


@pytest.fixture
def elbien_reference_rows():
    return [
        {"Z": 79, "TotBEn": 517015, "1e-ion": 93254, "2e-ion": 184793, "77e-ion": 516989},
        {"Z": 82, "TotBEn": 567985, "4e-ion": 250311, "78e-ion": 567893},
    ]


@pytest.mark.parametrize(
    "Z,Q,expected_MeV", [(79, 77, 0.332222), (79, 78, 0.423761), (79, 79, 0.517015), (82, 78, 0.317674)]
)
def test_elbien_nonbare_and_bare_corrections(elbien_reference_rows, Z, Q, expected_MeV):
    assert GetBindingEnergy(elbien_reference_rows, Z, Q) == pytest.approx(expected_MeV, rel=1e-12)


@pytest.mark.parametrize("lookup", [GetBindingEnergy, GetBindingEnergyFromNIST])
def test_neutral_requires_no_electronic_correction(lookup):
    assert lookup({}, 79, 0) == 0.0


@pytest.mark.parametrize("lookup", [GetBindingEnergy, GetBindingEnergyFromNIST])
@pytest.mark.parametrize("Z,Q", [(0, 0), (6, -1), (6, 7), (6, 1.5), (6.5, 1), (True, 1), (6, True)])
def test_invalid_charge_or_atomic_number_raises(lookup, Z, Q):
    with pytest.raises(ValueError):
        lookup({}, Z, Q)


def test_numpy_integer_charge_is_accepted(elbien_reference_rows):
    assert GetBindingEnergy(elbien_reference_rows, np.int64(79), np.int64(77)) == pytest.approx(0.332222)


def test_missing_residual_binding_cannot_be_replaced_by_zero(elbien_reference_rows):
    with pytest.raises(KeyError, match="3e-ion"):
        GetBindingEnergy(elbien_reference_rows, 79, 76)


def test_missing_neutral_binding_cannot_be_replaced_by_zero():
    with pytest.raises(KeyError, match="TotBEn"):
        GetBindingEnergy([{"Z": 79, "2e-ion": 184793}], 79, 77)


@pytest.mark.parametrize("residual_eV", [0, -1, 600000, float("nan"), float("inf")])
def test_invalid_elbien_residual_binding_raises(residual_eV):
    with pytest.raises(ValueError, match="binding energy"):
        GetBindingEnergy([{"Z": 79, "TotBEn": 517015, "2e-ion": residual_eV}], 79, 77)


@pytest.mark.parametrize("neutral_eV", [0, -1, float("nan"), float("inf")])
def test_invalid_neutral_binding_raises(neutral_eV):
    with pytest.raises(ValueError, match="neutral total binding"):
        GetBindingEnergy([{"Z": 79, "TotBEn": neutral_eV}], 79, 79)


@pytest.mark.parametrize(
    "entries",
    [
        [{"Ion Charge": "+1", "Binding Energy (eV)": "10"}],
        [{"Ion Charge": "0", "Binding Energy (eV)": "20"}],
    ],
)
def test_nist_requires_both_neutral_and_requested_ion(entries):
    with pytest.raises(KeyError, match="Missing NIST"):
        GetBindingEnergyFromNIST({"6": entries}, 6, 5)


def test_nist_duplicate_charge_is_ambiguous():
    entries = [{"Ion Charge": "0", "Binding Energy (eV)": "20"}] * 2
    with pytest.raises(ValueError, match="Duplicate NIST"):
        GetBindingEnergyFromNIST({"6": entries}, 6, 6)


def test_nist_nonphysical_totals_raise():
    entries = [
        {"Ion Charge": "0", "Binding Energy (eV)": "20"},
        {"Ion Charge": "+1", "Binding Energy (eV)": "30"},
    ]
    with pytest.raises(ValueError, match="residual-ion"):
        GetBindingEnergyFromNIST({"6": entries}, 6, 1)


def test_calibration_constructs_ionic_masses_and_recovers_atomic_mass(elbien_reference_rows):
    # Independent I_q values catch lookup errors that a shared round trip would hide.
    species = [
        ("197Au", 79, 197, 77, -30.0, 0.332222, "Y"),
        ("208Pb", 82, 208, 78, -20.0, 0.317674, "Y"),
        ("197Au", 79, 197, 79, -30.0, 0.517015, "Y"),
        ("198Au", 79, 198, 78, -31.0, 0.423761, "N"),
    ]
    data, ame = [], []
    for name, Z, A, Q, ME, I_q, flag in species:
        physical_moq = (A * amu + ME - Q * me + I_q) / (Q * amu)
        # Artificial observable x equals the physical m/q: exact linear curve.
        data.append([name, Z, Q, flag, physical_moq, 0.0, 0.0, 0.0, "", name, f"{name}{Q}+"])
        ame.append({"Element": name, "Z": Z, "A": A, "ME (keV)": ME * 1000, "ME_error (keV)": 1.0})
    nuclei, refs, unknowns = build_nuclei_from_data(np.array(data, dtype=object), ame, elbien_reference_rows)
    for row, datum in zip(nuclei, data, strict=True):
        assert row[13] == pytest.approx(datum[4], rel=1e-12)
        assert ame_moq_for_nuclear_state(row)[0] == pytest.approx(datum[4], rel=1e-12)
    fit = fit_calibration(refs, p=2)
    _, result, _ = predict_unknown_table(fit, unknowns)
    assert result["ME_state_keV"].iloc[0] == pytest.approx(-31000.0, rel=0, abs=1e-5)

"""Parsing, unit conversion and nuclear-state bookkeeping.

Arithmetic checks use rtol=1e-12 to tolerate floating-point rounding.
"""

from __future__ import annotations

import numpy as np
import pytest

from rionmass.constants import amu, me
from rionmass.processing import (
    ame_moq_for_nuclear_state,
    build_nuclei_from_data,
    filter_ground_states,
    get_isomer_energy,
    get_isomer_metadata,
    parse_ion_label,
    safe_float,
)


class TestSafeFloat:
    def test_plain_dot_decimal(self):
        assert safe_float("1.5") == 1.5

    def test_comma_decimal_locale(self):
        # European-locale ODS export: comma as decimal separator.
        assert safe_float("1,5") == 1.5

    def test_dot_thousands_comma_decimal(self):
        # "1.234,5" -> 1234.5 (dot=thousands, comma=decimal).
        assert safe_float("1.234,5") == 1234.5

    def test_passthrough_numeric_types(self):
        assert safe_float(3) == 3.0
        assert safe_float(3.25) == 3.25

    def test_none_raises_value_error(self):
        with pytest.raises(ValueError, match="Could not convert"):
            safe_float(None)

    def test_garbage_raises_value_error_with_location(self):
        with pytest.raises(ValueError, match="index 2, column 6"):
            safe_float("not-a-number", idx=2, column=6, array_name="Nuclei_Y")


class TestParseIonLabel:
    @pytest.mark.parametrize(
        "label,expected",
        [
            ("201mPt76+", ("201", "m", "Pt", "76")),
            ("204Pt78+", ("204", "", "Pt", "78")),
            (r"$^{201m}$Pt$^{76+}$", ("201", "m", "Pt", "76")),
            (r"$^{205}$Au$^{79+}$", ("205", "", "Au", "79")),
        ],
    )
    def test_parses_plain_and_latex_forms(self, label, expected):
        assert parse_ion_label(label) == expected

    def test_unparseable_label_raises(self):
        with pytest.raises(ValueError, match="Could not parse ion label"):
            parse_ion_label("not an ion")


class TestAmeMoqForNuclearState:
    def test_ground_state_formula(self):
        # Row: element,Z,A,N,Q,flag,x,xerr,width_x,width_x_err,ME_AME,MEerr_AME,
        #      Ebind,MoQ_AME,MoQerr_AME,isomer,display_name,species_id
        A, Q, ME, ME_err, E_bind = 204, 78, -30.0, 0.005, 0.05  # MeV, MeV, MeV
        row = ["204Pt", 78, A, A - Q, Q, "Y", 0.0, 0.0, 0.0, 0.0, ME, ME_err, E_bind, 0.0, 0.0, "", "204Pt", "204Pt78+"]
        moq, moq_err, me_state, me_state_err, e_iso, e_iso_err = ame_moq_for_nuclear_state(row)

        expected_mass_ion = A * amu + ME - Q * me + E_bind
        assert moq == pytest.approx(expected_mass_ion / amu / Q, rel=1e-12)
        assert moq_err == pytest.approx(ME_err / amu / Q, rel=1e-12)
        assert me_state == pytest.approx(ME, rel=1e-12)
        assert me_state_err == pytest.approx(ME_err, rel=1e-12)
        assert e_iso == 0.0
        assert e_iso_err == 0.0

    def test_isomer_excitation_energy_is_added(self):
        A, Q, ME, ME_err = 206, 81, -29.0, 0.003
        row = ["206Tl", 81, A, A - Q, Q, "Y", 0.0, 0.0, 0.0, 0.0, ME, ME_err, 0.0, 0.0, 0.0, "m", "206mTl", "206mTl81+"]
        e_iso_input = 2.6431  # MeV, a real isomer energy used as an example value only
        moq_gs, *_ = ame_moq_for_nuclear_state(row, isomer_excitation_energy=None)
        moq_iso, _, me_iso, _, e_iso, _ = ame_moq_for_nuclear_state(
            row, isomer_excitation_energy={"206mTl": e_iso_input}
        )
        assert e_iso == pytest.approx(e_iso_input)
        # The isomeric m/q must exceed the ground-state m/q by E_iso/(amu*Q).
        assert moq_iso - moq_gs == pytest.approx(e_iso_input / amu / Q, rel=1e-10)
        assert me_iso == pytest.approx(ME + e_iso_input, rel=1e-12)

    def test_isomer_energy_with_uncertainty_tuple(self):
        row = ["1X", 1, 1, 0, 1, "Y", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "m", "1mX", "1mX1+"]
        iso, disp, species = get_isomer_metadata(row)
        e, e_err = get_isomer_energy(row, {"1mX": (1.5, 0.2)})
        assert (e, e_err) == (1.5, 0.2)
        assert (iso, disp, species) == ("m", "1mX", "1mX1+")

    def test_unknown_isomer_key_defaults_to_ground_state(self):
        row = ["1X", 1, 1, 0, 1, "Y", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "", "1X", "1X1+"]
        e, e_err = get_isomer_energy(row, {"unrelated": 5.0})
        assert (e, e_err) == (0.0, 0.0)


class TestFilterGroundStates:
    def test_ground_only_false_returns_everything(self):
        import pandas as pd

        df = pd.DataFrame({"species_id": ["204Pt78+", "204mPt78+"], "isomer": ["", "m"]})
        out = filter_ground_states(df, ground_only=False)
        assert len(out) == 2

    def test_ground_only_true_drops_isomer_column_flag(self):
        import pandas as pd

        df = pd.DataFrame({"species_id": ["204Pt78+", "204Ir77+"], "isomer": ["", "m"]})
        out = filter_ground_states(df, ground_only=True)
        assert list(out["species_id"]) == ["204Pt78+"]

    def test_ground_only_true_drops_label_pattern_match(self):
        import pandas as pd

        # No isomer column at all -> must fall back to label-pattern detection.
        df = pd.DataFrame({"species_id": ["204Pt78+", "201mPt76+"]})
        out = filter_ground_states(df, ground_only=True)
        assert list(out["species_id"]) == ["204Pt78+"]


class TestBuildNucleiFromDataBindingEnergySource:
    """Covers the pluggable binding_energy_fn parameter (used by
    CalibrationConfig(elbien_source="nist")) alongside the default
    elbien_data path."""

    @staticmethod
    def _make_data_row(name_ame="12C", Z=6, Q=6, ref="Y", x=1.0, xerr=0.01):
        # Matches rionmass.io.build_data_from_ods's row layout: name_ame, Z,
        # charge, ref, x, x_err, width_x, width_x_err, isomer, name_full, species_id.
        return [name_ame, Z, Q, ref, x, xerr, 0.0, 0.0, "", name_ame, f"{name_ame}{Q}+"]

    @staticmethod
    def _ame_data():
        return [{"Element": "12C", "Z": 6, "A": 12, "ME (keV)": 0.0, "ME_error (keV)": 0.0}]

    def test_default_uses_elbien_data(self):
        data = np.array([self._make_data_row()], dtype=object)
        elbien_data = [{"Z": 6, "TotBEn": 1030, "6e-ion": 1030}]
        _nuclei, nuclei_y, nuclei_n = build_nuclei_from_data(data, self._ame_data(), elbien_data=elbien_data)
        assert len(nuclei_y) == 1 and len(nuclei_n) == 0
        # E_bind (index 12) should be 1030 eV -> 1030e-6 MeV.
        assert nuclei_y[0][12] == pytest.approx(1030e-6)

    def test_binding_energy_fn_overrides_elbien_data(self):
        data = np.array([self._make_data_row()], dtype=object)
        calls = []

        def fake_binding_energy(Z: int, Q: int) -> float:
            calls.append((Z, Q))
            return 0.5  # MeV, an arbitrary sentinel value

        nuclei, _y, _n = build_nuclei_from_data(data, self._ame_data(), binding_energy_fn=fake_binding_energy)
        assert calls == [(6, 6)]
        assert nuclei[0][12] == pytest.approx(0.5)

    def test_neither_elbien_data_nor_fn_raises(self):
        data = np.array([self._make_data_row()], dtype=object)
        with pytest.raises(ValueError, match="binding_energy_fn"):
            build_nuclei_from_data(data, self._ame_data())

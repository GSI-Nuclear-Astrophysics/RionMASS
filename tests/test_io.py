"""ElBiEn parsing and ODS import with synthetic files in tmp_path."""

from __future__ import annotations

import pytest

ezodf = pytest.importorskip("ezodf", reason="ODS import tests need the optional 'data' extra")

from rionmass.io import GetBindingEnergy, Read_elbien_file, build_data_from_ods  # noqa: E402


@pytest.fixture
def synthetic_elbien_file(tmp_path):
    """A tiny synthetic ElBiEn file: total binding for each bound-electron count (eV)."""
    path = tmp_path / "ElBiEn_synthetic.dat"
    path.write_text("# Z TotBEn 1e-ion 2e-ion 3e-ion\n6 1030 490 882 945\n78 130161 101401 190315 268344\n")
    return path


@pytest.fixture
def synthetic_ods_file(tmp_path):
    """A tiny synthetic ODS table: Label, mu_fit_Hz, sigma_Hz, width_Hz, width_err_Hz."""
    path = tmp_path / "synthetic_import.ods"
    doc = ezodf.newdoc(doctype="ods", filename=str(path))
    rows = [
        ("Label", "mu_fit_Hz", "sigma_Hz", "width_Hz", "width_err_Hz"),
        ("204Pt78+", 1_500_000.0, 0.5, 12.0, 0.3),
        ("205Au78+", 1_510_000.0, 0.6, 13.0, 0.4),
        ("206Tl81+", 1_520_000.0, 0.4, 11.0, 0.2),
    ]
    sheet = ezodf.Sheet("Sheet1", size=(len(rows), 5))
    doc.sheets += sheet
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            sheet[r, c].set_value(value)
    doc.save()
    return path


class TestReadElbienFile:
    def test_parses_z_and_totben(self, synthetic_elbien_file):
        data = Read_elbien_file(str(synthetic_elbien_file))
        assert {entry["Z"] for entry in data} == {6, 78}
        carbon = next(e for e in data if e["Z"] == 6)
        assert carbon["TotBEn"] == 1030
        assert carbon["1e-ion"] == 490
        assert carbon["2e-ion"] == 882
        assert carbon["3e-ion"] == 945

    def test_get_binding_energy_converts_ev_to_mev(self, synthetic_elbien_file):
        data = Read_elbien_file(str(synthetic_elbien_file))
        # C4+ retains two electrons; C5+ retains one.
        assert GetBindingEnergy(data, Z=6, Q=4) == pytest.approx((1030 - 882) / 1e6)
        assert GetBindingEnergy(data, Z=6, Q=5) == pytest.approx((1030 - 490) / 1e6)

    def test_missing_z_raises(self, synthetic_elbien_file):
        data = Read_elbien_file(str(synthetic_elbien_file))
        with pytest.raises(KeyError, match="Z=999"):
            GetBindingEnergy(data, Z=999, Q=1)

    def test_missing_file_raises(self, tmp_path):
        # Missing input files must fail explicitly.
        with pytest.raises(FileNotFoundError):
            Read_elbien_file(str(tmp_path / "does_not_exist.dat"))


class TestBuildDataFromOds:
    # Row layout (see rionmass.io.build_data_from_ods docstring): 0=name_ame,
    # 1=Z, 2=charge, 3=ref, 4=x, 5=x_err, 6=width_x, 7=width_x_err,
    # 8=isomer, 9=name_full, 10=species_id.
    _COL_REF = 3
    _COL_X = 4

    def test_frequency_observable_default(self, synthetic_ods_file):
        data, aux = build_data_from_ods(filename=str(synthetic_ods_file), sheet_index=0, verbose=False)
        assert len(data) == 3
        assert aux["x_label"] == "frequency"
        assert aux["x_unit"] == "Hz"
        assert list(data[:, self._COL_X].astype(float)) == pytest.approx([1_500_000.0, 1_510_000.0, 1_520_000.0])

    def test_time_observable_conversion(self, synthetic_ods_file):
        data, aux = build_data_from_ods(
            filename=str(synthetic_ods_file), sheet_index=0, observable="time", verbose=False
        )
        assert aux["x_label"] == "revolution_time"
        assert aux["x_unit"] == "ps"
        # T = 1e12 / f
        expected_T = 1e12 / 1_500_000.0
        assert float(data[0, self._COL_X]) == pytest.approx(expected_T, rel=1e-9)

    def test_unmeasured_list_flags_unknowns(self, synthetic_ods_file):
        data, _aux = build_data_from_ods(
            filename=str(synthetic_ods_file), sheet_index=0, unmeasured_list=["205Au78+"], verbose=False
        )
        idx = list(data[:, -1]).index("205Au78+")
        assert data[idx, self._COL_REF] == "N"
        # Every other row must remain flagged as a calibrant.
        other_idx = [i for i in range(len(data)) if i != idx]
        assert all(data[i, self._COL_REF] == "Y" for i in other_idx)

    def test_unknown_element_raises(self, tmp_path):
        path = tmp_path / "bad_element.ods"
        doc = ezodf.newdoc(doctype="ods", filename=str(path))
        sheet = ezodf.Sheet("Sheet1", size=(2, 5))
        doc.sheets += sheet
        sheet[0, 0].set_value("Label")
        sheet[1, 0].set_value("1Zz1+")  # "Zz" is not a real element symbol
        sheet[1, 1].set_value(1.0)
        sheet[1, 2].set_value(0.1)
        sheet[1, 3].set_value(1.0)
        doc.save()
        with pytest.raises(ValueError, match="Unknown element"):
            build_data_from_ods(filename=str(path), sheet_index=0, verbose=False)

    def test_harmonic_division(self, synthetic_ods_file):
        data_h1, _ = build_data_from_ods(filename=str(synthetic_ods_file), sheet_index=0, verbose=False)
        data_h2, _ = build_data_from_ods(filename=str(synthetic_ods_file), sheet_index=0, harmonic=2, verbose=False)
        assert float(data_h2[0, self._COL_X]) == pytest.approx(float(data_h1[0, self._COL_X]) / 2.0, rel=1e-12)

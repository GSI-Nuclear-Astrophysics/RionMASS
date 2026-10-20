"""Custom reference masses and changed-file replay on invented data only."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from conftest import make_row

from rionmass.api import export_result_tables, run_full_calibration_workflow
from rionmass.configuration import CalibrationConfig, ODSImportConfig
from rionmass.electronic import ElectronicInputs
from rionmass.processing import ame_moq_for_nuclear_state
from rionmass.reference_masses import (
    apply_reference_mass_overrides,
    normalize_reference_mass_overrides,
    parse_reference_mass_spec,
)
from rionmass.results import loo_reference_results
from rionmass.session import load_run_configuration, replay_run_configuration, save_run_configuration


@pytest.fixture
def reference_run(tmp_path, monkeypatch):
    ezodf = pytest.importorskip("ezodf")
    rows = [
        make_row(f"{12 + i}C", 6, 12 + i, 6, "Y", float(x), 0.01, 2 + 1e-3 * (x - 300) / 200, 1e-6)
        for i, x in enumerate(np.linspace(100, 500, 6))
    ]
    rows.append(make_row("18C", 6, 18, 6, "N", 350, 0.01, 2 + 1e-3 * 50 / 200, 1e-6))
    ame = [
        {"Element": r[0], "Z": r[1], "A": r[2], "ME (keV)": r[10] * 1000, "ME_error (keV)": r[11] * 1000} for r in rows
    ]
    monkeypatch.setattr("rionmass.api.Read_AME_barion", lambda: ame)
    path = tmp_path / "invented.ods"
    doc = ezodf.newdoc(doctype="ods", filename=str(path))
    sheet = ezodf.Sheet("Measurements", size=(len(rows) + 1, 5))
    doc.sheets += sheet
    for j, value in enumerate(("Label", "frequency", "sigma", "width", "width_sigma")):
        sheet[0, j].set_value(value)
    for i, row in enumerate(rows, 1):
        for j, value in enumerate((row[17], row[6], row[7], 1.0, 0.1)):
            sheet[i, j].set_value(value)
    doc.save()
    ods = ODSImportConfig(path, 0, unmeasured_list=("18C6+",))
    electronic = ElectronicInputs.from_corrections({(6, 6): 0.0}, covariance_MeV2=np.array([[1e-6]]))
    calibration = CalibrationConfig(electronic_inputs=electronic, p=2, compare_orders=(2,))
    return ods, calibration, rows, ame


def test_parse_and_validate_explicit_units_and_sources():
    values = parse_reference_mass_spec(["206Hg78+=-20000,2 # invented example", "# blank/comment", "72As=-60000,3"])
    assert values["206Hg"] == dict(mass_excess_keV=-20000.0, uncertainty_keV=2.0, source="invented example")
    assert values["72As"]["uncertainty_keV"] == 3
    assert (
        CalibrationConfig(
            elbien_source="nist", reference_mass_overrides={"206Hg": (-20000, 2)}
        ).reference_mass_overrides["206Hg"]["uncertainty_keV"]
        == 2
    )


@pytest.mark.parametrize(
    "values",
    [
        {"206Hg": (1, 0)},
        {"206Hg": (1, -1)},
        {"206Hg": (1, float("nan"))},
        {"206Hg": (float("inf"), 1)},
        {"206mHg": (1, 1)},
        {"1Zz": (1, 1)},
        {"206Hg": 1},
        {"206Hg": (True, 1)},
        {"206Hg": (1, 1), "206Hg78+": (2, 2)},
        {"206Hg": {"mass_excess_keV": 1, "uncertainty_keV": 1, "unexpected": 2}},
    ],
)
def test_invalid_custom_inputs_rejected(values):
    with pytest.raises(ValueError):
        CalibrationConfig(elbien_source="nist", reference_mass_overrides=values)


@pytest.mark.parametrize("text", ["206Hg=1", "206Hg=1,2,3", "206Hg=abc,1", "206Hg=1,2\n206Hg=3,4"])
def test_invalid_form_text_rejected(text):
    with pytest.raises(ValueError):
        parse_reference_mass_spec(text.splitlines())


def test_copy_overrides_updates_mass_uncertainty_isomers_and_charge_states():
    first = make_row("12C", 6, 12, 6, "Y", 100, 0.1, 2, 1e-6)
    second = make_row("12C", 6, 12, 5, "Y", 200, 0.1, 2.4, 1e-6)
    second[15:18] = ["m", "12mC", "12mC5+"]
    values = normalize_reference_mass_overrides({"12C": (10, 2, "invented")})
    updated, inputs = apply_reference_mass_overrides([first, second], values, {"12mC5+": (0.1, 0.003)})
    assert first[10] != 0.010
    assert updated[0][10:12] == [0.010, 0.002]
    assert updated[1][10:12] == [0.010, 0.002]
    assert inputs.ME_reference_state_keV.tolist() == pytest.approx([10, 110])
    assert inputs.ME_reference_state_err_keV.tolist() == pytest.approx([2, np.hypot(2, 3)])
    for row in updated:
        assert row[13:15] == pytest.approx(ame_moq_for_nuclear_state(row, {"12mC5+": (0.1, 0.003)})[:2])


def test_custom_references_drive_fit_and_loo_without_overwriting_ame(reference_run, tmp_path):
    ods, calibration, rows, ame = reference_run
    nominal = run_full_calibration_workflow(ods, calibration)
    overrides = {r[0]: (r[10] * 1000 + 2, 1, "Invented independent measurement") for r in rows[:-1]}
    custom_config = replace(calibration, reference_mass_overrides=overrides)
    result = run_full_calibration_workflow(ods, custom_config)
    assert result.unknown_df.ME_state_keV.iloc[0] - nominal.unknown_df.ME_state_keV.iloc[0] == pytest.approx(
        2, abs=1e-6
    )
    assert result.unknown_df.ME_AME_state_keV.iloc[0] == nominal.unknown_df.ME_AME_state_keV.iloc[0]
    assert result.extra_sys_keV == 0
    np.testing.assert_allclose(result.loo_df.Delta_ME_reference_keV, 0, atol=1e-6)
    np.testing.assert_allclose(result.loo_df.Delta_ME_state_keV, 2, atol=1e-6)
    assert result.loo_df.ME_reference_state_err_keV.tolist() == [1] * 6
    assert result.loo_df.loo_pull_no_extra.abs().max() < 1e-6
    # Imported AME rows and the AME reader are not modified.
    assert result.nuclei[0][10] == rows[0][10]
    assert ame[0]["ME (keV)"] == rows[0][10] * 1000
    table = loo_reference_results(result.loo_with_sys)
    assert "Pull vs reference" in table
    assert "ΔME vs AME2020 (keV)" in table
    assert "σ reference comparison (keV)" in table
    written = export_result_tables(result, str(tmp_path / "output"))
    assert any(p.endswith("reference_mass_inputs.csv") for p in written)
    saved = pd.read_csv(tmp_path / "output/reference_mass_inputs.csv")
    assert saved.reference_is_custom.all()
    wide = run_full_calibration_workflow(
        ods, replace(calibration, reference_mass_overrides={key: (v[0], 20) for key, v in overrides.items()})
    )
    assert wide.unknown_df.ME_stat_err_keV.iloc[0] > result.unknown_df.ME_stat_err_keV.iloc[0]
    path = tmp_path / "config.json"
    save_run_configuration(path, ods, custom_config, result)
    _, loaded, payload = load_run_configuration(path)
    assert loaded.reference_mass_overrides == custom_config.reference_mass_overrides
    assert payload["ame2020_inputs"][0]["ME (keV)"] == rows[0][10] * 1000
    report = replay_run_configuration(path)
    assert not report.warnings and not report.mismatches
    np.testing.assert_array_equal(report.result.fit.coeff, result.fit.coeff)


def test_unknown_or_absent_reference_override_rejected(reference_run):
    ods, calibration, rows, ame = reference_run
    for name in ("18C", "19C"):
        with pytest.raises(ValueError, match="not a retained calibration reference"):
            run_full_calibration_workflow(ods, replace(calibration, reference_mass_overrides={name: (10, 2)}))


def test_new_reference_without_ame_keeps_comparison_unavailable_and_replays(reference_run, monkeypatch, tmp_path):
    ods, calibration, rows, ame = reference_run
    monkeypatch.setattr("rionmass.api.Read_AME_barion", lambda: ame[1:])
    config = replace(calibration, reference_mass_overrides={"12C": (rows[0][10] * 1000, 2)})
    result = run_full_calibration_workflow(ods, config)
    assert np.isnan(result.loo_df.Delta_ME_state_keV.iloc[0])
    assert np.isfinite(result.loo_df.Delta_ME_reference_keV.iloc[0])
    assert np.isfinite(result.loo_df.loo_sigma_no_extra_keV.iloc[0])
    path = tmp_path / "new_reference.json"
    save_run_configuration(path, ods, config, result)
    report = replay_run_configuration(path)
    assert not report.mismatches


def test_added_ods_sheet_warns_but_preserves_results(reference_run, tmp_path):
    ezodf = pytest.importorskip("ezodf")
    ods, calibration, rows, ame = reference_run
    result = run_full_calibration_workflow(ods, calibration)
    path = tmp_path / "config.json"
    save_run_configuration(path, ods, calibration, result)
    doc = ezodf.opendoc(str(ods.filename))
    doc.sheets += ezodf.Sheet("Unrelated new sheet", size=(2, 2))
    doc.save()
    report = replay_run_configuration(path)
    assert any("SHA256 differs" in w for w in report.warnings)
    assert not report.mismatches
    np.testing.assert_array_equal(report.result.fit.coeff, result.fit.coeff)
    doc.sheets[0][2, 1].set_value(float(doc.sheets[0][2, 1].value) + 1)
    doc.save()
    changed = replay_run_configuration(path)
    assert changed.mismatches


def test_old_schema_one_without_overrides_still_replays(reference_run, tmp_path):
    ods, calibration, _, _ = reference_run
    result = run_full_calibration_workflow(ods, calibration)
    path = tmp_path / "old.json"
    save_run_configuration(path, ods, calibration, result)
    payload = json.loads(path.read_text())
    payload["schema_version"] = 1
    payload["calibration"].pop("reference_mass_overrides")
    for row in payload["verification"]["loo"]:
        for key in (
            "Delta_ME_reference_keV",
            "reference_mass_source",
            "ME_reference_state_keV",
            "ME_reference_state_err_keV",
        ):
            row.pop(key)
    path.write_text(json.dumps(payload))
    report = replay_run_configuration(path)
    assert not report.mismatches
    assert report.calibration.reference_mass_overrides == {}

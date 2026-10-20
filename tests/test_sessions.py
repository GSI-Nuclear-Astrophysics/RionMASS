"""Saved-run replay on invented measurements and reference inputs only."""

import json

import numpy as np
import pytest

from rionmass.bundles import create_run_directory
from rionmass.configuration import CalibrationConfig
from rionmass.session import (
    _sha256,
    configuration_values,
    load_run_configuration,
    replay_run_configuration,
    save_run_bundle,
    save_run_configuration,
)


def test_save_replay_freezes_external_inputs(monkeypatch, tmp_path, invented_run):
    ods, calibration, result = invented_run
    out, written = save_run_bundle(tmp_path / "outputs", "region 1", ods, calibration, result)
    assert "_RionMASS_CONFIG_region_1" in out.name
    assert all(str(out) == str(__import__("pathlib").Path(p).parent) for p in written)
    assert (out / "config.json").exists()
    assert (out / "final_mass_results.csv").exists()
    assert (out / "loo_reference_results.csv").exists()
    assert not list(out.glob("*.ods"))
    # Replay must not consult a later AME download or binding-energy table.
    monkeypatch.setattr("rionmass.api.Read_AME_barion", lambda: pytest.fail("AME must be frozen"))
    report = replay_run_configuration(out / "config.json")
    assert not report.mismatches
    assert not report.warnings
    assert report.calibration.loo_anchor_species == ("C06+",)
    assert report.calibration.electronic_workers == 2
    np.testing.assert_array_equal(report.result.fit.coeff, result.fit.coeff)
    np.testing.assert_allclose(report.result.unknown_df.ME_state_keV, result.unknown_df.ME_state_keV)
    assert report.calibration.electronic_inputs is not None
    assert calibration.electronic_inputs is not None
    np.testing.assert_allclose(
        report.calibration.electronic_inputs.covariance_MeV2,
        calibration.electronic_inputs.covariance_MeV2,
        equal_nan=True,
    )


def test_roundtrip_settings_only_and_null_covariance(tmp_path, invented_run):
    ods, calibration, _ = invented_run
    path = tmp_path / "config.json"
    save_run_configuration(path, ods, calibration)
    assert "NaN" not in path.read_text()
    loaded_ods, loaded_calibration, payload = load_run_configuration(path)
    assert configuration_values(loaded_ods, loaded_calibration) == configuration_values(ods, calibration)
    assert loaded_calibration.isomer_excitation_energy["invented_isomer"] == [0.1, 0.001]
    assert "verification" not in payload
    report = replay_run_configuration(path)
    assert not report.mismatches
    assert any("Settings-only" in s for s in report.warnings)


def test_changed_input_warns_and_replays(tmp_path, invented_run):
    ods, calibration, result = invented_run
    path = tmp_path / "config.json"
    save_run_configuration(path, ods, calibration, result)
    ods.filename.write_text("Changed invented input; selected measurements remain unchanged")
    report = replay_run_configuration(path)
    assert any("SHA256 differs" in warning for warning in report.warnings)
    assert not report.mismatches
    assert report.result.import_aux["input_sha256"] == _sha256(ods.filename)
    # Saving old results against a changed input is still disallowed. Replay
    # provides freshly recomputed results which may be saved as a new run.
    with pytest.raises(ValueError, match="ODS changed"):
        save_run_configuration(tmp_path / "wrong.json", ods, calibration, result)
    save_run_configuration(tmp_path / "recomputed.json", report.ods, report.calibration, report.result)


def test_original_input_can_be_relocated(tmp_path, invented_run):
    ods, calibration, result = invented_run
    path = tmp_path / "config.json"
    save_run_configuration(path, ods, calibration, result)
    relocated = tmp_path / "relocated.ods"
    relocated.write_bytes(ods.filename.read_bytes())
    report = replay_run_configuration(path, ods_override=relocated)
    assert not report.mismatches
    assert report.ods.filename == str(relocated)
    assert report.result.import_aux["input_sha256"] == _sha256(relocated)


def test_changed_checkpoint_detected(tmp_path, invented_run):
    ods, calibration, result = invented_run
    path = tmp_path / "config.json"
    save_run_configuration(path, ods, calibration, result)
    payload = json.loads(path.read_text())
    payload["verification"]["unknowns"][0]["ME_state_keV"] += 1
    path.write_text(json.dumps(payload))
    assert any("ME_state_keV" in m for m in replay_run_configuration(path).mismatches)


def test_schema_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema_version": 500, "software": "RionMASS"}))
    with pytest.raises(ValueError, match="Unsupported"):
        load_run_configuration(path)


def test_run_folder_names_cannot_escape_parent_and_never_overwrite(tmp_path):
    first = create_run_directory(tmp_path, "RionMASS", "../../region/label")
    second = create_run_directory(tmp_path, "RionMASS", "../../region/label")
    assert first.parent == second.parent == tmp_path
    assert second.name == first.name + "_02"
    assert create_run_directory(tmp_path, "RionMASS").name.endswith("_RionMASS_CONFIG")


def test_configured_external_binding_path_is_fingerprinted(tmp_path, invented_run):
    ods, _, _ = invented_run
    table = tmp_path / "invented_binding.dat"
    table.write_text("Invented binding input")
    path = tmp_path / "config.json"
    save_run_configuration(path, ods, CalibrationConfig(elbien_path=table))
    _, loaded, payload = load_run_configuration(path)
    assert loaded.elbien_path == str(table)
    assert payload["inputs"]["elbien_path"]["sha256"] == _sha256(table)

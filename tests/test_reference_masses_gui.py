"""Custom reference form and nonblocking replay warning; invented inputs only."""

from dataclasses import replace

import pytest

pytest.importorskip("PySide6")

from test_gui_smoke import qapp as qapp
from test_gui_smoke import wait_until
from test_reference_masses import reference_run as reference_run

from rionmass.api import run_full_calibration_workflow
from rionmass.configuration import ODSImportConfig
from rionmass.gui import main_window
from rionmass.session import save_run_configuration


def test_gui_collect_restore_edit_custom_references(qapp):
    win = main_window.MainWindow()
    win.elbien_source_combo.setCurrentText("nist")
    win.reference_masses_text.setPlainText("206Hg=-20000,2 # Invented example\n72As=-60000,3")
    ods, config = win._collect_configs()
    assert config.reference_mass_overrides["206Hg"]["uncertainty_keV"] == 2
    assert config.reference_mass_overrides["206Hg"]["source"] == "Invented example"
    win._restore_configs(ODSImportConfig("invented.ods", 0), config, {})
    _, restored = win._collect_configs()
    assert restored.reference_mass_overrides == config.reference_mass_overrides
    win.reference_masses_text.clear()
    assert win._collect_configs()[1].reference_mass_overrides == {}
    win.reference_masses_text.setPlainText("206Hg=1,-2")
    with pytest.raises(ValueError, match="positive"):
        win._collect_configs()
    win.close()


def test_gui_changed_input_replay_displays_warning_and_new_results(qapp, reference_run, tmp_path, monkeypatch):
    ods, config, rows, _ = reference_run
    config = replace(config, reference_mass_overrides={"12C": (rows[0][10] * 1000 + 2, 1, "Invented")})
    result = run_full_calibration_workflow(ods, config)
    path = tmp_path / "gui.json"
    save_run_configuration(path, ods, config, result)
    import ezodf

    doc = ezodf.opendoc(str(ods.filename))
    doc.sheets += ezodf.Sheet("Extra sheet", size=(2, 2))
    doc.save()
    win = main_window.MainWindow()
    monkeypatch.setattr(win, "_choose_config", lambda: str(path))
    warnings, errors = [], []
    monkeypatch.setattr(main_window.QMessageBox, "warning", lambda *a: warnings.append(a))
    monkeypatch.setattr(main_window.QMessageBox, "critical", lambda *a: errors.append(a))
    win._on_replay_config()
    wait_until(qapp, lambda: win._worker is None)
    assert not errors
    assert len(warnings) == 1 and "SHA256 differs" in warnings[0][-1]
    assert win._result is not None
    assert win.export_button.isEnabled()
    assert win.reference_inputs_model.rowCount() == 6
    assert "12C=" in win.reference_masses_text.toPlainText()
    assert win.loo_final_model._df["Reference source"].iloc[0] == "Custom: Invented"
    win.close()

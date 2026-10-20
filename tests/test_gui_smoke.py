"""Offscreen window construction and workflow wiring with synthetic inputs."""

from __future__ import annotations

import os
import threading

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6", reason="GUI tests need the optional 'gui' extra")


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def test_main_window_constructs(qapp):
    from rionmass.gui.main_window import MainWindow

    window = MainWindow()
    assert window.windowTitle() == "RionMASS — Mass Calibration"
    assert window.run_button is not None
    assert window.export_button.isEnabled() is False
    window.close()


def test_collect_configs_reads_form_fields(qapp):
    from rionmass.gui.main_window import MainWindow

    window = MainWindow()
    window.ods_edit.setText("dummy.ods")
    window.sheet_index_spin.setValue(3)
    window.elbien_edit.setText("dummy.dat")
    window.exclude_edit.setText("1, 2")
    window.unmeasured_edit.setText("205Au78+")
    window.p_spin.setValue(2)
    window.loo_anchors_edit.setText(" C06+ , C16+ ")
    window.loo_anchor_policy_combo.setCurrentIndex(1)

    ods_config, calibration_config = window._collect_configs()

    assert ods_config.filename == "dummy.ods"
    assert ods_config.sheet_index == 3
    assert ods_config.exclusion_list == (1, 2)
    assert ods_config.unmeasured_list == ("205Au78+",)
    assert calibration_config.p == 2
    assert calibration_config.loo_anchor_species == ("C06+", "C16+")
    assert calibration_config.loo_anchor_policy == "remove_from_validation"
    window.close()


def test_isomer_text_parsing_matches_configuration_helper(qapp):
    from rionmass.configuration import parse_isomer_energy_spec
    from rionmass.gui.main_window import MainWindow

    window = MainWindow()
    window.isomer_text.setPlainText("206mTl=2.6431\n205mAu=0.907,0.001")
    window.elbien_edit.setText("dummy.dat")
    _ods_config, calibration_config = window._collect_configs()

    assert calibration_config.isomer_excitation_energy == parse_isomer_energy_spec(
        ["206mTl=2.6431", "205mAu=0.907,0.001"]
    )
    window.close()


@pytest.fixture
def invented_calibration_result(linear_calibrants, unknown_row_on_line):
    import numpy as np
    import pandas as pd

    from rionmass.api import CalibrationResult
    from rionmass.fitting import fit_calibration, predict_unknown_table
    from rionmass.uncertainty import estimate_extra_systematic_from_loo, leave_one_out_self_calibration

    fit = fit_calibration(linear_calibrants, p=2)
    loo, _, _ = leave_one_out_self_calibration(linear_calibrants, p=2)
    sys = estimate_extra_systematic_from_loo(loo, p=2)
    _, unknowns, influences = predict_unknown_table(fit, [unknown_row_on_line], extra_sys_keV=sys["extra_sys_keV"])
    unknowns["ME_conditional_err_keV"] = unknowns["ME_total_err_keV"]
    unknowns["ME_electronic_err_keV"] = np.nan
    unknowns["ME_total_err_keV"] = np.nan
    return CalibrationResult(
        nuclei=linear_calibrants + [unknown_row_on_line],
        nuclei_y=linear_calibrants,
        nuclei_n=[unknown_row_on_line],
        import_aux={},
        fit=fit,
        loo_df=loo,
        loo_with_sys=sys["table"],
        extra_sys_keV=sys["extra_sys_keV"],
        systematic_result=sys,
        order_comparison=pd.DataFrame(),
        unknown_df=unknowns,
        unknown_influence=influences,
    )


def wait_until(qapp, condition, timeout_ms=5000):
    from PySide6.QtCore import QElapsedTimer
    from PySide6.QtTest import QTest

    timer = QElapsedTimer()
    timer.start()
    while not condition() and timer.elapsed() < timeout_ms:
        qapp.processEvents()
        QTest.qWait(5)
    assert condition(), "GUI task did not complete"


def test_background_calibration_and_export_with_unknown_electronic_error(
    qapp, monkeypatch, tmp_path, invented_calibration_result
):
    import pandas as pd

    from rionmass.gui import main_window

    window = main_window.MainWindow()
    source = tmp_path / "invented.ods"
    source.write_text("Invented input")
    binding = tmp_path / "invented.dat"
    binding.write_text("Invented binding input")
    window.ods_edit.setText(str(source))
    window.elbien_edit.setText(str(binding))
    window.output_parent_edit.setText(str(tmp_path / "outputs"))
    window.bundle_label_edit.setText("region 1")
    thread_ids = []
    release = threading.Event()
    entered = threading.Event()

    def workflow(*args):
        thread_ids.append(threading.get_ident())
        entered.set()
        assert release.wait(5)
        return invented_calibration_result

    messages = []
    monkeypatch.setattr(main_window, "run_full_calibration_workflow", workflow)
    monkeypatch.setattr(main_window.QMessageBox, "critical", lambda *args: messages.append(args))
    monkeypatch.setattr(main_window.QMessageBox, "warning", lambda *args: messages.append(args))
    monkeypatch.setattr(main_window.QMessageBox, "information", lambda *args: None)
    monkeypatch.setattr(main_window.QFileDialog, "getExistingDirectory", lambda *args: str(tmp_path))
    window._on_run()
    try:
        wait_until(qapp, entered.is_set)
        assert window._worker is not None
        assert not window.run_button.isEnabled()
        assert thread_ids[0] != threading.get_ident()
    finally:
        release.set()
        wait_until(qapp, lambda: window._worker is None)
    assert not messages
    assert window.export_button.isEnabled()
    assert window.final_model.rowCount() == 1
    assert window.loo_final_model.rowCount() == 6
    assert "LOO validation statistic" in window.diagnostics_label.text()
    assert "Done." in window.statusBar().currentMessage()
    assert "conditional uncertainties" in window.plot_vs_ame.figure.axes[0].get_legend_handles_labels()[1]
    window._on_export()
    wait_until(qapp, lambda: window._worker is None)
    assert window.export_button.isEnabled()
    outdir = window._bundle_dir
    assert outdir is not None
    assert "_RionMASS_CONFIG_region_1" in outdir.name
    table = pd.read_csv(outdir / "final_mass_results.csv")
    assert len(table) == 1
    assert table["σ combined (keV)"].isna().to_numpy().all()
    assert (outdir / "unknown_masses_polynomial.csv").exists()
    assert (outdir / "config.json").exists()
    window._on_save_config()
    wait_until(qapp, lambda: window._worker is None)
    assert window._bundle_dir == outdir
    assert len(list((tmp_path / "outputs").iterdir())) == 1
    window.close()


@pytest.mark.parametrize("failure", ["calibration", "display", "export"])
def test_failed_task_or_display_restores_buttons(qapp, monkeypatch, tmp_path, invented_calibration_result, failure):
    from rionmass.gui import main_window

    window = main_window.MainWindow()
    window.elbien_edit.setText("invented.dat")
    messages = []
    monkeypatch.setattr(main_window.QMessageBox, "critical", lambda *args: messages.append(args))
    monkeypatch.setattr(main_window.QMessageBox, "warning", lambda *args: messages.append(args))
    monkeypatch.setattr(main_window.QFileDialog, "getExistingDirectory", lambda *args: str(tmp_path))

    def fail(*args):
        raise RuntimeError("deliberate invented failure")

    if failure == "calibration":
        monkeypatch.setattr(main_window, "run_full_calibration_workflow", fail)
        window._on_run()
    elif failure == "display":
        monkeypatch.setattr(main_window, "run_full_calibration_workflow", lambda *args: invented_calibration_result)
        monkeypatch.setattr(window, "_populate", fail)
        window._on_run()
    else:
        window._result = invented_calibration_result
        window._result_configs = window._collect_configs()
        monkeypatch.setattr(window, "_save_bundle", fail)
        window._on_export()
    wait_until(qapp, lambda: window._worker is None)
    assert messages
    assert window.run_button.isEnabled()
    assert window.export_button.isEnabled() == (failure != "calibration")
    assert "running" not in window.statusBar().currentMessage().lower()
    window.close()


def test_config_replay_restores_all_fields_and_results(qapp, monkeypatch, tmp_path, invented_run):
    from rionmass.gui import main_window
    from rionmass.session import configuration_values, save_run_configuration

    ods, calibration, result = invented_run
    path = tmp_path / "config.json"
    save_run_configuration(path, ods, calibration, result)
    window = main_window.MainWindow()
    monkeypatch.setattr(main_window.QFileDialog, "getOpenFileName", lambda *a: (str(path), "JSON"))
    messages = []
    monkeypatch.setattr(main_window.QMessageBox, "critical", lambda *a: messages.append(a))
    monkeypatch.setattr(main_window.QMessageBox, "warning", lambda *a: messages.append(a))
    window._on_replay_config()
    wait_until(qapp, lambda: window._worker is None)
    assert not messages
    assert "Replay verified" in window.statusBar().currentMessage()
    assert window._result is not None
    assert window.final_model.rowCount() == 1
    assert window.loo_anchors_edit.text() == "C06+"
    collected_ods, collected_cal = window._collect_configs()
    assert collected_cal.electronic_workers == 2
    assert collected_cal.electronic_method == "refit"
    assert collected_cal.electronic_inputs is not None
    assert configuration_values(collected_ods, collected_cal)["ods"] == configuration_values(ods, calibration)["ods"]
    window.close()


def test_loading_config_does_not_run_analysis(qapp, monkeypatch, tmp_path, invented_run):
    from rionmass.gui import main_window
    from rionmass.session import save_run_configuration

    ods, calibration, _ = invented_run
    path = tmp_path / "config.json"
    save_run_configuration(path, ods, calibration)
    window = main_window.MainWindow()
    monkeypatch.setattr(main_window.QFileDialog, "getOpenFileName", lambda *a: (str(path), "JSON"))
    monkeypatch.setattr(main_window, "run_full_calibration_workflow", lambda *a, **kw: pytest.fail("No calculation"))
    window._on_load_config()
    assert window._result is None
    assert window._worker is None
    assert window.p_spin.value() == 2
    assert window.loo_anchors_edit.text() == "C06+"
    window.close()


def test_edited_settings_do_not_attach_previous_result_tables(qapp, monkeypatch, tmp_path, invented_run):
    import json

    from rionmass.gui import main_window

    ods, calibration, result = invented_run
    window = main_window.MainWindow()
    window._restore_configs(ods, calibration, {})
    window._pending_configs = (ods, calibration)
    window._calibration_ready(result)
    window.output_parent_edit.setText(str(tmp_path / "outputs"))
    window.bundle_label_edit.setText("setting")
    monkeypatch.setattr(main_window.QMessageBox, "information", lambda *a: None)
    window.p_spin.setValue(3)
    window._on_save_config()
    wait_until(qapp, lambda: window._worker is None)
    settings_folder = window._bundle_dir
    assert settings_folder is not None
    assert json.loads((settings_folder / "config.json").read_text())["calibration"]["p"] == 3
    assert not (settings_folder / "final_mass_results.csv").exists()
    window._on_export()
    wait_until(qapp, lambda: window._worker is None)
    result_folder = window._bundle_dir
    assert result_folder is not None and result_folder != settings_folder
    assert json.loads((result_folder / "config.json").read_text())["calibration"]["p"] == 2
    assert (result_folder / "final_mass_results.csv").exists()
    window.close()

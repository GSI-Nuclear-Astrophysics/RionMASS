"""Calibration form and results view using the shared workflow API."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields
from pathlib import Path
from typing import Any

import pandas as pd
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableView,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..api import CalibrationResult, run_full_calibration_workflow
from ..configuration import CalibrationConfig, ODSImportConfig, parse_isomer_energy_spec
from ..plotting import (
    plot_calibration_fit,
    plot_loo_residuals,
    plot_unknown_mass_excess_residuals,
    plot_unknown_mass_excess_vs_ame,
)
from ..reference_masses import parse_reference_mass_spec
from ..results import final_mass_results, loo_reference_results
from ..session import (
    ReplayReport,
    _verification,
    configuration_values,
    load_run_configuration,
    replay_run_configuration,
    save_run_bundle,
)
from .plot_canvas import PlotCanvas
from .table_model import DataFrameTableModel
from .task import TaskThread

# Column order matches models.IDX_* exactly — see rionmass.models module docstring.
_NUCLEI_COLUMNS = [
    "element",
    "Z",
    "A",
    "N",
    "Q",
    "flag",
    "x",
    "x_err",
    "width_x",
    "width_x_err",
    "ME_AME_MeV",
    "ME_err_AME_MeV",
    "E_bind_MeV",
    "MoQ_AME",
    "MoQ_err_AME",
    "isomer",
    "display_name",
    "species_id",
]


def _nuclei_to_dataframe(rows: list[list[Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=pd.Index(_NUCLEI_COLUMNS))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("RionMASS — Mass Calibration")
        self.resize(1400, 900)
        self._result: CalibrationResult | None = None
        self._worker: TaskThread | None = None
        self._task_name = ""
        self._loaded_ods: ODSImportConfig | None = None
        self._loaded_calibration: CalibrationConfig | None = None
        self._loaded_ame_data: list[dict[str, Any]] | None = None
        self._result_configs: tuple[ODSImportConfig, CalibrationConfig] | None = None
        self._pending_configs: tuple[ODSImportConfig, CalibrationConfig] | None = None
        self._bundle_key: str | None = None
        self._bundle_dir: Path | None = None
        self._bundle_files: list[str] = []

        splitter = QSplitter(self)
        splitter.addWidget(self._build_form_panel())
        splitter.addWidget(self._build_results_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)
        self.statusBar().showMessage("Fill in the import settings, then click Run.")

    def _build_form_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(420)
        scroll.setMaximumWidth(520)

        panel = QWidget()
        layout = QVBoxLayout(panel)

        session_group = QGroupBox("Saved runs: configuration and tables")
        session_form = QFormLayout(session_group)
        self.output_parent_edit = self._path_row(session_form, "Output parent folder", is_save=False, directory=True)
        self.bundle_label_edit = QLineEdit()
        self.bundle_label_edit.setPlaceholderText("Optional label, e.g. region or setting")
        session_form.addRow("Folder label", self.bundle_label_edit)
        self.save_config_button = QPushButton("Save config + tables")
        self.load_config_button = QPushButton("Load config…")
        self.replay_config_button = QPushButton("Replay config…")
        self.save_config_button.clicked.connect(self._on_save_config)
        self.load_config_button.clicked.connect(self._on_load_config)
        self.replay_config_button.clicked.connect(self._on_replay_config)
        session_buttons = QHBoxLayout()
        for button in (self.save_config_button, self.load_config_button, self.replay_config_button):
            session_buttons.addWidget(button)
        session_form.addRow(session_buttons)
        self.bundle_path_label = QLabel("DATE_RionMASS_CONFIG[_label]")
        self.bundle_path_label.setWordWrap(True)
        session_form.addRow(self.bundle_path_label)
        layout.addWidget(session_group)

        import_group = QGroupBox("1. Import (ODS + electron binding energies)")
        import_form = QFormLayout(import_group)
        self.ods_edit = self._path_row(import_form, "ODS file", is_save=False)
        self.sheet_index_spin = QSpinBox()
        self.sheet_index_spin.setRange(0, 9999)
        import_form.addRow("Sheet index", self.sheet_index_spin)
        self.elbien_source_combo = QComboBox()
        self.elbien_source_combo.addItems(["file", "nist"])
        self.elbien_source_combo.setToolTip(
            "'file': an ElBiEn_*.dat table (below). 'nist': scraped/cached NIST data, "
            "no file needed; cross-check this source before using it for a real analysis."
        )
        import_form.addRow("Binding energy source", self.elbien_source_combo)
        self.elbien_edit = self._path_row(import_form, "ElBiEn file (if source=file)", is_save=False)
        self.electronic_inputs_edit = self._path_row(import_form, "Electronic inputs JSON (optional)", is_save=False)
        self.frozen_inputs_label = QLabel()
        self.frozen_inputs_label.setWordWrap(True)
        import_form.addRow(self.frozen_inputs_label)
        self.electronic_independent_check = QCheckBox("Assume independent primitive table errors")
        self.electronic_independent_check.setToolTip(
            "Explicit covariance approximation. Missing uncertainties remain unknown. "
            "For explicit JSON inputs, specify the covariance there."
        )
        import_form.addRow(self.electronic_independent_check)
        self.exclude_edit = QLineEdit()
        self.exclude_edit.setPlaceholderText("e.g. 3, 7, 12")
        import_form.addRow("Exclude rows", self.exclude_edit)
        self.unmeasured_edit = QLineEdit()
        self.unmeasured_edit.setPlaceholderText("e.g. 205Au78+, 203Pt77+")
        import_form.addRow("Unmeasured (unknown) ions", self.unmeasured_edit)
        self.observable_combo = QComboBox()
        self.observable_combo.addItems(["frequency", "time"])
        import_form.addRow("Observable", self.observable_combo)
        self.harmonic_edit = QLineEdit()
        self.harmonic_edit.setPlaceholderText("blank = already fundamental frequency")
        import_form.addRow("Harmonic", self.harmonic_edit)
        layout.addWidget(import_group)

        fit_group = QGroupBox("2. Calibration fit")
        fit_form = QFormLayout(fit_group)
        self.p_spin = QSpinBox()
        self.p_spin.setRange(1, 12)
        self.p_spin.setValue(3)
        fit_form.addRow("Polynomial terms p", self.p_spin)
        self.itermax_spin = QSpinBox()
        self.itermax_spin.setRange(1, 100000)
        self.itermax_spin.setValue(100)
        fit_form.addRow("Max iterations", self.itermax_spin)
        self.tol_edit = QLineEdit("1e-14")
        fit_form.addRow("Tolerance", self.tol_edit)
        self.isomer_text = QPlainTextEdit()
        self.isomer_text.setPlaceholderText("One per line: 206mTl=2.6431 or 205mAu=0.907,0.001")
        self.isomer_text.setFixedHeight(80)
        fit_form.addRow("Isomer energies (MeV)", self.isomer_text)
        self.reference_masses_text = QPlainTextEdit()
        self.reference_masses_text.setPlaceholderText("NUCLIDE=ME_keV,sigma_keV # optional source")
        self.reference_masses_text.setFixedHeight(100)
        self.reference_masses_text.setToolTip(
            "Neutral-atom ground-state mass excess and positive 1-sigma uncertainty in keV. "
            "One retained reference nuclide per line, e.g. 206Hg=VALUE,ERROR # source. "
            "Applies across charge states; isomer excitation energies are added separately. "
            "Leave blank to use AME2020. Unknown ions are not promoted to references."
        )
        fit_form.addRow("Custom masses (keV)", self.reference_masses_text)
        self.compare_orders_edit = QLineEdit("2, 3, 4, 5")
        fit_form.addRow("Compare orders", self.compare_orders_edit)
        self.loo_anchors_edit = QLineEdit()
        self.loo_anchors_edit.setPlaceholderText("Comma-separated reference species ids; blank = ordinary LOO")
        self.loo_anchors_edit.setToolTip(
            "Keep selected references in the final calibration with their input uncertainties. "
            "Exclude their own holdout residuals from the LOO scatter estimate. "
            "Select anchors from calibration geometry and independent reference knowledge."
        )
        fit_form.addRow("LOO anchors", self.loo_anchors_edit)
        self.loo_anchor_policy_combo = QComboBox()
        self.loo_anchor_policy_combo.addItem("Keep anchors in other LOO training sets", "retain_in_training")
        self.loo_anchor_policy_combo.addItem("Remove anchors from other LOO training sets", "remove_from_validation")
        fit_form.addRow("Anchor validation policy", self.loo_anchor_policy_combo)
        layout.addWidget(fit_group)

        self.run_button = QPushButton("Run calibration")
        self.run_button.clicked.connect(self._on_run)
        layout.addWidget(self.run_button)

        export_group = QGroupBox("3. Export")
        export_layout = QHBoxLayout(export_group)
        self.export_button = QPushButton("Save results bundle (config + CSV)")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._on_export)
        export_layout.addWidget(self.export_button)
        layout.addWidget(export_group)

        layout.addStretch(1)
        scroll.setWidget(panel)
        return scroll

    def _path_row(self, form: QFormLayout, label: str, is_save: bool, directory: bool = False) -> QLineEdit:
        edit = QLineEdit()
        browse = QPushButton("Browse…")

        def _browse() -> None:
            if directory:
                path = QFileDialog.getExistingDirectory(self, label)
            else:
                path, _ = (QFileDialog.getSaveFileName if is_save else QFileDialog.getOpenFileName)(self, label)
            if path:
                edit.setText(path)

        browse.clicked.connect(_browse)
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(edit)
        row_layout.addWidget(browse)
        form.addRow(label, row)
        return edit

    def _build_results_panel(self) -> QTabWidget:
        self.tabs = QTabWidget()

        final_panel = QWidget()
        final_layout = QVBoxLayout(final_panel)
        self.diagnostics_label = QLabel("Calibration and LOO validation diagnostics will appear here.")
        self.diagnostics_label.setWordWrap(True)
        final_layout.addWidget(self.diagnostics_label)
        final_layout.addWidget(
            QLabel(
                "All uncertainties are in keV. Fit combines calibration and unknown-observable errors.\n"
                "Conditional combines fit and LOO extra scatter; combined also includes supplied electronic inputs."
            )
        )
        self.final_model = DataFrameTableModel(decimal_places=3)
        self.final_view = self._table_view(self.final_model)
        self.loo_final_model = DataFrameTableModel(decimal_places=3)
        self.loo_final_view = self._table_view(self.loo_final_model)
        final_tables = QTabWidget()
        final_tables.addTab(self.final_view, "Unknown masses")
        final_tables.addTab(self.loo_final_view, "LOO reference predictions")
        final_layout.addWidget(final_tables)
        comparison_note = QLabel(
            "ΔME = inferred state ME - AME2020 state ME. "
            "Isomer reference errors include the supplied excitation error.\n"
            "LOO prediction errors exclude the held-out reference error; comparison errors include it. "
            "With custom masses, LOO pulls/scatter use the supplied reference, and AME differences remain separate. "
            "Anchor holdouts are diagnostic only and do not set extra scatter."
        )
        comparison_note.setWordWrap(True)
        final_layout.addWidget(comparison_note)
        self.final_tab_index = self.tabs.addTab(final_panel, "Final results")

        self.reference_inputs_model = DataFrameTableModel(decimal_places=6)
        self.tabs.addTab(self._table_view(self.reference_inputs_model), "Reference mass inputs")
        self.nuclei_model = DataFrameTableModel()
        self.tabs.addTab(self._table_view(self.nuclei_model), "Nuclei")

        self.param_model = DataFrameTableModel()
        self.cov_model = DataFrameTableModel()
        self.corr_model = DataFrameTableModel()
        self.calibrant_model = DataFrameTableModel()
        self.flow_model = DataFrameTableModel()
        self.tabs.addTab(
            self._sub_tabs(
                [
                    ("Parameters", self.param_model),
                    ("Covariance", self.cov_model),
                    ("Correlation", self.corr_model),
                    ("Calibrants", self.calibrant_model),
                    ("Flow of information", self.flow_model),
                ]
            ),
            "Calibration",
        )

        self.loo_model = DataFrameTableModel()
        self.tabs.addTab(self._table_view(self.loo_model), "Leave-one-out")

        self.order_model = DataFrameTableModel()
        self.tabs.addTab(self._table_view(self.order_model), "Order comparison")

        self.electronic_input_model = DataFrameTableModel()
        self.electronic_correction_model = DataFrameTableModel()
        self.electronic_mass_cov_model = DataFrameTableModel()
        self.electronic_fit_cov_model = DataFrameTableModel()
        self.total_mass_cov_model = DataFrameTableModel()
        self.tabs.addTab(
            self._sub_tabs(
                [
                    ("Primitive inputs", self.electronic_input_model),
                    ("Ion corrections", self.electronic_correction_model),
                    ("Electronic mass covariance", self.electronic_mass_cov_model),
                    ("Electronic coefficient covariance", self.electronic_fit_cov_model),
                    ("Combined mass covariance", self.total_mass_cov_model),
                ]
            ),
            "Electronic uncertainty",
        )

        self.unknown_model = DataFrameTableModel()
        self.tabs.addTab(self._table_view(self.unknown_model), "Unknown details")

        self.plot_fit = PlotCanvas()
        self.plot_loo = PlotCanvas()
        self.plot_vs_ame = PlotCanvas()
        self.plot_residuals = PlotCanvas()
        plot_tabs = QTabWidget()
        plot_tabs.addTab(self.plot_fit, "Calibration fit")
        plot_tabs.addTab(self.plot_loo, "LOO residuals")
        plot_tabs.addTab(self.plot_vs_ame, "Unknown vs AME")
        plot_tabs.addTab(self.plot_residuals, "Unknown residuals")
        self.tabs.addTab(plot_tabs, "Plots")

        return self.tabs

    @staticmethod
    def _table_view(model: DataFrameTableModel) -> QTableView:
        view = QTableView()
        view.setModel(model)
        view.setAlternatingRowColors(True)
        model.modelReset.connect(view.resizeColumnsToContents)
        view.horizontalHeader().setStretchLastSection(True)
        return view

    def _sub_tabs(self, items: list[tuple[str, DataFrameTableModel]]) -> QTabWidget:
        tabs = QTabWidget()
        for name, model in items:
            tabs.addTab(self._table_view(model), name)
        return tabs

    def _collect_configs(self) -> tuple[ODSImportConfig, CalibrationConfig]:
        exclude = [int(v) for v in self.exclude_edit.text().split(",") if v.strip()]
        unmeasured = [v.strip() for v in self.unmeasured_edit.text().split(",") if v.strip()]
        harmonic_text = self.harmonic_edit.text().strip()
        harmonic = float(harmonic_text) if harmonic_text else None

        hidden_ods = (
            {
                f.name: getattr(self._loaded_ods, f.name)
                for f in fields(self._loaded_ods)
                if f.name in ("header", "revname")
            }
            if self._loaded_ods is not None
            else {}
        )
        ods_config = ODSImportConfig(
            **hidden_ods,
            filename=self.ods_edit.text(),
            sheet_index=self.sheet_index_spin.value(),
            exclusion_list=tuple(exclude),
            unmeasured_list=tuple(unmeasured),
            observable=self.observable_combo.currentText(),
            harmonic=harmonic,
        )

        isomer_lines = [line for line in self.isomer_text.toPlainText().splitlines() if line.strip()]
        isomer_energies = parse_isomer_energy_spec(isomer_lines)
        compare_orders = tuple(int(v) for v in self.compare_orders_edit.text().split(",") if v.strip())

        elbien_source = self.elbien_source_combo.currentText()
        visible_fields = {
            "elbien_source",
            "elbien_path",
            "isomer_excitation_energy",
            "reference_mass_overrides",
            "p",
            "iterationMax",
            "tol",
            "compare_orders",
            "loo_anchor_species",
            "loo_anchor_policy",
            "electronic_inputs_path",
            "electronic_assume_independent",
        }
        hidden_calibration = (
            {
                f.name: getattr(self._loaded_calibration, f.name)
                for f in fields(self._loaded_calibration)
                if f.name not in visible_fields
            }
            if self._loaded_calibration is not None
            else {}
        )
        if self.electronic_inputs_edit.text().strip():
            hidden_calibration["electronic_inputs"] = None
        elif self._loaded_calibration is not None and (
            elbien_source != self._loaded_calibration.elbien_source
            or self.electronic_independent_check.isChecked() != self._loaded_calibration.electronic_assume_independent
            or (self.elbien_edit.text() or None)
            != (str(self._loaded_calibration.elbien_path) if self._loaded_calibration.elbien_path else None)
        ):
            hidden_calibration["electronic_inputs"] = None
        calibration_config = CalibrationConfig(
            **hidden_calibration,
            elbien_source=elbien_source,
            elbien_path=(self.elbien_edit.text().strip() or None) if elbien_source == "file" else None,
            isomer_excitation_energy=isomer_energies,
            reference_mass_overrides=parse_reference_mass_spec(self.reference_masses_text.toPlainText().splitlines()),
            p=self.p_spin.value(),
            iterationMax=self.itermax_spin.value(),
            tol=float(self.tol_edit.text()),
            compare_orders=compare_orders,
            loo_anchor_species=tuple(v.strip() for v in self.loo_anchors_edit.text().split(",") if v.strip()),
            loo_anchor_policy=self.loo_anchor_policy_combo.currentData(),
            electronic_inputs_path=self.electronic_inputs_edit.text().strip() or None,
            electronic_assume_independent=self.electronic_independent_check.isChecked(),
        )
        return ods_config, calibration_config

    def _on_run(self) -> None:
        if self._worker is not None:
            return
        try:
            ods_config, calibration_config = self._collect_configs()
        except Exception as exc:
            QMessageBox.critical(self, "Invalid settings", str(exc))
            return

        self._pending_configs = (ods_config, calibration_config)
        frozen_ame = (
            self._loaded_ame_data
            if (
                self._loaded_ods is not None
                and ods_config.filename == self._loaded_ods.filename
                and ods_config.sheet_index == self._loaded_ods.sheet_index
            )
            else None
        )
        self._start_task(
            "Calibration",
            lambda: run_full_calibration_workflow(
                ods_config, calibration_config, **({"ame_data": frozen_ame} if frozen_ame is not None else {})
            ),
            self._calibration_ready,
        )

    def _start_task(self, name: str, operation: Callable[[], object], ready: Callable) -> None:
        self._task_name = name
        self.statusBar().showMessage(f"{name} running…")
        self.run_button.setEnabled(False)
        self.export_button.setEnabled(False)
        for button in (self.save_config_button, self.load_config_button, self.replay_config_button):
            button.setEnabled(False)
        self._worker = TaskThread(operation, self)
        self._worker.succeeded.connect(ready)
        self._worker.failed.connect(self._task_failed)
        self._worker.finished.connect(self._task_finished)
        self._worker.start()

    def _task_finished(self) -> None:
        worker, self._worker = self._worker, None
        self.run_button.setEnabled(True)
        self.export_button.setEnabled(self._result is not None)
        for button in (self.save_config_button, self.load_config_button, self.replay_config_button):
            button.setEnabled(True)
        if worker is not None:
            worker.deleteLater()

    def _task_failed(self, message: str) -> None:
        self.statusBar().showMessage(f"{self._task_name} failed.")
        QMessageBox.critical(self, f"{self._task_name} failed", message)

    def _calibration_ready(self, result: CalibrationResult) -> None:
        self._result = result
        self._result_configs = self._pending_configs
        try:
            self._populate(result)
        except Exception as exc:
            self.statusBar().showMessage("Calibration complete; a result view could not be drawn. Export is available.")
            QMessageBox.warning(self, "Display failed", f"Results are retained and can be exported.\n{exc}")
            return
        self.tabs.setCurrentIndex(self.final_tab_index)
        self.statusBar().showMessage(
            f"Done. chi2/dof = {result.fit.chi2:.4g}/{result.fit.dof} = {result.fit.red_chi2:.4g}; "
            f"LOO extra scatter = {result.extra_sys_keV:.4f} keV."
            + (
                " Electronic covariance incomplete; total errors may be unknown."
                if not result.electronic_uncertainty_complete
                else ""
            )
        )

    def _populate(self, result: CalibrationResult) -> None:
        self.final_model.set_dataframe(final_mass_results(result.unknown_df))
        self.loo_final_model.set_dataframe(loo_reference_results(result.loo_with_sys))
        diagnostic = result.systematic_result
        anchors = diagnostic.get("loo_anchor_species", ())
        anchor_note = (
            f"\nAnchors retained in final calibration: {', '.join(anchors)}. "
            f"LOO policy: {diagnostic.get('loo_anchor_policy')}; "
            f"{diagnostic['N']} residuals used, {diagnostic.get('n_excluded', 0)} anchor holdouts excluded."
            if anchors
            else ""
        )
        self.diagnostics_label.setText(
            f"Calibration χ²/dof: {result.fit.chi2:.5g}/{result.fit.dof} = {result.fit.red_chi2:.5g}\n"
            f"LOO validation statistic/dof: {diagnostic['chi2_raw']:.5g}/{diagnostic['dof']} "
            f"= {diagnostic['red_chi2_raw']:.5g}; after extra scatter: {diagnostic['red_chi2_with_sys']:.5g}.\n"
            "The extra-scatter term is set by held-out predictions, not the full-fit χ². "
            "LOO uses marginal pulls; correlated folds limit a formal χ² interpretation." + anchor_note
        )
        self.reference_inputs_model.set_dataframe(result.reference_inputs)
        self.nuclei_model.set_dataframe(_nuclei_to_dataframe(result.nuclei))
        self.param_model.set_dataframe(result.fit.parameter_table())
        self.cov_model.set_dataframe(result.fit.covariance_table())
        self.corr_model.set_dataframe(result.fit.correlation_table())
        self.calibrant_model.set_dataframe(result.fit.calibrant_table())
        self.flow_model.set_dataframe(result.fit.flow_table())
        self.loo_model.set_dataframe(result.loo_with_sys)
        self.order_model.set_dataframe(result.order_comparison)
        self.unknown_model.set_dataframe(result.unknown_df)
        if result.electronic is not None:
            electronic = result.electronic
            self.electronic_input_model.set_dataframe(electronic.inputs.primitive_table())
            self.electronic_correction_model.set_dataframe(electronic.inputs.correction_table())
            self.electronic_mass_cov_model.set_dataframe(
                electronic.mass_covariance_keV2.rename_axis("ion").reset_index()
            )
            self.electronic_fit_cov_model.set_dataframe(
                electronic.coefficient_covariance.rename_axis("parameter").reset_index()
            )
            self.total_mass_cov_model.set_dataframe(
                electronic.total_mass_covariance_keV2.rename_axis("ion").reset_index()
            )

        self.plot_fit.draw_with(plot_calibration_fit, result.fit, result.nuclei_n)
        self.plot_loo.draw_with(plot_loo_residuals, result.loo_with_sys, with_extra=True)
        self.plot_vs_ame.draw_with(plot_unknown_mass_excess_vs_ame, result.unknown_df, ground_only=True)
        self.plot_residuals.draw_with(plot_unknown_mass_excess_residuals, result.unknown_df, ground_only=True)

    def _output_parent(self) -> str | None:
        parent = self.output_parent_edit.text().strip()
        if not parent:
            parent = QFileDialog.getExistingDirectory(self, "Choose output parent folder")
            if parent:
                self.output_parent_edit.setText(parent)
        return parent or None

    def _save_bundle(
        self,
        parent: str,
        label: str,
        ods: ODSImportConfig,
        calibration: CalibrationConfig,
        result: CalibrationResult | None,
    ) -> list[str]:
        import json

        from ..session import _sha256

        key = json.dumps(
            {
                "parent": str(Path(parent).expanduser().resolve()),
                "label": label,
                "settings": configuration_values(ods, calibration),
                "input_hash": _sha256(Path(ods.filename).expanduser()),
                "results": _verification(result) if result is not None else None,
            },
            sort_keys=True,
        )
        if (
            key != self._bundle_key
            or self._bundle_dir is None
            or not self._bundle_dir.exists()
            or not all(Path(p).is_file() for p in self._bundle_files)
        ):
            self._bundle_dir, self._bundle_files = save_run_bundle(parent, label, ods, calibration, result)
            self._bundle_key = key
        return self._bundle_files

    def _on_save_config(self) -> None:
        if self._worker is not None:
            return
        try:
            ods, calibration = self._collect_configs()
        except Exception as exc:
            QMessageBox.critical(self, "Invalid settings", str(exc))
            return
        parent = self._output_parent()
        if not parent:
            return
        result = None
        if self._result_configs is not None and self._result is not None:
            previous = configuration_values(*self._result_configs)
            if previous == configuration_values(ods, calibration):
                result = self._result
        label = self.bundle_label_edit.text().strip()
        self._start_task("Save", lambda: self._save_bundle(parent, label, ods, calibration, result), self._export_ready)

    def _on_export(self) -> None:
        if self._result is None or self._worker is not None or self._result_configs is None:
            return
        parent = self._output_parent()
        if not parent:
            return
        ods, calibration = self._result_configs
        result = self._result
        label = self.bundle_label_edit.text().strip()
        self._start_task(
            "Export", lambda: self._save_bundle(parent, label, ods, calibration, result), self._export_ready
        )

    def _export_ready(self, written: list[str]) -> None:
        outdir = str(Path(written[0]).parent) if written else ""
        self.bundle_path_label.setText(outdir)
        self.statusBar().showMessage(f"Saved {len(written)} file(s) to {outdir}")
        QMessageBox.information(self, "Run saved", f"Configuration and available tables saved to:\n{outdir}")

    def _choose_config(self) -> str:
        path, _ = QFileDialog.getOpenFileName(self, "Open RionMASS configuration", "", "JSON (*.json)")
        return path

    def _restore_configs(self, ods: ODSImportConfig, calibration: CalibrationConfig, payload: dict[str, Any]) -> None:
        self._loaded_ods, self._loaded_calibration = ods, calibration
        self._loaded_ame_data = payload.get("ame2020_inputs")
        self.frozen_inputs_label.setText(
            "Using electronic inputs frozen in the saved run. Change the binding source/file or input JSON to replace."
            if calibration.electronic_inputs is not None
            else ""
        )
        self.ods_edit.setText(str(ods.filename))
        self.sheet_index_spin.setValue(ods.sheet_index)
        self.exclude_edit.setText(", ".join(map(str, ods.exclusion_list)))
        self.unmeasured_edit.setText(", ".join(ods.unmeasured_list))
        self.observable_combo.setCurrentText(ods.observable)
        self.harmonic_edit.setText(str(ods.harmonic) if ods.harmonic is not None else "")
        self.elbien_source_combo.setCurrentText(calibration.elbien_source)
        self.elbien_edit.setText(str(calibration.elbien_path) if calibration.elbien_path else "")
        self.electronic_inputs_edit.setText(
            str(calibration.electronic_inputs_path) if calibration.electronic_inputs_path else ""
        )
        self.electronic_independent_check.setChecked(calibration.electronic_assume_independent)
        self.p_spin.setMaximum(max(self.p_spin.maximum(), calibration.p))
        self.p_spin.setValue(calibration.p)
        self.itermax_spin.setMaximum(max(self.itermax_spin.maximum(), calibration.iterationMax))
        self.itermax_spin.setValue(calibration.iterationMax)
        self.tol_edit.setText(str(calibration.tol))
        self.compare_orders_edit.setText(", ".join(map(str, calibration.compare_orders)))
        self.loo_anchors_edit.setText(", ".join(calibration.loo_anchor_species))
        self.loo_anchor_policy_combo.setCurrentIndex(
            self.loo_anchor_policy_combo.findData(calibration.loo_anchor_policy)
        )
        self.reference_masses_text.setPlainText(
            "\n".join(
                f"{key}={value['mass_excess_keV']:.17g},{value['uncertainty_keV']:.17g} # {value['source']}"
                for key, value in calibration.reference_mass_overrides.items()
            )
        )
        self.isomer_text.setPlainText(
            "\n".join(
                f"{key}=" + (",".join(map(str, value)) if isinstance(value, (tuple, list)) else str(value))
                for key, value in calibration.isomer_excitation_energy.items()
            )
        )

    def _on_load_config(self) -> None:
        if self._worker is not None:
            return
        path = self._choose_config()
        if not path:
            return
        try:
            ods, calibration, payload = load_run_configuration(path)
            self._restore_configs(ods, calibration, payload)
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            return
        self.statusBar().showMessage("Configuration loaded. Run calibration or replay to recompute results.")

    def _on_replay_config(self) -> None:
        if self._worker is not None:
            return
        path = self._choose_config()
        if not path:
            return
        try:
            ods, _, _ = load_run_configuration(path)
        except Exception as exc:
            QMessageBox.critical(self, "Replay failed", str(exc))
            return
        override = None
        if not Path(ods.filename).is_file():
            override, _ = QFileDialog.getOpenFileName(self, "Locate original ODS input", "", "ODS (*.ods)")
            if not override:
                return
        self._start_task("Replay", lambda: replay_run_configuration(path, ods_override=override), self._replay_ready)

    def _replay_ready(self, report: ReplayReport) -> None:
        self._restore_configs(report.ods, report.calibration, report.payload)
        self._pending_configs = (report.ods, report.calibration)
        self._calibration_ready(report.result)
        if report.mismatches:
            self.statusBar().showMessage("Replay completed with differences; see verification message.")
            QMessageBox.warning(self, "Replay differences", "\n".join(report.warnings + report.mismatches))
        elif report.warnings:
            self.statusBar().showMessage("Replay completed; see verification message.")
            QMessageBox.warning(self, "Replay warnings", "\n".join(report.warnings))
        else:
            self.statusBar().showMessage("Replay verified against saved numerical checkpoints.")

    def closeEvent(self, event) -> None:
        if self._worker is not None:
            self.statusBar().showMessage(f"Wait for {self._task_name.lower()} to finish before closing.")
            event.ignore()
            return
        super().closeEvent(event)

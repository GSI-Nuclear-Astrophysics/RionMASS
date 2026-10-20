# Public API

RionMASS 1.0 uses semantic versioning. Public input, unit and result changes
are documented in [CHANGELOG.md](../CHANGELOG.md).

- `api`: `run_full_calibration_workflow`, `CalibrationResult`, `export_result_tables`.
- `configuration`: `ODSImportConfig`, `CalibrationConfig`, `parse_isomer_energy_spec`.
- `fitting`: `fit_calibration`, `predict_unknown_table`, `MassCalibration`.
- `uncertainty`: LOO validation, extra-scatter estimation, order comparisons
  and reference-deletion diagnostics.
- `electronic`: `ElectronicInputs`, input readers, covariance propagation
  and optional refitted trials.
- `session`: `save_run_bundle`, `replay_run_configuration`.
- `io`, `processing`, `models`, `results` and `plotting`: reference and
  spreadsheet input, state handling, labelled outputs and figures.

Names beginning with an underscore and GUI widget layouts are implementation
details. The `IDX_*` row format remains available for existing callers;
`NuclearState.from_row` offers named access.

`GetBindingEnergy`, `GetBindingEnergyFromNIST` and the retained `Ebind` fields
use cumulative ionization energy in MeV. Custom callbacks must use that convention.
`MassCalibration` retains a tuple interface; prefer named result columns in new code.

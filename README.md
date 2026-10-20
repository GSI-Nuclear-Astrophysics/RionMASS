# RionMASS

RionMASS converts storage-ring revolution frequencies or times into atomic
mass excesses. Python, command-line and Qt interfaces share a polynomial
calibration engine and return the same reference diagnostics and result tables.

The workflow includes AME2020 references, nuclear-state and charge-state
handling, custom reference masses, electronic-input covariance, leave-one-out
validation and saved-run replay. Frequency extraction belongs upstream;
RionMASS starts from observables and their uncertainties.

## Installation

Python 3.10 or later is required.

```sh
python -m pip install RionMASS              # calibration engine
python -m pip install 'RionMASS[data,plot]'  # spreadsheets and plots
python -m pip install 'RionMASS[gui]'        # complete desktop workflow
```

For development, install `python -m pip install -e '.[dev]'` from a checkout.

## Python workflow

```python
from rionmass.api import export_result_tables, run_full_calibration_workflow
from rionmass.configuration import CalibrationConfig, ODSImportConfig

ods = ODSImportConfig("frequency_observables.ods", 0, unmeasured_list=("15N7+",), observable="frequency")
calibration = CalibrationConfig(elbien_source="nist", p=3)
result = run_full_calibration_workflow(ods, calibration)
print(result.unknown_df)
export_result_tables(result, "results")
```

`p` counts polynomial coefficients; the degree is `p - 1`. ODS input uses
columns for ion label, centroid frequency, centroid error, width and width
error, with frequencies in Hz. Supply `harmonic` when input frequencies have
not been reduced to the fundamental. The time branch returns revolution times
in ps. Isomer excitation energies are supplied separately in MeV.

`python examples/quickstart.py` runs an offline demonstration with public
AME2020 records and the packaged NIST table. Its observables are constructed
to demonstrate the interfaces; they are not experimental measurements.

## Command line and desktop application

```sh
rionmass run --ods frequency_observables.ods --sheet-index 0 \
  --elbien-source nist --unmeasured 15N7+ --p 3 --outdir results
rionmass-gui
```

The GUI shows calibration plots, unknown masses, held-out reference predictions,
AME2020 differences and separate uncertainty terms. Calibration and export run
in background tasks. The selected references, order, states and electronic
inputs remain inspectable.

## Reference inputs and uncertainties

The GUI's **Custom masses (keV)** field and repeatable CLI `--reference-mass`
options accept `NUCLIDE=ME_KEV,SIGMA_KEV`, a neutral-atom ground-state mass
excess and positive one-standard-deviation error. Python callers use
`CalibrationConfig(reference_mass_overrides=...)`. Replacements enter the
calibration and LOO comparisons; AME2020 remains a separately labelled comparator.

Explicit **LOO anchors** remain in the final fit with their input uncertainty.
Their own holdouts are shown as diagnostics and excluded from the extra-scatter
estimate. The default policy retains them in other folds' training sets.
Select anchors from reference geometry or independent information before
examining residuals.

Result tables separate calibration-curve, unknown-observable, LOO extra-scatter
and electronic terms. The correction is cumulative ionization energy:
`I_q = B_neutral - B_ion`. ElBiEn columns count remaining electrons; NIST
total-binding rows use ion charge. Shared electronic inputs are propagated
through calibration and atomic-mass recovery, retaining cancellation and
cross-covariance. Missing covariance remains explicitly unavailable.

The extra-scatter estimate uses held-out predictions, not the full-fit reduced
chi-square. LOO folds share references, so the marginal-pull convention is an
empirical validation diagnostic. Unprovided reference or observable correlations
are not inferred. See [Scientific method](docs/SCIENTIFIC_METHOD.md) and
[Electronic inputs](docs/UNCERTAINTIES.md) for units and covariance assumptions.

## Saved runs and replay

**Save config + tables**, **Load config…** and **Replay config…** create or
restore `DATE_RionMASS_CONFIG_suffix` bundles. Completed runs freeze reference
records, electronic inputs, software versions, constants and numerical
checkpoints. Input spreadsheets are referenced rather than copied.

```python
from rionmass.session import save_run_bundle, replay_run_configuration

folder, files = save_run_bundle("outputs", "label", ods, calibration, result)
report = replay_run_configuration(folder / "config.json")
print(report.mismatches, report.warnings)
```

An edited ODS file produces a hash warning and is still replayed with the saved
sheet index. Numerical differences are reported separately. Settings-only
snapshots contain no completed-result checkpoints. See
[Reproducibility](docs/REPRODUCIBILITY.md).

## Development, citation and license

```sh
pytest
ruff check .
ruff format --check .
pyright
python -m build
twine check dist/*
python scripts/check_release.py dist/*
```

Tests use constructed numerical fixtures and published reference inputs.
See [CONTRIBUTING.md](CONTRIBUTING.md), [CITATION.cff](CITATION.cff),
[release instructions](docs/RELEASE.md) and [LICENSE](LICENSE).
For version 1.0.0, cite [doi:10.5281/zenodo.23501662](https://doi.org/10.5281/zenodo.23501662).
The license is GPL-3.0-or-later.

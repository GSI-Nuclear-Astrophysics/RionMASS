"""Offline calibration workflow using public reference data and constructed frequencies."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import ezodf
import numpy as np

from rionmass.api import export_result_tables, run_full_calibration_workflow
from rionmass.configuration import CalibrationConfig, ODSImportConfig
from rionmass.constants import ELEMENT_Z, amu, me
from rionmass.electronic import binding_energy_inputs
from rionmass.io import fetch_nist_electron_binding_table
from rionmass.session import replay_run_configuration, save_run_bundle

UNKNOWN = "15N7+"


def run_example(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    public = json.loads(Path(__file__).with_name("ame2020_reference_subset.json").read_text())
    records = public["records"]
    inputs = binding_energy_inputs(fetch_nist_electron_binding_table(), [(6, 6), (7, 7), (8, 8)], "nist", True)
    # Independent elemental totals are an explicit assumption for this example.
    inputs_json = {
        "format": "corrections",
        "source": "Packaged NIST total-binding cache; independent elemental totals for this example",
        "corrections": [
            {
                "Z": z,
                "q": q,
                "I_q_MeV": inputs.correction(z, q),
                "sigma_MeV": float(np.sqrt(inputs.covariance_MeV2[i, i])),
            }
            for i, (z, q) in enumerate(inputs.ions)
        ],
        "assume_independent": True,
    }
    electronic_path = output / "electronic.json"
    electronic_path.write_text(json.dumps(inputs_json, indent=2) + "\n")
    ods_path = output / "illustrative_observables.ods"
    document = ezodf.newdoc(doctype="ods", filename=str(ods_path))
    sheet = ezodf.Sheet("Illustrative", size=(len(records) + 1, 5))
    document.sheets += sheet
    header = ("Label", "mu_fit_Hz", "sigma_Hz", "width_Hz", "width_err_Hz")
    for column, value in enumerate(header):
        sheet[0, column].set_value(value)
    for row, record in enumerate(records, 1):
        a, z = record["A"], record["Z"]
        element = next(symbol for symbol, atomic_number in ELEMENT_Z.items() if atomic_number == z)
        moq = (a * amu + record["ME (keV)"] / 1000 - z * me + inputs.correction(z, z)) / (z * amu)
        frequency = 1_600_000 - 2_000_000 * (moq - 2)
        values = (f"{a}{element}{z}+", frequency, 0.5, 8.0, 0.2)
        for column, value in enumerate(values):
            sheet[row, column].set_value(value)
    document.save()
    ods = ODSImportConfig(ods_path, 0, unmeasured_list=(UNKNOWN,))
    calibration = CalibrationConfig(electronic_inputs=inputs, p=2, compare_orders=(2, 3))
    result = run_full_calibration_workflow(ods, calibration, ame_data=records)
    print("Illustrative frequencies; not experimental measurements.")
    print(result.fit.parameter_table().to_string(index=False))
    print(
        result.unknown_df[["species_id", "ME_state_keV", "ME_total_err_keV", "Delta_ME_state_keV"]].to_string(
            index=False
        )
    )
    files = export_result_tables(result, output / "tables")
    bundle, _ = save_run_bundle(output, "public_example", ods, calibration, result)
    replay = replay_run_configuration(bundle / "config.json")
    if replay.mismatches:
        raise RuntimeError(f"Example replay differs: {replay.mismatches}")
    print(f"Exported {len(files)} tables; saved-run replay agrees.")
    return ods, calibration, result, bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output:
        run_example(args.output)
    else:
        with tempfile.TemporaryDirectory() as directory:
            run_example(Path(directory))


if __name__ == "__main__":
    main()

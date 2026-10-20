"""Shared import, calibration, validation and mass-prediction workflow.

Python, CLI and GUI callers receive the same result tables and covariance
components. Reference and electronic inputs can be frozen for replay.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .configuration import CalibrationConfig, ODSImportConfig
from .electronic import (
    ElectronicPropagation,
    binding_energy_inputs,
    propagate_electronic_uncertainty,
    read_electronic_inputs,
)
from .fitting import fit_calibration, predict_unknown_table
from .io import (
    Read_AME_barion,
    Read_elbien_file,
    build_data_from_ods,
    fetch_nist_electron_binding_table,
)
from .models import CalibrationFit
from .processing import build_nuclei_from_data, get_isomer_energy
from .reference_masses import annotate_loo_references, apply_reference_mass_overrides, prepare_ame_comparators
from .results import final_mass_results, loo_reference_results
from .uncertainty import compare_polynomial_orders, estimate_extra_systematic_from_loo, leave_one_out_self_calibration


@dataclass
class CalibrationResult:
    """Tables, fitted model and covariance components from one calibration.

    electronic_uncertainty_complete describes the supplied electronic model;
    other measurement correlations remain the caller's responsibility.
    """

    nuclei: list[list[Any]]
    nuclei_y: list[list[Any]]
    nuclei_n: list[list[Any]]
    import_aux: dict[str, Any]
    fit: CalibrationFit
    loo_df: pd.DataFrame
    loo_with_sys: pd.DataFrame
    extra_sys_keV: float
    systematic_result: dict[str, Any]
    order_comparison: pd.DataFrame
    unknown_df: pd.DataFrame
    unknown_influence: dict[str, pd.DataFrame]
    electronic: ElectronicPropagation | None = None
    loo_electronic_covariance_keV2: pd.DataFrame | None = None
    electronic_uncertainty_complete: bool = False
    reference_inputs: pd.DataFrame = field(default_factory=pd.DataFrame)


def run_full_calibration_workflow(
    ods_config: ODSImportConfig,
    calibration_config: CalibrationConfig,
    *,
    ame_data: list[dict[str, Any]] | None = None,
) -> CalibrationResult:
    """Import observables, calibrate, validate and predict unknown masses.

    ame_data supplies frozen reference records for replay. Without it, the
    workflow reads AME2020. Input and fitting errors propagate to the caller.
    """
    input_path = Path(ods_config.filename).expanduser()
    input_sha256 = hashlib.sha256(input_path.read_bytes()).hexdigest() if input_path.is_file() else None
    data, aux = build_data_from_ods(
        filename=ods_config.filename,
        sheet_index=ods_config.sheet_index,
        exclusion_list=ods_config.exclusion_list,
        unmeasured_list=ods_config.unmeasured_list,
        observable=ods_config.observable,
        harmonic=ods_config.harmonic,
        revname=ods_config.revname,
        header=ods_config.header,
        verbose=False,
    )

    aux = dict(aux)
    if input_sha256 is not None:
        aux["input_sha256"] = input_sha256
    AME_data = (
        Read_AME_barion()
        if ame_data is None
        else [
            {k: (np.nan if v is None and k in ("ME (keV)", "ME_error (keV)") else v) for k, v in row.items()}
            for row in ame_data
        ]
    )

    ions = [(int(row[1]), int(row[2])) for row in data]
    if calibration_config.electronic_inputs is not None:
        electronic_inputs = calibration_config.electronic_inputs
    elif calibration_config.electronic_inputs_path is not None:
        electronic_inputs = read_electronic_inputs(calibration_config.electronic_inputs_path, ions)
    elif calibration_config.elbien_source == "nist":
        nist_data = fetch_nist_electron_binding_table(calibration_config.nist_cache_path)
        electronic_inputs = binding_energy_inputs(
            nist_data,
            ions,
            "nist",
            calibration_config.electronic_assume_independent,
        )
    else:
        # File-source configurations require this path during validation.
        assert calibration_config.elbien_path is not None
        elbien_data = Read_elbien_file(calibration_config.elbien_path)
        electronic_inputs = binding_energy_inputs(
            elbien_data,
            ions,
            "file",
            calibration_config.electronic_assume_independent,
        )
    AME_data = prepare_ame_comparators(data, AME_data, calibration_config.reference_mass_overrides)
    nuclei, nuclei_y, nuclei_n = build_nuclei_from_data(
        data,
        AME_data,
        isomer_excitation_energy=calibration_config.isomer_excitation_energy,
        binding_energy_fn=electronic_inputs.correction,
    )

    nuclei_y, reference_inputs = apply_reference_mass_overrides(
        nuclei_y, calibration_config.reference_mass_overrides, calibration_config.isomer_excitation_energy
    )

    fit = fit_calibration(
        nuclei_y,
        p=calibration_config.p,
        isomer_excitation_energy=calibration_config.isomer_excitation_energy,
        iterationMax=calibration_config.iterationMax,
        tol=calibration_config.tol,
        observable_label=aux["x_label"],
        observable_unit=aux["x_unit"],
    )

    loo_df, _loo_legacy_rows, _loo_fits = leave_one_out_self_calibration(
        nuclei_y,
        p=calibration_config.p,
        isomer_excitation_energy=calibration_config.isomer_excitation_energy,
        observable_label=aux["x_label"],
        observable_unit=aux["x_unit"],
        iterationMax=calibration_config.iterationMax,
        tol=calibration_config.tol,
        electronic_inputs=electronic_inputs,
        electronic_method=calibration_config.electronic_method,
        electronic_step_MeV=calibration_config.electronic_step_MeV,
        electronic_workers=calibration_config.electronic_workers,
        loo_anchor_species=calibration_config.loo_anchor_species,
        loo_anchor_policy=calibration_config.loo_anchor_policy,
    )

    loo_df = annotate_loo_references(loo_df, reference_inputs)

    systematic_result = estimate_extra_systematic_from_loo(loo_df, p=calibration_config.p, center_residuals=False)
    extra_sys_keV = systematic_result["extra_sys_keV"]
    loo_with_sys = systematic_result["table"]
    eligible_loo = loo_df.loc[loo_df["used_for_extra_scatter"]]
    systematic_result["electronic_uncertainty_complete"] = bool(eligible_loo["electronic_uncertainty_known"].all())

    order_comparison = compare_polynomial_orders(
        nuclei_y,
        p_values=calibration_config.compare_orders,
        isomer_excitation_energy=calibration_config.isomer_excitation_energy,
        observable_label=aux["x_label"],
        observable_unit=aux["x_unit"],
        electronic_inputs=electronic_inputs,
        electronic_method=calibration_config.electronic_method,
        electronic_step_MeV=calibration_config.electronic_step_MeV,
        electronic_workers=calibration_config.electronic_workers,
        loo_anchor_species=calibration_config.loo_anchor_species,
        loo_anchor_policy=calibration_config.loo_anchor_policy,
        iterationMax=calibration_config.iterationMax,
        tol=calibration_config.tol,
    )

    _legacy_unknown_rows, unknown_df, unknown_influence = predict_unknown_table(
        fit, nuclei_n, extra_sys_keV=extra_sys_keV
    )

    electronic = propagate_electronic_uncertainty(
        fit,
        nuclei_y,
        nuclei_n,
        electronic_inputs,
        extra_sys_keV=extra_sys_keV,
        method=calibration_config.electronic_method,
        iterationMax=calibration_config.iterationMax,
        tol=calibration_config.tol,
        step_MeV=calibration_config.electronic_step_MeV,
        workers=calibration_config.electronic_workers,
    )
    if len(unknown_df):
        unknown_df["ME_conditional_err_keV"] = unknown_df["ME_total_err_keV"]
        electronic_sigma = np.sqrt(np.maximum(np.diag(electronic.mass_covariance_keV2), 0))
        unknown_df["ME_electronic_err_keV"] = electronic_sigma
        unknown_df["electronic_uncertainty_known"] = np.isfinite(electronic_sigma)
        unknown_df["ME_total_err_keV"] = np.sqrt(np.maximum(np.diag(electronic.total_mass_covariance_keV2), 0))
        iso_sigma = np.array([get_isomer_energy(r, fit.isomer_excitation_energy)[1] * 1000 for r in nuclei_n])
        unknown_df["ME_gs_equiv_err_keV"] = np.sqrt(unknown_df["ME_total_err_keV"] ** 2 + iso_sigma**2)

    return CalibrationResult(
        reference_inputs=reference_inputs,
        nuclei=nuclei,
        nuclei_y=nuclei_y,
        nuclei_n=nuclei_n,
        import_aux=aux,
        fit=fit,
        loo_df=loo_df,
        loo_with_sys=loo_with_sys,
        extra_sys_keV=extra_sys_keV,
        systematic_result=systematic_result,
        order_comparison=order_comparison,
        unknown_df=unknown_df,
        unknown_influence=unknown_influence,
        electronic=electronic,
        loo_electronic_covariance_keV2=loo_df.attrs.get("electronic_covariance_keV2"),
        electronic_uncertainty_complete=bool(
            np.all(np.isfinite(electronic.total_mass_covariance_keV2))
            and systematic_result["electronic_uncertainty_complete"]
        ),
    )


def export_result_tables(result: CalibrationResult, outdir: str) -> list[str]:
    """Export result tables as CSV files and return their paths.

    The output directory is created if needed; existing filenames are replaced.
    """
    from pathlib import Path

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    written = []

    def _write(df: pd.DataFrame, name: str, index: bool = False) -> None:
        path = out / name
        df.to_csv(path, index=index)
        written.append(str(path))

    if not result.reference_inputs.empty:
        _write(result.reference_inputs, "reference_mass_inputs.csv")
    _write(result.fit.parameter_table(), "polynomial_parameters.csv")
    _write(result.fit.covariance_table(), "polynomial_covariance.csv", index=True)
    _write(result.fit.correlation_table(), "polynomial_correlation.csv", index=True)
    _write(result.fit.calibrant_table(), "calibrant_fit_table.csv")
    _write(result.fit.flow_table(), "flow_of_information_coefficients.csv")
    _write(result.loo_df, "leave_one_out_without_extra_sys.csv")
    _write(result.loo_with_sys, "leave_one_out_with_extra_sys.csv")
    _write(result.order_comparison, "polynomial_order_comparison.csv")
    _write(result.unknown_df, "unknown_masses_polynomial.csv")
    _write(final_mass_results(result.unknown_df), "final_mass_results.csv")
    _write(loo_reference_results(result.loo_with_sys), "loo_reference_results.csv")
    _write(
        pd.DataFrame([{k: v for k, v in result.systematic_result.items() if k != "table"}]),
        "loo_validation_summary.csv",
    )
    if result.electronic is not None:
        electronic = result.electronic
        inputs = electronic.inputs
        names = pd.Index(inputs.labels)
        _write(inputs.primitive_table(), "electronic_primitives.csv")
        _write(inputs.correction_table(), "electronic_corrections.csv")
        _write(
            pd.DataFrame(inputs.covariance_MeV2, index=names, columns=names),
            "electronic_input_covariance_MeV2.csv",
            index=True,
        )
        _write(electronic.mass_covariance_keV2, "electronic_mass_covariance_keV2.csv", index=True)
        _write(electronic.coefficient_covariance, "electronic_polynomial_covariance.csv", index=True)
        _write(electronic.mass_coefficient_covariance, "electronic_mass_polynomial_cross_covariance.csv", index=True)
        _write(electronic.conditional_mass_covariance_keV2, "mass_covariance_conditional_keV2.csv", index=True)
        _write(electronic.total_mass_covariance_keV2, "mass_covariance_keV2.csv", index=True)
        _write(
            pd.DataFrame(
                electronic.mass_jacobian_keV_per_MeV, index=electronic.mass_covariance_keV2.index, columns=names
            ),
            "electronic_mass_jacobian_keV_per_MeV.csv",
            index=True,
        )
        if result.loo_electronic_covariance_keV2 is not None:
            _write(result.loo_electronic_covariance_keV2, "loo_electronic_covariance_keV2.csv", index=True)

    for species_id, infl in result.unknown_influence.items():
        safe = species_id.replace("+", "p").replace("/", "_")
        _write(infl, f"prediction_influence_{safe}.csv")

    return written

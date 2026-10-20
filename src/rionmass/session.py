"""Versioned configuration snapshots and verified calibration replay.

No raw spectra or ODS files are copied. Resolved electronic inputs and the
selected AME2020 inputs are saved with completed runs to freeze external data.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy

from .api import CalibrationResult, export_result_tables, run_full_calibration_workflow
from .bundles import create_run_directory
from .configuration import CalibrationConfig, ODSImportConfig
from .electronic import ElectronicInputs
from .version import __version__

SCHEMA_VERSION = 2


def _plain(value: Any) -> Any:
    if isinstance(value, ElectronicInputs):
        return {f.name: _plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def configuration_values(ods: ODSImportConfig, calibration: CalibrationConfig) -> dict[str, Any]:
    """Lossless settings representation, including hidden GUI/API options."""
    return {
        "ods": {f.name: _plain(getattr(ods, f.name)) for f in fields(ods)},
        "calibration": {f.name: _plain(getattr(calibration, f.name)) for f in fields(calibration)},
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _verification(result: CalibrationResult) -> dict[str, Any]:
    columns = [
        "species_id",
        "ME_state_keV",
        "ME_stat_err_keV",
        "ME_conditional_err_keV",
        "ME_electronic_err_keV",
        "ME_total_err_keV",
        "Delta_ME_state_keV",
    ]
    return _plain(
        {
            "coefficients": result.fit.coeff,
            "extra_sys_keV": result.extra_sys_keV,
            "unknowns": result.unknown_df.reindex(columns=columns).to_dict(orient="records"),
            "loo": result.loo_with_sys.reindex(
                columns=[
                    "species_id",
                    "Delta_ME_state_keV",
                    "Delta_ME_reference_keV",
                    "reference_mass_source",
                    "ME_reference_state_keV",
                    "ME_reference_state_err_keV",
                    "loo_sigma_no_extra_keV",
                    "used_for_extra_scatter",
                ]
            ).to_dict(orient="records"),
        }
    )


def save_run_configuration(
    path: str | Path,
    ods: ODSImportConfig,
    calibration: CalibrationConfig,
    result: CalibrationResult | None = None,
) -> None:
    """Save all settings, input fingerprints and optional numerical checkpoints."""
    payload = configuration_values(ods, calibration)
    # Paths are absolute so a session can be opened from a different cwd.
    for block, keys in (
        ("ods", ("filename", "revname")),
        ("calibration", ("elbien_path", "nist_cache_path", "electronic_inputs_path")),
    ):
        for key in keys:
            if payload[block][key] is not None:
                payload[block][key] = str(Path(payload[block][key]).expanduser().resolve())
    active_files = {"ods": payload["ods"]["filename"]}
    if result is not None and result.electronic is not None:
        payload["calibration"]["electronic_inputs"] = _plain(result.electronic.inputs)
        payload["calibration"]["electronic_inputs_path"] = None
        payload["calibration"]["electronic_assume_independent"] = False
    elif calibration.electronic_inputs is None:
        key = (
            "electronic_inputs_path"
            if calibration.electronic_inputs_path
            else ("elbien_path" if calibration.elbien_source == "file" else "nist_cache_path")
        )
        if payload["calibration"].get(key):
            active_files[key] = payload["calibration"][key]
    if result is not None and result.import_aux.get("input_sha256") is not None:
        if _sha256(Path(active_files["ods"])) != result.import_aux["input_sha256"]:
            raise ValueError("The ODS changed after calibration; rerun before saving these results")
    payload.update(
        {
            "schema_version": SCHEMA_VERSION,
            "software": "RionMASS",
            "software_version": __version__,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "environment": {"numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__},
            "inputs": {key: {"path": p, "sha256": _sha256(Path(p))} for key, p in active_files.items()},
        }
    )
    if result is not None:
        # Ground-state AME entries; isomer energies stay in calibration settings.
        unique = {
            str(r[0]): {
                "Element": str(r[0]),
                "Z": int(r[1]),
                "A": int(r[2]),
                "ME (keV)": float(r[10]) * 1000,
                "ME_error (keV)": float(r[11]) * 1000,
            }
            for r in result.nuclei
        }
        payload["ame2020_inputs"] = _plain(list(unique.values()))
        payload["verification"] = _verification(result)
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_run_configuration(path: str | Path) -> tuple[ODSImportConfig, CalibrationConfig, dict[str, Any]]:
    """Load settings without running analysis or accessing input files."""
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") not in (1, SCHEMA_VERSION) or payload.get("software") != "RionMASS":
        raise ValueError("Unsupported RionMASS configuration schema or software")
    ods_dict, cal_dict = dict(payload["ods"]), dict(payload["calibration"])
    for block, keys in (
        (ods_dict, ("filename", "revname")),
        (cal_dict, ("elbien_path", "nist_cache_path", "electronic_inputs_path")),
    ):
        for key in keys:
            if block.get(key) is not None:
                p = Path(block[key]).expanduser()
                block[key] = str(p if p.is_absolute() else source.parent / p)
    for block, keys in (
        (ods_dict, ("exclusion_list", "unmeasured_list")),
        (cal_dict, ("compare_orders", "loo_anchor_species")),
    ):
        for key in keys:
            if key in block:
                block[key] = tuple(block[key])
    if cal_dict.get("electronic_inputs") is not None:
        model = cal_dict["electronic_inputs"]
        cal_dict["electronic_inputs"] = ElectronicInputs(
            labels=tuple(model["labels"]),
            values_MeV=np.asarray(model["values_MeV"], dtype=float),
            covariance_MeV2=np.asarray(model["covariance_MeV2"], dtype=float),
            ions=tuple(tuple(v) for v in model["ions"]),
            response=np.asarray(model["response"], dtype=float),
            source=model["source"],
        )
    return ODSImportConfig(**ods_dict), CalibrationConfig(**cal_dict), payload


def _compare(expected: Any, actual: Any, label: str = "results") -> list[str]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        return [
            message for key, value in expected.items() for message in _compare(value, actual.get(key), f"{label}.{key}")
        ]
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{label}: different row count"]
        return [
            message
            for i, (a, b) in enumerate(zip(expected, actual, strict=True))
            for message in _compare(a, b, f"{label}[{i}]")
        ]
    if (
        isinstance(expected, (int, float))
        and not isinstance(expected, bool)
        and isinstance(actual, (int, float))
        and not isinstance(actual, bool)
    ):
        if math.isclose(expected, actual, rel_tol=1e-10, abs_tol=1e-6):
            return []
    elif expected == actual:
        return []
    return [f"{label}: expected {expected!r}, obtained {actual!r}"]


@dataclass
class ReplayReport:
    result: CalibrationResult
    ods: ODSImportConfig
    calibration: CalibrationConfig
    payload: dict[str, Any]
    warnings: list[str]
    mismatches: list[str]


def replay_run_configuration(path: str | Path, *, ods_override: str | Path | None = None) -> ReplayReport:
    """Warn on changed file hashes, rerun calibration, and compare stored checkpoints."""
    ods, calibration, payload = load_run_configuration(path)
    if ods_override is not None:
        ods.filename = str(Path(ods_override).expanduser().resolve())
    warnings = []
    for key, entry in payload.get("inputs", {}).items():
        p = Path(ods.filename) if key == "ods" else Path(getattr(calibration, key))
        if _sha256(p) != entry["sha256"]:
            warnings.append(
                f"Input file changed since the saved run ({key}): {p}. "
                "SHA256 differs; replay continues with the saved sheet index and settings. "
                "Numerical checkpoints will be compared after recalculation."
            )
    if payload["software_version"] != __version__:
        warnings.append(f"Recorded RionMASS {payload['software_version']}; running {__version__}")
    for name, version in {"numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__}.items():
        if payload.get("environment", {}).get(name) != version:
            warnings.append(f"{name} version differs from saved environment")
    result = run_full_calibration_workflow(ods, calibration, ame_data=payload.get("ame2020_inputs"))
    mismatches = _compare(payload["verification"], _verification(result)) if "verification" in payload else []
    if "verification" not in payload:
        warnings.append("Settings-only configuration: no saved numerical checkpoints to verify")
    return ReplayReport(result, ods, calibration, payload, warnings, mismatches)


def save_run_bundle(
    parent: str | Path,
    label: str,
    ods: ODSImportConfig,
    calibration: CalibrationConfig,
    result: CalibrationResult | None = None,
) -> tuple[Path, list[str]]:
    """Save configuration and all available tables into one new dated folder."""
    out = create_run_directory(parent, "RionMASS", label)
    config = out / "config.json"
    save_run_configuration(config, ods, calibration, result)
    written = [str(config)]
    if result is not None:
        written.extend(export_result_tables(result, str(out)))
    return out, written

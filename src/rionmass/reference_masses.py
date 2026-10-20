"""Explicit ground-state atomic mass inputs, separate from AME comparisons."""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd

from .constants import ELEMENT_Z
from .models import IDX_ME_AME, IDX_MEERR_AME, IDX_MOQ_AME, IDX_MOQERR_AME
from .processing import ame_moq_for_nuclear_state, get_isomer_metadata


def reference_nuclide(label: str) -> str:
    """Canonical ground-state nuclide; an optional ionic charge is ignored."""
    match = re.fullmatch(r"([1-9]\d*)([A-Z][a-z]?)(?:[1-9]\d*\+)?", label.strip())
    if match is None or match[2] not in ELEMENT_Z or int(match[1]) < ELEMENT_Z[match[2]]:
        raise ValueError(f"Invalid ground-state reference nuclide {label!r}; use e.g. 206Hg (no isomer label).")
    return match[1] + match[2]


def normalize_reference_mass_overrides(values: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Accept (ME_keV, sigma_keV[, source]) pairs or explicit dictionaries."""
    if not isinstance(values, dict):
        raise ValueError("reference_mass_overrides must be a nuclide-to-mass dictionary")
    out = {}
    for name, specification in values.items():
        if not isinstance(name, str):
            raise ValueError("Reference nuclide labels must be strings")
        key = reference_nuclide(name)
        if key in out:
            raise ValueError(f"Duplicate custom reference for {key}")
        if isinstance(specification, dict):
            if set(specification) - {"mass_excess_keV", "uncertainty_keV", "source"}:
                raise ValueError(f"Unknown reference mass fields for {key}")
            mass = specification.get("mass_excess_keV")
            sigma = specification.get("uncertainty_keV")
            source = specification.get("source", "User supplied")
        elif isinstance(specification, (tuple, list)) and len(specification) in (2, 3):
            mass, sigma = specification[:2]
            source = specification[2] if len(specification) == 3 else "User supplied"
        else:
            raise ValueError(f"Supply mass excess and uncertainty in keV for {key}")
        try:
            if mass is None or sigma is None or isinstance(mass, bool) or isinstance(sigma, bool):
                raise ValueError("Boolean mass")
            mass, sigma = float(mass), float(sigma)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid mass excess or uncertainty for {key}") from exc
        if not math.isfinite(mass) or not math.isfinite(sigma) or sigma <= 0:
            raise ValueError(
                f"Custom reference {key} requires a finite mass excess and a positive finite 1-sigma uncertainty"
            )
        if not isinstance(source, str) or "\n" in source or "\r" in source:
            raise ValueError(f"Reference source for {key} must be a single line of text")
        out[key] = dict(mass_excess_keV=mass, uncertainty_keV=sigma, source=source.strip() or "User supplied")
    return out


def parse_reference_mass_spec(items: list[str]) -> dict[str, dict[str, Any]]:
    """Parse NUCLIDE=ME_KEV,SIGMA_KEV # optional source (one per line)."""
    specifications = {}
    for item in items:
        body, _, source = item.partition("#")
        if not body.strip():
            continue
        try:
            name, numbers = body.split("=", 1)
            mass, sigma = numbers.split(",")
        except ValueError as exc:
            raise ValueError(f"Expected NUCLIDE=ME_KEV,SIGMA_KEV # optional source; got {item!r}") from exc
        key = reference_nuclide(name)
        if key in specifications:
            raise ValueError(f"Duplicate custom reference for {key}")
        specifications[key] = (mass.strip(), sigma.strip(), source.strip() or "User supplied")
    return normalize_reference_mass_overrides(specifications)


def prepare_ame_comparators(data, ame, overrides):
    """Permit an explicitly supplied reference absent from AME, without inventing AME values."""
    available = {str(row[0]) for row in data if row[3] == "Y"}
    for name in overrides:
        if name not in available:
            raise ValueError(
                f"Custom reference {name} is not a retained calibration reference. "
                "Check exclusions and the unknown-ion list."
            )
    records = list(ame)
    known = {str(row["Element"]) for row in records}
    for row in data:
        name = str(row[0])
        if name in overrides and name not in known:
            mass_number = re.match(r"\d+", name)
            if mass_number is None:
                raise ValueError(f"Invalid imported nuclide label: {name}")
            a = int(mass_number[0])
            records.append({"Element": name, "Z": int(row[1]), "A": a, "ME (keV)": np.nan, "ME_error (keV)": np.nan})
            known.add(name)
    return records


def apply_reference_mass_overrides(rows, overrides, isomer_energies=None):
    """Copy calibration rows, replace their mass inputs, and record actual inputs."""
    references, provenance = [], []
    for original in rows:
        row = list(original)
        name = str(row[0])
        custom = overrides.get(name)
        ame_state = ame_moq_for_nuclear_state(original, isomer_energies)
        if custom is not None:
            row[IDX_ME_AME] = custom["mass_excess_keV"] / 1000
            row[IDX_MEERR_AME] = custom["uncertainty_keV"] / 1000
        state = ame_moq_for_nuclear_state(row, isomer_energies)
        row[IDX_MOQ_AME], row[IDX_MOQERR_AME] = state[:2]
        references.append(row)
        provenance.append(
            dict(
                species_id=get_isomer_metadata(row)[2],
                nuclide=name,
                reference_is_custom=custom is not None,
                reference_mass_source=("Custom: " + custom["source"]) if custom is not None else "AME2020",
                ME_reference_gs_keV=float(row[IDX_ME_AME]) * 1000,
                ME_reference_gs_err_keV=float(row[IDX_MEERR_AME]) * 1000,
                ME_reference_state_keV=state[2] * 1000,
                ME_reference_state_err_keV=state[3] * 1000,
                ME_AME_gs_keV=float(original[IDX_ME_AME]) * 1000,
                ME_AME_gs_err_keV=float(original[IDX_MEERR_AME]) * 1000,
                ME_AME_state_keV=ame_state[2] * 1000,
                ME_AME_state_err_keV=ame_state[3] * 1000,
            )
        )
    return references, pd.DataFrame(provenance)


def annotate_loo_references(loo, inputs):
    """Keep validation against supplied references and AME comparisons distinct."""
    out = loo.copy()
    out["Delta_ME_reference_keV"] = out["Delta_ME_state_keV"]
    for name in (
        "reference_is_custom",
        "reference_mass_source",
        "ME_reference_state_keV",
        "ME_reference_state_err_keV",
    ):
        out[name] = inputs[name].to_numpy()
    for name in ("ME_AME_state_keV", "ME_AME_state_err_keV"):
        out[name] = inputs[name].to_numpy()
    out["Delta_ME_state_keV"] = out["ME_state_keV"] - out["ME_AME_state_keV"]
    out["Delta_ME_gs_equiv_keV"] = out["Delta_ME_state_keV"]
    return out

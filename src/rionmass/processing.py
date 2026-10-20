"""Ion-label parsing, unit conversion and nuclear-state bookkeeping."""

from __future__ import annotations

import functools
import math
import re
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .constants import amu, me
from .models import IDX_A, IDX_EBIND, IDX_ME_AME, IDX_MEERR_AME, IDX_Q


def safe_float(value: Any, idx: int | None = None, column: int | None = None, array_name: str = "Nuclei") -> float:
    """Parse dot or decimal-comma numbers, including European thousands separators.

    Optional row and column indices identify malformed spreadsheet cells.
    """
    try:
        if value is None:
            raise ValueError("None")
        if isinstance(value, (float, int, np.floating, np.integer)):
            return float(value)
        cleaned = str(value).strip().replace("\u00a0", "").replace(" ", "")
        if "," in cleaned and "." in cleaned:
            cleaned = cleaned.replace(".", "").replace(",", ".")
        else:
            cleaned = cleaned.replace(",", ".")
        return float(cleaned)
    except Exception as exc:
        loc = ""
        if idx is not None and column is not None:
            loc = f" at index {idx}, column {column}, in {array_name}"
        raise ValueError(f"Could not convert value to float{loc}: {value!r} ({exc})") from None


def to_float_locale(value: Any) -> float:
    """Alias of :func:`safe_float` without row/column context, for table cells."""
    return safe_float(value)


def parse_ion_label(label: str) -> tuple[str, str, str, str]:
    """Parse a LaTeX or plain ion label into ``(A, isomer, element, charge)``.

    Accepts both ``$^{201m}$Pt$^{76+}$`` (LaTeX, as typically pasted from a
    template-fit spreadsheet) and ``201mPt76+`` (plain). Raises
    :class:`ValueError` if neither pattern matches.
    """
    clean = str(label).replace("$", "").replace("\\", "").replace("{", "").replace("}", "").strip()
    patterns = [
        r"\^(\d+)(m\d*|m)?([A-Z][a-z]?)\^(\d+)\+",
        r"(\d+)(m\d*|m)?([A-Z][a-z]?)(\d+)\+?",
    ]
    for pattern in patterns:
        match = re.search(pattern, clean)
        if match is not None:
            A = match.group(1)
            iso = match.group(2) or ""
            el = match.group(3)
            q = match.group(4)
            return A, iso, el, q
    raise ValueError(f"Could not parse ion label: {label}")


def get_isomer_metadata(nuc: Sequence[Any]) -> tuple[str, str, str]:
    """Extract ``(isomer, display_name, species_id)`` from a legacy row."""
    isomer = ""
    display_name = str(nuc[0])
    species_id = f"{nuc[0]}{int(float(nuc[IDX_Q]))}+"
    if len(nuc) > 15:
        isomer = str(nuc[15]).strip()
    if len(nuc) > 16:
        display_name = str(nuc[16]).strip()
    if len(nuc) > 17:
        species_id = str(nuc[17]).strip()
    return isomer, display_name, species_id


def get_isomer_energy(
    nuc: Sequence[Any], isomer_excitation_energy: dict[str, Any] | None = None
) -> tuple[float, float]:
    """Look up ``(excitation_energy_MeV, uncertainty_MeV)`` for an isomeric state.

    ``isomer_excitation_energy`` maps a species id or display name to either
    a bare float (energy, zero uncertainty) or a ``(value, uncertainty)``
    pair, both in MeV. Ground states (no match) return ``(0.0, 0.0)``.
    """
    if isomer_excitation_energy is None:
        return 0.0, 0.0
    _isomer, display_name, species_id = get_isomer_metadata(nuc)
    for key in (species_id, display_name):
        if key in isomer_excitation_energy:
            value = isomer_excitation_energy[key]
            if isinstance(value, (tuple, list)):
                return float(value[0]), float(value[1]) if len(value) > 1 else 0.0
            return float(value), 0.0
    return 0.0, 0.0


def ame_moq_for_nuclear_state(
    nuc: Sequence[Any], isomer_excitation_energy: dict[str, Any] | None = None
) -> tuple[float, float, float, float, float, float]:
    """AME-reference ionic m/q for the actual nuclear state a row represents.

    Combines the AME ground-state mass excess with the row's isomer
    excitation energy (if any) and cumulative ionization energy, following the
    ionic-mass convention ``M_ion = A*amu + ME - Q*me + E_bind``.

    Returns ``(MoQ_state, MoQ_state_err, ME_state, ME_state_err, E_iso,
    E_iso_err)``, with mass excesses and excitation energies in MeV; m/q is in u/e.
    """
    A = int(nuc[IDX_A])
    Q = float(nuc[IDX_Q])
    ME_gs = float(nuc[IDX_ME_AME])
    ME_gs_err = float(nuc[IDX_MEERR_AME])
    E_bind = float(nuc[IDX_EBIND])
    E_iso, E_iso_err = get_isomer_energy(nuc, isomer_excitation_energy)
    ME_state = ME_gs + E_iso
    ME_state_err = math.sqrt(ME_gs_err**2 + E_iso_err**2)
    M_ion_state = A * amu + ME_state - Q * me + E_bind
    MoQ_state = M_ion_state / amu / Q
    MoQ_state_err = ME_state_err / amu / Q
    return MoQ_state, MoQ_state_err, ME_state, ME_state_err, E_iso, E_iso_err


def build_nuclei_from_data(
    data: np.ndarray,
    AME_data: list[dict[str, Any]],
    elbien_data: list[dict[str, Any]] | None = None,
    isomer_excitation_energy: dict[str, Any] | None = None,
    binding_energy_fn: Callable[[int, int], float] | None = None,
) -> tuple[list[list[Any]], list[list[Any]], list[list[Any]]]:
    """Build ``(Nuclei, Nuclei_Y, Nuclei_N)`` legacy row lists from raw import data.

    ``data`` is the array produced by :func:`rionmass.io.build_data_from_ods`.
    ``Nuclei_Y`` holds calibrant rows (``ref == "Y"``), ``Nuclei_N`` holds
    unknown rows (``ref == "N"``); ``Nuclei`` holds both, in input order.

    The cumulative ionization energy ``I_Q = B_neutral - B_ion`` comes from ``elbien_data`` (via
    :func:`rionmass.io.GetBindingEnergy`) by default. Pass
    ``binding_energy_fn(Z, Q) -> MeV`` instead to use a different source
    (e.g. :func:`rionmass.io.GetBindingEnergyFromNIST` bound to a fetched
    NIST table). A custom callback must return ``I_Q``, not the remaining
    ion's total electronic binding energy. See :class:`rionmass.configuration.CalibrationConfig`'s
    ``elbien_source``, which is what most callers should use rather than
    calling this directly. Exactly one of ``elbien_data`` or
    ``binding_energy_fn`` must effectively be usable.
    """
    from .io import GetAMEData, GetBindingEnergy  # local import: avoids a module cycle

    get_binding_energy: Callable[[int, int], float]
    if binding_energy_fn is not None:
        get_binding_energy = binding_energy_fn
    elif elbien_data is not None:
        get_binding_energy = functools.partial(GetBindingEnergy, elbien_data)
    else:
        raise ValueError("Either elbien_data or binding_energy_fn must be given")

    Nuclei: list[list[Any]] = []
    Nuclei_Y: list[list[Any]] = []
    Nuclei_N: list[list[Any]] = []
    for ion in data:
        base = ion[:8].tolist()
        isomer = ion[8]
        display_name = ion[9]
        species_id = ion[10]
        ame_name = ion[0]
        Z = int(ion[1])
        q = int(ion[2])
        ref_flag = ion[3]
        ME, MEError, A, N = GetAMEData(AME_data, ame_name)
        EBindingEnergy = get_binding_energy(Z, q)
        Mass = A * amu + ME - q * me + EBindingEnergy
        if isomer_excitation_energy and display_name in isomer_excitation_energy:
            val = isomer_excitation_energy[display_name]
            Mass += float(val[0] if isinstance(val, (list, tuple)) else val)
        Mass_Q = Mass / amu / q
        # Convert mass-excess uncertainty from MeV to u/e.
        Mass_QError = MEError / amu / q
        aux = [*base, ME, MEError, EBindingEnergy, Mass_Q, Mass_QError]
        aux.insert(2, A)
        aux.insert(3, N)
        aux += [isomer, display_name, species_id]
        Nuclei.append(aux)
        if ref_flag == "Y":
            Nuclei_Y.append(aux)
        else:
            Nuclei_N.append(aux)
    return Nuclei, Nuclei_Y, Nuclei_N


def filter_ground_states(
    df: pd.DataFrame,
    ground_only: bool = False,
    isomer_col: str = "isomer",
    label_cols: tuple[str, ...] = ("species_id", "display_name", "label"),
) -> pd.DataFrame:
    """Return only ground-state rows of ``df`` if ``ground_only=True``.

    A row is treated as isomeric if ``isomer_col`` is present and non-empty,
    or if any of ``label_cols`` contains a mass-number-plus-``m`` pattern
    (e.g. ``201mPt``, ``204mPb``).
    """
    if not ground_only:
        return df.copy()

    out = df.copy()
    is_isomer = np.zeros(len(out), dtype=bool)

    if isomer_col in out.columns:
        iso = out[isomer_col].fillna("").astype(str).str.strip()
        is_isomer |= ~iso.isin(["", "0", "False", "false", "nan", "None"])

    for col in label_cols:
        if col in out.columns:
            labels = out[col].fillna("").astype(str)
            is_isomer |= labels.str.contains(r"\d+m\d*[A-Z][a-z]?", regex=True)

    return out.loc[~is_isomer].copy()

"""AME2020, electronic-energy tables and spreadsheet input.

ElBiEn columns count bound electrons; NIST total-binding rows use ion
charge. Both scalar lookups return I_q = B_neutral - B_ion in MeV.
ElectronicInputs carries uncertainties and shared-input covariance.
Spreadsheet and online-table readers use the optional data dependencies.
"""

from __future__ import annotations

import math
import operator
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, SupportsIndex

import numpy as np

from .constants import ELEMENT_Z
from .processing import parse_ion_label, to_float_locale


def Read_AME_barion() -> list[dict[str, Any]]:
    """Return AME2020 records with element, Z, A, mass excess and error in keV.

    The IAEA mass table is cached locally after its first download.
    """
    from .ame import read_ame_table

    data_list = []
    for row in read_ame_table():
        data_list.append(
            {
                "Element": str(row[5]) + str(row[6]),
                "Z": int(row[4]),
                "A": int(row[5]),
                "ME (keV)": float(row[8]),
                "ME_error (keV)": float(row[9]),
            }
        )
    return data_list


def Read_elbien_file(file_path: str | Path) -> list[dict[str, Any]]:
    """Read an ``ElBiEn_*.dat`` electron-binding-energy table.

    File format: whitespace-separated columns, ``#``-prefixed header/comment
    lines. Column 0 is ``Z``, column 1 is the neutral-atom total binding
    energy (``TotBEn``), and columns 2.. are the total binding energy for
    ions with 1, 2, 3, ... electrons remaining (stored under keys ``"1e-ion"``,
    ``"2e-ion"``, ...), all in eV as integers in the source file.
    """
    data_list = []
    with open(file_path) as f:
        lines = f.readlines()
    header_line = None
    for line in lines:
        if line.startswith("#"):
            if header_line is None:
                header_line = line
            continue
        if not line.strip():
            continue
        tokens = line.split()
        if len(tokens) < 2:
            continue
        entry: dict[str, Any] = {"Z": int(tokens[0]), "TotBEn": int(tokens[1])}
        for i in range(2, len(tokens)):
            entry[f"{i - 1}e-ion"] = int(float(tokens[i]))
        data_list.append(entry)
    return data_list


def GetAMEData(AME_data: list[dict[str, Any]], element: str) -> tuple[float, float, int, int]:
    """Look up ``(ME_MeV, ME_error_MeV, A, N)`` for ``element`` (e.g. ``"204Pt"``)."""
    for entry in AME_data:
        if str(element) == str(entry["Element"]):
            ME = float(entry["ME (keV)"]) / 1000.0
            MEerr = float(entry["ME_error (keV)"]) / 1000.0
            A = int(entry["A"])
            N = int(entry["A"]) - int(entry["Z"])
            return ME, MEerr, A, N
    raise KeyError(f"AME element {element!r} not found")


def _validate_electron_charge(Z: SupportsIndex, Q: SupportsIndex) -> tuple[int, int]:
    """Require integer atomic number and charge, with ``Z > 0`` and ``0 <= Q <= Z``."""
    if isinstance(Z, (bool, np.bool_)) or isinstance(Q, (bool, np.bool_)):
        raise ValueError("Z and Q must be integers, not booleans")
    try:
        z, q = operator.index(Z), operator.index(Q)
    except TypeError as exc:
        raise ValueError("Z and Q must be integers") from exc
    if z <= 0 or not 0 <= q <= z:
        raise ValueError(f"Require Z > 0 and 0 <= Q <= Z; got Z={z}, Q={q}")
    return z, q


def _ionization_energy_MeV(neutral_eV: float, residual_eV: float, Z: int, Q: int) -> float:
    """Validate total binding energies and return their difference in MeV."""
    if not math.isfinite(neutral_eV) or neutral_eV <= 0:
        raise ValueError(f"Invalid neutral total binding energy for Z={Z}: {neutral_eV}")
    if not math.isfinite(residual_eV) or residual_eV < 0 or residual_eV > neutral_eV:
        raise ValueError(f"Invalid residual-ion total binding energy for Z={Z}, Q={Q}: {residual_eV}")
    if Q < Z and residual_eV == 0:
        raise ValueError(f"Zero binding energy for a non-bare ion, Z={Z}, Q={Q}")
    return (neutral_eV - residual_eV) / 1e6


def GetBindingEnergy(elbien_data: list[dict[str, Any]], Z: SupportsIndex, Q: SupportsIndex) -> float:
    """Cumulative ionization energy ``I_Q`` (MeV) from an ElBiEn table.

    ``Q`` is the positive ion charge (electrons removed). The table columns
    count electrons *remaining*: ``I_Q = TotBEn - entry[f"{Z-Q}e-ion"]``.
    A bare ion (``Q == Z``) has no remaining electronic binding, so ``I_Q``
    equals ``TotBEn``. A neutral atom (``Q == 0``) requires no correction.

    Missing required data raise ``KeyError``; invalid charge states or
    nonphysical binding energies raise ``ValueError``. No uncertainty is
    returned or inferred from the table's precision.
    """
    Z, Q = _validate_electron_charge(Z, Q)
    if Q == 0:
        return 0.0
    match = next((entry for entry in elbien_data if int(entry["Z"]) == Z), None)
    if match is None:
        raise KeyError(f"No ElBiEn binding-energy data for Z={Z}")
    neutral_eV = float(match["TotBEn"])
    residual_eV = 0.0 if Q == Z else float(match[f"{Z - Q}e-ion"])
    return _ionization_energy_MeV(neutral_eV, residual_eV, Z, Q)


def build_data_from_ods(
    filename: str | Path,
    sheet_index: int,
    exclusion_list: Sequence[int] | None = None,
    unmeasured_list: Sequence[str] | None = None,
    observable: str = "frequency",
    harmonic: float | None = None,
    revname: str | Path | None = None,
    header: bool = True,
    verbose: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read a template-fit ODS table and build the legacy ``data`` array.

    Expected ODS columns: ``Label, mu_fit_Hz, mu_combined_sigma_Hz,
    rms_width_Hz, [width_err_Hz]``.

    Parameters
    ----------
    observable:
        ``"frequency"`` stores ``(f, sigma_f, width_f)`` [Hz] as the
        observable. ``"time"`` converts to revolution time via
        ``T = 1e12/f`` [ps] and propagates the frequency uncertainty
        through ``dT/df = -1e12/f**2`` (first-order/linear propagation).
    harmonic:
        If the ODS frequencies are still harmonic (not fundamental
        revolution) frequencies, divide by this harmonic number before any
        other conversion. Leave ``None`` if the ODS already stores
        fundamental frequencies.
    unmeasured_list:
        Species ids (e.g. ``"205Au78+"``) to mark as unknown (``ref="N"``)
        even though they otherwise look like ordinary rows; every other row
        is treated as a calibrant (``ref="Y"``).
    revname:
        If given, the built rows are additionally written out as a
        tab-separated text file at this path (legacy revolution-time-file
        format). The export contains the selected observable rows.

    Returns
    -------
    ``(data, aux)`` where ``data`` is the object-dtype legacy row array
    (``name_ame, Z, q, ref, x, x_err, width_x, width_x_err, isomer,
    name_full, species_id``) and ``aux`` carries the intermediate arrays
    (raw frequency/time series, parsed labels, ``x_label``/``x_unit``) for
    diagnostics and re-use.
    """
    import ezodf

    odsinfo = ezodf.opendoc(str(filename))
    sheet = odsinfo.sheets[sheet_index]
    exclusion_set = set(exclusion_list or [])
    start = 1 if header else 0

    names: list[str] = []
    f: list[float] = []
    fe: list[float] = []
    width: list[float] = []
    width_err: list[float] = []
    for row in range(start, sheet.nrows()):
        if row in exclusion_set:
            continue
        vals = [sheet[row, col].value if col < sheet.ncols() else None for col in range(5)]
        if vals[0] is None or str(vals[0]).strip() == "":
            continue
        names.append(str(vals[0]))
        f.append(to_float_locale(vals[1]))
        fe.append(to_float_locale(vals[2]))
        width.append(to_float_locale(vals[3]))
        width_err.append(to_float_locale(vals[4]) if vals[4] not in (None, "") else np.nan)

    f_arr = np.asarray(f, dtype=float)
    fe_arr = np.asarray(fe, dtype=float)
    width_arr = np.asarray(width, dtype=float)
    width_err_arr = np.asarray(width_err, dtype=float)
    if harmonic is not None:
        h = float(harmonic)
        f_arr = f_arr / h
        fe_arr = fe_arr / h
        width_arr = width_arr / h
        width_err_arr = width_err_arr / h

    parsed = np.array([parse_ion_label(x) for x in names], dtype=object)
    A_str = parsed[:, 0]
    isomer = parsed[:, 1]
    element = parsed[:, 2]
    charge = parsed[:, 3]
    name_ame = np.array([f"{A}{el}" for A, el in zip(A_str, element, strict=True)], dtype=object)
    name_full = np.array([f"{A}{iso}{el}" for A, iso, el in zip(A_str, isomer, element, strict=True)], dtype=object)
    species_id = np.array(
        [f"{A}{iso}{el}{q}+" for A, iso, el, q in zip(A_str, isomer, element, charge, strict=True)], dtype=object
    )
    ref = np.full(len(names), "Y", dtype="<U1")
    for bad in unmeasured_list or []:
        matches = np.where(species_id == bad)[0]
        if len(matches):
            ref[matches] = "N"
            if verbose:
                print(f"✓ Marked 'N' → {bad} indices {matches.tolist()}")
        elif verbose:
            print(f"⚠ Warning: could not find {bad}")

    observable_kind = str(observable).lower()
    if observable_kind.startswith("f"):
        x = f_arr.copy()
        xerr = fe_arr.copy()
        xwidth = width_arr.copy()
        xwidtherr = width_err_arr.copy()
        x_label = "frequency"
        x_unit = "Hz"
    elif observable_kind.startswith("t"):
        x = 1.0 / f_arr * 1e12
        xerr = fe_arr / (f_arr**2) * 1e12
        xwidth = width_arr / (f_arr**2) * 1e12
        xwidtherr = width_err_arr / (f_arr**2) * 1e12
        x_label = "revolution_time"
        x_unit = "ps"
    else:
        raise ValueError("observable must be 'frequency' or 'time'")

    Z = np.array([ELEMENT_Z.get(str(el), -1) for el in element], dtype=int)
    if np.any(Z < 0):
        missing = sorted({str(el) for el, z in zip(element, Z, strict=True) if z < 0})
        raise ValueError(f"Unknown element(s): {missing}")

    data = []
    for i in range(len(names)):
        data.append(
            [
                name_ame[i],
                int(Z[i]),
                int(charge[i]),
                ref[i],
                float(x[i]),
                float(xerr[i]),
                float(xwidth[i]),
                float(xwidtherr[i]) if np.isfinite(xwidtherr[i]) else np.nan,
                isomer[i],
                name_full[i],
                species_id[i],
            ]
        )
    data = np.asarray(data, dtype=object)

    if revname:
        with open(revname, "w") as fout:
            fout.write("\n".join(["\t".join(map(str, row)) for row in data]) + "\n")
        if verbose:
            print(f"Wrote {revname}")

    aux = dict(
        names_latex=np.asarray(names, dtype=object),
        f=f_arr,
        fe=fe_arr,
        width_f=width_arr,
        width_f_err=width_err_arr,
        x=x,
        xerr=xerr,
        xwidth=xwidth,
        xwidtherr=xwidtherr,
        x_label=x_label,
        x_unit=x_unit,
        A=A_str,
        isomer=isomer,
        element=element,
        charge=charge,
        name_ame=name_ame,
        name_full=name_full,
        species_id=species_id,
        ref=ref,
        Z=Z,
    )
    return data, aux


# Optional NIST total-binding source. Retrieval needs network access only
# when the selected cache is unavailable.


def _retrieve_electron_binding_energies() -> dict[str, Any]:
    """Retrieve total binding energies and marginal errors from NIST ASD."""
    import requests
    from bs4 import BeautifulSoup

    url = "https://physics.nist.gov/cgi-bin/ASD/ie.pl"
    data = {
        "spectra": "H-Og",
        "submit": "Retrieve Data",
        "units": "1",
        "format": "0",
        "order": "0",
        "at_num_out": "on",
        "ion_charge_out": "on",
        "e_out": "1",
        "unc_out": "on",
    }
    response = requests.post(url, data=data)
    if response.status_code != 200:
        raise RuntimeError(f"NIST ASD request failed with status {response.status_code}")
    soup = BeautifulSoup(response.content, "html.parser")
    table = soup.find("table", {"border": "1"})
    if table is None:
        raise RuntimeError("Binding-energy table not found in NIST ASD response")
    data_dict: dict[str, Any] = {}
    for row in table.find_all("tr")[1:]:
        cells = row.find_all("td")
        if len(cells) != 4:
            continue
        at_num, ion_charge, binding_energy, uncertainty = (c.text.strip() for c in cells)
        data_dict.setdefault(at_num, []).append(
            {"Ion Charge": ion_charge, "Binding Energy (eV)": binding_energy, "Uncertainty (eV)": uncertainty}
        )
    return data_dict


def _save_electron_binding_energies(data_dict: dict[str, Any], filename: str | Path) -> None:
    npz_data = {key: np.array(value) for key, value in data_dict.items()}
    np.savez(filename, **npz_data)  # pyright: ignore[reportArgumentType]  # dynamic str keys, not the allow_pickle kwarg


def _load_electron_binding_energies(filename: str | Path) -> dict[str, Any]:
    data = np.load(filename, allow_pickle=True)
    return {key: data[key].tolist() for key in data}


def _get_electron_binding_energies(filename: str | Path | None = None) -> dict[str, Any]:
    """Read the NIST cache; retrieve and save it only if absent.

    The default is the packaged cache, which supports offline use.
    Custom NPZ caches must come from a trusted source because they contain
    serialized Python objects.
    """
    if filename is None:
        filename = Path(__file__).parent / "data" / "electron_binding_energies.npz"
    if os.path.exists(filename):
        return _load_electron_binding_energies(filename)
    data_dict = _retrieve_electron_binding_energies()
    _save_electron_binding_energies(data_dict, filename)
    return data_dict


def fetch_nist_electron_binding_table(cache_path: str | Path | None = None) -> dict[str, Any]:
    """Return the NIST total-binding table from a supplied or bundled cache.

    CalibrationConfig(elbien_source="nist") selects this input for a workflow.
    """
    return _get_electron_binding_energies(cache_path)


def _convert_binding_energy_to_float(value: str) -> float:
    """Parse a NIST ASD binding-energy cell (``"[1 030.1085]"`` -> ``1030.1085``)."""
    return float(value.strip("[]()").replace(" ", ""))


def GetBindingEnergyFromNIST(nist_data: dict[str, Any], Z: SupportsIndex, Q: SupportsIndex) -> float:
    """Cumulative ionization energy ``I_Q`` (MeV) from NIST total binding energies.

    Entries are selected by ``Ion Charge`` (not list position). Subtract
    the total binding at charge ``Q`` from the neutral (charge 0) total.
    For bare ions use zero residual binding; NIST need not supply a bare
    row. Neutral atoms have zero correction.

    Missing required data raise ``KeyError``; invalid charge states or
    nonphysical binding energies raise ``ValueError``. The cached NIST
    uncertainties are not propagated; doing so requires their covariance.
    """
    Z, Q = _validate_electron_charge(Z, Q)
    if Q == 0:
        return 0.0
    entries = nist_data.get(str(Z))
    if not entries:
        raise KeyError(f"No NIST binding-energy data for Z={Z}")
    by_charge = {}
    for entry in entries:
        charge = int(entry["Ion Charge"])
        if charge in by_charge:
            raise ValueError(f"Duplicate NIST binding-energy entry for Z={Z}, Q={charge}")
        by_charge[charge] = entry
    if 0 not in by_charge or (Q != Z and Q not in by_charge):
        raise KeyError(f"Missing NIST neutral or charge-Q binding-energy entry for Z={Z}, Q={Q}")
    neutral_eV = _convert_binding_energy_to_float(by_charge[0]["Binding Energy (eV)"])
    residual_eV = 0.0 if Q == Z else _convert_binding_energy_to_float(by_charge[Q]["Binding Energy (eV)"])
    return _ionization_energy_MeV(neutral_eV, residual_eV, Z, Q)

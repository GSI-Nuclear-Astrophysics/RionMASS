"""Validated import, calibration and replay settings shared by all interfaces."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .electronic import ElectronicInputs
from .reference_masses import normalize_reference_mass_overrides


def parse_isomer_energy_spec(items: list[str]) -> dict[str, Any]:
    """Parse NAME=ENERGY_MEV[,UNCERTAINTY_MEV] for both CLI and GUI inputs."""
    out: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected NAME=ENERGY_MEV[,UNCERTAINTY_MEV], got {item!r}")
        name, value = item.split("=", 1)
        parts = value.split(",")
        if len(parts) == 1:
            out[name.strip()] = float(parts[0])
        elif len(parts) == 2:
            out[name.strip()] = (float(parts[0]), float(parts[1]))
        else:
            raise ValueError(f"Too many commas in isomer energy value: {value!r}")
    return out


@dataclass(slots=True)
class ODSImportConfig:
    """Inputs for :func:`rionmass.io.build_data_from_ods`.

    Attributes
    ----------
    filename:
        Path to the template-fit ODS table.
    sheet_index:
        Zero-based sheet index within the ODS file.
    exclusion_list:
        Row indices (within the sheet, before header adjustment) to skip.
    unmeasured_list:
        Species ids (e.g. ``"205Au78+"``) to flag as unknown (``ref="N"``);
        every other row becomes a calibrant.
    observable:
        ``"frequency"`` (default) or ``"time"``.
    harmonic:
        Divide the ODS frequency-like columns by this harmonic number before
        conversion. ``None`` (default) if the ODS already stores fundamental
        revolution frequencies.
    revname:
        Optional tab-separated export of the imported observable rows.
    header:
        Whether the sheet's first row is a header row to skip.
    """

    filename: str | Path
    sheet_index: int
    exclusion_list: tuple[int, ...] = ()
    unmeasured_list: tuple[str, ...] = ()
    observable: str = "frequency"
    harmonic: float | None = None
    revname: str | Path | None = None
    header: bool = True

    def __post_init__(self) -> None:
        if self.observable.lower()[:1] not in ("f", "t"):
            raise ValueError(f"observable must be 'frequency' or 'time', got {self.observable!r}")
        if self.sheet_index < 0:
            raise ValueError(f"sheet_index must be >= 0, got {self.sheet_index}")


@dataclass(slots=True)
class CalibrationConfig:
    """Inputs for :func:`rionmass.fitting.fit_calibration` and the uncertainty stages.

    Attributes
    ----------
    elbien_source:
        Source and indexing convention for the electronic correction:

        - ``"file"`` (default): a Rodrigues-et-al.-sourced ``ElBiEn_*.dat``
          table, read via :func:`rionmass.io.Read_elbien_file`. Requires
          :attr:`elbien_path`. Columns count bound electrons; the lookup
          subtracts the residual-ion binding from the neutral total.
        - ``"nist"``: scraped from ``physics.nist.gov`` (cached locally;
          the bundled total-binding cache supports offline use), via
          :func:`rionmass.io.fetch_nist_electron_binding_table`. Entries
          label ion charge; the lookup subtracts the binding at the
          requested charge from the neutral total.
    elbien_path:
        Path to the ``ElBiEn_*.dat`` table. Required when
        ``elbien_source="file"``; ignored otherwise.
    nist_cache_path:
        Optional override for where the NIST table is cached/read from
        (defaults to the package's bundled cache). Ignored when
        ``elbien_source="file"``.
    isomer_excitation_energy:
        ``{species_id_or_display_name: energy_MeV}`` or
        ``{...: (energy_MeV, uncertainty_MeV)}``, for isomeric calibrants or
        unknowns.
    reference_mass_overrides:
        Ground-state neutral atomic mass excesses and 1-sigma uncertainties,
        in keV: {"206Hg": (ME_keV, sigma_keV[, source])}. These replace AME
        calibration inputs for retained references, across charge states.
        Isomer excitation energies are added separately. Original
        AME values are preserved for comparison and saved-run replay.
    p:
        Number of polynomial terms (``degree = p - 1``).
    iterationMax:
        Maximum iterations for the x-error-propagation loop in
        :func:`rionmass.fitting.fit_calibration`.
    tol:
        Convergence tolerance for that iteration.
    compare_orders:
        Polynomial-order values to try in
        :func:`rionmass.uncertainty.compare_polynomial_orders` (diagnostic
        only; :attr:`p` is what is actually fit).
    electronic_inputs, electronic_inputs_path:
        Explicit electronic-input model or its JSON file. Overrides the
        selected binding table; supply at most one. Values/uncertainties
        are in MeV, covariance in MeV**2. Missing uncertainties remain unknown.
    electronic_assume_independent:
        Explicitly assume independent primitive binding-energy errors when
        constructing a model from a table (default False). This does not
        assign missing marginal uncertainties. It is an approximation for
        NIST total binding energies, which can share underlying inputs.
        With explicit electronic inputs, set their covariance/model instead.
    electronic_method:
        "fixed_weights" (analytic Jacobian at final calibration weights) or
        "refit" (finite differences of the full iterative calibration).
    loo_anchor_species:
        Explicit reference species excluded from the LOO scatter estimate.
        They always remain in the final calibration with their input errors.
        Select from geometry/independent reference knowledge, not residual size.
    loo_anchor_policy:
        "retain_in_training" (default): retain anchors when predicting other
        references. "remove_from_validation": omit anchors from those training
        sets too. Anchor holdouts are separate diagnostics under either policy.
    electronic_step_MeV, electronic_workers:
        Finite-difference step and thread count for the refit method.
    """

    elbien_source: str = "file"
    elbien_path: str | Path | None = None
    nist_cache_path: str | Path | None = None
    isomer_excitation_energy: dict[str, Any] = field(default_factory=dict)
    p: int = 3
    iterationMax: int = 100
    tol: float = 1e-14
    compare_orders: tuple[int, ...] = (2, 3, 4, 5)
    electronic_inputs: ElectronicInputs | None = None
    electronic_inputs_path: str | Path | None = None
    electronic_assume_independent: bool = False
    electronic_method: str = "fixed_weights"
    electronic_step_MeV: float = 0.001
    electronic_workers: int = 1
    loo_anchor_species: tuple[str, ...] = ()
    loo_anchor_policy: str = "retain_in_training"

    reference_mass_overrides: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.reference_mass_overrides = normalize_reference_mass_overrides(self.reference_mass_overrides)
        if self.loo_anchor_policy not in ("retain_in_training", "remove_from_validation"):
            raise ValueError("loo_anchor_policy must be retain_in_training or remove_from_validation")
        if isinstance(self.loo_anchor_species, str) or any(
            not isinstance(v, str) or not v.strip() or v != v.strip() for v in self.loo_anchor_species
        ):
            raise ValueError("loo_anchor_species must contain nonempty species ids without surrounding whitespace")
        if len(set(self.loo_anchor_species)) != len(self.loo_anchor_species):
            raise ValueError("loo_anchor_species must not contain duplicate species ids")
        if self.elbien_source not in ("file", "nist"):
            raise ValueError(f"elbien_source must be 'file' or 'nist', got {self.elbien_source!r}")
        if self.electronic_inputs is not None and self.electronic_inputs_path is not None:
            raise ValueError("Supply electronic_inputs or electronic_inputs_path, not both")
        if (
            self.elbien_source == "file"
            and self.elbien_path is None
            and self.electronic_inputs is None
            and self.electronic_inputs_path is None
        ):
            raise ValueError("elbien_path is required when elbien_source='file'")
        if self.electronic_method not in ("fixed_weights", "refit"):
            raise ValueError("electronic_method must be fixed_weights or refit")
        if not math.isfinite(self.electronic_step_MeV) or self.electronic_step_MeV <= 0 or self.electronic_workers < 1:
            raise ValueError("Electronic differentiation step and workers must be positive")
        if self.electronic_assume_independent and (
            self.electronic_inputs is not None or self.electronic_inputs_path is not None
        ):
            raise ValueError(
                "Set the covariance/model in explicit electronic inputs, not electronic_assume_independent"
            )
        if self.p < 1:
            raise ValueError(f"p (polynomial terms) must be >= 1, got {self.p}")
        if self.iterationMax < 1:
            raise ValueError(f"iterationMax must be >= 1, got {self.iterationMax}")

"""Electronic correction inputs and correlated nuisance propagation.

Primitive energies are in MeV; their covariance is in MeV**2. A correction
is a linear combination of primitives: either B_neutral - B_ion, a sum of
successive ionization energies, or an explicitly supplied I_q. Unknown
variances/covariances are NaN, never an implicit zero. Nuclear-state rows
sharing (Z, q) share the same correction (electronic ground-state convention).

The nominal fit remains conditional on these external inputs. Electronic
covariance is propagated separately, so it is not counted again in the
conditional fit covariance. Independence from AME and frequency errors is
assumed; this module does not supply their cross-covariance.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import amu, me
from .fitting import fit_calibration
from .io import GetBindingEnergy, GetBindingEnergyFromNIST, _convert_binding_energy_to_float, _validate_electron_charge
from .models import IDX_A, IDX_EBIND, IDX_Q, IDX_X, IDX_XERR, IDX_Z, CalibrationFit
from .processing import get_isomer_metadata

IonKey = tuple[int, int]


def propagate_partial_covariance(left: np.ndarray, covariance: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Compute left @ C @ right.T; unknown terms affect only nonzero sensitivities."""
    out = np.zeros((len(left), len(right)))
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            weights = np.outer(a, b)
            active = weights != 0
            terms = weights[active] * covariance[active]
            out[i, j] = np.sum(terms)
            if i == j and np.array_equal(left, right) and np.isfinite(out[i, j]):
                tolerance = 32 * np.finfo(float).eps * max(1, len(terms)) * np.sum(np.abs(terms))
                if out[i, j] < -tolerance:
                    raise ValueError("Electronic covariance gives a negative projected variance")
                out[i, j] = max(out[i, j], 0)
    return out


def _clean_roundoff(matrix: np.ndarray) -> np.ndarray:
    out = matrix.copy()
    if out.size:
        tolerance = 32 * np.finfo(float).eps * max(out.shape) * np.maximum(1, np.sum(np.abs(out), axis=1))
        out[np.abs(out) <= tolerance[:, None]] = 0
    return out


def _covariance_from_sigmas(
    labels: Sequence[str],
    sigmas: Sequence[float | None],
    assume_independent: bool = False,
    source_groups: Mapping[str, str] | None = None,
) -> np.ndarray:
    n = len(labels)
    c = np.full((n, n), np.nan)
    for i, s in enumerate(sigmas):
        if s is not None:
            if not np.isfinite(s) or s < 0:
                raise ValueError(f"Invalid electronic uncertainty for {labels[i]}: {s}")
            c[i, i] = s**2
    for i in range(n):
        for j in range(i):
            si, sj = sigmas[i], sigmas[j]
            if assume_independent:
                c[i, j] = c[j, i] = 0
            elif source_groups is not None and labels[i] in source_groups and labels[j] in source_groups:
                if source_groups[labels[i]] != source_groups[labels[j]]:
                    c[i, j] = c[j, i] = 0
                elif si is not None and sj is not None:
                    c[i, j] = c[j, i] = si * sj
    return c


@dataclass(frozen=True)
class ElectronicInputs:
    """A linear electronic-input model with an explicit, possibly incomplete covariance.

    ``corrections = response @ values_MeV`` in ``ions`` order. ``source``
    records provenance and correlation assumptions. Input covariance may
    contain NaN for unknown entries. Known zeros must be explicitly supplied.
    ``with_covariance`` can replace missing information after an evaluation.
    """

    labels: tuple[str, ...]
    values_MeV: np.ndarray
    covariance_MeV2: np.ndarray
    ions: tuple[IonKey, ...]
    response: np.ndarray
    source: str

    def __post_init__(self) -> None:
        n = len(self.labels)
        values = np.array(self.values_MeV, dtype=float, copy=True)
        cov = np.array(self.covariance_MeV2, dtype=float, copy=True)
        response = np.array(self.response, dtype=float, copy=True)
        if len(set(self.labels)) != n or len(set(self.ions)) != len(self.ions):
            raise ValueError("Electronic primitive labels and ion keys must be unique")
        if values.shape != (n,) or cov.shape != (n, n) or response.shape != (len(self.ions), n):
            raise ValueError("Electronic input dimensions do not match labels/ions")
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(response)):
            raise ValueError("Electronic energies and response must be finite")
        if np.any(np.isinf(cov)) or not np.allclose(cov, cov.T, rtol=1e-12, atol=0, equal_nan=True):
            raise ValueError("Electronic covariance must be symmetric, finite or NaN")
        if np.any(np.diag(cov) < 0):
            raise ValueError("Electronic variances must be nonnegative")
        for i in range(n):
            if cov[i, i] == 0:
                if np.any(np.isfinite(cov[i]) & (cov[i] != 0)):
                    raise ValueError("A zero-variance input cannot have nonzero covariance")
                cov[i, :] = cov[:, i] = 0
        for i in range(n):
            for j in range(i):
                if np.all(np.isfinite([cov[i, i], cov[j, j], cov[i, j]])):
                    if abs(cov[i, j]) > np.sqrt(cov[i, i] * cov[j, j]) * (1 + 1e-12):
                        raise ValueError("Electronic covariance violates the correlation bound")
        known = np.flatnonzero(np.isfinite(np.diag(cov)))
        known_cov = cov[np.ix_(known, known)]
        if len(known) and np.all(np.isfinite(known_cov)):
            tolerance = 1e-12 * max(float(np.max(np.diag(known_cov))), np.finfo(float).tiny)
            if np.min(np.linalg.eigvalsh(known_cov)) < -tolerance:
                raise ValueError("Electronic covariance must be positive semidefinite")
        for z, q in self.ions:
            _validate_electron_charge(z, q)
        for array in (values, cov, response):
            array.setflags(write=False)
        object.__setattr__(self, "values_MeV", values)
        object.__setattr__(self, "covariance_MeV2", cov)
        object.__setattr__(self, "response", response)

    def correction(self, Z: int, Q: int) -> float:
        """Return I_q in MeV; require the requested ion in this model."""
        key = _validate_electron_charge(Z, Q)
        try:
            index = self.ions.index(key)
        except ValueError as exc:
            raise KeyError(f"No electronic correction for Z={Z}, Q={Q}") from exc
        return float(self.response[index] @ self.values_MeV)

    def sensitivities(self, rows: Sequence[Sequence[Any]]) -> np.ndarray:
        result = np.zeros((len(rows), len(self.labels)))
        for i, row in enumerate(rows):
            key = (int(row[IDX_Z]), int(row[IDX_Q]))
            self.correction(*key)
            result[i] = self.response[self.ions.index(key)]
            if not np.isclose(result[i] @ self.values_MeV, float(row[IDX_EBIND]), rtol=0, atol=1e-12):
                raise ValueError(f"Electronic central values disagree with the mass row for {key}")
        return result

    def with_covariance(self, covariance_MeV2: np.ndarray, source: str) -> ElectronicInputs:
        return replace(self, covariance_MeV2=covariance_MeV2, source=f"{self.source}; {source}")

    def primitive_table(self) -> pd.DataFrame:
        sigma = np.sqrt(np.diag(self.covariance_MeV2))
        return pd.DataFrame(
            {
                "primitive": self.labels,
                "energy_MeV": self.values_MeV,
                "sigma_MeV": sigma,
                "uncertainty_known": np.isfinite(sigma),
                "source": self.source,
            }
        )

    def correction_table(self) -> pd.DataFrame:
        cov = propagate_partial_covariance(self.response, self.covariance_MeV2, self.response)
        sigma = np.sqrt(np.maximum(np.diag(cov), 0))
        return pd.DataFrame(
            {
                "Z": [k[0] for k in self.ions],
                "q": [k[1] for k in self.ions],
                "I_q_MeV": self.response @ self.values_MeV,
                "sigma_MeV": sigma,
                "uncertainty_known": np.isfinite(sigma),
                "source": self.source,
            }
        )

    @classmethod
    def from_corrections(
        cls,
        corrections_MeV: Mapping[IonKey, float],
        sigmas_MeV: Mapping[IonKey, float | None] | None = None,
        covariance_MeV2: np.ndarray | None = None,
        assume_independent: bool = False,
        source: str = "user-supplied I_q",
    ) -> ElectronicInputs:
        ions = tuple(sorted(corrections_MeV))
        if any(
            not np.isfinite(value) or value < 0 or (key[1] == 0 and value != 0)
            for key, value in corrections_MeV.items()
        ):
            raise ValueError("I_q must be finite and nonnegative, and zero for neutral atoms")
        if covariance_MeV2 is not None and assume_independent:
            raise ValueError("Supply explicit covariance or an independence assumption, not both")
        labels = tuple(f"I(Z={z},q={q})" for z, q in ions)
        sigmas = [None if sigmas_MeV is None else sigmas_MeV.get(key) for key in ions]
        covariance = (
            _covariance_from_sigmas(labels, sigmas, assume_independent) if covariance_MeV2 is None else covariance_MeV2
        )
        if assume_independent:
            source += "; primitive errors assumed independent"
        return cls(labels, np.array([corrections_MeV[k] for k in ions]), covariance, ions, np.eye(len(ions)), source)


def _optional_sigma(value: Any) -> float | None:
    if value is None or str(value).strip() in ("", "--", "..."):
        return None
    result = _convert_binding_energy_to_float(str(value)) / 1e6
    if not np.isfinite(result) or result < 0:
        raise ValueError(f"Invalid NIST uncertainty: {value}")
    return result


def _binding_total_MeV(table: Any, z: int, charge: int, source: str) -> tuple[float, float | None]:
    if source == "file":
        row = next(r for r in table if int(r["Z"]) == z)
        return float(row["TotBEn"] if charge == 0 else row[f"{z - charge}e-ion"]) / 1e6, None
    entry = next(r for r in table[str(z)] if int(r["Ion Charge"]) == charge)
    return (
        _convert_binding_energy_to_float(entry["Binding Energy (eV)"]) / 1e6,
        _optional_sigma(entry.get("Uncertainty (eV)")),
    )


def binding_energy_inputs(
    table: Any,
    ions: Sequence[IonKey],
    source: str = "file",
    assume_independent: bool = False,
) -> ElectronicInputs:
    """Build B_neutral - B_ion models from ElBiEn or NIST *total* binding tables.

    ElBiEn has no quoted uncertainties. NIST marginal uncertainties are
    retained, but cross-covariances default to unknown. Independence is an
    explicit approximation, not inferred from different table rows.
    """
    if source not in ("file", "nist"):
        raise ValueError("Binding-energy source must be file or nist")
    keys = tuple(sorted(set(_validate_electron_charge(*key) for key in ions)))
    energies: dict[str, float] = {}
    sigmas: dict[str, float | None] = {}
    coefficients: list[dict[str, float]] = []
    for z, q in keys:
        if q == 0:
            coefficients.append({})
            continue
        if source == "file":
            GetBindingEnergy(table, z, q)
        else:
            GetBindingEnergyFromNIST(table, z, q)

        coeff = {}
        for charge, sign in ((0, 1.0), (q, -1.0)):
            if charge == z:
                continue
            label = f"B(Z={z},n={z - charge})"
            energies[label], sigmas[label] = _binding_total_MeV(table, z, charge, source)
            coeff[label] = sign
        coefficients.append(coeff)
    labels = tuple(sorted(energies))
    response = np.array([[c.get(label, 0.0) for label in labels] for c in coefficients]).reshape(len(keys), len(labels))
    covariance = _covariance_from_sigmas(labels, [sigmas[k] for k in labels], assume_independent)
    provenance = f"{source} total binding energies; unspecified covariance is unknown"
    if assume_independent:
        provenance += "; primitive errors assumed independent"
    return ElectronicInputs(labels, np.array([energies[k] for k in labels]), covariance, keys, response, provenance)


def ionization_energy_inputs(
    table: Mapping[str, Sequence[Mapping[str, Any]]],
    ions: Sequence[IonKey],
    covariance_MeV2: np.ndarray | None = None,
    source_groups: Mapping[str, str] | None = None,
    assume_independent: bool = False,
) -> ElectronicInputs:
    """Sum individual NIST-style ionization energies for charges 0 through q-1.

    Each entry must contain ``Ion Charge`` and ``Ionization Energy (eV)``;
    ``Uncertainty (eV)`` is optional. The table must contain individual
    ionization energies, not total binding energies. Primitive labels are
    ``IE(Z=...,q=...)``, lexically sorted. Explicit covariance is preferred.
    Optional source groups model full correlation within each supplied group
    and independence between supplied groups, matching NIST's sum convention.
    This is a stated nuisance model, not recovered covariance information.
    """
    if assume_independent and source_groups is not None:
        raise ValueError("Source groups and independent primitive errors are alternative models")
    if covariance_MeV2 is not None and (assume_independent or source_groups is not None):
        raise ValueError("Supply explicit covariance or a correlation model, not both")
    keys = tuple(sorted(set(_validate_electron_charge(*key) for key in ions)))
    energies, sigmas, coefficients = {}, {}, []
    for z, q in keys:
        by_charge = {}
        for r in table.get(str(z), []):
            charge = int(r["Ion Charge"])
            if charge in by_charge:
                raise ValueError(f"Duplicate ionization-energy entry for Z={z}, Q={charge}")
            by_charge[charge] = r
        coeff = {}
        for charge in range(q):
            if charge not in by_charge:
                raise KeyError(f"Missing ionization energy for Z={z}, Q={charge}")
            entry = by_charge[charge]
            value = _convert_binding_energy_to_float(entry["Ionization Energy (eV)"]) / 1e6
            if not np.isfinite(value) or value <= 0:
                raise ValueError("Individual ionization energies must be positive and finite")
            label = f"IE(Z={z},q={charge})"
            energies[label], sigmas[label] = value, _optional_sigma(entry.get("Uncertainty (eV)"))
            coeff[label] = 1.0
        coefficients.append(coeff)
    labels = tuple(sorted(energies))
    response = np.array([[c.get(k, 0.0) for k in labels] for c in coefficients]).reshape(len(keys), len(labels))
    covariance = (
        _covariance_from_sigmas(labels, [sigmas[k] for k in labels], assume_independent, source_groups)
        if covariance_MeV2 is None
        else covariance_MeV2
    )
    source = "individual ionization energies; I_q=sum(IE_0,...,IE_(q-1))"
    if source_groups is not None:
        source += "; supplied source groups: full within-group correlation, independent between groups"
    if assume_independent:
        source += "; primitive errors assumed independent"
    return ElectronicInputs(labels, np.array([energies[k] for k in labels]), covariance, keys, response, source)


def read_electronic_inputs(path: str | Path, ions: Sequence[IonKey]) -> ElectronicInputs:
    """Read explicit corrections or individual ionization energies from a JSON file.

    Formats and units are documented in README.md. No network access or
    uncertainty estimation is performed. JSON null covariance entries mean unknown.
    """
    spec = json.loads(Path(path).read_text())
    covariance = spec.get("covariance_MeV2")
    if covariance is not None:
        covariance = np.asarray(covariance, dtype=float)
    if spec.get("format", "corrections") == "ionization_energies":
        result = ionization_energy_inputs(
            spec["table"], ions, covariance, spec.get("source_groups"), spec.get("assume_independent", False)
        )
        return replace(result, source=f"{result.source}; {spec.get('source', 'explicit ionization-energy JSON')}")
    if spec.get("format", "corrections") != "corrections":
        raise ValueError("Unknown electronic-input JSON format")
    entries = spec["corrections"]
    keys = [_validate_electron_charge(r["Z"], r["q"]) for r in entries]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate electronic correction in JSON")
    if covariance is not None:
        if covariance.shape != (len(keys), len(keys)):
            raise ValueError("JSON covariance must match the correction entries")
        order = sorted(range(len(keys)), key=lambda i: keys[i])
        covariance = covariance[np.ix_(order, order)]
    result = ElectronicInputs.from_corrections(
        {k: r["I_q_MeV"] for k, r in zip(keys, entries, strict=True)},
        {k: r.get("sigma_MeV") for k, r in zip(keys, entries, strict=True)},
        covariance,
        spec.get("assume_independent", False),
        spec.get("source", "explicit electronic-input JSON"),
    )
    for key in ions:
        result.correction(*key)
    return result


def _row_labels(rows: Sequence[Sequence[Any]]) -> list[str]:
    names = [get_isomer_metadata(r)[2] for r in rows]
    return [name if names.count(name) == 1 else f"{name}#{i}" for i, name in enumerate(names)]


@dataclass
class ElectronicPropagation:
    """Electronic covariance components, separate from conditional fit/measurement errors."""

    inputs: ElectronicInputs
    mass_jacobian_keV_per_MeV: np.ndarray
    coefficient_jacobian: np.ndarray
    mass_covariance_keV2: pd.DataFrame
    coefficient_covariance: pd.DataFrame
    mass_coefficient_covariance: pd.DataFrame
    conditional_mass_covariance_keV2: pd.DataFrame
    total_mass_covariance_keV2: pd.DataFrame
    method: str


def propagate_electronic_uncertainty(
    fit: CalibrationFit,
    references: Sequence[Sequence[Any]],
    unknowns: Sequence[Sequence[Any]],
    inputs: ElectronicInputs,
    extra_sys_keV: float = 0,
    method: str = "fixed_weights",
    iterationMax: int = 100,
    tol: float = 1e-14,
    step_MeV: float = 0.001,
    workers: int = 1,
) -> ElectronicPropagation:
    """Propagate shared electronic nuisances through calibration and atomic-mass recovery.

    At fixed final fit weights, J_beta=R J_ref/(q_ref*u) and
    J_mass=1000*[diag(q_unknown*u) V J_beta-J_unknown]. ``refit`` instead
    differentiates the complete iterative fit using centered finite differences
    (step in MeV); weights and coefficients are recomputed, x0/xscale retained.
    The latter is a local numerical approximation, check step-size stability.
    Electronic inputs are assumed independent of frequency/AME/isomer errors.
    The conditional mass covariance assumes independent unknown frequency errors
    and an independent per-mass LOO extra term; it is not an all-source covariance.
    """
    if method not in ("fixed_weights", "refit"):
        raise ValueError("Electronic propagation method must be fixed_weights or refit")
    if not np.isfinite(step_MeV) or step_MeV <= 0 or workers < 1:
        raise ValueError("Electronic differentiation step and workers must be positive")
    if len(references) != len(fit.x) or [get_isomer_metadata(r)[2] for r in references] != fit.labels:
        raise ValueError("References must match the fit's order and labels")
    jr, ju = inputs.sensitivities(references), inputs.sensitivities(unknowns)
    qref = np.array([float(r[IDX_Q]) for r in references])
    qu = np.array([float(r[IDX_Q]) for r in unknowns])
    xu = np.array([float(r[IDX_X]) for r in unknowns])
    sxu = np.array([float(r[IDX_XERR]) for r in unknowns])
    v = fit.design_vector(xu)
    # The response must not inherit reduced-chi-square covariance inflation.
    weights = 1 / np.clip(fit.sigma_eff_y, 1e-300, np.inf) ** 2
    response = np.linalg.pinv(fit.design.T @ (weights[:, None] * fit.design)) @ (fit.design.T * weights)
    jb = response @ (jr / (qref[:, None] * amu))
    if method == "refit":

        def differentiate(k):
            if not np.any(jr[:, k]):
                return np.zeros(fit.p)
            coeffs = []
            for sign in (1, -1):
                changed = [list(r) for r in references]
                for i, row in enumerate(changed):
                    row[IDX_EBIND] = float(row[IDX_EBIND]) + sign * step_MeV * jr[i, k]
                changed_fit = fit_calibration(
                    changed,
                    p=fit.p,
                    x0=fit.x0,
                    xscale=fit.xscale,
                    isomer_excitation_energy=fit.isomer_excitation_energy,
                    iterationMax=iterationMax,
                    tol=tol,
                )
                coeffs.append(changed_fit.coeff)
            return (coeffs[0] - coeffs[1]) / (2 * step_MeV)

        if workers == 1:
            derivatives = list(map(differentiate, range(len(inputs.labels))))
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                derivatives = list(pool.map(differentiate, range(len(inputs.labels))))
        jb = np.array(derivatives).T.reshape(fit.p, len(inputs.labels))
    jm = 1000 * _clean_roundoff((qu * amu)[:, None] * (v @ jb) - ju)
    cm = propagate_partial_covariance(jm, inputs.covariance_MeV2, jm)
    cb = propagate_partial_covariance(jb, inputs.covariance_MeV2, jb)
    cmb = propagate_partial_covariance(jm, inputs.covariance_MeV2, jb)
    conditional = (qu * amu * 1000)[:, None] * (v @ fit.cov @ v.T) * (qu * amu * 1000)[None, :]
    derivative = fit.derivative_vector_dx(xu) @ fit.coeff
    conditional += np.diag((qu * amu * 1000 * derivative * sxu) ** 2 + extra_sys_keV**2)
    names, coeff_names = pd.Index(_row_labels(unknowns)), pd.Index([f"a{k}" for k in range(fit.p)])
    return ElectronicPropagation(
        inputs,
        jm,
        jb,
        pd.DataFrame(cm, index=names, columns=names),
        pd.DataFrame(cb, index=coeff_names, columns=coeff_names),
        pd.DataFrame(cmb, index=names, columns=coeff_names),
        pd.DataFrame(conditional, index=names, columns=names),
        pd.DataFrame(conditional + cm, index=names, columns=names),
        method,
    )


def electronic_mass_trials(
    fit: CalibrationFit,
    references: Sequence[Sequence[Any]],
    unknowns: Sequence[Sequence[Any]],
    inputs: ElectronicInputs,
    replicates: int = 5000,
    seed: int | None = None,
    workers: int = 1,
    iterationMax: int = 100,
    tol: float = 1e-14,
) -> np.ndarray:
    """Parametric Gaussian electronic-input trials with a shared draw and full refit.

    Requires complete PSD input covariance. Returns state mass excesses in
    keV, shape (replicates, unknowns), conditional on other inputs. This is
    an electronic component/linearity check, not an end-to-end bootstrap.
    Draws are generated before threading, so seed results do not depend on
    workers. Non-Gaussian electronic-input laws require a different sampler.
    """
    if replicates < 2 or workers < 1 or not np.all(np.isfinite(inputs.covariance_MeV2)):
        raise ValueError("Trials need >=2 replicates, positive workers and complete electronic covariance")
    if len(references) != len(fit.x) or [get_isomer_metadata(r)[2] for r in references] != fit.labels:
        raise ValueError("References must match the fit's order and labels")
    jr, ju = inputs.sensitivities(references), inputs.sensitivities(unknowns)
    draws = np.random.default_rng(seed).multivariate_normal(
        np.zeros(len(inputs.labels)), inputs.covariance_MeV2, size=replicates, check_valid="raise"
    )
    v = fit.design_vector([r[IDX_X] for r in unknowns])
    q = np.array([r[IDX_Q] for r in unknowns], dtype=float)
    a = np.array([r[IDX_A] for r in unknowns], dtype=float)
    ionization = np.array([r[IDX_EBIND] for r in unknowns], dtype=float)

    def trial(draw):
        changed = [list(r) for r in references]
        for i, row in enumerate(changed):
            row[IDX_EBIND] = float(row[IDX_EBIND]) + jr[i] @ draw
        varied = fit_calibration(
            changed,
            p=fit.p,
            x0=fit.x0,
            xscale=fit.xscale,
            isomer_excitation_energy=fit.isomer_excitation_energy,
            iterationMax=iterationMax,
            tol=tol,
        )
        return 1000 * (q * amu * (v @ varied.coeff) - a * amu + q * me - ionization - ju @ draw)

    if workers == 1:
        return np.asarray(list(map(trial, draws)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return np.asarray(list(pool.map(trial, draws)))

"""Polynomial mass calibration in the coordinate z = (x - x0) / xscale.

Centering and scaling condition the fit without changing its physical
predictions. Iterative weighted least squares folds reference-frequency or
time errors into mass-to-charge errors through the current curve derivative.
Electronic-input covariance is propagated separately.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from .constants import amu, me
from .models import (
    IDX_A,
    IDX_EBIND,
    IDX_FLAG,
    IDX_ME_AME,
    IDX_MEERR_AME,
    IDX_N,
    IDX_Q,
    IDX_X,
    IDX_XERR,
    IDX_Z,
    CalibrationFit,
)
from .processing import ame_moq_for_nuclear_state, get_isomer_energy, get_isomer_metadata, safe_float


def _weighted_fit(
    design: np.ndarray, y: np.ndarray, sigma: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One weighted-least-squares solve; returns ``(coeff, cov, weights, R)``.

    ``R = cov @ design.T @ diag(weights)`` is the coefficient-response matrix used
    for the flow-of-information (Audi-style) significance and prediction
    influence — see :meth:`rionmass.models.CalibrationFit.prediction_influence`.
    """
    w = 1.0 / np.clip(sigma, 1e-300, np.inf) ** 2
    normal = design.T @ (w[:, None] * design)
    rhs = design.T @ (w * y)
    cov = np.linalg.pinv(normal)
    coeff = cov @ rhs
    R = cov @ design.T @ np.diag(w)
    return coeff, cov, w, R


def fit_calibration(
    Nuclei_Y: Sequence[Sequence[Any]],
    p: int = 3,
    x0: float | None = None,
    xscale: float | None = None,
    isomer_excitation_energy: dict[str, Any] | None = None,
    iterationMax: int = 100,
    tol: float = 1e-14,
    observable_label: str = "x",
    observable_unit: str = "arb.",
    inflate_cov_by_redchi2: bool = False,
    verbose: bool = False,
) -> CalibrationFit:
    """Fit the centered/scaled calibration polynomial to calibrant ions.

    Parameters
    ----------
    Nuclei_Y:
        Calibrant legacy rows (``ref == "Y"``), as built by
        :func:`rionmass.processing.build_nuclei_from_data`.
    p:
        Number of polynomial terms (``degree = p - 1``). Must satisfy
        ``len(Nuclei_Y) >= p``.
    x0, xscale:
        Centering/scaling constants. If omitted, ``x0`` is the
        inverse-reference-variance-weighted mean of the calibrant ``x`` values and
        ``xscale`` is half their range (``1.0`` if the range is zero).
    inflate_cov_by_redchi2:
        If ``True``, rescale the final covariance by ``red_chi2`` when it
        exceeds 1. Default False; the workflow estimates validation scatter
        separately through leave-one-out predictions.

    Returns
    -------
    A :class:`rionmass.models.CalibrationFit` with the fitted coefficients,
    covariance, correlation, per-calibrant residuals/pulls, and
    flow-of-information tables.

    This fit covariance is conditional on electronic corrections. Use
    :func:`rionmass.electronic.propagate_electronic_uncertainty` or the full
    workflow to propagate their separate, shared nuisance covariance.

    Raises
    ------
    ValueError
        If fewer calibrants than polynomial terms are supplied.
    """
    labels = []
    x, sx, y, sy = [], [], [], []
    for i, nuc in enumerate(Nuclei_Y):
        labels.append(get_isomer_metadata(nuc)[2])
        x.append(safe_float(nuc[IDX_X], i, IDX_X, "Nuclei_Y"))
        sx.append(safe_float(nuc[IDX_XERR], i, IDX_XERR, "Nuclei_Y"))
        moq, moqerr, *_ = ame_moq_for_nuclear_state(nuc, isomer_excitation_energy)
        y.append(moq)
        sy.append(moqerr)
    x = np.asarray(x, dtype=float)
    sx = np.asarray(sx, dtype=float)
    y = np.asarray(y, dtype=float)
    sy = np.asarray(sy, dtype=float)
    if len(x) < p:
        raise ValueError(f"Need at least p={p} calibrants; got {len(x)}")
    if x0 is None:
        x0 = float(np.average(x, weights=1.0 / np.clip(sy, 1e-300, np.inf) ** 2))
    if xscale is None:
        span = float(np.ptp(x))
        xscale = span / 2.0 if span > 0 else 1.0
    z = (x - x0) / xscale
    design = np.vstack([z**k for k in range(p)]).T
    sigma_eff = sy.copy()
    coeff = np.zeros(p)
    cov = np.eye(p)
    R = np.zeros((p, len(x)))
    last_chi2 = np.inf
    for iteration in range(int(iterationMax)):
        coeff_old = coeff.copy()
        coeff, cov, _w, R = _weighted_fit(design, y, sigma_eff)
        dydx = np.zeros_like(x)
        for k in range(1, p):
            dydx += coeff[k] * k * z ** (k - 1) / xscale
        sigma_eff = np.sqrt(sy**2 + (dydx * sx) ** 2)
        residual = design @ coeff - y
        chi2 = float(np.sum((residual / sigma_eff) ** 2))
        if verbose:
            print(f"iteration={iteration:03d} chi2={chi2:.8g} coeff={coeff}")
        converged_coeff = np.linalg.norm(coeff - coeff_old) <= tol * max(1.0, np.linalg.norm(coeff))
        converged_chi2 = abs(chi2 - last_chi2) <= max(tol, tol * abs(last_chi2))
        if converged_coeff and converged_chi2:
            break
        last_chi2 = chi2
    # Match the final coefficient covariance to the last effective weights.
    coeff, cov, _w, R = _weighted_fit(design, y, sigma_eff)
    yfit = design @ coeff
    residual = yfit - y
    chi2 = float(np.sum((residual / sigma_eff) ** 2))
    dof = int(len(x) - p)
    red_chi2 = chi2 / dof if dof > 0 else np.nan
    if inflate_cov_by_redchi2 and np.isfinite(red_chi2) and red_chi2 > 1.0:
        # Inflation changes uncertainty, not the estimator's response to its inputs.
        cov = cov * red_chi2
    H = design @ R
    significance = np.diag(H)
    flow_coeff = R.T * design
    pulls = residual / sigma_eff
    return CalibrationFit(
        p=p,
        degree=p - 1,
        x0=float(x0),
        xscale=float(xscale),
        coeff=coeff,
        cov=cov,
        chi2=chi2,
        dof=dof,
        red_chi2=float(red_chi2),
        x=x,
        sx=sx,
        y=y,
        sy=sy,
        sigma_eff_y=sigma_eff,
        design=design,
        yfit=yfit,
        residuals=residual,
        pulls=pulls,
        flow_coeff=flow_coeff,
        significance=significance,
        R=R,
        labels=labels,
        observable_label=observable_label,
        observable_unit=observable_unit,
        isomer_excitation_energy=isomer_excitation_energy,
    )


def predict_unknown_table(
    fit: CalibrationFit,
    Nuclei_N: Sequence[Sequence[Any]],
    extra_sys_keV: float = 0.0,
) -> tuple[list[list[Any]], pd.DataFrame, dict[str, pd.DataFrame]]:
    """Predict unknown-ion masses from a fitted calibration.

    Errors returned here are conditional on the row's electronic correction.
    The full workflow adds correlated electronic-input covariance separately.

    ``extra_sys_keV`` is the leave-one-out empirical systematic (see
    :func:`rionmass.uncertainty.estimate_extra_systematic_from_loo`), added
    in quadrature to the fit+observable statistical error to form
    ``ME_total_err_keV``. Pass ``0.0`` to report the statistical error alone.

    Returns ``(legacy_rows, detail_df, influence)``: ``legacy_rows`` contains
    one row per unknown but does not use the deprecated 10-column layout;
    ``detail_df`` is the documented, column-named result table; ``influence``
    maps each unknown's species id to its signed calibrant-influence table.
    """
    legacy_rows = []
    detail_rows = []
    influence = {}
    for i, nuc in enumerate(Nuclei_N):
        x = safe_float(nuc[IDX_X], i, IDX_X, "Nuclei_N")
        sx = safe_float(nuc[IDX_XERR], i, IDX_XERR, "Nuclei_N")
        pred = fit.predict_moq(x, sx)
        MoQ_N = float(pred["moq"])
        MoQ_curve_err = float(pred["moq_curve_err"])
        MoQ_x_err = float(pred["moq_x_err"])
        MoQ_stat_err = float(pred["moq_err"])
        Z_N = int(nuc[IDX_Z])
        A_N = int(nuc[IDX_A])
        _N_N = int(nuc[IDX_N])
        Q_N = float(nuc[IDX_Q])
        Flag = nuc[IDX_FLAG]
        EBindingEnergy_N = float(nuc[IDX_EBIND])
        _isomer, display_name, species_id = get_isomer_metadata(nuc)
        E_iso, E_iso_err = get_isomer_energy(nuc, fit.isomer_excitation_energy)
        Mass_ion_N = MoQ_N * Q_N * amu
        ME_N_state = Mass_ion_N - A_N * amu + Q_N * me - EBindingEnergy_N
        MEError_curve_keV = MoQ_curve_err * Q_N * amu * 1000.0
        MEError_x_keV = MoQ_x_err * Q_N * amu * 1000.0
        MEError_stat_keV = MoQ_stat_err * Q_N * amu * 1000.0
        MEError_total_keV = math.sqrt(MEError_stat_keV**2 + float(extra_sys_keV) ** 2)
        ME_N_AME_gs = float(nuc[IDX_ME_AME])
        MEError_N_AME_gs = float(nuc[IDX_MEERR_AME])
        ME_N_AME_state = ME_N_AME_gs + E_iso
        MEError_N_AME_state_keV = math.sqrt(MEError_N_AME_gs**2 + E_iso_err**2) * 1000.0
        Delta_ME_state_keV = (ME_N_state - ME_N_AME_state) * 1000.0
        ME_N_gs_equiv = ME_N_state - E_iso
        MEError_N_gs_equiv_keV = math.sqrt((MEError_total_keV / 1000.0) ** 2 + E_iso_err**2) * 1000.0
        Delta_ME_gs_equiv_keV = (ME_N_gs_equiv - ME_N_AME_gs) * 1000.0
        MoQ_N_AME_state, MoQError_N_AME_state, *_ = ame_moq_for_nuclear_state(nuc, fit.isomer_excitation_energy)
        legacy_rows.append(
            [
                display_name,
                Flag,
                Z_N,
                A_N,
                Q_N,
                (x - fit.x0) / fit.xscale,
                Delta_ME_state_keV,
                ME_N_state * 1000.0,
                MEError_total_keV,
                MEError_N_AME_state_keV,
                E_iso * 1000.0,
                ME_N_AME_state * 1000.0,
                ME_N_gs_equiv * 1000.0,
                MEError_N_gs_equiv_keV,
                Delta_ME_gs_equiv_keV,
                MoQ_N,
                MoQ_stat_err,
                MoQ_N_AME_state,
                MoQError_N_AME_state,
                species_id,
            ]
        )
        detail_rows.append(
            {
                "ion": display_name,
                "species_id": species_id,
                "flag": Flag,
                "Z": Z_N,
                "A": A_N,
                "q": Q_N,
                "x": x,
                "x_err": sx,
                "z": (x - fit.x0) / fit.xscale,
                "MoQ_fit": MoQ_N,
                "MoQ_curve_err": MoQ_curve_err,
                "MoQ_x_err": MoQ_x_err,
                "MoQ_stat_err": MoQ_stat_err,
                "ME_state_keV": ME_N_state * 1000.0,
                "ME_stat_err_keV": MEError_stat_keV,
                "ME_curve_err_keV": MEError_curve_keV,
                "ME_x_err_keV": MEError_x_keV,
                "ME_extra_sys_keV": float(extra_sys_keV),
                "ME_total_err_keV": MEError_total_keV,
                "ME_AME_state_keV": ME_N_AME_state * 1000.0,
                "ME_AME_state_err_keV": MEError_N_AME_state_keV,
                "Delta_ME_state_keV": Delta_ME_state_keV,
                "E_iso_keV": E_iso * 1000.0,
                "ME_gs_equiv_keV": ME_N_gs_equiv * 1000.0,
                "ME_gs_equiv_err_keV": MEError_N_gs_equiv_keV,
                "Delta_ME_gs_equiv_keV": Delta_ME_gs_equiv_keV,
            }
        )
        influence[species_id] = fit.prediction_influence(x)
    return legacy_rows, pd.DataFrame(detail_rows), influence


def MassCalibration(
    Nuclei_Y: Sequence[Sequence[Any]],
    Nuclei_N: Sequence[Sequence[Any]],
    OutputMatrixOrNot_Y: bool = False,
    p: int = 3,
    A0_Y: Sequence[float] | None = None,
    iterationMax: int = 100,
    tol: float = 1e-14,
    isomer_excitation_energy: dict[str, Any] | None = None,
    observable_label: str = "x",
    observable_unit: str = "arb.",
    extra_sys_keV: float = 0.0,
    return_details: bool = False,
    verbose: bool = False,
):
    """Compatibility wrapper returning (rows, coeff, chi2).

    return_details adds the fit, named result table and influence tables.
    A0_Y is accepted for older callers; weighted linear least squares needs
    no coefficient seed. See predict_unknown_table for the result-row format.
    """
    fit = fit_calibration(
        Nuclei_Y,
        p=p,
        isomer_excitation_energy=isomer_excitation_energy,
        iterationMax=iterationMax,
        tol=tol,
        observable_label=observable_label,
        observable_unit=observable_unit,
        verbose=verbose,
    )
    rows, detail, influence = predict_unknown_table(fit, Nuclei_N, extra_sys_keV=extra_sys_keV)
    if OutputMatrixOrNot_Y:
        from .plotting import print_calibration_report

        print_calibration_report(fit)
    if return_details:
        return rows, fit.coeff, fit.chi2, fit, detail, influence
    return rows, fit.coeff, fit.chi2

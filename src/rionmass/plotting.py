"""Diagnostic plots and the text calibration report.

Callers may apply a Matplotlib style before calling or pass a pre-styled
``ax``.

Requires the optional ``matplotlib`` dependency (the ``plot`` extra);
nothing in :mod:`rionmass.fitting` or :mod:`rionmass.uncertainty` imports
this module, so the core calibration engine works without it installed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from .models import IDX_X, IDX_XERR, CalibrationFit
from .processing import filter_ground_states, safe_float


def print_calibration_report(fit: CalibrationFit) -> None:
    """Print the fitted polynomial, its parameter/correlation tables, and the
    Audi-style flow-of-information table (whose significance column must sum
    to ``p``, printed as a self-check)."""
    print("\n=== Calibration polynomial ===")
    print(f"Observable: {fit.observable_label} [{fit.observable_unit}]")
    print("Polynomial: m/q = sum_k a_k z^k,  z = (x - x0)/xscale")
    print(f"x0 = {fit.x0:.12g} {fit.observable_unit}")
    print(f"xscale = {fit.xscale:.12g} {fit.observable_unit}")
    print(f"degree = {fit.degree}  p_terms = {fit.p}")
    print(f"chi2/dof = {fit.chi2:.6g}/{fit.dof} = {fit.red_chi2:.6g}")
    print("\nParameter table:")
    print(fit.parameter_table().to_string(index=False))
    print("\nCorrelation matrix:")
    print(fit.correlation_table().to_string())
    print("\nFlow/significance table:")
    print(fit.flow_table().to_string(index=False))
    print(f"Sum significance = {np.sum(fit.significance):.12g} (should equal p={fit.p})")


def plot_calibration_fit(fit: CalibrationFit, Nuclei_N: Sequence[Sequence[Any]] | None = None, ax=None):
    """Plot calibrant m/q vs. observable, the fitted curve + 1-sigma band, and unknowns."""
    import matplotlib.pyplot as plt

    if ax is None:
        _fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(fit.x, fit.y, yerr=fit.sigma_eff_y, fmt="o", label="calibrants")
    xx = np.linspace(np.min(fit.x), np.max(fit.x), 400)
    pred = fit.predict_moq(xx, 0.0)
    ax.plot(xx, pred["moq"], label=f"degree {fit.degree}")
    ax.fill_between(xx, pred["moq"] - pred["moq_curve_err"], pred["moq"] + pred["moq_curve_err"], alpha=0.2)
    if Nuclei_N:
        xu = np.array([safe_float(n[IDX_X]) for n in Nuclei_N])
        yu = fit.predict_moq(xu, np.array([safe_float(n[IDX_XERR]) for n in Nuclei_N]))["moq"]
        ax.plot(xu, yu, "s", label="unknowns")
    ax.set_xlabel(f"{fit.observable_label} [{fit.observable_unit}]")
    ax.set_ylabel("ionic m/q [u/e]")
    ax.legend()
    return ax


def plot_loo_residuals(loo_df: pd.DataFrame, with_extra: bool = False, ax=None):
    """Plot each calibrant's leave-one-out mass-excess residual with its LOO error bar."""
    import matplotlib.pyplot as plt

    if ax is None:
        _fig, ax = plt.subplots(figsize=(9, 4))
    reference_col = "Delta_ME_reference_keV" if "Delta_ME_reference_keV" in loo_df else "Delta_ME_state_keV"
    y = np.asarray(loo_df[reference_col], dtype=float)
    if with_extra and "loo_sigma_with_extra_keV" in loo_df:
        e = np.asarray(loo_df["loo_sigma_with_extra_keV"], dtype=float)
    else:
        e = np.asarray(loo_df["loo_sigma_no_extra_keV"], dtype=float)
    x = np.arange(len(loo_df))
    labels_col = "species_id" if "species_id" in loo_df.columns else "held_out_species_id"
    labels = loo_df[labels_col].astype(str)
    ax.axhline(0, color="k", lw=1)
    used = (
        loo_df["used_for_extra_scatter"].to_numpy(dtype=bool)
        if "used_for_extra_scatter" in loo_df
        else np.ones(len(loo_df), dtype=bool)
    )
    ax.errorbar(x[used], y[used], yerr=e[used], fmt="o", capsize=3, label="Used for validation scatter")
    if np.any(~used):
        ax.errorbar(x[~used], y[~used], yerr=e[~used], fmt="x", capsize=3, label="Anchor holdout: diagnostic only")
        ax.legend()
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right")
    custom = "reference_is_custom" in loo_df and bool(loo_df["reference_is_custom"].any())
    ax.set_ylabel("LOO ME_fit - ME_reference [keV]" if custom else "LOO ME_fit - ME_AME [keV]")
    ax.grid(True, alpha=0.3)
    return ax


def _empty_unknown_plot(ax, title):
    ax.set_title(title)
    ax.text(0.5, 0.5, "No finite unknown masses in this selection", ha="center", va="center", transform=ax.transAxes)
    return ax


def _measurement_errors(df, column):
    errors = np.asarray(df[column], dtype=float).copy() if column in df else np.full(len(df), np.nan)
    errors[errors < 0] = np.nan
    conditional = np.zeros(len(df), dtype=bool)
    if column == "ME_total_err_keV" and "ME_conditional_err_keV" in df:
        fallback = np.asarray(df["ME_conditional_err_keV"], dtype=float)
        conditional = ~np.isfinite(errors) & np.isfinite(fallback) & (fallback >= 0)
        errors[conditional] = fallback[conditional]
    return errors, conditional


def _available_errors(errors):
    # Matplotlib cannot determine the shape of an all-NaN error array.
    return errors if errors is not None and np.any(np.isfinite(errors)) else None


def _error_groups(errors, conditional):
    known = np.isfinite(errors)
    return (
        (known & ~conditional, "combined uncertainties"),
        (known & conditional, "conditional uncertainties"),
        (~known, "uncertainty unavailable"),
    )


def plot_unknown_mass_excess_vs_ame(
    unknown_df: pd.DataFrame,
    ax=None,
    label_col: str = "species_id",
    exp_col: str = "ME_state_keV",
    exp_err_col: str = "ME_total_err_keV",
    ame_col: str = "ME_AME_state_keV",
    ame_err_col: str = "ME_AME_state_err_keV",
    title: str = "Unknown mass excess: experiment vs AME",
    ground_only: bool = False,
):
    """Scatter measured vs. AME/reference mass excess for unknown ions.

    Finite masses are retained when their full uncertainty is unavailable.
    Default total errors fall back to explicitly labelled conditional errors
    where available; otherwise draw the central values without those bars.
    Set ``ground_only=True`` to drop isomeric rows (see
    :func:`rionmass.processing.filter_ground_states`).
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _fig, ax = plt.subplots(figsize=(6.5, 6.0))

    if unknown_df.empty:
        return _empty_unknown_plot(ax, title)
    df = filter_ground_states(unknown_df, ground_only=ground_only)

    x = np.asarray(df[ame_col], dtype=float)
    y = np.asarray(df[exp_col], dtype=float)

    xerr = np.asarray(df[ame_err_col], dtype=float) if ame_err_col in df else None
    yerr, conditional = _measurement_errors(df, exp_err_col)

    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        return _empty_unknown_plot(ax, title)
    x = x[finite]
    y = y[finite]
    xerr_plot = xerr[finite] if xerr is not None else None
    yerr_plot = yerr[finite]
    conditional = conditional[finite]
    labels = df.loc[finite, label_col].astype(str).to_numpy()

    for mask, description in _error_groups(yerr_plot, conditional):
        if np.any(mask):
            ax.errorbar(
                x[mask],
                y[mask],
                xerr=_available_errors(xerr_plot[mask] if xerr_plot is not None else None),
                yerr=_available_errors(yerr_plot[mask]),
                fmt="o",
                capsize=3,
                label=description,
            )

    lo = min(np.nanmin(x), np.nanmin(y))
    hi = max(np.nanmax(x), np.nanmax(y))
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    xx = np.linspace(lo - pad, hi + pad, 200)
    ax.plot(xx, xx, "--", lw=1, label="experiment = AME")

    for xi, yi, lab in zip(x, y, labels, strict=True):
        ax.annotate(lab, (xi, yi), xytext=(5, 5), textcoords="offset points", fontsize=9)

    ax.set_xlabel("AME/reference mass excess of same state [keV]")
    ax.set_ylabel("Measured mass excess [keV]")
    if ground_only:
        title += " — ground states only"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    return ax


def plot_unknown_mass_excess_residuals(
    unknown_df: pd.DataFrame,
    ax=None,
    label_col: str = "species_id",
    residual_col: str = "Delta_ME_state_keV",
    exp_err_col: str = "ME_total_err_keV",
    ame_err_col: str = "ME_AME_state_err_keV",
    title: str = "Unknown mass residuals relative to AME",
    ground_only: bool = False,
):
    """Plot mass residuals with available errors, labelling conditional bars.

    An unavailable combined uncertainty never hides a finite mass result.
    Residual error bars combine the selected measurement and AME marginal
    errors; their unknown cross-covariance is not inferred here.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _fig, ax = plt.subplots(figsize=(9, 4))

    if unknown_df.empty:
        return _empty_unknown_plot(ax, title)
    df = filter_ground_states(unknown_df, ground_only=ground_only)

    y = np.asarray(df[residual_col], dtype=float)
    yerr_exp, conditional = _measurement_errors(df, exp_err_col)
    yerr_ame = np.array(df[ame_err_col], dtype=float, copy=True) if ame_err_col in df else np.full(len(df), np.nan)
    yerr_ame[yerr_ame < 0] = np.nan
    yerr = np.sqrt(yerr_exp**2 + yerr_ame**2)

    labels = df[label_col].astype(str).to_numpy()
    x = np.arange(len(df))

    finite = np.isfinite(y)
    if not np.any(finite):
        return _empty_unknown_plot(ax, title)
    x = x[finite]
    y = y[finite]
    yerr = yerr[finite]
    labels = labels[finite]
    conditional = conditional[finite]

    ax.axhline(0.0, color="k", lw=1)
    for mask, description in _error_groups(yerr, conditional):
        if np.any(mask):
            ax.errorbar(x[mask], y[mask], yerr=_available_errors(yerr[mask]), fmt="o", capsize=3, label=description)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_ylabel(r"$ME_{\mathrm{exp}} - ME_{\mathrm{AME}}$ [keV]")
    if ground_only:
        title += " — ground states only"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax

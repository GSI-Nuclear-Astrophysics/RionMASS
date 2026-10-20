"""Uncertainty-budget diagnostics for a fitted calibration.

The empirical systematic is a single additive term estimated through
leave-one-out self-calibration.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .electronic import ElectronicInputs, propagate_electronic_uncertainty, propagate_partial_covariance
from .fitting import fit_calibration, predict_unknown_table
from .models import IDX_X, CalibrationFit
from .processing import get_isomer_metadata


def leave_one_out_self_calibration(
    Nuclei_Y: Sequence[Sequence[Any]],
    p: int = 3,
    isomer_excitation_energy: dict[str, Any] | None = None,
    observable_label: str = "x",
    observable_unit: str = "arb.",
    iterationMax: int = 100,
    tol: float = 1e-14,
    electronic_inputs: ElectronicInputs | None = None,
    electronic_method: str = "fixed_weights",
    electronic_step_MeV: float = 0.001,
    electronic_workers: int = 1,
    loo_anchor_species: Sequence[str] = (),
    loo_anchor_policy: str = "retain_in_training",
) -> tuple[pd.DataFrame, list[list[Any]], list[CalibrationFit]]:
    """Hold out each calibrant in turn, refit, and predict it as an unknown.

    Each residual is the held-out prediction minus the supplied reference
    mass excess. Eligible residuals enter
    :func:`estimate_extra_systematic_from_loo`.

    Explicit ``loo_anchor_species`` are excluded from the scatter estimate,
    but their held-out predictions remain in the output as diagnostics.
    By default they stay in every other fold's training set. The alternative
    ``remove_from_validation`` omits them from those training sets too.
    Neither policy removes an anchor from the final calibration or suppresses
    its input uncertainty. Choose anchors before inspecting validation residuals.

    Returns ``(loo_df, legacy_rows, fits)`` — one row/entry per held-out
    calibrant, in ``Nuclei_Y`` order.

    With electronic inputs, the shared calibration/recovery contribution is
    included in each pull denominator when known. Otherwise that denominator
    remains conditional and ``electronic_uncertainty_known`` is False.
    ``loo_df.attrs['electronic_covariance_keV2']`` retains the electronic
    cross-covariance between residuals. The empirical extra-scatter estimator
    still uses marginal denominators, not a joint correlated chi-square.
    """
    if loo_anchor_policy not in ("retain_in_training", "remove_from_validation"):
        raise ValueError("loo_anchor_policy must be retain_in_training or remove_from_validation")
    if isinstance(loo_anchor_species, str) or len(set(loo_anchor_species)) != len(loo_anchor_species):
        raise ValueError("loo_anchor_species must be a sequence of distinct species ids")
    anchors = set(loo_anchor_species)
    labels = [get_isomer_metadata(n)[2] for n in Nuclei_Y]
    for label in anchors:
        if labels.count(label) != 1:
            raise ValueError(f"LOO anchor {label!r} must match exactly one reference species")
    if anchors and len(Nuclei_Y) - len(anchors) <= p:
        raise ValueError("Too few non-anchor LOO residuals for N-p validation; reduce p or use more references")
    rows = []
    legacy_rows = []
    fits = []
    electronic_jacobians = []
    for i, nuc in enumerate(Nuclei_Y):
        is_anchor = labels[i] in anchors
        # Anchor holdouts remain diagnostic only. Other folds obey the selected
        # training policy, while the final all-reference calibration is untouched.
        train = [
            Nuclei_Y[j]
            for j in range(len(Nuclei_Y))
            if j != i and (is_anchor or loo_anchor_policy == "retain_in_training" or labels[j] not in anchors)
        ]
        test = [nuc]
        fit = fit_calibration(
            train,
            p=p,
            isomer_excitation_energy=isomer_excitation_energy,
            iterationMax=iterationMax,
            tol=tol,
            observable_label=observable_label,
            observable_unit=observable_unit,
        )
        legacy, detail, _ = predict_unknown_table(fit, test, extra_sys_keV=0.0)
        d = detail.iloc[0].to_dict()
        d["held_out_index"] = i
        d["held_out_species_id"] = labels[i]
        d["used_for_extra_scatter"] = not is_anchor
        d["loo_role"] = "Anchor diagnostic only" if is_anchor else "Validation reference"
        d["loo_anchor_policy"] = loo_anchor_policy
        train_x = np.asarray(fit.x)
        x = float(nuc[IDX_X])
        d["loo_extrapolation"] = bool(x < train_x.min() or x > train_x.max())
        d["loo_training_species"] = ";".join(get_isomer_metadata(n)[2] for n in train)
        # The comparator is the supplied reference, which may override AME.
        denom = math.sqrt(d["ME_total_err_keV"] ** 2 + d["ME_AME_state_err_keV"] ** 2)
        d["loo_sigma_conditional_keV"] = denom
        if electronic_inputs is not None:
            electronic = propagate_electronic_uncertainty(
                fit,
                train,
                test,
                electronic_inputs,
                method=electronic_method,
                iterationMax=iterationMax,
                tol=tol,
                step_MeV=electronic_step_MeV,
                workers=electronic_workers,
            )
            electronic_jacobians.append(electronic.mass_jacobian_keV_per_MeV[0])
            variance = float(electronic.mass_covariance_keV2.iloc[0, 0])
            d["ME_electronic_err_keV"] = math.sqrt(max(variance, 0)) if np.isfinite(variance) else np.nan
            d["electronic_uncertainty_known"] = np.isfinite(variance)
            if np.isfinite(variance):
                denom = math.sqrt(denom**2 + max(variance, 0))
        else:
            d["ME_electronic_err_keV"] = np.nan
            d["electronic_uncertainty_known"] = False
        d["loo_sigma_no_extra_keV"] = denom
        d["loo_pull_no_extra"] = d["Delta_ME_state_keV"] / denom if denom > 0 else np.nan
        d["fit_chi2"] = fit.chi2
        d["fit_red_chi2"] = fit.red_chi2
        rows.append(d)
        legacy_rows.append(legacy[0])
        fits.append(fit)
    table = pd.DataFrame(rows)
    table.attrs["loo_anchor_species"] = tuple(loo_anchor_species)
    table.attrs["loo_anchor_policy"] = loo_anchor_policy
    if electronic_inputs is not None:
        j = np.asarray(electronic_jacobians)
        cov = propagate_partial_covariance(j, electronic_inputs.covariance_MeV2, j)
        covariance_labels = pd.Index([f"{get_isomer_metadata(r)[2]}#{i}" for i, r in enumerate(Nuclei_Y)])
        table.attrs["electronic_covariance_keV2"] = pd.DataFrame(
            cov, index=covariance_labels, columns=covariance_labels
        )
    return table, legacy_rows, fits


def estimate_extra_systematic_from_loo(
    loo_df: pd.DataFrame,
    p: int,
    residual_col: str | None = None,
    sigma_col: str = "loo_sigma_no_extra_keV",
    center_residuals: bool = False,
    dof: int | None = None,
) -> dict[str, Any]:
    """Estimate an additive LOO-scatter allowance in keV.

    Solve sum(r_i**2 / (s_i**2 + extra**2)) = dof, returning zero when the
    uninflated sum is already no larger than dof. The default dof is N-p,
    or N-p-1 for centered residuals. Shared LOO training sets make this an
    empirical marginal-pull prescription rather than an independent chi-square
    test. It measures validation scatter within the supplied reference network.

    Rows marked used_for_extra_scatter=False remain visible but do not enter
    the residual count or estimate. The returned table includes uncertainties
    and pulls with the fitted allowance.
    """
    if residual_col is None:
        residual_col = "Delta_ME_reference_keV" if "Delta_ME_reference_keV" in loo_df else "Delta_ME_state_keV"
    r = np.asarray(loo_df[residual_col], dtype=float)
    s = np.asarray(loo_df[sigma_col], dtype=float)
    eligible = (
        loo_df["used_for_extra_scatter"].to_numpy(dtype=bool)
        if "used_for_extra_scatter" in loo_df
        else np.ones(len(loo_df), dtype=bool)
    )
    mask = np.isfinite(r) & np.isfinite(s) & (s > 0) & eligible
    r, s = r[mask], s[mask]
    N = len(r)
    if N == 0:
        raise ValueError("No valid LOO residuals")
    w = 1.0 / s**2
    weighted_mean = np.sum(w * r) / np.sum(w)
    mean_err = math.sqrt(1.0 / np.sum(w))
    if center_residuals:
        r_for_chi2 = r - weighted_mean
    else:
        r_for_chi2 = r.copy()
    if dof is None:
        degrees = N - p - (1 if center_residuals else 0)
        if not np.all(eligible) and degrees <= 0:
            raise ValueError("Too few eligible LOO residuals for N-p validation")
        dof = max(1, degrees)
    if dof <= 0:
        raise ValueError("LOO degrees of freedom must be positive")

    def chi2(sys: float) -> float:
        return float(np.sum((r_for_chi2 / np.sqrt(s**2 + sys**2)) ** 2))

    chi2_raw = chi2(0.0)
    red_raw = chi2_raw / dof
    if red_raw <= 1.0:
        sys = 0.0
    else:
        hi = max(1.0, np.max(np.abs(r_for_chi2)), np.max(s))
        while chi2(hi) / dof > 1.0:
            hi *= 2.0
        sys = float(brentq(lambda val: chi2(val) / dof - 1.0, 0.0, hi))  # pyright: ignore[reportArgumentType]  # brentq(full_output=False) always returns float
    out: dict[str, Any] = dict(
        N=N,
        n_excluded=int(np.count_nonzero(~eligible)),
        loo_anchor_species=loo_df.attrs.get("loo_anchor_species", ()),
        loo_anchor_policy=loo_df.attrs.get("loo_anchor_policy", "retain_in_training"),
        p=p,
        dof=dof,
        center_residuals=center_residuals,
        chi2_raw=chi2_raw,
        red_chi2_raw=red_raw,
        normalized_chi_raw=math.sqrt(red_raw),
        extra_sys_keV=sys,
        chi2_with_sys=chi2(sys),
        red_chi2_with_sys=chi2(sys) / dof,
        normalized_chi_with_sys=math.sqrt(chi2(sys) / dof),
        weighted_mean_residual_keV=weighted_mean,
        weighted_mean_error_keV=mean_err,
    )
    result_df = loo_df.copy()
    result_df["extra_sys_keV"] = np.where(eligible, sys, 0.0)
    result_df["loo_sigma_with_extra_keV"] = np.sqrt(result_df[sigma_col] ** 2 + result_df["extra_sys_keV"] ** 2)
    result_df["loo_pull_with_extra"] = result_df[residual_col] / result_df["loo_sigma_with_extra_keV"]
    out["table"] = result_df
    return out


def compare_polynomial_orders(
    Nuclei_Y: Sequence[Sequence[Any]],
    p_values: Sequence[int] = (2, 3, 4),
    isomer_excitation_energy: dict[str, Any] | None = None,
    observable_label: str = "x",
    observable_unit: str = "arb.",
    electronic_inputs: ElectronicInputs | None = None,
    electronic_method: str = "fixed_weights",
    electronic_step_MeV: float = 0.001,
    electronic_workers: int = 1,
    iterationMax: int = 100,
    tol: float = 1e-14,
    loo_anchor_species: Sequence[str] = (),
    loo_anchor_policy: str = "retain_in_training",
) -> pd.DataFrame:
    """Compare polynomial orders using held-out reference residuals and pulls.

    An order requires more references than coefficients so each holdout can
    still be fitted. These diagnostics inform, but do not select, the order.
    """
    rows = []
    for p in p_values:
        if len(Nuclei_Y) <= p:
            continue
        try:
            loo, _, _ = leave_one_out_self_calibration(
                Nuclei_Y,
                p=p,
                isomer_excitation_energy=isomer_excitation_energy,
                observable_label=observable_label,
                observable_unit=observable_unit,
                electronic_inputs=electronic_inputs,
                electronic_method=electronic_method,
                electronic_step_MeV=electronic_step_MeV,
                electronic_workers=electronic_workers,
                iterationMax=iterationMax,
                tol=tol,
                loo_anchor_species=loo_anchor_species,
                loo_anchor_policy=loo_anchor_policy,
            )
            loo = loo.loc[loo["used_for_extra_scatter"]]
            r = np.asarray(loo["Delta_ME_state_keV"], dtype=float)
            pull = np.asarray(loo["loo_pull_no_extra"], dtype=float)
            rows.append(
                {
                    "p_terms": p,
                    "degree": p - 1,
                    "n_loo": len(loo),
                    "loo_anchor_species": ";".join(loo_anchor_species),
                    "loo_anchor_policy": loo_anchor_policy,
                    "mean_residual_keV": np.nanmean(r),
                    "rms_residual_keV": math.sqrt(np.nanmean(r**2)),
                    "median_abs_residual_keV": np.nanmedian(np.abs(r)),
                    "pull_mean": np.nanmean(pull),
                    "pull_rms": math.sqrt(np.nanmean(pull**2)),
                    "electronic_uncertainty_complete": bool(np.all(loo["electronic_uncertainty_known"].to_numpy())),
                }
            )
        except Exception as exc:
            rows.append({"p_terms": p, "degree": p - 1, "error": str(exc)})
    return pd.DataFrame(rows)


def calibrant_geometry_sweep(
    Nuclei_Y: Sequence[Sequence[Any]],
    Nuclei_N: Sequence[Sequence[Any]],
    candidate_labels: Sequence[str],
    p: int = 3,
    isomer_excitation_energy: dict[str, Any] | None = None,
    observable_label: str = "x",
    observable_unit: str = "arb.",
    iterationMax: int = 100,
    tol: float = 1e-14,
) -> pd.DataFrame:
    """Measure prediction shifts after removing each candidate reference.

    Refit the calibration and LOO extra scatter for each removal, comparing
    unknown masses with the all-reference baseline. candidate_labels must match
    reference species IDs; a missing label raises ValueError. Other fit inputs
    are held fixed.

    Returns one row per (candidate_removed, unknown), with masses and shifts in
    keV. Fewer than p remaining references gives skipped=True and NaN shifts.
    Exactly p references still gives a mass prediction, but LOO scatter is NaN:
    its estimation requires at least p+1 references. A valid mass shift can
    therefore accompany an unavailable extra-scatter estimate.
    """
    label_to_index = {get_isomer_metadata(nuc)[2]: i for i, nuc in enumerate(Nuclei_Y)}
    missing = [label for label in candidate_labels if label not in label_to_index]
    if missing:
        raise ValueError(f"candidate_labels not found in Nuclei_Y: {missing}")

    def _fit_and_predict(calibrants: Sequence[Sequence[Any]]) -> tuple[pd.Series, float]:
        fit = fit_calibration(
            calibrants,
            p=p,
            isomer_excitation_energy=isomer_excitation_energy,
            iterationMax=iterationMax,
            tol=tol,
            observable_label=observable_label,
            observable_unit=observable_unit,
        )
        # A saturated fit predicts masses but cannot estimate LOO scatter.
        if len(calibrants) > p:
            loo_df, _, _ = leave_one_out_self_calibration(
                calibrants,
                p=p,
                isomer_excitation_energy=isomer_excitation_energy,
                observable_label=observable_label,
                observable_unit=observable_unit,
                iterationMax=iterationMax,
                tol=tol,
            )
            systematic = estimate_extra_systematic_from_loo(loo_df, p=p)
            extra_sys_keV = systematic["extra_sys_keV"]
        else:
            extra_sys_keV = np.nan
        _, detail_df, _ = predict_unknown_table(fit, Nuclei_N, extra_sys_keV=extra_sys_keV)
        return cast(pd.Series, detail_df.set_index("species_id")["ME_state_keV"]), extra_sys_keV

    baseline_me, baseline_extra_sys = _fit_and_predict(Nuclei_Y)

    rows = []
    for label in candidate_labels:
        remaining = [nuc for i, nuc in enumerate(Nuclei_Y) if i != label_to_index[label]]
        skipped = len(remaining) < p
        perturbed_me: pd.Series
        perturbed_extra_sys = np.nan
        if not skipped:
            perturbed_me, perturbed_extra_sys = _fit_and_predict(remaining)
        for unknown_id, baseline_value in baseline_me.items():
            perturbed_value = np.nan if skipped else perturbed_me.get(unknown_id, np.nan)
            rows.append(
                dict(
                    candidate_removed=label,
                    unknown=unknown_id,
                    skipped=skipped,
                    n_calibrants_baseline=len(Nuclei_Y),
                    n_calibrants_without_candidate=len(remaining),
                    ME_state_keV_baseline=baseline_value,
                    ME_state_keV_without_candidate=perturbed_value,
                    delta_ME_state_keV=np.nan if skipped else perturbed_value - baseline_value,
                    extra_sys_keV_baseline=baseline_extra_sys,
                    extra_sys_keV_without_candidate=perturbed_extra_sys,
                    delta_extra_sys_keV=np.nan if skipped else perturbed_extra_sys - baseline_extra_sys,
                )
            )
    return pd.DataFrame(rows)

"""Readable result tables, without changing mass or uncertainty calculations."""

from __future__ import annotations

# ruff: noqa: RUF001 -- Greek sigma is intentional in uncertainty headings.
import numpy as np
import pandas as pd

from .processing import parse_ion_label

_SUPERSCRIPTS = str.maketrans("0123456789+m", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺ᵐ")


def ion_display_label(species_id: str) -> str:
    """Unicode isotope/charge label for plain text tables and CSV exports."""
    try:
        a, isomer, element, q = parse_ion_label(species_id)
        return (a + isomer).translate(_SUPERSCRIPTS) + element + (q + "+").translate(_SUPERSCRIPTS)
    except ValueError:
        return species_id


def final_mass_results(unknown_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize state masses; keep conditional and combined errors distinct.

    Fit sigma contains calibration-curve and unknown-observable errors.
    Conditional sigma adds LOO extra scatter. Combined sigma also includes
    the supplied electronic component, if evaluated. Unknown inputs remain
    NaN; this function never fills them with zero or overwrites original data.
    """
    columns = {
        "ME (keV)": "ME_state_keV",
        "ΔME vs AME2020 (keV)": "Delta_ME_state_keV",
        "σ AME2020 (keV)": "ME_AME_state_err_keV",
        "σ fit (keV)": "ME_stat_err_keV",
        "σ LOO extra (keV)": "ME_extra_sys_keV",
        "σ electronic (keV)": "ME_electronic_err_keV",
        "σ conditional (keV)": "ME_conditional_err_keV",
        "σ combined (keV)": "ME_total_err_keV",
    }
    labels = (
        unknown_df["species_id"]
        if "species_id" in unknown_df
        else unknown_df["ion"]
        if "ion" in unknown_df
        else pd.Series(dtype=str)
    )
    out = pd.DataFrame({"Ion": [ion_display_label(str(v)) for v in labels]})
    for heading, source in columns.items():
        out[heading] = unknown_df[source].to_numpy() if source in unknown_df else np.full(len(out), np.nan)
    # Older conditional-only predictions did not expose this separate column.
    if "ME_conditional_err_keV" not in unknown_df and "ME_electronic_err_keV" not in unknown_df:
        out["σ conditional (keV)"] = out["σ combined (keV)"]
        out["σ combined (keV)"] = np.nan
    out["Uncertainty status"] = np.where(
        np.isfinite(out["σ combined (keV)"]),
        "Electronic inputs included",
        "Electronic term unavailable",
    )
    return out


def loo_reference_results(loo_df: pd.DataFrame) -> pd.DataFrame:
    """Held-out state predictions and comparison errors, with anchor roles.

    Prediction sigma excludes the supplied reference-mass error. Comparison
    sigma includes it; with-extra sigma is the validation denominator.
    Unknown electronic terms stay unavailable rather than silently zeroed.
    """
    labels = (
        loo_df["species_id"]
        if "species_id" in loo_df
        else loo_df["held_out_species_id"]
        if "held_out_species_id" in loo_df
        else pd.Series(dtype=str)
    )
    out = pd.DataFrame({"Ion": [ion_display_label(str(v)) for v in labels]})
    columns = {
        "LOO ME (keV)": "ME_state_keV",
        "ΔME vs AME2020 (keV)": "Delta_ME_state_keV",
        "σ AME2020 (keV)": "ME_AME_state_err_keV",
        "σ prediction conditional (keV)": "ME_total_err_keV",
        "σ electronic (keV)": "ME_electronic_err_keV",
        "σ comparison (keV)": "loo_sigma_no_extra_keV",
        "σ comparison + extra (keV)": "loo_sigma_with_extra_keV",
        "Pull": "loo_pull_no_extra",
    }
    for heading, source in columns.items():
        out[heading] = loo_df[source].to_numpy() if source in loo_df else np.full(len(out), np.nan)
    if "reference_is_custom" in loo_df and bool(loo_df["reference_is_custom"].any()):
        out["Reference source"] = loo_df["reference_mass_source"].to_numpy()
        out["Reference ME (keV)"] = loo_df["ME_reference_state_keV"].to_numpy()
        out["σ reference (keV)"] = loo_df["ME_reference_state_err_keV"].to_numpy()
        out["ΔME vs reference (keV)"] = loo_df["Delta_ME_reference_keV"].to_numpy()
        out = out.rename(
            columns={
                "Pull": "Pull vs reference",
                "σ comparison (keV)": "σ reference comparison (keV)",
                "σ comparison + extra (keV)": "σ reference comparison + extra (keV)",
            }
        )
    out["σ prediction combined (keV)"] = np.sqrt(
        out["σ prediction conditional (keV)"] ** 2 + out["σ electronic (keV)"] ** 2
    )
    out["Validation use"] = loo_df["loo_role"].to_numpy() if "loo_role" in loo_df else "Validation reference"
    out["Held-out geometry"] = (
        np.where(loo_df["loo_extrapolation"].to_numpy(), "Extrapolation", "Interpolation")
        if "loo_extrapolation" in loo_df
        else "Unavailable"
    )
    out["Comparison status"] = (
        np.where(
            loo_df["electronic_uncertainty_known"].to_numpy(),
            "Electronic inputs included",
            "Conditional: electronic term unavailable",
        )
        if "electronic_uncertainty_known" in loo_df
        else "Conditional: electronic term unavailable"
    )
    return out

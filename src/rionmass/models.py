"""Ion-row conventions and calibration results.

The IDX_* constants define the row format used by the numerical API.
NuclearState provides named access to those fields. CalibrationFit stores
the centered polynomial, coefficient covariance and reference diagnostics.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# Ion-row field order is part of the public API.
# One row = one ion (a calibrant or an unknown). Column semantics, in order:
#   element name (AME form), Z, A, N, Q, reference flag ('Y'=calibrant,
#   'N'=unknown), observable x, its uncertainty, template-fit width of x,
#   width uncertainty, AME mass excess (MeV) and its uncertainty, electron
#   correction I_Q = B_neutral - B_ion (MeV), AME ionic m/q (u/e) and its uncertainty, isomer
#   label, display name, species id.
IDX_ELEMENT = 0
IDX_Z = 1
IDX_A = 2
IDX_N = 3
IDX_Q = 4
IDX_FLAG = 5
IDX_X = 6  # generic observable: frequency or revolution time
IDX_XERR = 7
IDX_WIDTH_X = 8
IDX_WIDTH_XERR = 9
IDX_ME_AME = 10
IDX_MEERR_AME = 11
IDX_EBIND = 12
IDX_MOQ_AME = 13
IDX_MOQERR_AME = 14
IDX_ISOMER = 15
IDX_DISPLAY_NAME = 16
IDX_SPECIES_ID = 17


@dataclass(frozen=True, slots=True)
class NuclearState:
    """Read-only named view of an ion row; construct with from_row."""

    element: str
    Z: int
    A: int
    N: int
    Q: float
    flag: str
    x: float
    x_err: float
    width_x: float
    width_x_err: float
    me_ame_MeV: float
    me_ame_err_MeV: float
    e_bind_MeV: float
    moq_ame: float
    moq_ame_err: float
    isomer: str
    display_name: str
    species_id: str

    @classmethod
    def from_row(cls, row: Sequence[Any]) -> NuclearState:
        """Build a :class:`NuclearState` from a legacy ``Nuclei``-style row."""
        return cls(
            element=str(row[IDX_ELEMENT]),
            Z=int(row[IDX_Z]),
            A=int(row[IDX_A]),
            N=int(row[IDX_N]),
            Q=float(row[IDX_Q]),
            flag=str(row[IDX_FLAG]),
            x=float(row[IDX_X]),
            x_err=float(row[IDX_XERR]),
            width_x=float(row[IDX_WIDTH_X]),
            width_x_err=float(row[IDX_WIDTH_XERR]),
            me_ame_MeV=float(row[IDX_ME_AME]),
            me_ame_err_MeV=float(row[IDX_MEERR_AME]),
            e_bind_MeV=float(row[IDX_EBIND]),
            moq_ame=float(row[IDX_MOQ_AME]),
            moq_ame_err=float(row[IDX_MOQERR_AME]),
            isomer=str(row[IDX_ISOMER]) if len(row) > IDX_ISOMER else "",
            display_name=str(row[IDX_DISPLAY_NAME]) if len(row) > IDX_DISPLAY_NAME else str(row[IDX_ELEMENT]),
            species_id=(
                str(row[IDX_SPECIES_ID])
                if len(row) > IDX_SPECIES_ID
                else f"{row[IDX_ELEMENT]}{int(float(row[IDX_Q]))}+"
            ),
        )


@dataclass
class CalibrationFit:
    """Result of :func:`rionmass.fitting.fit_calibration`.

    The fitted model is the centered/scaled polynomial

    .. math:: m/q = \\sum_{k=0}^{p-1} a_k z^k, \\qquad z = (x - x_0) / x_{\\rm scale}

    ``x0``/``xscale`` are the centering/scaling constants chosen from the
    calibrant data (see :func:`rionmass.fitting.fit_calibration`); ``coeff``
    and ``cov`` are expressed in this *scaled* basis, not in raw powers of
    ``x``. Use design_vector when propagating this covariance.
    """

    p: int
    degree: int
    x0: float
    xscale: float
    coeff: np.ndarray
    cov: np.ndarray
    chi2: float
    dof: int
    red_chi2: float
    x: np.ndarray
    sx: np.ndarray
    y: np.ndarray
    sy: np.ndarray
    sigma_eff_y: np.ndarray
    design: np.ndarray
    yfit: np.ndarray
    residuals: np.ndarray
    pulls: np.ndarray
    flow_coeff: np.ndarray
    significance: np.ndarray
    R: np.ndarray
    labels: list[str]
    observable_label: str
    observable_unit: str
    isomer_excitation_energy: dict[str, Any] | None = field(default=None)

    @property
    def param_errors(self) -> np.ndarray:
        """1-sigma uncertainty on each fitted coefficient, sqrt(diag(cov))."""
        return np.sqrt(np.clip(np.diag(self.cov), 0.0, np.inf))

    @property
    def corr(self) -> np.ndarray:
        """Parameter correlation matrix, derived from :attr:`cov`."""
        d = self.param_errors
        denom = np.outer(d, d)
        out = np.full_like(self.cov, np.nan, dtype=float)
        mask = denom > 0
        out[mask] = self.cov[mask] / denom[mask]
        return out

    def z(self, x: Any) -> np.ndarray:
        """Map a raw observable value to the fit's centered/scaled coordinate."""
        return (np.asarray(x, dtype=float) - self.x0) / self.xscale

    def design_vector(self, x: Any) -> np.ndarray:
        z = np.atleast_1d(self.z(x))
        return np.vstack([z**k for k in range(self.p)]).T

    def derivative_vector_dx(self, x: Any) -> np.ndarray:
        """d(design row)/dx, used to propagate the observable's uncertainty."""
        z = np.atleast_1d(self.z(x))
        cols = []
        for k in range(self.p):
            if k == 0:
                cols.append(np.zeros_like(z))
            else:
                cols.append(k * z ** (k - 1) / self.xscale)
        return np.vstack(cols).T

    def predict_moq(self, x: Any, sx: Any = 0.0) -> dict[str, np.ndarray]:
        """Predict m/q (and its error budget) at observable value(s) ``x``.

        Returns curve error (from the fit covariance), the propagated
        observable-uncertainty contribution, and their quadrature sum
        ``moq_err``. The workflow adds electronic and LOO-scatter terms separately.
        """
        scalar = np.ndim(x) == 0
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        sx_arr = np.broadcast_to(np.asarray(sx, dtype=float), x_arr.shape)
        V = self.design_vector(x_arr)
        D = self.derivative_vector_dx(x_arr)
        y = V @ self.coeff
        dydx = D @ self.coeff
        curve_var = np.einsum("ij,jk,ik->i", V, self.cov, V)
        x_var = (dydx * sx_arr) ** 2
        total_var = np.clip(curve_var, 0.0, np.inf) + x_var
        out = dict(
            moq=y,
            moq_curve_err=np.sqrt(np.clip(curve_var, 0.0, np.inf)),
            moq_x_err=np.sqrt(x_var),
            moq_err=np.sqrt(total_var),
            dydx=dydx,
        )
        if scalar:
            return {k: np.asarray(v)[0] for k, v in out.items()}
        return out

    def prediction_influence(self, x_unknown: float) -> pd.DataFrame:
        """Signed per-calibrant influence on the predicted m/q at ``x_unknown``.

        ``h_i = v(x*)^T R[:, i]``; ``sum_i h_i == 1`` for models with an
        intercept (Audi-style flow-of-information convention).
        """
        v = self.design_vector(float(x_unknown))[0]
        h = v @ self.R
        var_terms = (h * self.sigma_eff_y) ** 2
        denom = np.sum(var_terms)
        frac = var_terms / denom if denom > 0 else np.full_like(var_terms, np.nan)
        return pd.DataFrame(
            {
                "calibrant": self.labels,
                "signed_prediction_weight": h,
                "variance_fraction": frac,
                "x": self.x,
                "pull": self.pulls,
                "significance_leverage": self.significance,
            }
        )

    def parameter_table(self) -> pd.DataFrame:
        rows = [{"parameter": f"a{k}", "value": self.coeff[k], "sigma": self.param_errors[k]} for k in range(self.p)]
        return pd.DataFrame(rows)

    def covariance_table(self) -> pd.DataFrame:
        names = pd.Index([f"a{k}" for k in range(self.p)])
        return pd.DataFrame(self.cov, index=names, columns=names)

    def correlation_table(self) -> pd.DataFrame:
        names = pd.Index([f"a{k}" for k in range(self.p)])
        return pd.DataFrame(self.corr, index=names, columns=names)

    def flow_table(self) -> pd.DataFrame:
        df = pd.DataFrame(self.flow_coeff, columns=pd.Index([f"flow_to_a{k}" for k in range(self.p)]))
        df.insert(0, "calibrant", np.asarray(self.labels, dtype=object))
        df["significance_leverage"] = self.significance
        df["x"] = self.x
        df["pull"] = self.pulls
        return df

    def calibrant_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "calibrant": self.labels,
                "x": self.x,
                "x_err": self.sx,
                "moq_ref": self.y,
                "moq_ref_err": self.sy,
                "moq_fit": self.yfit,
                "moq_residual_fit_minus_ref": self.residuals,
                "sigma_eff_y": self.sigma_eff_y,
                "pull": self.pulls,
                "significance_leverage": self.significance,
            }
        )

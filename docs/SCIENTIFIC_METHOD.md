# Calibration and reference diagnostics

The observable is fundamental revolution frequency in Hz or revolution time
in ps. Ionic mass-to-charge is reported in u/e:

```text
y = [A*U + ME_gs + E_iso - q*m_e*c² + I_q] / (q*U)
I_q = B_neutral - B_ion
P(x) = sum_k a_k*z^k, k=0,...,p-1
z = (x-x0)/xscale
sigma_eff,i² = sigma_y,i² + [P'(x_i)*sigma_x,i]²
```

`U = m_u*c²`, mass excesses, excitation energies and electronic corrections
use MeV. `q` is positive ion charge. Mass recovery reverses the first equation;
an isomer retains its state mass excess unless an excitation is explicitly
subtracted. The electronic model uses ground-state configurations and neglects
isotope-dependent electronic binding.

The default center is the inverse-reference-variance-weighted mean of the
reference observables; the scale is half their span. Coefficients and covariance
use this centered basis. Effective weights iteratively propagate observable
errors. Reference-row and observable errors are independent in this estimator;
it is not a general joint-input covariance fit.

## Reference influence

At the final weights, `R = (V.T*W*V)^+*V.T*W` maps references to coefficients.
The diagonal of `V*R` gives leverage. Coefficient information flow resolves
that dependence by parameter. At an unknown position, `v(x*)*R` gives signed
prediction weights. Squared weights times effective variances give the
prediction-specific curve-variance contributions.

A reference that anchors a polynomial coefficient need not dominate the
prediction error near an unknown. Reference-deletion refits directly measure
the resulting change in each unknown mass.

## Leave-one-out validation

Hold out each reference, refit the others and predict the held-out state.
Comparison errors include both the prediction and held-out reference errors.
Known electronic contributions are included; missing ones remain conditional.

The extra-scatter routine solves

```text
sum_i r_i² / (sigma_i² + sigma_extra²) = nu
nu = max(1, N-p)  # uncentered eligible residuals
```

It returns zero if the uninflated sum is already no larger than `nu`. Shared
LOO training sets make this an empirical marginal-pull convention, separate
from the full-calibration goodness-of-fit statistic.

Explicit anchors remain in the final fit. Their own holdouts remain visible
as diagnostics but do not enter the extra-scatter estimate. `retain_in_training`
keeps them in other folds; `remove_from_validation` removes them from those
training sets too. The policy and actual fold membership are exported.

Order comparisons and deletion studies assess reference span and prediction
stability. Model or reference-set changes are sensitivities rather than
additional independent statistical measurements.

## Joint mass covariance

The conditional curve and unknown-observable terms, supplied electronic
covariance and an independent per-mass LOO allowance enter the returned mass
covariance. Electronic sensitivities include both calibration and mass recovery.
Rows sharing element and charge share their electronic correction.

Unprovided AME correlations, observable correlations and a different correlation
model for the LOO allowance are not supplied by this matrix. See
[UNCERTAINTIES.md](UNCERTAINTIES.md) for electronic inputs and output fields.

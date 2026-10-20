# Electronic inputs and uncertainty propagation


The binding-energy API returns the **cumulative ionization energy**
`I_q = B_neutral - B_ion`, in MeV, for removal of `q` electrons from a
neutral atom. The mass conversion is
`M_ion*c^2 = M_atom*c^2 - q*m_e*c^2 + I_q`; recovery of the atomic mass
subtracts the same `I_q`. `ElBiEn_*.dat` columns count electrons remaining,
so the residual-ion binding is column `Z-q`. NIST total-binding entries
are selected by their `Ion Charge` field at charge `q`.

Public function names and the `Ebind`/`e_bind_MeV` row fields are retained;
they now explicitly mean `I_q`. Existing hand-built rows and custom
`binding_energy_fn` callbacks must use that convention. Electronic-energy
uncertainties and correlations are propagated separately through the
`ElectronicInputs` model described below. The original scalar lookup
functions still return central values only. No uncertainty is inferred from
the table's number of digits.

Definitions: [NIST ASD ionization-energy and total-binding-energy help](https://physics.nist.gov/PhysRefData/ASD/Html/iehelp.html).

## Electronic uncertainties and covariance

`run_full_calibration_workflow` propagates electronic inputs through both
the reference calibration and the final atomic-mass correction. The 2007
file contains no uncertainties, so its primitive covariance is unknown.
The bundled NIST cache supplies marginal uncertainties where available,
but not the full covariance between binding totals. These missing entries
remain `NaN`; no independence or error size is assumed by default.

Rows sharing `(Z,q)` share the same electronic correction, including nuclear
ground states/isomers. The model assumes electronic ground states and a common
binding correction for each element/charge; it does not model isotope-dependent
binding or electronically excited ions. Electronic errors are assumed independent of AME, frequency and nuclear
excitation-energy errors. Supply the evaluated electronic covariance rather
than adding an independent error to every mass or calibrant row.

For primitive inputs `theta`, write `I_ref = A_ref @ theta` and
`I_unknown = A_unknown @ theta`. At the final calibration weights the
response matrix is `R`, and the unknown polynomial design is `V`. Then

```text
J_beta = R @ diag(1/(q_ref*u)) @ A_ref
J_mass = 1000 * [diag(q_unknown*u) @ V @ J_beta - A_unknown]
C_mass,electronic = J_mass @ C_theta @ J_mass.T             [keV^2]
C_beta,electronic = J_beta @ C_theta @ J_beta.T
C_mass,beta      = J_mass @ C_theta @ J_beta.T
```

`u` is the atomic mass constant energy equivalent in MeV. This preserves
cross-terms and cancellation between reference and unknown corrections.
The nominal calibration is conditional on the electronic central values;
electronic covariance is propagated as a separate nuisance component,
not added again to the reference errors used by the nominal weighted fit.
This is propagation for the chosen estimator, not a generalized-least-squares
fit optimizing weights with a complete electronic covariance.

Python callers can supply a complete or partial model via
`CalibrationConfig(electronic_inputs=model)`. Explicit corrections:

```python
from rionmass.electronic import ElectronicInputs

model = ElectronicInputs.from_corrections(
    corrections_MeV=evaluated_I_q,  # {(Z,q): energy_MeV}; cover every input ion
    covariance_MeV2=evaluated_covariance,  # in sorted (Z,q) order
    source="Evaluation reference, version and covariance assumptions",
)
```

Alternatively provide `sigmas_MeV={(Z,q): sigma_or_None}`. Cross-covariance
stays unknown unless explicitly supplied or `assume_independent=True` is
selected and justified. Known zero uncertainty and unavailable uncertainty
are different inputs. For table-based workflows,
`electronic_assume_independent=True` explicitly models independent primitive
binding errors; it is an approximation for NIST totals and is never the default.

Prefer `ionization_energy_inputs(table, ions, covariance_MeV2=...)` when
individual ionization-energy evaluations are available. It sums only charge
stages `0,...,q-1` and shares these primitives across charge states. Entries
must contain `Ion Charge`, `Ionization Energy (eV)` and, where available,
`Uncertainty (eV)`. The total-binding cache is **not** an individual-ionization
table and cannot be used here. No additional data download occurs automatically.
Covariance follows lexically sorted primitive labels `IE(Z=...,q=...)`.
An optional `source_groups` mapping models full correlation within a supplied
literature group and independence between groups, following NIST's sum
convention. This is an explicit approximation; it does not recover unknown
correlations from a reference string or infer missing uncertainties.

For CLI/GUI use, supply `electronic_inputs_path="electronic.json"`. The file
overrides the binding table. Its correction format is, for example:

```json
{
  "format": "corrections",
  "source": "NIST bundled carbon total; document the evaluation used",
  "corrections": [
    {"Z": 6, "q": 6, "I_q_MeV": 0.0010301085, "sigma_MeV": 4e-10}
  ]
}
```

Include all required ions. Optional `covariance_MeV2` follows the **correction
array order** in JSON (the reader reorders internally); `null` means unknown.
An explicit covariance is authoritative over marginal `sigma_MeV` entries.
Optional `assume_independent: true` is an alternative to an explicit covariance.
For individual energies use `format: "ionization_energies"`, a `table`
mapping atomic-number strings to the entries described above, and optional
`covariance_MeV2` or `source_groups`. Explicit covariance and correlation
assumptions are alternative inputs, not combined adjustments.

```bash
rionmass run --ods template.ods --sheet-index 0 --electronic-inputs electronic.json \
    --electronic-method refit --electronic-workers 4 --outdir results/
```

`electronic_method="fixed_weights"` is the analytic first-order default.
`"refit"` differentiates the complete iterative calibration by centered
finite differences; `electronic_step_MeV` defaults to 0.001 and should be
varied to check numerical stability. Threads are optional. For a nonlinear
check, `electronic_mass_trials(fit, references, unknowns, model,
replicates=5000, seed=..., workers=...)` draws shared Gaussian primitive
inputs and refits. It requires a complete PSD covariance; it samples only
the electronic component, conditional on other inputs. Draws are generated
before threading, so results for a fixed seed do not depend on thread count.

The result exposes `result.electronic`, with mass, coefficient and cross-
covariance matrices, sensitivity matrices and provenance. CSV exports include
the primitive inputs, ion corrections, input covariance, electronic covariance
components and the combined state-mass covariance. `fit.cov` and the original
coefficient tables remain **conditional**; the separate electronic component
can be combined when known. Mass outputs distinguish
`ME_conditional_err_keV`, `ME_electronic_err_keV` and `ME_total_err_keV`.
An unknown electronic contribution gives `NaN` in the combined error/matrix,
rather than a falsely complete uncertainty. A genuinely cancelling unknown
primitive does not make the affected mass uncertainty unknown.

LOO pull denominators include the electronic contribution when known;
otherwise they remain conditional with an explicit status flag. The LOO
electronic covariance is exported. The existing empirical extra-scatter
estimate uses marginal pull denominators, not a joint correlated chi-square.
An incomplete electronic budget can still be absorbed in that conditional
scatter term; the software does not label it as a complete budget.

The combined mass covariance includes the conditional calibration covariance,
independent unknown-observable errors, electronic covariance and an independent
per-mass empirical extra term. It does not include unprovided AME correlations,
frequency/template/alignment correlations or an unspecified correlation model
for the empirical extra term. Use it under those stated assumptions. It covers
state mass excesses; additional uncertainties in externally supplied excitation
energies for ground-state conversion require their own joint propagation.

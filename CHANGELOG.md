# Changelog

## 1.0.1 — 2026-10-20

- Accept three-column ODS inputs containing ion label, centroid frequency and
  centroid uncertainty. Width and width-uncertainty columns are optional;
  missing or blank values remain unavailable. Existing inputs retain their
  column layout and calibration results.

## 1.0.0 — release preparation

- Centered polynomial calibration and iterative observable-error propagation.
- Shared Python, command-line and desktop workflows.
- Information flow, reference leverage and prediction-specific influence.
- LOO validation, anchor policies and polynomial-order comparisons.
- Custom neutral-atom reference masses and uncertainties.
- Shared electronic-input covariance, cancellation and incomplete-budget reporting.
- Labelled final results, AME2020 comparisons and background export tasks.
- Dated run bundles with frozen references and numerical replay.
- Offline public-reference example and distribution-content checks.

Both electronic readers use `I_q = B_neutral - B_ion`. ElBiEn residual binding
uses `Z-q` bound electrons; NIST rows use charge `q`. Earlier development
versions used inconsistent indices and omitted the required binding difference.
Partially stripped-ion calibrations from those versions need refitting from
their observables. The eV-to-MeV divisor remains `1e6`.

The release date and archive DOI are added when the version is published.

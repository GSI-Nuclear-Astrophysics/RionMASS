# Contributing

Install with `python -m pip install -e '.[dev]'`. Run `pytest`, `ruff check .`,
`ruff format --check .` and `pyright` before submitting a change.

For numerical changes, state the physical or statistical motivation, units and
validation. Test independently derived identities or known results with explicit
tolerances. Use constructed observables and published reference data.

Comments should explain units, assumptions or non-obvious numerical choices.
Public functions should document input and output units. Record API and
convention changes in `CHANGELOG.md`.

Open pull requests against `main`. Keep measurement inputs, analysis outputs,
local paths and credentials out of commits and release artifacts.

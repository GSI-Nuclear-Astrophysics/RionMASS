# Reproducing a calibration

Polynomial calibration, LOO refits and extra-scatter estimates are deterministic
for fixed inputs and numerical-library versions. Optional electronic Monte
Carlo trials use an explicit seed and generate draws before dispatching workers.

A completed result bundle preserves import and calibration settings, input
SHA-256, resolved AME records, electronic values and covariance, numerical
checkpoints, constants and software versions. Unavailable uncertainty entries
remain unavailable. Input spreadsheets are referenced, not copied.

`replay_run_configuration` restores inputs, refits and compares unrounded mass
values and errors. Default tolerances are absolute `1e-6` keV and relative
`1e-10`: numerical comparison tolerances, not physical accuracy claims.
A changed input hash gives a warning and replay continues with the saved
sheet index. Sheet insertion or reordering may change the selected content.

The ordinary AME2020 reader caches the IAEA table after its first download.
Completed-result replay uses frozen records. The packaged NIST table supports
offline electronic lookup. An explicit electronic JSON file or cache path
selects another evaluation. Freeze covariance assumptions with central energies.

Public tests cover parsing, unit conversion, calibration identities, reference
diagnostics, electronic cancellation, anchors, overrides, replay, plotting and
GUI tasks. They use constructed observables and published reference inputs.
Check a fresh wheel installation outside the checkout to detect missing files.

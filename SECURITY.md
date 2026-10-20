# Security

RionMASS is a local analysis application. Readers can retrieve AME2020 from
the IAEA or binding energies from NIST when a selected cache is absent.
Completed-result replay can use frozen inputs without those calls.

Only load trusted input files. The legacy NPZ binding cache contains serialized
Python objects and should come from a trusted release or a cache you created.
JSON electronic inputs do not use pickle.

Report security problems privately to D.FreireFernandez@gsi.de, with the affected
version and a minimal reproducer excluding measurement data and credentials.

#!/usr/bin/env bash
# Exercise the public test suite in the code-only container.
set -euo pipefail
docker run --rm --entrypoint python "${RIONMASS_IMAGE:-rionmass:release}" -m pytest -q

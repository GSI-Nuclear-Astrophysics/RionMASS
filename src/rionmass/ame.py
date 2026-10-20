"""Read the public AME2020 fixed-width mass table from IAEA.

The optional data dependencies include fortranformat. Downloads are cached
under ~/.ame unless a cache path is supplied.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Any

#: AME2020 fixed-width mass-table endpoint (IAEA Atomic Mass Data Center).
AME_DATA_URL = "https://www-nds.iaea.org/amdc/ame2020/mass_1.mas20.txt"

_CACHE_DIR = Path.home() / ".ame"
_CACHE_FILE = _CACHE_DIR / "ame.data"

# AME2020's mass table has a 36-line descriptive header before the data
# rows begin; this count is specific to the AME2020 file format.
_HEADER_LINES = 36

# Fixed-width column spec for AME2020's mass table (Fortran format string).
# Columns, 1-indexed by field: cc(a4) NZ(a1) N(i3) Z(i5) A(i5) EL(a3) O(a4)
# then mass-excess (f14.6) +/- (f12.6) [keV], binding-energy/A (f13.5) +/-
# (f10.5) [keV], beta-decay-energy flag (a2), beta-decay-energy (f13.5) +/-
# (f11.5) [keV], atomic-mass integer part (i3), atomic-mass fractional part
# (f13.6) +/- (f12.6) [micro-u].
_AME2020_FORMAT = "(a4,a1,i3,i5,i5,i5,1x,a3,a4,1x,f14.6,f12.6,f13.5,1x,f10.5,1x,a2,f13.5,f11.5,1x,i3,1x,f13.6,f12.6)"


def _ensure_cached(cache_file: Path) -> None:
    """Download the AME2020 mass table to ``cache_file`` if not already present."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    if cache_file.exists():
        return
    request = urllib.request.Request(AME_DATA_URL, headers={"User-Agent": "Magic Browser"})
    with urllib.request.urlopen(request) as response:
        cache_file.write_bytes(response.read())


def read_ame_table(cache_file: str | Path | None = None) -> list[list[Any]]:
    """Parse AME2020 rows, downloading the table only if the cache is absent.

    In the native field order, row[3:7] contains N, Z, A and the element symbol;
    row[8:10] contains mass excess and its standard uncertainty in keV.
    """
    import fortranformat as ff

    path = Path(cache_file) if cache_file is not None else _CACHE_FILE
    _ensure_cached(path)

    line_reader = ff.FortranRecordReader(_AME2020_FORMAT)
    table: list[list[Any]] = []
    with open(path) as f:
        for _ in range(_HEADER_LINES):
            next(f)
        for line in f:
            # Preserve the extrapolation flag while making AME markers parseable.
            if "*" in line:
                line = line.replace("*", "0")
            if "#" in line:
                line = line.replace("#", ".")
                line = "sys " + line
            else:
                line = "exp " + line
            row = line_reader.read(line)
            row = [value.strip() if isinstance(value, str) else value for value in row]
            table.append(row)
    return table

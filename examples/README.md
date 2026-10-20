# Offline workflow example

`quickstart.py` uses five public AME2020 records and the packaged NIST
binding-energy cache. It constructs a frequency-to-mass relation solely to
exercise spreadsheet import, calibration, CSV export and configuration replay.
The frequencies are not measured and are not a performance benchmark.

```sh
python examples/quickstart.py
python examples/quickstart.py --output example_output
```

The retained output includes input ODS, electronic JSON, result tables and
a replay bundle. All data in this example can be redistributed. The mass
records come from the [IAEA AME2020 table](https://www-nds.iaea.org/amdc/ame2020/mass_1.mas20.txt).

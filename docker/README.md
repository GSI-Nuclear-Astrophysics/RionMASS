# Code-only container

Build with `docker build -t rionmass:release .`. The image contains source,
public tests and the offline example. It does not include measurement inputs.

```sh
docker run --rm rionmass:release --help
docker run --rm --entrypoint python rionmass:release examples/quickstart.py
bash docker/run_checks.sh
```

For your own analysis, mount an input directory read-only and a separate output
directory writable, then pass their container paths to `rionmass run`.

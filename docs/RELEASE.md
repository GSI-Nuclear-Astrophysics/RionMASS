# Publishing a reviewed release

Build and inspect the exact version before creating a public release:

```sh
python -m build
twine check dist/*
python scripts/check_release.py dist/*
```

Install the wheel in a fresh environment outside the checkout. Check the CLI,
the offline example and the optional GUI. Keep the source-distribution and
wheel hashes with the release notes. Only reviewed source, public documentation,
public examples and tests belong in a code archive.

## PyPI

The release workflow runs manually and requires an explicit `publish` input.
Configure a PyPI Trusted Publisher for `GSI-Nuclear-Astrophysics/RionMASS`, workflow
`release.yml`, environment `pypi`. Protect that environment with release reviewers.
The publishing job uses OIDC; a repository API token is not needed.

The workflow builds, tests, inspects the artifacts and checks their metadata
before upload. Configure Trusted Publishing before invoking it. A source commit
or tag alone does not invoke this workflow.

## Zenodo

Enable the repository in Zenodo's GitHub integration and archive the reviewed
GitHub release, or upload the reviewed source archive as a new software record.
The license, authors, version and repository URL must match the code metadata.

`.zenodo.json` supplies the GitHub archive metadata when both it and
`CITATION.cff` are present. After the record is published, add the actual
version DOI and release date to the citation file and SoftwareX metadata.
Do not reuse another version's DOI as if it identified this source.

References: [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/using-a-publisher/)
and [Zenodo software metadata](https://help.zenodo.org/docs/github/describe-software/).

The manual publishing workflow defaults to TestPyPI and runs the full development checks, including the offscreen GUI tests. Configure a separate trusted publisher for each index.

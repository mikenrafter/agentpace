# Packaging the local ETA fork

Use the PEP 517 Hatchling backend declared in pyproject.toml. For Nix,
buildPythonApplication needs pyproject = true and build-system = [ hatchling ].
There are no third-party runtime Python dependencies. Keep the forced wheel
includes from pyproject.toml: the schemas, example plan and public upstream seed
rows are used by the legacy commands. The executable is agentpace, calling
agentpace.cli:main. The package version currently follows upstream.

Run the unittest suite with PYTHONPATH=src python -m unittest discover -s tests
-t .; pytest is not needed. Native plan locking uses fcntl, so the fork's local
commands currently require a POSIX host. The default diagnostic timezone is UTC;
other AGENTPACE_TIMEZONE values require a zoneinfo database (or Python tzdata).
No T3 installation, network connection or private history is required to import
the package, show help or run tests.

Local estimates require caller-provided local rows. No fitted model or personal
history is included. AGENTPACE_HOME relocates local artifacts; config, data,
timings and ledger paths have CLI overrides. T3 database access remains read-only
and optional, configured through AGENTPACE_T3_DB. See the CLI guide for defaults.
Keep local reports, replay input files and wheel/test output outside the source
checkout when packaging. The example local plan uses synthetic durations.

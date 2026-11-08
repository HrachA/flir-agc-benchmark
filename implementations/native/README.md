# Native AGC implementation

This backend is part of the [combined FLIR AGC repository](../../README.md). Use its shared root environment, setup scripts and runner.

- [Setup and full reproduction](../../README.md#setup-on-windows--powershell)
- [Combined report and all-algorithm comparison](../../docs/REPORT.md)
- [Archived backend report](../../results/published/native/REPORT.md)
- Package: `agc_native`; configuration: `configs/default.json`.
- Run tests from this directory with the root environment: `../../.venv/Scripts/python.exe -m unittest discover -s tests -v`.

The algorithm and experiment source files are preserved from the original independent experiment. All root commands pass absolute dataset/output paths and run each package in its own working directory.
